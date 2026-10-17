"""Persistent local worker for Optimist Prime.

The bot runner still performs one bounded bot cycle when called directly. The
worker keeps one process alive, loads the Reddit client once, and starts later
cycles on the configured cadence. Keeping one process alive avoids repeated
imports and records startup progress in the local log.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from bot_logging import (
    clear_log_context,
    configure_logging,
    console_log_path,
    current_run_id,
    current_log_path,
    ensure_run_id,
    log_event,
    set_log_phase,
)
from env_file import clear_inherited_runtime_environment, load_local_env
from adaptive_scheduler import AdaptiveSchedulerState, choose_next_interval
from runtime_activity import read_cycle_summary, write_cycle_summary
from fast_test_mode import FAST_INTERVAL_SECONDS, get_fast_test_mode
from moderator_activity import get_moderator_activity_status


DEFAULT_INTERVAL_SECONDS = 60
MIN_INTERVAL_SECONDS = 60
MIN_ADAPTIVE_INTERVAL_SECONDS = 2
MAX_INTERVAL_SECONDS = 900


def get_fast_cycle_mode() -> dict:
    """Combine the independent manual-test and moderator-activity boosts."""
    manual = get_fast_test_mode()
    moderator = get_moderator_activity_status()
    active_modes = [
        ("manual_test", manual),
        ("moderator_activity", moderator),
    ]
    active = [(name, status) for name, status in active_modes if status.get("active")]
    if not active:
        return {"active": False, "remaining_seconds": 0, "expires_at": None}
    # A status label aids diagnostics; the effective interval is the same for
    # both controls, and the longer-lived control keeps the combined window on.
    remaining = max(int(status.get("remaining_seconds", 0) or 0) for _, status in active)
    source = "moderator_activity" if any(name == "moderator_activity" for name, _ in active) else "manual_test"
    return {
        "active": True,
        "source": source,
        "remaining_seconds": remaining,
        "expires_at": max(
            float(status.get("expires_at") or 0)
            for _, status in active
        ),
    }


def _interval_seconds(value: int | None = None) -> int:
    """Return a bounded cycle interval from an argument or local .env."""
    if value is None:
        raw = os.environ.get("BOT_LOCAL_CYCLE_INTERVAL_SECONDS", "").strip()
        try:
            value = int(raw) if raw else DEFAULT_INTERVAL_SECONDS
        except ValueError:
            value = DEFAULT_INTERVAL_SECONDS
    return min(MAX_INTERVAL_SECONDS, max(MIN_INTERVAL_SECONDS, int(value)))


def _next_cycle_deadline(
    previous_deadline: float,
    interval_seconds: int,
    *,
    fast_test_active: bool,
    now: float,
) -> float:
    """Keep the normal cadence start-based; fast mode pauses after each cycle."""
    if fast_test_active:
        return now + interval_seconds
    return previous_deadline + interval_seconds


def _exit_code(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return 1


def _scheduler_state_from_summary(
    summary: object,
    *,
    default_interval: int,
    minimum_seconds: int,
    maximum_seconds: int,
) -> AdaptiveSchedulerState:
    """Load compatible interval state while discarding obsolete rate history."""
    fallback = min(maximum_seconds, max(minimum_seconds, int(default_interval)))
    raw = summary.get("scheduler") if isinstance(summary, dict) else None
    if not isinstance(raw, dict):
        return AdaptiveSchedulerState(last_interval_seconds=fallback)
    try:
        interval = int(raw.get("lastIntervalSeconds", fallback))
        if interval <= 0:
            raise ValueError("saved interval must be positive")
    except (TypeError, ValueError):
        return AdaptiveSchedulerState(last_interval_seconds=fallback)
    try:
        samples = max(0, int(raw.get("samples", 0) or 0))
    except (TypeError, ValueError):
        samples = 0
    return AdaptiveSchedulerState(
        last_interval_seconds=min(maximum_seconds, max(minimum_seconds, interval)),
        samples=samples,
    )


def _adaptive_sample_from_summary(
    summary: object, *, expected_run_id: str | None
) -> dict[str, int] | None:
    """Return a complete fresh sample, or None when any enabled scan is uncertain."""
    if not isinstance(summary, dict) or not expected_run_id:
        return None
    if summary.get("runId") != expected_run_id or summary.get("adaptiveMetricsVersion") != 1:
        return None

    counts: dict[str, int] = {}
    enabled_sources = 0
    for name, count_key in (
        ("posts", "eligiblePosts"),
        ("comments", "eligibleComments"),
        ("privateChat", "privateChatMessagesHandled"),
    ):
        enabled = summary.get(f"{name}ScanEnabled")
        usable = summary.get(f"{name}ScanUsable")
        if not isinstance(enabled, bool) or not isinstance(usable, bool):
            return None
        if not enabled:
            counts[count_key] = 0
            continue
        enabled_sources += 1
        if not usable:
            return None
        value = summary.get(count_key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        counts[count_key] = value
    return counts if enabled_sources else None


def _write_scheduler_summary(
    summary: object,
    *,
    scheduler_state: AdaptiveSchedulerState,
    target_items: int,
    interval_seconds: int,
) -> None:
    """Persist cadence even when a fresh workload sample cannot be used."""
    base = summary if isinstance(summary, dict) else {}
    try:
        write_cycle_summary(
            {
                **base,
                "scheduler": scheduler_state.to_dict(),
                "adaptiveSchedulingEnabled": True,
                "adaptiveTargetItemsPerRun": target_items,
                "nextIntervalSeconds": interval_seconds,
            }
        )
    except OSError as exc:
        log_event(
            "scheduler.adaptive_state.failed",
            "Adaptive scheduler state could not be recorded",
            level=logging.WARNING,
            component="scheduler",
            worker="local_worker",
            error_type=type(exc).__name__,
        )


def _wait_for_next_cycle(
    delay: float,
    *,
    status_reader=get_fast_cycle_mode,
    sleeper=time.sleep,
    poll_seconds: float = 1.0,
    mode_was_active: bool = False,
) -> str:
    """Wait for the deadline or a fast-cycle mode transition.

    Return ``activated`` or ``deactivated`` when either boost changes state, and
    ``deadline`` when the already scheduled cycle boundary arrives.
    """
    was_active = bool(mode_was_active)
    deadline = time.monotonic() + max(0.0, delay)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "deadline"
        sleeper(min(max(0.05, poll_seconds), remaining))
        active = bool(status_reader().get("active"))
        if active and not was_active:
            return "activated"
        if was_active and not active:
            return "deactivated"
        was_active = active


def _run_cycle(run_bot: Callable[[], object], stream, *, cycle_number: int) -> int:
    """Run one bot cycle and write a terminal scheduler event."""
    clear_log_context()
    run_id = ensure_run_id()
    set_log_phase("scheduler.startup")
    cycle_started = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    stream.write(
        f"\n=== Optimist Prime local cycle {cycle_number} started {started} "
        f"run_id={run_id} structured_log={current_log_path()} ===\n"
    )
    stream.flush()

    exit_code = 0
    with redirect_stdout(stream), redirect_stderr(stream):
        try:
            # bot_runner reconfigures logging for each cycle so direct runs and
            # persistent-worker runs share the same structured log contract.
            log_path = configure_logging(also_stderr=True)
            log_event(
                "scheduler.cycle.started",
                "Local scheduled bot cycle started",
                component="scheduler",
                worker="local_worker",
                cycle_number=cycle_number,
                working_directory=str(Path.cwd()),
                structured_log_path=str(log_path),
            )
            result = run_bot()
            exit_code = _exit_code(result)
        except SystemExit as exc:
            exit_code = _exit_code(exc.code)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            traceback.print_exc()
            log_event(
                "scheduler.cycle.failed",
                "Local scheduled bot cycle raised an unexpected exception",
                level=logging.ERROR,
                component="scheduler",
                worker="local_worker",
                cycle_number=cycle_number,
                error_type=type(exc).__name__,
            )
            exit_code = 1

    duration_ms = round((time.monotonic() - cycle_started) * 1000, 1)
    outcome = "success" if exit_code == 0 else "failed"
    set_log_phase("scheduler.complete")
    log_event(
        "scheduler.cycle.completed" if exit_code == 0 else "scheduler.cycle.failed",
        "Local scheduled bot cycle finished"
        if exit_code == 0
        else "Local scheduled bot cycle failed",
        level=logging.INFO if exit_code == 0 else logging.ERROR,
        component="scheduler",
        worker="local_worker",
        cycle_number=cycle_number,
        duration_ms=duration_ms,
        exit_code=exit_code,
        outcome=outcome,
    )
    finished = datetime.now(timezone.utc).isoformat()
    stream.write(
        f"=== Optimist Prime local cycle {cycle_number} finished {finished} "
        f"run_id={run_id} duration_ms={duration_ms} exit={exit_code} "
        f"outcome={outcome} ===\n"
    )
    stream.flush()
    return exit_code


def run_worker(
    *,
    max_cycles: int | None = None,
    interval_seconds: int | None = None,
    runner: Callable[[], object] | None = None,
) -> int:
    """Run the persistent local worker, optionally for a bounded test run."""
    # Do not pass credentials as command-line arguments. Load the operator's
    # gitignored .env and prevent inherited shell credentials from taking over.
    clear_inherited_runtime_environment()
    load_local_env(override=True)
    os.environ["BOT_SCHEDULER_CONTEXT"] = "1"
    interval = _interval_seconds(interval_seconds)
    adaptive_enabled = os.environ.get("BOT_ADAPTIVE_SCHEDULING_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    try:
        adaptive_min_interval = max(
            MIN_ADAPTIVE_INTERVAL_SECONDS,
            min(
                MAX_INTERVAL_SECONDS,
                int(os.environ.get("BOT_ADAPTIVE_MIN_INTERVAL_SECONDS", str(MIN_ADAPTIVE_INTERVAL_SECONDS))),
            ),
        )
    except ValueError:
        adaptive_min_interval = MIN_ADAPTIVE_INTERVAL_SECONDS
    try:
        normal_setting = int(os.environ.get("BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS", str(interval)))
    except ValueError:
        normal_setting = interval
    normal_interval = min(
        MAX_INTERVAL_SECONDS,
        max(adaptive_min_interval, normal_setting),
    )
    try:
        adaptive_max_interval = min(
            MAX_INTERVAL_SECONDS,
            max(normal_interval, int(os.environ.get("BOT_ADAPTIVE_MAX_INTERVAL_SECONDS", str(normal_interval)))),
        )
    except ValueError:
        adaptive_max_interval = normal_interval
    target_raw = os.environ.get("BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN")
    if target_raw is None or not target_raw.strip():
        target_raw = os.environ.get("BOT_ADAPTIVE_TARGET_COMMENTS_PER_RUN", "3")
    try:
        target_items = max(1, int(target_raw))
    except (TypeError, ValueError):
        target_items = 3
    prior_summary = read_cycle_summary()
    scheduler_state = _scheduler_state_from_summary(
        prior_summary,
        default_interval=interval,
        minimum_seconds=adaptive_min_interval,
        maximum_seconds=adaptive_max_interval,
    )
    normal_next_interval = interval
    if adaptive_enabled:
        normal_next_interval = min(
            adaptive_max_interval,
            max(adaptive_min_interval, int(scheduler_state.last_interval_seconds)),
        )
    path = console_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Importing bot_runner also imports the project's feature handlers. It is
    # deliberately performed once, after logging is active, so a slow first
    # import is visible and cannot delay every recurring cycle.
    import_started = time.monotonic()
    worker_run_id = ensure_run_id()
    set_log_phase("scheduler.startup")
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        started = datetime.now(timezone.utc).isoformat()
        stream.write(
            f"\n=== Optimist Prime local worker started {started} "
            f"run_id={worker_run_id} structured_log={current_log_path()} ===\n"
        )
        stream.write(
            "Worker is starting once and will run one cycle about every "
            f"{interval} seconds. Reddit request starts, retries, timeouts, "
            "and each cycle result will appear below.\n"
        )
        stream.flush()

        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                log_path = configure_logging(also_stderr=True)
                log_event(
                    "scheduler.worker.import.started",
                    "Loading Optimist Prime bot modules once for the persistent worker",
                    component="scheduler",
                    worker="local_worker",
                    interval_seconds=interval,
                    structured_log_path=str(log_path),
                )
                if runner is None:
                    from bot_runner import main as runner
                import_duration_ms = round(
                    (time.monotonic() - import_started) * 1000, 1
                )
                log_event(
                    "scheduler.worker.import.completed",
                    "Optimist Prime bot modules loaded; recurring cycles are ready",
                    component="scheduler",
                    worker="local_worker",
                    interval_seconds=interval,
                    duration_ms=import_duration_ms,
                )
                log_event(
                    "scheduler.worker.ready",
                    "Persistent local worker is ready for configured cycles",
                    component="scheduler",
                    worker="local_worker",
                    interval_seconds=interval,
                )
                stream.write(
                    f"Worker ready after {import_duration_ms}ms; recurring cycle "
                    f"interval is {interval}s.\n"
                )
                stream.flush()
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                traceback.print_exc()
                log_event(
                    "scheduler.worker.import.failed",
                    "Persistent worker could not load the bot modules",
                    level=logging.ERROR,
                    component="scheduler",
                    worker="local_worker",
                    interval_seconds=interval,
                    duration_ms=round((time.monotonic() - import_started) * 1000, 1),
                    error_type=type(exc).__name__,
                )
                return 1

            next_cycle_at = time.monotonic()
            cycle_number = 0
            last_exit_code = 0
            while max_cycles is None or cycle_number < max_cycles:
                cycle_number += 1
                last_exit_code = _run_cycle(
                    runner, stream, cycle_number=cycle_number
                )
                if max_cycles is not None and cycle_number >= max_cycles:
                    break

                # Keep cycle starts approximately interval seconds apart. If a
                # slow cycle misses one or more intervals, record that fact and
                # begin the next cycle immediately rather than creating overlap.
                next_interval = interval
                cycle_summary = read_cycle_summary()
                fast_cycle = get_fast_cycle_mode()
                fast_test_active = bool(fast_cycle.get("active"))
                if fast_test_active:
                    next_interval = FAST_INTERVAL_SECONDS
                    event_name = (
                        "scheduler.fast_test.active"
                        if fast_cycle.get("source") == "manual_test"
                        else "scheduler.moderator_activity.fast_cycles"
                    )
                    log_event(
                        event_name,
                        "Fast polling is active; cycle overlap and the shared Reddit request budget remain unchanged",
                        component="scheduler",
                        worker="local_worker",
                        source=fast_cycle.get("source"),
                        interval_seconds=next_interval,
                        remaining_seconds=fast_cycle.get("remaining_seconds"),
                    )
                elif adaptive_enabled and last_exit_code != 0:
                    # A failed cycle may leave the previous summary on disk.
                    # Keep the last known-good cadence; never treat stale work
                    # as a fresh sample or reset a busy schedule to the baseline.
                    next_interval = scheduler_state.last_interval_seconds
                    _write_scheduler_summary(
                        cycle_summary,
                        scheduler_state=scheduler_state,
                        target_items=target_items,
                        interval_seconds=next_interval,
                    )
                    log_event(
                        "scheduler.adaptive.failure_hold",
                        "Adaptive timing held at the current interval after a failed cycle",
                        level=logging.WARNING,
                        component="scheduler",
                        worker="local_worker",
                        interval_seconds=next_interval,
                    )
                elif adaptive_enabled:
                    sample = _adaptive_sample_from_summary(
                        cycle_summary,
                        expected_run_id=current_run_id(),
                    )
                    if sample is None:
                        next_interval = scheduler_state.last_interval_seconds
                        _write_scheduler_summary(
                            cycle_summary,
                            scheduler_state=scheduler_state,
                            target_items=target_items,
                            interval_seconds=next_interval,
                        )
                        log_event(
                            "scheduler.adaptive.sample_hold",
                            "Adaptive timing held because this cycle had no fresh usable workload sample",
                            level=logging.WARNING,
                            component="scheduler",
                            worker="local_worker",
                            interval_seconds=next_interval,
                            run_id_matches=bool(
                                isinstance(cycle_summary, dict)
                                and cycle_summary.get("runId") == current_run_id()
                            ),
                        )
                    else:
                        next_interval, scheduler_state = choose_next_interval(
                            scheduler_state,
                            eligible_posts=sample["eligiblePosts"],
                            eligible_comments=sample["eligibleComments"],
                            private_chat_messages=sample["privateChatMessagesHandled"],
                            target_items=target_items,
                            minimum_seconds=adaptive_min_interval,
                            maximum_seconds=adaptive_max_interval,
                        )
                        _write_scheduler_summary(
                            cycle_summary,
                            scheduler_state=scheduler_state,
                            target_items=target_items,
                            interval_seconds=next_interval,
                        )
                if not fast_test_active:
                    normal_next_interval = next_interval
                # Normal scheduling targets a start-to-start interval and
                # starts an overdue cycle immediately after the previous one
                # finishes. Fast testing instead pauses for the full interval
                # after completion, preventing rapid back-to-back cycles when
                # a cycle itself takes longer than 3 seconds.
                now = time.monotonic()
                next_cycle_at = _next_cycle_deadline(
                    next_cycle_at,
                    next_interval,
                    fast_test_active=fast_test_active,
                    now=now,
                )
                delay = next_cycle_at - now
                if delay <= 0:
                    missed = max(1, int((-delay) // interval) + 1)
                    log_event(
                        "scheduler.cycle.overdue",
                        "A cycle took longer than the configured interval; no overlapping cycle was started",
                        level=logging.WARNING,
                        component="scheduler",
                        worker="local_worker",
                        interval_seconds=next_interval,
                        missed_intervals=missed,
                        adaptive_scheduling=adaptive_enabled,
                    )
                    stream.write(
                        f"Cycle {cycle_number} exceeded the {next_interval}s interval; "
                        "starting the next cycle without overlap.\n"
                    )
                    stream.flush()
                    next_cycle_at = now
                    continue

                log_event(
                    "scheduler.cycle.waiting",
                    "Persistent worker is waiting for the next cycle",
                    component="scheduler",
                    worker="local_worker",
                    interval_seconds=next_interval,
                    next_cycle_in_seconds=round(delay, 1),
                    adaptive_scheduling=adaptive_enabled,
                    target_items=target_items if adaptive_enabled else None,
                )
                stream.write(
                    f"Next cycle in about {round(delay, 1)} seconds; worker remains running."
                    + (" Adaptive timing is active." if adaptive_enabled else "")
                    + "\n"
                )
                stream.flush()
                wait_mode_was_active = fast_test_active
                while True:
                    wait_result = _wait_for_next_cycle(
                        delay,
                        status_reader=get_fast_cycle_mode,
                        mode_was_active=wait_mode_was_active,
                    )
                    if wait_result == "activated":
                        next_cycle_at = time.monotonic()
                        active_mode = get_fast_cycle_mode()
                        if active_mode.get("source") == "moderator_activity":
                            event_name = "scheduler.moderator_activity.fast_cycle_started"
                            message = "Moderator conversation detected; starting the next cycle now"
                        else:
                            event_name = "scheduler.fast_test.started"
                            message = "Temporary fast testing requested; starting the next cycle now"
                        log_event(
                            event_name,
                            message,
                            component="scheduler",
                            worker="local_worker",
                            interval_seconds=FAST_INTERVAL_SECONDS,
                            source=active_mode.get("source"),
                        )
                        stream.write(
                            "Fast polling enabled; starting the next cycle now, then waiting 3 seconds after each cycle finishes.\n"
                        )
                        stream.flush()
                        break
                    if wait_result == "deactivated":
                        # Do not let a pending three-second deadline launch
                        # one more fast cycle after stop or lease expiry.
                        next_cycle_at = time.monotonic() + normal_next_interval
                        delay = next_cycle_at - time.monotonic()
                        wait_mode_was_active = False
                        log_event(
                            "scheduler.fast_test.stopped",
                            "Fast testing ended; normal cycle timing is restored",
                            component="scheduler",
                            worker="local_worker",
                            interval_seconds=normal_next_interval,
                        )
                        stream.write(
                            "Fast testing ended; returning to normal cycle timing.\n"
                        )
                        stream.flush()
                        continue
                    break

            log_event(
                "scheduler.worker.stopped",
                "Persistent local worker stopped after its requested cycle limit",
                component="scheduler",
                worker="local_worker",
                cycles=cycle_number,
                exit_code=last_exit_code,
            )
            return last_exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimist Prime continuous local worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and stop (useful for diagnostics and tests)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        help="Override the local cycle interval for this worker process",
    )
    args = parser.parse_args(argv)
    return run_worker(
        max_cycles=1 if args.once else None,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
