"""Health diagnostics dump for agents and operators.

Implements Application Standard 20 (health diagnostics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reddit_request_budget import (
    read_request_budget_snapshot,
    resolve_request_budget_settings,
)
from moderator_activity import get_moderator_activity_status

REPO_ROOT = Path(__file__).resolve().parent

_PROCESS_STARTED_AT = time.time()
_ENVIRONMENT_LOAD_LOCK = threading.RLock()


def _safe_read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _public_response_safety_summary(state: Any) -> dict[str, Any]:
    """Expose aggregate held/recent response counts without Reddit identities."""
    empty = {
        "ledgerReadable": True,
        "pending": 0,
        "uncertain": 0,
        "countedLastHour": 0,
        "ordinaryLastHour": 0,
        "moderatorLastHour": 0,
    }
    if not isinstance(state, dict):
        return {**empty, "ledgerReadable": False}
    ledger = state.get("public_response_ledger", {})
    if not isinstance(ledger, dict):
        return {**empty, "ledgerReadable": False}

    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - 3600
    result = dict(empty)
    for record in ledger.values():
        if not isinstance(record, dict):
            result["ledgerReadable"] = False
            continue
        status = record.get("status")
        if status == "pending":
            result["pending"] += 1
        elif status == "uncertain":
            result["uncertain"] += 1
        if status not in {"pending", "uncertain", "completed"}:
            continue
        created = record.get("created_at")
        if isinstance(created, bool) or not isinstance(created, (int, float)) or created < cutoff:
            continue
        result["countedLastHour"] += 1
        tier = record.get("tier")
        if tier == "moderator":
            result["moderatorLastHour"] += 1
        elif tier == "ordinary":
            result["ordinaryLastHour"] += 1
    return result


def _recent_audit_errors(limit: int = 10) -> list[dict[str, Any]]:
    from audit_log import audit_log_path

    entries = _safe_read_json(audit_log_path(), [])
    if not isinstance(entries, list):
        return []
    errors: list[dict[str, Any]] = []
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        outcome = str(entry.get("outcome", "")).lower()
        if "error" in outcome or "fail" in outcome or entry.get("error"):
            errors.append(entry)
            if len(errors) >= limit:
                break
    return errors


def _tail_log_lines(path: Path, max_lines: int = 30) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


def _structured_log_records(path: Path, max_lines: int = 2000) -> list[dict[str, Any]]:
    """Read the retained JSONL tail without failing the diagnostics command."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-max_lines:]:
        try:
            record = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _record_timestamp(record: dict[str, Any]) -> datetime | None:
    return _parse_run_timestamp(record.get("timestamp"))


def _record_time_text(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    value = record.get("timestamp")
    return str(value) if value else None


def _request_snapshot(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return safe request metadata suitable for the diagnostics payload."""
    if not record:
        return None
    snapshot: dict[str, Any] = {}
    for source, target in (
        ("timestamp", "timestamp"),
        ("eventName", "event"),
        ("phase", "phase"),
        ("requestId", "requestId"),
        ("attempt", "attempt"),
        ("requestMethod", "method"),
        ("requestPath", "path"),
        ("timeoutSeconds", "timeoutSeconds"),
        ("durationMs", "durationMs"),
        ("statusCode", "statusCode"),
        ("retryable", "retryable"),
        ("retryReason", "retryReason"),
        ("errorType", "errorType"),
    ):
        if source in record and record[source] is not None:
            snapshot[target] = record[source]
    return snapshot or None


def _operational_log_summary(path: Path) -> dict[str, Any]:
    """Summarize scheduler and Reddit transport evidence from structured logs."""
    records = _structured_log_records(path)
    starts: list[dict[str, Any]] = []
    terminals: list[dict[str, Any]] = []
    reddit_records: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []
    retry_records: list[dict[str, Any]] = []
    timeout_records: list[dict[str, Any]] = []
    worker_import_starts: list[dict[str, Any]] = []
    worker_import_terminals: list[dict[str, Any]] = []
    worker_ready_records: list[dict[str, Any]] = []
    for record in records:
        event = str(record.get("eventName", ""))
        if event == "scheduler.cycle.started":
            starts.append(record)
        elif event in {"scheduler.cycle.completed", "scheduler.cycle.failed"}:
            terminals.append(record)
        elif event == "scheduler.worker.import.started":
            worker_import_starts.append(record)
        elif event in {
            "scheduler.worker.import.completed",
            "scheduler.worker.import.failed",
        }:
            worker_import_terminals.append(record)
        elif event == "scheduler.worker.ready":
            worker_ready_records.append(record)
        if event.startswith("reddit."):
            reddit_records.append(record)
        if event.startswith("reddit.request."):
            request_records.append(record)
        if event == "reddit.retry":
            retry_records.append(record)

        error_type = str(record.get("errorType", ""))
        retry_reason = str(record.get("retryReason", ""))
        # Count request attempts, not both the attempt failure and PRAW's
        # separate retry-scheduled log line for that same timeout.
        if event in {
            "reddit.request.retryable_failure",
            "reddit.request.failed",
        } and ("timeout" in error_type.lower() or "timeout" in retry_reason.lower()):
            timeout_records.append(record)

    last_start = starts[-1] if starts else None
    last_terminal = terminals[-1] if terminals else None
    last_start_time = _record_timestamp(last_start or {})
    last_terminal_time = _record_timestamp(last_terminal or {})
    cycle_in_progress = bool(
        last_start_time
        and (last_terminal_time is None or last_start_time > last_terminal_time)
    )
    cycle_age_seconds = None
    if cycle_in_progress and last_start_time:
        cycle_age_seconds = round(
            max(0.0, (datetime.now(timezone.utc) - last_start_time).total_seconds()),
            1,
        )

    last_cycle_run_id = (last_start or {}).get("correlationId")
    cycle_records = (
        [
            record
            for record in records
            if last_cycle_run_id
            and record.get("correlationId") == last_cycle_run_id
        ]
        if last_cycle_run_id
        else []
    )
    cycle_retries = sum(1 for record in cycle_records if record.get("eventName") == "reddit.retry")
    cycle_timeouts = sum(
        1
        for record in cycle_records
        if record.get("eventName") in {
            "reddit.request.retryable_failure",
            "reddit.request.failed",
        }
        and (
            "timeout" in str(record.get("errorType", "")).lower()
            or "timeout" in str(record.get("retryReason", "")).lower()
        )
    )

    last_worker_import_start = worker_import_starts[-1] if worker_import_starts else None
    last_worker_import_terminal = (
        worker_import_terminals[-1] if worker_import_terminals else None
    )
    worker_import_start_time = _record_timestamp(last_worker_import_start or {})
    worker_import_terminal_time = _record_timestamp(last_worker_import_terminal or {})
    worker_import_in_progress = bool(
        worker_import_start_time
        and (
            worker_import_terminal_time is None
            or worker_import_start_time > worker_import_terminal_time
        )
    )
    last_worker_ready = worker_ready_records[-1] if worker_ready_records else None
    worker_interval_seconds = (
        (last_worker_ready or last_worker_import_start or {}).get("intervalSeconds")
    )

    last_reddit_record = request_records[-1] if request_records else None
    last_retry = retry_records[-1] if retry_records else None
    last_timeout = timeout_records[-1] if timeout_records else None
    last_failure = next(
        (
            record
            for record in reversed(reddit_records)
            if str(record.get("eventName", "")).endswith((".failed", ".retryable_failure"))
        ),
        None,
    )

    return {
        "scheduler": {
            "logRecordsRetained": len(records),
            "lastCycleStartedAt": _record_time_text(last_start),
            "lastCycleFinishedAt": _record_time_text(last_terminal),
            "lastCycleOutcome": (last_terminal or {}).get("outcome") if last_terminal else None,
            "lastCycleDurationMs": (last_terminal or {}).get("durationMs") if last_terminal else None,
            "lastCycleExitCode": (last_terminal or {}).get("exitCode") if last_terminal else None,
            "lastCycleRunId": last_cycle_run_id,
            "cycleInProgress": cycle_in_progress,
            "currentCycleAgeSeconds": cycle_age_seconds,
            "currentCycleRunId": last_cycle_run_id if cycle_in_progress else None,
            "currentCycleRetryCount": cycle_retries if cycle_in_progress else 0,
            "currentCycleTimeoutCount": cycle_timeouts if cycle_in_progress else 0,
            "workerImportInProgress": worker_import_in_progress,
            "workerImportStartedAt": _record_time_text(last_worker_import_start),
            "workerReadyAt": _record_time_text(last_worker_ready),
            "workerImportDurationMs": (
                last_worker_import_terminal.get("durationMs")
                if last_worker_import_terminal
                else None
            ),
            "cycleIntervalSeconds": worker_interval_seconds,
        },
        "redditRequests": {
            "logRecordsRetained": len(reddit_records),
            "retryCount": len(retry_records),
            "retryableFailureCount": sum(
                1 for record in reddit_records if record.get("eventName") == "reddit.request.retryable_failure"
            ),
            "timeoutCount": len(timeout_records),
            "lastRequest": _request_snapshot(last_reddit_record),
            "lastRetry": _request_snapshot(last_retry),
            "lastTimeout": _request_snapshot(last_timeout),
            "lastFailure": _request_snapshot(last_failure),
        },
    }


def _configured_reddit_timeout_seconds() -> int:
    raw = os.environ.get("REDDIT_API_TIMEOUT_SECONDS", "").strip()
    try:
        value = int(raw) if raw else 15
    except ValueError:
        value = 15
    return min(60, max(5, value))


def _parse_run_timestamp(value: Any) -> datetime | None:
    """Parse a stats timestamp, returning None for missing/invalid markers."""
    if not value or value == "Never":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def _last_run_recency(value: Any, *, fresh_hours: int = 6) -> tuple[bool, float | None]:
    """Return freshness and age for the last aggregate completed-run marker."""
    parsed = _parse_run_timestamp(value)
    if parsed is None:
        return False, None
    age_hours = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 3600)
    return age_hours <= fresh_hours, round(age_hours, 2)


def _build_operational_status(
    *,
    last_run_proven: bool,
    last_run_recent: bool,
    last_run_age_hours: float | None,
    state_readable: bool,
    reddit_credentials_ready: bool,
    llm_key_ready: bool,
    running_in_actions: bool,
    local_configuration_ready: bool | None = None,
) -> dict[str, Any]:
    """Build a human-readable status while retaining machine-readable facts."""
    requested = "Inspect the configured Reddit community and perform only enabled actions."
    runtime_name = "the CI runner" if running_in_actions else "the local runtime"

    if not state_readable:
        return {
            "summary": "The worker cannot safely continue because its runtime state is unavailable.",
            "requested": requested,
            "observed": "The encrypted state could not be read.",
            "confirmation": "No new run is considered safe to confirm while required state is unreadable.",
            "blocker": "The encryption key is missing or the encrypted state is invalid.",
            "nextAction": "Restore the state encryption key or repair the encrypted state before running the worker.",
            "nextActionAt": None,
            "userActionRequired": True,
            "severity": "error",
            "incidentId": "runtime-state-unreadable",
        }

    if not last_run_proven:
        observed = f"No valid completed-run marker is available for {runtime_name}."
        blocker = "A completed worker run cannot be confirmed."
        if not running_in_actions and not reddit_credentials_ready:
            blocker += " Local Reddit credentials are also not configured."
        next_action = (
            "Configure Reddit and LLM credentials for this runtime, then run a controlled Safe Mode check."
        )
        return {
            "summary": "The worker has not produced a completed run that can be confirmed.",
            "requested": requested,
            "observed": observed,
            "confirmation": "There is no evidence of a completed worker operation yet.",
            "blocker": blocker,
            "nextAction": next_action,
            "nextActionAt": None,
            "userActionRequired": True,
            "severity": "warning",
            "incidentId": "worker:no-completed-run",
        }

    if not last_run_recent:
        age_text = (
            f" about {last_run_age_hours:g} hours ago"
            if last_run_age_hours is not None
            else ""
        )
        return {
            "summary": "A completed run exists, but recent worker operation is not confirmed.",
            "requested": requested,
            "observed": f"The last completed worker run was recorded{age_text}.",
            "confirmation": "The old completion marker is valid, but it does not prove the worker is operating now.",
            "blocker": "No recent completed run has been observed.",
            "nextAction": "Check the local worker and configuration, then run a controlled Safe Mode check.",
            "nextActionAt": None,
            "userActionRequired": True,
            "severity": "warning",
            "incidentId": "worker:no-recent-completed-run",
        }

    readiness_blocker = None
    if not running_in_actions and not reddit_credentials_ready:
        readiness_blocker = "This local machine has no Reddit credentials, so a new local live run cannot start."
    elif not running_in_actions and not llm_key_ready:
        readiness_blocker = "This local machine has no LLM key, so a new local live run cannot generate responses."
    elif not running_in_actions and local_configuration_ready is False:
        readiness_blocker = (
            "The local .env does not contain the complete operator configuration; "
            "a background worker must not depend on an inherited shell credential."
        )

    return {
        "summary": "A recent completed worker run has been observed.",
        "requested": requested,
        "observed": "The completion marker is within the recent-activity window.",
        "confirmation": "A run completed, but this marker does not prove that every enabled feature succeeded.",
        "blocker": readiness_blocker,
        "nextAction": "Continue monitoring the next completed run and inspect errors if the status changes."
        if readiness_blocker is None
        else "Configure the missing local credential before starting another live run.",
        "nextActionAt": None,
        "userActionRequired": readiness_blocker is not None,
        "severity": "warning" if readiness_blocker else "ok",
        "incidentId": "local-readiness-blocked" if readiness_blocker else None,
    }


def dump_diagnostics(
    *,
    load_environment: bool = False,
    environment_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return diagnostics while keeping local environment loading isolated."""
    if not load_environment:
        return _dump_diagnostics_loaded(
            load_environment=False,
            environment_path=environment_path,
        )

    from env_file import clear_inherited_runtime_environment, load_local_env

    env_path = Path(environment_path) if environment_path else REPO_ROOT / ".env"
    with _ENVIRONMENT_LOAD_LOCK:
        original_environment = dict(os.environ)
        try:
            try:
                clear_inherited_runtime_environment()
                load_local_env(env_path, override=True)
            except Exception:
                # Diagnostics must remain available even if an optional local
                # env file is malformed or inaccessible.
                pass
            return _dump_diagnostics_loaded(
                load_environment=True,
                environment_path=environment_path,
            )
        finally:
            os.environ.clear()
            os.environ.update(original_environment)


def _dump_diagnostics_loaded(
    *,
    load_environment: bool = False,
    environment_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a machine-readable health snapshot (no secrets).

    The read-only function does not implicitly load the operator's ``.env``.
    Executable/UI boundaries opt into environment loading explicitly so test
    processes and contributor tools cannot inherit local settings merely
    by importing or calling diagnostics.
    """
    from bot_logging import console_log_path, current_log_path, log_dir

    env_path = Path(environment_path) if environment_path else REPO_ROOT / ".env"
    local_env_values: dict[str, str] = {}
    if load_environment:
        from env_file import load_local_env

        try:
            # The CLI/UI are local readiness checks. The local .env is
            # the source of truth, so an inherited Codex/shell key must not
            # make an incomplete operator configuration look ready for live use.
            from env_file import clear_inherited_runtime_environment

            clear_inherited_runtime_environment()
            local_env_values = load_local_env(env_path, override=True)
        except Exception:
            # Diagnostics must remain available even if an optional local env
            # file is malformed or inaccessible.
            pass

    try:
        # Avoid importing bot_runner (heavy / side-effectful); keep in sync with BOT_VERSION there.
        version = "2.0"
        import ast

        runner = REPO_ROOT / "bot_runner.py"
        if runner.exists():
            for line in runner.read_text(encoding="utf-8").splitlines():
                if line.startswith("BOT_VERSION"):
                    version = ast.literal_eval(line.split("=", 1)[1].strip())
                    break
    except Exception:
        version = "unknown"

    def configured_path(environment_name: str, default: Path) -> Path:
        raw = os.environ.get(environment_name, "").strip()
        if not raw:
            return default
        candidate = Path(os.path.expandvars(os.path.expanduser(raw)))
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    def outside_repository(path: Path) -> bool:
        try:
            path.resolve().relative_to(REPO_ROOT.resolve())
            return False
        except ValueError:
            return True

    configured_state_path = configured_path(
        "BOT_RUNTIME_STATE_FILE", REPO_ROOT / "data" / "bot_state.json"
    )
    configured_stats_path = configured_path(
        "BOT_RUNTIME_STATS_FILE", REPO_ROOT / "data" / "stats.json"
    )
    configured_audit_path = configured_path(
        "BOT_RUNTIME_AUDIT_LOG_FILE",
        configured_path("AUDIT_LOG_PATH", REPO_ROOT / "data" / "audit_log.json"),
    )
    state_path = configured_state_path
    encrypted_state_path = (
        configured_state_path
        if configured_state_path.suffix.lower() == ".enc"
        else REPO_ROOT / "data" / "bot_state.enc"
    )
    stats_path = configured_stats_path
    audit_path = configured_audit_path
    activity_path = configured_path(
        "BOT_RUNTIME_ACTIVITY_FILE", REPO_ROOT / "data" / "activity.json"
    )
    cycle_summary_path = configured_path(
        "BOT_RUNTIME_CYCLE_SUMMARY_FILE", REPO_ROOT / "data" / "cycle_summary.json"
    )
    rules_reference_path = configured_path(
        "BOT_COMMUNITY_RULES_REFERENCE_FILE",
        REPO_ROOT / "data" / "community_rules.reference.json",
    )
    state_path_outside_repo = outside_repository(state_path)
    stats_path_outside_repo = outside_repository(stats_path)
    audit_path_outside_repo = outside_repository(audit_path)
    activity_path_outside_repo = outside_repository(activity_path)
    cycle_summary_path_outside_repo = outside_repository(cycle_summary_path)
    request_budget_settings = resolve_request_budget_settings(
        os.environ,
        default_root=REPO_ROOT,
    )
    request_budget_snapshot = read_request_budget_snapshot(
        request_budget_settings.path,
        requests_per_minute=request_budget_settings.requests_per_minute,
        window_seconds=request_budget_settings.window_seconds,
        max_requests=request_budget_settings.max_requests,
    )
    request_budget_outside_repo = outside_repository(request_budget_settings.path)
    prefs_path = REPO_ROOT / "data" / "settings_gui_prefs.json"
    log_path = current_log_path()
    console_path = console_log_path()
    operational_log = _operational_log_summary(log_path)
    activity_snapshot = _safe_read_json(activity_path, {})
    cycle_summary = _safe_read_json(cycle_summary_path, {})
    try:
        from community_rules import load_reference, reference_status

        community_rules_status = reference_status(load_reference(rules_reference_path))
    except Exception as exc:
        community_rules_status = {
            "warnings": [f"community rules reference unavailable: {type(exc).__name__}"],
        }
    community_rules_status["path"] = str(rules_reference_path)
    community_rules_status["outsideRepository"] = outside_repository(rules_reference_path)
    adaptive_enabled = os.environ.get("BOT_ADAPTIVE_SCHEDULING_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }

    stats = _safe_read_json(stats_path, {})
    state = {}
    state_read_error = None
    state_backend_requested = os.environ.get("BOT_STATE_BACKEND", "").strip().lower()
    actions_runtime = os.environ.get("GITHUB_ACTIONS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    state_backend = (
        "encrypted"
        if state_backend_requested in {"encrypted", "fernet"}
        or encrypted_state_path.exists()
        or os.environ.get("BOT_STATE_ENCRYPTION_KEY")
        or actions_runtime
        else "plaintext-local"
    )
    state_readable = True
    if state_backend == "encrypted":
        encryption_key = os.environ.get("BOT_STATE_ENCRYPTION_KEY", "").strip()
        if encryption_key:
            try:
                from state_store import load_state as load_runtime_state

                state = load_runtime_state(encrypted_state_path, {}, dry_run=True)
            except Exception as exc:
                state_readable = False
                state_read_error = type(exc).__name__
        else:
            state_readable = False
            state_read_error = "BOT_STATE_ENCRYPTION_KEY is not configured; encrypted state is not readable"
    else:
        state = _safe_read_json(state_path, {})
    catch_up_metrics: dict[str, Any] = {}
    public_response_metrics = _public_response_safety_summary(state)
    if isinstance(state, dict):
        try:
            from catch_up import catch_up_summary, ensure_catch_up_state

            ensure_catch_up_state(state)
            catch_up_metrics = catch_up_summary(state)
        except Exception as exc:
            catch_up_metrics = {"error": type(exc).__name__}
    last_run = stats.get("last_run") if isinstance(stats, dict) else None
    if isinstance(state, dict):
        last_run = last_run or state.get("last_check_completed") or state.get("last_check")

    review_metrics = {}
    if isinstance(state, dict):
        review_metrics = (
            state.get("reported_user_reviews", {}).get("metrics", {})
            if isinstance(state.get("reported_user_reviews", {}), dict)
            else {}
        )
    if not isinstance(review_metrics, dict):
        review_metrics = {}
    try:
        from history_review_ledger import diagnostic_ledger_metrics

        history_ledger_metrics = diagnostic_ledger_metrics()
    except Exception as exc:
        history_ledger_metrics = {"ledger_available": False, "read_error": type(exc).__name__}
    last_run_proven = (
        _parse_run_timestamp(stats.get("last_run")) is not None
        if isinstance(stats, dict)
        else False
    )
    last_run_recent, last_run_age_hours = _last_run_recency(
        stats.get("last_run") if isinstance(stats, dict) else None
    )
    if last_run_recent:
        operational_status = "recent completed run observed"
    elif last_run_proven:
        operational_status = "stale completed run observed"
    else:
        operational_status = "no completed run observed"

    reddit_client_configured = bool(
        os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")
    )
    reddit_refresh_token_configured = bool(
        reddit_client_configured and os.environ.get("REDDIT_REFRESH_TOKEN")
    )
    # Report the configured provider without requiring or exposing an API key.
    default_reasoning_effort = "low"
    try:
        from llm_client import (
            DEFAULT_LLM_REASONING_EFFORT,
            PROVIDER_PRESETS,
        )

        default_reasoning_effort = DEFAULT_LLM_REASONING_EFFORT
        provider = os.environ.get("BOT_LLM_PROVIDER", "").strip().lower()
        provider_known = provider in PROVIDER_PRESETS
        preset = PROVIDER_PRESETS.get(provider, {})
        active_llm_key_names = tuple(
            name.strip()
            for name in preset.get("api_key_envs", "").split(",")
            if name.strip()
        )
        effective_llm_model = os.environ.get("LLM_MODEL", "").strip()
        effective_base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or preset.get("base_url")
            or ""
        ).strip()
    except Exception:
        provider = "unknown"
        provider_known = False
        active_llm_key_names = ()
        effective_llm_model = "unknown"
        effective_base_url = ""

    raw_thinking = os.environ.get("LLM_THINKING_ENABLED", "").strip().lower()
    if not raw_thinking:
        llm_thinking_enabled: bool | None = None
        llm_thinking_setting = "default"
    elif raw_thinking in {"1", "true", "yes", "on"}:
        llm_thinking_enabled = True
        llm_thinking_setting = "true"
    elif raw_thinking in {"0", "false", "no", "off"}:
        llm_thinking_enabled = False
        llm_thinking_setting = "false"
    else:
        llm_thinking_enabled = None
        llm_thinking_setting = "invalid"

    llm_reasoning_effort = (
        os.environ.get("LLM_REASONING_EFFORT", default_reasoning_effort)
        .strip()
        .lower()
        or default_reasoning_effort
    )
    llm_key_configured = any(os.environ.get(name, "").strip() for name in active_llm_key_names)

    llm_configuration_errors: list[str] = []
    if not provider_known:
        llm_configuration_errors.append("unknown provider")
    elif not effective_llm_model:
        llm_configuration_errors.append("model is not configured")
    if provider_known and not effective_base_url:
        llm_configuration_errors.append("endpoint is not configured")
    if not llm_key_configured:
        llm_configuration_errors.append("provider API key is missing")
    llm_configuration_ready = not llm_configuration_errors

    def credential_source(names: tuple[str, ...]) -> str:
        if any(str(local_env_values.get(name, "")).strip() for name in names):
            return "local .env"
        if any(os.environ.get(name, "").strip() for name in names):
            return "process environment"
        return "missing"

    local_reddit_credentials_ready = bool(
        local_env_values.get("REDDIT_CLIENT_ID", "").strip()
        and local_env_values.get("REDDIT_CLIENT_SECRET", "").strip()
        and local_env_values.get("REDDIT_REFRESH_TOKEN", "").strip()
    )
    local_llm_key_ready = any(
        str(local_env_values.get(name, "")).strip() for name in active_llm_key_names
    )
    local_configuration_ready = (
        local_reddit_credentials_ready and local_llm_key_ready and llm_configuration_ready
        if load_environment
        else None
    )

    reddit_credential_source = credential_source(
        (
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "REDDIT_REFRESH_TOKEN",
        )
    )
    llm_credential_source = credential_source(active_llm_key_names)
    operational_status_detail = _build_operational_status(
        last_run_proven=last_run_proven,
        last_run_recent=last_run_recent,
        last_run_age_hours=last_run_age_hours,
        state_readable=state_readable,
        reddit_credentials_ready=reddit_refresh_token_configured,
        llm_key_ready=llm_key_configured,
        running_in_actions=actions_runtime,
        local_configuration_ready=local_configuration_ready,
    )

    reply_limits: dict[str, Any]
    try:
        # Diagnostics runs in a lightweight process, so it can safely refresh
        # the configuration modules after loading the operator's local .env.
        import importlib
        import config as runtime_config
        import runtime_settings as runtime_settings_module
        importlib.reload(runtime_config)
        importlib.reload(runtime_settings_module)
        effective_settings = runtime_settings_module.resolve_runtime_settings()
        community_rules_status["contentModerationEnabled"] = runtime_config.CONTENT_MODERATION_ENABLED
        daily_user_replies = (
            state.get("daily_non_moderator_replies")
            if isinstance(state, dict)
            else 0
        )
        if not isinstance(daily_user_replies, (int, float)):
            daily_user_replies = state.get("daily_replies", 0) if isinstance(state, dict) else 0
        reply_limits = {
            "ordinaryUserPerRunCap": effective_settings.max_replies_per_run,
            "ordinaryUserDailyCap": effective_settings.max_replies_per_day,
            "sameUserCooldownHours": runtime_config.SAME_USER_COOLDOWN_HOURS,
            "sameUserRepliesBeforeCooldown": runtime_config.SAME_USER_REPLIES_BEFORE_COOLDOWN,
            "moderatorsBypassLimits": runtime_config.MODERATOR_REPLIES_BYPASS_LIMITS,
            "selfFlairRequestsEnabled": runtime_config.USER_CUSTOM_FLAIR_ENABLED,
            "dailyOrdinaryUserRepliesUsed": max(0, int(daily_user_replies or 0)),
            "dailyTotalRepliesIncludingModerators": max(
                0,
                int(state.get("daily_replies", 0) or 0)
                if isinstance(state, dict)
                else 0,
            ),
        }
    except Exception:
        reply_limits = {
            "ordinaryUserPerRunCap": None,
            "ordinaryUserDailyCap": None,
            "sameUserCooldownHours": None,
            "sameUserRepliesBeforeCooldown": None,
            "moderatorsBypassLimits": None,
            "selfFlairRequestsEnabled": None,
            "dailyOrdinaryUserRepliesUsed": None,
            "dailyTotalRepliesIncludingModerators": None,
        }

    return {
        "ok": True,
        "actionId": "diagnostics.dump",
        "app": "Optimist Prime",
        "version": version,
        "pid": os.getpid(),
        "uptimeSeconds": round(time.time() - _PROCESS_STARTED_AT, 1),
        "startedAt": datetime.fromtimestamp(_PROCESS_STARTED_AT, tz=timezone.utc).isoformat(),
        "now": datetime.now(timezone.utc).isoformat(),
        "mode": {
            "subreddit": os.environ.get("BOT_SUBREDDIT", ""),
            "safeMode": os.environ.get("BOT_SAFE_MODE", "true"),
            "expectedBotUsername": os.environ.get("BOT_EXPECTED_USERNAME", ""),
            "runtime": "GitHub Actions" if actions_runtime else "local process",
        },
        "paths": {
            "repoRoot": str(REPO_ROOT),
            "envFile": str(env_path),
            "envFileExists": env_path.exists(),
            "stateFile": str(state_path),
            "stateFileOutsideRepository": state_path_outside_repo,
            "encryptedStateFile": str(encrypted_state_path),
            "statsFile": str(stats_path),
            "statsFileOutsideRepository": stats_path_outside_repo,
            "auditLogFile": str(audit_path),
            "auditLogOutsideRepository": audit_path_outside_repo,
            "prefsFile": str(prefs_path),
            "logDir": str(log_dir()),
            "logFile": str(log_path),
            "logFileExists": log_path.exists(),
            "consoleLogFile": str(console_path),
            "consoleLogFileExists": console_path.exists(),
            "redditRequestBudgetFile": str(request_budget_settings.path),
            "redditRequestBudgetOutsideRepository": request_budget_outside_repo,
            "activityFile": str(activity_path),
            "activityFileOutsideRepository": activity_path_outside_repo,
            "cycleSummaryFile": str(cycle_summary_path),
            "cycleSummaryFileOutsideRepository": cycle_summary_path_outside_repo,
            "communityRulesReferenceFile": str(rules_reference_path),
        },
        "integrations": {
            "activeLlmProvider": provider,
            "activeLlmKeyNames": list(active_llm_key_names),
            "activeLlmModel": effective_llm_model,
            "activeLlmEndpoint": effective_base_url,
            "llmThinkingEnabled": llm_thinking_enabled,
            "llmThinkingSetting": llm_thinking_setting,
            "llmReasoningEffort": llm_reasoning_effort,
            "llmConfigurationReady": llm_configuration_ready,
            "llmConfigurationErrors": llm_configuration_errors,
            "redditRefreshTokenConfigured": reddit_refresh_token_configured,
            "redditClientConfigured": reddit_client_configured,
            "redditCredentialSource": reddit_credential_source,
            "llmKeyConfigured": llm_key_configured,
            "llmCredentialSource": llm_credential_source,
            "localRedditConfigurationReady": local_reddit_credentials_ready if load_environment else None,
            "localLlmConfigurationReady": local_llm_key_ready if load_environment else None,
            "localConfigurationReady": local_configuration_ready,
        },
        "background": {
            "runtime": "github-actions" if actions_runtime else "local",
            "lastKnownStateMarker": last_run,
            "lastCompletedRun": stats.get("last_run") if isinstance(stats, dict) else None,
            "lastRunProven": last_run_proven,
            "lastRunFresh": last_run_recent,
            "lastRunAgeHours": last_run_age_hours,
            "operationalStatus": operational_status,
            "operationalStatusDetail": operational_status_detail,
            "stateBackend": state_backend,
            "runtimeStateReadable": state_readable,
            "runtimeStateOutsideRepository": state_path_outside_repo,
            "runtimeStateReadError": state_read_error,
            "redditRequests": {
                "configuredTimeoutSeconds": _configured_reddit_timeout_seconds(),
                **operational_log["redditRequests"],
                "requestBudget": {
                    **request_budget_settings.to_dict(),
                    "usedRequests": request_budget_snapshot.used_requests,
                    "remainingRequests": request_budget_snapshot.remaining_requests,
                    "resetSeconds": round(request_budget_snapshot.reset_seconds, 1),
                    "readable": request_budget_snapshot.readable,
                    "error": request_budget_snapshot.error,
                    "outsideRepository": request_budget_outside_repo,
                },
            },
            "scheduler": operational_log["scheduler"],
            "moderatorActivity": {
                **get_moderator_activity_status(),
                "boostIntervalSeconds": 3,
                "quietPeriodSeconds": 180,
            },
            "adaptiveScheduling": {
                "enabled": adaptive_enabled,
                "targetItemsPerRun": os.environ.get(
                    "BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN",
                    os.environ.get("BOT_ADAPTIVE_TARGET_COMMENTS_PER_RUN", "3"),
                ),
                "adjustmentPercent": 15,
                "countedSources": [
                    "eligiblePosts",
                    "eligibleComments",
                    "privateChatMessagesHandled",
                ],
                "minimumIntervalSeconds": os.environ.get("BOT_ADAPTIVE_MIN_INTERVAL_SECONDS", "2"),
                "normalIntervalSeconds": os.environ.get("BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS", "60"),
                "maximumIntervalSeconds": os.environ.get("BOT_ADAPTIVE_MAX_INTERVAL_SECONDS", "60"),
                "latestCycleSummary": cycle_summary if isinstance(cycle_summary, dict) else {},
            },
            "communityRules": community_rules_status,
            "flairWrites": {
                "paused": bool(
                    isinstance(state, dict)
                    and isinstance(state.get("reddit_flair_permission_pause"), dict)
                    and state["reddit_flair_permission_pause"].get("blocked") is True
                ),
                "reason": (
                    state.get("reddit_flair_permission_pause", {}).get("reason")
                    if isinstance(state, dict)
                    and isinstance(state.get("reddit_flair_permission_pause"), dict)
                    else None
                ),
                "since": (
                    state.get("reddit_flair_permission_pause", {}).get("since")
                    if isinstance(state, dict)
                    and isinstance(state.get("reddit_flair_permission_pause"), dict)
                    else None
                ),
            },
            "capturedActivity": {
                "updatedAt": activity_snapshot.get("updatedAt") if isinstance(activity_snapshot, dict) else None,
                "itemCount": len(activity_snapshot.get("items", [])) if isinstance(activity_snapshot, dict) and isinstance(activity_snapshot.get("items"), list) else 0,
                "subreddit": activity_snapshot.get("subreddit") if isinstance(activity_snapshot, dict) else None,
            },
            "catchUp": catch_up_metrics,
            "publicResponseSafety": public_response_metrics,
            "replyLimits": reply_limits,
            "reportedUserReview": {
                "reportsSeen": review_metrics.get("reports_seen", 0),
                "usersReviewed": review_metrics.get("users_reviewed", 0),
                "reportsSkipped": review_metrics.get("reports_skipped", 0),
                "reviewsCompleted": review_metrics.get("reviews_completed", 0),
                "retryableFailures": review_metrics.get("retryable_failures", 0),
                "lastReportScan": review_metrics.get("last_report_scan"),
                "lastSuccessfulReview": review_metrics.get("last_successful_review"),
                "lastModeratorNotification": review_metrics.get("last_moderator_notification"),
                "safeModePreventedDelivery": review_metrics.get("safe_mode_prevented_delivery", 0),
                "ruleReferenceWarnings": review_metrics.get("rule_reference_warnings", []),
            },
            "historyReviewLedger": {
                "available": history_ledger_metrics.get("ledger_available", False),
                "reviewsCompleted": history_ledger_metrics.get("reviews_completed", 0),
                "failedJevAttempts": history_ledger_metrics.get("failed_jev_attempts", 0),
                "incompleteJevAttempts": history_ledger_metrics.get("incomplete_jev_attempts", 0),
                "skippedDuplicateItems": history_ledger_metrics.get("skipped_duplicate_items", 0),
                "skippedNoNewHistory": history_ledger_metrics.get("skipped_no_new_history", 0),
                "historyGaps": history_ledger_metrics.get("history_gaps", 0),
            },
        },
        "recentErrors": _recent_audit_errors(),
        "recentLogLines": _tail_log_lines(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump Optimist Prime health diagnostics")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--environment-path",
        default=None,
        help="local .env path to load for this diagnostics process",
    )
    args = parser.parse_args(argv)
    payload = dump_diagnostics(
        load_environment=True,
        environment_path=args.environment_path,
    )
    if args.pretty:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        json.dump(payload, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
