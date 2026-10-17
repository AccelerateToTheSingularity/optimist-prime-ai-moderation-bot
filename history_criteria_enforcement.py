"""Optional history-criterion enforcement actions."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import config
from history_gate_jev import configured_history_criteria_ids
from moderation_rules import history_criterion_score_is_current


_STATE_KEY = "history_criteria_ban_actions"
_REVIEW_STATE_KEY = "history_criteria_reviews"
_NOTIFICATION_STATE_KEY = "reported_user_reviews"
_GENERIC_BAN_MESSAGE = (
    "You have been banned from this subreddit after a history review matched "
    "moderator-configured criteria. If you think this was a mistake, contact the moderators."
)
_GENERIC_BAN_REASON = "Matched moderator-configured history criteria"


@dataclass(frozen=True)
class HistoryCriteriaBanOutcome:
    status: str
    threshold_percent: int
    triggering_scores: tuple[tuple[str, float], ...]
    error_type: str | None = None

    @property
    def action_taken(self) -> bool:
        return self.status == "banned"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def triggering_history_criteria_scores(
    assessment: Any,
    *,
    threshold_percent: int | None = None,
) -> tuple[tuple[str, float], ...]:
    """Return configured criteria scores strictly above the ban percentage."""
    configured_threshold = (
        config.HISTORY_REVIEW_BAN_THRESHOLD_PERCENT
        if threshold_percent is None else threshold_percent
    )
    if configured_threshold is None:
        return ()
    try:
        configured_threshold = int(configured_threshold)
    except (TypeError, ValueError):
        return ()
    if not 1 <= configured_threshold <= 100:
        return ()
    threshold = configured_threshold / 100.0
    probabilities = getattr(assessment, "criterion_probabilities", {})
    if not isinstance(probabilities, dict):
        return ()
    rule_order = {
        name: index
        for index, name in enumerate(getattr(assessment, "rule_names", ()))
    }
    matches: list[tuple[str, float]] = []
    for name, raw_score in probabilities.items():
        if (
            name not in configured_history_criteria_ids()
            or isinstance(raw_score, bool)
            or not history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
        ):
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and threshold < score <= 1.0:
            matches.append((name, score))
    return tuple(sorted(matches, key=lambda item: (-item[1], rule_order.get(item[0], 10_000), item[0])))


def _action_key(username: str, source_key: str) -> str:
    return _stable_key(username, source_key)


def history_autoban_action_key(username: str, source_key: str) -> str:
    """Return the stable state key for one score-based ban action."""
    return _action_key(username, source_key)


def history_autoban_notification_key(
    username: str, source_key: str, outcome_status: str = "banned"
) -> str:
    """Return the stable internal-Modmail key for one score-based ban action."""
    return f"autoban:{_action_key(username, source_key)[:20]}:{outcome_status}"


def _persist(persist_state, state: dict) -> None:
    if persist_state is not None:
        persist_state(state)


def _stable_key(username: str, source_key: str) -> str:
    value = chr(10).join((username.casefold(), str(source_key or "").strip()))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_history_assessment(
    state: dict,
    username: str,
    source_key: str,
    assessment: Any,
    *,
    history_item_count: int,
    history_incomplete: bool,
    persist_state=None,
) -> str:
    """Persist score-only results so delivery retries do not repeat Jev."""
    key = _stable_key(username, source_key)
    records = state.setdefault(_REVIEW_STATE_KEY, {})
    if not isinstance(records, dict):
        records = {}
        state[_REVIEW_STATE_KEY] = records
    raw_probabilities = getattr(assessment, "criterion_probabilities", {}) or {}
    revisions = getattr(assessment, "criterion_score_revisions", {}) or {}
    if not isinstance(revisions, dict):
        revisions = {}
    probabilities = (
        {
            str(name): score
            for name, score in raw_probabilities.items()
            if str(name) in configured_history_criteria_ids() and history_criterion_score_is_current(str(name), revisions)
        }
        if isinstance(raw_probabilities, dict)
        else {}
    )
    records[key] = {
        "username": str(username),
        "criterion_probabilities": probabilities,
        "criterion_score_revisions": {
            name: revisions.get(name, 1) for name in probabilities
        },
        "outdated_criterion_ids": sorted({
            str(name) for name in raw_probabilities
            if str(name) in configured_history_criteria_ids()
            and not history_criterion_score_is_current(str(name), revisions)
        } | {
            str(name) for name in (getattr(assessment, "outdated_criterion_ids", ()) or ())
            if str(name) in configured_history_criteria_ids() and str(name) not in probabilities
        }),
        "matched_rules": [
            str(name) for name in (getattr(assessment, "matched_rules", ()) or ())
            if str(name) in probabilities
        ],
        "rule_names": [
            str(name) for name in (getattr(assessment, "rule_names", ()) or ())
            if str(name) in configured_history_criteria_ids()
        ],
        "match_threshold": getattr(assessment, "match_threshold", None),
        "scores_complete": bool(getattr(assessment, "scores_complete", True)),
        "missing_criterion_ids": [
            str(name) for name in (getattr(assessment, "missing_criterion_ids", ()) or ())
            if str(name) in configured_history_criteria_ids()
        ],
        "evidence_truncated_items": int(getattr(assessment, "evidence_truncated_items", 0) or 0),
        "history_item_count": max(0, int(history_item_count)),
        "history_incomplete": bool(history_incomplete),
        "served_model": getattr(assessment, "served_model", None),
        "updated_at": _now_iso(),
    }
    if len(records) > 500:
        ordered = sorted(
            records.items(),
            key=lambda pair: str(pair[1].get("updated_at", "")) if isinstance(pair[1], dict) else "",
        )
        for old_key, _record in ordered[: len(records) - 500]:
            records.pop(old_key, None)
    _persist(persist_state, state)
    return key


def load_history_assessment(state: dict, username: str, source_key: str):
    """Restore saved criterion scores for this same review event, if present."""
    records = state.get(_REVIEW_STATE_KEY, {})
    record = records.get(_stable_key(username, source_key)) if isinstance(records, dict) else None
    if not isinstance(record, dict) or str(record.get("username", "")).casefold() != username.casefold():
        return None
    return _assessment_from_record(record)


def _assessment_from_record(record: dict):
    probabilities = record.get("criterion_probabilities", {})
    if not isinstance(probabilities, dict):
        return None
    from history_gate_jev import HistoryGateAssessment

    try:
        revisions = record.get("criterion_score_revisions", {})
        if not isinstance(revisions, dict):
            revisions = {}
        active_probabilities: dict[str, float] = {}
        for raw_name, raw_score in probabilities.items():
            name = str(raw_name)
            if (
                name not in configured_history_criteria_ids()
                or isinstance(raw_score, bool)
                or not history_criterion_score_is_current(name, revisions)
            ):
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score) and 0.0 <= score <= 1.0:
                active_probabilities[name] = score
        raw_match_threshold = record.get("match_threshold")
        match_threshold = float(raw_match_threshold) if raw_match_threshold is not None else None
        saved_rule_names = tuple(
            name for name in (str(value) for value in record.get("rule_names", ()))
            if name in configured_history_criteria_ids()
        )
        rule_names = tuple(dict.fromkeys((
            *saved_rule_names,
            *sorted(configured_history_criteria_ids() - set(saved_rule_names)),
        )))
        missing_criterion_ids = tuple(
            name for name in sorted(configured_history_criteria_ids())
            if name not in active_probabilities
        )
        outdated_criterion_ids = tuple(sorted({
            str(name) for name in probabilities
            if str(name) in configured_history_criteria_ids()
            and not history_criterion_score_is_current(str(name), revisions)
        } | {
            str(name) for name in record.get("outdated_criterion_ids", ())
            if str(name) in configured_history_criteria_ids() and str(name) not in active_probabilities
        }))
        assessment = HistoryGateAssessment(
            rule_names=rule_names,
            bundle_size=0,
            bundles=(),
            matched_rules=tuple(
                name for name, score in active_probabilities.items()
                if match_threshold is not None and score >= match_threshold
            ),
            calls=0,
            input_tokens=0,
            cost=0.0,
            served_model=record.get("served_model"),
            classifier="jev_full_history_noul",
            item_attribution_available=False,
            criterion_probabilities=active_probabilities,
            match_threshold=match_threshold,
            evidence_truncated_items=int(record.get("evidence_truncated_items", 0) or 0),
            scores_complete=not missing_criterion_ids,
            missing_criterion_ids=missing_criterion_ids,
            criterion_score_revisions={
                name: revisions.get(name, 1) for name in active_probabilities
            },
            outdated_criterion_ids=outdated_criterion_ids,
        )
        return (
            assessment,
            max(0, int(record.get("history_item_count", 0) or 0)),
            bool(record.get("history_incomplete", False)),
        )
    except (TypeError, ValueError):
        return None


def latest_history_assessment(state: dict, username: str):
    """Return the most recent saved profile result for a user, if available."""
    records = state.get(_REVIEW_STATE_KEY, {}) if isinstance(state, dict) else {}
    target = str(username or "").strip().casefold()
    if not target or not isinstance(records, dict):
        return None
    candidates = [
        (str(record.get("updated_at") or ""), str(key))
        for key, record in records.items()
        if isinstance(record, dict)
        and str(record.get("username") or "").strip().casefold() == target
    ]
    for updated_at, source_key in sorted(candidates, reverse=True):
        record = records.get(source_key)
        loaded = _assessment_from_record(record) if isinstance(record, dict) else None
        if loaded is not None:
            return (*loaded, updated_at)
    return None


def queue_review_modmail(
    state: dict,
    *,
    notification_key: str,
    username: str,
    subject: str,
    body: str,
    report_keys: list[str] | None = None,
    action_key: str | None = None,
    persist_state=None,
) -> str:
    """Save an internal-Modmail notice before attempting its delivery."""
    root = state.setdefault(_NOTIFICATION_STATE_KEY, {})
    if not isinstance(root, dict):
        root = {}
        state[_NOTIFICATION_STATE_KEY] = root
    pending = root.setdefault("pending_notifications", {})
    sent = root.setdefault("sent_notifications", {})
    if not isinstance(pending, dict):
        pending = {}
        root["pending_notifications"] = pending
    if not isinstance(sent, dict):
        sent = {}
        root["sent_notifications"] = sent
    key = str(notification_key or "").strip()
    if not key:
        raise ValueError("notification_key is required")
    if key not in sent:
        pending.setdefault(key, {
            "username": str(username),
            "subject": str(subject),
            "body": str(body),
            "report_keys": list(report_keys or ()),
            "action_key": str(action_key or "").strip() or None,
            "created_at": _now_iso(),
        })
    _persist(persist_state, state)
    return key


def deliver_review_modmail(subreddit, state: dict, notification_key: str, *, persist_state=None) -> bool:
    """Send one pending internal notice; leave it queued on any delivery error."""
    root = state.get(_NOTIFICATION_STATE_KEY, {})
    pending = root.get("pending_notifications", {}) if isinstance(root, dict) else {}
    if not isinstance(pending, dict):
        return False
    record = pending.get(notification_key)
    if not isinstance(record, dict):
        return False
    from modmail_support import send_internal_notification

    send_internal_notification(
        subreddit,
        str(record.get("subject") or "History criteria review"),
        str(record.get("body") or ""),
    )
    pending.pop(notification_key, None)
    sent = root.setdefault("sent_notifications", {})
    if not isinstance(sent, dict):
        sent = {}
        root["sent_notifications"] = sent
    sent[notification_key] = {"sent_at": _now_iso(), "subject": record.get("subject", "")}
    if len(sent) > 1000:
        oldest = sorted(sent.items(), key=lambda pair: str(pair[1].get("sent_at", "")) if isinstance(pair[1], dict) else "")
        for old_key, _value in oldest[: len(sent) - 1000]:
            sent.pop(old_key, None)
    action_key = str(record.get("action_key") or "").strip()
    action_map = state.get(_STATE_KEY, {})
    action_record = action_map.get(action_key) if isinstance(action_map, dict) else None
    if isinstance(action_record, dict):
        action_record["moderator_notice_status"] = "sent"
        action_record["moderator_notice_sent_at"] = _now_iso()
    _persist(persist_state, state)
    return True


def reconcile_pending_autoban_notices(subreddit, state: dict, *, persist_state=None) -> int:
    """Recover interrupted bans and finish their moderator notices after restart."""
    action_map = state.get(_STATE_KEY, {})
    if not isinstance(action_map, dict):
        return 0
    root = state.get(_NOTIFICATION_STATE_KEY, {})
    pending = root.get("pending_notifications", {}) if isinstance(root, dict) else {}
    sent = root.get("sent_notifications", {}) if isinstance(root, dict) else {}
    if not isinstance(pending, dict):
        pending = {}
    if not isinstance(sent, dict):
        sent = {}
    delivered = 0
    for key, record in list(action_map.items()):
        if not isinstance(record, dict):
            continue
        username = str(record.get("username") or "").strip()
        source_key = str(record.get("source_key") or "").strip()
        current_status = str(record.get("status") or "")
        if current_status in {"checking_existing_ban", "ban_pending", "retryable"}:
            if not username or not source_key:
                continue
            scores = record.get("triggering_scores", [])
            try:
                current_scores = [
                    (str(name), float(score)) for name, score in scores
                    if str(name) in configured_history_criteria_ids()
                    and history_criterion_score_is_current(str(name), record.get("criterion_score_revisions", {}))
                ]
                if len(current_scores) != len(scores):
                    record["triggering_scores"] = [[name, score] for name, score in current_scores]
                    if not current_scores:
                        record.update({
                            "status": "cancelled_criterion_definition_changed",
                            "updated_at": _now_iso(),
                        })
                        _persist(persist_state, state)
                        continue
                scores = current_scores
                assessment = type("SavedScoreAssessment", (), {
                    "criterion_probabilities": {str(name): float(score) for name, score in scores},
                    "rule_names": tuple(str(name) for name, _score in scores),
                    "criterion_score_revisions": record.get("criterion_score_revisions", {}),
                    # A pending action is created only after the complete
                    # assessment passed the eligibility checks. Restore that
                    # fact so a restart can safely finish the interrupted ban.
                    "scores_complete": True,
                    "missing_criterion_ids": (),
                    "outdated_criterion_ids": (),
                })()
            except (TypeError, ValueError):
                continue
            if bool(record.get("history_incomplete", False)):
                record.update({
                    "status": "cancelled_incomplete_review",
                    "updated_at": _now_iso(),
                })
                _persist(persist_state, state)
                continue
            if (
                not config.AUTOMATED_BAN_ENABLED
                or config.SAFE_MODE
                or config.HISTORY_REVIEW_BAN_THRESHOLD_PERCENT is None
            ):
                try:
                    already_banned = next(iter(subreddit.banned(redditor=username, limit=1)), None) is not None
                except Exception:
                    continue
                if already_banned:
                    outcome = HistoryCriteriaBanOutcome(
                        "already_banned",
                        int(record.get("threshold_percent", config.HISTORY_REVIEW_BAN_THRESHOLD_PERCENT or 0)),
                        tuple((str(name), float(score)) for name, score in scores),
                    )
                    record.update({"status": "already_banned", "updated_at": _now_iso()})
                else:
                    record.update({"status": "cancelled_disabled", "updated_at": _now_iso()})
                    _persist(persist_state, state)
                    continue
            elif not triggering_history_criteria_scores(assessment):
                record.update({
                    "status": "cancelled_threshold_changed",
                    "updated_at": _now_iso(),
                })
                _persist(persist_state, state)
                continue
            else:
                outcome = apply_history_criteria_ban(
                    subreddit, username, assessment, state,
                    source_key=source_key, persist_state=persist_state,
                )
                record = action_map.get(key, record)
            if outcome.status in {"ban_failed", "ban_check_failed"}:
                failure_key = history_autoban_notification_key(username, source_key, outcome.status)
                if failure_key not in sent and failure_key not in pending:
                    queue_review_modmail(
                        state,
                        notification_key=failure_key,
                        username=username,
                        subject=f"History auto-ban needs attention: u/{username} [{failure_key[-24:]}]",
                        body=format_autoban_notice(
                            username, outcome,
                            reference_links=record.get("notice_links", ()),
                        ),
                        action_key=key,
                        persist_state=persist_state,
                    )
                    try:
                        if deliver_review_modmail(subreddit, state, failure_key, persist_state=persist_state):
                            delivered += 1
                    except Exception as exc:
                        print(f"    Auto-ban notice is queued for retry ({type(exc).__name__}).")
                continue
        if record.get("status") not in {"banned", "already_banned"}:
            continue
        if record.get("moderator_notice_status") == "sent":
            continue
        notification_key = str(record.get("notification_key") or "").strip()
        if not notification_key:
            notification_key = history_autoban_notification_key(
                username, source_key,
                str(record.get("status") or "banned"),
            )
            record["notification_key"] = notification_key
        if notification_key in sent:
            record["moderator_notice_status"] = "sent"
            continue
        if notification_key in pending:
            continue
        scores = record.get("triggering_scores", [])
        try:
            outcome = HistoryCriteriaBanOutcome(
                str(record["status"]),
                int(record.get("threshold_percent", config.HISTORY_REVIEW_BAN_THRESHOLD_PERCENT or 0)),
                tuple((str(name), float(score)) for name, score in scores),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not username or not outcome.triggering_scores:
            continue
        subject = f"History auto-ban result: u/{username} [{notification_key[-24:]}]"
        body = format_autoban_notice(
            username, outcome,
            reference_links=record.get("notice_links", ()),
        )
        queue_review_modmail(
            state,
            notification_key=notification_key,
            username=username,
            subject=subject,
            body=body,
            action_key=str(key),
            persist_state=persist_state,
        )
        try:
            if deliver_review_modmail(
                subreddit, state, notification_key, persist_state=persist_state
            ):
                delivered += 1
        except Exception as exc:
            print(f"    Auto-ban notice is queued for retry ({type(exc).__name__}).")
    if delivered:
        _persist(persist_state, state)
    return delivered


def apply_history_criteria_ban(
    subreddit,
    username: str,
    assessment: Any,
    state: dict,
    *,
    source_key: str,
    dry_run: bool = False,
    history_complete: bool = True,
    notice_links: list[str] | None = None,
    persist_state=None,
) -> HistoryCriteriaBanOutcome:
    """Apply one permanent ban, guarded by the live ban list and durable state.

    A complete history listing and complete score set are required before this
    high-impact action can proceed.
    """
    threshold = config.HISTORY_REVIEW_BAN_THRESHOLD_PERCENT
    if threshold is None:
        return HistoryCriteriaBanOutcome("disabled", 0, ())
    matches = triggering_history_criteria_scores(assessment, threshold_percent=threshold)
    if not config.AUTOMATED_BAN_ENABLED or config.SAFE_MODE:
        return HistoryCriteriaBanOutcome("disabled", threshold, matches)
    if (
        not history_complete
        or not bool(getattr(assessment, "scores_complete", False))
        or bool(getattr(assessment, "missing_criterion_ids", ()))
        or bool(getattr(assessment, "outdated_criterion_ids", ()))
    ):
        return HistoryCriteriaBanOutcome("not_eligible", threshold, ())
    if not matches:
        return HistoryCriteriaBanOutcome("not_eligible", threshold, ())
    if dry_run:
        return HistoryCriteriaBanOutcome("would_ban", threshold, matches)

    target = str(username or "").strip()
    if not target:
        return HistoryCriteriaBanOutcome("invalid_target", threshold, matches)
    stable_source = str(source_key or "").strip() or "history-review"
    key = _action_key(target, stable_source)
    action_map = state.setdefault(_STATE_KEY, {})
    if not isinstance(action_map, dict):
        action_map = {}
        state[_STATE_KEY] = action_map
    existing = action_map.get(key)
    if isinstance(existing, dict) and existing.get("status") in {"banned", "already_banned"}:
        return HistoryCriteriaBanOutcome(
            str(existing["status"]),
            int(existing.get("threshold_percent", threshold)),
            tuple(
                (str(name), float(score))
                for name, score in existing.get("triggering_scores", [])
                if isinstance(name, str) and isinstance(score, (int, float))
            ),
        )

    record = dict(existing) if isinstance(existing, dict) else {}
    record.update({
        "username": target,
        "source_key": stable_source,
        "status": "checking_existing_ban",
        "threshold_percent": threshold,
        "triggering_scores": [[name, score] for name, score in matches],
        "criterion_score_revisions": {
            name: (getattr(assessment, "criterion_score_revisions", {}) or {}).get(name, 1)
            for name, _score in matches
        },
        "notification_key": history_autoban_notification_key(
            target, stable_source, "checking"
        ),
        "moderator_notice_status": "pending",
        "updated_at": _now_iso(),
    })
    if notice_links is not None:
        record["notice_links"] = [
            str(link).strip() for link in notice_links if str(link).strip()
        ]
    action_map[key] = record
    try:
        _persist(persist_state, state)
        already_banned = next(iter(subreddit.banned(redditor=target, limit=1)), None) is not None
    except Exception as exc:
        record.update({
            "status": "retryable",
            "notification_key": history_autoban_notification_key(target, stable_source, "ban_check_failed"),
            "moderator_notice_status": "pending",
            "last_error_type": type(exc).__name__,
            "updated_at": _now_iso(),
        })
        _persist(persist_state, state)
        return HistoryCriteriaBanOutcome("ban_check_failed", threshold, matches, type(exc).__name__)

    if already_banned:
        record.update({
            "status": "already_banned",
            "notification_key": history_autoban_notification_key(target, stable_source, "already_banned"),
            "moderator_notice_status": "pending",
            "updated_at": _now_iso(),
        })
        record.pop("last_error_type", None)
        _persist(persist_state, state)
        return HistoryCriteriaBanOutcome("already_banned", threshold, matches)

    record.update({"status": "ban_pending", "updated_at": _now_iso()})
    _persist(persist_state, state)
    try:
        from reddit_observability import single_write_attempt

        with single_write_attempt():
            subreddit.banned.add(
                target,
                ban_reason=_GENERIC_BAN_REASON,
                ban_message=_GENERIC_BAN_MESSAGE,
            )
    except Exception as exc:
        # A later cycle checks Reddit's ban list before attempting this same
        # action again, covering an API timeout after Reddit accepted the ban.
        record.update({
            "status": "retryable",
            "notification_key": history_autoban_notification_key(target, stable_source, "ban_failed"),
            "moderator_notice_status": "pending",
            "last_error_type": type(exc).__name__,
            "updated_at": _now_iso(),
        })
        _persist(persist_state, state)
        return HistoryCriteriaBanOutcome("ban_failed", threshold, matches, type(exc).__name__)

    record.update({
        "status": "banned",
        "notification_key": history_autoban_notification_key(target, stable_source, "banned"),
        "moderator_notice_status": "pending",
        "banned_at": _now_iso(),
        "updated_at": _now_iso(),
    })
    record.pop("last_error_type", None)
    if len(action_map) > 2000:
        ordered = sorted(
            action_map.items(),
            key=lambda pair: str(pair[1].get("updated_at", "")) if isinstance(pair[1], dict) else "",
        )
        for old_key, _value in ordered[: len(action_map) - 2000]:
            action_map.pop(old_key, None)
    _persist(persist_state, state)
    return HistoryCriteriaBanOutcome("banned", threshold, matches)


def display_criterion_name(value: str) -> str:
    import re

    normalized = str(value).strip().casefold()
    name = re.sub(r"(?:[_\s]+detector)$", "", str(value).strip(), flags=re.IGNORECASE)
    words = re.sub(r"[_-]+", " ", name).split()
    return " ".join(
        word.upper() if word.casefold() in {"ai", "agi"}
        else word[:1].upper() + word[1:]
        for word in words
    )


def format_history_criteria_matches(assessment: Any, *, minimum_score: float | None = None) -> str:
    """List every criterion at or above the review line with its score."""
    threshold = (
        getattr(assessment, "match_threshold", None)
        if minimum_score is None else minimum_score
    )
    if threshold is None:
        return ""
    threshold = float(threshold)
    probabilities = getattr(assessment, "criterion_probabilities", {})
    order = {
        name: index for index, name in enumerate(getattr(assessment, "rule_names", ()))
    }
    matches = []
    if isinstance(probabilities, dict):
        for name, raw in probabilities.items():
            if (
                name not in configured_history_criteria_ids()
                or isinstance(raw, bool)
                or not history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
            ):
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score) and threshold <= score <= 1:
                matches.append((name, score))
    matches.sort(key=lambda item: (-item[1], order.get(item[0], 10_000), item[0]))
    return ", ".join(
        f"{display_criterion_name(name)} ({round(score * 100)}%)"
        for name, score in matches
    )


def format_autoban_notice(
    username: str,
    outcome: HistoryCriteriaBanOutcome,
    *,
    reference_links: list[str] | tuple[str, ...] = (),
) -> str:
    matches = ", ".join(
        f"{display_criterion_name(name)} ({round(score * 100)}%)"
        for name, score in outcome.triggering_scores
    )
    if outcome.status == "banned":
        message = (
            f"u/{username} was banned after a history review matched moderator-configured criteria. "
            f"Matched criteria: {matches}."
        )
    elif outcome.status == "already_banned":
        message = (
            f"u/{username} was already banned; no new ban was made. "
            f"Matched criteria: {matches}."
        )
    elif outcome.status == "would_ban":
        message = f"[DRY RUN] Would ban u/{username}. Matched criteria: {matches}."
    elif outcome.status in {"ban_check_failed", "ban_failed", "invalid_target"}:
        message = (
            f"Configured criteria matched u/{username} above the {outcome.threshold_percent}% ban setting "
            f"({matches}), but Reddit did not confirm a ban. A moderator should check the ban list."
        )
    else:
        message = f"No automatic ban was made for u/{username}."
    links = [str(link).strip() for link in reference_links if str(link).strip()]
    if links:
        message += "\n\nRelated Reddit item(s):\n" + "\n".join(links)
    return message
