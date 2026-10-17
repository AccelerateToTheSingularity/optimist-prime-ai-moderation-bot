"""Idempotent writes for native subreddit moderator review notes.

Reddit's moderator-note endpoint does not expose an idempotency key.  Each
write therefore carries a stable marker, and the local state records a write
intent before the POST so an interrupted request can be reconciled later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any


_STATE_KEY = "moderator_review_notes"
_MAX_NOTE_LENGTH = 250
_MAX_RECONCILE_CHECKS = 5
_MAX_RECONCILES_PER_CALL = 25
_REMOTE_NOTE_LIMIT = 100
_MAX_COMPLETED_RECORDS = 10_000
_LABEL = "USER_SUMMARY"
_logger = logging.getLogger(__name__)


def _warn_status(status: str, error_type: str | None = None) -> None:
    if error_type:
        _logger.warning(
            "Native moderator review note status=%s error_type=%s",
            status,
            error_type,
        )
    else:
        _logger.warning("Native moderator review note status=%s", status)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _subreddit_name(subreddit: Any) -> str:
    """Return the configured subreddit name without hydrating a lazy object."""
    value = getattr(subreddit, "display_name", None)
    if not value:
        try:
            value = str(subreddit)
        except Exception:
            return ""
    name = str(value or "").strip()
    if name.casefold().startswith("r/"):
        name = name[2:]
    return name


def _review_marker(
    username: str,
    subreddit_name: str,
    review_key: str,
    thing: str | None = None,
) -> str:
    identity = [username.casefold(), subreddit_name.casefold(), review_key]
    # Keep existing markers stable for unlinked notes; linked notes have a
    # separate marker per Reddit item so each report can be reconciled safely.
    if thing:
        identity.append(thing)
    payload = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"[review-ref:{digest}]"


def _notes_interface(subreddit: Any):
    try:
        notes = subreddit.mod.notes
    except Exception:
        return None
    if not callable(getattr(notes, "create", None)):
        return None
    if not callable(getattr(notes, "redditors", None)):
        return None
    return notes


def _root_and_records(state: dict) -> dict[str, dict[str, Any]]:
    root = state.get(_STATE_KEY)
    if not isinstance(root, dict):
        root = {}
        state[_STATE_KEY] = root
    records = root.get("records")
    if not isinstance(records, dict):
        records = {}
        root["records"] = records
    return records


def _persist(persist_state, state: dict) -> None:
    if not callable(persist_state):
        _warn_status("skipped_no_durable_store", "PersistenceCallbackUnavailable")
        raise ValueError("durable state persistence is required before a moderator-note write")
    try:
        result = persist_state(state)
    except Exception as exc:
        _warn_status("failed", type(exc).__name__)
        raise
    if result is False:
        _warn_status("failed", "PersistenceRejected")
        raise RuntimeError("moderator-note state could not be persisted")


def _note_id(note: Any) -> str:
    return str(getattr(note, "id", "") or "").strip()


def _note_text(note: Any) -> str:
    return str(getattr(note, "note", "") or "")


def _matching_note(notes_interface, record: dict[str, Any]):
    """Find this exact operation among a bounded window of the user's notes."""
    username = str(record.get("username") or "")
    marker = str(record.get("marker") or "")
    if not username or not marker:
        return None
    listing = notes_interface.redditors(
        username,
        all_notes=True,
        limit=_REMOTE_NOTE_LIMIT,
    )
    for note in listing:
        note_id = _note_id(note)
        note_text = _note_text(note)
        if note_id and marker in note_text:
            return note
    return None


def _set_delivered(record: dict[str, Any], note: Any) -> None:
    record.update({
        "status": "delivered",
        "note_id": _note_id(note),
        "updated_at": _now_iso(),
        "last_error_type": None,
    })


def _trim_completed_records(records: dict[str, dict[str, Any]]) -> None:
    """Retain completed dedup keys beyond the callers' replay windows.

    Pending and uncertain operations are never evicted because doing so could
    allow an uncertain Reddit POST to be repeated.
    """
    completed = [
        (str(record.get("created_at") or ""), marker)
        for marker, record in records.items()
        if isinstance(record, dict)
        and record.get("status") in {"delivered", "failed"}
    ]
    excess = len(completed) - _MAX_COMPLETED_RECORDS
    if excess <= 0:
        return
    for _created_at, marker in sorted(completed)[:excess]:
        records.pop(marker, None)


def _status_after_check(
    state: dict,
    record: dict[str, Any],
    notes_interface,
    persist_state,
) -> str:
    """Reconcile one possibly-written note without resending an uncertain POST."""
    checks = int(record.get("reconcile_checks", 0) or 0)
    if checks >= _MAX_RECONCILE_CHECKS:
        record["status"] = "needs_attention"
        record["updated_at"] = _now_iso()
        _warn_status("needs_attention", str(record.get("last_error_type") or "Unknown"))
        _persist(persist_state, state)
        return "needs_attention"

    record["reconcile_checks"] = checks + 1
    record["status"] = "uncertain"
    record["updated_at"] = _now_iso()
    # Save the bounded-check counter before the Reddit read as well.  A process
    # restart must not reset an unresolved note into an unbounded polling loop.
    _persist(persist_state, state)
    try:
        note = _matching_note(notes_interface, record)
    except Exception as exc:
        record["last_error_type"] = type(exc).__name__
        record["updated_at"] = _now_iso()
        _warn_status("uncertain", type(exc).__name__)
        _persist(persist_state, state)
        return "uncertain"

    # The operation marker is the durable idempotency key. Reddit may
    # normalize note text, so marker visibility confirms reconciliation even
    # when the returned representation differs from the submitted body.
    if note is not None:
        _set_delivered(record, note)
        _trim_completed_records(_root_and_records(state))
        _persist(persist_state, state)
        return "delivered"

    record["last_error_type"] = "NoteNotVisibleYet"
    record["updated_at"] = _now_iso()
    _warn_status("uncertain", "NoteNotVisibleYet")
    if record["reconcile_checks"] >= _MAX_RECONCILE_CHECKS:
        record["status"] = "needs_attention"
        _warn_status("needs_attention", "NoteNotVisibleYet")
        _persist(persist_state, state)
        return "needs_attention"
    _persist(persist_state, state)
    return "uncertain"


def _is_definite_rejection(exc: Exception) -> bool:
    """Identify responses that explicitly rejected the create request."""
    if type(exc).__name__ == "RedditAPIException":
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        return False
    return 400 <= status_code < 500


def record_review_note(
    subreddit,
    state: dict,
    *,
    username: str,
    review_key: str,
    result_text: str,
    thing: str | None = None,
    dry_run: bool = False,
    persist_state=None,
) -> str:
    """Write one native moderator note, returning its delivery state.

    The same ``review_key`` maps to the same marker across callers and process
    restarts.  If an existing intent is unresolved, this function only checks
    for its marker; it never repeats that uncertain POST.
    """
    if dry_run:
        return "skipped_dry_run"
    if not callable(persist_state):
        _warn_status("skipped_no_durable_store", "PersistenceCallbackUnavailable")
        return "skipped_no_durable_store"

    target = str(username or "").strip()
    key = str(review_key or "").strip()
    result = " ".join(str(result_text or "").split())
    community = _subreddit_name(subreddit)
    linked_thing = str(thing or "").strip()
    if not target or not key or not result or not community:
        return "invalid"
    if linked_thing and not re.fullmatch(r"t[13]_[A-Za-z0-9]+", linked_thing):
        return "invalid"

    notes_interface = _notes_interface(subreddit)
    if notes_interface is None:
        _warn_status("skipped_unavailable", "NativeNotesInterfaceUnavailable")
        return "skipped_unavailable"

    marker = _review_marker(target, community, key, linked_thing or None)
    body = f"{result} {marker}"
    if len(body) > _MAX_NOTE_LENGTH:
        return "invalid"

    records = _root_and_records(state)
    existing = records.get(marker)
    if isinstance(existing, dict):
        status = str(existing.get("status") or "")
        if status == "delivered":
            return "already_delivered"
        if status == "failed":
            _warn_status("failed", str(existing.get("last_error_type") or "Unknown"))
            return "failed"
        return _status_after_check(state, existing, notes_interface, persist_state)

    record = {
        "username": target,
        "subreddit": community,
        "marker": marker,
        "body": body,
        "thing": linked_thing or None,
        "status": "prepared",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "note_id": None,
        "write_attempts": 0,
        "reconcile_checks": 0,
        "last_error_type": None,
    }
    records[marker] = record
    # A failure here must escape before the Reddit POST.  The caller supplies
    # the same state-store hook used by the normal worker/action path.
    _persist(persist_state, state)

    record["write_attempts"] = 1
    record["updated_at"] = _now_iso()
    try:
        # Keep startup imports light; all actual note writes still use the
        # existing observable client and its shared request budget.
        from reddit_observability import single_write_attempt

        with single_write_attempt():
            create_kwargs = {
                "redditor": target,
                "note": body,
                "label": _LABEL,
            }
            if linked_thing:
                create_kwargs["thing"] = linked_thing
            note = notes_interface.create(**create_kwargs)
    except Exception as exc:
        record["last_error_type"] = type(exc).__name__
        record["updated_at"] = _now_iso()
        if _is_definite_rejection(exc):
            record["status"] = "failed"
            _warn_status("failed", type(exc).__name__)
            _trim_completed_records(records)
            _persist(persist_state, state)
            return "failed"
        record["status"] = "uncertain"
        _warn_status("uncertain", type(exc).__name__)
        _persist(persist_state, state)
        return "uncertain"

    if not _note_id(note) or _note_text(note) != body:
        record["status"] = "uncertain"
        record["last_error_type"] = "UnverifiedCreateResponse"
        record["updated_at"] = _now_iso()
        _warn_status("uncertain", "UnverifiedCreateResponse")
        _persist(persist_state, state)
        return "uncertain"

    _set_delivered(record, note)
    _trim_completed_records(records)
    _persist(persist_state, state)
    return "delivered"


def reconcile_review_notes(
    subreddit,
    state: dict,
    *,
    dry_run: bool = False,
    persist_state=None,
) -> dict[str, int]:
    """Check pending note markers and record confirmed deliveries.

    Reconciliation is read-only with respect to Reddit.  A missing marker or a
    failed listing stays uncertain because Reddit may expose a successful
    write after a delay.  Repeated checks are bounded per note and per call.
    """
    counts = {
        "checked": 0,
        "delivered": 0,
        "uncertain": 0,
        "needs_attention": 0,
        "failed": 0,
        "skipped": 0,
    }
    records = _root_and_records(state)
    pending = [
        (marker, record)
        for marker, record in records.items()
        if isinstance(record, dict)
        and record.get("status") in {"prepared", "uncertain"}
    ]
    if dry_run or not callable(persist_state):
        counts["skipped"] = len(pending)
        if pending and not dry_run:
            _warn_status("skipped_no_durable_store", "PersistenceCallbackUnavailable")
        return counts

    notes_interface = _notes_interface(subreddit)
    if notes_interface is None:
        counts["skipped"] = len(pending)
        if pending:
            _warn_status("skipped_unavailable", "NativeNotesInterfaceUnavailable")
        return counts

    community = _subreddit_name(subreddit).casefold()
    for _marker, record in pending[:_MAX_RECONCILES_PER_CALL]:
        if str(record.get("subreddit") or "").casefold() != community:
            counts["skipped"] += 1
            continue
        if int(record.get("reconcile_checks", 0) or 0) >= _MAX_RECONCILE_CHECKS:
            record["status"] = "needs_attention"
            record["updated_at"] = _now_iso()
            _persist(persist_state, state)
            counts["needs_attention"] += 1
            continue
        counts["checked"] += 1
        status = _status_after_check(state, record, notes_interface, persist_state)
        if status == "delivered":
            counts["delivered"] += 1
        elif status == "needs_attention":
            counts["needs_attention"] += 1
        else:
            counts["uncertain"] += 1

    counts["skipped"] += max(0, len(pending) - _MAX_RECONCILES_PER_CALL)
    return counts
