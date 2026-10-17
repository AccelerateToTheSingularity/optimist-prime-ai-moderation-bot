"""Durable idempotency and rolling limits for public bot comments."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Callable

import config
from bot_utils import is_bot_owned_author


ORDINARY_GLOBAL_PER_HOUR = 150
ORDINARY_PER_THREAD_PER_HOUR = 15
MODERATOR_GLOBAL_PER_HOUR = 600
MODERATOR_PER_THREAD_PER_HOUR = 60
LEDGER_RETENTION_SECONDS = (config.MAX_AGE_HOURS + 2) * 3600


def _now_timestamp(now: float | None = None) -> float:
    return float(now if now is not None else datetime.now(timezone.utc).timestamp())


def _ledger(state: dict) -> dict[str, dict[str, Any]]:
    if "public_response_ledger" not in state:
        value = {}
        state["public_response_ledger"] = value
    else:
        value = state["public_response_ledger"]
    if not isinstance(value, dict):
        # This ledger is the at-most-once record for public side effects. A
        # bad value must stop the run; replacing it with an empty dictionary
        # could publish a second reply to work we already completed.
        raise RuntimeError(
            "Public response ledger is unreadable; refusing to post without "
            "its idempotency record"
        )
    for key, record in value.items():
        created = record.get("created_at") if isinstance(record, dict) else None
        if (
            not isinstance(key, str)
            or not isinstance(record, dict)
            or record.get("status") not in {"pending", "uncertain", "completed", "abstained"}
            or isinstance(created, bool)
            or not isinstance(created, (int, float))
            or not math.isfinite(float(created))
            or not isinstance(record.get("thread_id"), str)
            or not record.get("thread_id")
            or record.get("tier") not in {"ordinary", "moderator"}
        ):
            raise RuntimeError(
                "Public response ledger contains an unreadable record; refusing "
                "to post without its idempotency record"
            )
    return value


def _source_key(source_id: str, source_type: str = "comment", action: str = "reply") -> str:
    prefix = "post" if source_type == "post" else "comment"
    if prefix == "comment":
        # A comment can receive no more than one bot response, regardless of
        # which feature happened to generate it.
        return f"comment:{source_id.removeprefix('t1_')}"
    return f"post:{source_id.removeprefix('t3_')}:{action}"


def _thread_id(source: Any, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit).removeprefix("t3_")
    submission = getattr(source, "submission", None)
    post_id = getattr(submission, "id", None)
    if post_id:
        return str(post_id).removeprefix("t3_")
    return str(getattr(source, "id", "unknown")).removeprefix("t3_")


def _verified_moderator_tier(state: dict, author_name: str | None) -> bool:
    """Grant the higher ceiling only from this cycle's successfully fetched roster."""
    if not author_name or state.get("_current_moderators_verified") is not True:
        return False
    names = state.get("_current_moderators", [])
    return author_name.casefold() in {
        str(name).casefold() for name in names if isinstance(name, str)
    }


def _prune(ledger: dict[str, dict[str, Any]], now: float) -> None:
    cutoff = now - LEDGER_RETENTION_SECONDS
    for key, record in list(ledger.items()):
        created = record.get("created_at")
        if record.get("status") not in {"pending", "uncertain"} and created < cutoff:
            ledger.pop(key, None)


def _hourly_counts(
    ledger: dict[str, dict[str, Any]],
    now: float,
    thread_id: str,
    tier: str,
) -> tuple[int, int]:
    cutoff = now - 3600
    global_count = 0
    thread_count = 0
    for record in ledger.values():
        if not isinstance(record, dict):
            continue
        if record.get("status") not in {"pending", "uncertain", "completed"}:
            continue
        if record.get("tier") != tier:
            continue
        timestamp = record.get("created_at")
        if not isinstance(timestamp, (int, float)) or timestamp < cutoff:
            continue
        global_count += 1
        if str(record.get("thread_id", "")) == thread_id:
            thread_count += 1
    return global_count, thread_count


def _caps(tier: str) -> tuple[int, int]:
    if tier == "moderator":
        return MODERATOR_GLOBAL_PER_HOUR, MODERATOR_PER_THREAD_PER_HOUR
    return ORDINARY_GLOBAL_PER_HOUR, ORDINARY_PER_THREAD_PER_HOUR


def response_capacity_available(
    state: dict,
    thread_id: str,
    *,
    author_name: str | None = None,
    now: float | None = None,
) -> tuple[bool, str | None]:
    """Check rolling counts in the author's verified response tier."""
    timestamp = _now_timestamp(now)
    ledger = _ledger(state)
    _prune(ledger, timestamp)
    tier = "moderator" if _verified_moderator_tier(state, author_name) else "ordinary"
    global_limit, thread_limit = _caps(tier)
    # Ordinary users and verified moderators have independent budgets. The
    # moderator allowance is four times larger and does not borrow or consume
    # the ordinary-user pool.
    global_count, thread_count = _hourly_counts(
        ledger, timestamp, str(thread_id), tier
    )
    if global_count >= global_limit:
        return False, f"{tier}_global"
    if thread_count >= thread_limit:
        return False, f"{tier}_thread"
    return True, None


def _legacy_processed(state: dict, source_id: str, source_type: str) -> bool:
    item_id = str(source_id).removeprefix("t1_").removeprefix("t3_")
    if source_type == "post":
        return f"post_{item_id}" in set(state.get("summon_responses", []))
    return (
        item_id in set(state.get("replied_to_comments", []))
        or item_id in set(state.get("summon_responses", []))
    )


def _clear_pending_history_item(state: dict, source_id: str, source_type: str) -> bool:
    """Remove a catch-up retry once its public response is terminal."""
    kind = "posts" if source_type == "post" else "comments"
    normalized_id = str(source_id).removeprefix("t1_").removeprefix("t3_")
    catch_up = state.get("catch_up")
    bucket = catch_up.get(kind) if isinstance(catch_up, dict) else None
    pending = bucket.get("pending") if isinstance(bucket, dict) else None
    if not isinstance(pending, list) or not any(
        isinstance(item, dict) and str(item.get("id")) == normalized_id
        for item in pending
    ):
        return False
    from catch_up import clear_completed_retry_items

    clear_completed_retry_items(state, kind, {normalized_id})
    return True


def _owned_child_status(
    source: Any,
    bot_username: str,
    expected_hash: str | None = None,
) -> bool | None:
    """Return whether Optimist has a direct child; None means the listing is partial."""
    children = getattr(source, "replies", None)
    if children is None:
        return None
    incomplete = False
    try:
        for child in children:
            # PRAW represents unexpanded reply pages with MoreComments nodes.
            # Their presence means an absent Optimist child is not conclusive.
            if type(child).__name__ == "MoreComments":
                incomplete = True
                continue
            if not is_bot_owned_author(
                getattr(getattr(child, "author", None), "name", None),
                bot_username,
            ):
                continue
            if expected_hash:
                body = str(getattr(child, "body", "") or "")
                if hashlib.sha256(body.encode("utf-8")).hexdigest() != expected_hash:
                    continue
            return True
        return None if incomplete else False
    except Exception:
        return None


def _has_owned_child(
    source: Any,
    bot_username: str,
    expected_hash: str | None = None,
) -> bool:
    return _owned_child_status(source, bot_username, expected_hash) is True


def _reconcile_existing_reply(
    source: Any,
    bot_username: str,
    *,
    expected_hash: str | None = None,
    source_type: str = "comment",
) -> bool | None:
    """Return found / not observed / unknown for an ambiguous prior write."""
    initial = _owned_child_status(source, bot_username, expected_hash)
    if initial is True:
        return True

    # Comment.refresh() fetches its current replies.  This path is used only
    # for a durable pending marker after a prior uncertain Reddit write.
    refresh = getattr(source, "refresh", None)
    if callable(refresh):
        try:
            refresh()
            refreshed = _owned_child_status(source, bot_username, expected_hash)
            if refreshed is True:
                return True
            if source_type == "comment" and hasattr(source, "parent_id"):
                return refreshed
        except Exception:
            return None

    if source_type == "comment" and hasattr(source, "parent_id"):
        # Lightweight sources used by isolated callers may expose a complete
        # replies list without PRAW's refresh method. PRAW comments
        # take the refresh path above.
        return initial

    # Post-origin outputs are rare pending recoveries.  Load the post's
    # comments once and check its direct children without querying per reply.
    comments = getattr(source, "comments", None)
    replace_more = getattr(comments, "replace_more", None)
    list_comments = getattr(comments, "list", None)
    if callable(replace_more) and callable(list_comments):
        try:
            replace_more(limit=0)
            source_fullname = f"t3_{str(getattr(source, 'id', '')).removeprefix('t3_')}"
            for child in list_comments():
                if getattr(child, "parent_id", None) != source_fullname:
                    continue
                if not is_bot_owned_author(
                    getattr(getattr(child, "author", None), "name", None),
                    bot_username,
                ):
                    continue
                if expected_hash:
                    body = str(getattr(child, "body", "") or "")
                    if hashlib.sha256(body.encode("utf-8")).hexdigest() != expected_hash:
                        continue
                return True
            return False
        except Exception:
            return None

    return None


def public_response_already_handled(
    source: Any,
    state: dict,
    *,
    source_id: str,
    bot_username: str,
    persist_state: Callable[[dict], Any] | None,
    source_type: str = "comment",
    action: str = "reply",
) -> bool:
    """Skip completed work and reconcile any write interrupted mid-flight."""
    key = _source_key(source_id, source_type, action)
    ledger = _ledger(state)
    record = ledger.get(key)
    if isinstance(record, dict):
        status = record.get("status")
        if status in {"completed", "abstained", "uncertain"}:
            if _clear_pending_history_item(state, source_id, source_type) and callable(persist_state):
                persist_state(state)
            return True
        if status == "pending":
            found = _reconcile_existing_reply(
                source,
                bot_username,
                expected_hash=(
                    record.get("reply_body_hash") if source_type == "post" else None
                ),
                source_type=source_type,
            )
            if found is True:
                record["status"] = "completed"
                record["reconciled_at"] = _now_timestamp()
                record["reply_id"] = "reconciled"
                _clear_pending_history_item(state, source_id, source_type)
                if callable(persist_state):
                    persist_state(state)
                return True
            if found is False:
                # Reddit's comment listing is not authoritative enough to
                # prove that an earlier POST was never accepted (deleted,
                # delayed, or omitted replies can all look like absence).
                # Once a write may have reached Reddit, never retry it
                # automatically. The lost response is preferable to a
                # duplicate public comment.
                record["status"] = "uncertain"
                record["uncertain_at"] = _now_timestamp()
                _clear_pending_history_item(state, source_id, source_type)
                if callable(persist_state):
                    persist_state(state)
                return True
            # Unknown means Reddit did not give a trustworthy read. Do not
            # risk a second public comment while the first may exist. Retain
            # an explicit terminal marker to avoid repeated reconciliation
            # reads or a future automatic retry.
            record["status"] = "uncertain"
            record["uncertain_at"] = _now_timestamp()
            _clear_pending_history_item(state, source_id, source_type)
            if callable(persist_state):
                persist_state(state)
            return True
    if _legacy_processed(state, source_id, source_type):
        if _clear_pending_history_item(state, source_id, source_type) and callable(persist_state):
            persist_state(state)
        return True
    return False


def record_public_abstention(
    state: dict,
    *,
    source_id: str,
    thread_id: str,
    author_name: str | None,
    source_type: str = "comment",
    action: str = "reply",
    now: float | None = None,
    persist_state: Callable[[dict], Any] | None = None,
) -> None:
    """Remember the model's one-turn decision not to publish."""
    timestamp = _now_timestamp(now)
    ledger = _ledger(state)
    _prune(ledger, timestamp)
    ledger[_source_key(source_id, source_type, action)] = {
        "status": "abstained",
        "created_at": timestamp,
        "thread_id": str(thread_id),
        "tier": "moderator" if _verified_moderator_tier(state, author_name) else "ordinary",
        "source_id": str(source_id),
    }
    _clear_pending_history_item(state, source_id, source_type)
    if callable(persist_state):
        persist_state(state)


def _notify_cap_once(state: dict, subreddit: Any, persist_state, reason: str, tier: str) -> None:
    """Send at most one private modmail notice per cap and UTC hour."""
    now = _now_timestamp()
    hour_key = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H")
    key = f"{tier}:{reason}:{hour_key}"
    notices = state.setdefault("public_response_cap_notices", {})
    if not isinstance(notices, dict):
        notices = {}
        state["public_response_cap_notices"] = notices
    stale_before = datetime.fromtimestamp(now - 2 * 86400, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H"
    )
    notices = {
        notice_key: value
        for notice_key, value in notices.items()
        if notice_key.rsplit(":", 1)[-1] >= stale_before
    }
    state["public_response_cap_notices"] = notices
    if notices.get(key):
        return
    # Persist before contacting Reddit so a timeout cannot produce repeated
    # moderator mail. A missed notice is preferable to a duplicate burst.
    notices[key] = "pending"
    if callable(persist_state):
        persist_state(state)
    try:
        subreddit.message(
            subject="Optimist Prime public response limit reached",
            message=(
                f"The {tier} public response limit ({'hourly' if reason.endswith('global') else 'per-thread'}) "
                "was reached. Further public comments are being held until the rolling hour clears."
            ),
        )
        notices[key] = "sent"
        if callable(persist_state):
            persist_state(state)
    except Exception as exc:
        print(f"    ⚠️ Could not send private response-limit notice ({type(exc).__name__})")


def notify_public_response_limit(
    state: dict,
    subreddit: Any,
    persist_state,
    *,
    author_name: str | None,
    reason: str,
) -> None:
    tier = "moderator" if _verified_moderator_tier(state, author_name) else "ordinary"
    if subreddit is not None:
        _notify_cap_once(state, subreddit, persist_state, reason, tier)


def post_public_comment(
    source: Any,
    text: str,
    state: dict,
    *,
    source_id: str,
    thread_id: str | None = None,
    author_name: str | None = None,
    bot_username: str,
    persist_state: Callable[[dict], Any] | None,
    source_type: str = "comment",
    action: str = "reply",
    subreddit: Any = None,
    dry_run: bool = False,
    validate: Callable[[Any, str], Any] | None = None,
    item_type: str = "public response",
) -> Any | None:
    """Persist a reservation, post once, then persist the verified outcome."""
    if dry_run:
        print(f"       [DRY RUN] Would post {item_type} for {source_id}")
        return None
    if not callable(persist_state):
        raise RuntimeError("Public comments require a durable state-save callback")

    if public_response_already_handled(
        source,
        state,
        source_id=source_id,
        bot_username=bot_username,
        persist_state=persist_state,
        source_type=source_type,
        action=action,
    ):
        return None

    timestamp = _now_timestamp()
    ledger = _ledger(state)
    _prune(ledger, timestamp)
    resolved_thread = _thread_id(source, thread_id)
    tier = "moderator" if _verified_moderator_tier(state, author_name) else "ordinary"
    global_limit, thread_limit = _caps(tier)
    global_count, thread_count = _hourly_counts(
        ledger, timestamp, resolved_thread, tier
    )
    reason = "global" if global_count >= global_limit else (
        "thread" if thread_count >= thread_limit else None
    )
    if reason:
        print(
            f"    ⏸️ Public response limit reached for {tier} ({reason}); "
            "no comment was posted"
        )
        if subreddit is not None:
            _notify_cap_once(state, subreddit, persist_state, reason, tier)
        return None

    # Recover the at-most-once invariant if older local state lost its ledger
    # entry after Reddit accepted a comment. This check runs only for a reply
    # that is otherwise ready to publish, never for every scanned comment.
    # Comment.refresh() provides Reddit's current direct-reply context.
    if source_type != "post":
        existing = _reconcile_existing_reply(
            source,
            bot_username,
            source_type="comment",
        )
        if existing is True:
            ledger[_source_key(source_id, source_type, action)] = {
                "status": "completed",
                "created_at": timestamp,
                "thread_id": resolved_thread,
                "tier": tier,
                "source_id": str(source_id),
                "action": action,
                "reply_id": "reconciled",
            }
            _clear_pending_history_item(state, source_id, source_type)
            persist_state(state)
            print(f"       ⏭️ Existing Optimist reply found for {source_id}; skipped duplicate")
            return None
        if existing is None:
            # Keep the source retryable, but do not publish while Reddit's
            # current reply context is unavailable or incomplete.
            print(f"       ⏸️ Could not verify existing replies for {source_id}; skipped to avoid a duplicate")
            return None

    key = _source_key(source_id, source_type, action)
    ledger[key] = {
        "status": "pending",
        "created_at": timestamp,
        "thread_id": resolved_thread,
        "tier": tier,
        "source_id": str(source_id),
        "action": action,
        "reply_body_hash": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
    }
    # If this fails, no Reddit write is attempted.
    persist_state(state)

    try:
        response = source.reply(text)
        if validate is not None:
            validate(response, item_type)
        elif response is None or not getattr(response, "id", None):
            raise ValueError("Reddit did not return a public comment ID")
    except Exception as exc:
        # Keep the durable pending marker. On the next cycle we reconcile it;
        # if Reddit cannot prove the outcome, the source is held permanently
        # instead of risking a second public comment.
        print(f"       ⚠️ {item_type} outcome is uncertain; it will be reconciled before any further action ({type(exc).__name__})")
        return None

    ledger[key].update(
        status="completed",
        reply_id=str(getattr(response, "id", "")),
        confirmed_at=_now_timestamp(),
    )
    _clear_pending_history_item(state, source_id, source_type)
    # A failure here must stop the worker. The already-persisted pending entry
    # will be reconciled against Reddit after restart.
    persist_state(state)
    return response


__all__ = [
    "ORDINARY_GLOBAL_PER_HOUR",
    "ORDINARY_PER_THREAD_PER_HOUR",
    "MODERATOR_GLOBAL_PER_HOUR",
    "MODERATOR_PER_THREAD_PER_HOUR",
    "post_public_comment",
    "notify_public_response_limit",
    "public_response_already_handled",
    "record_public_abstention",
    "response_capacity_available",
]
