"""Natural-language moderator assistant for recent Reddit Modmail threads."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import config
from moderator_action_router import process_moderator_turn
from modmail_support import (
    message_author,
    message_body,
    reply_private_note,
)
SUBJECT = "Optimist Prime moderator assistant"
OPENING_BODY = (
    "I’m ready. Ask naturally for a user's history review, a focused question, or a flair change. "
    "For example: ‘Optimist Prime, review u/example_user’s history’ or ‘set this person’s flair to an approved value.’"
)
RECENT_DAYS = 7
LISTING_LIMIT = 100
MAX_CONVERSATIONS_PER_CYCLE = 12
MAX_SEEN_MESSAGE_IDS = 5000


def _assistant_state(state: dict) -> dict:
    entry = state.setdefault("moderator_modmail_assistant", {})
    if not isinstance(entry, dict):
        entry = {}
        state["moderator_modmail_assistant"] = entry
    entry.setdefault("conversation_id", None)
    entry.setdefault("processed_message_ids", [])
    entry.setdefault("pending_reply", None)
    entry.setdefault("conversation_cursors", {})
    return entry


def _latest_activity(conversation: Any) -> datetime | None:
    raw = getattr(conversation, "last_updated", None)
    if isinstance(raw, datetime):
        value = raw
    elif isinstance(raw, (int, float)):
        value = datetime.fromtimestamp(raw, tz=timezone.utc)
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _conversation_signature(conversation: Any) -> tuple[int, str] | None:
    updated = _latest_activity(conversation)
    if updated is None:
        return None
    try:
        count = max(0, int(getattr(conversation, "num_messages", 0) or 0))
    except (TypeError, ValueError):
        count = 0
    return count, updated.isoformat()


def _collect_recent_conversations(subreddit, entry: dict, now: datetime, metrics: dict):
    found: dict[str, dict[str, Any]] = {}
    for state_name in ("all", "appeals", "mod"):
        try:
            listing = subreddit.modmail.conversations(
                state=state_name, sort="recent", limit=LISTING_LIMIT
            )
            for conversation in listing:
                conversation_id = str(getattr(conversation, "id", "") or "").strip()
                if not conversation_id:
                    continue
                row = found.setdefault(
                    conversation_id,
                    {"conversation": conversation, "appeal": False},
                )
                if state_name == "appeals":
                    row["appeal"] = True
                if (
                    state_name == "mod"
                    and bool(getattr(conversation, "is_internal", False))
                    and str(getattr(conversation, "subject", "")).strip() == SUBJECT
                ):
                    entry["conversation_id"] = conversation_id
        except Exception as exc:
            metrics["failures"] += 1
            print(f"  ⚠️ Could not list {state_name} Modmail conversations ({type(exc).__name__})")

    if not entry.get("conversation_id"):
        if not config.MODMAIL_ASSISTANT_ENABLED:
            return []
        # If no saved team thread exists, create one only during a real run.
        if metrics.get("dry_run"):
            return []
        conversation = subreddit.modmail.create(
            subject=SUBJECT, body=OPENING_BODY, recipient=None
        )
        if not bool(getattr(conversation, "is_internal", False)):
            raise RuntimeError("Reddit did not create the moderator assistant as a private thread")
        entry["conversation_id"] = str(conversation.id)
        found[str(conversation.id)] = {"conversation": conversation, "appeal": False}

    cutoff = now - timedelta(days=RECENT_DAYS)
    recent = []
    for conversation_id, row in found.items():
        activity = _latest_activity(row["conversation"])
        if activity is None or activity < cutoff:
            metrics["stale_conversations_skipped"] += 1
            continue
        recent.append((activity, conversation_id, row["conversation"], bool(row["appeal"])))
    recent.sort(key=lambda item: (item[0], item[1]))
    return recent


def _is_internal_message(message: Any) -> bool:
    return bool(getattr(message, "is_internal", False))


def _find_existing_pending_reply(messages: list[Any], bot_username: str, pending: dict) -> bool:
    source_id = str(pending.get("source_message_id") or "")
    expected_body = str(pending.get("body") or "")
    source_index = next(
        (index for index, message in enumerate(messages) if str(getattr(message, "id", "")) == source_id),
        -1,
    )
    if source_index < 0:
        return False
    return any(
        _is_internal_message(message)
        and message_author(message).casefold() == bot_username.casefold()
        and message_body(message).strip() == expected_body.strip()
        for message in messages[source_index + 1 :]
    )


def _remember_processed(entry: dict, message_id: str) -> None:
    ids = [str(value) for value in entry.get("processed_message_ids", []) if value]
    if message_id and message_id not in ids:
        ids.append(message_id)
    entry["processed_message_ids"] = ids[-MAX_SEEN_MESSAGE_IDS:]


def _context_before(messages: list[Any], index: int) -> str:
    rows = []
    for message in messages[max(0, index - 6):index]:
        author = message_author(message) or "unknown"
        body = re.sub(r"\s+", " ", message_body(message)).strip()
        if body:
            rows.append(f"u/{author}: {body[:450]}")
    return "\n".join(rows)[-2200:]


def _store_cursor(entry: dict, conversation_id: str, message_id: str, *, signature=None) -> None:
    cursors = entry.setdefault("conversation_cursors", {})
    row = cursors.setdefault(conversation_id, {})
    if message_id:
        row["last_message_id"] = message_id
    if signature is not None:
        row["num_messages"], row["last_updated"] = signature


def _conversation_messages(conversation):
    try:
        return list(getattr(conversation, "messages", []) or [])
    except Exception:
        raise


def _verified_moderators(state: dict) -> set[str] | None:
    """Return only the moderator roster verified during this bot cycle."""
    if state.get("_current_moderators_verified") is not True:
        return None
    names = state.get("_current_moderators")
    if not isinstance(names, list):
        return None
    return {str(name).strip().casefold() for name in names if str(name).strip()}


def _retry_pending_reply(
    subreddit,
    state: dict,
    entry: dict,
    bot_username: str,
    current_moderators: set[str],
    *,
    dry_run: bool,
    persist_state,
) -> tuple[bool, bool]:
    pending = entry.get("pending_reply")
    if not isinstance(pending, dict):
        return False, False
    conversation_id = str(
        pending.get("conversation_id") or entry.get("conversation_id") or ""
    ).strip()
    if not conversation_id:
        return False, False
    conversation = subreddit.modmail(conversation_id)
    messages = _conversation_messages(conversation)
    source_id = str(pending.get("source_message_id") or "")
    if not source_id:
        raise RuntimeError("Unable to verify the author of a saved Modmail reply")
    if _find_existing_pending_reply(messages, bot_username, pending):
        _remember_processed(entry, source_id)
        _store_cursor(entry, conversation_id, source_id)
        entry["pending_reply"] = None
        if persist_state and not dry_run:
            persist_state(state)
        return True, True
    source_message = next(
        (message for message in messages
         if str(getattr(message, "id", "") or "") == source_id),
        None,
    )
    if source_message is None:
        raise RuntimeError("Unable to verify the author of a saved Modmail reply")
    source_author = message_author(source_message).casefold()
    if not source_author or source_author not in current_moderators:
        _remember_processed(entry, source_id)
        _store_cursor(entry, conversation_id, source_id)
        entry["pending_reply"] = None
        if persist_state and not dry_run:
            persist_state(state)
        return True, False
    if dry_run:
        return True, False
    reply_private_note(conversation, str(pending.get("body") or ""))
    _remember_processed(entry, source_id)
    _store_cursor(entry, conversation_id, source_id)
    entry["pending_reply"] = None
    if persist_state:
        persist_state(state)
    return True, True


def _prepare_response(
    conversation,
    message,
    *,
    subreddit,
    reddit,
    state: dict,
    llm_model,
    jev_client,
    bot_username: str,
    appeal: bool,
    dedicated: bool,
    trusted_context: str,
    persist_state,
):
    author_name = message_author(message)
    body = message_body(message)
    comment_context = SimpleNamespace(
        id=str(getattr(message, "id", "") or ""),
        author=SimpleNamespace(name=author_name),
        body=body,
    )
    tokens = 0
    cost = 0.0
    participant = getattr(conversation, "participant", None)
    parent_author = (
        str(getattr(participant, "name", "") or "").strip()
        if appeal else None
    )
    turn = process_moderator_turn(
        reddit,
        subreddit,
        state,
        llm_model,
        author_name=author_name,
        text=body,
        channel="modmail",
        source_id=str(getattr(message, "id", "") or ""),
        conversation_key=f"modmail:{getattr(conversation, 'id', '')}",
        bot_username=bot_username,
        trusted_context=trusted_context,
        verified_moderator=True,
        interaction=comment_context,
        parent_author=parent_author,
        is_summon=dedicated,
        jev_client=jev_client,
        persist_state=persist_state,
    )
    if turn is None:
        raise RuntimeError("moderator Modmail turn was not interpreted")
    usage = turn.get("token_info", {})
    tokens = int(usage.get("total_tokens", 0) or 0)
    cost = float(usage.get("cost", 0.0) or 0.0)
    state = turn.get("state", state)
    response = (
        turn.get("response") if turn.get("handled")
        else turn.get("conversation_response")
    )
    if str(response or "").strip() == "[[NO_REPLY]]":
        return None, state, tokens, cost, False
    return str(response or ""), state, tokens, cost, True


def process_modmail_assistant(
    subreddit,
    reddit,
    state: dict,
    llm_model,
    *,
    bot_username: str,
    dry_run: bool = False,
    jev_client=None,
    persist_state: Callable[[dict], None] | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Check active, appeal, and moderator-only Modmail updated in the last week."""
    metrics = {
        "available": 0,
        "conversations_checked": 0,
        "stale_conversations_skipped": 0,
        "commands_seen": 0,
        "replies_sent": 0,
        "failures": 0,
        "tokens": 0,
        "cost": 0.0,
        "dry_run": dry_run,
    }
    if not config.MODMAIL_ASSISTANT_ENABLED:
        return metrics

    current_moderators = _verified_moderators(state)
    if current_moderators is None:
        metrics["failures"] += 1
        return metrics

    entry = _assistant_state(state)
    try:
        retry_handled, retry_sent = _retry_pending_reply(
            subreddit, state, entry, bot_username, current_moderators,
            dry_run=dry_run, persist_state=persist_state,
        )
        if retry_handled:
            metrics["replies_sent"] = int(retry_sent)
            return metrics
    except Exception as exc:
        metrics["failures"] += 1
        print(f"  ⚠️ A saved Modmail reply remains retryable ({type(exc).__name__})")
        return metrics

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    recent = _collect_recent_conversations(
        subreddit, entry, current_time.astimezone(timezone.utc), metrics
    )
    already_processed = {
        str(value) for value in entry.get("processed_message_ids", []) if value
    }
    cursors = entry.setdefault("conversation_cursors", {})
    inspected = 0

    for activity, conversation_id, summary, appeal in recent:
        if inspected >= MAX_CONVERSATIONS_PER_CYCLE:
            break
        signature = _conversation_signature(summary)
        cursor = cursors.get(conversation_id, {})
        if (
            signature is not None
            and cursor.get("num_messages") == signature[0]
            and cursor.get("last_updated") == signature[1]
            and cursor.get("last_message_id")
        ):
            continue
        inspected += 1
        metrics["conversations_checked"] += 1
        try:
            messages = _conversation_messages(summary)
        except Exception:
            metrics["failures"] += 1
            continue
        metrics["available"] += len(messages)
        cursor_id = str(cursor.get("last_message_id") or "")
        cursor_index = next(
            (i for i, message in enumerate(messages)
             if str(getattr(message, "id", "") or "") == cursor_id),
            -1,
        ) if cursor_id else -1
        start_index = cursor_index + 1 if cursor_index >= 0 else 0
        dedicated = (
            conversation_id == str(entry.get("conversation_id") or "")
            and bool(getattr(summary, "is_internal", False))
        )
        last_examined_id = cursor_id

        for index in range(start_index, len(messages)):
            message = messages[index]
            message_id = str(getattr(message, "id", "") or "").strip()
            if not message_id:
                continue
            author_name = message_author(message)
            body = message_body(message)
            if message_id in already_processed:
                last_examined_id = message_id
                continue
            if not author_name or author_name.casefold() == bot_username.casefold():
                _remember_processed(entry, message_id)
                already_processed.add(message_id)
                last_examined_id = message_id
                continue
            if author_name.casefold() not in current_moderators:
                _remember_processed(entry, message_id)
                already_processed.add(message_id)
                last_examined_id = message_id
                continue

            metrics["commands_seen"] += 1
            if dry_run:
                return metrics
            response, state, tokens, cost, handled = _prepare_response(
                summary,
                message,
                subreddit=subreddit,
                reddit=reddit,
                state=state,
                llm_model=llm_model,
                jev_client=jev_client,
                bot_username=bot_username,
                appeal=appeal,
                dedicated=dedicated,
                trusted_context=_context_before(messages, index),
                persist_state=persist_state,
            )
            metrics["tokens"] += tokens
            metrics["cost"] += cost
            entry = _assistant_state(state)
            last_examined_id = message_id
            if not handled or not str(response or "").strip():
                _remember_processed(entry, message_id)
                already_processed.add(message_id)
                _store_cursor(entry, conversation_id, last_examined_id)
                continue

            entry["pending_reply"] = {
                "conversation_id": conversation_id,
                "source_message_id": message_id,
                "body": str(response),
            }
            _store_cursor(entry, conversation_id, message_id)
            if persist_state:
                persist_state(state)
            try:
                # Internal notes remain hidden from appeal participants and any
                # other non-moderator recipients of this Modmail conversation.
                reply_private_note(summary, str(response))
            except Exception:
                metrics["failures"] += 1
                raise
            _remember_processed(entry, message_id)
            already_processed.add(message_id)
            entry["pending_reply"] = None
            metrics["replies_sent"] += 1
            if persist_state:
                persist_state(state)
            return metrics

        _store_cursor(entry, conversation_id, last_examined_id, signature=signature)

    if persist_state and not dry_run:
        persist_state(state)
    return metrics
