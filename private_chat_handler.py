"""One-to-one Reddit private conversations using the published message API."""

from __future__ import annotations

import hashlib
import logging
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import config
from bot_logging import log_event
from bot_utils import (
    check_user_cooldown,
    is_bot_owned_author,
    refresh_moderators,
    record_reply_success,
)
from llm_client import model_without_per_run_cap
from jev_interaction_router import route_public_interaction
from moderator_action_router import process_moderator_turn
from persona import NO_PUBLIC_REPLY, generate_conversation_response_from_text
from reddit_conversation_transport import (
    mark_message_read,
    send_message_reply,
    unread_private_messages,
)
from self_flair_requests import (
    handle_self_custom_flair_request,
)


_PRIVATE_HISTORY_LIMIT = 24
_PROCESSED_MESSAGE_LIMIT = 3000
_PRIVATE_FAILURE_BACKOFF_SECONDS = 60
_PRIVATE_FAILURE_BACKOFF_MAX_SECONDS = 900
_JEV_HISTORY_FAILURE_REPLY = (
    "Jev couldn't complete the history review, so I don't have a report. "
    "I won't retry the same analysis automatically. Send a new review request "
    "if you'd like me to try again."
)


@dataclass(frozen=True)
class PrivateChatCycleMetrics:
    """Count-only result used by the adaptive scheduler."""

    messages_handled: int = 0
    scan_complete: bool = True
    messages_failed: int = 0
    messages_deferred: int = 0


def _author_name(message) -> str | None:
    author = getattr(message, "author", None)
    name = getattr(author, "name", None) if author else None
    return str(name).strip() if name else None


def _destination_name(message) -> str:
    destination = getattr(message, "dest", None) or getattr(message, "destination", None)
    return str(getattr(destination, "name", destination) or "").strip()


def _parent_id(message) -> str | None:
    value = str(getattr(message, "parent_id", "") or "").strip()
    if value.startswith("t4_"):
        value = value[3:]
    return value or None


def _conversation_key(message, username: str, state: dict) -> str:
    explicit = str(getattr(message, "conversation_id", "") or "").strip()
    mapping = state.setdefault("private_message_conversation_ids", {})
    if explicit:
        key = f"private:{username.casefold()}:{explicit}"
    else:
        # PRAW's supported one-to-one inbox messages often have neither a
        # conversation ID nor a parent ID. Use the user/bot pair as the stable
        # private conversation identity instead of accidentally starting a
        # fresh transcript for every incoming message.
        key = f"private:{username.casefold()}"
        _merge_legacy_private_history(state, username, key)
    mapping[str(getattr(message, "id", ""))] = key
    if len(mapping) > _PROCESSED_MESSAGE_LIMIT * 2:
        for old in list(mapping)[:_PROCESSED_MESSAGE_LIMIT]:
            mapping.pop(old, None)
    return key


def reply_to_latest_private_chat(
    reddit,
    state: dict,
    *,
    username: str,
    bot_username: str,
    text: str,
    request_text: str = "",
) -> bool:
    """Reply in the newest one-to-one Chat already recorded for this user.

    This deliberately does not create a new conversation. If no recorded
    message can be fetched, the fetched message is not between these two
    accounts, or Reddit rejects the reply, it returns False without retrying.
    """
    target = str(username or "").strip().casefold()
    bot = str(bot_username or "").strip().casefold()
    if not target or not bot or not str(text or "").strip():
        return False

    prefix = f"private:{target}"
    mapping = state.get("private_message_conversation_ids")
    conversations = state.get("private_conversations")
    if not isinstance(mapping, dict) or not isinstance(conversations, dict):
        return False

    latest_by_conversation: dict[str, tuple[float, int, str]] = {}
    for index, (message_id, conversation_key) in enumerate(mapping.items()):
        message_id = str(message_id or "").strip()
        key = str(conversation_key or "").strip()
        if not message_id or not (key == prefix or key.startswith(prefix + ":")):
            continue
        row = conversations.get(key)
        try:
            updated_at = float(row.get("updated_at", 0) or 0) if isinstance(row, dict) else 0.0
        except (TypeError, ValueError):
            updated_at = 0.0
        candidate = (updated_at, index, message_id)
        if key not in latest_by_conversation or candidate[:2] > latest_by_conversation[key][:2]:
            latest_by_conversation[key] = candidate

    if not latest_by_conversation:
        return False
    conversation_key, candidate = max(
        latest_by_conversation.items(),
        key=lambda item: item[1][:2],
    )
    message_id = candidate[2]

    try:
        anchor = reddit.inbox.message(message_id)
        author = _author_name(anchor)
        destination = _destination_name(anchor).casefold()
        if not author or {author.casefold(), destination} != {target, bot}:
            return False
        response = send_message_reply(anchor, text)
    except Exception as exc:
        logging.getLogger("optimist_prime").info(
            "Existing private Chat reply was not delivered",
            extra={
                "event_name": "private_chat.existing_reply.not_delivered",
                "component": "private_chat",
                "error_type": type(exc).__name__,
            },
        )
        return False

    response_id = str(getattr(response, "id", "") or "").strip()
    if response_id:
        mapping[response_id] = conversation_key
    row = conversations.setdefault(conversation_key, {"history": []})
    history = row.setdefault("history", [])
    if request_text:
        history.append({
            "role": "user",
            "text": f"[Requested in subreddit comment] {str(request_text)[:1450]}",
        })
    history.append({"role": "assistant", "text": str(text)[:1500]})
    row["history"] = history[-_PRIVATE_HISTORY_LIMIT:]
    row["updated_at"] = datetime.now(timezone.utc).timestamp()
    record_reply_success(state, moderator=True)
    return True


def _merge_legacy_private_history(state: dict, username: str, canonical_key: str) -> None:
    """Fold old message-keyed DM transcripts into the stable per-user history."""
    conversations = state.setdefault("private_conversations", {})
    contexts = state.setdefault("conversation_contexts", {})
    mapping = state.setdefault("private_message_conversation_ids", {})
    prefix = f"private:{username.casefold()}:"
    legacy_keys = {
        key for key in conversations
        if isinstance(key, str)
        and key.startswith(prefix)
        and mapping.get(key[len(prefix):]) == key
    }
    legacy_keys.update(
        key for key in contexts
        if isinstance(key, str)
        and key.startswith(prefix)
        and mapping.get(key[len(prefix):]) == key
    )
    legacy_keys.discard(canonical_key)
    if not legacy_keys:
        return

    history_rows: list[tuple[float, list]] = []
    newest_updated_at = 0.0
    for key in {canonical_key, *legacy_keys}:
        row = conversations.get(key)
        if not isinstance(row, dict):
            continue
        try:
            updated_at = float(row.get("updated_at", 0) or 0)
        except (TypeError, ValueError):
            updated_at = 0.0
        newest_updated_at = max(newest_updated_at, updated_at)
        history = row.get("history")
        if isinstance(history, list):
            history_rows.append((updated_at, history))

    merged_history: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for _updated_at, history in sorted(history_rows, key=lambda item: item[0]):
        for row in history:
            if not isinstance(row, dict):
                continue
            copied = dict(row)
            identity = (
                str(copied.get("role", "user")),
                str(copied.get("text", "")),
                str(copied.get("source_message_id", "")),
            )
            # Successful reply rows do not currently carry a source ID; never
            # collapse two identical sentences from separate turns.
            if identity[2] and identity in seen:
                continue
            if identity[2]:
                seen.add(identity)
            merged_history.append(copied)

    if merged_history:
        conversations[canonical_key] = {
            "history": merged_history[-_PRIVATE_HISTORY_LIMIT:],
            "updated_at": newest_updated_at,
        }
    for key in legacy_keys:
        conversations.pop(key, None)

    # Retain the newest code-owned follow-up metadata while removing the
    # obsolete message-specific keys.
    context_rows: list[tuple[float, dict]] = []
    for key in {canonical_key, *legacy_keys}:
        row = contexts.get(key)
        if isinstance(row, dict):
            try:
                updated_at = float(row.get("updated_at", 0) or 0)
            except (TypeError, ValueError):
                updated_at = 0.0
            context_rows.append((updated_at, row))
    if context_rows:
        _updated_at, latest = max(context_rows, key=lambda item: item[0])
        contexts[canonical_key] = dict(latest)
    for key in legacy_keys:
        contexts.pop(key, None)

    for message_id, old_key in list(mapping.items()):
        if old_key in legacy_keys:
            mapping[message_id] = canonical_key


def _private_trusted_context(history: list) -> str:
    """Build a bounded transcript that keeps the newest private turns intact."""
    rows = [
        f"{row.get('role', 'user')}: {str(row.get('text', '')).replace(chr(10), ' ')[:1200]}"
        for row in history[-_PRIVATE_HISTORY_LIMIT:]
        if isinstance(row, dict)
    ]
    return "\n".join(rows)[-4500:]


def _message_shim(message, author_name: str):
    return SimpleNamespace(
        id=str(getattr(message, "id", "")),
        fullname=str(getattr(message, "fullname", "") or ""),
        author=SimpleNamespace(name=author_name),
        body=str(getattr(message, "body", "") or ""),
        parent_id=getattr(message, "parent_id", None),
        created_utc=getattr(message, "created_utc", 0),
        permalink=getattr(message, "permalink", ""),
        link_id=None,
        submission_id=None,
    )


def _record_user_reply(state: dict, username: str) -> None:
    now = datetime.now(timezone.utc).timestamp()
    recent = state.setdefault("recent_user_replies", {})
    row = recent.get(username)
    if not isinstance(row, dict) or (now - float(row.get("first_reply_time", 0) or 0)) >= config.SAME_USER_COOLDOWN_HOURS * 3600:
        recent[username] = {"count": 1, "first_reply_time": now}
    else:
        row["count"] = int(row.get("count", 0) or 0) + 1


def _record_private_message_failure(
    state: dict,
    message_id: str,
    exc: Exception,
    *,
    stage: str,
) -> int:
    """Persist a bounded retry delay without retaining private message text."""
    failures = state.setdefault("private_message_failures", {})
    previous = failures.get(message_id)
    try:
        attempts = max(0, int(previous.get("attempts", 0))) if isinstance(previous, dict) else 0
    except (TypeError, ValueError):
        attempts = 0
    attempts += 1
    delay = min(
        _PRIVATE_FAILURE_BACKOFF_MAX_SECONDS,
        _PRIVATE_FAILURE_BACKOFF_SECONDS * (2 ** min(attempts - 1, 4)),
    )
    failures[message_id] = {
        "attempts": attempts,
        "retry_after": datetime.now(timezone.utc).timestamp() + delay,
        "stage": str(stage)[:80],
        "error_type": type(exc).__name__[:120],
    }
    if len(failures) > _PROCESSED_MESSAGE_LIMIT:
        for old_id in list(failures)[:len(failures) - _PROCESSED_MESSAGE_LIMIT]:
            failures.pop(old_id, None)

    fingerprint = hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:12]
    logging.getLogger("optimist_prime").error(
        "Private message processing failed; retry deferred",
        extra={
            "event_name": "private_chat.message.failed",
            "component": "private_chat",
            "phase": "engagement.private_chat",
            "error_type": type(exc).__name__,
            "cause_error_type": (
                type(exc.__cause__).__name__ if exc.__cause__ is not None else None
            ),
            "operation_stage": str(stage)[:80],
            "message_fingerprint": fingerprint,
            "retry_after_seconds": delay,
            # Preserve source locations for diagnosis without serializing the
            # exception text, which could contain private message content.
            "stack_trace": "".join(traceback.format_tb(exc.__traceback__))[-12000:],
        },
    )
    return delay


def _send_and_record(message, text: str, *, state: dict, author: str, moderator: bool, conversation_key: str):
    response = send_message_reply(message, text)
    message_id = str(getattr(message, "id", ""))
    state.setdefault("private_message_conversation_ids", {})[str(getattr(response, "id", ""))] = conversation_key
    history_row = state.setdefault("private_conversations", {}).setdefault(conversation_key, {"history": []})
    history = history_row.setdefault("history", [])
    history.extend([
        {"role": "user", "text": str(getattr(message, "body", "") or "")[:1500]},
        {"role": "assistant", "text": str(text or "")[:1500]},
    ])
    history_row["history"] = history[-_PRIVATE_HISTORY_LIMIT:]
    history_row["updated_at"] = datetime.now(timezone.utc).timestamp()
    processed = state.setdefault("processed_private_messages", [])
    if message_id not in processed:
        processed.append(message_id)
    state["processed_private_messages"] = processed[-_PROCESSED_MESSAGE_LIMIT:]
    if not moderator:
        _record_user_reply(state, author)
    record_reply_success(state, moderator=moderator)
    try:
        mark_message_read(message)
    except Exception as exc:
        print(f"  ⚠️ Private reply sent but could not mark the message read ({type(exc).__name__})")
    return response


def process_private_chat(
    reddit,
    model,
    state: dict,
    bot_username: str,
    *,
    dry_run: bool = False,
    persist_state=None,
    jev_client=None,
) -> tuple[int, int, float, dict, PrivateChatCycleMetrics]:
    """Handle direct private messages; group chat rooms are intentionally unsupported."""
    if dry_run:
        state = deepcopy(state)
    try:
        messages = unread_private_messages(reddit, limit=100)
    except Exception as exc:
        print(f"  ⚠️ Could not read private messages ({type(exc).__name__})")
        return 0, 0, 0.0, state, PrivateChatCycleMetrics(scan_complete=False)

    unread = [
        message for message in messages
        if _destination_name(message).casefold() == str(bot_username).casefold()
    ]
    if not unread:
        return 0, 0, 0.0, state, PrivateChatCycleMetrics()

    subreddit = reddit.subreddit(config.SUBREDDIT)
    roster = refresh_moderators(subreddit, state)
    current_mods = roster or set()
    processed_order = list(dict.fromkeys(
        str(value) for value in state.get("processed_private_messages", []) if value
    ))
    processed = set(processed_order)

    def remember_processed(message_id: str) -> None:
        if message_id not in processed:
            processed.add(message_id)
            processed_order.append(message_id)

    pending = state.setdefault("private_message_pending_replies", {})
    failures = state.setdefault("private_message_failures", {})

    def complete_private_abstention(
        message,
        message_id: str,
        body: str,
        conversation_key: str,
    ) -> None:
        """Persist a no-reply decision without sending its internal marker."""
        history_row = state.setdefault("private_conversations", {}).setdefault(
            conversation_key, {"history": []}
        )
        history = history_row.setdefault("history", [])
        if not any(
            isinstance(row, dict) and row.get("source_message_id") == message_id
            for row in history
        ):
            history.append({
                "role": "user",
                "text": body[:1500],
                "source_message_id": message_id,
            })
        history_row["history"] = history[-_PRIVATE_HISTORY_LIMIT:]
        history_row["updated_at"] = datetime.now(timezone.utc).timestamp()
        state.setdefault("private_message_conversation_ids", {})[
            message_id
        ] = conversation_key
        pending.pop(message_id, None)
        remember_processed(message_id)
        failures.pop(message_id, None)
        if callable(persist_state):
            persist_state(state)
        try:
            mark_message_read(message)
        except Exception as exc:
            print(
                "  ⚠️ Private abstention recorded but could not mark the message read "
                f"({type(exc).__name__})"
            )

    replies_sent = total_tokens = 0
    total_cost = 0.0
    messages_handled = 0
    messages_failed = 0
    messages_deferred = 0
    scan_complete = True
    counted_message_ids: set[str] = set()

    for message in sorted(unread, key=lambda item: float(getattr(item, "created_utc", 0) or 0)):
        message_id = str(getattr(message, "id", "") or "")
        if not message_id:
            continue
        if message_id in processed:
            try:
                mark_message_read(message)
            except Exception:
                pass
            continue
        failure = failures.get(message_id)
        terminal_jev_failure = (
            isinstance(failure, dict)
            and failure.get("error_type") == "JevHistoryReviewError"
        )
        if isinstance(failure, dict) and not terminal_jev_failure:
            try:
                retry_after = float(failure.get("retry_after", 0) or 0)
            except (TypeError, ValueError):
                retry_after = 0
            if retry_after > datetime.now(timezone.utc).timestamp():
                messages_deferred += 1
                scan_complete = False
                continue
        author_name = _author_name(message)
        # The private message is already addressed to Optimist. Suppress only
        # Optimist's own accounts; a human username ending in "bot" remains
        # a valid user conversation.
        if not author_name or is_bot_owned_author(author_name, bot_username):
            remember_processed(message_id)
            failures.pop(message_id, None)
            continue
        body = str(getattr(message, "body", "") or "").strip()
        if not body or body.casefold() == "[deleted]":
            remember_processed(message_id)
            failures.pop(message_id, None)
            continue
        created = float(getattr(message, "created_utc", 0) or 0)
        if (
            created
            and (datetime.now(timezone.utc).timestamp() - created) > config.MAX_AGE_HOURS * 3600
            and not terminal_jev_failure
        ):
            remember_processed(message_id)
            failures.pop(message_id, None)
            try:
                mark_message_read(message)
            except Exception:
                pass
            continue

        moderator = author_name.casefold() in current_mods
        if terminal_jev_failure and not pending.get(message_id):
            # These failures were already submitted to Jev. Queue one plain
            # failure reply and use the durable send path without repeating
            # the same analysis just because the source message stayed unread.
            conversation_key = _conversation_key(message, author_name, state)
            pending[message_id] = {
                "text": _JEV_HISTORY_FAILURE_REPLY,
                "conversation_key": conversation_key,
            }
            if callable(persist_state):
                persist_state(state)
        if not moderator and not terminal_jev_failure:
            if int(state.get("daily_non_moderator_replies", 0) or 0) >= config.MAX_REPLIES_PER_DAY:
                continue
            if int(state.get("_cycle_non_moderator_replies", 0) or 0) >= config.MAX_REPLIES_PER_RUN:
                continue
            if check_user_cooldown(author_name, state.get("recent_user_replies", {})):
                continue
        if dry_run:
            print(f"  🧪 [DRY RUN] Would answer a private message from u/{author_name}")
            continue

        # Count a distinct message once it passes the normal eligibility,
        # duplicate, age, and user-limit gates and enters actual processing.
        # This includes retryable replies because they still consumed work.
        if message_id not in counted_message_ids:
            counted_message_ids.add(message_id)
            messages_handled += 1

        processing_stage = "conversation_context"
        try:
            conversation_key = _conversation_key(message, author_name, state)
            queued_reply = pending.get(message_id)
            if isinstance(queued_reply, dict) and queued_reply.get("text"):
                queued_text = str(queued_reply["text"])
                queued_conversation_key = str(
                    queued_reply.get("conversation_key") or conversation_key
                )
                if queued_text.strip() == NO_PUBLIC_REPLY:
                    processing_stage = "persist_private_abstention"
                    complete_private_abstention(
                        message,
                        message_id,
                        body,
                        queued_conversation_key,
                    )
                    continue
                processing_stage = "send_queued_reply"
                _send_and_record(
                    message,
                    queued_text,
                    state=state,
                    author=author_name,
                    moderator=moderator,
                    conversation_key=queued_conversation_key,
                )
                pending.pop(message_id, None)
                remember_processed(message_id)
                state.setdefault("private_message_failures", {}).pop(message_id, None)
                if callable(persist_state):
                    persist_state(state)
                replies_sent += 1
                continue
            history_row = state.setdefault("private_conversations", {}).setdefault(conversation_key, {"history": []})
            history = history_row.get("history", [])
            trusted_context = _private_trusted_context(history)
            message_model = model_without_per_run_cap(model) if moderator else model
            shim = _message_shim(message, author_name)
            response_text = None
            if moderator:
                processing_stage = "moderator_action_selection_and_response"
                turn = process_moderator_turn(
                    reddit,
                    subreddit,
                    state,
                    message_model,
                    author_name=author_name,
                    text=body,
                    channel="private",
                    source_id=message_id,
                    conversation_key=conversation_key,
                    bot_username=bot_username,
                    verified_moderator=True,
                    interaction=shim,
                    trusted_context=trusted_context,
                    conversation_history=history,
                    conversation_updated_at=history_row.get("updated_at"),
                    jev_client=jev_client,
                    dry_run=dry_run,
                    persist_state=persist_state,
                )
                if turn is None:
                    raise RuntimeError("moderator turn was not interpreted")
                usage = turn.get("token_info", {})
                total_tokens += int(usage.get("total_tokens", 0) or 0)
                total_cost += float(usage.get("cost", 0.0) or 0.0)
                state = turn.get("state", state)
                response_text = (
                    turn.get("response") if turn.get("handled")
                    else turn.get("conversation_response")
                )
            else:
                processing_stage = "public_interaction_route"
                allowed_actions = []
                if config.USER_CUSTOM_FLAIR_ENABLED:
                    allowed_actions.append("self_custom_flair")
                routed = route_public_interaction(
                    jev_client,
                    text=body,
                    channel="private",
                    summoned=True,
                    conversation_context=trusted_context,
                    allowed_actions=allowed_actions,
                )
                usage = routed.get("token_info", {}) if isinstance(routed, dict) else {}
                if isinstance(usage, dict):
                    total_tokens += int(
                        usage.get("total_tokens", usage.get("input_tokens", 0)) or 0
                    )
                    total_cost += float(usage.get("cost", 0.0) or 0.0)
                if not isinstance(routed, dict) or not routed.get("complete"):
                    error = routed.get("error") if isinstance(routed, dict) else None
                    raise RuntimeError(
                        f"Jev could not route this private message"
                        f" ({str(error or 'incomplete decision')[:100]})"
                    )

                route = str(routed.get("route", "") or "").strip().casefold()
                action = str(routed.get("action", "") or "").strip().casefold()
                if route in {"no_reply", "not_addressed"}:
                    response_text = NO_PUBLIC_REPLY
                elif route == "conversation":
                    processing_stage = "conversation_response_generation"
                    response_text, token_info = generate_conversation_response_from_text(
                        body,
                        f"Private conversation with u/{author_name}.\n{trusted_context}",
                        message_model,
                    )
                    total_tokens += int(token_info.get("total_tokens", 0) or 0)
                    total_cost += float(token_info.get("cost", 0.0) or 0.0)
                elif route == "clarify":
                    response_text = (
                        "I’m not sure what you’d like me to do. Could you say a bit more?"
                    )
                elif route == "unsupported_mod_action":
                    response_text = (
                        "I can’t do moderator-only actions. A moderator for this "
                        "subreddit can request those."
                    )
                elif route == "self_custom_flair":
                    processing_stage = "self_flair_request"
                    intent = {"action": action}
                    requested_flair = str(routed.get("requested_flair", "") or "").strip()
                    if requested_flair:
                        intent["label"] = requested_flair
                    response_text, used_tokens, used_cost, state = (
                        handle_self_custom_flair_request(
                            shim,
                            subreddit,
                            reddit,
                            message_model,
                            state,
                            intent,
                            dry_run=dry_run,
                            jev_client=jev_client,
                            intent_override=intent,
                        )
                    )
                    total_tokens += used_tokens
                    total_cost += used_cost
                else:
                    raise RuntimeError("Jev returned an unsupported private-message route")
            if not response_text:
                raise ValueError("Private conversation produced an empty response")

            if str(response_text).strip() == NO_PUBLIC_REPLY:
                processing_stage = "persist_private_abstention"
                complete_private_abstention(
                    message, message_id, body, conversation_key
                )
                continue

            pending[message_id] = {"text": response_text, "conversation_key": conversation_key}
            if callable(persist_state):
                processing_stage = "persist_pending_reply"
                persist_state(state)
            processing_stage = "send_reply"
            _send_and_record(
                message,
                response_text,
                state=state,
                author=author_name,
                moderator=moderator,
                conversation_key=conversation_key,
            )
            if moderator and isinstance(turn, dict):
                turn["_response_delivered"] = True
            pending.pop(message_id, None)
            remember_processed(message_id)
            state.setdefault("private_message_failures", {}).pop(message_id, None)
            if callable(persist_state):
                processing_stage = "persist_completed_reply"
                persist_state(state)
            replies_sent += 1
        except Exception as exc:
            messages_failed += 1
            scan_complete = False
            retry_delay = _record_private_message_failure(
                state, message_id, exc, stage=processing_stage
            )
            if callable(persist_state):
                try:
                    persist_state(state)
                except Exception as persist_exc:
                    log_event(
                        "private_chat.retry_state_save_failed",
                        "Private message retry metadata could not be saved",
                        level=logging.ERROR,
                        component="private_chat",
                        error_type=type(persist_exc).__name__,
                        operation_stage="persist_retry_state",
                    )
            print(
                "  ⚠️ A private message could not be processed "
                f"({type(exc).__name__}); retrying in about {retry_delay}s"
            )
            continue

    state["processed_private_messages"] = processed_order[-_PROCESSED_MESSAGE_LIMIT:]
    conversations = state.setdefault("private_conversations", {})
    if len(conversations) > 500:
        oldest = sorted(
            conversations,
            key=lambda key: float(conversations[key].get("updated_at", 0) or 0)
            if isinstance(conversations.get(key), dict) else 0,
        )
        for old_key in oldest[:len(conversations) - 500]:
            conversations.pop(old_key, None)
    message_map = state.setdefault("private_message_conversation_ids", {})
    if len(message_map) > _PROCESSED_MESSAGE_LIMIT * 2:
        for old_key in list(message_map)[:len(message_map) - _PROCESSED_MESSAGE_LIMIT * 2]:
            message_map.pop(old_key, None)
    return (
        replies_sent,
        total_tokens,
        total_cost,
        state,
        PrivateChatCycleMetrics(
            messages_handled=messages_handled,
            scan_complete=scan_complete,
            messages_failed=messages_failed,
            messages_deferred=messages_deferred,
        ),
    )


__all__ = ["process_private_chat"]
