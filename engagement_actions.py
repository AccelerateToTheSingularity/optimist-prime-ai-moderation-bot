"""Configurable action modes for user-facing bot engagement.

Actions are selected by the operator and remain subject to feature settings,
Safe Mode, dry-run, and success-before-claim checks in their callers.
"""

from __future__ import annotations

from typing import Any

from bot_comment_format import format_bot_comment
from bot_utils import validate_reply_response
from modmail_support import send_internal_notification
from public_response_safety import post_public_comment

ENGAGEMENT_ACTIONS = frozenset({"comment", "report", "modmail", "log"})


def normalize_engagement_action(action: str | None) -> str:
    """Return a supported engagement action, defaulting to a public reply."""
    normalized = str(action or "comment").strip().lower()
    return normalized if normalized in ENGAGEMENT_ACTIONS else "comment"


def _content_type(content_obj: Any) -> str:
    return "post" if hasattr(content_obj, "title") else "comment"


def _permalink(content_obj: Any) -> str:
    raw = str(getattr(content_obj, "permalink", "") or "")
    if raw.startswith(("http://", "https://")):
        return raw
    return f"https://www.reddit.com{raw}" if raw else ""


def _safe_audit(
    event_type: str,
    content_obj: Any,
    author_name: str,
    source_text: str,
    action: str,
    message: str,
) -> None:
    """Audit without turning a successful Reddit action into a retry."""
    try:
        from audit_log import log_audit_event

        log_audit_event(
            event_type,
            str(getattr(content_obj, "id", "")),
            author_name or "[deleted]",
            source_text,
            action,
            True,
            message=message,
        )
    except Exception as exc:
        print(f"    ⚠️ Engagement audit logging failed: {exc}")


def execute_engagement_action(
    content_obj: Any,
    subreddit: Any,
    author_name: str,
    source_text: str,
    response_text: str,
    *,
    event_type: str,
    action_mode: str | None = None,
    dry_run: bool = False,
    response_state: dict | None = None,
    persist_state=None,
    bot_username: str | None = None,
) -> bool:
    """Execute one configured engagement action and return success.

    ``comment`` posts the generated response, ``report`` reports the source
    item for moderator review, ``modmail`` sends the response privately, and
    ``log`` records the event without contacting Reddit.  The function never
    performs a ban or removal.
    """
    action = normalize_engagement_action(action_mode)
    content_type = _content_type(content_obj)
    item_id = str(getattr(content_obj, "id", "") or "")
    response_text = str(response_text or "").strip()
    source_text = str(source_text or "")

    if dry_run:
        print(
            f"       [DRY RUN] Would {action} {event_type} for {content_type} "
            f"{item_id} by u/{author_name}: {response_text[:120]}"
        )
        return True

    try:
        if action == "comment":
            if response_state is None or not callable(persist_state) or not bot_username:
                print(
                    f"       ❌ Refusing {event_type} public reply without durable response state"
                )
                return False
            source_type = "post" if hasattr(content_obj, "title") else "comment"
            reply = post_public_comment(
                content_obj,
                format_bot_comment(response_text),
                response_state,
                source_id=item_id,
                author_name=author_name,
                bot_username=bot_username,
                persist_state=persist_state,
                source_type=source_type,
                action=event_type,
                subreddit=subreddit,
                dry_run=dry_run,
                validate=validate_reply_response,
                item_type=f"{event_type} reply",
            )
            if reply is None:
                return False
            try:
                reply.mod.distinguish(sticky=False)
            except Exception as distinguish_error:
                # The reply was already created and validated.  Do not turn a
                # failed optional presentation step into a duplicate-reply
                # retry on the next polling cycle.
                print(f"       ⚠️ {event_type} reply posted but could not distinguish it: {distinguish_error}")
            print(f"       ✅ Posted {event_type} reply for {content_type} {item_id}")
            _safe_audit(
                event_type,
                content_obj,
                author_name,
                source_text,
                "replied",
                "Posted public engagement reply",
            )
            return True

        if action == "report":
            content_obj.report(
                reason=f"Optimist Prime {event_type}: {response_text[:95]}"
            )
            print(f"       ✅ Reported {event_type} source {content_type} {item_id}")
            _safe_audit(
                event_type,
                content_obj,
                author_name,
                source_text,
                "reported",
                "Reported the engagement source instead of replying",
            )
            return True

        if action == "modmail":
            if not item_id:
                raise ValueError(
                    "Refusing to send an engagement Modmail without a stable source ID"
                )
            link = _permalink(content_obj)
            body = (
                f"**Optimist Prime {event_type} escalation**\n\n"
                f"**Author:** u/{author_name}\n"
                f"**Source:** {link or item_id}\n\n"
                f"**Generated response/context:**\n{response_text}"
            )
            # A Reddit write can succeed even when its acknowledgement is
            # lost.  Key the internal conversation by source ID so a retry
            # finds the accepted message instead of creating another one.
            subject = (
                f"Optimist Prime {str(event_type).title()[:24]} "
                f"[{item_id[:24]}]"
            )
            send_internal_notification(subreddit, subject, body)
            print(f"       ✅ Sent {event_type} modmail for {content_type} {item_id}")
            _safe_audit(
                event_type,
                content_obj,
                author_name,
                source_text,
                "modmail",
                "Sent private engagement escalation",
            )
            return True

        # ``log`` is intentionally local-only.  A logging failure means the
        # event remains retryable because no Reddit side effect occurred.
        from audit_log import log_audit_event

        log_audit_event(
            event_type,
            item_id,
            author_name or "[deleted]",
            source_text,
            "logged",
            True,
            message="Engagement event logged without a Reddit reply",
        )
        print(f"       📝 Logged {event_type} without a Reddit reply")
        return True
    except Exception as exc:
        print(f"       ❌ Engagement action {action} failed for {item_id}: {exc}")
        return False
