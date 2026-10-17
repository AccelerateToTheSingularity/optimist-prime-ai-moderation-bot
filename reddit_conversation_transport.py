"""Supported Reddit message transport adapters used by the conversation layer."""

from __future__ import annotations

from bot_utils import validate_reply_response
from persona import NO_PUBLIC_REPLY
from public_response_safety import post_public_comment


# Reddit's published Data API exposes private-message inbox/reply operations.
# It does not publish native group-room history or send operations for this
# authenticated PRAW client, so no undocumented chat endpoints are attempted.
PRIVATE_MESSAGES_SUPPORTED = True
NATIVE_GROUP_CHAT_SUPPORTED = False


def is_private_message(item) -> bool:
    """Recognize a PRAW private-message object, excluding comment notifications."""
    try:
        from praw.models import Message
        if isinstance(item, Message):
            return True
    except Exception:
        pass
    return bool(getattr(item, "_reddit_private_message", False))


def unread_private_messages(reddit, *, limit: int = 100) -> list:
    """Read unread private messages without marking them read during fetch."""
    try:
        rows = reddit.inbox.unread(mark_read=False, limit=limit)
    except TypeError:
        # Older PRAW signatures still default to non-destructive reads.
        rows = reddit.inbox.unread(limit=limit)
    return [item for item in rows if is_private_message(item)]


def _public_message_text(text: str) -> str:
    value = str(text or "")
    if value.strip() == NO_PUBLIC_REPLY:
        raise ValueError("Internal no-reply marker cannot be sent to Reddit")
    return value


def send_message_reply(message, text: str):
    """Reply in the same private-message conversation and return its response."""
    response = message.reply(_public_message_text(text))
    if response is None or not getattr(response, "id", None):
        raise ValueError("Reddit did not confirm the private reply")
    return response


def send_private_message(reddit, username: str, subject: str, text: str):
    """Start a supported one-to-one private conversation with a Redditor."""
    response = reddit.redditor(username).message(
        subject=subject[:100], message=_public_message_text(text)
    )
    # PRAW's documented Redditor.message implementation returns None after a
    # successful /api/compose response. Reaching here without an exception is
    # Reddit's positive acceptance result; requiring a message object would
    # incorrectly send the public-comment fallback as a duplicate.
    return response if response is not None else True


def mark_message_read(message) -> None:
    mark_read = getattr(message, "mark_read", None)
    if callable(mark_read):
        mark_read()


def remove_processed_instruction(comment) -> bool:
    """Remove a moderator's successfully processed instruction and verify it."""
    comment.mod.remove()
    refresh = getattr(comment, "refresh", None)
    if callable(refresh):
        refresh()
    removed_by = getattr(comment, "banned_by", None)
    category = getattr(comment, "removed_by_category", None)
    return bool(removed_by or category)


__all__ = [
    "NATIVE_GROUP_CHAT_SUPPORTED",
    "PRIVATE_MESSAGES_SUPPORTED",
    "is_private_message",
    "mark_message_read",
    "remove_processed_instruction",
    "send_message_reply",
    "send_private_message",
    "unread_private_messages",
]
