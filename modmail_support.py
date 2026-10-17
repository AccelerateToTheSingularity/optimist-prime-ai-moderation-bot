"""Small helpers for private, moderator-only Modmail conversations."""

from __future__ import annotations

from typing import Any


def find_internal_conversation(subreddit, subject: str, *, limit: int = 100):
    """Find the newest internal moderator conversation with this exact subject."""
    conversations = subreddit.modmail.conversations(
        state="mod", sort="recent", limit=max(1, int(limit))
    )
    for conversation in conversations:
        if (
            bool(getattr(conversation, "is_internal", False))
            and str(getattr(conversation, "subject", "")).strip() == subject
        ):
            return conversation
    return None


def find_or_create_internal_conversation(
    subreddit,
    subject: str,
    opening_body: str,
):
    """Reuse an existing private team thread or create one with no user recipient."""
    existing = find_internal_conversation(subreddit, subject)
    if existing is not None:
        return existing, False
    conversation = subreddit.modmail.create(
        subject=subject,
        body=opening_body,
        recipient=None,
    )
    if not getattr(conversation, "id", None):
        raise RuntimeError("Reddit did not return an ID for the internal Modmail conversation")
    return conversation, True


def send_internal_notification(subreddit, subject: str, body: str):
    """Create one private team thread, reusing an exact-subject retry if present."""
    conversation, created = find_or_create_internal_conversation(
        subreddit, subject, body
    )
    if not bool(getattr(conversation, "is_internal", False)):
        raise RuntimeError("Reddit did not return an internal moderator conversation")
    return conversation, created


def reply_internal(conversation, body: str):
    """Reply only as an internal moderator note, never to an outside user."""
    if not bool(getattr(conversation, "is_internal", False)):
        raise ValueError("Refusing to reply in a Modmail conversation that is not internal")
    if not str(body or "").strip():
        raise ValueError("Refusing to send an empty internal Modmail reply")
    return conversation.reply(body=body, internal=True)


def reply_private_note(conversation, body: str):
    """Send a moderator-only note in internal mail, appeals, or other Modmail."""
    if not str(body or "").strip():
        raise ValueError("Refusing to send an empty private Modmail note")
    reply = getattr(conversation, "reply", None)
    if not callable(reply):
        raise TypeError("The Modmail conversation cannot accept a reply")
    return reply(body=body, internal=True)


def message_body(message: Any) -> str:
    return str(
        getattr(message, "body_markdown", None)
        or getattr(message, "body", None)
        or ""
    )


def message_author(message: Any) -> str:
    author = getattr(message, "author", None)
    return str(getattr(author, "name", "") or "").strip()
