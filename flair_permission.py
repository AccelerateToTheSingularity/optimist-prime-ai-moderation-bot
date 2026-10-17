"""Shared pause for moderator-flair writes rejected by Reddit permissions."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from bot_logging import log_event


STATE_KEY = "reddit_flair_permission_pause"


def is_missing_flair_scope(exc: BaseException) -> bool:
    """Recognize an explicit 403 scope failure without pausing on other 403s."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status is not None:
        try:
            if int(status) != 403:
                return False
        except (TypeError, ValueError):
            return False

    details = " ".join(
        str(value or "")
        for value in (
            getattr(exc, "error_type", None),
            getattr(exc, "message", None),
            type(exc).__name__,
            exc,
        )
    ).casefold().replace("_", " ").replace("-", " ")
    return "403" in details and any(
        marker in details
        for marker in ("insufficient scope", "insufficientscope", "flair")
    )


def is_flair_permission_paused(state: dict[str, Any]) -> bool:
    pause = state.get(STATE_KEY)
    return isinstance(pause, dict) and pause.get("blocked") is True


def pause_flair_writes(state: dict[str, Any], exc: BaseException) -> bool:
    """Pause all flair writes after Reddit explicitly denies a flair scope."""
    if not is_missing_flair_scope(exc):
        return False
    if is_flair_permission_paused(state):
        return True
    state[STATE_KEY] = {
        "blocked": True,
        "reason": "Reddit denied required flair access (403 InsufficientScope)",
        "since": datetime.now(timezone.utc).isoformat(),
    }
    log_event(
        "flair.permission_paused",
        "Reddit denied moderator flair access; all flair changes are paused",
        level=logging.WARNING,
        component="flair_permission",
        required_scopes="flair,modflair",
    )
    return True


def record_flair_write_failure(state: dict[str, Any] | None, exc: BaseException) -> bool:
    """Remember a scope failure when the caller has the durable bot state."""
    if state is None:
        return False
    return pause_flair_writes(state, exc)


def clear_flair_permission_pause(state: dict[str, Any]) -> bool:
    """Clear the shared pause after the helper verifies required flair scopes."""
    return state.pop(STATE_KEY, None) is not None
