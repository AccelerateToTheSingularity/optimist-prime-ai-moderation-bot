"""Exact self-service flair requests using an operator-managed allowlist."""

from __future__ import annotations

import re
from typing import Any

import config
from bot_utils import is_bot_owned_author, normalize_reddit_markdown_for_intent
from flair_permission import is_flair_permission_paused
from flair_utils import normalize_flair_label, set_user_flair


_QUOTED_LABEL = re.compile(
    r"\b(?:flair|label)\b.{0,40}?[\"'“](.{1,80}?)[\"'”]",
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_LABEL = re.compile(
    r"\b(?:flair|label)\s+(?:to|as|saying|reading|should be)\s+"
    r"([A-Za-z][A-Za-z0-9&' /-]{1,29}?)(?=\s*(?:[.!?,;]|$))",
    re.IGNORECASE,
)


def parse_self_custom_flair_request(text: str) -> dict[str, Any] | None:
    """Recognize an explicit request to set the speaker's own exact flair."""
    body = normalize_reddit_markdown_for_intent(text).strip()
    lower = body.casefold()
    if "flair" not in lower or _is_non_actionable_request(lower):
        return None
    mentioned_users = re.findall(r"\bu/([a-z0-9_-]{1,20})\b", lower)
    if any(not is_bot_owned_author(username) for username in mentioned_users):
        return None
    self_directed = bool(re.search(r"\b(?:my|me|myself|for me|for my account)\b", lower))
    action_requested = bool(re.search(r"\b(?:set|change|make|give|request|want)\b", lower))
    if not self_directed or not action_requested:
        return None
    match = _QUOTED_LABEL.search(body) or _PLAIN_LABEL.search(body)
    label = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
    normalized = normalize_flair_label(label, limit=min(30, config.USER_FLAIR_MAX_LEN))
    if normalized:
        return {"action": "set", "label": normalized}
    if label:
        return {"action": "invalid"}
    return None


def _is_non_actionable_request(lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:don't|dont|do not|never|no longer)\s+"
            r"(?:(?:want|need|ask|request)\s+(?:me\s+to\s+|you\s+to\s+)?)?"
            r"(?:set|change|make|give|request|want|need)\b"
            r"|\b(?:not asking|not requesting|what if|hypothetically|pretend|imagine)\b",
            lower,
        )
    )


def handle_self_custom_flair_request(
    comment,
    subreddit,
    reddit,
    llm_model,
    state: dict,
    intent: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    jev_client=None,
    intent_override: dict[str, Any] | None = None,
) -> tuple[str, int, float, dict]:
    """Set an exact allowlisted flair without inspecting account history."""
    allowed = tuple(
        normalize_flair_label(value, limit=min(30, config.USER_FLAIR_MAX_LEN))
        for value in config.USER_CUSTOM_FLAIR_ALLOWED_VALUES
    )
    allowed = tuple(value for value in allowed if value)
    if not config.USER_CUSTOM_FLAIR_ENABLED:
        return "Custom flair requests are disabled. No flair was changed.", 0, 0.0, state
    if not allowed:
        return "Custom flair is not configured for this subreddit. No flair was changed.", 0, 0.0, state
    if not dry_run and is_flair_permission_paused(state):
        return "Reddit has paused flair changes because the bot lacks flair-management access.", 0, 0.0, state

    author = getattr(comment, "author", None)
    username = str(getattr(author, "name", "") or "").strip()
    if not username:
        return "I couldn't identify your Reddit account, so I left the flair unchanged.", 0, 0.0, state
    selected = intent_override if isinstance(intent_override, dict) else intent or {}
    if selected.get("action") == "invalid":
        return "Please use one of the exact flair labels configured by the moderators.", 0, 0.0, state
    requested = normalize_flair_label(
        str(selected.get("label") or selected.get("flair_label") or ""),
        limit=min(30, config.USER_FLAIR_MAX_LEN),
    )
    canonical = next((value for value in allowed if value.casefold() == requested.casefold()), None)
    if canonical is None:
        choices = ", ".join(f"**{value}**" for value in allowed[:10])
        if not choices:
            return "Please request one of the flair labels configured by the moderators.", 0, 0.0, state
        return f"Please choose one of these configured flair labels: {choices}.", 0, 0.0, state
    if not set_user_flair(subreddit, username, canonical, dry_run=dry_run, state=state):
        return "I couldn't update your flair. No change was recorded as complete.", 0, 0.0, state
    if dry_run:
        return f"Your flair would be set to **{canonical}**. Preview only; nothing was changed.", 0, 0.0, state
    return f"Your flair is now **{canonical}**.", 0, 0.0, state
