"""Small, policy-neutral helpers for operator-approved user flair."""

from __future__ import annotations

import re

import config


def normalize_flair_label(value: str, *, limit: int = 30) -> str:
    """Return a safe, bounded flair label or an empty string."""
    label = re.sub(r"\s+", " ", str(value or "")).strip()
    if (
        not label
        or len(label) > limit
        or "|" in label
        or "\n" in label
        or "\r" in label
        or re.search(r"https?://|\bu/|\br/", label, re.IGNORECASE)
    ):
        return ""
    return label


def first_flair_template_id(subreddit) -> str | None:
    """Return the first available user-flair template, if Reddit provides one."""
    try:
        for template in subreddit.flair.templates:
            if template.get("flair_type") == "USER_FLAIR":
                template_id = template.get("id")
                return str(template_id) if template_id else None
    except Exception:
        return None
    return None


def set_user_flair(
    subreddit,
    username: str,
    label: str,
    *,
    dry_run: bool = False,
    state: dict | None = None,
) -> bool:
    """Set one already-approved label and preserve the flair safety pause."""
    from flair_permission import is_flair_permission_paused, record_flair_write_failure

    if not dry_run and state is not None and is_flair_permission_paused(state):
        return False
    if dry_run:
        print(f"    [DRY RUN] Would set flair for u/{username}: {label!r}")
        return True
    try:
        template_id = first_flair_template_id(subreddit)
        if template_id:
            subreddit.flair.set(username, text=label, flair_template_id=template_id)
        else:
            subreddit.flair.set(username, text=label or None)
        return True
    except Exception as exc:
        record_flair_write_failure(state, exc)
        print(f"    Could not set flair for u/{username} ({type(exc).__name__}).")
        return False
