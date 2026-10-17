"""
Shared utility functions for bot detection and common checks.
"""

import re
from copy import deepcopy
from datetime import datetime, timezone

import config
from config import BOT_INDICATORS


def is_likely_bot(
    author_name: str | None,
    runtime_bot_username: str | None = None,
) -> bool:
    """
    Check if an author is likely a bot based on username patterns.

    Checks:
    - BOT_OWNED_USERNAMES and optional runtime bot identity
    - BOT_INDICATORS regex patterns
    - Common bot username suffixes (endswith "bot", "-mod")
    - "automod" substring
    """
    if not author_name:
        return True  # Treat deleted users as bots

    name_lower = author_name.lower()

    # Check owned bot usernames first (plus runtime identity when provided)
    if is_bot_owned_author(author_name, runtime_bot_username):
        return True

    # Check common bot suffixes
    if name_lower.endswith("bot") or name_lower.endswith("-mod") or "automod" in name_lower:
        return True

    # Check regex patterns
    for pattern in BOT_INDICATORS:
        if re.search(pattern.lower(), name_lower):
            return True

    return False


def is_bot_owned_author(author_name: str | None, runtime_bot_username: str | None = None) -> bool:
    """Return True for deleted authors, the runtime bot identity, or configured aliases."""
    if not author_name:
        return True
    normalized = author_name.strip().casefold()
    known = {
        name.strip().casefold()
        for name in config.BOT_OWNED_USERNAMES
        if name and name.strip()
    }
    if config.EXPECTED_BOT_USERNAME:
        known.add(config.EXPECTED_BOT_USERNAME.strip().casefold())
    if runtime_bot_username:
        known.add(runtime_bot_username.strip().casefold())
    return normalized in known


def normalize_reddit_markdown_for_intent(text: str) -> str:
    """Undo Markdown punctuation escapes before parsing Reddit usernames."""
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", str(text or ""))


def is_too_old(created_utc: float) -> bool:
    """Check if a post/comment is older than MAX_AGE_HOURS."""
    age_seconds = datetime.now(timezone.utc).timestamp() - created_utc
    age_hours = age_seconds / 3600
    return age_hours > config.MAX_AGE_HOURS


def check_user_cooldown(author_name: str | None, recent_replies: dict) -> bool:
    """Check if we've recently replied to this user too many times."""
    from config import SAME_USER_COOLDOWN_HOURS, SAME_USER_REPLIES_BEFORE_COOLDOWN

    if not author_name or author_name not in recent_replies:
        return False

    user_data = recent_replies[author_name]
    reply_count = user_data.get("count", 0)
    first_reply_time = user_data.get("first_reply_time", 0)

    # If under the limit, allow reply
    if reply_count < SAME_USER_REPLIES_BEFORE_COOLDOWN:
        return False

    # Over limit - check if cooldown has expired
    hours_since = (datetime.now(timezone.utc).timestamp() - first_reply_time) / 3600
    return hours_since < SAME_USER_COOLDOWN_HOURS


def get_daily_non_moderator_reply_count(state: dict) -> int:
    """Return the daily reply count that is subject to the ordinary-user cap.

    ``daily_replies`` is retained as the historical total, including moderator
    replies.  New state records the separately limited ordinary-user count.
    The fallback keeps an older state file conservative during migration.
    """
    value = state.get("daily_non_moderator_replies")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    legacy_total = state.get("daily_replies", 0)
    if isinstance(legacy_total, (int, float)) and not isinstance(legacy_total, bool):
        return max(0, int(legacy_total))
    return 0


def record_reply_success(state: dict, *, moderator: bool) -> None:
    """Record one successful engagement action without claiming failed work.

    Moderator actions remain visible in the total operational count, but do not
    consume the ordinary-user daily allowance.
    """
    state["daily_replies"] = max(0, int(state.get("daily_replies", 0) or 0)) + 1
    if not moderator:
        state["daily_non_moderator_replies"] = (
            get_daily_non_moderator_reply_count(state) + 1
        )
        state["_cycle_non_moderator_replies"] = max(
            0, int(state.get("_cycle_non_moderator_replies", 0) or 0)
        ) + 1


def get_cached_moderators(state: dict, subreddit, *, force_refresh: bool = False) -> set:
    """
    Get moderator set from cache, refreshing from Reddit if stale.
    Updates state in-place with cached data.
    """
    from config import MOD_CACHE_REFRESH_DAYS

    now = datetime.now(timezone.utc).timestamp()
    cache_max_age = MOD_CACHE_REFRESH_DAYS * 24 * 3600  # Convert days to seconds

    cached_mods = state.get("moderator_cache", {})
    last_refresh = cached_mods.get("last_refresh", 0)
    mod_list = cached_mods.get("moderators", [])

    # Check if cache is fresh enough
    if not force_refresh and mod_list and (now - last_refresh) < cache_max_age:
        return set(m.lower() for m in mod_list)

    # Cache is stale or empty - refresh from Reddit
    try:
        fresh_mods = [mod.name for mod in subreddit.moderator()]
        state["moderator_cache"] = {
            "moderators": fresh_mods,
            "last_refresh": now
        }
        print(f"    🔄 Refreshed moderator cache ({len(fresh_mods)} mods)")
        return set(m.lower() for m in fresh_mods)
    except Exception as e:
        print(f"    ⚠️ Could not refresh mod cache: {e}")
        # Return stale cache if available, otherwise empty
        return set(m.lower() for m in mod_list)


def refresh_moderators(subreddit, state: dict) -> set[str] | None:
    """Read Reddit's current moderator roster; never authorize from stale cache."""
    now = datetime.now(timezone.utc).timestamp()
    try:
        fresh_mods = [str(mod.name) for mod in subreddit.moderator() if getattr(mod, "name", None)]
    except Exception as exc:
        state["_current_moderators_verified"] = False
        state["_current_moderators"] = []
        print(f"    ⚠️ Could not verify the current moderator list: {type(exc).__name__}")
        return None
    state["moderator_cache"] = {"moderators": fresh_mods, "last_refresh": now}
    state["_current_moderators_verified"] = True
    state["_current_moderators"] = [name.casefold() for name in fresh_mods]
    return {name.casefold() for name in fresh_mods}


def is_current_moderator(author_name: str | None, state: dict, subreddit) -> bool:
    """Use this cycle's live roster when available for uncapped conversation replies."""
    if not author_name:
        return False
    if "_current_moderators_verified" in state:
        return bool(
            state.get("_current_moderators_verified")
            and author_name.casefold() in set(state.get("_current_moderators", []))
        )
    return is_moderator(author_name, state, subreddit)


def is_moderator(author_name: str | None, state: dict, subreddit) -> bool:
    """Check if a user is a moderator of the subreddit."""
    if not author_name:
        return False
    mods = get_cached_moderators(state, subreddit)
    return author_name.lower() in mods


def validate_reply_response(response, item_type="comment") -> bool:
    """
    Validate that a Reddit API reply response has expected attributes.

    Args:
        response: The PRAW response object from reply()
        item_type: Description for error messages (e.g. "comment", "post")

    Returns:
        True if valid, raises ValueError if not
    """
    if response is None:
        raise ValueError(f"Reddit API returned None response for {item_type}")
    if not hasattr(response, "id") or not response.id:
        raise ValueError(f"Reddit API returned response without valid ID for {item_type}")
    if not hasattr(response, "permalink"):
        raise ValueError(f"Reddit API returned response without permalink for {item_type}")
    return True


def is_content_approved(content_obj) -> bool:
    """Return True if Reddit content is moderator-approved."""
    try:
        return bool(getattr(content_obj, "approved", False))
    except Exception:
        return False


def claim_action(state: dict, key: str, *, max_keys: int = 5000) -> bool:
    """
    Record a one-time action key in bot state. Returns True if this is the first claim.

    Keys are namespaced by the feature and Reddit item they track.
    """
    action_keys = state.setdefault("action_keys", [])
    if not isinstance(action_keys, list):
        action_keys = []
        state["action_keys"] = action_keys
    if key in action_keys:
        return False
    action_keys.append(key)
    if len(action_keys) > max_keys:
        state["action_keys"] = action_keys[-max_keys:]
    return True


def begin_action(state: dict, key: str, *, max_pending: int = 5000) -> bool:
    """Reserve an action without marking it complete.

    A reservation is retryable: only :func:`complete_action` moves the key to
    the completed list.  This prevents a failed Reddit API call from making
    the item look permanently processed on the next run.
    """
    if has_action_claim(state, key):
        return False
    pending = state.setdefault("pending_action_keys", {})
    if not isinstance(pending, dict):
        pending = {}
        state["pending_action_keys"] = pending
    if key in pending:
        # A stale reservation from an interrupted run is safe to retry.  The
        # current worker is serialized by the state lock.
        pending.pop(key, None)
    pending[key] = {"started_at": datetime.now(timezone.utc).timestamp()}
    if len(pending) > max_pending:
        oldest = sorted(
            pending.items(),
            key=lambda pair: pair[1].get("started_at", 0)
            if isinstance(pair[1], dict)
            else 0,
        )
        for old_key, _ in oldest[:-max_pending]:
            pending.pop(old_key, None)
    return True


def complete_action(state: dict, key: str, *, max_keys: int = 5000) -> None:
    """Commit a previously reserved action after its side effect succeeds."""
    pending = state.get("pending_action_keys", {})
    if isinstance(pending, dict):
        pending.pop(key, None)
    completed = state.setdefault("action_keys", [])
    if not isinstance(completed, list):
        completed = []
        state["action_keys"] = completed
    if key not in completed:
        completed.append(key)
        if len(completed) > max_keys:
            state["action_keys"] = completed[-max_keys:]
    failures = state.get("action_failures", {})
    if isinstance(failures, dict):
        failures.pop(key, None)


def fail_action(state: dict, key: str, error: str | None = None) -> None:
    """Release a reservation and retain bounded retry diagnostics."""
    pending = state.get("pending_action_keys", {})
    if isinstance(pending, dict):
        pending.pop(key, None)
    failures = state.setdefault("action_failures", {})
    if not isinstance(failures, dict):
        failures = {}
        state["action_failures"] = failures
    if isinstance(failures, dict):
        failures[key] = {
            "failed_at": datetime.now(timezone.utc).timestamp(),
            "error": (error or "operation failed")[:500],
        }
        if len(failures) > 1000:
            oldest = sorted(
                failures.items(),
                key=lambda pair: pair[1].get("failed_at", 0)
                if isinstance(pair[1], dict)
                else 0,
            )
            state["action_failures"] = dict(oldest[-1000:])


def copy_state_for_dry_run(state: dict) -> dict:
    """Return an isolated state object for callers that promise no writes."""
    return deepcopy(state)


def has_action_claim(state: dict, key: str) -> bool:
    """Return True if an action key was already claimed."""
    action_keys = state.get("action_keys", [])
    return isinstance(action_keys, list) and key in action_keys
