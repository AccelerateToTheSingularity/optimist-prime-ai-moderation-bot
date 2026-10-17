"""Centralized configuration for Optimist Prime."""

import os
import sys
from pathlib import Path

from reddit_request_budget import resolve_request_budget_settings


def _configure_windows_console_output() -> None:
    """Keep status output safe on Windows installations using cp1252.

    Optimist Prime uses a few emoji status markers in its operator output.
    Python can otherwise raise UnicodeEncodeError while reporting a handled
    Reddit/API failure, which can make the surrounding operation appear to
    have failed as well.  Reconfigure only streams that expose the modern
    TextIO API; UTF-8-capable streams are left unchanged.
    """
    if os.name != "nt":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        encoding = str(getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if stream is None or not hasattr(stream, "reconfigure") or encoding in {"utf8", "utf"}:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            # A redirected or host-managed stream may not permit changes.
            # Logging and the operation itself must remain usable either way.
            pass


_configure_windows_console_output()


def _env_on(name: str, default: bool = False) -> bool:
    """Read a BOT_* boolean env var; empty/unset uses default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_bounded_int(name: str, minimum: int, maximum: int) -> int | None:
    """Read an explicitly supplied integer policy value within its valid range."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if minimum <= value <= maximum else None


# The target community is required. A blank value prevents accidental actions
# against a sample community during first-run setup.
SUBREDDIT = os.environ.get("BOT_SUBREDDIT", "").strip()

# Reddit API requests may retry after transient network failures. Keep each
# attempt bounded, and record the effective timeout in the local operational log.
REDDIT_API_TIMEOUT_SECONDS = min(
    60, max(5, _env_int("REDDIT_API_TIMEOUT_SECONDS", 15))
)

# Local runtime outputs. The repository-relative defaults work for
# development and tests. Set local paths in .env to keep runtime files separate
# from source control.
RUNTIME_STATE_FILE = os.environ.get("BOT_RUNTIME_STATE_FILE", "data/bot_state.json").strip() or "data/bot_state.json"
RUNTIME_STATS_FILE = os.environ.get("BOT_RUNTIME_STATS_FILE", "data/stats.json").strip() or "data/stats.json"
RUNTIME_ACTIVITY_FILE = os.environ.get("BOT_RUNTIME_ACTIVITY_FILE", "data/activity.json").strip() or "data/activity.json"
RUNTIME_CYCLE_SUMMARY_FILE = os.environ.get("BOT_RUNTIME_CYCLE_SUMMARY_FILE", "data/cycle_summary.json").strip() or "data/cycle_summary.json"

# Reddit documents 100 queries per minute per OAuth client ID, averaged over
# ten minutes. Keep a shared local budget so the worker and read-only observer
# cannot unknowingly spend the same allowance at the same time. The resolver
# also prevents local configuration from raising this safety ceiling.
_REDDIT_REQUEST_BUDGET_SETTINGS = resolve_request_budget_settings(
    os.environ,
    default_root=Path(__file__).resolve().parent,
)
REDDIT_REQUEST_BUDGET_FILE = str(_REDDIT_REQUEST_BUDGET_SETTINGS.path)
REDDIT_REQUEST_BUDGET_PER_MINUTE = _REDDIT_REQUEST_BUDGET_SETTINGS.requests_per_minute
REDDIT_REQUEST_BUDGET_WINDOW_SECONDS = _REDDIT_REQUEST_BUDGET_SETTINGS.window_seconds
REDDIT_REQUEST_BUDGET_MAX_REQUESTS = _REDDIT_REQUEST_BUDGET_SETTINGS.max_requests

# Feature switches default off in a new installation.
POST_TLDR_ENABLED = _env_on("BOT_POST_TLDR_ENABLED", False)
COMMENT_TLDR_ENABLED = _env_on("BOT_COMMENT_TLDR_ENABLED", False)
COMMENT_SUMMARY_ENABLED = _env_on("BOT_COMMENT_SUMMARY_ENABLED", False)

POST_WORD_THRESHOLD = max(1, _env_int("BOT_POST_WORD_THRESHOLD", 250))
COMMENT_WORD_THRESHOLD = max(1, _env_int("BOT_COMMENT_WORD_THRESHOLD", 400))
MAX_TLDR_PER_RUN = max(0, _env_int("BOT_MAX_TLDR_PER_RUN", 1))
MAX_TLDR_PER_DAY = max(0, _env_int("BOT_MAX_TLDR_PER_DAY", 10))
MAX_AGE_HOURS = max(1, _env_int("BOT_MAX_AGE_HOURS", 24))
# The worker keeps these limits deliberately separate from per-user history
# limits.  They bound the amount of subreddit work one cycle can inspect while
# the catch-up cursor lets a later cycle continue through a longer outage.
POST_SCAN_LIMIT = max(1, _env_int("BOT_POST_SCAN_LIMIT", 100))
COMMENT_SCAN_LIMIT = max(1, _env_int("BOT_COMMENT_SCAN_LIMIT", 100))
COMMENT_MILESTONES = tuple(
    sorted({
        int(value.strip())
        for value in os.environ.get("BOT_COMMENT_MILESTONES", "").split(",")
        if value.strip().isdigit() and int(value.strip()) > 0
    })
)
POST_TLDR_PIN = os.environ.get("BOT_POST_TLDR_PIN", "").lower() in ("1", "true", "yes", "on")
COMMENT_SUMMARY_PIN = os.environ.get("BOT_COMMENT_SUMMARY_PIN", "").lower() in ("1", "true", "yes", "on")
REDDIT_REFERENCE_TLDR_ENABLED = _env_on("BOT_REDDIT_REFERENCE_TLDR_ENABLED", False)
COMMENT_TLDR_PARENT_CONTEXT_ENABLED = _env_on("BOT_COMMENT_TLDR_PARENT_CONTEXT_ENABLED", False)

# Reply/Conversation Settings
CONVERSATIONAL_REPLIES_ENABLED = _env_on("BOT_CONVERSATIONAL_REPLIES_ENABLED", False)
MAX_REPLIES_PER_RUN = max(
    0, _env_int("BOT_MAX_REPLIES_PER_RUN", 2)
)  # Maximum ordinary-user replies in one run
MAX_REPLIES_PER_DAY = max(
    0, _env_int("BOT_MAX_REPLIES_PER_DAY", 10)
)  # Daily cap for ordinary-user replies
# All structured decisions in one worker cycle share this allowance.
JEV_MAX_CALLS_PER_RUN = max(
    0, _env_int("BOT_JEV_MAX_CALLS_PER_RUN", 30)
)
MAX_REPLY_WORDS = 75  # Target max words for conversational replies (keep it tight)

# Rate limiting
SAME_USER_COOLDOWN_HOURS = max(
    1, _env_int("BOT_SAME_USER_COOLDOWN_HOURS", 1)
)  # Cooldown window for ordinary users
SAME_USER_REPLIES_BEFORE_COOLDOWN = max(
    1, _env_int("BOT_SAME_USER_REPLIES_BEFORE_COOLDOWN", 5)
)  # Replies in the window before cooldown applies
MODERATOR_REPLIES_BYPASS_LIMITS = _env_on(
    "BOT_MODERATOR_REPLIES_BYPASS_LIMITS", False
)  # Moderator-specific rate limits are optional
MOD_CACHE_REFRESH_DAYS = 3  # Refresh moderator list from Reddit every N days

# Summon detection patterns (case-insensitive)
# These patterns will trigger the bot to respond
# PHILOSOPHY: Only trigger when someone is DIRECTLY addressing the bot themselves
# NOT when they're telling others to summon it or mentioning AI in general
#
# ❌ "Why don't you ask ai?" - telling someone else to use AI
# ❌ "Ask the bot about this" - telling someone else to summon
# ❌ "Someone should summon the bot" - indirect suggestion
# ✅ "Hey bot, what do you think?" - directly addressing the bot
# ✅ "Optimist Prime, help me out" - using the bot's name
# ✅ "I summon the bot" - first-person summoning
SUMMON_PATTERNS = [
    # Direct name mentions (always triggers - they're talking TO the bot)
    r"\boptimist\s*prime\b",
    
    # Greetings DIRECTLY addressing the bot (greeting + bot term = talking TO it)
    r"\b(hey|hi|hello|yo|sup)\s+(optimist\s*prime|bot|mod\s*bot|tldr\s*bot)\b",
    
    # First-person summons only ("I summon", "I'm calling", etc.)
    r"\bI('m| am)?\s*(summon|summoning|calling|paging)\s+(the\s+)?(bot|optimist)\b",
    
    # Mod bot as direct address (specific enough to be intentional)
    r"\bmod\s*bot\b",
]

# Bot identification patterns (to avoid responding to other bots)
BOT_INDICATORS = [
    r"bot\b",
    r"Bot\b", 
    r"auto[\-_]?mod",
    r"AutoModerator",
]

# Optional bot accounts to ignore, supplied by the operator.
BOT_OWNED_USERNAMES = [
    name.strip()
    for name in os.environ.get("BOT_OWNED_USERNAMES", "").split(",")
    if name.strip()
]

# ===== CROSSPOST SETTINGS =====
# Crossposting is off until the operator supplies both communities and their
# own selection guidance.
CROSSPOST_ENABLED = _env_on("BOT_CROSSPOST_ENABLED", False)
CROSSPOST_SOURCE_SUB = os.environ.get("BOT_CROSSPOST_SOURCE_SUB", SUBREDDIT).strip()
CROSSPOST_TARGET_SUB = os.environ.get("BOT_CROSSPOST_TARGET_SUB", "").strip()
CROSSPOST_SELECTION_GUIDANCE = os.environ.get("BOT_CROSSPOST_SELECTION_GUIDANCE", "").strip()
CROSSPOST_SELECTION_THRESHOLD_PERCENT = _env_optional_bounded_int(
    "BOT_CROSSPOST_SELECTION_THRESHOLD_PERCENT", 1, 100
)
CROSSPOST_MAX_PER_DAY = _env_optional_bounded_int("BOT_CROSSPOST_MAX_PER_DAY", 1, 100)
CROSSPOST_MIN_HOURS_OLD = _env_optional_bounded_int("BOT_CROSSPOST_MIN_HOURS_OLD", 0, 8760)
CROSSPOST_MAX_HOURS_OLD = _env_optional_bounded_int("BOT_CROSSPOST_MAX_HOURS_OLD", 1, 8760)
CROSSPOST_TIME_WINDOW_UTC = (
    _env_optional_bounded_int("BOT_CROSSPOST_EARLIEST_HOUR_UTC", 0, 23),
    _env_optional_bounded_int("BOT_CROSSPOST_LATEST_HOUR_UTC", 0, 23),
)
CROSSPOST_LOOKBACK_DAYS = _env_optional_bounded_int("BOT_CROSSPOST_LOOKBACK_DAYS", 1, 3650)

# User-requested flair is limited to exact operator-approved values. No public
# history scoring or model-generated labels are included in the starter setup.
USER_CUSTOM_FLAIR_ENABLED = _env_on("BOT_USER_CUSTOM_FLAIR_ENABLED", False)
USER_CUSTOM_FLAIR_ALLOWED_VALUES = tuple(
    value.strip()
    for value in os.environ.get("BOT_USER_CUSTOM_FLAIR_ALLOWED_VALUES", "").split(",")
    if value.strip()
)
USER_FLAIR_MAX_LEN = 64

# Safe mode is on by default for a fresh public setup.
SAFE_MODE = _env_on("BOT_SAFE_MODE", True)

# Content moderation (LLM-based rule violation detection)
CONTENT_MODERATION_ENABLED = os.environ.get("BOT_CONTENT_MODERATION_ENABLED", "").lower() in ("1", "true", "yes", "on")
AUTOMATED_BAN_ENABLED = _env_on("BOT_AUTOMATED_BAN_ENABLED", False)
CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED = _env_on(
    "BOT_CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED", False
)
# A second AI pass can make modmail easier to read, but the original rule
# evaluation already produced a reason.  Keep this optional enrichment off by
# default so moderation does not consume an avoidable extra call.
MODERATION_AI_SUMMARY_ENABLED = _env_on(
    "BOT_MODERATION_AI_SUMMARY_ENABLED", False
)

# Engagement action mode. Log locally unless the operator explicitly chooses
# public comments, reports, or private moderator mail.
def _parse_engagement_action_mode(raw: str) -> str:
    """Return a supported delivery mode, defaulting safely to local logging."""
    normalized = str(raw or "").strip().lower()
    return normalized if normalized in {"comment", "report", "modmail", "log"} else "log"


ENGAGEMENT_ACTION_MODE = _parse_engagement_action_mode(
    os.environ.get("BOT_ENGAGEMENT_ACTION_MODE", "log")
)

# Discrete rule-based moderation (new system)
MODERATION_RULES_FILE = os.environ.get("BOT_MODERATION_RULES_FILE", "data/rules.json")
MODERATION_RULES_JSON = os.environ.get("BOT_MODERATION_RULES_JSON", "")  # Inline JSON override
MODERATION_RULES_WIKI_PAGE = os.environ.get("BOT_MODERATION_RULES_WIKI_PAGE", "config/moderation_rules")
COMMUNITY_RULES_REFERENCE_FILE = os.environ.get(
    "BOT_COMMUNITY_RULES_REFERENCE_FILE", "data/community_rules.reference.json"
).strip() or "data/community_rules.reference.json"
COMMUNITY_RULES_MAX_AGE_DAYS = max(1, _env_int("BOT_COMMUNITY_RULES_MAX_AGE_DAYS", 30))

# The history sample size controls how much personal data leaves Reddit for
# provider analysis and how far the bot searches each Reddit history listing,
# so require an explicit operator choice before any review.
HISTORY_REVIEW_MAX_ITEMS = _env_optional_bounded_int(
    "BOT_HISTORY_REVIEW_MAX_ITEMS", 1, 250
)
HISTORY_CRITERIA_FILE = (
    os.environ.get("BOT_HISTORY_CRITERIA_FILE", "data/history_criteria.json").strip()
    or "data/history_criteria.json"
)

# Reported-user history review uses only operator-authored criteria.
REPORTED_USER_REVIEW_ENABLED = _env_on("BOT_REPORTED_USER_REVIEW_ENABLED", False)
_history_ban_threshold_raw = os.environ.get("BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", "").strip()
try:
    _history_ban_threshold_value = int(_history_ban_threshold_raw) if _history_ban_threshold_raw else None
    HISTORY_REVIEW_BAN_THRESHOLD_PERCENT = (
        _history_ban_threshold_value
        if _history_ban_threshold_value is not None and 1 <= _history_ban_threshold_value <= 100
        else None
    )
except ValueError:
    HISTORY_REVIEW_BAN_THRESHOLD_PERCENT = None
_history_match_threshold_raw = os.environ.get(
    "BOT_HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT", ""
).strip()
try:
    _history_match_threshold_value = int(_history_match_threshold_raw) if _history_match_threshold_raw else None
    HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT = (
        _history_match_threshold_value
        if _history_match_threshold_value is not None and 1 <= _history_match_threshold_value <= 100
        else None
    )
except ValueError:
    HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT = None
MODMAIL_ASSISTANT_ENABLED = _env_on("BOT_MODMAIL_ASSISTANT_ENABLED", False)
REPORTED_USER_REVIEW_ACTION = os.environ.get("BOT_REPORTED_USER_REVIEW_ACTION", "log").strip().lower() or "log"
if REPORTED_USER_REVIEW_ACTION not in {"modmail", "log"}:
    REPORTED_USER_REVIEW_ACTION = "log"
# History scope is a privacy and moderation-policy choice, so it has no implicit
# public default. History routes stay unavailable until the operator sets it.
HISTORY_REVIEW_LOOKBACK_DAYS = _env_optional_bounded_int(
    "BOT_HISTORY_REVIEW_LOOKBACK_DAYS", 1, 3650
)
REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN = max(0, _env_int("BOT_REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 3))
REPORTED_USER_REVIEW_NOTIFY_ON_NO_MATCH = _env_on("BOT_REPORTED_USER_REVIEW_NOTIFY_ON_NO_MATCH", False)
REPORTED_USER_REVIEW_STATE_RETENTION_DAYS = max(1, _env_int("BOT_REPORTED_USER_REVIEW_STATE_RETENTION_DAYS", 30))

# Report-triggered, explicitly requested, and diagnostic history reviews use
# the same sample cap and a single batched request for configured criteria.
HISTORY_GATE_JEV_MAX_CALLS_PER_RUN = JEV_MAX_CALLS_PER_RUN

def _bounded_moderator_history_caps(
    total_limit: int,
    comments: int,
    posts: int,
) -> tuple[int, int]:
    """Bound each type to the combined limit; the collector enforces the total."""
    total_limit = max(0, int(total_limit))
    comments = max(0, int(comments))
    posts = max(0, int(posts))
    return min(total_limit, comments), min(total_limit, posts)


def moderator_history_report_caps() -> tuple[int, int]:
    """Return report item caps while enforcing the combined maximum."""
    configured_limit = HISTORY_REVIEW_MAX_ITEMS or 0
    return _bounded_moderator_history_caps(
        configured_limit,
        configured_limit,
        configured_limit,
    )


MODERATOR_HISTORY_REPORT_BATCH_SIZE = min(
    75, max(10, _env_int("BOT_MODERATOR_HISTORY_REPORT_BATCH_SIZE", 50))
)

# History reviews inspect the target community by default. Operators may add
# other communities explicitly; an empty value never means all of Reddit.
HISTORY_SUBREDDITS = tuple(
    value.strip().casefold().removeprefix("r/").rstrip("/")
    for value in os.environ.get("BOT_HISTORY_SUBREDDITS", "").split(",")
    if value.strip()
)
# Adaptive scheduling is opt-in per environment. The operator can enable the
# persistent worker's combined post/comment/private-chat workload timing.
ADAPTIVE_SCHEDULING_ENABLED = _env_on("BOT_ADAPTIVE_SCHEDULING_ENABLED", False)
ADAPTIVE_TARGET_ITEMS_PER_RUN = max(
    1,
    _env_int(
        "BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN",
        _env_int("BOT_ADAPTIVE_TARGET_COMMENTS_PER_RUN", 3),
    ),
)
ADAPTIVE_TARGET_COMMENTS_PER_RUN = ADAPTIVE_TARGET_ITEMS_PER_RUN  # legacy import alias
ADAPTIVE_MIN_INTERVAL_SECONDS = min(
    900,
    max(2, _env_int("BOT_ADAPTIVE_MIN_INTERVAL_SECONDS", 2)),
)
ADAPTIVE_NORMAL_INTERVAL_SECONDS = max(
    ADAPTIVE_MIN_INTERVAL_SECONDS,
    min(900, _env_int("BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS", 60)),
)
ADAPTIVE_MAX_INTERVAL_SECONDS = max(
    ADAPTIVE_NORMAL_INTERVAL_SECONDS,
    min(900, _env_int("BOT_ADAPTIVE_MAX_INTERVAL_SECONDS", 60)),
)

# Select the provider and model in the local environment file.
LLM_PROVIDER = os.environ.get("BOT_LLM_PROVIDER", "").strip().lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
LLM_THINKING_ENABLED = _env_on("LLM_THINKING_ENABLED", False)
LLM_REASONING_EFFORT = (
    os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()
)

# Account identity guard. It may be blank during Safe Mode setup and testing;
# set it to the expected bot account before enabling live Reddit writes.
EXPECTED_BOT_USERNAME = os.environ.get("BOT_EXPECTED_USERNAME", "").strip()
