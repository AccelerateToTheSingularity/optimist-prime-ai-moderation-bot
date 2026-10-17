"""Environment-driven settings for bounded local runs.

This module applies runtime limits and feature switches. Community policy
remains in operator-owned local configuration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import config

# Store base patterns so apply_runtime_settings can reset before appending.
_BASE_SUMMON_PATTERNS: list[str] | None = None


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


@dataclass
class RuntimeSettings:
    subreddit: str = config.SUBREDDIT
    post_tldr_enabled: bool = config.POST_TLDR_ENABLED
    comment_tldr_enabled: bool = config.COMMENT_TLDR_ENABLED
    comment_summary_enabled: bool = config.COMMENT_SUMMARY_ENABLED
    crosspost_enabled: bool = config.CROSSPOST_ENABLED
    inbox_replies_enabled: bool = False
    summons_enabled: bool = False
    max_llm_calls_per_run: int | None = None
    max_tldr_per_run: int = config.MAX_TLDR_PER_RUN
    max_replies_per_run: int = config.MAX_REPLIES_PER_RUN
    max_tldr_per_day: int = config.MAX_TLDR_PER_DAY
    max_replies_per_day: int = config.MAX_REPLIES_PER_DAY
    post_scan_limit: int | None = None
    comment_scan_limit: int | None = None
    max_age_hours: int = config.MAX_AGE_HOURS
    post_word_threshold: int = config.POST_WORD_THRESHOLD
    safe_mode: bool = config.SAFE_MODE
    extra_summon_patterns: list[str] = field(default_factory=list)

    def describe(self) -> str:
        llm_cap = (
            str(self.max_llm_calls_per_run)
            if self.max_llm_calls_per_run is not None
            else "unlimited"
        )
        subreddit = f"r/{self.subreddit}" if self.subreddit else "unset"
        return ", ".join(
            (
                f"subreddit={subreddit}",
                f"max_llm_calls={llm_cap}",
                f"post_tldr={self.post_tldr_enabled}",
                f"summons={self.summons_enabled}",
                f"inbox_replies={self.inbox_replies_enabled}",
                f"crosspost={self.crosspost_enabled}",
                f"scan={self.post_scan_limit or config.POST_SCAN_LIMIT} posts/"
                f"{self.comment_scan_limit or config.COMMENT_SCAN_LIMIT} comments",
                f"age={self.max_age_hours}h",
            )
        )


def resolve_runtime_settings(bot_username: str | None = None) -> RuntimeSettings:
    subreddit = _env_str("BOT_SUBREDDIT", config.SUBREDDIT)

    settings = RuntimeSettings(
        subreddit=subreddit,
        post_tldr_enabled=config.POST_TLDR_ENABLED,
        comment_tldr_enabled=config.COMMENT_TLDR_ENABLED,
        comment_summary_enabled=config.COMMENT_SUMMARY_ENABLED,
        crosspost_enabled=config.CROSSPOST_ENABLED,
        inbox_replies_enabled=False,
        summons_enabled=False,
        max_tldr_per_run=config.MAX_TLDR_PER_RUN,
        max_replies_per_run=config.MAX_REPLIES_PER_RUN,
        max_tldr_per_day=config.MAX_TLDR_PER_DAY,
        max_replies_per_day=config.MAX_REPLIES_PER_DAY,
        post_scan_limit=config.POST_SCAN_LIMIT,
        comment_scan_limit=config.COMMENT_SCAN_LIMIT,
        max_age_hours=config.MAX_AGE_HOURS,
        post_word_threshold=config.POST_WORD_THRESHOLD,
    )

    # Per-feature environment overrides take precedence over an optional preset.
    for attr, env_name in (
        ("post_tldr_enabled", "BOT_POST_TLDR_ENABLED"),
        ("comment_tldr_enabled", "BOT_COMMENT_TLDR_ENABLED"),
        ("comment_summary_enabled", "BOT_COMMENT_SUMMARY_ENABLED"),
        ("crosspost_enabled", "BOT_CROSSPOST_ENABLED"),
        ("inbox_replies_enabled", "BOT_INBOX_REPLIES_ENABLED"),
        ("summons_enabled", "BOT_SUMMONS_ENABLED"),
    ):
        value = _env_bool(env_name)
        if value is not None:
            setattr(settings, attr, value)

    if (max_llm := _env_int("BOT_MAX_LLM_CALLS_PER_RUN")) is not None:
        settings.max_llm_calls_per_run = max(0, max_llm)
    if (scan_limit := _env_int("BOT_POST_SCAN_LIMIT")) is not None:
        settings.post_scan_limit = max(1, scan_limit)
    if (comment_scan_limit := _env_int("BOT_COMMENT_SCAN_LIMIT")) is not None:
        settings.comment_scan_limit = max(1, comment_scan_limit)
    if (max_age_hours := _env_int("BOT_MAX_AGE_HOURS")) is not None:
        settings.max_age_hours = max(1, max_age_hours)
    if (word_threshold := _env_int("BOT_POST_WORD_THRESHOLD")) is not None:
        settings.post_word_threshold = max(0, word_threshold)
    if (max_tldr_run := _env_int("BOT_MAX_TLDR_PER_RUN")) is not None:
        settings.max_tldr_per_run = max(0, max_tldr_run)
    if (max_tldr_day := _env_int("BOT_MAX_TLDR_PER_DAY")) is not None:
        settings.max_tldr_per_day = max(0, max_tldr_day)
    if (max_replies_run := _env_int("BOT_MAX_REPLIES_PER_RUN")) is not None:
        settings.max_replies_per_run = max(0, max_replies_run)
    if (max_replies_day := _env_int("BOT_MAX_REPLIES_PER_DAY")) is not None:
        settings.max_replies_per_day = max(0, max_replies_day)

    safe_mode = _env_bool("BOT_SAFE_MODE")
    if safe_mode is not None:
        settings.safe_mode = safe_mode

    if bot_username:
        safe_name = re.escape(bot_username)
        settings.extra_summon_patterns.extend(
            (rf"\bu/{safe_name}\b", rf"\b{safe_name}\b")
        )

    return settings


def apply_runtime_settings(settings: RuntimeSettings) -> None:
    """Apply resolved generic runtime settings to the shared config module."""
    config.SUBREDDIT = settings.subreddit
    config.POST_TLDR_ENABLED = settings.post_tldr_enabled
    config.COMMENT_TLDR_ENABLED = settings.comment_tldr_enabled
    config.COMMENT_SUMMARY_ENABLED = settings.comment_summary_enabled
    config.CROSSPOST_ENABLED = settings.crosspost_enabled
    config.MAX_TLDR_PER_RUN = settings.max_tldr_per_run
    config.MAX_REPLIES_PER_RUN = settings.max_replies_per_run
    config.MAX_TLDR_PER_DAY = settings.max_tldr_per_day
    config.MAX_REPLIES_PER_DAY = settings.max_replies_per_day
    config.POST_SCAN_LIMIT = max(1, settings.post_scan_limit or config.POST_SCAN_LIMIT)
    config.COMMENT_SCAN_LIMIT = max(
        1, settings.comment_scan_limit or config.COMMENT_SCAN_LIMIT
    )
    config.MAX_AGE_HOURS = max(1, settings.max_age_hours)
    config.POST_WORD_THRESHOLD = max(0, settings.post_word_threshold)
    config.SAFE_MODE = settings.safe_mode

    if settings.extra_summon_patterns:
        global _BASE_SUMMON_PATTERNS
        if _BASE_SUMMON_PATTERNS is None:
            _BASE_SUMMON_PATTERNS = list(config.SUMMON_PATTERNS)
        config.SUMMON_PATTERNS = _BASE_SUMMON_PATTERNS + settings.extra_summon_patterns
