"""Generic settings exposed by the local Optimist Prime setup UI."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_client import PROVIDER_PRESETS


# Core target and safety controls remain visible when the operator customizes
# which optional fields appear in the setup screen.
NON_HIDEABLE_KEYS = frozenset(
    {
        "BOT_SAFE_MODE",
        "BOT_CONTENT_MODERATION_ENABLED",
        "BOT_AUTOMATED_BAN_ENABLED",
        "BOT_SUBREDDIT",
        "BOT_EXPECTED_USERNAME",
        "BOT_HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT",
        "BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT",
        "BOT_HISTORY_REVIEW_LOOKBACK_DAYS",
        "BOT_HISTORY_REVIEW_MAX_ITEMS",
        "BOT_CROSSPOST_SELECTION_THRESHOLD_PERCENT",
        "BOT_CROSSPOST_SOURCE_SUB",
        "BOT_CROSSPOST_TARGET_SUB",
        "BOT_CROSSPOST_SELECTION_GUIDANCE",
        "BOT_CROSSPOST_MAX_PER_DAY",
        "BOT_CROSSPOST_MIN_HOURS_OLD",
        "BOT_CROSSPOST_MAX_HOURS_OLD",
        "BOT_CROSSPOST_EARLIEST_HOUR_UTC",
        "BOT_CROSSPOST_LATEST_HOUR_UTC",
        "BOT_CROSSPOST_LOOKBACK_DAYS",
    }
)

# Provider credentials and less common tuning stay behind Show advanced.
ADVANCED_KEYS = frozenset(
    {
        "LLM_MODEL",
        "LLM_THINKING_ENABLED",
        "LLM_REASONING_EFFORT",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GLM_API_KEY",
        "ZHIPU_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "XAI_API_KEY",
        "LLM_API_KEY",
        "REDDIT_DEVELOPER_USERNAME",
        "REDDIT_OAUTH_SCOPES",
        "REDDIT_APP_NAME",
        "REDDIT_API_TIMEOUT_SECONDS",
        "REDDIT_REQUEST_BUDGET_PER_MINUTE",
        "BOT_HISTORY_CRITERIA_FILE",
        "BOT_HISTORY_SUBREDDITS",
        "BOT_MAX_LLM_CALLS_PER_RUN",
        "BOT_MAX_TLDR_PER_RUN",
        "BOT_MAX_TLDR_PER_DAY",
        "BOT_MAX_REPLIES_PER_RUN",
        "BOT_MAX_REPLIES_PER_DAY",
        "BOT_SAME_USER_COOLDOWN_HOURS",
        "BOT_SAME_USER_REPLIES_BEFORE_COOLDOWN",
        "BOT_MODERATOR_REPLIES_BYPASS_LIMITS",
        "BOT_POST_SCAN_LIMIT",
        "BOT_COMMENT_SCAN_LIMIT",
        "BOT_MAX_AGE_HOURS",
        "BOT_POST_WORD_THRESHOLD",
        "BOT_MAX_MODERATION_LLM_CALLS_PER_RUN",
        "BOT_MODERATION_RULES_WIKI_PAGE",
        "BOT_COMMENT_TLDR_PARENT_CONTEXT_ENABLED",
        "BOT_POST_TLDR_PIN",
        "BOT_COMMENT_SUMMARY_PIN",
        "BOT_MODERATION_AI_SUMMARY_ENABLED",
        "BOT_HISTORY_REVIEW_LOOKBACK_DAYS",
        "BOT_HISTORY_REVIEW_MAX_ITEMS",
        "BOT_REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN",
        "BOT_REPORTED_USER_REVIEW_STATE_RETENTION_DAYS",
        "BOT_MODERATOR_HISTORY_REPORT_BATCH_SIZE",
        "BOT_COMMUNITY_RULES_REFERENCE_FILE",
        "BOT_COMMUNITY_RULES_MAX_AGE_DAYS",
        "BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN",
        "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS",
        "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS",
        "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS",
        "BOT_CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED",
        "BOT_RUNTIME_STATE_FILE",
        "BOT_RUNTIME_STATS_FILE",
        "BOT_RUNTIME_AUDIT_LOG_FILE",
        "BOT_RUNTIME_REQUEST_BUDGET_FILE",
        "BOT_RUNTIME_ACTIVITY_FILE",
        "BOT_RUNTIME_CYCLE_SUMMARY_FILE",
        "BOT_LOG_DIR",
        "BOT_CONSOLE_LOG_FILE",
    }
)


@dataclass
class SettingField:
    key: str
    label: str
    section: str
    field_type: str  # bool, int, float, str, text, password, choice, percent
    default: str = ""
    choices: list[str] = field(default_factory=list)
    help_text: str = ""
    hideable: bool = True
    advanced: bool = False
    minimum: int | None = None
    maximum: int | None = None
    step: int | None = None


def _provider_choices() -> list[str]:
    return [""] + [name for name in PROVIDER_PRESETS if name != "custom"] + ["custom"]


def _field(
    key: str,
    label: str,
    section: str,
    field_type: str,
    default: str = "",
    choices: list[str] | None = None,
    help_text: str = "",
    minimum: int | None = None,
    maximum: int | None = None,
    step: int | None = None,
) -> SettingField:
    return SettingField(
        key=key,
        label=label,
        section=section,
        field_type=field_type,
        default=default,
        choices=choices or [],
        help_text=help_text,
        hideable=key not in NON_HIDEABLE_KEYS,
        advanced=key in ADVANCED_KEYS,
        minimum=minimum,
        maximum=maximum,
        step=step,
    )


SETTING_FIELDS: list[SettingField] = [
    # --- AI providers ---
    _field(
        "BOT_LLM_PROVIDER",
        "Text-generation provider",
        "AI providers",
        "choice",
        "",
        _provider_choices(),
        help_text="Choose a service before enabling AI features. Structured decisions use the separately configured OpenRouter key.",
    ),
    _field("OPENROUTER_API_KEY", "OpenRouter API key", "AI providers", "password"),
    _field(
        "LLM_MODEL",
        "Text-generation model",
        "AI providers",
        "str",
        "",
        help_text="Enter a model currently supported by the selected provider.",
    ),
    _field(
        "OPENAI_BASE_URL",
        "Custom OpenAI-compatible API URL",
        "AI providers",
        "str",
        help_text="Used with the custom provider, or to override a compatible provider endpoint.",
    ),
    _field("OPENAI_API_KEY", "OpenAI API key", "AI providers", "password"),
    _field("ANTHROPIC_API_KEY", "Anthropic API key", "AI providers", "password"),
    _field("GEMINI_API_KEY", "Google Gemini API key", "AI providers", "password"),
    _field("DEEPSEEK_API_KEY", "DeepSeek API key", "AI providers", "password"),
    _field("GLM_API_KEY", "GLM API key", "AI providers", "password"),
    _field("ZHIPU_API_KEY", "Zhipu API key", "AI providers", "password"),
    _field("GROQ_API_KEY", "Groq API key", "AI providers", "password"),
    _field("MISTRAL_API_KEY", "Mistral API key", "AI providers", "password"),
    _field("TOGETHER_API_KEY", "Together AI API key", "AI providers", "password"),
    _field("XAI_API_KEY", "xAI API key", "AI providers", "password"),
    _field("LLM_API_KEY", "Generic provider API key", "AI providers", "password"),
    _field(
        "LLM_THINKING_ENABLED",
        "Request provider reasoning mode",
        "AI providers",
        "bool",
        "false",
    ),
    _field(
        "LLM_REASONING_EFFORT",
        "Reasoning effort",
        "AI providers",
        "choice",
        "low",
        ["low", "high", "max"],
    ),
    # --- Reddit account and target ---
    _field("REDDIT_CLIENT_ID", "Reddit client ID", "Reddit", "str"),
    _field("REDDIT_CLIENT_SECRET", "Reddit client secret", "Reddit", "password"),
    _field("REDDIT_REFRESH_TOKEN", "Reddit refresh token", "Reddit", "password"),
    _field(
        "REDDIT_OAUTH_SCOPES",
        "Requested Reddit permissions",
        "Reddit",
        "str",
        "identity read",
        help_text="Starts with account identity and read access. Add only scopes needed by features you choose to enable.",
    ),
    _field(
        "BOT_EXPECTED_USERNAME",
        "Expected bot account (required for live actions)",
        "Reddit",
        "str",
        help_text="Set this to the Reddit account name for the refresh token. Live runs stop if it is blank or Reddit authenticates as another account.",
    ),
    _field(
        "REDDIT_DEVELOPER_USERNAME",
        "Developer name in Reddit User-Agent",
        "Reddit",
        "str",
        help_text="Optional name required by Reddit's API identification format; this is not an account credential.",
    ),
    _field(
        "REDDIT_APP_NAME",
        "Application name in Reddit User-Agent",
        "Reddit",
        "str",
        "OptimistPrimeModBot",
    ),
    _field("BOT_SUBREDDIT", "Target subreddit", "Reddit", "str"),
    _field(
        "REDDIT_API_TIMEOUT_SECONDS",
        "Reddit request timeout (seconds)",
        "Reddit",
        "int",
        "15",
        help_text="Timeout for each Reddit HTTP attempt; the application bounds this value to 5–60 seconds.",
        minimum=5,
        maximum=60,
    ),
    # --- Safety and moderation policy ---
    _field("BOT_SAFE_MODE", "Safe Mode (prevent Reddit writes)", "Safety", "bool", "true"),
    _field(
        "BOT_CONTENT_MODERATION_ENABLED",
        "Enable operator-defined content checks",
        "Safety",
        "bool",
        "false",
    ),
    _field(
        "BOT_AUTOMATED_BAN_ENABLED",
        "Allow automated bans",
        "Safety",
        "bool",
        "false",
        help_text="A global opt-in. Safe Mode must also be off, and a moderation rule or history criterion must authorize the action.",
    ),
    _field(
        "BOT_HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT",
        "History-review match threshold (%)",
        "History review",
        "int",
        "",
        help_text="Required to run history scoring. It labels which operator-defined criteria matched; it does not authorize a ban.",
        minimum=1,
        maximum=100,
    ),
    _field(
        "BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT",
        "Optional history-review ban threshold (%)",
        "Safety",
        "int",
        "",
        help_text="Leave blank to disable history-based bans. A threshold is only used when automated bans, a user-authored criterion, and an eligible review route are also enabled. Model scores are not calibrated probabilities.",
        minimum=1,
        maximum=100,
    ),
    _field(
        "BOT_HISTORY_REVIEW_LOOKBACK_DAYS",
        "History-review lookback (days)",
        "History review",
        "int",
        "",
        help_text="Required when history review is configured. Choose the oldest public history the bot may inspect.",
        minimum=1,
        maximum=3650,
    ),
    _field(
        "BOT_CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED",
        "Post a public explanation after a configured removal",
        "Safety",
        "bool",
        "false",
    ),
    _field(
        "BOT_MODERATION_RULES_FILE",
        "Local content-rules file",
        "Moderation rules",
        "str",
        "data/rules.json",
    ),
    _field(
        "BOT_MODERATION_RULES_WIKI_PAGE",
        "Optional Reddit wiki page for rules",
        "Moderation rules",
        "str",
        "config/moderation_rules",
    ),
    _field(
        "BOT_COMMUNITY_RULES_REFERENCE_FILE",
        "Community-rules reference file",
        "Moderation rules",
        "str",
        "data/community_rules.reference.json",
    ),
    _field(
        "BOT_COMMUNITY_RULES_MAX_AGE_DAYS",
        "Maximum age of cached rules (days)",
        "Moderation rules",
        "int",
        "30",
        minimum=1,
    ),
    _field(
        "BOT_REPORTED_USER_REVIEW_ENABLED",
        "Review reported users against local criteria",
        "History review",
        "bool",
        "false",
        help_text="Requires criteria that you write in the local history-criteria file. No criteria are bundled or enabled by default.",
    ),
    _field(
        "BOT_HISTORY_CRITERIA_FILE",
        "Local history-criteria file",
        "History review",
        "str",
        "data/history_criteria.json",
        help_text="Write your own review criteria in this ignored local file. The public example starts empty.",
    ),
    _field(
        "BOT_HISTORY_SUBREDDITS",
        "Additional communities to include in history reviews",
        "History review",
        "str",
        "",
        help_text="Comma-separated names. Blank means the target subreddit only.",
    ),
    _field(
        "BOT_REPORTED_USER_REVIEW_ACTION",
        "Where reported-user review results go",
        "History review",
        "choice",
        "log",
        ["log", "modmail"],
    ),
    _field(
        "BOT_REPORTED_USER_REVIEW_NOTIFY_ON_NO_MATCH",
        "Notify moderators when no criterion matches",
        "History review",
        "bool",
        "false",
    ),
    _field(
        "BOT_HISTORY_REVIEW_MAX_ITEMS",
        "Maximum history items sent for review",
        "History review",
        "int",
        "",
        help_text="Required before history reviews can run. Choose how many public history items may be sent to the AI provider, from 1 to 250.",
        minimum=1,
        maximum=250,
    ),
    _field(
        "BOT_REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN",
        "History-review provider calls per run",
        "History review",
        "int",
        "3",
        minimum=0,
    ),
    _field(
        "BOT_REPORTED_USER_REVIEW_STATE_RETENTION_DAYS",
        "History-review record retention (days)",
        "History review",
        "int",
        "30",
        minimum=1,
    ),
    _field(
        "BOT_MODERATOR_HISTORY_REPORT_BATCH_SIZE",
        "History report batch size",
        "History review",
        "int",
        "50",
        minimum=10,
        maximum=75,
    ),
    # --- Summaries and conversations ---
    _field("BOT_POST_TLDR_ENABLED", "Summarize eligible posts", "Features", "bool", "false"),
    _field("BOT_COMMENT_TLDR_ENABLED", "Summarize eligible comments", "Features", "bool", "false"),
    _field("BOT_COMMENT_SUMMARY_ENABLED", "Summarize discussions", "Features", "bool", "false"),
    _field("BOT_REDDIT_REFERENCE_TLDR_ENABLED", "Summarize linked Reddit discussions", "Features", "bool", "false"),
    _field("BOT_POST_TLDR_PIN", "Pin post summaries", "Features", "bool", "false"),
    _field("BOT_COMMENT_SUMMARY_PIN", "Pin discussion summaries", "Features", "bool", "false"),
    _field("BOT_COMMENT_TLDR_PARENT_CONTEXT_ENABLED", "Include parent context in comment summaries", "Features", "bool", "false"),
    _field("BOT_CONVERSATIONAL_REPLIES_ENABLED", "Enable contextual replies", "Features", "bool", "false"),
    _field("BOT_INBOX_REPLIES_ENABLED", "Enable private inbox replies", "Features", "bool", "false"),
    _field("BOT_SUMMONS_ENABLED", "Respond to direct summons", "Features", "bool", "false"),
    _field(
        "BOT_ENGAGEMENT_ACTION_MODE",
        "How generated conversational responses are delivered",
        "Features",
        "choice",
        "log",
        ["comment", "report", "modmail", "log"],
    ),
    _field("BOT_MAX_REPLIES_PER_RUN", "Maximum public replies per run", "Features", "int", "2", minimum=0),
    _field("BOT_MAX_REPLIES_PER_DAY", "Maximum public replies per day", "Features", "int", "10", minimum=0),
    _field("BOT_SAME_USER_REPLIES_BEFORE_COOLDOWN", "Replies before user cooldown", "Features", "int", "5", minimum=1),
    _field("BOT_SAME_USER_COOLDOWN_HOURS", "User cooldown (hours)", "Features", "int", "1", minimum=1),
    _field("BOT_MODERATOR_REPLIES_BYPASS_LIMITS", "Let moderator replies bypass ordinary reply caps", "Features", "bool", "false"),
    _field(
        "BOT_MODMAIL_ASSISTANT_ENABLED",
        "Enable private moderator-assistant replies",
        "Features",
        "bool",
        "false",
    ),
    _field(
        "BOT_MODERATION_AI_SUMMARY_ENABLED",
        "Add AI summaries to moderator reports",
        "Features",
        "bool",
        "false",
    ),
    # --- Optional community tools ---
    _field(
        "BOT_USER_CUSTOM_FLAIR_ENABLED",
        "Enable self-service flair requests",
        "Community tools",
        "bool",
        "false",
        help_text="Requests are limited to exact values in the operator allowlist below.",
    ),
    _field(
        "BOT_USER_CUSTOM_FLAIR_ALLOWED_VALUES",
        "Allowed self-service flair values",
        "Community tools",
        "str",
        "",
        help_text="Comma-separated exact flair text. Leave blank to accept no requests.",
    ),
    _field(
        "BOT_CROSSPOST_ENABLED",
        "Enable scheduled crossposting",
        "Community tools",
        "bool",
        "false",
        help_text="Requires communities and selection guidance supplied by the operator.",
    ),
    _field("BOT_CROSSPOST_SOURCE_SUB", "Crosspost source community", "Community tools", "str"),
    _field("BOT_CROSSPOST_TARGET_SUB", "Crosspost target community", "Community tools", "str"),
    _field(
        "BOT_CROSSPOST_SELECTION_GUIDANCE",
        "Your crosspost selection guidance",
        "Community tools",
        "str",
        help_text="Describe what is suitable to crosspost for your communities. No selection policy is built in.",
    ),
    _field(
        "BOT_CROSSPOST_SELECTION_THRESHOLD_PERCENT",
        "Crosspost selection cutoff (%)",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Choose the minimum model score that counts as a match to your guidance.",
        minimum=1,
        maximum=100,
    ),
    _field(
        "BOT_CROSSPOST_MAX_PER_DAY",
        "Maximum crossposts per day",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Choose a daily posting limit from 1 to 100.",
        minimum=1,
        maximum=100,
    ),
    _field(
        "BOT_CROSSPOST_MIN_HOURS_OLD",
        "Minimum source-post age (hours)",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Choose the youngest eligible source post, from 0 to 8,760 hours.",
        minimum=0,
        maximum=8760,
    ),
    _field(
        "BOT_CROSSPOST_MAX_HOURS_OLD",
        "Maximum source-post age (hours)",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Choose the oldest eligible source post, from 1 to 8,760 hours.",
        minimum=1,
        maximum=8760,
    ),
    _field(
        "BOT_CROSSPOST_EARLIEST_HOUR_UTC",
        "Earliest crosspost hour (UTC)",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Choose an hour from 0 to 23 UTC.",
        minimum=0,
        maximum=23,
    ),
    _field(
        "BOT_CROSSPOST_LATEST_HOUR_UTC",
        "Latest crosspost hour (UTC)",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Must be the same as or later than the earliest hour.",
        minimum=0,
        maximum=23,
    ),
    _field(
        "BOT_CROSSPOST_LOOKBACK_DAYS",
        "Duplicate search window (days)",
        "Community tools",
        "int",
        "",
        help_text="Required when crossposting is enabled. Choose how far back to search the target community for duplicates.",
        minimum=1,
        maximum=3650,
    ),
    # --- Runtime limits and local files ---
    _field("BOT_MAX_LLM_CALLS_PER_RUN", "Maximum text-provider calls per run", "Runtime limits", "int", minimum=0),
    _field("BOT_JEV_MAX_CALLS_PER_RUN", "Maximum structured-decision calls per run", "Runtime limits", "int", "30", minimum=0),
    _field("BOT_MAX_TLDR_PER_RUN", "Maximum summaries per run", "Runtime limits", "int", "1", minimum=0),
    _field("BOT_MAX_TLDR_PER_DAY", "Maximum summaries per day", "Runtime limits", "int", "10", minimum=0),
    _field("BOT_POST_SCAN_LIMIT", "Posts scanned per run", "Runtime limits", "int", "100", minimum=1),
    _field("BOT_COMMENT_SCAN_LIMIT", "Comments scanned per run", "Runtime limits", "int", "100", minimum=1),
    _field("BOT_MAX_AGE_HOURS", "Work age limit (hours)", "Runtime limits", "int", "24", minimum=1),
    _field("BOT_POST_WORD_THRESHOLD", "Minimum post length for summary (words)", "Runtime limits", "int", "250", minimum=0),
    _field("BOT_MAX_MODERATION_LLM_CALLS_PER_RUN", "Maximum moderation evidence calls per run", "Runtime limits", "int", minimum=0),
    _field("BOT_ADAPTIVE_SCHEDULING_ENABLED", "Adjust worker interval to workload", "Runtime limits", "bool", "false"),
    _field("BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN", "Work items targeted per run", "Runtime limits", "int", "3", minimum=1),
    _field("BOT_ADAPTIVE_MIN_INTERVAL_SECONDS", "Shortest worker interval (seconds)", "Runtime limits", "int", "2", minimum=2, maximum=900),
    _field("BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS", "Normal worker interval (seconds)", "Runtime limits", "int", "60", minimum=2, maximum=900),
    _field("BOT_ADAPTIVE_MAX_INTERVAL_SECONDS", "Longest worker interval (seconds)", "Runtime limits", "int", "60", minimum=2, maximum=900),
    _field("REDDIT_REQUEST_BUDGET_PER_MINUTE", "Reddit request budget per minute", "Runtime limits", "int", "90", minimum=1, maximum=90),
    _field("BOT_RUNTIME_STATE_FILE", "Runtime state file", "Local files", "str", "data/bot_state.json"),
    _field("BOT_RUNTIME_STATS_FILE", "Aggregate statistics file", "Local files", "str", "data/stats.json"),
    _field("BOT_RUNTIME_AUDIT_LOG_FILE", "Moderation audit-log file", "Local files", "str", "data/audit_log.json"),
    _field("BOT_RUNTIME_REQUEST_BUDGET_FILE", "Reddit request-budget file", "Local files", "str"),
    _field("BOT_RUNTIME_ACTIVITY_FILE", "Captured activity snapshot", "Local files", "str", "data/activity.json"),
    _field("BOT_RUNTIME_CYCLE_SUMMARY_FILE", "Latest worker-cycle summary", "Local files", "str", "data/cycle_summary.json"),
    _field("BOT_LOG_DIR", "Log folder", "Local files", "str", "logs"),
    _field("BOT_CONSOLE_LOG_FILE", "Worker console log", "Local files", "str"),
]


def sections() -> list[str]:
    """Return each settings section in its display order."""
    seen: list[str] = []
    for setting in SETTING_FIELDS:
        if setting.section not in seen:
            seen.append(setting.section)
    return seen


def fields_by_section() -> dict[str, list[SettingField]]:
    """Group registered fields while preserving their display order."""
    grouped: dict[str, list[SettingField]] = {}
    for setting in SETTING_FIELDS:
        grouped.setdefault(setting.section, []).append(setting)
    return grouped
