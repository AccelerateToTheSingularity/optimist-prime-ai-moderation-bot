"""Observable rotating JSONL logging for local and CI bot runs.

Implements Application Standard 14 (observable logging).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
APP_LOGGER_NAME = "optimist_prime"

_RUN_ID: ContextVar[str | None] = ContextVar("optimist_prime_run_id", default=None)
_PHASE: ContextVar[str | None] = ContextVar("optimist_prime_phase", default=None)
_REQUEST_ID: ContextVar[str | None] = ContextVar("optimist_prime_request_id", default=None)
_REQUEST_ATTEMPT: ContextVar[int | None] = ContextVar(
    "optimist_prime_request_attempt", default=None
)
_REQUEST_TIMEOUT: ContextVar[float | int | None] = ContextVar(
    "optimist_prime_request_timeout", default=None
)
_RETRY_PENDING: ContextVar[bool] = ContextVar(
    "optimist_prime_retry_pending", default=False
)


def new_run_id() -> str:
    """Return a short identifier that links one worker cycle's log records."""
    return uuid.uuid4().hex[:12]


def ensure_run_id(value: str | None = None) -> str:
    """Set or reuse the current cycle identifier without exposing credentials."""
    current = _RUN_ID.get()
    run_id = (value or current or new_run_id()).strip()
    if not run_id:
        run_id = new_run_id()
    _RUN_ID.set(run_id)
    return run_id


def current_run_id() -> str | None:
    return _RUN_ID.get()


def set_log_phase(phase: str | None) -> None:
    """Set the human-readable operation phase attached to later log records."""
    value = str(phase or "").strip()
    _PHASE.set(value or None)


def current_log_phase() -> str | None:
    return _PHASE.get()


def set_request_observation(
    request_id: str | None,
    attempt: int | None,
    timeout_seconds: float | int | None = None,
) -> None:
    """Expose the current low-level request to the PRAW retry log filter."""
    _REQUEST_ID.set(request_id)
    _REQUEST_ATTEMPT.set(attempt)
    _REQUEST_TIMEOUT.set(timeout_seconds)
    _RETRY_PENDING.set(False)


def current_request_observation() -> tuple[str | None, int | None, float | int | None]:
    return _REQUEST_ID.get(), _REQUEST_ATTEMPT.get(), _REQUEST_TIMEOUT.get()


def retry_pending() -> bool:
    """Return whether PRAW has just announced another attempt for this request."""
    return _RETRY_PENDING.get()


def mark_retry_pending() -> None:
    """Keep the current request identity across PRAW's recursive retry."""
    _RETRY_PENDING.set(True)


def clear_log_context() -> None:
    """Reset per-process context so isolated tests cannot affect one another."""
    _RUN_ID.set(None)
    _PHASE.set(None)
    _REQUEST_ID.set(None)
    _REQUEST_ATTEMPT.set(None)
    _REQUEST_TIMEOUT.set(None)
    _RETRY_PENDING.set(False)


def _safe_request_path(value: str) -> str:
    """Keep retry logs useful without retaining query strings or credentials."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[unparseable]"
    path = parsed.path or "/"
    return path[:500]


class PrawRetryLogFilter(logging.Filter):
    """Turn PRAW's terse retry warning into a safe structured event.

    The custom requestor below records every actual attempt.  This filter also
    normalizes PRAW's own retry message so an operator sees the retry decision
    without a raw exception representation or query string in the JSON log.
    """

    _RETRY_RE = re.compile(
        r"^Retrying due to (?P<reason>.+?) status: (?P<method>[A-Za-z]+) (?P<url>\S+)$"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "prawcore" or record.levelno < logging.WARNING:
            return True
        message = record.getMessage()
        match = self._RETRY_RE.match(message)
        if not match:
            return True

        reason = match.group("reason").strip()
        reason_type = reason.split("(", 1)[0].strip() or "unknown"
        request_id, attempt, timeout_seconds = current_request_observation()
        record.msg = "Reddit request retry scheduled"
        record.args = ()
        record.event_name = "reddit.retry"
        record.component = "reddit"
        record.correlation_id = record.__dict__.get("correlation_id") or current_run_id()
        record.phase = record.__dict__.get("phase") or current_log_phase()
        record.request_id = request_id
        record.attempt = attempt
        record.request_method = match.group("method").upper()
        record.request_path = _safe_request_path(match.group("url"))
        record.timeout_seconds = timeout_seconds
        record.retry_reason = reason_type[:120]
        record.retry_scheduled = True
        record.retryable = True
        mark_retry_pending()
        return True


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr, key in (
            ("event_name", "eventName"),
            ("correlation_id", "correlationId"),
            ("action_id", "actionId"),
            ("component", "component"),
            ("log_path", "logPath"),
            ("structured_log_path", "structuredLogPath"),
            ("subreddit", "subreddit"),
            ("profile", "profile"),
            ("auth_mode", "authMode"),
            ("phase", "phase"),
            ("request_id", "requestId"),
            ("attempt", "attempt"),
            ("duration_ms", "durationMs"),
            ("timeout_seconds", "timeoutSeconds"),
            ("request_method", "requestMethod"),
            ("request_path", "requestPath"),
            ("status_code", "statusCode"),
            ("retryable", "retryable"),
            ("retry_scheduled", "retryScheduled"),
            ("retry_reason", "retryReason"),
            ("error_type", "errorType"),
            ("provider_error_type", "providerErrorType"),
            ("provider_status_code", "providerStatusCode"),
            ("provider_error_reason", "providerErrorReason"),
            ("cause_error_type", "causeErrorType"),
            ("operation_stage", "operationStage"),
            ("provider", "provider"),
            ("model_name", "modelName"),
            ("choice_count", "choiceCount"),
            ("finish_reason", "finishReason"),
            ("content_length", "contentLength"),
            ("reasoning_content_length", "reasoningContentLength"),
            ("refusal_present", "refusalPresent"),
            ("tool_call_count", "toolCallCount"),
            ("prompt_tokens", "promptTokens"),
            ("completion_tokens", "completionTokens"),
            ("failure_reason", "failureReason"),
            ("item_count", "itemCount"),
            ("resulting_batch_count", "resultingBatchCount"),
            ("message_fingerprint", "messageFingerprint"),
            ("retry_after_seconds", "retryAfterSeconds"),
            ("stack_trace", "stackTrace"),
            ("outcome", "outcome"),
            ("worker", "worker"),
            ("working_directory", "workingDirectory"),
            ("exit_code", "exitCode"),
            ("cycle_number", "cycleNumber"),
            ("interval_seconds", "intervalSeconds"),
            ("next_cycle_in_seconds", "nextCycleInSeconds"),
            ("missed_intervals", "missedIntervals"),
            ("cycles", "cycles"),
            ("request_budget_per_minute", "requestBudgetPerMinute"),
            ("request_budget_window_seconds", "requestBudgetWindowSeconds"),
            ("request_budget_max_requests", "requestBudgetMaxRequests"),
            ("budget_requests_per_minute", "budgetRequestsPerMinute"),
            ("budget_window_seconds", "budgetWindowSeconds"),
            ("budget_max_requests", "budgetMaxRequests"),
            ("budget_used_requests", "budgetUsedRequests"),
            ("budget_remaining_requests", "budgetRemainingRequests"),
            ("budget_reset_seconds", "budgetResetSeconds"),
            ("wait_seconds", "waitSeconds"),
            ("waited_seconds", "waitedSeconds"),
            # Keep aggregate operation metrics separate from user content,
            # usernames, IDs, prompts, and credentials.
            ("tldrs_generated", "tldrsGenerated"),
            ("replies_sent", "repliesSent"),
            ("summons_handled", "summonsHandled"),
            ("crossposts_made", "crosspostsMade"),
            ("users_banned", "usersBanned"),
            ("tokens_used", "tokensUsed"),
            ("cost", "cost"),
            ("catch_up_posts_fetched", "catchUpPostsFetched"),
            ("catch_up_comments_fetched", "catchUpCommentsFetched"),
            ("catch_up_posts_complete", "catchUpPostsComplete"),
            ("catch_up_comments_complete", "catchUpCommentsComplete"),
            ("private_chat_messages_failed", "privateChatMessagesFailed"),
            ("private_chat_messages_deferred", "privateChatMessagesDeferred"),
        ):
            value = getattr(record, attr, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class OperatorLogFormatter(logging.Formatter):
    """Render safe, useful progress lines for the hidden worker's console log."""

    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event_name", ""))
        method = getattr(record, "request_method", None)
        path = getattr(record, "request_path", None)
        attempt = getattr(record, "attempt", None)
        timeout = getattr(record, "timeout_seconds", None)
        duration = getattr(record, "duration_ms", None)
        error_type = getattr(record, "error_type", None)
        status = getattr(record, "status_code", None)

        if method and path and event == "reddit.request.started":
            message = f"Reddit {method} {path} attempt {attempt} started (timeout {timeout}s)"
        elif method and path and event == "reddit.request.retryable_failure":
            detail = f"failed retryably ({error_type or 'transient error'}"
            if duration is not None:
                detail += f", {duration}ms"
            message = f"Reddit {method} {path} attempt {attempt} {detail})"
        elif method and path and event == "reddit.request.failed":
            detail = f"failed ({error_type or 'error'}"
            if duration is not None:
                detail += f", {duration}ms"
            message = f"Reddit {method} {path} attempt {attempt} {detail})"
        elif method and path and event == "reddit.retry":
            message = (
                f"Reddit {method} {path} retry scheduled after attempt {attempt}"
                f" ({getattr(record, 'retry_reason', None) or 'transient failure'})"
            )
        elif method and path and event in {
            "reddit.request.completed",
            "reddit.request.retryable_response",
        }:
            detail = f"status {status}" if status is not None else "response received"
            if duration is not None:
                detail += f", {duration}ms"
            if event == "reddit.request.retryable_response":
                detail += ", retryable"
            message = f"Reddit {method} {path} attempt {attempt} {detail}"
        elif event == "reddit.request_budget.waiting":
            wait = getattr(record, "wait_seconds", None)
            message = "Reddit request safety budget is full; waiting"
            if wait is not None:
                message += f" about {wait}s for a slot"
        elif event == "reddit.request_budget.acquired":
            waited = getattr(record, "waited_seconds", None)
            remaining = getattr(record, "budget_remaining_requests", None)
            message = "Reddit request safety budget slot acquired"
            if waited is not None:
                message += f" after {waited}s"
            if remaining is not None:
                message += f" ({remaining} remaining)"
        else:
            message = record.getMessage()

        run_id = getattr(record, "correlation_id", None)
        phase = getattr(record, "phase", None)
        context = " ".join(
            part
            for part in (
                f"run={run_id}" if run_id else "",
                f"phase={phase}" if phase else "",
            )
            if part
        )
        prefix = f"{record.levelname}"
        if context:
            prefix += f" [{context}]"
        return f"{prefix} {message}"


def log_dir() -> Path:
    raw = os.environ.get("BOT_LOG_DIR", "").strip()
    if not raw:
        return DEFAULT_LOG_DIR
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def current_log_path() -> Path:
    return log_dir() / f"{APP_LOGGER_NAME}.jsonl"


def console_log_path() -> Path:
    """Return the local capture file used by a background worker."""
    raw = os.environ.get("BOT_CONSOLE_LOG_FILE", "").strip()
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw)))
    return log_dir() / f"{APP_LOGGER_NAME}.console.log"


def configure_logging(*, retention_days: int = 14, also_stderr: bool = True) -> Path:
    """Configure app logging; return the active JSONL log path."""
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = current_log_path()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers when reconfigured in tests
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = TimedRotatingFileHandler(
        filename=path,
        when="midnight",
        backupCount=retention_days,
        encoding="utf-8",
        utc=True,
    )
    retry_filter = PrawRetryLogFilter()
    file_handler.addFilter(retry_filter)
    file_handler.setFormatter(JsonLineFormatter())
    root.addHandler(file_handler)

    if also_stderr:
        stream = logging.StreamHandler()
        stream.addFilter(PrawRetryLogFilter())
        stream.setFormatter(OperatorLogFormatter())
        root.addHandler(stream)

    logging.getLogger(APP_LOGGER_NAME).info(
        "logging configured",
        extra={"event_name": "logging.configured", "component": "bot_logging"},
    )
    return path


def get_logger(name: str = APP_LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    event_name: str,
    message: str,
    *,
    level: int = logging.INFO,
    correlation_id: str | None = None,
    action_id: str | None = None,
    component: str | None = None,
    **extra: Any,
) -> None:
    payload = {"event_name": event_name}
    correlation_id = correlation_id or current_run_id()
    if correlation_id:
        payload["correlation_id"] = correlation_id
    if action_id:
        payload["action_id"] = action_id
    if component:
        payload["component"] = component
    payload.update(extra)
    if "phase" not in payload and current_log_phase():
        payload["phase"] = current_log_phase()
    get_logger().log(level, message, extra=payload)
