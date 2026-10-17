"""Observable PRAW request handling for the local Optimist Prime worker.

PRAW retries transient Reddit failures internally.  That is useful behavior,
but its default log line does not tell the operator which scheduled cycle is
waiting, which attempt failed, or how long the attempt took.  This module
keeps PRAW's retry policy and adds safe per-attempt events around the actual
HTTP request.  It never records request bodies, headers, query strings, or
credential values.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import logging
from threading import Lock
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from prawcore import Requestor
from prawcore.sessions import Session as PrawcoreSession

from bot_logging import log_event, retry_pending, set_request_observation
from reddit_request_budget import (
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_WINDOW_SECONDS,
    RedditRequestBudget,
)


@dataclass
class _SingleWriteState:
    """Guard one native moderator-note POST within its caller's context."""

    attempted: bool = False
    retry_allowed_after_401: bool = False
    lock: Lock = field(default_factory=Lock)

    def claim_attempt(self) -> None:
        with self.lock:
            if self.attempted and not self.retry_allowed_after_401:
                raise RuntimeError(
                    "a repeated POST to /api/mod/notes was blocked within "
                    "single_write_attempt"
                )
            self.attempted = True
            self.retry_allowed_after_401 = False

    def observe_response(self, status_code: int | None) -> None:
        if status_code != 401:
            return
        # Reddit rejected the request as unauthenticated, so PRAW may safely
        # refresh credentials and repeat it. A later failure remains guarded.
        with self.lock:
            self.retry_allowed_after_401 = True


_single_write_state: ContextVar[_SingleWriteState | None] = ContextVar(
    "optimist_prime_single_write_attempt",
    default=None,
)


@contextmanager
def single_write_attempt():
    """Prevent PRAW from retransmitting a moderator-note write in this scope.

    PRAW retries POST requests after transport failures and 5xx responses.
    Reddit provides no idempotency key for moderator notes, so a repeated
    request could create duplicate notes. The guard allows only an explicit
    401 retry for PRAW's normal token-refresh flow. Nested scopes share one
    guard; each fresh outer scope starts a new independent write.
    """
    existing = _single_write_state.get()
    if existing is not None:
        yield
        return

    token: Token[_SingleWriteState | None] = _single_write_state.set(
        _SingleWriteState()
    )
    try:
        yield
    finally:
        _single_write_state.reset(token)


def _is_moderator_note_write(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.rstrip("/") == "/api/mod/notes"


def _request_parts(args: tuple[Any, ...]) -> tuple[str, str, str]:
    method = str(args[0] if args else "unknown").upper()
    raw_url = str(args[1] if len(args) > 1 else "")
    try:
        parsed = urlsplit(raw_url)
        host = (parsed.hostname or "unknown").lower()
        path = (parsed.path or "/")[:500]
    except ValueError:
        host = "unknown"
        path = "[unparseable]"
    return method, host, path


def _request_key(method: str, host: str, path: str) -> tuple[str, str, str]:
    return method, host, path


def _original_exception(error: BaseException) -> BaseException:
    original = getattr(error, "original_exception", None)
    return original if isinstance(original, BaseException) else error


def _is_retryable_exception(error: BaseException) -> bool:
    original = _original_exception(error)
    return isinstance(original, PrawcoreSession.RETRY_EXCEPTIONS)


class ObservableRequestor(Requestor):
    """PRAW requestor that logs safe timing and retry evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        budget_path = kwargs.pop("request_budget_path", None)
        budget_limit = kwargs.pop("request_budget_limit", None)
        budget_window_seconds = kwargs.pop(
            "request_budget_window_seconds", DEFAULT_WINDOW_SECONDS
        )
        budget_per_minute = kwargs.pop(
            "request_budget_per_minute", DEFAULT_REQUESTS_PER_MINUTE
        )
        super().__init__(*args, **kwargs)
        self._attempts: dict[tuple[str, str, str], int] = {}
        self._request_ids: dict[tuple[str, str, str], str] = {}
        effective_window_seconds = max(1, int(budget_window_seconds))
        effective_requests_per_minute = min(
            DEFAULT_REQUESTS_PER_MINUTE,
            max(1, int(budget_per_minute)),
        )
        derived_max_requests = max(
            1,
            effective_requests_per_minute * effective_window_seconds // 60,
        )
        effective_max_requests = (
            derived_max_requests
            if budget_limit is None
            else min(derived_max_requests, max(1, int(budget_limit)))
        )
        self._request_budget = (
            RedditRequestBudget(
                budget_path,
                max_requests=effective_max_requests,
                window_seconds=effective_window_seconds,
                requests_per_minute=effective_requests_per_minute,
            )
            if budget_path
            else None
        )

    def _begin_attempt(
        self, key: tuple[str, str, str], method: str, host: str, path: str, timeout: Any
    ) -> tuple[str, int, float]:
        # A retry warning is emitted between PRAW attempts.  If no such
        # warning is pending, this is a new logical request even when it uses
        # the same endpoint as an exhausted prior request.
        if not retry_pending():
            self._attempts.pop(key, None)
            self._request_ids.pop(key, None)
        request_id = self._request_ids.setdefault(key, uuid.uuid4().hex[:12])
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt
        set_request_observation(request_id, attempt, timeout)
        started = time.monotonic()
        log_event(
            "reddit.request.started",
            f"Reddit {method} {path} request attempt started",
            level=logging.INFO,
            component="reddit",
            request_id=request_id,
            attempt=attempt,
            request_method=method,
            request_path=path,
            timeout_seconds=timeout,
        )
        return request_id, attempt, started

    def _finish_request(
        self, key: tuple[str, str, str], request_id: str, attempt: int
    ) -> None:
        self._attempts.pop(key, None)
        self._request_ids.pop(key, None)
        set_request_observation(None, None, None)

    def _reserve_request_budget(self, method: str, path: str) -> None:
        """Reserve one slot for the actual HTTP attempt before it starts."""

        if self._request_budget is None:
            return

        def report_wait(wait_seconds: float) -> None:
            log_event(
                "reddit.request_budget.waiting",
                "Reddit request safety budget is full; waiting for a slot",
                level=logging.WARNING,
                component="reddit",
                request_method=method,
                request_path=path,
                wait_seconds=round(max(0.0, wait_seconds), 1),
                budget_requests_per_minute=self._request_budget.requests_per_minute,
                budget_window_seconds=self._request_budget.window_seconds,
                budget_max_requests=self._request_budget.max_requests,
            )

        acquisition = self._request_budget.acquire(on_wait=report_wait)
        if acquisition.waited_seconds > 0:
            log_event(
                "reddit.request_budget.acquired",
                "Reddit request safety budget slot acquired",
                level=logging.INFO,
                component="reddit",
                request_method=method,
                request_path=path,
                waited_seconds=round(acquisition.waited_seconds, 1),
                budget_used_requests=acquisition.used_requests,
                budget_remaining_requests=acquisition.remaining_requests,
                budget_reset_seconds=round(acquisition.reset_seconds, 1),
            )

    def request(self, *args: Any, timeout: float | None = None, **kwargs: Any):
        method, host, path = _request_parts(args)
        key = _request_key(method, host, path)
        write_state = _single_write_state.get()
        guarded_note_write = bool(
            write_state is not None and _is_moderator_note_write(method, path)
        )
        if guarded_note_write:
            # Claim before waiting for or reserving a request-budget slot so a
            # blocked PRAW retry consumes neither another HTTP attempt nor a
            # second budget reservation.
            write_state.claim_attempt()
        self._reserve_request_budget(method, path)
        # PRAW currently passes prawcore's default timeout (16 seconds) as an
        # explicit per-call argument, which would otherwise override the
        # configured requestor timeout.  The local bot's bounded setting is
        # the source of truth for every Reddit HTTP attempt.
        effective_timeout = self.timeout
        request_id, attempt, started = self._begin_attempt(
            key, method, host, path, effective_timeout
        )

        try:
            response = super().request(*args, timeout=effective_timeout, **kwargs)
        except Exception as exc:  # PRAW wraps network errors in RequestException.
            if guarded_note_write:
                error_response = getattr(exc, "response", None)
                try:
                    error_status_code = int(
                        getattr(error_response, "status_code", None)
                    )
                except (TypeError, ValueError):
                    error_status_code = None
                write_state.observe_response(error_status_code)
            original = _original_exception(exc)
            retryable = _is_retryable_exception(exc)
            log_event(
                "reddit.request.retryable_failure" if retryable else "reddit.request.failed",
                f"Reddit {method} {path} request attempt failed",
                level=logging.WARNING if retryable else logging.ERROR,
                component="reddit",
                request_id=request_id,
                attempt=attempt,
                request_method=method,
                request_path=path,
                timeout_seconds=effective_timeout,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
                retryable=retryable,
                error_type=type(original).__name__,
            )
            if not retryable:
                self._finish_request(key, request_id, attempt)
            raise

        status_code = getattr(response, "status_code", None)
        try:
            status_code_int = int(status_code)
        except (TypeError, ValueError):
            status_code_int = None
        if guarded_note_write:
            write_state.observe_response(status_code_int)
        retryable_response = status_code_int in (
            set(PrawcoreSession.RETRY_STATUSES) | {401}
        )
        log_event(
            "reddit.request.retryable_response"
            if retryable_response
            else "reddit.request.completed",
            f"Reddit {method} {path} request returned",
            level=logging.WARNING if retryable_response else logging.INFO,
            component="reddit",
            request_id=request_id,
            attempt=attempt,
            request_method=method,
            request_path=path,
            timeout_seconds=effective_timeout,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            status_code=status_code_int,
            retryable=retryable_response,
        )
        if not retryable_response:
            self._finish_request(key, request_id, attempt)
        return response


def observable_requestor_settings(
    timeout_seconds: int,
    *,
    request_budget_path: str | Path | None = None,
    request_budget_limit: int | None = None,
    request_budget_window_seconds: int = DEFAULT_WINDOW_SECONDS,
    request_budget_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
) -> dict[str, Any]:
    """Return the PRAW settings needed to make the timeout effective.

    ``timeout=`` is retained for PRAW's configuration surface, while
    ``requestor_kwargs`` is the setting that reaches prawcore's actual
    ``requests`` call in the supported PRAW versions used by this project.
    """
    requestor_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
    if request_budget_path:
        effective_window_seconds = max(1, int(request_budget_window_seconds))
        effective_requests_per_minute = min(
            DEFAULT_REQUESTS_PER_MINUTE,
            max(1, int(request_budget_per_minute)),
        )
        derived_max_requests = max(
            1,
            effective_requests_per_minute * effective_window_seconds // 60,
        )
        effective_max_requests = (
            derived_max_requests
            if request_budget_limit is None
            else min(derived_max_requests, max(1, int(request_budget_limit)))
        )
        requestor_kwargs.update(
            {
                "request_budget_path": str(request_budget_path),
                "request_budget_limit": effective_max_requests,
                "request_budget_window_seconds": effective_window_seconds,
                "request_budget_per_minute": effective_requests_per_minute,
            }
        )
    return {
        "requestor_class": ObservableRequestor,
        "requestor_kwargs": requestor_kwargs,
        "timeout": timeout_seconds,
    }


__all__ = [
    "ObservableRequestor",
    "observable_requestor_settings",
    "single_write_attempt",
]
