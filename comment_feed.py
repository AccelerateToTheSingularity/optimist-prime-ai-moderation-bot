"""Local, read-only browser feed of recent comments from a subreddit.

This is deliberately separate from ``bot_runner.py``.  It is a maintainer
view, not another bot phase: it authenticates with the operator's local Reddit
configuration, reads recent comments, and keeps a bounded in-memory cache.
It never posts, replies, moderates, changes flair, sends modmail, or writes
bot runtime state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hmac
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlsplit

from bot_logging import configure_logging, ensure_run_id, log_event, set_log_phase
from env_file import clear_inherited_runtime_environment, load_local_env
from local_http_security import is_local_http_request
from reddit_observability import observable_requestor_settings
from reddit_request_budget import resolve_request_budget_settings


DEFAULT_SUBREDDIT = ""
DEFAULT_PORT = 8090
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_MAX_ITEMS = 100
DEFAULT_TIMEOUT_SECONDS = 15
MIN_TIMEOUT_SECONDS = 5
MAX_TIMEOUT_SECONDS = 60
MAX_BODY_CHARS = 20_000
SESSION_TOKEN_HEADER = "X-Optimist-Prime-Feed-Token"
ALLOWED_REDDIT_HOSTS = {"reddit.com", "www.reddit.com", "old.reddit.com", "redd.it"}


class FeedError(RuntimeError):
    """Base error for configuration and feed startup failures."""


class FeedConfigurationError(FeedError):
    """The local configuration cannot support a read-only Reddit connection."""


class FeedIdentityError(FeedError):
    """Reddit authenticated as an account other than the configured account."""


class FeedServiceStopped(FeedError):
    """An operation was requested after the feed had been stopped."""


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    """Read a possibly lazy PRAW attribute without allowing it to abort a row."""
    try:
        return getattr(value, name)
    except Exception:
        return default


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _reddit_url(value: Any) -> str:
    """Return only an http(s) Reddit URL; reject arbitrary input."""
    raw = _as_text(value).strip()
    if not raw:
        return ""
    candidate = urljoin("https://www.reddit.com", raw)
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in ALLOWED_REDDIT_HOSTS:
        return ""
    return candidate


def _iso_utc(timestamp: float | int | None) -> str | None:
    if timestamp is None:
        return None
    try:
        value = float(timestamp)
        if value <= 0:
            return None
        return (
            dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _numeric(value: Any, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class FeedComment:
    """The small, non-PRAW representation sent to the local browser."""

    comment_id: str
    author: str
    body: str
    created_utc: float
    score: int | None
    permalink: str
    subreddit: str
    link_title: str
    body_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.comment_id,
            "author": self.author,
            "body": self.body,
            "createdAt": _iso_utc(self.created_utc),
            "createdUtc": self.created_utc,
            "score": self.score,
            "permalink": self.permalink,
            "subreddit": self.subreddit,
            "linkTitle": self.link_title,
            "bodyTruncated": self.body_truncated,
        }


def normalize_comment(
    comment: Any,
    *,
    fallback_subreddit: str = DEFAULT_SUBREDDIT,
    max_body_chars: int = MAX_BODY_CHARS,
) -> FeedComment | None:
    """Convert one PRAW-like comment to safe, bounded feed data.

    The function intentionally does not access ``comment.submission``.  That
    property can trigger an extra Reddit request for every row.  PRAW's
    comment listing normally supplies ``link_title`` already.
    """
    comment_id = _as_text(_safe_attr(comment, "id")).strip()
    if not comment_id:
        return None

    raw_author = _safe_attr(comment, "author")
    if raw_author is None:
        author = "[deleted]"
    elif isinstance(raw_author, str):
        author = raw_author.strip() or "[unknown]"
    else:
        author = _as_text(_safe_attr(raw_author, "name"), "[unknown]").strip()
        author = author or "[unknown]"

    body = _as_text(_safe_attr(comment, "body"), "")
    body_truncated = len(body) > max_body_chars
    if body_truncated:
        body = body[:max_body_chars]

    created = _numeric(_safe_attr(comment, "created_utc"))
    created_utc = float(created) if created is not None else 0.0
    score_value = _numeric(_safe_attr(comment, "score"), integer=True)
    score = int(score_value) if score_value is not None else None

    subreddit_obj = _safe_attr(comment, "subreddit")
    subreddit = _as_text(_safe_attr(subreddit_obj, "display_name"), "").strip()
    subreddit = subreddit or fallback_subreddit

    permalink = _reddit_url(_safe_attr(comment, "permalink"))
    link_title = _as_text(_safe_attr(comment, "link_title"), "").strip()
    return FeedComment(
        comment_id=comment_id,
        author=author,
        body=body,
        created_utc=created_utc,
        score=score,
        permalink=permalink,
        subreddit=subreddit,
        link_title=link_title,
        body_truncated=body_truncated,
    )


def _safe_error(exc: BaseException, *, max_chars: int = 240) -> str:
    """Make a useful UI error while removing values from local configuration."""
    message = f"{type(exc).__name__}: {_as_text(exc).strip()}".strip()
    for key, value in os.environ.items():
        if not value or len(value) < 4:
            continue
        if key.startswith(("REDDIT_", "DEEPSEEK_", "OPENAI_", "ANTHROPIC_", "GEMINI_")):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?i)(client_secret|password|refresh_token|api_key)=([^&\s]+)", r"\1=[redacted]", message)
    if len(message) > max_chars:
        message = message[: max_chars - 1].rstrip() + "…"
    return message or type(exc).__name__


def _normalise_username(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("u/"):
        value = value[2:]
    return value.casefold()


def _configured_timeout(environment: Mapping[str, str] | None = None) -> int:
    values = environment if environment is not None else os.environ
    raw = _as_text(values.get("REDDIT_API_TIMEOUT_SECONDS"), "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(MIN_TIMEOUT_SECONDS, min(MAX_TIMEOUT_SECONDS, value))


def create_read_only_reddit(
    *,
    environment: Mapping[str, str] | None = None,
    praw_module: Any | None = None,
    timeout_seconds: int | None = None,
    expected_username: str | None = None,
) -> tuple[Any, str]:
    """Authenticate, verify the account, then switch the PRAW object to read-only."""
    values = environment if environment is not None else os.environ
    client_id = _as_text(values.get("REDDIT_CLIENT_ID"), "").strip()
    client_secret = _as_text(values.get("REDDIT_CLIENT_SECRET"), "").strip()
    refresh_token = _as_text(values.get("REDDIT_REFRESH_TOKEN"), "").strip()

    missing = []
    if not client_id:
        missing.append("REDDIT_CLIENT_ID")
    if not client_secret:
        missing.append("REDDIT_CLIENT_SECRET")
    if not _as_text(values.get("REDDIT_DEVELOPER_USERNAME"), "").strip():
        missing.append("REDDIT_DEVELOPER_USERNAME")
    if not refresh_token:
        missing.append("REDDIT_REFRESH_TOKEN")
    if missing:
        raise FeedConfigurationError("Missing local Reddit setting(s): " + ", ".join(missing))

    if praw_module is None:
        try:
            import praw as praw_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise FeedConfigurationError("PRAW is not installed in the selected Python environment") from exc

    app_name = _as_text(values.get("REDDIT_APP_NAME"), "OptimistPrimeModBot").strip() or "OptimistPrimeModBot"
    developer = _as_text(values.get("REDDIT_DEVELOPER_USERNAME"), "").strip()
    user_agent = f"script:{app_name}:comment-feed/1.0 (by /u/{developer})"
    actual_timeout = timeout_seconds if timeout_seconds is not None else _configured_timeout(values)
    budget_settings = resolve_request_budget_settings(
        values,
        default_root=Path(__file__).resolve().parent,
    )
    kwargs: dict[str, Any] = {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent,
    }
    kwargs.update(
        observable_requestor_settings(
            actual_timeout,
            request_budget_path=budget_settings.path,
            request_budget_limit=budget_settings.max_requests,
            request_budget_window_seconds=budget_settings.window_seconds,
            request_budget_per_minute=budget_settings.requests_per_minute,
        )
    )
    kwargs["refresh_token"] = refresh_token

    try:
        reddit = praw_module.Reddit(**kwargs)
        me = reddit.user.me()
    except Exception as exc:
        raise FeedError(f"Reddit authentication failed: {_safe_error(exc)}") from exc

    account_name = _as_text(_safe_attr(me, "name"), "").strip()
    if not account_name:
        raise FeedError("Reddit authentication returned no account name")

    configured_expected = expected_username
    if configured_expected is None:
        configured_expected = _as_text(values.get("BOT_EXPECTED_USERNAME"), "").strip()
    if configured_expected and _normalise_username(account_name) != _normalise_username(configured_expected):
        raise FeedIdentityError(
            f"Reddit authenticated as u/{account_name}, not the configured account"
        )

    try:
        reddit.read_only = True
    except Exception as exc:
        raise FeedError("Reddit connected, but the client could not be placed in read-only mode") from exc

    log_event(
        "feed.reddit_authenticated",
        "read-only Reddit client authenticated",
        component="comment_feed",
        account=account_name,
    )
    return reddit, account_name


class CommentFeedService:
    """Poll recent comments and expose a thread-safe, bounded snapshot."""

    def __init__(
        self,
        reddit: Any,
        subreddit: str = DEFAULT_SUBREDDIT,
        *,
        account_name: str = "",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_items: int = DEFAULT_MAX_ITEMS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.reddit = reddit
        self.subreddit_name = subreddit.strip()
        if not self.subreddit_name:
            raise FeedConfigurationError(
                "Set BOT_SUBREDDIT or pass --subreddit before starting the comment feed"
            )
        self.account_name = account_name.strip()
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.max_items = max(1, min(100, int(max_items)))
        self._clock = clock
        self._subreddit = reddit.subreddit(self.subreddit_name)
        self._comments: dict[str, FeedComment] = {}
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._stopped = False
        self._paused = False
        self._refreshing = False
        self._state = "starting"
        self._started_at = self._clock()
        self._last_success_at: float | None = None
        self._next_refresh_at: float | None = None
        self._last_error: str | None = None
        self._last_fetch_count = 0
        self._last_added_count = 0
        self._refresh_count = 0
        self._failed_refresh_count = 0

    def start(self) -> None:
        with self._state_lock:
            if self._stopped:
                raise FeedServiceStopped("The comment feed has already been stopped")
            if self._worker and self._worker.is_alive():
                return
            self._state = "starting"
            self._worker = threading.Thread(
                target=self._poll_loop,
                name="optimist-prime-comment-feed",
                daemon=True,
            )
            self._worker.start()
        log_event("feed.started", "read-only comment feed started", component="comment_feed")

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._state_lock:
                paused = self._paused
            if paused:
                self._wake_event.wait(0.25)
                self._wake_event.clear()
                continue
            self.refresh_now()
            self._wake_event.wait(self.poll_seconds)
            self._wake_event.clear()
            if self._stop_event.is_set():
                return

    def refresh_now(self) -> bool:
        """Fetch once.  A failure leaves the previous cache intact and is retryable."""
        with self._refresh_lock:
            with self._state_lock:
                if self._stopped:
                    return False
                self._refreshing = True
            try:
                raw_comments = list(self._subreddit.comments(limit=self.max_items))
                incoming: dict[str, FeedComment] = {}
                for raw in raw_comments:
                    normalized = normalize_comment(
                        raw,
                        fallback_subreddit=self.subreddit_name,
                    )
                    if normalized is not None:
                        incoming[normalized.comment_id] = normalized
            except Exception as exc:
                error = _safe_error(exc)
                now = self._clock()
                with self._state_lock:
                    self._failed_refresh_count += 1
                    self._last_error = error
                    self._state = "paused" if self._paused else "error"
                    self._next_refresh_at = None if self._paused else now + self.poll_seconds
                    self._refreshing = False
                log_event(
                    "feed.refresh.failed",
                    "comment feed refresh failed; cached comments retained",
                    component="comment_feed",
                    error_type=type(exc).__name__,
                )
                return False

            now = self._clock()
            with self._state_lock:
                if self._stopped:
                    self._refreshing = False
                    return False
                old_ids = set(self._comments)
                self._comments.update(incoming)
                ordered = sorted(
                    self._comments.values(),
                    key=lambda item: (item.created_utc, item.comment_id),
                    reverse=True,
                )
                self._comments = {
                    item.comment_id: item for item in ordered[: self.max_items]
                }
                self._last_fetch_count = len(incoming)
                self._last_added_count = len(set(incoming) - old_ids)
                self._refresh_count += 1
                self._last_success_at = now
                self._last_error = None
                self._state = "paused" if self._paused else "connected"
                self._next_refresh_at = None if self._paused else now + self.poll_seconds
                self._refreshing = False
            log_event(
                "feed.refresh.succeeded",
                "read-only comment feed refreshed",
                component="comment_feed",
                fetched_count=len(incoming),
                added_count=len(set(incoming) - old_ids),
                cached_count=len(self._comments),
            )
            return True

    def pause(self) -> bool:
        with self._state_lock:
            if self._stopped:
                return False
            self._paused = True
            self._state = "paused"
            self._next_refresh_at = None
            self._wake_event.set()
        log_event("feed.paused", "automatic comment feed updates paused", component="comment_feed")
        return True

    def resume(self) -> bool:
        with self._state_lock:
            if self._stopped:
                return False
            self._paused = False
            self._state = "starting"
            self._next_refresh_at = self._clock()
            self._wake_event.set()
        log_event("feed.resumed", "automatic comment feed updates resumed", component="comment_feed")
        return True

    def stop(self) -> None:
        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
            self._paused = False
            self._state = "stopped"
            self._next_refresh_at = None
            self._stop_event.set()
            self._wake_event.set()
        log_event("feed.stopped", "read-only comment feed stopped", component="comment_feed")

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            ordered = sorted(
                self._comments.values(),
                key=lambda item: (item.created_utc, item.comment_id),
                reverse=True,
            )
            status = self._state
            if status == "starting":
                message = "Connecting to Reddit and loading recent comments."
            elif status == "connected":
                message = "Connected; showing the newest comments returned by Reddit."
            elif status == "paused":
                message = "Automatic updates are paused; the cached comments remain visible."
            elif status == "error":
                message = "The latest read failed; cached comments remain visible while the feed retries."
            else:
                message = "The read-only comment feed is stopped."
            return {
                "ok": True,
                "status": {
                    "state": status,
                    "message": message,
                    "subreddit": self.subreddit_name,
                    "accountName": self.account_name or None,
                    "readOnly": True,
                    "writeOperations": False,
                    "startedAt": _iso_utc(self._started_at),
                    "lastSuccessfulRefresh": _iso_utc(self._last_success_at),
                    "nextRefreshAt": _iso_utc(self._next_refresh_at),
                    "lastError": self._last_error,
                    "lastFetchCount": self._last_fetch_count,
                    "lastAddedCount": self._last_added_count,
                    "cachedCount": len(ordered),
                    "refreshCount": self._refresh_count,
                    "failedRefreshCount": self._failed_refresh_count,
                    "refreshInProgress": self._refreshing,
                    "pollSeconds": self.poll_seconds,
                    "maxItems": self.max_items,
                },
                "comments": [item.to_dict() for item in ordered],
            }


def _script_json(value: Any) -> str:
    """Embed JSON in a script without allowing markup termination."""
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


FEED_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Optimist Prime — subreddit comment feed</title>
  <meta name="description" content="A local read-only view of recent subreddit comments.">
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: Canvas; color: CanvasText; }
    main, header, .panel { max-width: 1100px; margin: 0 auto; }
    header { padding: 28px 20px 12px; }
    h1 { margin: 0 0 8px; font-size: clamp(1.5rem, 3vw, 2.25rem); }
    .subtitle { margin: 0; opacity: .78; }
    .panel { margin: 12px auto; padding: 20px; border: 1px solid color-mix(in srgb, CanvasText 25%, transparent); border-radius: 12px; }
    .notice { background: color-mix(in srgb, Highlight 13%, Canvas); }
    .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 16px 0; }
    .status-item { padding: 12px; border-radius: 8px; background: color-mix(in srgb, CanvasText 8%, Canvas); }
    .label { display: block; font-size: .8rem; opacity: .7; margin-bottom: 4px; }
    .value { font-weight: 650; overflow-wrap: anywhere; }
    .status-badge { display: inline-block; padding: 4px 9px; border-radius: 999px; font-weight: 700; }
    .status-connected { background: #18794e; color: white; }
    .status-starting { background: #8a5a00; color: white; }
    .status-paused { background: #5b5b5b; color: white; }
    .status-error { background: #a52626; color: white; }
    .status-stopped { background: #5b5b5b; color: white; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 16px; }
    button, .link-button { min-height: 40px; padding: 8px 14px; border: 1px solid color-mix(in srgb, CanvasText 35%, transparent); border-radius: 7px; background: ButtonFace; color: ButtonText; font: inherit; cursor: pointer; }
    button:focus-visible, a:focus-visible { outline: 3px solid Highlight; outline-offset: 2px; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    a { color: LinkText; }
    .comment { margin: 12px 0; padding: 18px; border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius: 10px; }
    .comment:first-child { border-color: Highlight; }
    .comment-meta { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: baseline; margin-bottom: 10px; }
    .author { font-weight: 700; }
    .muted { opacity: .72; font-size: .9rem; }
    .post-title { margin: 0 0 12px; font-weight: 650; overflow-wrap: anywhere; }
    .body { margin: 0 0 14px; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.45; }
    .comment-link { font-size: .9rem; }
    .error { color: #c93636; min-height: 1.3em; overflow-wrap: anywhere; }
    .empty { padding: 30px 8px; text-align: center; opacity: .75; }
    [hidden] { display: none !important; }
  </style>
</head>
<body>
  <header>
    <h1>Optimist Prime comment feed</h1>
    <p class="subtitle">Recent public comments arriving in <span id="subreddit-name">your subreddit</span></p>
  </header>
  <section class="panel notice" aria-label="Read-only explanation">
    <strong>Read-only view.</strong>
    This page can read recent comments, but it never posts, replies, moderates, changes flair, sends modmail, or changes bot state.
    It shows a bounded live cache, not a permanent archive; Reddit may omit or remove content.
  </section>
  <section class="panel" aria-label="Feed status">
    <div class="status-grid">
      <div class="status-item"><span class="label">Connection</span><span id="connection-status" class="value status-badge status-starting">Loading</span></div>
      <div class="status-item"><span class="label">Authenticated account</span><span id="account-name" class="value">Checking…</span></div>
      <div class="status-item"><span class="label">Comments shown</span><span id="comment-count" class="value">0</span></div>
      <div class="status-item"><span class="label">Last successful refresh</span><span id="last-refresh" class="value">Not yet</span></div>
      <div class="status-item"><span class="label">Next automatic check</span><span id="next-refresh" class="value">Waiting…</span></div>
    </div>
    <p id="status-message" role="status" aria-live="polite">Connecting…</p>
    <p id="error-message" class="error" role="alert"></p>
    <div class="controls">
      <button id="refresh-button" type="button">Refresh now</button>
      <button id="pause-button" type="button">Pause updates</button>
      <button id="resume-button" type="button" hidden>Resume updates</button>
      <button id="stop-button" type="button">Stop feed</button>
      <a id="subreddit-link" class="link-button" href="#" target="_blank" rel="noopener noreferrer" hidden>Open subreddit</a>
    </div>
    <p id="operation-message" class="muted" role="status" aria-live="polite"></p>
  </section>
  <main id="comment-feed" aria-live="polite" aria-label="Recent subreddit comments"></main>
  <p id="empty-message" class="panel empty" hidden>No comments have been loaded yet.</p>
  <script>
  (() => {
    "use strict";
    const sessionToken = __SESSION_TOKEN__;
    const refreshButton = document.getElementById("refresh-button");
    const pauseButton = document.getElementById("pause-button");
    const resumeButton = document.getElementById("resume-button");
    const stopButton = document.getElementById("stop-button");
    const feed = document.getElementById("comment-feed");
    const emptyMessage = document.getElementById("empty-message");
    const operationMessage = document.getElementById("operation-message");
    let stopped = false;

    function safeRedditUrl(value) {
      try {
        const url = new URL(value || "", "https://www.reddit.com");
        const hosts = new Set(["reddit.com", "www.reddit.com", "old.reddit.com", "redd.it"]);
        if ((url.protocol === "https:" || url.protocol === "http:") && hosts.has(url.hostname.toLowerCase())) {
          return url.href;
        }
      } catch (error) { /* an absent or malformed permalink simply has no link */ }
      return "";
    }

    function displayTime(value) {
      if (!value) return "Not available";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? "Not available" : date.toLocaleString();
    }

    function statusClass(state) {
      const allowed = new Set(["connected", "starting", "paused", "error", "stopped"]);
      return allowed.has(state) ? "status-" + state : "status-starting";
    }

    function createComment(comment) {
      const article = document.createElement("article");
      article.className = "comment";
      const meta = document.createElement("div");
      meta.className = "comment-meta";
      const author = document.createElement("span");
      author.className = "author";
      author.textContent = "u/" + (comment.author || "[unknown]");
      const when = document.createElement("span");
      when.className = "muted";
      when.textContent = displayTime(comment.createdAt);
      const score = document.createElement("span");
      score.className = "muted";
      score.textContent = comment.score === null || comment.score === undefined ? "score unavailable" : "score " + comment.score;
      meta.append(author, when, score);
      article.append(meta);

      if (comment.linkTitle) {
        const title = document.createElement("h2");
        title.className = "post-title";
        title.textContent = comment.linkTitle;
        article.append(title);
      }

      const body = document.createElement("p");
      body.className = "body";
      body.textContent = comment.body || "[no comment text returned]";
      article.append(body);

      const permalink = safeRedditUrl(comment.permalink);
      if (permalink) {
        const link = document.createElement("a");
        link.className = "comment-link";
        link.href = permalink;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "Open this comment on Reddit";
        article.append(link);
      }
      return article;
    }

    function render(payload) {
      const status = payload && payload.status ? payload.status : {};
      const state = status.state || "error";
      const badge = document.getElementById("connection-status");
      badge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
      badge.className = "value status-badge " + statusClass(state);
      document.getElementById("subreddit-name").textContent = status.subreddit ? "r/" + status.subreddit : "the subreddit";
      document.getElementById("account-name").textContent = status.accountName ? "u/" + status.accountName : "Not identified";
      const comments = Array.isArray(payload.comments) ? payload.comments : [];
      document.getElementById("comment-count").textContent = String(comments.length);
      document.getElementById("last-refresh").textContent = displayTime(status.lastSuccessfulRefresh);
      document.getElementById("next-refresh").textContent = status.nextRefreshAt ? displayTime(status.nextRefreshAt) : (state === "paused" ? "Paused" : "Waiting…");
      document.getElementById("status-message").textContent = status.message || "No status message available.";
      document.getElementById("error-message").textContent = status.lastError || "";
      feed.replaceChildren();
      comments.forEach((comment) => feed.append(createComment(comment)));
      emptyMessage.hidden = comments.length !== 0;
      stopped = state === "stopped";
      refreshButton.disabled = stopped;
      pauseButton.hidden = stopped || state === "paused";
      resumeButton.hidden = stopped || state !== "paused";
      pauseButton.disabled = stopped;
      stopButton.disabled = stopped;
      const subredditLink = document.getElementById("subreddit-link");
      if (status.subreddit) {
        subredditLink.href = "https://www.reddit.com/r/" + encodeURIComponent(status.subreddit) + "/";
        subredditLink.hidden = false;
      } else {
        subredditLink.hidden = true;
      }
    }

    async function getFeed() {
      const response = await fetch("/api/feed", { cache: "no-store" });
      if (!response.ok) throw new Error("Feed status request failed");
      return response.json();
    }

    async function dispatch(action) {
      refreshButton.disabled = true;
      pauseButton.disabled = true;
      resumeButton.disabled = true;
      operationMessage.textContent = "Working…";
      try {
        const response = await fetch("/api/actions", {
          method: "POST",
          cache: "no-store",
          headers: { "Content-Type": "application/json", ["X-Optimist-Prime-Feed-Token"]: sessionToken },
          body: JSON.stringify({ action })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error("Feed action was rejected");
        render(payload);
        operationMessage.textContent = payload.message || "Action completed.";
      } catch (error) {
        operationMessage.textContent = "The local feed control failed; the cached view remains visible.";
        try { render(await getFeed()); } catch (ignored) { /* keep the existing view */ }
      } finally {
        if (!stopped) {
          refreshButton.disabled = false;
          pauseButton.disabled = false;
          resumeButton.disabled = false;
        }
      }
    }

    refreshButton.addEventListener("click", () => dispatch("feed.refresh"));
    pauseButton.addEventListener("click", () => dispatch("feed.pause"));
    resumeButton.addEventListener("click", () => dispatch("feed.resume"));
    stopButton.addEventListener("click", () => dispatch("feed.stop"));

    async function poll() {
      if (!stopped) {
        try { render(await getFeed()); } catch (error) {
          operationMessage.textContent = "The browser could not refresh the local status; the last view remains visible.";
        }
      }
      window.setTimeout(poll, 4000);
    }

    getFeed().then(render).catch(() => {
      operationMessage.textContent = "The local feed is starting; waiting for its first status update.";
    });
    window.setTimeout(poll, 4000);
  })();
  </script>
</body>
</html>
"""


def render_feed_page(session_token: str) -> str:
    return FEED_PAGE_TEMPLATE.replace("__SESSION_TOKEN__", _script_json(session_token))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _make_handler(service: CommentFeedService, session_token: str):
    class FeedRequestHandler(BaseHTTPRequestHandler):
        server_version = "OptimistPrimeCommentFeed/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid writing comment text or arbitrary browser request data to stdout.
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

        def _path(self) -> str:
            return urlsplit(self.path).path

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if not is_local_http_request(self.headers, self.server.server_address):
                self.send_error(403, "This feed interface accepts local requests only.")
                return
            path = self._path()
            if path == "/":
                self._send(200, render_feed_page(session_token).encode("utf-8"), "text/html; charset=utf-8")
                return
            if path in {"/api/feed", "/api/status"}:
                self._send_json(200, service.snapshot())
                return
            self._send_json(404, {"ok": False, "message": "Not found"})

        def _authorised(self) -> bool:
            supplied = self.headers.get(SESSION_TOKEN_HEADER, "")
            return bool(supplied) and hmac.compare_digest(supplied, session_token)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if not is_local_http_request(
                self.headers,
                self.server.server_address,
                require_origin=True,
            ):
                self.send_error(403, "This feed interface accepts local requests only.")
                return
            if self._path() != "/api/actions":
                self._send_json(404, {"ok": False, "message": "Not found"})
                return
            if not self._authorised():
                self._send_json(403, {"ok": False, "message": "A valid local feed session is required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > 4096:
                self._send_json(400, {"ok": False, "message": "Invalid request body"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"ok": False, "message": "Request must be JSON"})
                return
            action = payload.get("action") if isinstance(payload, dict) else None
            if action == "feed.refresh":
                succeeded = service.refresh_now()
                message = "Feed refreshed." if succeeded else "Refresh failed; cached comments were retained for retry."
            elif action == "feed.pause":
                succeeded = service.pause()
                message = "Automatic updates paused." if succeeded else "The feed is already stopped."
            elif action == "feed.resume":
                succeeded = service.resume()
                message = "Automatic updates resumed." if succeeded else "The feed is stopped."
            elif action == "feed.stop":
                succeeded = True
                service.stop()
                message = "Feed stopped. No further Reddit requests will be made."
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send_json(400, {"ok": False, "message": "Unknown feed action"})
                return
            response = service.snapshot()
            response["actionSucceeded"] = succeeded
            response["message"] = message
            self._send_json(200, response)

    return FeedRequestHandler


class FeedHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: Any, service: CommentFeedService, session_token: str):
        self.feed_service = service
        self.session_token = session_token
        super().__init__(address, handler)


def create_feed_server(
    service: CommentFeedService,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    session_token: str | None = None,
) -> tuple[FeedHTTPServer, str]:
    token = session_token or secrets.token_urlsafe(32)
    handler = _make_handler(service, token)
    return FeedHTTPServer((host, port), handler, service, token), token


def _subreddit_name(value: str) -> str:
    cleaned = value.strip()
    if cleaned.lower().startswith("r/"):
        cleaned = cleaned[2:]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,20}", cleaned):
        raise argparse.ArgumentTypeError("subreddit must be a simple Reddit community name")
    return cleaned


def _bounded_float(minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number") from exc
        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum:g} and {maximum:g}")
        return parsed

    return parse


def _bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a whole number") from exc
        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local read-only Reddit comment feed")
    parser.add_argument(
        "--subreddit",
        type=_subreddit_name,
        default=None,
        help="community to display (required unless BOT_SUBREDDIT is set in .env)",
    )
    parser.add_argument("--poll-seconds", type=_bounded_float(1, 3600), default=DEFAULT_POLL_SECONDS, help="automatic refresh interval (default: 15)")
    parser.add_argument("--max-items", type=_bounded_int(1, 100), default=DEFAULT_MAX_ITEMS, help="maximum comments kept in memory (default: 100)")
    parser.add_argument("--port", type=_bounded_int(1, 65535), default=DEFAULT_PORT, help="localhost port (default: 8090)")
    parser.add_argument("--timeout-seconds", type=_bounded_int(MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS), default=None, help="bounded Reddit API timeout")
    parser.add_argument("--expected-account", default=None, help="optional account name that must authenticate")
    parser.add_argument("--once", "--check", dest="once", action="store_true", help="authenticate and perform one read-only refresh, then exit")
    parser.add_argument("--no-browser", action="store_true", help="run the local server without opening a browser tab")
    parser.add_argument(
        "--enable-feed",
        action="store_true",
        help="Explicitly enable the optional read-only Reddit comment viewer",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.enable_feed:
        print("The optional live comment viewer is disabled. No Reddit connection was opened.")
        print("Use --enable-feed only when you intentionally want to watch comments.")
        return 0
    clear_inherited_runtime_environment()
    load_local_env(override=True)
    configured_subreddit = args.subreddit or os.environ.get("BOT_SUBREDDIT", "").strip()
    if not configured_subreddit:
        print("Set BOT_SUBREDDIT in .env or pass --subreddit before starting the comment feed.")
        return 2
    try:
        subreddit_name = _subreddit_name(configured_subreddit)
    except argparse.ArgumentTypeError as exc:
        print(f"Invalid subreddit: {exc}")
        return 2
    run_id = ensure_run_id()
    set_log_phase("comment_feed")
    log_path = configure_logging(also_stderr=not args.no_browser or args.once)
    log_event(
        "feed.startup",
        "starting local read-only comment feed",
        component="comment_feed",
        log_path=str(log_path),
        subreddit=subreddit_name,
        poll_seconds=args.poll_seconds,
        max_items=args.max_items,
        timeout_seconds=args.timeout_seconds or _configured_timeout(),
    )

    try:
        reddit, account_name = create_read_only_reddit(
            timeout_seconds=args.timeout_seconds,
            expected_username=args.expected_account,
        )
        service = CommentFeedService(
            reddit,
            subreddit_name,
            account_name=account_name,
            poll_seconds=args.poll_seconds,
            max_items=args.max_items,
        )
        if args.once:
            if service.refresh_now():
                snapshot = service.snapshot()
                print(
                    f"Read-only check succeeded: u/{account_name} read "
                    f"r/{subreddit_name}; {snapshot['status']['cachedCount']} comment(s) loaded."
                )
                service.stop()
                return 0
            print("Read-only check reached Reddit but the comment refresh failed; it is retryable.")
            service.stop()
            return 1

        server, _token = create_feed_server(service, port=args.port)
        service.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Read-only comment feed running at {url}")
        print("This feed does not post, moderate, send modmail, change flair, or write bot state.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            print("Stopping the local comment feed.")
        finally:
            service.stop()
            server.server_close()
        return 0
    except OSError as exc:
        log_event("feed.startup.failed", "comment feed could not bind its local port", component="comment_feed", error_type=type(exc).__name__)
        print(f"Could not start the local feed: {_safe_error(exc)}")
        return 1
    except FeedError as exc:
        log_event("feed.startup.failed", "comment feed startup failed", component="comment_feed", error_type=type(exc).__name__)
        print(f"Could not start the local feed: {_safe_error(exc)}")
        return 1
    except Exception as exc:
        log_event("feed.startup.failed", "unexpected comment feed startup failure", component="comment_feed", error_type=type(exc).__name__)
        print(f"Could not start the local feed: {_safe_error(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
