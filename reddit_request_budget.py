"""Shared local Reddit request budget for all PRAW clients.

Reddit documents a limit of 100 queries per minute per OAuth client ID,
averaged over a ten-minute window.  The local bot deliberately reserves
a 10 percent margin: at most 90 requests per minute, represented as 900 HTTP
attempts in any rolling ten-minute window.

The budget is backed by a small SQLite file so the scheduled worker and the
read-only moderator console can coordinate across processes.  It counts each
actual HTTP attempt, including an attempt made by PRAW's retry handling.  It
contains timestamps only; it never stores credentials, URLs, bodies, or
response data.
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


OFFICIAL_REDDIT_REQUESTS_PER_MINUTE = 100
DEFAULT_REQUESTS_PER_MINUTE = 90
DEFAULT_WINDOW_SECONDS = 600
DEFAULT_BUDGET_FILENAME = "reddit_request_budget.sqlite3"
_MAX_SLEEP_SECONDS = 30.0


def _safe_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _expanded_path(value: str, root: Path) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(value)))
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True)
class RequestBudgetSettings:
    """Effective budget settings used by one local installation."""

    path: Path
    requests_per_minute: int
    window_seconds: int
    max_requests: int
    official_requests_per_minute: int = OFFICIAL_REDDIT_REQUESTS_PER_MINUTE

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "officialRequestsPerMinute": self.official_requests_per_minute,
            "safetyRequestsPerMinute": self.requests_per_minute,
            "windowSeconds": self.window_seconds,
            "maxRequestsPerWindow": self.max_requests,
        }


@dataclass(frozen=True)
class RequestBudgetSnapshot:
    """Safe status information for diagnostics and the native console."""

    path: Path
    requests_per_minute: int
    window_seconds: int
    max_requests: int
    used_requests: int
    remaining_requests: int
    reset_seconds: float
    readable: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "safetyRequestsPerMinute": self.requests_per_minute,
            "windowSeconds": self.window_seconds,
            "maxRequestsPerWindow": self.max_requests,
            "usedRequests": self.used_requests,
            "remainingRequests": self.remaining_requests,
            "resetSeconds": round(max(0.0, self.reset_seconds), 1),
            "readable": self.readable,
            "error": self.error,
        }


@dataclass(frozen=True)
class RequestBudgetAcquisition:
    """Result returned after one HTTP attempt slot has been reserved."""

    waited_seconds: float
    used_requests: int
    remaining_requests: int
    reset_seconds: float


def resolve_request_budget_settings(
    environment: Mapping[str, str] | None = None,
    *,
    default_root: str | Path | None = None,
) -> RequestBudgetSettings:
    """Resolve the bounded local budget without exposing any secret.

    ``REDDIT_REQUEST_BUDGET_PER_MINUTE`` may lower the default for a developer
    or test installation, but it can never raise the configured safety ceiling
    above 90 requests per minute.
    """

    values = environment if environment is not None else os.environ
    root = Path(default_root) if default_root is not None else Path(__file__).resolve().parent
    raw_rate = str(values.get("REDDIT_REQUEST_BUDGET_PER_MINUTE", "") or "").strip()
    requests_per_minute = min(
        DEFAULT_REQUESTS_PER_MINUTE,
        max(1, _safe_int(raw_rate, DEFAULT_REQUESTS_PER_MINUTE)),
    )

    raw_path = str(values.get("BOT_RUNTIME_REQUEST_BUDGET_FILE", "") or "").strip()
    if raw_path:
        path = _expanded_path(raw_path, root)
    else:
        raw_state = str(values.get("BOT_RUNTIME_STATE_FILE", "") or "").strip()
        if raw_state:
            path = _expanded_path(raw_state, root).with_name(DEFAULT_BUDGET_FILENAME)
        else:
            path = root / "data" / DEFAULT_BUDGET_FILENAME

    max_requests = max(
        1,
        math.floor(requests_per_minute * DEFAULT_WINDOW_SECONDS / 60),
    )
    return RequestBudgetSettings(
        path=path,
        requests_per_minute=requests_per_minute,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        max_requests=max_requests,
    )


class RedditRequestBudget:
    """Coordinate Reddit request attempts between local processes."""

    _TABLE = "request_attempts"

    def __init__(
        self,
        path: str | Path,
        *,
        max_requests: int,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        requests_per_minute: int | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.path = Path(path)
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self.requests_per_minute = requests_per_minute or max(
            1, math.floor(self.max_requests * 60 / self.window_seconds)
        )
        self._clock = clock
        self._sleeper = sleeper

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _try_acquire(self) -> tuple[float, int, float]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self._TABLE} (requested_at REAL NOT NULL)"
            )
            now = float(self._clock())
            if not math.isfinite(now):
                raise ValueError("request budget clock returned a non-finite value")
            cutoff = now - self.window_seconds
            connection.execute(
                f"DELETE FROM {self._TABLE} WHERE requested_at <= ?", (cutoff,)
            )
            used = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {self._TABLE} WHERE requested_at > ?",
                    (cutoff,),
                ).fetchone()[0]
            )
            if used < self.max_requests:
                connection.execute(
                    f"INSERT INTO {self._TABLE} (requested_at) VALUES (?)", (now,)
                )
                used += 1
                connection.execute("COMMIT")
                return 0.0, used, 0.0

            oldest_row = connection.execute(
                f"SELECT MIN(requested_at) FROM {self._TABLE} WHERE requested_at > ?",
                (cutoff,),
            ).fetchone()
            oldest = float(oldest_row[0]) if oldest_row and oldest_row[0] is not None else now
            wait_seconds = max(0.01, oldest + self.window_seconds - now)
            connection.execute("COMMIT")
            return wait_seconds, used, wait_seconds
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def acquire(
        self,
        *,
        on_wait: Callable[[float], None] | None = None,
    ) -> RequestBudgetAcquisition:
        """Wait until the next slot is available, then reserve it.

        The sleep is deliberately chunked so a long quiet period can still be
        interrupted by a process shutdown or a future cooperative worker
        cancellation mechanism.
        """

        waited = 0.0
        while True:
            wait_seconds, used, reset_seconds = self._try_acquire()
            if wait_seconds <= 0:
                return RequestBudgetAcquisition(
                    waited_seconds=waited,
                    used_requests=used,
                    remaining_requests=max(0, self.max_requests - used),
                    reset_seconds=reset_seconds,
                )
            if on_wait is not None:
                on_wait(wait_seconds)
            sleep_seconds = min(wait_seconds, _MAX_SLEEP_SECONDS)
            self._sleeper(sleep_seconds)
            waited += sleep_seconds

    def snapshot(self) -> RequestBudgetSnapshot:
        return read_request_budget_snapshot(
            self.path,
            requests_per_minute=self.requests_per_minute,
            window_seconds=self.window_seconds,
            max_requests=self.max_requests,
            clock=self._clock,
        )


def read_request_budget_snapshot(
    path: str | Path,
    *,
    requests_per_minute: int,
    window_seconds: int,
    max_requests: int,
    clock: Callable[[], float] = time.time,
) -> RequestBudgetSnapshot:
    """Read budget usage without creating or changing the database."""

    budget_path = Path(path)
    settings = {
        "path": budget_path,
        "requests_per_minute": max(1, int(requests_per_minute)),
        "window_seconds": max(1, int(window_seconds)),
        "max_requests": max(1, int(max_requests)),
    }
    if not budget_path.exists():
        return RequestBudgetSnapshot(**settings, used_requests=0, remaining_requests=settings["max_requests"], reset_seconds=0.0)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(budget_path), timeout=1.0)
        connection.execute("PRAGMA busy_timeout=1000")
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (RedditRequestBudget._TABLE,),
        ).fetchone()
        if table_exists is None:
            return RequestBudgetSnapshot(**settings, used_requests=0, remaining_requests=settings["max_requests"], reset_seconds=0.0)
        now = float(clock())
        cutoff = now - settings["window_seconds"]
        used = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {RedditRequestBudget._TABLE} WHERE requested_at > ?",
                (cutoff,),
            ).fetchone()[0]
        )
        oldest_row = connection.execute(
            f"SELECT MIN(requested_at) FROM {RedditRequestBudget._TABLE} WHERE requested_at > ?",
            (cutoff,),
        ).fetchone()
        reset_seconds = 0.0
        if used >= settings["max_requests"] and oldest_row and oldest_row[0] is not None:
            reset_seconds = max(0.0, float(oldest_row[0]) + settings["window_seconds"] - now)
        return RequestBudgetSnapshot(
            **settings,
            used_requests=used,
            remaining_requests=max(0, settings["max_requests"] - used),
            reset_seconds=reset_seconds,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return RequestBudgetSnapshot(
            **settings,
            used_requests=0,
            remaining_requests=settings["max_requests"],
            reset_seconds=0.0,
            readable=False,
            error=type(exc).__name__,
        )
    finally:
        if connection is not None:
            connection.close()


__all__ = [
    "DEFAULT_REQUESTS_PER_MINUTE",
    "DEFAULT_WINDOW_SECONDS",
    "OFFICIAL_REDDIT_REQUESTS_PER_MINUTE",
    "RedditRequestBudget",
    "RequestBudgetAcquisition",
    "RequestBudgetSettings",
    "RequestBudgetSnapshot",
    "read_request_budget_snapshot",
    "resolve_request_budget_settings",
]
