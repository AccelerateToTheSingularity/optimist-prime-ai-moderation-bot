"""Local, durable de-duplication state for configured history reviews.

Only Reddit item IDs, usernames, timestamps, and review status are stored here.
Post/comment text remains in memory for the one review that needs it.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from state_store import runtime_state_path


class _ClosingSQLiteConnection(sqlite3.Connection):
    """Make ``with connection`` close the handle as well as commit/rollback."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def item_fullname(item: Any) -> str:
    """Return a stable typed Reddit item ID for a history item."""
    kind = str(getattr(item, "kind", "") or "").casefold()
    item_id = str(getattr(item, "item_id", "") or "").strip()
    prefix = "t3_" if kind == "post" else "t1_" if kind == "comment" else ""
    if not prefix or not item_id:
        return ""
    return prefix + item_id


def _profile_key(username: str) -> str:
    return str(username or "").strip().casefold()


def _runtime_ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser()
    else:
        configured = os.environ.get("BOT_RUNTIME_HISTORY_REVIEW_LEDGER_FILE", "").strip()
        if configured:
            resolved = Path(os.path.expandvars(os.path.expanduser(configured)))
        else:
            state_path = runtime_state_path()
            resolved = state_path.with_name("history_review_ledger.sqlite3")
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


@dataclass(frozen=True)
class ReviewAttempt:
    review_id: str
    allowed: bool = True
    reason: str = ""
    accepted_fullnames: frozenset[str] = frozenset()


class HistoryReviewLedger:
    """SQLite-backed deduplication ledger for requested history reviews."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock=None,
        read_only: bool = False,
    ) -> None:
        import time

        self.path = _runtime_ledger_path(path)
        self._clock = clock or time.time
        self.read_only = bool(read_only)
        if not self.read_only:
            self._initialise()

    def _connect(self, *, create: bool = False) -> sqlite3.Connection:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.read_only and self.path.exists():
            connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro", uri=True, timeout=5.0,
                factory=_ClosingSQLiteConnection,
            )
        else:
            connection = sqlite3.connect(
                str(self.path), timeout=30.0, factory=_ClosingSQLiteConnection
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialise(self) -> None:
        with self._connect(create=True) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reviewed_items (
                    profile_key TEXT NOT NULL,
                    username TEXT NOT NULL,
                    fullname TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    submitted_at REAL NOT NULL,
                    source TEXT NOT NULL,
                    review_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (profile_key, fullname)
                );
                CREATE INDEX IF NOT EXISTS idx_reviewed_items_submitted
                    ON reviewed_items(submitted_at);
                CREATE TABLE IF NOT EXISTS review_attempts (
                    review_id TEXT PRIMARY KEY,
                    profile_key TEXT NOT NULL,
                    username TEXT NOT NULL,
                    source TEXT NOT NULL,
                    attempted_at REAL NOT NULL,
                    finished_at REAL,
                    status TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    post_count INTEGER NOT NULL,
                    comment_count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_attempts_profile_time
                    ON review_attempts(profile_key, attempted_at);
                CREATE TABLE IF NOT EXISTS ledger_counters (
                    metric TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                """
            )
    def now(self) -> float:
        return float(self._clock())

    def seen_fullnames(self, username: str) -> set[str]:
        if not self.path.exists():
            return set()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT fullname FROM reviewed_items WHERE profile_key=?",
                (_profile_key(username),),
            ).fetchall()
        return {str(row["fullname"]) for row in rows}

    def start_review(
        self,
        username: str,
        items: Iterable[Any],
        *,
        source: str,
        now: float | None = None,
    ) -> ReviewAttempt:
        """Persist selected IDs before sending them to Jev."""
        if self.read_only:
            raise RuntimeError("cannot start a review with a read-only ledger")
        timestamp = self.now() if now is None else float(now)
        profile = _profile_key(username)
        selected = list(dict.fromkeys(
            (item_fullname(item), str(getattr(item, "kind", "")))
            for item in items
            if item_fullname(item)
        ))
        review_id = uuid.uuid4().hex
        with self._connect(create=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = {
                str(row["fullname"])
                for row in connection.execute(
                    "SELECT fullname FROM reviewed_items WHERE profile_key=?",
                    (profile,),
                ).fetchall()
            }
            selected = [row for row in selected if row[0] not in existing]
            if not selected:
                connection.rollback()
                return ReviewAttempt("", False, "items_already_submitted")
            posts = sum(kind == "post" for _fullname, kind in selected)
            comments = sum(kind == "comment" for _fullname, kind in selected)
            connection.execute(
                "INSERT INTO review_attempts(review_id, profile_key, username, source, attempted_at, status, item_count, post_count, comment_count) VALUES(?,?,?,?,?,?,?,?,?)",
                (review_id, profile, str(username), str(source), timestamp, "attempted", len(selected), posts, comments),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO reviewed_items(profile_key, username, fullname, kind, submitted_at, source, review_id, status) VALUES(?,?,?,?,?,?,?,?)",
                [
                    (profile, str(username), fullname, kind, timestamp, str(source), review_id, "attempted")
                    for fullname, kind in selected
                ],
            )
            connection.commit()
        return ReviewAttempt(
            review_id,
            accepted_fullnames=frozenset(fullname for fullname, _kind in selected),
        )

    def finish_review(self, review_id: str, *, status: str, now: float | None = None) -> None:
        if self.read_only or not review_id:
            return
        timestamp = self.now() if now is None else float(now)
        with self._connect(create=True) as connection:
            prior = connection.execute(
                "SELECT status FROM review_attempts WHERE review_id=?",
                (str(review_id),),
            ).fetchone()
            if prior is None or str(prior["status"]) != "attempted":
                return
            connection.execute(
                "UPDATE review_attempts SET finished_at=?, status=? WHERE review_id=?",
                (timestamp, str(status), str(review_id)),
            )
            connection.execute(
                "UPDATE reviewed_items SET status=? WHERE review_id=?",
                (str(status), str(review_id)),
            )
            if status == "complete":
                self._increment_counter(connection, "reviews_completed", 1, timestamp)
            elif status == "failed":
                self._increment_counter(connection, "failed_jev_attempts", 1, timestamp)
            elif status == "incomplete":
                self._increment_counter(connection, "incomplete_jev_attempts", 1, timestamp)

    @staticmethod
    def _increment_counter(connection: sqlite3.Connection, metric: str, amount: int, now: float) -> None:
        connection.execute(
            "INSERT INTO ledger_counters(metric, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(metric) DO UPDATE SET value=value+excluded.value, updated_at=excluded.updated_at",
            (str(metric), int(amount), float(now)),
        )

    def increment_metric(self, metric: str, amount: int = 1, *, now: float | None = None) -> None:
        if self.read_only or not int(amount):
            return
        timestamp = self.now() if now is None else float(now)
        with self._connect(create=True) as connection:
            self._increment_counter(connection, metric, int(amount), timestamp)

    def metrics(self) -> dict[str, Any]:
        result = {
            "ledger_available": self.path.exists(),
            "reviews_completed": 0,
            "failed_jev_attempts": 0,
            "incomplete_jev_attempts": 0,
            "skipped_duplicate_items": 0,
            "skipped_no_new_history": 0,
            "history_gaps": 0,
        }
        if not self.path.exists():
            return result
        with self._connect() as connection:
            counters = connection.execute(
                "SELECT metric, value FROM ledger_counters"
            ).fetchall()
            for row in counters:
                name = str(row["metric"])
                if name in result:
                    result[name] = int(row["value"])
        return result


def diagnostic_ledger_metrics(path: str | Path | None = None) -> dict[str, Any]:
    """Read local ledger counts without creating a missing database."""
    return HistoryReviewLedger(path, read_only=True).metrics()
