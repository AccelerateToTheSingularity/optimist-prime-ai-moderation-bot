"""Tests for the shared cross-process Reddit request budget."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reddit_request_budget import (
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_WINDOW_SECONDS,
    RedditRequestBudget,
    resolve_request_budget_settings,
)
from reddit_observability import observable_requestor_settings


class TestRedditRequestBudget(unittest.TestCase):
    def test_requestor_settings_cannot_raise_the_safety_ceiling(self):
        settings = observable_requestor_settings(
            9,
            request_budget_path="C:/temp/reddit-budget.sqlite3",
            request_budget_limit=9999,
            request_budget_window_seconds=600,
            request_budget_per_minute=999,
        )

        self.assertEqual(settings["requestor_kwargs"]["request_budget_per_minute"], 90)
        self.assertEqual(settings["requestor_kwargs"]["request_budget_limit"], 900)

    def test_settings_keep_a_ten_percent_margin_and_follow_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "private" / "bot_state.json"
            settings = resolve_request_budget_settings(
                {
                    "REDDIT_REQUEST_BUDGET_PER_MINUTE": "999",
                    "BOT_RUNTIME_STATE_FILE": str(state_path),
                },
                default_root=tmp,
            )

            self.assertEqual(settings.official_requests_per_minute, 100)
            self.assertEqual(settings.requests_per_minute, DEFAULT_REQUESTS_PER_MINUTE)
            self.assertEqual(settings.window_seconds, DEFAULT_WINDOW_SECONDS)
            self.assertEqual(settings.max_requests, 900)
            self.assertEqual(
                settings.path,
                state_path.with_name("reddit_request_budget.sqlite3"),
            )

    def test_budget_expires_old_attempts_from_the_rolling_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [1_000.0]
            budget = RedditRequestBudget(
                Path(tmp) / "budget.sqlite3",
                max_requests=2,
                window_seconds=60,
                requests_per_minute=2,
                clock=lambda: now[0],
            )

            budget.acquire()
            budget.acquire()
            self.assertEqual(budget.snapshot().used_requests, 2)

            now[0] = 1_060.1
            acquisition = budget.acquire()

            self.assertEqual(acquisition.waited_seconds, 0)
            self.assertEqual(acquisition.used_requests, 1)
            self.assertEqual(budget.snapshot().used_requests, 1)

    def test_full_budget_waits_in_interruptible_chunks_then_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [2_000.0]
            sleeps: list[float] = []

            def advance(seconds: float) -> None:
                sleeps.append(seconds)
                now[0] += seconds

            budget = RedditRequestBudget(
                Path(tmp) / "budget.sqlite3",
                max_requests=1,
                window_seconds=60,
                requests_per_minute=1,
                clock=lambda: now[0],
                sleeper=advance,
            )
            budget.acquire()
            waits: list[float] = []

            acquisition = budget.acquire(on_wait=waits.append)

            self.assertEqual(sleeps, [30.0, 30.0])
            self.assertEqual(waits, [60.0, 30.0])
            self.assertEqual(acquisition.waited_seconds, 60.0)
            self.assertEqual(acquisition.used_requests, 1)


if __name__ == "__main__":
    unittest.main()
