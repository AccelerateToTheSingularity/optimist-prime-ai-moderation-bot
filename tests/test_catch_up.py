"""Regression tests for bounded outage catch-up and retry tracking."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from catch_up import (
    catch_up_summary,
    clear_completed_retry_items,
    collect_catch_up_items,
    ensure_catch_up_state,
    remember_retryable_item,
)


class _Item:
    def __init__(self, item_id, created_utc, kind):
        self.id = item_id
        self.fullname = f"{'t3' if kind == 'posts' else 't1'}_{item_id}"
        self.created_utc = created_utc


class _PagedSubreddit:
    display_name = "example_community"

    def __init__(self, posts=None, comments=None):
        self.posts = list(posts or [])
        self.comments_data = list(comments or [])
        self.calls = []
        self.fail = False

    def _page(self, values, limit, params):
        self.calls.append((limit, dict(params or {})))
        if self.fail:
            raise RuntimeError("temporary Reddit listing failure")
        start = 0
        after = (params or {}).get("after")
        if after:
            matched = False
            for index, item in enumerate(values):
                if item.fullname == after:
                    start = index + 1
                    matched = True
                    break
            if not matched:
                return []
            return values[start : start + limit]
        before = (params or {}).get("before")
        if before:
            end = None
            for index, item in enumerate(values):
                if item.fullname == before:
                    end = index
                    break
            if end is None:
                return []
            return values[:end][:limit]
        return values[:limit]

    def new(self, limit=100, params=None):
        return self._page(self.posts, limit, params)

    def comments(self, limit=100, params=None):
        return self._page(self.comments_data, limit, params)


class TestCatchUp(unittest.TestCase):
    def test_initial_backlog_continues_across_cycles_without_losing_items(self):
        now = 1_800_000_000.0
        posts = [_Item(f"p{i}", now - i * 3600, "posts") for i in range(5)]
        subreddit = _PagedSubreddit(posts=posts)
        state = {}

        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            first = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)
            second = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)
            third = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)

        self.assertFalse(first.complete)
        self.assertTrue(second.complete or third.complete)
        self.assertEqual(
            {item.id for result in (first, second, third) for item in result.items},
            {"p0", "p1", "p2", "p3", "p4"},
        )
        self.assertTrue(any(params.get("after") for _, params in subreddit.calls))
        self.assertIsNone(state["catch_up"]["posts"]["backfill_after_fullname"])

    def test_new_items_and_gap_backfill_are_both_seen(self):
        now = 1_800_000_000.0
        posts = [_Item(f"p{i}", now - i * 3600, "posts") for i in range(5)]
        subreddit = _PagedSubreddit(posts=posts)
        state = {}

        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            initial = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)
            # New posts arrive while the first run is offline.  The old
            # frontier is no longer in the latest bounded page.
            subreddit.posts[0:0] = [
                _Item("new0", now + 10, "posts"),
                _Item("new1", now + 5, "posts"),
            ]
            gap = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)
            older = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)

        self.assertIn("new0", {item.id for item in gap.items})
        self.assertIn("new1", {item.id for item in gap.items})
        self.assertIn("p4", {item.id for item in older.items})
        self.assertTrue(initial.items)

    def test_large_new_activity_is_paginated_without_starving_old_backlog(self):
        now = 1_800_000_000.0
        posts = [_Item(f"p{i}", now - i * 3600, "posts") for i in range(5)]
        subreddit = _PagedSubreddit(posts=posts)
        state = {}

        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            first = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)
            subreddit.posts[0:0] = [
                _Item(f"new{i}", now + 20 - i, "posts") for i in range(5)
            ]
            results = [
                collect_catch_up_items(subreddit, "posts", state, 3, now_utc=now)
                for _ in range(4)
            ]

        seen = {item.id for result in [first, *results] for item in result.items}
        self.assertTrue({f"new{i}" for i in range(5)}.issubset(seen))
        self.assertTrue({"p3", "p4"}.issubset(seen))
        self.assertTrue(results[-1].complete)

    def test_retry_descriptor_contains_identity_only_and_clears_after_success(self):
        state = {}
        item = SimpleNamespace(id="c123", fullname="t1_c123", created_utc=123.0, body="private text")
        ensure_catch_up_state(state)
        remember_retryable_item(state, "comments", item, "temporary failure")
        descriptor = state["catch_up"]["comments"]["pending"][0]
        self.assertEqual(descriptor["id"], "c123")
        self.assertNotIn("body", descriptor)
        self.assertNotIn("private text", str(descriptor))
        clear_completed_retry_items(state, "comments", {"c123"})
        self.assertEqual(state["catch_up"]["comments"]["pending"], [])

    def test_pending_retries_do_not_break_the_per_run_item_cap(self):
        now = 1_800_000_000.0
        retry = _Item("retry", now - 1, "comments")
        comments = [_Item(f"c{i}", now - i - 10, "comments") for i in range(5)]
        subreddit = _PagedSubreddit(comments=comments)

        class _Reddit:
            def info(self, fullnames):
                self.requested = list(fullnames)
                return [retry]

        state = {
            "catch_up": {
                "comments": {
                    "pending": [{"id": "retry", "fullname": "t1_retry"}],
                }
            }
        }
        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            result = collect_catch_up_items(
                subreddit, "comments", state, 2, reddit=_Reddit(), now_utc=now
            )

        self.assertEqual(len(result.items), 2)
        self.assertEqual([item.id for item in result.items][0], "retry")

    def test_listing_failure_preserves_frontier_and_is_not_complete(self):
        state = {"catch_up": {"posts": {"cursor_fullname": "t3_old"}}}
        subreddit = _PagedSubreddit(posts=[])
        subreddit.fail = True
        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            result = collect_catch_up_items(subreddit, "posts", state, 3, now_utc=1_800_000_000.0)
        self.assertFalse(result.complete)
        self.assertEqual(state["catch_up"]["posts"]["cursor_fullname"], "t3_old")
        self.assertIn("listing failed", state["catch_up"]["posts"]["last_error"])

    def test_expired_cursor_fallback_records_an_incomplete_gap_and_rebases(self):
        now = 1_800_000_000.0
        comments = [_Item(f"c{i}", now - i * 60, "comments") for i in range(4)]
        subreddit = _PagedSubreddit(comments=comments)
        state = {
            "catch_up": {
                "comments": {
                    "newest_fullname": "t1_test001",
                    "newest_created_utc": now - 40 * 3600,
                    "cursor_fullname": "t1_older",
                    "cursor_created_utc": now - 40 * 3600,
                }
            }
        }
        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            result = collect_catch_up_items(subreddit, "comments", state, 3, now_utc=now)

        summary = catch_up_summary(state)["comments"]
        self.assertEqual({item.id for item in result.items}, {"c0", "c1", "c2"})
        self.assertFalse(result.complete)
        self.assertIsNone(result.error)
        self.assertIn("older than the configured processing window", result.incomplete_gap["reason"])
        self.assertFalse(summary["lastScanComplete"])
        self.assertIsNotNone(summary["incompleteGap"])
        self.assertIsNone(summary["lastError"])
        self.assertEqual(state["catch_up"]["comments"]["cursor_fullname"], "t1_c2")

    def test_stale_but_recent_cursor_is_paginated_back_to_anchor_across_cycles(self):
        now = 1_800_000_000.0
        comments = [_Item(f"c{i}", now - i * 60, "comments") for i in range(5)]
        subreddit = _PagedSubreddit(comments=comments)
        state = {
            "catch_up": {
                "comments": {
                    "newest_fullname": "t1_c4",
                    "newest_created_utc": now - 4 * 60,
                    "cursor_fullname": "t1_c4",
                    "cursor_created_utc": now - 4 * 60,
                }
            }
        }
        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            results = [
                collect_catch_up_items(subreddit, "comments", state, 2, now_utc=now)
                for _ in range(4)
            ]

        seen = {item.id for result in results for item in result.items}
        self.assertEqual(seen, {f"c{i}" for i in range(5)})
        self.assertTrue(results[-1].complete)
        self.assertIsNone(results[-1].incomplete_gap)

    def test_recent_cursor_that_reddit_cannot_reach_is_not_reported_complete(self):
        now = 1_800_000_000.0
        comments = [_Item("new", now - 60, "comments"), _Item("next", now - 120, "comments")]
        subreddit = _PagedSubreddit(comments=comments)
        state = {
            "catch_up": {
                "comments": {
                    "newest_fullname": "t1_test0000002",
                    "newest_created_utc": now - 3600,
                    "cursor_fullname": "t1_test0000002",
                    "cursor_created_utc": now - 3600,
                }
            }
        }
        with patch("catch_up.config.MAX_AGE_HOURS", 36):
            result = collect_catch_up_items(subreddit, "comments", state, 4, now_utc=now)

        self.assertFalse(result.complete)
        self.assertIsNone(result.error)
        self.assertEqual({item.id for item in result.items}, {"new", "next"})
        self.assertIn("before the saved position was reached", result.incomplete_gap["reason"])
        self.assertIsNotNone(catch_up_summary(state)["comments"]["incompleteGap"])

    def test_summary_is_safe_for_diagnostics(self):
        state = {}
        ensure_catch_up_state(state)
        summary = catch_up_summary(state)
        self.assertEqual(summary["posts"]["pendingRetryCount"], 0)
        self.assertNotIn("body", str(summary))


if __name__ == "__main__":
    unittest.main()
