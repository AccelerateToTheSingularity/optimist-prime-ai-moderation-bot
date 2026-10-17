"""Persistence and deduplication tests for history reviews."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from history_review_collection import collect_unseen_user_history
from history_review_ledger import (
    HistoryReviewLedger,
    diagnostic_ledger_metrics,
)
from user_history import HistoryItem


class _Listing:
    def __init__(self, items):
        self.items = list(items)
        self.requested_limits = []

    def new(self, limit=None):
        self.requested_limits.append(limit)
        return iter(self.items if limit is None else self.items[:limit])


class _Reddit:
    def __init__(self, comments, posts):
        self.user = SimpleNamespace(
            comments=_Listing(comments),
            submissions=_Listing(posts),
        )

    def redditor(self, username):
        return self.user


def _raw_item(item_id, kind, created_utc):
    return SimpleNamespace(
        id=item_id,
        fullname=("t3_" if kind == "post" else "t1_") + item_id,
        body=f"comment {item_id}" if kind == "comment" else None,
        title=f"post {item_id}" if kind == "post" else None,
        selftext="post body" if kind == "post" else None,
        score=1,
        created_utc=created_utc,
        permalink=f"/r/example_community/comments/{item_id}/x/",
        subreddit_name_prefixed="r/example_community",
    )


def _history_item(item_id, kind="comment", created_utc=2_000_000_000.0):
    return HistoryItem(
        item_id=item_id,
        kind=kind,
        subreddit="example_community",
        created_utc=created_utc,
        score=1,
        permalink=f"https://www.reddit.com/r/example_community/comments/{item_id}/x/",
        text=f"evidence {item_id}",
    )


class TestHistoryReviewLedger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "history.sqlite3"

    def test_deduplication_survives_restart_and_prefers_posts_for_each_250_item_batch(self):
        now = 2_000_000_000.0
        posts = [_raw_item(f"p{i}", "post", now - i) for i in range(300)]
        comments = [_raw_item(f"c{i}", "comment", now - 1000 - i) for i in range(260)]
        reddit = _Reddit(comments, posts)
        ledger = HistoryReviewLedger(self.path)
        options = dict(
            lookback_days=365,
            max_comments=250,
            max_posts=250,
            max_items=250,
            max_scan_comments=1000,
            max_scan_posts=1000,
            allowed_subreddits=frozenset({"example_community"}),
            prefer_posts=True,
            now_utc=now,
        )

        first = collect_unseen_user_history(
            reddit, "example_user", ledger=ledger, **options
        ).history
        self.assertEqual(len(first.items), 250)
        self.assertTrue(all(item.kind == "post" for item in first.items))
        first_attempt = ledger.start_review("example_user", first.items, source="reported")
        self.assertTrue(first_attempt.allowed)
        ledger.finish_review(first_attempt.review_id, status="complete")

        # A new ledger object represents the bot restarting between reports.
        ledger_after_restart = HistoryReviewLedger(self.path)
        second = collect_unseen_user_history(
            reddit,
            "example_user",
            ledger=ledger_after_restart,
            **options,
        ).history
        self.assertEqual(len(second.items), 250)
        self.assertEqual(sum(item.kind == "post" for item in second.items), 50)
        self.assertEqual(sum(item.kind == "comment" for item in second.items), 200)
        self.assertEqual(len({item.item_id for item in (*first.items, *second.items)}), 500)
        second_attempt = ledger_after_restart.start_review(
            "example_user", second.items, source="moderator"
        )
        self.assertTrue(second_attempt.allowed)

        third = collect_unseen_user_history(
            reddit,
            "example_user",
            ledger=ledger_after_restart,
            **options,
        ).history
        self.assertEqual([item.item_id for item in third.items], [f"c{i}" for i in range(200, 260)])
        third_attempt = ledger_after_restart.start_review(
            "example_user", third.items, source="reported"
        )
        self.assertTrue(third_attempt.allowed)

        final = collect_unseen_user_history(
            reddit,
            "example_user",
            ledger=ledger_after_restart,
            **options,
        ).history
        self.assertEqual(final.items, ())
        self.assertEqual(final.duplicate_items_skipped, 560)
        self.assertFalse(ledger_after_restart.start_review("example_user", final.items, source="reported").allowed)

    def test_duplicate_items_inside_one_input_are_sent_once(self):
        ledger = HistoryReviewLedger(self.path)
        duplicate = _history_item("same")
        attempt = ledger.start_review("example_user", [duplicate, duplicate], source="reported")
        self.assertTrue(attempt.allowed)
        self.assertEqual(attempt.accepted_fullnames, frozenset({"t1_same"}))

    def test_read_only_diagnostics_do_not_create_a_missing_database(self):
        metrics = diagnostic_ledger_metrics(self.path)
        self.assertFalse(metrics["ledger_available"])
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
