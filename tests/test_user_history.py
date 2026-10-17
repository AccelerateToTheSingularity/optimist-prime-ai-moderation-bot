"""Tests for the shared cross-Reddit history collector."""

import unittest
from types import SimpleNamespace

from user_history import collect_user_history


class _Listing:
    def __init__(self, values):
        self.values = values
        self.requested_limits = []

    def new(self, limit=None):
        self.requested_limits.append(limit)
        return self.values if limit is None else self.values[:limit]


class _Reddit:
    def __init__(self, comments, posts):
        self._redditor = SimpleNamespace(
            comments=_Listing(comments),
            submissions=_Listing(posts),
        )

    def redditor(self, username):
        return self._redditor


def _comment(item_id, subreddit, created, body="comment"):
    return SimpleNamespace(
        id=item_id,
        body=body,
        score=2,
        subreddit=SimpleNamespace(display_name=subreddit),
        created_utc=created,
        permalink=f"/r/{subreddit}/comments/{item_id}/x/",
    )


def _post(item_id, subreddit, created, title="title", body="body"):
    return SimpleNamespace(
        id=item_id,
        title=title,
        selftext=body,
        score=3,
        subreddit=SimpleNamespace(display_name=subreddit),
        created_utc=created,
        permalink=f"/r/{subreddit}/comments/{item_id}/x/",
    )


class TestUserHistory(unittest.TestCase):
    def test_collects_comments_and_posts_with_type_and_total_limits(self):
        now = 2_000_000_000.0
        reddit = _Reddit(
            [_comment("c1", "one", now - 10), _comment("c2", "two", now - 20)],
            [_post("p1", "three", now - 1), _post("p2", "four", now - 2)],
        )
        history = collect_user_history(
            reddit,
            "example_user",
            lookback_days=365,
            max_comments=2,
            max_posts=2,
            max_items=3,
            now_utc=now,
        )
        self.assertEqual([item.item_id for item in history.items], ["c1", "c2", "p1"])
        self.assertEqual(len(history.comments), 2)
        self.assertEqual(len(history.posts), 1)
        self.assertTrue(history.truncated)

    def test_posts_fill_the_limit_before_comments_and_comments_fill_the_remainder(self):
        now = 2_000_000_000.0
        reddit = _Reddit(
            [_comment(f"c{i}", "one", now - 10 - i) for i in range(5)],
            [_post(f"p{i}", "two", now - i) for i in range(3)],
        )
        history = collect_user_history(
            reddit,
            "example_user",
            max_comments=250,
            max_posts=250,
            max_items=4,
            prefer_posts=True,
            now_utc=now,
        )

        self.assertEqual([item.kind for item in history.items], ["post", "post", "post", "comment"])
        self.assertEqual([item.item_id for item in history.items], ["p0", "p1", "p2", "c0"])
        self.assertEqual(reddit._redditor.submissions.requested_limits, [4])
        self.assertEqual(reddit._redditor.comments.requested_limits, [1])

    def test_full_post_sample_skips_comment_listing(self):
        now = 2_000_000_000.0
        reddit = _Reddit(
            [_comment("c1", "one", now - 10)],
            [_post(f"p{i}", "two", now - i) for i in range(3)],
        )
        history = collect_user_history(
            reddit,
            "example_user",
            max_comments=250,
            max_posts=250,
            max_items=3,
            prefer_posts=True,
            now_utc=now,
        )

        self.assertEqual([item.item_id for item in history.items], ["p0", "p1", "p2"])
        self.assertEqual(reddit._redditor.submissions.requested_limits, [3])
        self.assertEqual(reddit._redditor.comments.requested_limits, [])

    def test_comment_fallback_runs_if_post_listing_fails(self):
        now = 2_000_000_000.0

        class FailedListing:
            def new(self, limit=None):
                raise RuntimeError("post listing unavailable")

        reddit = _Reddit([_comment("c1", "one", now - 1)], [])
        reddit._redditor.submissions = FailedListing()
        history = collect_user_history(
            reddit,
            "example_user",
            max_comments=250,
            max_posts=250,
            max_items=2,
            prefer_posts=True,
            now_utc=now,
        )

        self.assertEqual([item.item_id for item in history.items], ["c1"])
        self.assertTrue(any(error.startswith("posts:") for error in history.errors))

    def test_deleted_and_old_items_are_excluded(self):
        now = 2_000_000_000.0
        reddit = _Reddit(
            [_comment("deleted", "one", now - 10, "[deleted]"), _comment("old", "one", now - 200000)],
            [],
        )
        history = collect_user_history(reddit, "example_user", lookback_days=1, now_utc=now)
        self.assertEqual(history.items, ())

    def test_listing_subreddit_name_avoids_lazy_about_request(self):
        now = 2_000_000_000.0

        class ListingComment:
            id = "c1"
            body = "useful comment"
            score = 2
            created_utc = now - 10
            permalink = "/r/one/comments/c1/x/"
            subreddit_name_prefixed = "r/one"

            @property
            def subreddit(self):
                raise AssertionError("history should not hydrate subreddit metadata")

        reddit = _Reddit([ListingComment()], [])
        history = collect_user_history(reddit, "example_user", now_utc=now)
        self.assertEqual(history.items[0].subreddit, "one")

    def test_topic_filter_can_reach_matching_comments_beyond_irrelevant_items(self):
        now = 2_000_000_000.0
        comments = [
            _comment("off1", "unrelated", now - 1),
            _comment("off2", "unrelated", now - 2),
            _comment("on1", "example_community", now - 3),
            _comment("off3", "unrelated", now - 4),
            _comment("on2", "example_other_community", now - 5),
        ]
        history = collect_user_history(
            _Reddit(comments, []),
            "example_user",
            max_comments=2,
            max_posts=0,
            max_items=None,
            max_scan_comments=5,
            max_scan_posts=0,
            allowed_subreddits=frozenset({"example_community", "example_other_community"}),
            now_utc=now,
        )
        self.assertEqual([item.item_id for item in history.comments], ["on1", "on2"])
        self.assertTrue(history.comments_cap_reached)
        self.assertFalse(history.comments_scan_limited)

    def test_scan_safety_cap_is_reported_when_topic_filter_finds_too_few_items(self):
        now = 2_000_000_000.0
        history = collect_user_history(
            _Reddit(
                [_comment(f"off{i}", "unrelated", now - i) for i in range(1, 6)],
                [],
            ),
            "example_user",
            max_comments=10,
            max_posts=0,
            max_items=None,
            max_scan_comments=4,
            max_scan_posts=0,
            allowed_subreddits=frozenset({"example_community"}),
            now_utc=now,
        )
        self.assertEqual(history.items, ())
        self.assertTrue(history.comments_scan_limited)
        self.assertTrue(history.to_dict()["comments_scan_limited"])

    def test_reddit_listing_ceiling_is_reported_as_partial_scan(self):
        now = 2_000_000_000.0
        history = collect_user_history(
            _Reddit(
                [_comment(f"off{i}", "unrelated", now - i) for i in range(1000)],
                [],
            ),
            "example_user",
            max_comments=400,
            max_posts=0,
            max_items=None,
            max_scan_comments=1000,
            max_scan_posts=0,
            allowed_subreddits=frozenset({"example_community"}),
            now_utc=now,
        )
        self.assertEqual(history.comments_seen, 1000)
        self.assertEqual(history.items, ())
        self.assertTrue(history.comments_scan_limited)

    def test_subreddit_fallback_does_not_touch_a_lazy_attribute(self):
        now = 2_000_000_000.0

        class ListingComment:
            id = "c1"
            body = "useful comment"
            score = 2
            created_utc = now - 10
            permalink = "/r/example_community/comments/c1/x/"

            @property
            def subreddit(self):
                raise AssertionError("history must not trigger a lazy subreddit fetch")

        reddit = _Reddit([ListingComment()], [])
        history = collect_user_history(
            reddit,
            "example_user",
            max_comments=1,
            max_posts=0,
            max_items=None,
            max_scan_comments=1,
            max_scan_posts=0,
            allowed_subreddits=frozenset({"example_community"}),
            now_utc=now,
        )
        self.assertEqual(history.items, ())


if __name__ == "__main__":
    unittest.main()
