"""Unit tests for the local read-only Reddit comment feed."""

from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from comment_feed import (
    CommentFeedService,
    FeedConfigurationError,
    FeedIdentityError,
    create_feed_server,
    create_read_only_reddit,
    main as comment_feed_main,
    normalize_comment,
    parse_args,
    render_feed_page,
)
from reddit_observability import ObservableRequestor


def _comment(
    comment_id: str,
    *,
    author: str | None = "example_user",
    created: float = 1_000.0,
    score: int = 1,
    body: str = "hello",
    title: str = "A post",
):
    return SimpleNamespace(
        id=comment_id,
        author=None if author is None else SimpleNamespace(name=author),
        body=body,
        created_utc=created,
        score=score,
        permalink=f"/r/example_community/comments/{comment_id}/post/",
        subreddit=SimpleNamespace(display_name="example_community"),
        link_title=title,
    )


class _FakeSubreddit:
    def __init__(self, comments):
        self.comments_to_return = comments
        self.limits = []
        self.error = None

    def comments(self, limit=None):
        self.limits.append(limit)
        if self.error:
            raise self.error
        return list(self.comments_to_return)


class _FakeReddit:
    def __init__(self, comments):
        self.fake_subreddit = _FakeSubreddit(comments)
        self.read_only = True

    def subreddit(self, name):
        self.subreddit_name = name
        return self.fake_subreddit


class TestCommentFeedNormalization(unittest.TestCase):
    def test_normalizes_deleted_author_and_reddit_permalink(self):
        item = normalize_comment(_comment("c1", author=None, body="<not html>"))
        self.assertIsNotNone(item)
        self.assertEqual(item.author, "[deleted]")
        self.assertEqual(item.body, "<not html>")
        self.assertEqual(
            item.permalink,
            "https://www.reddit.com/r/example_community/comments/c1/post/",
        )

    def test_rejects_comment_without_an_id_and_bounds_large_body(self):
        self.assertIsNone(normalize_comment(SimpleNamespace(id="")))
        item = normalize_comment(_comment("c2", body="x" * 20), max_body_chars=10)
        self.assertEqual(item.body, "x" * 10)
        self.assertTrue(item.body_truncated)


class TestCommentFeedService(unittest.TestCase):
    def test_refresh_deduplicates_and_keeps_newest_first_with_bounded_cache(self):
        reddit = _FakeReddit(
            [_comment("old", created=100), _comment("new", created=300), _comment("middle", created=200)]
        )
        service = CommentFeedService(reddit, "example_community", max_items=2)
        self.assertTrue(service.refresh_now())
        snapshot = service.snapshot()
        self.assertEqual([item["id"] for item in snapshot["comments"]], ["new", "middle"])
        self.assertEqual(reddit.fake_subreddit.limits, [2])

        reddit.fake_subreddit.comments_to_return = [
            _comment("new", created=300, body="updated"),
            _comment("newer", created=400),
        ]
        self.assertTrue(service.refresh_now())
        snapshot = service.snapshot()
        self.assertEqual([item["id"] for item in snapshot["comments"]], ["newer", "new"])
        self.assertEqual(snapshot["comments"][1]["body"], "updated")

    def test_failed_refresh_keeps_cache_and_remains_retryable(self):
        reddit = _FakeReddit([_comment("c1")])
        service = CommentFeedService(reddit, "example_community")
        self.assertTrue(service.refresh_now())
        reddit.fake_subreddit.error = RuntimeError("temporary Reddit failure")
        self.assertFalse(service.refresh_now())
        failed = service.snapshot()
        self.assertEqual(failed["status"]["state"], "error")
        self.assertEqual([item["id"] for item in failed["comments"]], ["c1"])
        self.assertEqual(failed["status"]["failedRefreshCount"], 1)

        reddit.fake_subreddit.error = None
        reddit.fake_subreddit.comments_to_return = [_comment("c2", created=2_000)]
        self.assertTrue(service.refresh_now())
        recovered = service.snapshot()
        self.assertEqual(recovered["status"]["state"], "connected")
        self.assertEqual(recovered["status"]["refreshCount"], 2)

    def test_pause_and_resume_control_automatic_state_without_writing_reddit(self):
        service = CommentFeedService(_FakeReddit([]), "example_community")
        self.assertTrue(service.pause())
        self.assertEqual(service.snapshot()["status"]["state"], "paused")
        self.assertTrue(service.resume())
        self.assertEqual(service.snapshot()["status"]["state"], "starting")
        service.stop()
        self.assertEqual(service.snapshot()["status"]["state"], "stopped")


class TestCommentFeedAuthentication(unittest.TestCase):
    def test_subreddit_is_not_assumed(self):
        self.assertIsNone(parse_args([]).subreddit)
        with self.assertRaisesRegex(FeedConfigurationError, "BOT_SUBREDDIT or pass --subreddit"):
            CommentFeedService(_FakeReddit([]), "")

    def test_feed_is_disabled_without_explicit_enable_flag(self):
        with patch("comment_feed.create_read_only_reddit") as create_reddit:
            self.assertEqual(comment_feed_main(["--once", "--no-browser"]), 0)
        create_reddit.assert_not_called()

    def test_enabled_feed_requires_an_explicit_subreddit_before_authentication(self):
        with patch("comment_feed.clear_inherited_runtime_environment"), \
             patch("comment_feed.load_local_env"), \
             patch.dict("os.environ", {}, clear=True), \
             patch("comment_feed.create_read_only_reddit") as create_reddit:
            self.assertEqual(comment_feed_main(["--enable-feed", "--once"]), 2)
        create_reddit.assert_not_called()

    def test_authentication_checks_account_before_setting_read_only(self):
        captured = {}

        class FakePrawReddit:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.read_only = False
                self.user = SimpleNamespace(me=lambda: SimpleNamespace(name="example_bot"))

        env = {
            "REDDIT_CLIENT_ID": "client-id",
            "REDDIT_CLIENT_SECRET": "client-secret",
            "REDDIT_REFRESH_TOKEN": "refresh-token",
            "REDDIT_APP_NAME": "Optimist Prime",
            "REDDIT_DEVELOPER_USERNAME": "example_bot",
            "BOT_EXPECTED_USERNAME": "example_bot",
        }
        reddit, account = create_read_only_reddit(
            environment=env,
            praw_module=SimpleNamespace(Reddit=FakePrawReddit),
            timeout_seconds=7,
        )
        self.assertEqual(account, "example_bot")
        self.assertTrue(reddit.read_only)
        self.assertEqual(captured["refresh_token"], "refresh-token")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(captured["requestor_kwargs"]["timeout"], 7)
        self.assertEqual(captured["requestor_kwargs"]["request_budget_limit"], 900)
        self.assertEqual(
            captured["requestor_kwargs"]["request_budget_window_seconds"], 600
        )
        self.assertEqual(
            captured["requestor_kwargs"]["request_budget_per_minute"], 90
        )
        self.assertIs(captured["requestor_class"], ObservableRequestor)
        self.assertNotIn("password", captured)

    def test_authentication_rejects_the_wrong_account(self):
        class FakePrawReddit:
            def __init__(self, **kwargs):
                self.read_only = False
                self.user = SimpleNamespace(me=lambda: SimpleNamespace(name="someone_else"))

        env = {
            "REDDIT_CLIENT_ID": "client-id",
            "REDDIT_CLIENT_SECRET": "client-secret",
            "REDDIT_REFRESH_TOKEN": "refresh-token",
            "REDDIT_DEVELOPER_USERNAME": "example_developer",
            "BOT_EXPECTED_USERNAME": "example_bot",
        }
        with self.assertRaises(FeedIdentityError):
            create_read_only_reddit(
                environment=env,
                praw_module=SimpleNamespace(Reddit=FakePrawReddit),
            )

    def test_authentication_requires_a_developer_username_for_the_user_agent(self):
        env = {
            "REDDIT_CLIENT_ID": "client-id",
            "REDDIT_CLIENT_SECRET": "client-secret",
            "REDDIT_REFRESH_TOKEN": "refresh-token",
            "BOT_EXPECTED_USERNAME": "example_bot",
        }
        with self.assertRaisesRegex(FeedConfigurationError, "REDDIT_DEVELOPER_USERNAME"):
            create_read_only_reddit(
                environment=env,
                praw_module=SimpleNamespace(Reddit=object),
            )


class TestCommentFeedWebInterface(unittest.TestCase):
    def test_page_uses_dom_text_rendering_and_safe_token_embedding(self):
        page = render_feed_page("</script><img src=x>")
        self.assertIn("textContent", page)
        self.assertIn("document.createElement", page)
        self.assertNotIn("innerHTML", page)
        self.assertNotIn("</script><img", page)

    def test_reading_is_public_but_controls_require_local_session_token(self):
        service = CommentFeedService(_FakeReddit([_comment("c1")]), "example_community")
        server, token = create_feed_server(service, port=0, session_token="test-token")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(f"{base}/api/feed") as response:
                feed = json.loads(response.read().decode("utf-8"))
            self.assertTrue(feed["ok"])
            self.assertEqual(feed["comments"], [])

            request = Request(
                f"{base}/api/actions",
                data=json.dumps({"action": "feed.refresh"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Origin": base},
                method="POST",
            )
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 403)

            request.add_header("X-Optimist-Prime-Feed-Token", token)
            with urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertTrue(result["actionSucceeded"])
            self.assertEqual(result["status"]["cachedCount"], 1)

            rebound = Request(
                f"{base}/",
                headers={"Host": f"attacker.example:{server.server_port}"},
            )
            with self.assertRaises(HTTPError) as context:
                urlopen(rebound)
            self.assertEqual(context.exception.code, 403)
        finally:
            service.stop()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
