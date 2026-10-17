"""Regression tests for per-authority conversational reply limits."""

from datetime import date, datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reply_handler
import summon_handler


def _item(item_id: str, author: str, body: str = "Hello bot"):
    subreddit = SimpleNamespace(display_name="example_community")
    return SimpleNamespace(
        id=item_id,
        created_utc=datetime.now(timezone.utc).timestamp(),
        subreddit=subreddit,
        body=body,
        author=SimpleNamespace(name=author),
        submission=SimpleNamespace(id="submission-1"),
    )


class TestReplyLimits(unittest.TestCase):
    def setUp(self):
        subreddit_patch = patch.object(
            reply_handler.config, "SUBREDDIT", "example_community"
        )
        subreddit_patch.start()
        self.addCleanup(subreddit_patch.stop)

    def test_moderator_bypasses_daily_and_per_run_caps(self):
        ordinary = _item("ordinary-1", "ordinary_user")
        moderator = _item("moderator-1", "example_mod")
        reddit = SimpleNamespace(
            inbox=SimpleNamespace(comment_replies=lambda limit: [ordinary, moderator])
        )
        state = {
            "daily_reset_date": date.today().isoformat(),
            "daily_replies": 50,
            "daily_non_moderator_replies": 50,
            "recent_user_replies": {},
            "replied_to_comments": [],
        }

        with (
            patch.object(reply_handler.config, "MAX_REPLIES_PER_DAY", 50),
            patch.object(reply_handler.config, "MAX_REPLIES_PER_RUN", 1),
            patch.object(reply_handler.config, "MODERATOR_REPLIES_BYPASS_LIMITS", False),
            patch.object(
                reply_handler,
                "is_current_moderator",
                side_effect=lambda name, state, subreddit: name == "example_mod",
            ),
            patch.object(
                reply_handler,
                "route_public_interaction",
                side_effect=AssertionError("verified moderators must use process_moderator_turn"),
            ),
            patch.object(
                reply_handler,
                "process_moderator_turn",
                return_value={
                    "kind": "moderator_turn",
                    "route": "conversation",
                    "handled": False,
                    "conversation_response": "A useful answer",
                    "token_info": {"total_tokens": 4, "cost": 0.0},
                },
            ) as turn,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                return_value=("A useful answer", {"total_tokens": 3, "cost": 0.0}),
            ) as chat,
            patch.object(reply_handler, "execute_engagement_action", return_value=True),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit, object(), state, "example_bot", dry_run=False
            )

        self.assertEqual(replies, 1)
        self.assertEqual(updated["daily_replies"], 51)
        self.assertEqual(updated["daily_non_moderator_replies"], 50)
        self.assertIn("moderator-1", updated["replied_to_comments"])
        self.assertNotIn("ordinary-1", updated["replied_to_comments"])
        self.assertEqual(updated["recent_user_replies"], {})
        turn.assert_called_once()
        chat.assert_not_called()

    def test_ordinary_users_still_obey_per_run_cap_without_thread_cap(self):
        first = _item("ordinary-1", "ordinary_one")
        second = _item("ordinary-2", "ordinary_two")
        reddit = SimpleNamespace(
            inbox=SimpleNamespace(comment_replies=lambda limit: [first, second])
        )
        state = {
            "daily_reset_date": date.today().isoformat(),
            "daily_replies": 0,
            "daily_non_moderator_replies": 0,
            "recent_user_replies": {},
            "replied_to_comments": [],
        }

        with (
            patch.object(reply_handler.config, "MAX_REPLIES_PER_DAY", 50),
            patch.object(reply_handler.config, "MAX_REPLIES_PER_RUN", 1),
            patch.object(reply_handler.config, "MODERATOR_REPLIES_BYPASS_LIMITS", False),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value={
                    "complete": True,
                    "route": "conversation",
                    "action": "",
                    "token_info": {"input_tokens": 2, "cost": 0.001},
                },
            ) as route,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                return_value=("A useful answer", {"total_tokens": 3, "cost": 0.0}),
            ),
            patch.object(reply_handler, "execute_engagement_action", return_value=True),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit, object(), state, "example_bot", dry_run=False
            )

        self.assertEqual(replies, 1)
        self.assertEqual(updated["daily_non_moderator_replies"], 1)
        self.assertEqual(len(updated["replied_to_comments"]), 1)
        route.assert_called_once()

    def test_moderator_summon_bypasses_zero_ordinary_user_caps(self):
        moderator_comment = _item(
            "moderator-summon", "example_mod", "Hey bot, what do you think?"
        )
        subreddit = SimpleNamespace(
            display_name="example_community",
            comments=lambda limit: [moderator_comment],
            new=lambda limit: [],
        )
        state = {
            "daily_reset_date": date.today().isoformat(),
            "daily_replies": 0,
            "daily_non_moderator_replies": 0,
            "recent_user_replies": {},
            "summon_responses": [],
        }

        with (
            patch.object(summon_handler.config, "MAX_REPLIES_PER_DAY", 0),
            patch.object(summon_handler.config, "MAX_REPLIES_PER_RUN", 0),
            patch.object(summon_handler.config, "MODERATOR_REPLIES_BYPASS_LIMITS", False),
            patch.object(summon_handler, "is_current_moderator", return_value=True),
            patch.object(
                summon_handler,
                "process_moderator_turn",
                return_value={
                    "kind": "moderator_turn",
                    "route": "conversation",
                    "handled": False,
                    "conversation_response": "A useful answer",
                    "token_info": {"total_tokens": 3, "cost": 0.0},
                },
            ) as turn,
            patch.object(
                summon_handler,
                "generate_conversational_response",
                return_value=("A useful answer", {"total_tokens": 3, "cost": 0.0}),
            ) as chat,
            patch.object(summon_handler, "execute_engagement_action", return_value=True),
            patch("time.sleep"),
        ):
            handled, _, _, updated = summon_handler.check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                reddit=None,
                dry_run=False,
            )

        self.assertEqual(handled, 1)
        self.assertEqual(updated["daily_replies"], 1)
        self.assertEqual(updated["daily_non_moderator_replies"], 0)
        self.assertIn("moderator-summon", updated["summon_responses"])
        turn.assert_called_once()
        chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
