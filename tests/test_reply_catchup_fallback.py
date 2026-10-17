"""Tests for recovering direct bot replies omitted from Reddit inbox listings."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reply_handler


def _comment(
    item_id: str,
    author: str,
    parent_id: str,
    body: str = "A question for the bot",
):
    return SimpleNamespace(
        id=item_id,
        author=SimpleNamespace(name=author),
        body=body,
        parent_id=parent_id,
        created_utc=datetime.now(timezone.utc).timestamp(),
        subreddit=SimpleNamespace(display_name="example_community"),
        submission=SimpleNamespace(id="submission-1"),
    )


def _state():
    return {
        "daily_replies": 0,
        "daily_non_moderator_replies": 0,
        "recent_user_replies": {},
        "replied_to_comments": [],
    }


def _route(route: str, action: str = "", **metadata):
    return {
        "complete": True,
        "route": route,
        "action": action,
        "token_info": {"input_tokens": 2, "cost": 0.001},
        **metadata,
    }


class _Reddit:
    def __init__(self, inbox_items, parents):
        self.inbox = SimpleNamespace(comment_replies=lambda limit: inbox_items)
        self.parents = parents
        self.info_calls = []

    def info(self, *, fullnames):
        self.info_calls.append(list(fullnames))
        return [self.parents[fullname] for fullname in fullnames if fullname in self.parents]


class TestReplyCatchupFallback(unittest.TestCase):
    def setUp(self):
        subreddit_patch = patch.object(
            reply_handler.config, "SUBREDDIT", "example_community"
        )
        subreddit_patch.start()
        self.addCleanup(subreddit_patch.stop)

    def test_direct_reply_missing_from_inbox_is_recovered_from_recent_comments(self):
        # A human account whose name ends in "bot" must not be filtered out
        # when replying directly to Optimist.
        item = _comment("reply-1", "example_human_bot", "t1_parent-1")
        parent = SimpleNamespace(
            id="parent-1",
            fullname="t1_parent-1",
            author=SimpleNamespace(name="example_bot"),
            body="Here is the earlier bot response.",
        )
        reddit = _Reddit([], {"t1_parent-1": parent})

        with (
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value=_route("conversation"),
            ) as route,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                return_value=("A useful answer", {"total_tokens": 3, "cost": 0.0}),
            ) as generate,
            patch.object(reply_handler, "execute_engagement_action", return_value=True),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
            patch.object(reply_handler.config, "USER_CUSTOM_FLAIR_ENABLED", True),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                _state(),
                "example_bot",
                dry_run=False,
                recent_comments=[parent, item],
            )

        self.assertEqual(replies, 1)
        self.assertIn("reply-1", updated["replied_to_comments"])
        self.assertEqual(reddit.info_calls, [["t1_parent-1"]])
        route.assert_called_once()
        self.assertEqual(route.call_args.kwargs["parent_author"], "example_bot")
        self.assertEqual(
            route.call_args.kwargs["parent_text"],
            "Here is the earlier bot response.",
        )
        self.assertEqual(
            route.call_args.kwargs["allowed_actions"],
            ["self_custom_flair"],
        )
        self.assertFalse(generate.call_args.kwargs["allow_abstain"])

    def test_direct_reply_can_be_deliberately_left_unanswered(self):
        item = _comment("reply-no-reply", "example_user", "t1_parent-no-reply")
        parent = SimpleNamespace(
            id="parent-no-reply",
            fullname="t1_parent-no-reply",
            author=SimpleNamespace(name="example_bot"),
        )
        reddit = _Reddit([item], {"t1_parent-no-reply": parent})
        state = _state()
        persist_calls = []

        with (
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value=_route("no_reply"),
            ) as route,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                side_effect=AssertionError("Jev already selected no_reply"),
            ) as generate,
            patch.object(
                reply_handler,
                "execute_engagement_action",
                side_effect=AssertionError("an abstained turn must not publish"),
            ),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                state,
                "example_bot",
                dry_run=False,
                persist_state=lambda value: persist_calls.append(value.copy()),
                recent_comments=[item],
            )

        self.assertEqual(replies, 0)
        route.assert_called_once()
        generate.assert_not_called()
        self.assertEqual(
            updated["public_response_ledger"]["comment:reply-no-reply"]["status"],
            "abstained",
        )
        self.assertTrue(persist_calls)

    def test_disabled_flair_features_are_not_offered_to_jev(self):
        item = _comment("reply-flair-disabled", "example_user", "t1_test000004-disabled")
        reddit = _Reddit([item], {})

        with (
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value=_route("conversation"),
            ) as route,
            patch.object(reply_handler.config, "USER_CUSTOM_FLAIR_ENABLED", False),
            patch.object(
                reply_handler,
                "generate_conversational_response",
                return_value=("A useful answer", {"total_tokens": 3, "cost": 0.0}),
            ),
            patch.object(reply_handler, "execute_engagement_action", return_value=True),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
        ):
            replies, _, _, _ = reply_handler.check_inbox_replies(
                reddit,
                object(),
                _state(),
                "example_bot",
                dry_run=False,
            )

        self.assertEqual(replies, 1)
        self.assertEqual(route.call_args.kwargs["allowed_actions"], [])


    def test_public_reply_cap_prevents_spending_a_route_call(self):
        item = _comment("reply-at-cap", "example_user", "t1_test000004-at-cap")
        reddit = _Reddit([item], {})

        with (
            patch.object(
                reply_handler,
                "response_capacity_available",
                return_value=(False, "ordinary_global"),
            ),
            patch.object(reply_handler.config, "ENGAGEMENT_ACTION_MODE", "comment"),
            patch.object(reply_handler, "notify_public_response_limit") as notify,
            patch.object(
                reply_handler,
                "route_public_interaction",
                side_effect=AssertionError("a capped reply must not consume a Jev call"),
            ) as route,
            patch.object(reply_handler, "is_current_moderator", return_value=False),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                _state(),
                "example_bot",
                dry_run=False,
            )

        self.assertEqual(replies, 0)
        self.assertNotIn("reply-at-cap", updated["replied_to_comments"])
        notify.assert_called_once()
        route.assert_not_called()

    def test_invalid_jev_choice_stays_retryable_without_reply_or_ledger_entry(self):
        item = _comment("reply-invalid-route", "example_user", "t1_test000004-invalid")
        reddit = _Reddit([item], {})
        state = _state()
        persist_calls = []

        with (
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value={
                    "complete": False,
                    "route": None,
                    "action": "",
                    "token_info": {"input_tokens": 2, "cost": 0.001},
                },
            ) as route,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                side_effect=AssertionError("an invalid route must not generate prose"),
            ),
            patch.object(
                reply_handler,
                "execute_engagement_action",
                side_effect=AssertionError("an invalid route must not publish"),
            ),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
        ):
            replies, tokens, cost, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                state,
                "example_bot",
                dry_run=False,
                persist_state=lambda value: persist_calls.append(value.copy()),
            )

        self.assertEqual(replies, 0)
        self.assertEqual(tokens, 2)
        self.assertEqual(cost, 0.001)
        self.assertNotIn("reply-invalid-route", updated["replied_to_comments"])
        self.assertEqual(updated.get("public_response_ledger", {}), {})
        self.assertEqual(
            updated["catch_up"]["comments"]["pending"][0]["id"],
            "reply-invalid-route",
        )
        self.assertFalse(persist_calls)
        route.assert_called_once()


    def test_selected_custom_flair_text_is_passed_to_handler(self):
        item = _comment("reply-custom-flair", "example_user", "t1_test000004-custom")
        reddit = _Reddit([item], {})
        state = _state()

        with (
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value=_route(
                    "self_custom_flair", "set", requested_flair="Helpful analyst"
                ),
            ) as route,
            patch.object(reply_handler.config, "USER_CUSTOM_FLAIR_ENABLED", True),
            patch.object(
                reply_handler,
                "handle_self_custom_flair_request",
                return_value=("Custom flair updated.", 4, 0.02, state),
            ) as handle,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                side_effect=AssertionError("selected actions do not generate chat prose"),
            ),
            patch.object(reply_handler, "execute_engagement_action", return_value=True),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
        ):
            replies, tokens, cost, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                state,
                "example_bot",
                dry_run=False,
            )

        self.assertEqual(replies, 1)
        self.assertEqual(tokens, 6)
        self.assertAlmostEqual(cost, 0.021)
        self.assertIn("reply-custom-flair", updated["replied_to_comments"])
        route.assert_called_once()
        self.assertEqual(
            handle.call_args.kwargs["intent_override"],
            {"action": "set", "label": "Helpful analyst"},
        )

    def test_inbox_and_catchup_duplicate_is_processed_once(self):
        item = _comment("reply-1", "example_user", "t1_parent-1")
        parent = SimpleNamespace(
            id="parent-1",
            fullname="t1_parent-1",
            author=SimpleNamespace(name="example_bot"),
        )
        reddit = _Reddit([item], {"t1_parent-1": parent})

        with (
            patch.object(
                reply_handler,
                "route_public_interaction",
                return_value=_route("conversation"),
            ) as route,
            patch.object(
                reply_handler,
                "generate_conversational_response",
                return_value=("A useful answer", {"total_tokens": 3, "cost": 0.0}),
            ),
            patch.object(reply_handler, "execute_engagement_action", return_value=True),
            patch.object(reply_handler, "is_current_moderator", return_value=False),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                _state(),
                "example_bot",
                dry_run=False,
                recent_comments=[item, item],
            )

        self.assertEqual(replies, 1)
        self.assertEqual(updated["replied_to_comments"].count("reply-1"), 1)
        self.assertEqual(reddit.info_calls, [])
        route.assert_called_once()

    def test_non_bot_parent_is_not_treated_as_a_bot_reply(self):
        item = _comment("reply-2", "example_user", "t1_test000004-2")
        parent = SimpleNamespace(
            id="parent-2",
            fullname="t1_test000004-2",
            author=SimpleNamespace(name="another_user"),
        )
        reddit = _Reddit([], {"t1_test000004-2": parent})
        state = _state()

        with patch.object(
            reply_handler,
            "route_public_interaction",
            side_effect=AssertionError("non-bot parent must not trigger an AI call"),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                state,
                "example_bot",
                dry_run=True,
                recent_comments=[item],
            )

        self.assertEqual(replies, 0)
        self.assertEqual(state, _state())
        self.assertNotIn("reply-2", updated["replied_to_comments"])

    def test_human_username_ending_in_bot_is_not_treated_as_optimist(self):
        item = _comment("reply-example_human_bot", "example_user", "t1_test000004-example_human_bot")
        parent = SimpleNamespace(
            id="parent-example_human_bot",
            fullname="t1_test000004-example_human_bot",
            author=SimpleNamespace(name="example_human_bot"),
        )
        reddit = _Reddit([], {"t1_test000004-example_human_bot": parent})
        state = _state()

        with patch.object(
            reply_handler,
            "route_public_interaction",
            side_effect=AssertionError("a reply to a human must not trigger Optimist"),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                state,
                "example_bot",
                dry_run=True,
                recent_comments=[item],
            )

        self.assertEqual(replies, 0)
        self.assertEqual(reddit.info_calls, [["t1_test000004-example_human_bot"]])
        self.assertNotIn("reply-example_human_bot", updated["replied_to_comments"])

    def test_dry_run_does_not_persist_parent_cache(self):
        item = _comment("reply-3", "example_user", "t1_test000004-3")
        parent = SimpleNamespace(
            id="parent-3",
            fullname="t1_test000004-3",
            author=SimpleNamespace(name="example_bot"),
        )
        reddit = _Reddit([], {"t1_test000004-3": parent})
        state = _state()

        reply_handler.check_inbox_replies(
            reddit,
            object(),
            state,
            "example_bot",
            dry_run=True,
            recent_comments=[item],
        )

        self.assertNotIn("reply_parent_authors", state)

    def test_already_replied_comment_is_filtered_before_parent_lookup(self):
        item = _comment("reply-4", "example_user", "t1_test000004-4")
        parent = SimpleNamespace(
            id="parent-4",
            fullname="t1_test000004-4",
            author=SimpleNamespace(name="example_bot"),
        )
        reddit = _Reddit([], {"t1_test000004-4": parent})
        state = _state()
        state["replied_to_comments"] = ["reply-4"]

        with patch.object(
            reply_handler,
            "route_public_interaction",
            side_effect=AssertionError("already-replied comments must not reach AI"),
        ):
            replies, _, _, updated = reply_handler.check_inbox_replies(
                reddit,
                object(),
                state,
                "example_bot",
                dry_run=True,
                recent_comments=[item],
            )

        self.assertEqual(replies, 0)
        self.assertEqual(reddit.info_calls, [])
        self.assertEqual(updated["replied_to_comments"], ["reply-4"])


if __name__ == "__main__":
    unittest.main()
