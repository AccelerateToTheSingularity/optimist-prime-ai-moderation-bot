"""Tests for summon_handler.py pattern matching."""
import unittest
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bot_comment_format import format_bot_comment
from bot_utils import is_bot_owned_author
from summon_handler import (
    check_for_summons,
    is_bot_mentioned,
    is_jev_interaction_candidate,
    is_summon,
)


class FakeJevChoiceClient:
    def __init__(self, choice, *, input_tokens=0, cost=0.0, max_calls=5, calls_used=0):
        self.choice = choice
        self.ready = True
        self.max_calls = max_calls
        self.calls_used = calls_used
        self.last_usage = {"input_tokens": input_tokens, "cost": cost}

    def choose_many_detailed(self, *, state, questions):
        self.calls_used += 1
        self.last_usage = {"input_tokens": 23, "cost": 0.012}
        return {
            "public_interaction": {
                "choice": self.choice,
                "confidence": 0.91,
                "probabilities": {self.choice: 0.91},
            }
        }


class TestIsSummon(unittest.TestCase):
    def test_broad_bot_reference_reaches_jev_without_deciding_it_is_a_summon(self):
        self.assertTrue(is_jev_interaction_candidate("mod bot, can you review this?"))
        self.assertTrue(is_jev_interaction_candidate("The bot's last answer was confusing."))
        self.assertFalse(is_jev_interaction_candidate("I disagree with this point."))
        self.assertFalse(is_jev_interaction_candidate('Someone quoted "Hey bot, help me."'))
        self.assertFalse(is_jev_interaction_candidate("!example status"))

    def test_detects_optimist_prime(self):
        self.assertTrue(is_summon("Hey Optimist Prime, what do you think?"))

    def test_detects_bot_mention(self):
        self.assertTrue(is_summon("Hey bot, help me out"))

    def test_detects_mod_bot(self):
        self.assertTrue(is_summon("mod bot please summarize this"))

    def test_detects_short_mod_bot_review_without_greeting(self):
        self.assertTrue(is_summon("mod bot review"))
        self.assertFalse(is_summon("A mod bot review would be useful."))

    def test_detects_plain_bot_vocative_and_account_mention(self):
        self.assertTrue(is_summon("Optimist Prime, can you scan this person's history?"))
        self.assertTrue(is_summon("u/example_bot can you help me with my flair?"))

    def test_natural_direct_address_with_polite_modal_is_a_summon(self):
        self.assertTrue(is_summon("Optimist, could you review this request?"))
        self.assertTrue(is_summon("Hey Optimist, would you check this request?"))

    def test_markdown_profile_link_with_escaped_underscores_can_summon(self):
        request = (
            r"[u/example_bot]"
            r"(https://www.reddit.com/user/example_bot/) turn on my flair please"
        )
        self.assertTrue(is_bot_mentioned(request, "example_bot"))
        self.assertTrue(is_summon(request, "example_bot"))

    def test_markdown_profile_link_without_a_request_is_not_a_summon(self):
        mention = (
            r"[u/example_bot]"
            r"(https://www.reddit.com/user/example_bot/) made a good point"
        )
        self.assertTrue(is_bot_mentioned(mention, "example_bot"))
        self.assertFalse(is_summon(mention, "example_bot"))

    def test_mention_without_direct_request_is_eligible_for_judgment_not_a_summon(self):
        text = "Optimist Prime made an interesting point earlier."
        self.assertFalse(is_summon(text))
        self.assertTrue(is_bot_mentioned(text))
        self.assertFalse(is_bot_mentioned('Someone quoted "Optimist Prime made a point."'))

    def test_unmentioned_nearby_conversation_is_not_eligible(self):
        self.assertFalse(is_bot_mentioned("I think this is an interesting point."))

    def test_detects_first_person_summon(self):
        self.assertTrue(is_summon("I summon the bot"))

    def test_not_a_summon(self):
        self.assertFalse(is_summon("This is a regular comment about gardening"))

    def test_not_indirect_suggestion(self):
        self.assertFalse(is_summon("Someone should ask the bot about this"))
        self.assertFalse(is_summon("You can just summon the bot and ask what flair you would like."))
        self.assertFalse(is_summon("I hate these cringe X posts."))

    def test_quoted_or_hypothetical_bot_request_is_not_a_summon(self):
        self.assertFalse(is_summon('Someone said, "Hey bot, scan this user."'))
        self.assertFalse(is_summon("> Hey Optimist Prime, get in here\nI was quoting that request."))

    def test_positive_get_in_here_example_remains_a_summon(self):
        self.assertTrue(is_summon("WHAT THE FUCK IS THIS THREAD, OPTIMIST PRIME GET IN HERE"))

    def test_custom_self_flair_request_requires_direct_bot_address(self):
        self.assertTrue(is_summon('Hey mod bot, set my flair to "Community volunteer".'))
        self.assertFalse(is_summon('I want to change my flair to "Community volunteer".'))

    def test_unrelated_history_and_view_phrases_do_not_summon_moderator_report(self):
        self.assertFalse(
            is_summon(
                "You can scan everything in your body today. You cannot prevent many diseases. "
                "Genes, family history. No one can prevent them, not even a doctor."
            )
        )
        self.assertFalse(
            is_summon("It is the user's problem. That's a bad take; reconsider your point of view.")
        )

    def test_bot_footer_does_not_trigger_summon(self):
        self.assertFalse(
            is_summon(
                "Helpful answer\n\n---\n"
                "*^(AI assistant · mention the bot, mod bot, or use !bot)*"
            )
        )

    def test_current_footer_does_not_trigger_summon(self):
        # Footer still contains "Optimist Prime"; strip_bot_footer must remove it.
        self.assertFalse(is_summon(format_bot_comment("Helpful answer")))



class TestNoSummonNoExtraWork(unittest.TestCase):

    def test_bot_mention_in_comment_gets_one_jev_judgment_and_can_abstain(self):
        comment = SimpleNamespace(
            id="ordinary-1",
            created_utc=time.time(),
            author=SimpleNamespace(name="example_user"),
            body="You can just summon the bot and ask what flair you would like.",
            parent_id="t1_test04",
        )
        subreddit = SimpleNamespace(display_name="example_community")
        reddit = SimpleNamespace(info=MagicMock())
        model = MagicMock()
        with              patch("summon_handler.route_public_interaction", return_value={
                 "complete": True,
                 "route": "not_addressed",
                 "action": "",
                 "token_info": {"input_tokens": 11, "cost": 0.001},
             }) as route:
            result = check_for_summons(
                subreddit,
                model,
                {},
                "example_bot",
                reddit=reddit,
                recent_comments=[comment],
                recent_posts=[],
            )
        self.assertEqual(result[0], 0)
        reddit.info.assert_called_once_with(fullnames=["t1_test04"])
        model.generate_content.assert_not_called()
        route.assert_called_once()
        self.assertEqual(route.call_args.kwargs["channel"], "comment")
        self.assertEqual(route.call_args.kwargs["source_id"], "ordinary-1")

    def test_moderator_mention_gets_action_menu_for_unlisted_command_wording(self):
        now = time.time()
        comment = SimpleNamespace(
            id="mention-only-mod",
            created_utc=now,
            author=SimpleNamespace(name="example_mod"),
            body=(
                r"[u/example_bot]"
                r"(https://www.reddit.com/user/example_bot/) review u/example_user"
            ),
            parent_id="t3_test004",
            link_id="t3_test004",
            submission=SimpleNamespace(
                id="thread", title="A discussion", selftext=""
            ),
        )
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        reddit = SimpleNamespace(info=MagicMock(return_value=[SimpleNamespace(
            fullname="t3_test004", author=SimpleNamespace(name="example_user")
        )]))
        model = MagicMock()

        with              patch("summon_handler.is_current_moderator", return_value=True), \
             patch("summon_handler.route_public_interaction") as public_router, \
             patch("summon_handler.process_moderator_turn", return_value={
                 "route": "no_reply",
                 "handled": False,
                 "conversation_response": "[[NO_REPLY]]",
                 "token_info": {"total_tokens": 3, "cost": 0.0},
             }) as moderator_turn, \
             patch("summon_handler.generate_conversational_response", return_value=(
                 "[[NO_REPLY]]", {"total_tokens": 3, "cost": 0.0}
             )) as conversation:
            handled, _tokens, _cost, state = check_for_summons(
                subreddit,
                model,
                {},
                "example_bot",
                reddit=reddit,
                recent_comments=[comment],
                recent_posts=[],
            )

        self.assertEqual(handled, 0)
        self.assertEqual(state["public_response_ledger"]["comment:mention-only-mod"]["status"], "abstained")
        moderator_turn.assert_called_once()
        self.assertFalse(moderator_turn.call_args.kwargs["is_summon"])
        public_router.assert_not_called()
        conversation.assert_not_called()



class TestParentAuthorCache(unittest.TestCase):
    """Parent batch cache must key by fullname (t1_xxx), not bare id."""

    def test_cache_uses_fullname_keys(self):
        parent = MagicMock()
        parent.fullname = "t1_test07"
        parent.name = "abc123"
        parent.author = MagicMock()
        parent.author.name = "example_bot"

        cache = {}
        cache[parent.fullname] = parent.author.name

        self.assertIn("t1_test07", cache)
        self.assertNotIn("abc123", cache)
        self.assertEqual(cache["t1_test07"], "example_bot")

    def test_skips_reply_to_bot_via_parent_cache(self):
        """When parent is a migrated bot alias, skip before summon check."""
        comment = MagicMock()
        comment.id = "c1"
        comment.created_utc = 9_999_999_999
        comment.author = MagicMock()
        comment.author.name = "regular_user"
        comment.body = "hey bot help"
        comment.parent_id = "t1_test008"

        runtime_bot_username = "future-runtime-name"
        parent_author_cache = {"t1_test008": "legacy_example_account"}

        parent_id = comment.parent_id
        with patch("config.BOT_OWNED_USERNAMES", ["legacy_example_account"]):
            skip = (
                parent_id in parent_author_cache
                and is_bot_owned_author(parent_author_cache[parent_id], runtime_bot_username)
            )
        self.assertTrue(skip)
        self.assertNotEqual(parent_author_cache[parent_id], runtime_bot_username)


class TestSummonPostScanReuse(unittest.TestCase):
    def test_runner_supplied_posts_avoid_a_second_new_listing(self):
        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit.comments.return_value = []
        subreddit.new.side_effect = AssertionError("unexpected duplicate post fetch")

        handled, tokens, cost, _state = check_for_summons(
            subreddit,
            None,
            {},
            "example_bot",
            dry_run=True,
            recent_posts=[],
        )

        self.assertEqual((handled, tokens, cost), (0, 0, 0.0))
        subreddit.new.assert_not_called()

    def test_runner_supplied_posts_do_not_add_the_fallback_fetch_delay(self):
        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit.comments.return_value = []

        with patch("summon_handler.time.sleep") as sleep:
            check_for_summons(
                subreddit,
                None,
                {},
                "example_bot",
                dry_run=True,
                recent_posts=[],
            )

        sleep.assert_not_called()


class TestJevRoutedSummons(unittest.TestCase):
    def _comment(self, comment_id="jev-comment", body="Hey Optimist Prime, what do you think?"):
        return SimpleNamespace(
            id=comment_id,
            created_utc=time.time(),
            author=SimpleNamespace(name="example_user"),
            body=body,
            parent_id="t3_test004",
            link_id="t3_test004",
            link_title="A useful discussion",
            submission=SimpleNamespace(
                id="thread", title="A useful discussion", selftext="Opening context."
            ),
        )

    def test_conversation_uses_one_bounded_jev_call_then_text_generation(self):
        comment = self._comment()
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        jev_client = FakeJevChoiceClient(
            "conversation", input_tokens=0, cost=0.0
        )
        state = {}

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_conversational_response", return_value=(
                 "A useful answer.", {"total_tokens": 13, "cost": 0.005}
             )) as generate, \
             patch("summon_handler.execute_engagement_action", return_value=True), \
             patch("summon_handler.record_reply_success"):
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=jev_client,
            )

        self.assertEqual(handled, 1)
        self.assertEqual(jev_client.calls_used, 1)
        self.assertEqual((tokens, cost), (36, 0.017))
        self.assertFalse(generate.call_args.kwargs["allow_abstain"])
        self.assertEqual(state["summon_responses"], [comment.id])

    def test_direct_reply_to_known_bot_comment_is_a_candidate_without_a_mention(self):
        comment = self._comment("direct-reply", "Thanks, that clears it up.")
        comment.parent_id = "t1_bot-parent"
        comment.parent = MagicMock(side_effect=AssertionError("parent lookup should be cached"))
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        jev_client = FakeJevChoiceClient("conversation")

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_conversational_response", return_value=(
                 "Glad that helped.", {"total_tokens": 4, "cost": 0.001}
             )) as generate, \
             patch("summon_handler.execute_engagement_action", return_value=True), \
             patch("summon_handler.record_reply_success"):
            handled, _tokens, _cost, _state = check_for_summons(
                subreddit,
                object(),
                {"reply_parent_authors": {"t1_bot-parent": "example_bot"}},
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=jev_client,
            )

        self.assertEqual(handled, 1)
        self.assertEqual(jev_client.calls_used, 1)
        self.assertFalse(generate.call_args.kwargs["is_summon"])
        comment.parent.assert_not_called()

    def test_no_reply_decision_is_persisted_as_public_abstention(self):
        comment = self._comment("no-reply", "Hey Optimist, thanks.")
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        jev_client = FakeJevChoiceClient("no_reply")
        persist = MagicMock()
        state = {}

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_conversational_response") as generate:
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                persist_state=persist,
                jev_client=jev_client,
            )

        self.assertEqual(handled, 0)
        self.assertEqual(jev_client.calls_used, 1)
        self.assertEqual((tokens, cost), (23, 0.012))
        self.assertEqual(
            state["public_response_ledger"]["comment:no-reply"]["status"],
            "abstained",
        )
        self.assertIn("no-reply", state["summon_responses"])
        self.assertGreaterEqual(persist.call_count, 1)
        generate.assert_not_called()

    def test_invalid_jev_choice_remains_retryable_without_legacy_fallback(self):
        comment = self._comment("invalid-choice")
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        jev_client = FakeJevChoiceClient("not_a_menu_option")
        state = {}

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_conversational_response") as generate:
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=jev_client,
            )

        self.assertEqual(handled, 0)
        self.assertEqual(jev_client.calls_used, 1)
        self.assertEqual((tokens, cost), (23, 0.012))
        self.assertNotIn(
            "comment:invalid-choice",
            state.get("public_response_ledger", {}),
        )
        self.assertEqual(
            state["catch_up"]["comments"]["pending"][0]["id"],
            "invalid-choice",
        )
        generate.assert_not_called()

    def test_unavailable_jev_remains_retryable_without_calling_a_second_classifier(self):
        comment = self._comment("missing-jev")
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        state = {}

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_conversational_response") as generate:
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=None,
            )

        self.assertEqual((handled, tokens, cost), (0, 0, 0.0))
        self.assertNotIn(
            "comment:missing-jev",
            state.get("public_response_ledger", {}),
        )
        self.assertEqual(
            state["catch_up"]["comments"]["pending"][0]["id"],
            "missing-jev",
        )
        generate.assert_not_called()

    def test_incomplete_jev_route_can_be_retried_on_a_later_cycle(self):
        comment = self._comment("route-retry")
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        state = {}
        incomplete = {"complete": False, "error": "jev_unavailable", "token_info": {}}
        complete_no_reply = {
            "complete": True,
            "route": "no_reply",
            "action": "",
            "token_info": {"input_tokens": 2, "cost": 0.001},
        }

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.route_public_interaction", side_effect=(
                 incomplete, complete_no_reply
             )) as route:
            first = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=object(),
            )
            self.assertNotIn(
                "comment:route-retry",
                state.get("public_response_ledger", {}),
            )
            second = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=object(),
            )

        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], 0)
        self.assertEqual(route.call_count, 2)
        self.assertEqual(
            state["public_response_ledger"]["comment:route-retry"]["status"],
            "abstained",
        )

    def test_incomplete_top_level_post_route_stays_retryable(self):
        post = SimpleNamespace(
            id="post-route-retry",
            created_utc=time.time(),
            author=SimpleNamespace(name="example_user"),
            title="Optimist, could you take a look?",
            selftext="",
        )
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        state = {}

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.route_public_interaction", return_value={
                 "complete": False,
                 "error": "jev_unavailable",
                 "token_info": {},
             }):
            handled, _tokens, _cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[],
                recent_posts=[post],
                jev_client=object(),
            )

        self.assertEqual(handled, 0)
        self.assertNotIn("post:post-route-retry:post_summon", state.get("public_response_ledger", {}))
        self.assertEqual(
            state["catch_up"]["posts"]["pending"][0]["id"],
            "post-route-retry",
        )

    def test_disabled_flair_routes_are_not_offered_to_jev(self):
        comment = self._comment("disabled-actions")
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        result = {
            "complete": True,
            "route": "not_addressed",
            "action": "",
            "token_info": {"input_tokens": 1, "cost": 0.001},
        }

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.config.USER_CUSTOM_FLAIR_ENABLED", False), \
             patch("summon_handler.route_public_interaction", return_value=result) as route:
            check_for_summons(
                subreddit,
                object(),
                {},
                "example_bot",
                recent_comments=[comment],
                recent_posts=[],
                jev_client=object(),
            )

        self.assertEqual(route.call_count, 1)
        self.assertEqual(route.call_args.kwargs["allowed_actions"], [])

    def test_top_level_post_requires_direct_summon_and_uses_jev_once(self):
        post = SimpleNamespace(
            id="direct-post",
            created_utc=time.time(),
            author=SimpleNamespace(name="example_user"),
            title="Hey Optimist Prime, what do you think of this?",
            selftext="The proposal text.",
        )
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        jev_client = FakeJevChoiceClient("conversation")

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_post_summon_response", return_value=(
                 "A response.", {"total_tokens": 7, "cost": 0.004}
             )) as generate, \
             patch("summon_handler.execute_engagement_action", return_value=True), \
             patch("summon_handler.record_reply_success"):
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                {},
                "example_bot",
                recent_comments=[],
                recent_posts=[post],
                jev_client=jev_client,
            )

        self.assertEqual(handled, 1)
        self.assertEqual(jev_client.calls_used, 1)
        self.assertEqual((tokens, cost), (30, 0.016))
        self.assertEqual(generate.call_count, 1)

    def test_top_level_mention_without_direct_summon_reaches_jev_for_final_judgment(self):
        post = SimpleNamespace(
            id="mention-only-post",
            created_utc=time.time(),
            author=SimpleNamespace(name="example_user"),
            title="Optimist Prime made an interesting point.",
            selftext="",
        )
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        no_reply = {
            "complete": True,
            "route": "not_addressed",
            "action": "",
            "token_info": {"input_tokens": 23, "cost": 0.012},
        }

        with              patch("summon_handler.route_public_interaction", return_value=no_reply) as route:
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                {},
                "example_bot",
                recent_comments=[],
                recent_posts=[post],
                jev_client=object(),
            )

        self.assertEqual((handled, tokens, cost), (0, 23, 0.012))
        route.assert_called_once()
        self.assertFalse(route.call_args.kwargs["summoned"])

    def test_unusual_direct_post_summon_is_not_blocked_by_fixed_phrase_list(self):
        post = SimpleNamespace(
            id="unusual-direct-post",
            created_utc=time.time(),
            author=SimpleNamespace(name="example_user"),
            title="Could Optimist weigh in on this?",
            selftext="A question for the community.",
        )
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        self.assertFalse(is_summon(f"{post.title} {post.selftext}"))
        jev_client = FakeJevChoiceClient("conversation")

        with              patch("summon_handler.is_current_moderator", return_value=False), \
             patch("summon_handler.generate_post_summon_response", return_value=(
                 "A response.", {"total_tokens": 7, "cost": 0.004}
             )) as generate, \
             patch("summon_handler.execute_engagement_action", return_value=True), \
             patch("summon_handler.record_reply_success"):
            handled, _tokens, _cost, _state = check_for_summons(
                subreddit,
                object(),
                {},
                "example_bot",
                recent_comments=[],
                recent_posts=[post],
                jev_client=jev_client,
            )

        self.assertEqual(handled, 1)
        self.assertEqual(jev_client.calls_used, 1)
        self.assertIs(generate.call_args.args[0], post)

    def test_moderator_top_level_history_review_uses_moderator_route_only(self):
        post = SimpleNamespace(
            id="mod-history-post",
            created_utc=time.time(),
            author=SimpleNamespace(name="example_mod"),
            title="Hey Optimist, review u/example_user's history.",
            selftext="",
        )
        subreddit = SimpleNamespace(display_name="example_community", _reddit=None)
        state = {}
        jev_client = object()

        with              patch("summon_handler.is_current_moderator", return_value=True), \
             patch("summon_handler.process_moderator_turn", return_value={
                 "route": "history_review",
                 "handled": True,
                 "success": True,
                 "response": "Private report",
                 "token_info": {"total_tokens": 31, "cost": 0.02},
             }) as moderator_turn, \
             patch("summon_handler.route_public_interaction") as public_router, \
             patch("summon_handler.execute_engagement_action") as public_reply:
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                recent_comments=[],
                recent_posts=[post],
                jev_client=jev_client,
            )

        self.assertEqual((handled, tokens, cost), (1, 31, 0.02))
        self.assertEqual(moderator_turn.call_count, 1)
        self.assertEqual(moderator_turn.call_args.kwargs["channel"], "post")
        self.assertIn("u/example_user", moderator_turn.call_args.kwargs["text"])
        self.assertIs(moderator_turn.call_args.kwargs["jev_client"], jev_client)
        public_router.assert_not_called()
        public_reply.assert_not_called()
        self.assertIn("post_mod-history-post", state["summon_responses"])

    def test_ordinary_comments_do_not_trigger_parent_lookup(self):
        comment = MagicMock()
        comment.id = "ordinary"
        comment.created_utc = 9_999_999_999
        comment.author = MagicMock()
        comment.author.name = "regular_user"
        comment.body = "This is a normal discussion comment."
        comment.parent_id = "t1_test04"

        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit.comments.return_value = [comment]

        reddit = MagicMock()
        check_for_summons(
            subreddit,
            None,
            {},
            "example_bot",
            reddit=reddit,
            dry_run=True,
            recent_posts=[],
        )

        reddit.info.assert_not_called()
        comment.parent.assert_not_called()


class TestModeratorHistorySummonIntegration(unittest.TestCase):
    def test_missing_parent_from_partial_batch_cache_uses_commenter_or_thread_op(self):
        from datetime import datetime, timezone

        for parent_id, expected_author in (
            ("t1_source-comment", "comment_author"),
            ("t3_post456-post", "thread_op"),
        ):
            with self.subTest(parent_id=parent_id):
                comment = MagicMock()
                comment.id = f"review-{parent_id}"
                comment.created_utc = datetime.now(timezone.utc).timestamp()
                comment.author = SimpleNamespace(name="example_mod")
                comment.body = "hey optimist - review"
                comment.parent_id = parent_id
                comment.parent.return_value = SimpleNamespace(
                    author=SimpleNamespace(name=expected_author)
                )
                comment.submission = SimpleNamespace(
                    id="thread", title="A discussion", selftext=""
                )
                comment.link_id = "t3_test004"

                subreddit = MagicMock()
                subreddit.display_name = "example_community"
                subreddit._reddit = None
                subreddit.comments.return_value = [comment]
                # Simulate a partial batch result: it contains some other parent,
                # so the cache is non-empty, but it missed this request's parent.
                reddit = SimpleNamespace(info=MagicMock(return_value=[SimpleNamespace(
                    fullname="t1_test00009", author=SimpleNamespace(name="unrelated_user")
                )]))
                state = {}
                with                      patch("summon_handler.is_current_moderator", return_value=True), \
                     patch("summon_handler.process_moderator_turn", return_value={
                         "kind": "moderator_turn", "route": "history_review", "handled": True,
                         "success": True, "response": "History report", "state": state,
                     }) as turn, \
                     patch("summon_handler.build_full_context", return_value="context"), \
                     patch("summon_handler.reply_to_latest_private_chat", return_value=True), \
                     patch("summon_handler.remove_processed_instruction", return_value=True), \
                     patch("summon_handler.config.MODERATOR_REPLIES_BYPASS_LIMITS", False), \
                     patch("summon_handler.config.MAX_REPLIES_PER_RUN", 10):
                    check_for_summons(
                        subreddit,
                        object(),
                        state,
                        "example_bot",
                        reddit=reddit,
                        dry_run=False,
                        recent_posts=[],
                        recent_comments=[comment],
                    )

                comment.parent.assert_called_once()
                self.assertEqual(turn.call_args.kwargs["parent_author"], expected_author)

    def test_moderator_history_command_targets_the_replied_to_author_and_uses_existing_chat(self):
        from datetime import datetime, timezone

        from config import MAX_REPLIES_PER_RUN

        comment = MagicMock()
        comment.id = "mod-request"
        comment.created_utc = datetime.now(timezone.utc).timestamp()
        comment.author = MagicMock()
        comment.author.name = "example_mod"
        comment.body = "Hey mod bot, scan this person's history and summarize their recent community contributions."
        comment.parent_id = "t1_test00a"
        comment.parent.return_value = SimpleNamespace(
            author=SimpleNamespace(name="history_subject")
        )

        jev_client = object()
        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit._reddit = None
        subreddit.comments.return_value = [comment]

        with              patch("summon_handler.is_current_moderator", return_value=True), \
             patch("summon_handler.process_moderator_turn", return_value={
                 "kind": "moderator_turn", "route": "history_review", "handled": True,
                 "success": True, "response": "History report",
                 "token_info": {"total_tokens": 25, "cost": 0.01}, "state": {},
             }) as turn, \
             patch("summon_handler.execute_engagement_action", return_value=True) as post, \
             patch("summon_handler.reply_to_latest_private_chat", return_value=True) as chat_reply, \
             patch("summon_handler.remove_processed_instruction", return_value=True) as remove, \
             patch("summon_handler.config.MODERATOR_REPLIES_BYPASS_LIMITS", False), \
             patch("summon_handler.config.MAX_REPLIES_PER_RUN", max(1, MAX_REPLIES_PER_RUN)):
            state = {}
            handled, tokens, cost, _state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                reddit=object(),
                dry_run=False,
                recent_posts=[],
                recent_comments=[comment],
                jev_client=jev_client,
            )

        self.assertTrue(turn.called, f"moderator turn was not called: {turn.call_args}")
        self.assertTrue(chat_reply.called, f"route was {turn.return_value.get('route')!r}")
        self.assertEqual((handled, tokens, cost), (1, 25, 0.01))
        self.assertEqual(turn.call_args.kwargs["parent_author"], "history_subject")
        self.assertTrue(turn.call_args.kwargs["verified_moderator"])
        self.assertIs(turn.call_args.kwargs["jev_client"], jev_client)
        chat_reply.assert_called_once()
        self.assertEqual(chat_reply.call_args.kwargs["username"], "example_mod")
        self.assertEqual(chat_reply.call_args.kwargs["text"], "History report")
        remove.assert_called_once_with(comment)
        post.assert_not_called()
        self.assertIn("mod-request", state["summon_responses"])

    def test_public_history_dry_run_does_not_reply_remove_or_claim_request(self):
        from datetime import datetime, timezone

        comment = MagicMock()
        comment.id = "mod-request-dry-run"
        comment.created_utc = datetime.now(timezone.utc).timestamp()
        comment.author = MagicMock(name="author")
        comment.author.name = "example_mod"
        comment.body = "Hey mod bot, review u/example_user."
        comment.parent_id = "t1_test00a"
        comment.parent.return_value = SimpleNamespace(author=SimpleNamespace(name="subject"))

        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit._reddit = None
        subreddit.comments.return_value = [comment]
        state = {}
        persist = MagicMock()
        with              patch("summon_handler.is_current_moderator", return_value=True), \
             patch("summon_handler.process_moderator_turn", return_value={
                 "kind": "moderator_turn", "route": "history_review", "handled": True,
                 "success": True, "response": "History report", "state": state,
             }), \
             patch("summon_handler.reply_to_latest_private_chat") as chat_reply, \
             patch("summon_handler.remove_processed_instruction") as remove:
            _handled, _tokens, _cost, result_state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                reddit=object(),
                dry_run=True,
                recent_posts=[],
                recent_comments=[comment],
                persist_state=persist,
            )

        self.assertIsNotNone(result_state)
        chat_reply.assert_not_called()
        remove.assert_not_called()
        persist.assert_not_called()
        self.assertNotIn("mod-request-dry-run", result_state.get("summon_responses", []))

    def test_public_history_request_is_removed_and_claimed_when_chat_reply_fails(self):
        from datetime import datetime, timezone

        comment = MagicMock()
        comment.id = "mod-request-no-chat"
        comment.created_utc = datetime.now(timezone.utc).timestamp()
        comment.author = MagicMock(name="author")
        comment.author.name = "example_mod"
        comment.body = "Hey mod bot, review u/example_user."
        comment.parent_id = "t1_test00a"
        comment.parent.return_value = SimpleNamespace(author=SimpleNamespace(name="subject"))

        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit._reddit = None
        subreddit.comments.return_value = [comment]
        state = {}
        persist = MagicMock()
        with              patch("summon_handler.is_current_moderator", return_value=True), \
             patch("summon_handler.process_moderator_turn", return_value={
                 "kind": "moderator_turn", "route": "history_review", "handled": True,
                 "success": True, "response": "History report", "state": state,
             }), \
             patch("summon_handler.reply_to_latest_private_chat", side_effect=RuntimeError("rejected")) as chat_reply, \
             patch("summon_handler.remove_processed_instruction", return_value=True) as remove, \
             patch("summon_handler.execute_engagement_action") as post:
            handled, _tokens, _cost, result_state = check_for_summons(
                subreddit,
                object(),
                state,
                "example_bot",
                reddit=object(),
                dry_run=False,
                recent_posts=[],
                recent_comments=[comment],
                persist_state=persist,
            )

        self.assertEqual(handled, 0)
        self.assertIs(result_state, state)
        chat_reply.assert_called_once()
        remove.assert_called_once_with(comment)
        post.assert_not_called()
        self.assertIn("mod-request-no-chat", state["summon_responses"])
        self.assertGreaterEqual(persist.call_count, 1)

    def test_handled_public_history_request_is_not_analyzed_again(self):
        from datetime import datetime, timezone

        comment = MagicMock()
        comment.id = "already-handled-history"
        comment.created_utc = datetime.now(timezone.utc).timestamp()
        comment.author = MagicMock(name="author")
        comment.author.name = "example_mod"
        comment.body = "Hey mod bot, review u/example_user."

        subreddit = MagicMock()
        subreddit.display_name = "example_community"
        subreddit.comments.return_value = [comment]
        with              patch("summon_handler.process_moderator_turn") as analyze, \
             patch("summon_handler.reply_to_latest_private_chat") as chat_reply, \
             patch("summon_handler.remove_processed_instruction") as remove:
            _handled, _tokens, _cost, _state = check_for_summons(
                subreddit,
                object(),
                {"summon_responses": ["already-handled-history"]},
                "example_bot",
                dry_run=False,
                recent_posts=[],
                recent_comments=[comment],
            )

        analyze.assert_not_called()
        chat_reply.assert_not_called()
        remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
