"""Tests for exact, operator-allowlisted self-service flair values."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config
from self_flair_requests import (
    handle_self_custom_flair_request,
    parse_self_custom_flair_request,
)


ALLOWED_LABEL = "Community volunteer"


class TestSelfFlairRequestParser(unittest.TestCase):
    def setUp(self):
        bot_identity = patch.object(config, "EXPECTED_BOT_USERNAME", "example_bot")
        bot_identity.start()
        self.addCleanup(bot_identity.stop)

    def test_exact_custom_request_is_self_directed(self):
        intent = parse_self_custom_flair_request(
            f'Hey Optimist Prime, set my flair to "{ALLOWED_LABEL}".'
        )
        self.assertEqual(intent, {"action": "set", "label": ALLOWED_LABEL})

    def test_rejects_suggestions_and_values_without_exact_wording(self):
        self.assertIsNone(parse_self_custom_flair_request(
            "Hey Optimist Prime, suggest a custom flair for me."
        ))
        self.assertIsNone(parse_self_custom_flair_request(
            "Hey Optimist Prime, give me a custom flair that suits me."
        ))

    def test_bot_username_mentions_are_not_third_party_targets(self):
        self.assertEqual(
            parse_self_custom_flair_request(
                "u/example_bot, set my flair to Community volunteer"
            ),
            {"action": "set", "label": ALLOWED_LABEL},
        )
        self.assertEqual(
            parse_self_custom_flair_request(
                r"[u/example_bot]"
                r"(https://www.reddit.com/user/example_bot/) "
                "set my flair to Community volunteer"
            ),
            {"action": "set", "label": ALLOWED_LABEL},
        )

    def test_rejects_third_party_and_unrelated_requests(self):
        self.assertIsNone(parse_self_custom_flair_request(
            f'Hey Optimist Prime, set u/example_user\'s flair to "{ALLOWED_LABEL}".'
        ))
        self.assertIsNone(parse_self_custom_flair_request(
            "My flair is visible next to my name."
        ))

    def test_quoted_hypothetical_and_negated_requests_do_not_trigger(self):
        for text in (
            f'I saw someone write "set my flair to {ALLOWED_LABEL}" as an example.',
            f"Don't set my flair to {ALLOWED_LABEL}.",
            "I don't want a custom flair for me.",
            "What if I asked you to set a custom flair for me?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_self_custom_flair_request(text))

    def test_overlong_label_is_routed_to_clear_feedback(self):
        intent = parse_self_custom_flair_request(
            'Hey Optimist Prime, set my flair to "A label that is definitely far too long".'
        )
        self.assertEqual(intent, {"action": "invalid"})


class TestSelfFlairRequestHandler(unittest.TestCase):
    def setUp(self):
        bot_identity = patch.object(config, "EXPECTED_BOT_USERNAME", "example_bot")
        bot_identity.start()
        self.addCleanup(bot_identity.stop)
        self.comment = SimpleNamespace(author=SimpleNamespace(name="example_user"))
        self.subreddit = SimpleNamespace(display_name="example_community")

    def test_exact_allowlisted_value_is_applied_without_model_review(self):
        class NoModelCall:
            def decide_many(self, **_kwargs):
                raise AssertionError("Allowlisted values need no model review")

        state = {}
        with patch.object(config, "USER_CUSTOM_FLAIR_ENABLED", True), \
             patch.object(config, "USER_CUSTOM_FLAIR_ALLOWED_VALUES", (ALLOWED_LABEL,)), \
             patch("self_flair_requests.set_user_flair", return_value=True) as apply:
            reply, tokens, cost, updated = handle_self_custom_flair_request(
                self.comment,
                self.subreddit,
                object(),
                object(),
                state,
                {"action": "set", "label": ALLOWED_LABEL},
                jev_client=NoModelCall(),
            )

        self.assertIn(ALLOWED_LABEL, reply)
        self.assertEqual((tokens, cost), (0, 0.0))
        self.assertIs(updated, state)
        apply.assert_called_once_with(
            self.subreddit,
            "example_user",
            ALLOWED_LABEL,
            dry_run=False,
            state=state,
        )

    def test_empty_allowlist_rejects_the_value_without_any_flair_change(self):
        class NoModelCall:
            def decide_many(self, **_kwargs):
                raise AssertionError("An unset allowlist must fail closed")

        with patch.object(config, "USER_CUSTOM_FLAIR_ENABLED", True), \
             patch.object(config, "USER_CUSTOM_FLAIR_ALLOWED_VALUES", ()), \
             patch("self_flair_requests.set_user_flair") as apply:
            reply, _tokens, _cost, _state = handle_self_custom_flair_request(
                self.comment,
                self.subreddit,
                object(),
                object(),
                {},
                {"action": "set", "label": ALLOWED_LABEL},
                jev_client=NoModelCall(),
            )

        self.assertIn("not configured", reply)
        apply.assert_not_called()

    def test_non_exact_value_is_rejected_even_when_a_similar_value_is_allowed(self):
        with patch.object(config, "USER_CUSTOM_FLAIR_ENABLED", True), \
             patch.object(config, "USER_CUSTOM_FLAIR_ALLOWED_VALUES", (ALLOWED_LABEL,)), \
             patch("self_flair_requests.set_user_flair") as apply:
            reply, _tokens, _cost, _state = handle_self_custom_flair_request(
                self.comment,
                self.subreddit,
                object(),
                object(),
                {},
                {"action": "set", "label": ALLOWED_LABEL + " extra"},
            )

        self.assertIn("choose one of these configured flair labels", reply)
        apply.assert_not_called()

    def test_disabled_self_flair_makes_no_flair_change(self):
        with patch.object(config, "USER_CUSTOM_FLAIR_ENABLED", False), \
             patch.object(config, "USER_CUSTOM_FLAIR_ALLOWED_VALUES", (ALLOWED_LABEL,)), \
             patch("self_flair_requests.set_user_flair") as apply:
            reply, *_ = handle_self_custom_flair_request(
                self.comment,
                self.subreddit,
                object(),
                object(),
                {},
                {"action": "set", "label": ALLOWED_LABEL},
            )

        self.assertIn("disabled", reply)
        apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
