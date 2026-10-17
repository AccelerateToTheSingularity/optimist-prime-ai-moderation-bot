"""Tests for action idempotency helpers."""

import unittest
from unittest.mock import patch

import config

from bot_utils import (
    begin_action,
    claim_action,
    complete_action,
    fail_action,
    has_action_claim,
    is_bot_owned_author,
    is_likely_bot,
    get_daily_non_moderator_reply_count,
    record_reply_success,
)


class TestClaimAction(unittest.TestCase):
    def test_first_claim_succeeds(self):
        state = {}
        self.assertTrue(claim_action(state, "tldr:t3_abc"))
        self.assertIn("tldr:t3_abc", state["action_keys"])

    def test_second_claim_fails(self):
        state = {}
        self.assertTrue(claim_action(state, "mod:t1_x"))
        self.assertFalse(claim_action(state, "mod:t1_x"))
        self.assertTrue(has_action_claim(state, "mod:t1_x"))

    def test_action_is_only_completed_after_success(self):
        state = {}
        self.assertTrue(begin_action(state, "reply:t1_x"))
        self.assertFalse(has_action_claim(state, "reply:t1_x"))
        fail_action(state, "reply:t1_x", "temporary API error")
        self.assertFalse(has_action_claim(state, "reply:t1_x"))
        self.assertTrue(begin_action(state, "reply:t1_x"))
        complete_action(state, "reply:t1_x")
        self.assertTrue(has_action_claim(state, "reply:t1_x"))


class TestBotOwnedAuthor(unittest.TestCase):
    def test_runtime_identity_is_owned(self):
        self.assertTrue(is_bot_owned_author("another_bot_account", "another_bot_account"))

    def test_legacy_aliases_are_owned(self):
        with patch.object(config, "BOT_OWNED_USERNAMES", ["legacy_example_account"]):
            self.assertTrue(
                is_bot_owned_author("legacy_example_account", "different-runtime-name")
            )
            self.assertTrue(
                is_bot_owned_author("LEGACY_EXAMPLE_ACCOUNT", "different-runtime-name")
            )

    def test_runtime_identity_is_case_insensitive(self):
        self.assertTrue(is_bot_owned_author("Future-App-Name", "future-app-name"))

    def test_regular_user_is_not_owned(self):
        self.assertFalse(is_bot_owned_author("ordinary-user", "another_bot_account"))

    def test_migrated_name_is_likely_a_bot(self):
        self.assertTrue(is_likely_bot("helperbot"))

    def test_runtime_identity_is_likely_a_bot(self):
        self.assertTrue(is_likely_bot("future-runtime-name", "future-runtime-name"))
        self.assertFalse(is_likely_bot("ordinary-user", "future-runtime-name"))

    def test_simulated_automated_reply_chain_stops_immediately(self):
        processed = 0
        for _ in range(20):
            if is_likely_bot("helperbot", "future-runtime-name"):
                break
            processed += 1
        self.assertEqual(processed, 0)


class TestReplyAccounting(unittest.TestCase):
    def test_legacy_combined_counter_is_conservative_until_migrated(self):
        self.assertEqual(
            get_daily_non_moderator_reply_count({"daily_replies": 7}),
            7,
        )

    def test_moderator_reply_counts_in_total_but_not_ordinary_user_allowance(self):
        state = {"daily_replies": 4, "daily_non_moderator_replies": 4}
        record_reply_success(state, moderator=True)
        self.assertEqual(state["daily_replies"], 5)
        self.assertEqual(state["daily_non_moderator_replies"], 4)

    def test_ordinary_user_reply_counts_in_both_counters(self):
        state = {"daily_replies": 4, "daily_non_moderator_replies": 4}
        record_reply_success(state, moderator=False)
        self.assertEqual(state["daily_replies"], 5)
        self.assertEqual(state["daily_non_moderator_replies"], 5)


if __name__ == "__main__":
    unittest.main()
