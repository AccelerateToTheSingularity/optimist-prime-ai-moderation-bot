"""Tests for content_moderation.py parsing logic."""

import unittest
from unittest.mock import MagicMock, patch

from llm_client import LLMQuotaExhausted
from content_moderation import (
    IMPLEMENTED_ACTIONS,
    evaluate_rules,
    handle_moderation_action,
    _rule_eval_max_output_tokens,
)
from moderation_rules import ModerationRule
from ordered_gate import QUESTION_NAME


class TestModerationActions(unittest.TestCase):
    def test_unsupported_action_returns_false(self):
        subreddit = MagicMock()
        content = MagicMock()
        content.id = "source1"
        content.permalink = "/r/test/1"
        result = handle_moderation_action(
            subreddit, content, "user", "reason", "unsupported_action", dry_run=False,
        )
        self.assertFalse(result)

    def test_supported_actions_set(self):
        self.assertIn("report", IMPLEMENTED_ACTIONS)
        self.assertNotIn("unsupported_action", IMPLEMENTED_ACTIONS)

    def test_optional_public_removal_explanation_is_gated(self):
        subreddit = MagicMock()
        content = MagicMock()
        content.permalink = "/r/test/1"
        content.title = "A post"
        content.reply.return_value = MagicMock(
            id="reply1", permalink="/r/test/comments/reply1"
        )
        with patch("config.CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED", True):
            self.assertTrue(
                handle_moderation_action(
                    subreddit, content, "user", "Reason", "remove", dry_run=False,
                    state={}, bot_username="OptimistPrime",
                    persist_state=lambda _state: None,
                )
            )
        content.mod.remove.assert_called_once_with()
        content.reply.assert_called_once()

    def test_failed_optional_removal_explanation_does_not_retry_removal(self):
        subreddit = MagicMock()
        content = MagicMock()
        content.id = "source2"
        content.permalink = "/r/test/1"
        content.title = "A post"
        content.reply.side_effect = RuntimeError("temporary reply failure")
        with patch("config.CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED", True):
            self.assertTrue(
                handle_moderation_action(
                    subreddit, content, "user", "Reason", "remove", dry_run=False,
                    state={}, bot_username="OptimistPrime",
                    persist_state=lambda _state: None,
                )
            )
        content.mod.remove.assert_called_once_with()

    def test_moderation_modmail_does_not_make_an_extra_ai_summary_call_by_default(self):
        subreddit = MagicMock()
        content = MagicMock()
        content.permalink = "/r/test/1"
        with patch("config.MODERATION_AI_SUMMARY_ENABLED", False), \
             patch("mod_attention.generate_mod_attention_summary") as summary:
            self.assertTrue(
                handle_moderation_action(
                    subreddit,
                    content,
                    "user",
                    "Reason",
                    "modmail",
                    dry_run=False,
                    llm_model=object(),
                )
            )
        summary.assert_not_called()
        subreddit.message.assert_called_once()


class TestRuleEvalTokens(unittest.TestCase):
    def test_scales_with_rule_count(self):
        self.assertEqual(_rule_eval_max_output_tokens(10), 500)
        self.assertEqual(_rule_eval_max_output_tokens(21), 840)

    def test_evidence_quota_preserves_selected_rule_for_retry(self):
        class FakeJev:
            ready = True
            calls_used = 0
            max_calls = 10
            last_usage = {"input_tokens": 5, "cost": 0.0}

            def choose_many(self, *, state, questions):
                self.calls_used += 1
                return {QUESTION_NAME: "ordered_rule_1"}

        class QuotaModel:
            def generate_content(self, contents, generation_config=None):
                raise LLMQuotaExhausted("limit reached")
        rules = [ModerationRule(
            name="spam_detector", description="spam?",
            conditions={"stop_on_match": True},
        )]
        from ordered_gate import ModerationEvidencePending
        with self.assertRaises(ModerationEvidencePending) as raised:
            evaluate_rules("content", rules, QuotaModel(), jev_client=FakeJev())
        self.assertEqual(
            raised.exception.pending_selection["selected_rule_names"],
            ["spam_detector"],
        )


if __name__ == "__main__":
    unittest.main()
