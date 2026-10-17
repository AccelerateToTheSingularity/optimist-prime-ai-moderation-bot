"""Focused runner safety regressions."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from bot_runner import (
    canary_is_dry_run,
    is_expected_bot_account,
    is_test_comment_allowed,
    load_state,
    normalize_target_post_id,
    _post_scan_is_needed,
    _comment_scan_is_needed,
    _preview_tldr_and_discussion_summary,
    _preview_history_review,
    _moderation_input_fingerprint,
    _pending_moderation_selection,
    _save_pending_moderation_selection,
    _ModerationEvidenceBudgetModel,
    live_account_configuration_error,
)
from content_moderation import evaluate_rules
from llm_client import LLMQuotaExhausted
from moderation_rules import ModerationRule
from ordered_gate import ModerationEvidencePending, NONE_LABEL, QUESTION_NAME


class TestRunnerSafety(unittest.TestCase):
    @staticmethod
    def _runtime(**overrides):
        values = {
            "post_tldr_enabled": False,
            "comment_tldr_enabled": False,
            "comment_summary_enabled": False,
            "max_tldr_per_run": 0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_test_comment_is_blocked_by_dry_run_or_safe_mode(self):
        self.assertFalse(is_test_comment_allowed(True))

    def test_pending_moderation_decision_persists_only_for_unchanged_item_text(self):
        state = {}
        body = "synthetic moderation fixture text"
        fingerprint = _moderation_input_fingerprint(body)
        selection = {
            "rules_fingerprint": "synthetic-rules-fingerprint",
            "decisions": {"synthetic_rule": True},
            "selected_rule_names": ["synthetic_rule"],
        }
        _save_pending_moderation_selection(
            state, "comments", "fixture-id", fingerprint, selection
        )

        self.assertEqual(
            _pending_moderation_selection(state, "comments", "fixture-id", fingerprint),
            selection,
        )
        self.assertNotIn(body, json.dumps(state))
        self.assertIsNone(
            _pending_moderation_selection(
                state, "comments", "fixture-id",
                _moderation_input_fingerprint("edited fixture text"),
            )
        )

    def test_test_comment_requires_live_mode(self):
        self.assertTrue(is_test_comment_allowed(False))

    def test_live_mode_requires_an_expected_authenticated_account(self):
        self.assertIsNotNone(
            live_account_configuration_error(expected_username="", safe_mode=False)
        )
        self.assertIsNone(
            live_account_configuration_error(expected_username="example_bot", safe_mode=False)
        )
        self.assertIsNone(
            live_account_configuration_error(expected_username="", safe_mode=True)
        )

    def test_canary_normalizes_t3_prefix_and_requires_the_expected_account(self):
        self.assertEqual(normalize_target_post_id(" t3_test01 "), "test01")
        self.assertTrue(
            is_expected_bot_account(
                "example_bot", "example_bot"
            )
        )
        self.assertFalse(is_expected_bot_account("another-account", "example_bot"))

    def test_canary_is_preview_until_explicit_live_switch_and_safe_mode_wins(self):
        self.assertTrue(canary_is_dry_run(canary_mode=True, live_once=False, safe_mode=False))
        self.assertFalse(canary_is_dry_run(canary_mode=True, live_once=True, safe_mode=False))
        self.assertTrue(canary_is_dry_run(canary_mode=True, live_once=True, safe_mode=True))

    def test_state_migration_adds_missing_fields_without_dropping_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(
                json.dumps({
                    "future_field": {"keep": True},
                    "stats": {"total_tldrs_generated": 7},
                }),
                encoding="utf-8",
            )
            state = load_state(path)
        self.assertEqual(state["future_field"], {"keep": True})
        self.assertEqual(state["stats"]["total_tldrs_generated"], 7)
        self.assertIn("total_cost", state["stats"])
        self.assertIn("reported_user_reviews", state)

    def test_content_preview_generates_both_outputs_without_reddit_writes(self):
        # Use an ordinary list-like container so PRAW's top-level comment
        # iteration behavior is represented without a network connection.
        class FakeComments:
            def __iter__(self):
                return iter([SimpleNamespace(body=f"discussion point {i}") for i in range(6)])

            def replace_more(self, **kwargs):
                self.asserted_limit = kwargs["limit"]

        post = SimpleNamespace(
            id="preview123",
            title="A sample long-form AI post",
            selftext="useful content " * 300,
            num_comments=25,
            created_utc=time.time(),
            permalink="/r/example_community/comments/preview123/sample/",
            comments=FakeComments(),
        )
        subreddit = SimpleNamespace(new=lambda limit: [post])
        usage = {"total_tokens": 10, "cost": 0.0}
        output = StringIO()
        with patch("bot_runner.config.POST_TLDR_ENABLED", True), \
             patch("bot_runner.config.COMMENT_SUMMARY_ENABLED", True), \
             patch("bot_runner.config.COMMENT_MILESTONES", (10, 20)), \
             patch("bot_runner.generate_tldr", return_value=("preview TLDR", usage)) as tldr, \
             patch("bot_runner.generate_comment_summary", return_value=("preview summary", usage)) as summary, \
             redirect_stdout(output):
            result = _preview_tldr_and_discussion_summary(subreddit, object())

        self.assertEqual(result, 0)
        tldr.assert_called_once()
        summary.assert_called_once()
        self.assertEqual(post.comments.asserted_limit, 1)
        self.assertIn("preview TLDR", output.getvalue())
        self.assertIn("preview summary", output.getvalue())
        self.assertIn("nothing was posted, pinned, or saved", output.getvalue())
        self.assertFalse(hasattr(post, "reply"))

    def test_history_preview_reuses_report_analysis_without_persisting_state(self):
        reddit = SimpleNamespace(subreddit=lambda name: SimpleNamespace(name=name))
        model = SimpleNamespace(calls_used=1)
        jev = SimpleNamespace(calls_used=2)
        with patch("bot_runner.handle_moderator_history_request", return_value=(
            "preview table", 123, 0.004, {"must_not_be_saved": True}
        )) as handle:
            result = _preview_history_review(
                reddit,
                model,
                target_username="example_user",
                requester_username="example_bot",
                llm_model_name="provider/example-model",
                jev_client=jev,
            )

        self.assertEqual(result, ("preview table", 123, 0.004, 1, 2))
        self.assertEqual(handle.call_args.args[4], {})
        self.assertEqual(handle.call_args.args[5], {
            "needs_history": True, "target_username": "example_user"
        })
        self.assertTrue(handle.call_args.kwargs["dry_run"])
        self.assertTrue(handle.call_args.kwargs["diagnostic"])
        self.assertEqual(
            handle.call_args.kwargs["analysis_model_name"],
            "provider/example-model",
        )
        self.assertNotIn("persist_state", handle.call_args.kwargs)

    def test_idle_cycle_skips_broad_post_scan(self):
        with patch("bot_runner.config.CONTENT_MODERATION_ENABLED", False):
            self.assertFalse(_post_scan_is_needed(self._runtime()))

    def test_zero_moderation_evidence_budget_still_allows_jev_no_match(self):
        class Jev:
            ready = True
            max_calls = 1
            calls_used = 0
            last_usage = {}

            def choose_many(self, *, state, questions):
                self.calls_used += 1
                return {QUESTION_NAME: NONE_LABEL}

        class ForbiddenEvidenceModel:
            def generate_content(self, *_args, **_kwargs):
                raise AssertionError("No-match Jev decision needs no explanation call")

        jev = Jev()
        rules = [ModerationRule(
            name="spam_detector",
            description="Does the author post unsolicited spam?",
            conditions={"stop_on_match": True},
        )]
        result = evaluate_rules(
            "ordinary discussion",
            rules,
            _ModerationEvidenceBudgetModel(ForbiddenEvidenceModel(), max_calls=0),
            jev_client=jev,
        )

        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(result["matches"], [])
        self.assertFalse(result["decisions"]["spam_detector"]["matched"])

    def test_jev_positive_reaches_the_evidence_cap_only_after_classification(self):
        class Jev:
            ready = True
            max_calls = 1
            calls_used = 0
            last_usage = {}

            def choose_many(self, *, state, questions):
                self.calls_used += 1
                return {QUESTION_NAME: "ordered_rule_1"}

        class CountingEvidenceModel:
            calls = 0

            def generate_content(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("zero cap must prevent the explanation request")

        jev = Jev()
        evidence_model = CountingEvidenceModel()
        rules = [ModerationRule(
            name="spam_detector",
            description="Does the author post unsolicited spam?",
            conditions={"stop_on_match": True},
        )]
        with self.assertRaises(ModerationEvidencePending) as raised:
            evaluate_rules(
                "ordinary discussion",
                rules,
                _ModerationEvidenceBudgetModel(evidence_model, max_calls=0),
                jev_client=jev,
            )

        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(evidence_model.calls, 0)
        self.assertEqual(raised.exception.pending_selection["selected_rule_names"], ["spam_detector"])

    def test_moderation_evidence_budget_counts_successful_calls(self):
        class EvidenceModel:
            def __init__(self):
                self.calls = 0

            def generate_content(self, contents, generation_config=None):
                self.calls += 1
                return "result"

        inner = EvidenceModel()
        model = _ModerationEvidenceBudgetModel(inner, max_calls=1)
        self.assertEqual(model.generate_content("prompt"), "result")
        self.assertEqual(model.calls_used, 1)
        with self.assertRaises(LLMQuotaExhausted):
            model.generate_content("second prompt")
        self.assertEqual(inner.calls, 1)

    def test_broad_post_scan_is_kept_for_a_consuming_phase(self):
        with patch("bot_runner.config.CONTENT_MODERATION_ENABLED", False):
            self.assertTrue(
                _post_scan_is_needed(
                    self._runtime(post_tldr_enabled=True, max_tldr_per_run=1)
                )
            )



if __name__ == "__main__":
    unittest.main()
