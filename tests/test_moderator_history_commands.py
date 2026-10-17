"""Tests for explicit moderator history reports."""

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config
from history_gate_jev import HistoryGateAssessment, JevHistoryReviewError, assess_history_nouls_with_jev
from history_criteria_test_support import (
    ConfiguredHistoryCriteriaTestCase,
    synthetic_history_criteria,
    synthetic_history_criteria_rules,
    scores_for_questions,
)
from history_review_ledger import HistoryReviewLedger
from llm_client import LLMIncompleteOutput, LLMQuotaExhausted, wrap_with_rate_limit
from moderator_history_commands import (
    _evaluate_history_batch_with_fallback,
    _merge_chunk_reports,
    _display_rule_name,
    _format_history_report,
    handle_moderator_history_request,
    parse_moderator_history_request,
)
from reported_user_review import HistoryFinding
from user_history import HistoryItem, UserHistory


def _history(comment_count=1, post_count=0):
    now = datetime(2026, 9, 28, tzinfo=timezone.utc).timestamp()
    items = tuple(
        HistoryItem(
            item_id=f"c{index}",
            kind="comment",
            subreddit="example_community",
            created_utc=now - index,
            score=1,
            permalink=f"https://www.reddit.com/r/example_community/comments/c{index}/",
            text=f"Comment {index}",
        )
        for index in range(comment_count)
    ) + tuple(
        HistoryItem(
            item_id=f"p{index}",
            kind="post",
            subreddit="example_other_community",
            created_utc=now - 1000 - index,
            score=2,
            permalink=f"https://reddit.test/posts/p{index}",
            text=f"Post {index}",
        )
        for index in range(post_count)
    )
    return UserHistory("example_user", items, comment_count, post_count)


class TestModeratorHistoryRequestParser(unittest.TestCase):
    def test_criterion_display_name_is_formatted_for_readers(self):
        self.assertEqual(
            _display_rule_name("sample_repeated_link_posting"),
            "Sample Repeated Link Posting",
        )
        self.assertEqual(
            _display_rule_name("sample_direct_personal_attack"),
            "Sample Direct Personal Attack",
        )
        self.assertEqual(
            _display_rule_name("sample_contact_information"),
            "Sample Contact Information",
        )

    def test_history_review_limit_stays_within_supported_bounds(self):
        with patch.object(config, "HISTORY_REVIEW_MAX_ITEMS", 250):
            self.assertEqual(config.moderator_history_report_caps(), (250, 250))

    def test_chunk_report_synthesis_forbids_boilerplate_disclaimers(self):
        class Model:
            prompt = ""

            def generate_content(self, prompt, generation_config=None):
                self.prompt = prompt
                return SimpleNamespace(
                    text='{"summary":"Balanced summary.","answer":"","themes":[]}',
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=1, candidates_token_count=1
                    ),
                )

        model = Model()
        _merge_chunk_reports(
            "example_user", _history(), [], [{"summary": "Balanced summary."}], [], None, model
        )
        self.assertIn("Do not add generic disclaimers", model.prompt)
        self.assertIn("Do not list recurring interests", model.prompt)

    def test_incomplete_large_report_splits_and_preserves_evidence_indexes(self):
        history = _history(12)
        calls = []

        def evaluate(_username, _reported, batch, _rules, _model, **_kwargs):
            calls.append(len(batch.items))
            if len(batch.items) > 6:
                raise LLMIncompleteOutput(
                    "provider cut off report", reason="output_limit"
                )
            first = batch.items[0]
            return [HistoryFinding(
                evidence_index=0,
                item_id=first.item_id,
                kind=first.kind,
                subreddit=first.subreddit,
                permalink=first.permalink,
                timestamp="2026-09-28",
                score=first.score,
                excerpt=first.text,
                rule_name="example_rule",
                confidence=0.9,
                explanation="Evidence",
            )], {"history_summary": "summary", "total_tokens": 1, "cost": 0.0}

        with patch("moderator_history_commands.evaluate_history", side_effect=evaluate):
            parts = _evaluate_history_batch_with_fallback(
                "example_user", history, [], object(), None
            )

        self.assertEqual(calls, [12, 6, 6])
        findings = [finding for part, _usage in parts for finding in part]
        self.assertEqual([finding.evidence_index for finding in findings], [0, 6])

    def test_incomplete_small_report_is_not_retried_or_split(self):
        with patch(
            "moderator_history_commands.evaluate_history",
            side_effect=LLMIncompleteOutput("empty final", reason="empty_final_answer"),
        ) as evaluate:
            with self.assertRaises(LLMIncompleteOutput):
                _evaluate_history_batch_with_fallback(
                    "example_user", _history(10), [], object(), None
                )
        evaluate.assert_called_once()

    def test_generic_scan_uses_parent_target_and_has_no_narrow_question(self):
        intent = parse_moderator_history_request(
            "Hey mod bot, scan this guy's history for me."
        )
        self.assertIsNotNone(intent)
        self.assertTrue(intent["needs_history"])
        self.assertIsNone(intent["target_username"])
        self.assertIsNone(intent["question"])

    def test_private_assistant_context_accepts_short_named_scan(self):
        intent = parse_moderator_history_request(
            "scan u/example_user", explicit_bot_context=True
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent["target_username"], "example_user")
        self.assertTrue(intent["needs_history"])

    def test_private_assistant_context_accepts_username_without_u_prefix(self):
        intent = parse_moderator_history_request(
            "Scan example_user", explicit_bot_context=True
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent["target_username"], "example_user")
        self.assertTrue(intent["needs_history"])

    def test_bare_target_does_not_treat_generic_words_as_usernames(self):
        for body in ("Scan this person's history", "Scan username", "Review someone"):
            with self.subTest(body=body):
                intent = parse_moderator_history_request(body, explicit_bot_context=True)
                if body == "Scan this person's history":
                    self.assertIsNone(intent["target_username"])
                    self.assertTrue(intent["needs_history"])
                else:
                    self.assertIsNone(intent)

    def test_private_assistant_context_does_not_turn_unrelated_chat_into_a_scan(self):
        self.assertIsNone(parse_moderator_history_request(
            "That user's history is interesting", explicit_bot_context=True
        ))

    def test_explicit_user_and_question_are_preserved(self):
        intent = parse_moderator_history_request(
            "Hey mod bot, scan u/example_user's history. I want to know if they repeatedly post the same link."
        )
        self.assertEqual(intent["target_username"], "example_user")
        self.assertIn("repeatedly post the same link", intent["question"])

    def test_direct_question_is_recognized(self):
        intent = parse_moderator_history_request(
            "Hey mod bot, tell me whether this person repeatedly posts the same link."
        )
        self.assertTrue(intent["needs_history"])
        self.assertIn("repeatedly posts the same link", intent["question"])

    def test_ordinary_discussion_phrases_do_not_become_history_commands(self):
        examples = (
            "You can scan everything in your body today. You cannot prevent many diseases. "
            "Genes, family history. No one can prevent them, not even AI.",
            "It is the user's problem. That explanation is confusing; reconsider the instructions.",
            "Hey mod bot, what hobbies does this person enjoy?",
        )
        for body in examples:
            with self.subTest(body=body):
                self.assertIsNone(parse_moderator_history_request(body))
class TestModeratorHistoryRequestHandling(ConfiguredHistoryCriteriaTestCase):
    def setUp(self):
        super().setUp()
        self.threshold_patch = patch.object(config, "HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT", 67)
        self.threshold_patch.start()
        self.addCleanup(self.threshold_patch.stop)
        self.lookback_patch = patch.object(config, "HISTORY_REVIEW_LOOKBACK_DAYS", 90)
        self.lookback_patch.start()
        self.addCleanup(self.lookback_patch.stop)
        self.ledger_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.ledger_dir.cleanup)
        self.review_ledger = HistoryReviewLedger(
            Path(self.ledger_dir.name) / "history-review.sqlite3"
        )
        self.ledger_patch = patch(
            "moderator_history_commands.HistoryReviewLedger",
            return_value=self.review_ledger,
        )
        self.ledger_patch.start()
        self.addCleanup(self.ledger_patch.stop)
        self.criteria_patches = [
            patch(
                "moderator_history_commands.load_history_criteria_definitions",
                return_value=synthetic_history_criteria(),
            ),
            patch(
                "history_gate_jev.load_history_criteria",
                return_value=synthetic_history_criteria(),
            ),
        ]
        for patcher in self.criteria_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_non_moderator_is_refused_before_history_is_read(self):
        intent = parse_moderator_history_request("Mod bot, scan u/example_user's history")
        with patch("moderator_history_commands.collect_user_history") as collect:
            reply, _tokens, _cost, state = handle_moderator_history_request(
                SimpleNamespace(author=SimpleNamespace(name="member")),
                object(), object(), object(), {}, intent,
                requester_is_moderator=False,
                parent_author=None,
            )
        self.assertIn("moderators only", reply)
        self.assertEqual(state, {})
        collect.assert_not_called()

    def test_missing_parent_target_asks_instead_of_guessing(self):
        intent = parse_moderator_history_request("Mod bot, scan this person's history")
        with patch("moderator_history_commands.collect_user_history") as collect:
            reply, *_ = handle_moderator_history_request(
                SimpleNamespace(author=SimpleNamespace(name="mod")),
                object(), object(), object(), {}, intent,
                requester_is_moderator=True,
                parent_author=None,
            )
        self.assertIn("reply directly", reply)
        collect.assert_not_called()

    def test_moderator_can_explicitly_request_a_report_about_their_own_account(self):
        intent = parse_moderator_history_request(
            "Hey Optimist Prime, scan u/example_mod's history for me.",
            explicit_bot_context=True,
        )
        comment = SimpleNamespace(author=SimpleNamespace(name="example_mod"))
        with self.assertRaisesRegex(JevHistoryReviewError, "unavailable"):
            handle_moderator_history_request(
                comment,
                object(),
                object(),
                object(),
                {},
                intent,
                requester_is_moderator=True,
                parent_author=None,
                jev_client=SimpleNamespace(ready=False),
            )

    def test_empty_criteria_stops_before_collecting_history(self):
        intent = parse_moderator_history_request(
            "Mod bot, scan u/example_user's history. Tell me whether they repeatedly post the same link."
        )
        with patch("moderator_history_commands.load_history_criteria_definitions", return_value={}), \
             patch("moderator_history_commands.collect_user_history") as collect:
            with self.assertRaisesRegex(JevHistoryReviewError, "No active operator-defined history criteria"):
                handle_moderator_history_request(
                    SimpleNamespace(author=SimpleNamespace(name="example_mod")),
                    object(), object(), object(), {}, intent,
                    requester_is_moderator=True,
                    parent_author=None,
                    jev_client=SimpleNamespace(ready=True, max_calls=3, calls_used=0),
                )
        collect.assert_not_called()

    def test_explicit_target_overrides_parent_and_report_uses_one_existing_history_scan(self):
        intent = parse_moderator_history_request(
            "Mod bot, scan u/example_user's history. Tell me whether they repeatedly post the same link."
        )
        history = _history()
        finding = HistoryFinding(
            evidence_index=0,
            item_id="c0",
            kind="comment",
            subreddit="example_community",
            permalink=history.items[0].permalink,
            timestamp=datetime.fromtimestamp(history.items[0].created_utc, timezone.utc).isoformat(),
            score=1,
            excerpt="I repeatedly posted the same link to unrelated threads.",
            rule_name="sample_repeated_link_posting",
            confidence=0.9,
            explanation="The fictional author directly described repeating the same link.",
        )
        jev_calls = []

        def decide_many(*, state, questions):
            jev_calls.append((state, questions))
            return scores_for_questions(questions)

        jev_client = SimpleNamespace(
            ready=True,
            calls_used=0,
            max_calls=1,
            last_usage={"input_tokens": 12, "cost": 0.0002},
            served_model="typesafe/jev-test",
            decide_many=decide_many,
        )
        with patch("moderator_history_commands.load_history_subreddits", return_value=frozenset({"example_community", "example_other_community"})), \
             patch("moderator_history_commands.collect_user_history", return_value=history) as collect, \
             patch("moderator_history_commands.evaluate_history", return_value=(
                 [finding],
                 {
                     "history_summary": "Recurring pattern of personal insults.",
                     "history_themes": ["community gardening"],
                     "history_answer": "The sample describes repeated personal insults.",
                     "total_tokens": 75,
                     "cost": 0.001,
                 },
             )) as evaluate, \
             patch("moderator_history_commands._merge_chunk_reports") as merge:
            report, tokens, cost, _state = handle_moderator_history_request(
                SimpleNamespace(author=SimpleNamespace(name="example_mod")),
                object(), object(), object(), {}, intent,
                requester_is_moderator=True,
                parent_author="different_parent",
                jev_client=jev_client,
            )

        self.assertIn("No configured criterion matched in the reviewed history sample.", report)
        self.assertNotIn("ban", report.casefold())
        self.assertNotIn("History report for u/example_user", report)
        self.assertNotIn("| Configured criterion |", report)
        self.assertEqual(len(jev_calls[0][1]), 3)
        self.assertTrue(all(question["type"] == "noul" for question in jev_calls[0][1].values()))
        self.assertIn(f"{len(history.comments)} comment", report)
        self.assertNotIn("**Criterion-specific matches:**", report)
        self.assertNotIn("Sample Direct Personal Attack", report)
        self.assertNotIn("90% evidence confidence", report)
        self.assertNotIn("Recurring pattern of personal insults", report)
        self.assertNotIn("Closest scores", report)
        self.assertNotIn("Overall view", report)
        self.assertNotIn("history analysis", report)
        self.assertNotIn("average", report.casefold())
        self.assertNotIn("Recurring interests", report)
        self.assertNotIn(history.items[0].permalink, report)
        self.assertEqual(tokens, 87)
        self.assertAlmostEqual(cost, 0.0012)
        self.assertEqual(collect.call_args.args[1], "example_user")
        self.assertEqual(collect.call_args.kwargs["max_comments"], config.HISTORY_REVIEW_MAX_ITEMS)
        self.assertEqual(collect.call_args.kwargs["max_posts"], config.HISTORY_REVIEW_MAX_ITEMS)
        self.assertEqual(collect.call_args.kwargs["max_items"], config.HISTORY_REVIEW_MAX_ITEMS)
        self.assertTrue(collect.call_args.kwargs["prefer_posts"])
        self.assertEqual(
            collect.call_args.kwargs["max_scan_comments"],
            config.HISTORY_REVIEW_MAX_ITEMS,
        )
        self.assertEqual(
            collect.call_args.kwargs["max_scan_posts"],
            config.HISTORY_REVIEW_MAX_ITEMS,
        )
        self.assertEqual(collect.call_args.kwargs["allowed_subreddits"], frozenset({"example_community", "example_other_community"}))
        self.assertEqual(evaluate.call_args.kwargs["question"], intent["question"])
        self.assertEqual(len(jev_calls), 1)
        self.assertEqual(len(jev_calls[0][1]), 3)
        self.assertEqual(len(jev_calls[0][0]["reddit_history"]), 1)
        merge.assert_not_called()

    def test_two_call_profile_analyzes_full_capped_history_in_one_call_without_bypassing_limit(self):
        history = _history(80, 20)
        model = wrap_with_rate_limit(
            SimpleNamespace(generate_content=lambda *_args, **_kwargs: None), 2
        )
        intent = {"needs_history": True, "question": None}
        batch_sizes = []

        def evaluate(_user, _reports, batch, _rules, _model, **_kwargs):
            batch_sizes.append(len(batch.items))
            return [], {
                "history_summary": "A broad evidence-based summary.",
                "history_themes": ["gardening"],
                "history_answer": "",
                "total_tokens": 100,
                "cost": 0.0,
            }

        jev_calls = []

        def decide_many(*, state, questions):
            jev_calls.append((state, questions))
            return scores_for_questions(
                questions, positives=("sample_repeated_link_posting",), positive_score=0.93
            )

        jev_client = SimpleNamespace(
            ready=True,
            calls_used=0,
            max_calls=1,
            last_usage={"input_tokens": 1, "cost": 0.0001},
            served_model="typesafe/jev-test",
            decide_many=decide_many,
        )
        rules = synthetic_history_criteria_rules()
        with patch("moderator_history_commands.load_history_subreddits", return_value=frozenset({"example_community"})), \
             patch("moderator_history_commands.collect_user_history", return_value=history), \
             patch("moderator_history_commands.evaluate_history", side_effect=evaluate), \
             patch("moderator_history_commands._merge_chunk_reports") as merge:
            report, *_ = handle_moderator_history_request(
                SimpleNamespace(author=SimpleNamespace(name="mod")),
                object(), object(), model, {}, intent,
                requester_is_moderator=True,
                parent_author="example_user",
                jev_client=jev_client,
            )

        self.assertEqual(batch_sizes, [config.HISTORY_REVIEW_MAX_ITEMS])
        self.assertEqual(len(jev_calls), 1)
        self.assertEqual(len(jev_calls[0][1]), 3)
        self.assertTrue(all(question["type"] == "noul" for question in jev_calls[0][1].values()))
        self.assertEqual(
            len(jev_calls[0][0]["reddit_history"]),
            config.HISTORY_REVIEW_MAX_ITEMS,
        )
        self.assertIn("Possible matches against configured criteria:", report)
        self.assertIn("No moderation action was taken", report)
        self.assertNotIn("ban", report.casefold())
        self.assertNotIn("0.930", report)
        self.assertNotIn("I didn't find clear evidence", report)
        self.assertNotIn("80 matching comments", report)
        self.assertNotIn("20 matching posts", report)
        merge.assert_not_called()
        self.assertEqual(model.calls_used, 0)  # the patched evaluator owns no real provider call

    def test_manual_review_still_runs_jev_when_summary_call_limit_is_exhausted(self):
        history = _history(2)
        jev_calls = []

        def decide_many(*, state, questions):
            jev_calls.append((state, questions))
            return scores_for_questions(questions)

        jev_client = SimpleNamespace(
            ready=True,
            calls_used=0,
            max_calls=1,
            last_usage={"input_tokens": 1, "cost": 0.0},
            served_model="typesafe/jev-test",
            decide_many=decide_many,
        )
        intent = {"needs_history": True, "question": None}
        comment = SimpleNamespace(
            author=SimpleNamespace(name="mod"),
            fullname="t4_request",
            permalink="https://reddit.test/request",
        )
        with patch("moderator_history_commands._remaining_llm_calls", return_value=0), \
             patch("moderator_history_commands.load_history_subreddits", return_value=frozenset({"example_community"})), \
             patch("moderator_history_commands.collect_user_history", return_value=history), \
             patch("moderator_history_commands.evaluate_history") as evaluate:
            report, _tokens, _cost, _state = handle_moderator_history_request(
                comment, object(), object(), object(), {}, intent,
                requester_is_moderator=True,
                parent_author="example_user",
                jev_client=jev_client,
                diagnostic=True,
            )

        self.assertEqual(len(jev_calls), 1)
        self.assertEqual(len(jev_calls[0][1]), 3)
        evaluate.assert_not_called()
        self.assertIn("Jev scored all 3 operator-configured criteria against the same history", report)

    def test_debug_report_shows_each_configured_criterion_score(self):
        rules = synthetic_history_criteria_rules()

        def decide_many(*, state, questions):
            return scores_for_questions(
                questions, positives=("sample_repeated_link_posting",), positive_score=0.78
            )

        jev_client = SimpleNamespace(
            ready=True,
            calls_used=0,
            max_calls=1,
            last_usage={"input_tokens": 1, "cost": 0.0},
            served_model="typesafe/jev-test",
            decide_many=decide_many,
        )
        history = _history(2)
        assessment = assess_history_nouls_with_jev(
            history, rules, jev_client
        )

        finding = HistoryFinding(
            evidence_index=0,
            item_id=history.items[0].item_id,
            kind="comment",
            subreddit="example_community",
            permalink=history.items[0].permalink,
            timestamp="2026-10-16T00:00:00+00:00",
            score=1,
            excerpt="A synthetic comment repeatedly targets one fictional user with insults.",
            rule_name="sample_direct_personal_attack",
            confidence=0.91,
            explanation="The synthetic author repeatedly directs insults at a fictional participant.",
        )
        report = _format_history_report(
            "example_user",
            history,
            {"summary": "A concise summary."},
            [finding],
            gate_assessment=assessment,
            diagnostic=True,
        )

        self.assertIn("Jev scored all 3 operator-configured criteria against the same history", report)
        self.assertIn("| Sample Repeated Link Posting | 0.780 · flagged | — |", report)
        self.assertIn("Jev flagged 1 configured criterion/criteria", report)
        self.assertIn("uncalibrated outputs", report)
        self.assertIn("Sample Direct Personal Attack", report)
        self.assertIn("[View comment](<https://www.reddit.com/r/example_community/comments/c0/>)", report)
        self.assertNotIn("bundles of up to", report)

    def test_diagnostic_rechecks_history_without_reading_or_creating_the_review_ledger(self):
        history = _history(2)
        jev_client = SimpleNamespace(
            ready=True,
            calls_used=0,
            max_calls=1,
            last_usage={"input_tokens": 1, "cost": 0.0},
            served_model="typesafe/jev-test",
            decide_many=lambda *, state, questions: scores_for_questions(questions),
        )
        comment = SimpleNamespace(
            author=SimpleNamespace(name="mod"),
            fullname="t4_request",
            permalink="https://reddit.test/request",
        )
        with patch("moderator_history_commands.load_history_subreddits", return_value=frozenset({"example_community"})), \
             patch("moderator_history_commands.collect_user_history", return_value=history) as collect, \
             patch("moderator_history_commands.collect_unseen_user_history", side_effect=AssertionError("ledger lookup")), \
             patch("moderator_history_commands.HistoryReviewLedger", side_effect=AssertionError("ledger creation")):
            report, *_ = handle_moderator_history_request(
                comment,
                object(),
                object(),
                SimpleNamespace(max_calls=0, calls_used=0),
                {},
                {"needs_history": True, "target_username": "example_user", "wants_flair": False},
                requester_is_moderator=True,
                parent_author=None,
                jev_client=jev_client,
                diagnostic=True,
            )

        self.assertEqual(collect.call_count, 1)
        self.assertEqual(collect.call_args.kwargs["max_items"], config.HISTORY_REVIEW_MAX_ITEMS)
        self.assertIn("Sample: 2 items (2 comments, 0 posts).", report)
        self.assertIn("Jev scored all 3 operator-configured criteria", report)

    def test_exhausted_text_model_budget_does_not_block_structured_history_review(self):
        model = wrap_with_rate_limit(
            SimpleNamespace(generate_content=lambda *_args, **_kwargs: SimpleNamespace(text="ok")), 1
        )
        model.generate_content("consume the only call")
        intent = parse_moderator_history_request("Mod bot, scan u/example_user's history")
        history = _history()
        jev = SimpleNamespace(
            ready=True,
            calls_used=0,
            max_calls=1,
            last_usage={"input_tokens": 1, "cost": 0.0},
            served_model="typesafe/jev-test",
            decide_many=lambda *, state, questions: scores_for_questions(questions),
        )
        with patch("moderator_history_commands.load_history_subreddits", return_value=frozenset({"example_community"})), \
             patch("moderator_history_commands.collect_user_history", return_value=history) as collect:
            report, *_ = handle_moderator_history_request(
                SimpleNamespace(author=SimpleNamespace(name="mod")),
                object(), object(), model, {}, intent,
                requester_is_moderator=True,
                parent_author=None,
                jev_client=jev,
                diagnostic=True,
            )
        collect.assert_called_once()
        self.assertIn("Jev scored all 3 operator-configured criteria against the same history", report)

    def test_report_shows_evaluated_counts_and_omits_debugging_details(self):
        history = replace(
            _history(1, 1),
            comments_cap_reached=True,
            posts_cap_reached=True,
            comments_scan_limited=True,
            posts_scan_limited=True,
        )
        finding = HistoryFinding(
            evidence_index=0,
            item_id=history.items[0].item_id,
            kind="comment",
            subreddit="example_community",
            permalink=history.items[0].permalink,
            timestamp=datetime.fromtimestamp(history.items[0].created_utc, timezone.utc).isoformat(),
            score=1,
            excerpt="*bold* [fake link](https://bad.example) `code`",
            rule_name="sample_reporting_criterion",
            confidence=0.9,
            explanation="Evidence.",
        )

        report = _format_history_report(
            "example_user",
            history,
            {
                "summary": "This person's posts mainly discuss gardening projects and local events.",
                "themes": ["robotics", "open-source models"],
                "answer": "Supported.",
            },
            [finding],
        )

        self.assertIn(
            "Reviewed 1 comment(s) and 1 post(s) from u/example_user.",
            report,
        )
        self.assertIn("The criteria review did not finish", report)
        self.assertNotIn("90%", report)
        self.assertNotIn("Recurring interests", report)
        self.assertNotIn("matching comments", report)
        self.assertNotIn("matching posts", report)
        self.assertNotIn("2026-09-", report)
        self.assertNotIn("sample limit", report)
        self.assertNotIn("partial", report)
        self.assertNotIn("Evidence", report)
        self.assertIn("u/example_user", report)
        self.assertNotIn("bold", report)

    def test_broad_report_uses_plain_language_when_history_review_is_incomplete(self):
        history = _history(4, 2)
        report = _format_history_report(
            "example_user",
            history,
            {
                "summary": "This person often discusses gardening and community events.",
                "themes": ["robotics"],
                "answer": "",
            },
            [],
        )

        self.assertIn(
            "The criteria review did not finish, so there is no result.",
            report,
        )
        self.assertNotIn("gardening", report)
        self.assertNotIn("Recurring interests", report)
        self.assertIn("u/example_user", report)

    def test_partial_reddit_history_does_not_complete_the_moderator_ledger_attempt(self):
        criteria = synthetic_history_criteria()
        history = replace(
            _history(),
            errors=("Reddit did not return the complete comment listing.",),
        )
        assessment = HistoryGateAssessment(
            rule_names=tuple(criteria),
            bundle_size=0,
            bundles=(),
            matched_rules=(),
            calls=1,
            input_tokens=12,
            cost=0.0002,
            served_model="test",
            classifier="jev_full_history_noul",
            item_attribution_available=False,
            criterion_probabilities={name: 0.03 for name in criteria},
            scores_complete=True,
        )
        intent = parse_moderator_history_request("Mod bot, scan u/example_user's history.")
        jev_client = SimpleNamespace(ready=True, calls_used=0, max_calls=1)

        with patch("moderator_history_commands.collect_unseen_user_history", return_value=SimpleNamespace(
            history=history,
            duplicate_items_skipped=0,
        )), patch("moderator_history_commands.load_history_subreddits", return_value=frozenset({"example_community"})), \
             patch("moderator_history_commands.assess_history_nouls_with_jev", return_value=assessment), \
             patch("moderator_history_commands._remaining_llm_calls", return_value=0):
            report, *_ = handle_moderator_history_request(
                SimpleNamespace(
                    author=SimpleNamespace(name="example_mod"),
                    fullname="t1-review-request",
                    permalink="https://reddit.test/request",
                ),
                SimpleNamespace(),
                object(),
                object(),
                {},
                intent,
                requester_is_moderator=True,
                parent_author=None,
                jev_client=jev_client,
            )

        self.assertIn("History collection had gaps", report)
        self.assertEqual(self.review_ledger.metrics()["incomplete_jev_attempts"], 1)
        self.assertEqual(self.review_ledger.metrics()["reviews_completed"], 0)

    def test_plain_report_omits_unrequested_summary(self):
        history = _history(1, 1)
        report = _format_history_report(
            "example_user",
            history,
            {"summary": "", "answer": "", "themes": []},
            [],
        )

        self.assertNotIn("First sentence", report)
        self.assertNotIn("The rest of this long explanation", report)


if __name__ == "__main__":
    unittest.main()
