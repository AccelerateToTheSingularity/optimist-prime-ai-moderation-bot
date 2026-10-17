from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config
from history_gate_jev import (
    JevHistoryReviewError,
    assess_history_nouls_with_jev,
    build_history_criteria_questions,
    load_history_criteria,
    required_history_noul_calls,
)
from history_criteria_test_support import (
    ConfiguredHistoryCriteriaTestCase,
    SYNTHETIC_HISTORY_CRITERIA_IDS,
    synthetic_history_criteria_rules,
    scores_for_questions,
)
from user_history import HistoryItem, UserHistory


def _history(count: int, *, text_length: int = 20) -> UserHistory:
    items = tuple(
        HistoryItem(
            item_id=f"c{index}",
            kind="comment",
            subreddit="example_community",
            created_utc=2_000_000_000 - index,
            score=1,
            permalink=f"https://reddit.test/comments/{index}",
            text=(f"Synthetic comment {index}. " + "ordinary words " * text_length),
        )
        for index in range(count)
    )
    return UserHistory("synthetic_user", items, count, 0)


class _FakeJev:
    ready = True
    max_calls = 3

    def __init__(self, *, overrides=None, omit=None):
        self.calls_used = 0
        self.requests = []
        self.last_usage = {"input_tokens": 1200, "cost": 0.0002}
        self.served_model = "typesafe/jev-test"
        self.overrides = dict(overrides or {})
        self.omit = omit

    def decide_many(self, *, state, questions):
        self.calls_used += 1
        self.requests.append((state, questions))
        scores = scores_for_questions(questions)
        scores.update(self.overrides)
        if self.omit:
            scores.pop(self.omit, None)
        return scores


class HistoryGateJevTests(ConfiguredHistoryCriteriaTestCase):
    def setUp(self):
        super().setUp()
        self.threshold_patch = patch.object(config, "HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT", 67)
        self.threshold_patch.start()
        self.addCleanup(self.threshold_patch.stop)
        self.rules = synthetic_history_criteria_rules()

    def test_isolated_fixture_has_three_generic_example_criteria_with_complete_fields(self):
        guides = load_history_criteria()
        self.assertEqual(len(guides), 3)
        self.assertEqual(set(guides), SYNTHETIC_HISTORY_CRITERIA_IDS)
        self.assertIn("sample_repeated_link_posting", guides)
        attacks = guides["sample_direct_personal_attack"]
        self.assertEqual(attacks["title"], "Repeated direct personal attacks")
        self.assertIn("another participant", attacks["description"])
        self.assertIn("quotation", attacks["not_sufficient_by_itself"].casefold())
        self.assertIn("not enough", attacks["not_sufficient_by_itself"].casefold())
        for guide in guides.values():
            self.assertTrue(all(guide[field].strip() for field in (
                "title", "description", "high_confidence_yes",
                "not_sufficient_by_itself", "closest_boundaries",
            )))
        spam = guides["sample_repeated_link_posting"]
        self.assertIn("same link", spam["description"])
        self.assertIn("disclosed announcement", spam["not_sufficient_by_itself"])

    def test_live_question_includes_each_substantive_guide_field(self):
        questions = build_history_criteria_questions(self.rules)
        self.assertEqual(len(questions), len(SYNTHETIC_HISTORY_CRITERIA_IDS))
        prompt_text = " ".join(json.dumps(value).casefold() for value in questions.values())
        self.assertIn("repeated direct personal attacks", prompt_text)
        self.assertIn("the author's own statements", prompt_text)
        for name, guide in load_history_criteria().items():
            question = questions[name]
            self.assertEqual(question["type"], "noul")
            combined = " ".join((question["instructions"], *question["criteria"].values()))
            for field in ("description", "high_confidence_yes", "not_sufficient_by_itself", "closest_boundaries"):
                self.assertIn(guide[field], combined)
        spam = questions["sample_repeated_link_posting"]
        prompt = " ".join((spam["instructions"], *spam["criteria"].values())).casefold()
        self.assertIn("same link", prompt)
        self.assertIn("disclosed announcement", prompt)

    def test_missing_match_threshold_stops_before_provider_request(self):
        with patch.object(config, "HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT", None):
            client = _FakeJev()
            with self.assertRaisesRegex(JevHistoryReviewError, "Set BOT_HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT"):
                assess_history_nouls_with_jev(_history(1), self.rules, client)
            self.assertEqual(client.requests, [])

    def test_missing_lookback_stops_before_provider_request(self):
        with patch.object(config, "HISTORY_REVIEW_LOOKBACK_DAYS", None):
            client = _FakeJev()
            with self.assertRaisesRegex(JevHistoryReviewError, "Set BOT_HISTORY_REVIEW_LOOKBACK_DAYS"):
                assess_history_nouls_with_jev(_history(1), self.rules, client)
            self.assertEqual(client.requests, [])

    def test_missing_sample_limit_stops_before_provider_request(self):
        with patch.object(config, "HISTORY_REVIEW_MAX_ITEMS", None):
            client = _FakeJev()
            with self.assertRaisesRegex(JevHistoryReviewError, "Set BOT_HISTORY_REVIEW_MAX_ITEMS"):
                assess_history_nouls_with_jev(_history(1), self.rules, client)
            self.assertEqual(client.requests, [])

    def test_150_item_history_uses_one_request_and_returns_all_independent_scores(self):
        target = "sample_repeated_link_posting"
        other = "sample_direct_personal_attack"
        client = _FakeJev(overrides={target: 0.93, other: 0.88})
        result = assess_history_nouls_with_jev(_history(150), self.rules, client)

        self.assertEqual(len(client.requests), 1)
        state, questions = client.requests[0]
        self.assertEqual(len(state["reddit_history"]), 150)
        self.assertEqual(len(questions), 3)
        self.assertTrue(all(item["kind"] == "comment" for item in state["reddit_history"]))
        self.assertEqual(set(result.criterion_probabilities), SYNTHETIC_HISTORY_CRITERIA_IDS)
        self.assertEqual(result.matched_rules, (target, other))
        self.assertEqual(result.positive_criterion_count, 2)
        self.assertEqual(result.match_threshold, 0.67)
        self.assertEqual(result.classifier, "jev_full_history_noul")
        self.assertEqual(result.calls, 1)
        self.assertFalse(result.item_attribution_available)
        self.assertEqual(result.input_tokens, 1200)
        self.assertAlmostEqual(result.cost, 0.0002)

    def test_item_counts_one_21_84_150_and_250_each_use_one_provider_request(self):
        for count in (1, 21, 84, 150, 250):
            with self.subTest(count=count):
                client = _FakeJev()
                assessment = assess_history_nouls_with_jev(_history(count), self.rules, client)
                self.assertEqual(assessment.calls, 1)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(len(client.requests[0][0]["reddit_history"]), count)

    def test_scores_are_raw_sorted_and_multiple_matches_are_allowed(self):
        first = "sample_repeated_link_posting"
        second = "sample_direct_personal_attack"
        third = "sample_contact_information"
        client = _FakeJev(overrides={first: 0.80, second: 0.67, third: 0.669})
        assessment = assess_history_nouls_with_jev(_history(1), self.rules, client)
        self.assertEqual(assessment.matched_rules, (first, second))
        self.assertEqual(assessment.positive_criterion_count, 2)
        self.assertEqual(list(assessment.criterion_probabilities.values()), sorted(
            assessment.criterion_probabilities.values(), reverse=True
        ))
        self.assertAlmostEqual(assessment.criterion_probabilities[third], 0.669)

    def test_missing_or_invalid_score_is_preserved_as_an_incomplete_partial_result(self):
        for client in (
            _FakeJev(omit="sample_repeated_link_posting"),
            _FakeJev(overrides={"sample_repeated_link_posting": float("nan")}),
            _FakeJev(overrides={"sample_repeated_link_posting": True}),
        ):
            with self.subTest(client=client):
                result = assess_history_nouls_with_jev(_history(1), self.rules, client)
                self.assertFalse(result.scores_complete)
                self.assertIn("sample_repeated_link_posting", result.missing_criterion_ids)
                self.assertNotIn("sample_repeated_link_posting", result.criterion_probabilities)
                self.assertEqual(result.matched_rules, ())

    def test_no_valid_criterion_scores_still_fails_instead_of_looking_like_no_match(self):
        client = _FakeJev()
        client.decide_many = lambda **_kwargs: {}
        with self.assertRaisesRegex(JevHistoryReviewError, "review incomplete"):
            assess_history_nouls_with_jev(_history(1), self.rules, client)

    def test_over_cap_and_exhausted_provider_allowance_stop_before_a_request(self):
        client = _FakeJev()
        with self.assertRaisesRegex(JevHistoryReviewError, "at most 250"):
            assess_history_nouls_with_jev(_history(251), self.rules, client)
        self.assertEqual(client.requests, [])

        client.max_calls = 0
        with self.assertRaisesRegex(JevHistoryReviewError, "no remaining request allowance"):
            assess_history_nouls_with_jev(_history(1), self.rules, client)
        self.assertEqual(client.requests, [])

    def test_request_estimate_is_one_and_requires_the_complete_criterion_set(self):
        self.assertEqual(required_history_noul_calls(250, 3), 1)
        with self.assertRaisesRegex(JevHistoryReviewError, "all 3"):
            required_history_noul_calls(150, 2)
        with self.assertRaisesRegex(JevHistoryReviewError, "at most 250"):
            required_history_noul_calls(251, 3)

    def test_unconfigured_criteria_are_rejected_before_a_provider_request(self):
        extra = SimpleNamespace(
            name="unconfigured_example_criterion",
            description="An isolated test-only criterion.",
            active=True,
            order=190,
        )
        client = _FakeJev()
        with self.assertRaisesRegex(JevHistoryReviewError, "operator catalog"):
            assess_history_nouls_with_jev(_history(1), [*self.rules, extra], client)
        self.assertEqual(client.requests, [])

    def test_updated_example_guide_scores_carry_the_new_definition_revision(self):
        name = "sample_repeated_link_posting"
        client = _FakeJev(overrides={name: 0.96})
        result = assess_history_nouls_with_jev(_history(1), self.rules, client)
        self.assertEqual(result.criterion_score_revisions[name], 2)
        self.assertIn(name, result.matched_rules)
        self.assertEqual(len(client.requests), 1)

    def test_long_item_uses_three_window_excerpt_and_reports_truncation(self):
        text = "START_MARKER " + "x" * 2500 + "MIDDLE_MARKER " + "y" * 2500 + "END_MARKER"
        client = _FakeJev()
        assessment = assess_history_nouls_with_jev(
            UserHistory("synthetic_user", (
                HistoryItem("long", "comment", "example_community", 1, 1, "", text),
            ), 1, 0),
            self.rules,
            client,
        )
        excerpt = client.requests[0][0]["reddit_history"][0]["text"]
        self.assertEqual(assessment.evidence_truncated_items, 1)
        self.assertIn("START_MARKER", excerpt)
        self.assertIn("MIDDLE_MARKER", excerpt)
        self.assertIn("END_MARKER", excerpt)
        self.assertIn("text omitted", excerpt)

if __name__ == "__main__":
    unittest.main()
