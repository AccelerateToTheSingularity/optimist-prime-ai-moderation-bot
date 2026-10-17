import unittest
from types import SimpleNamespace

from history_criteria_test_support import (
    ConfiguredHistoryCriteriaTestCase,
    current_history_criteria_revisions,
)
from review_result import format_no_new_history_result, format_review_result
from user_history import HistoryItem, UserHistory


def history(comments=2, posts=0, errors=()):
    items = tuple(
        HistoryItem(str(i), kind, "example_community", 1.0, 1, "", "evidence")
        for i, kind in enumerate(["comment"] * comments + ["post"] * posts)
    )
    return UserHistory("example_user", items, 900, 100, errors=errors)


def assessment(scores, matched=(), *, truncated=0, rules=None):
    names = tuple(rules or scores)
    return SimpleNamespace(
        matched_rules=tuple(matched),
        rule_names=names,
        criterion_probabilities=dict(scores),
        criterion_score_revisions={
            name: current_history_criteria_revisions().get(name, 1)
            for name in scores
        },
        match_threshold=0.67,
        evidence_truncated_items=truncated,
    )


class ReviewResultTests(ConfiguredHistoryCriteriaTestCase):
    def test_pass_is_a_concise_plain_language_recommendation(self):
        result = format_review_result(
            history(150),
            assessment({
                "sample_repeated_link_posting": 0.24,
                "sample_contact_information": 0.49,
            }),
        )
        self.assertTrue(result.completed)
        self.assertEqual(
            result.text,
            "150 comments scanned for [u/example_user](https://www.reddit.com/user/example_user/). No configured criterion matched.",
        )

    def test_single_match_names_the_operator_configured_criterion(self):
        result = format_review_result(
            history(24),
            assessment({"sample_repeated_link_posting": 0.59}, ("sample_repeated_link_posting",)),
            display_name=lambda _name: "Example Repeated Link Posting",
        )
        self.assertEqual(
            result.text,
            "24 comments scanned for [u/example_user](https://www.reddit.com/user/example_user/). Matched configured criteria: Example Repeated Link Posting.",
        )

    def test_posts_are_counted_and_multiple_positive_matches_are_listed_compactly(self):
        result = format_review_result(
            history(120, 30),
            assessment({
                "sample_repeated_link_posting": 0.91,
                "sample_direct_personal_attack": 0.62,
                "sample_contact_information": 0.2,
            }, ("sample_repeated_link_posting", "sample_direct_personal_attack")),
            display_name=lambda name: {
                "sample_repeated_link_posting": "Repeated Link Posting",
                "sample_direct_personal_attack": "Direct Personal Attacks",
            }[name],
        )
        self.assertEqual(
            result.text,
            "120 comments and 30 posts scanned for [u/example_user](https://www.reddit.com/user/example_user/). Matched configured criteria: Repeated Link Posting and Direct Personal Attacks.",
        )

    def test_many_matches_are_all_listed_in_the_review_result(self):
        labels = {
            "sample_repeated_link_posting": "Repeated Link Posting",
            "sample_direct_personal_attack": "Direct Personal Attacks",
            "sample_contact_information": "Contact Information Disclosure",
        }
        scores = {name: 0.8 for name in labels}
        result = format_review_result(
            history(100), assessment(scores, tuple(labels), rules=tuple(labels)),
            display_name=labels.__getitem__,
        )
        for label in labels.values():
            with self.subTest(criterion=label):
                self.assertIn(label, result.text)
        self.assertNotIn("additional criteria matches", result.text)
        self.assertNotIn("| Criterion | Score |", result.text)

    def test_long_item_truncation_is_disclosed(self):
        result = format_review_result(
            history(1), assessment({"sample_repeated_link_posting": 0.10}, truncated=1)
        )
        self.assertIn("Some text in 1 long item(s) was shortened", result.text)

    def test_missing_data_or_analysis_never_passes(self):
        for sample, current in (
            (history(0), assessment({})),
            (history(errors=("HTTP error",)), assessment({})),
            (history(), None),
        ):
            with self.subTest(sample=sample):
                result = format_review_result(sample, current)
                self.assertFalse(result.completed)
                self.assertIn("Review incomplete", result.text)
                self.assertNotIn("Passed", result.text)

    def test_partial_match_on_failed_history_scan_is_not_called_a_pass_or_ban_recommendation(self):
        result = format_review_result(
            history(errors=("listing failed",)),
            assessment(
                {"sample_repeated_link_posting": 0.7},
                ("sample_repeated_link_posting",),
            ),
        )
        self.assertFalse(result.completed)
        self.assertIn(
            "Review incomplete; possible configured match: sample_repeated_link_posting.",
            result.text,
        )
        self.assertNotIn("Recommend banning user", result.text)
        self.assertNotIn("0.700", result.text)

    def test_no_new_history_does_not_turn_an_incomplete_earlier_review_into_a_pass(self):
        previous = (
            SimpleNamespace(matched_rules=(), scores_complete=False),
            12,
            True,
            "2026-10-01T00:00:00+00:00",
        )
        result = format_no_new_history_result("example_user", previous)
        self.assertIn("No new posts or comments were available for [u/example_user]", result)
        self.assertIn("earlier review from 01 Oct 2026 did not finish", result)
        self.assertIn("neither a match nor a no-match result", result)
        self.assertNotIn("found no configured criterion", result)

    def test_no_new_history_does_not_repeat_an_old_definition_match(self):
        previous = (
            SimpleNamespace(
                matched_rules=("sample_repeated_link_posting",),
                scores_complete=True,
            ),
            12,
            False,
            "2026-10-01T00:00:00+00:00",
        )
        result = format_no_new_history_result(
            "example_user",
            previous,
            display_name=lambda _name: "Example Spam Pattern",
        )
        self.assertIn("The earlier review used an older criterion definition", result)
        self.assertIn("isn't complete under the current settings", result)
        self.assertNotIn("Example Spam Pattern", result)


if __name__ == "__main__":
    unittest.main()
