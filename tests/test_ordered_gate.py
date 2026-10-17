import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from content_moderation import evaluate_rules
from llm_client import LLMQuotaExhausted, extract_token_info
from moderation_rules import ModerationRule
from ordered_gate import (
    MATCH_LABEL,
    NONE_LABEL,
    NO_MATCH_LABEL,
    NOT_REACHED_LABEL,
    QUESTION_NAME,
    ModerationEvidencePending,
    parse_evidence_response,
    parse_evidence_responses,
)


class FakeJev:
    ready = True
    max_calls = 10
    last_usage = {"input_tokens": 17, "cost": 0.0001}
    served_model = "typesafe/jev-test"

    def __init__(self, choice):
        self.choice = choice
        self.calls_used = 0
        self.states = []
        self.questions = []

    def choose_many(self, *, state, questions):
        self.calls_used += 1
        self.states.append(state)
        self.questions.append(questions)
        if isinstance(self.choice, Exception):
            raise self.choice
        return self.choice


class OrderedGateTests(unittest.TestCase):
    def rules(self):
        return [
            ModerationRule(name=name, description=f"Does this match {name}?", order=index,
                           conditions={"stop_on_match": True})
            for index, name in enumerate(("first", "second", "third"))
        ]

    def model(self, raw):
        response = SimpleNamespace(
            text=raw,
            usage_metadata=SimpleNamespace(
                prompt_token_count=23,
                candidates_token_count=7,
            ),
        )
        return Mock(generate_content=Mock(return_value=response))

    def mixed_rules(self):
        return [
            ModerationRule(
                name=name,
                description=f"Does this match {name}?",
                order=index,
                conditions={"stop_on_match": stop_on_match},
            )
            for index, (name, stop_on_match) in enumerate((
                ("first_nonstop", False),
                ("first_stop", True),
                ("later_nonstop", False),
                ("later_stop", True),
                ("unreached", False),
            ))
        ]

    def multi_evidence(self, *names):
        return json.dumps({
            "items": [
                {"rule": name, "reason": f"Reason for {name}", "evidence": "text"}
                for name in names
            ]
        })

    def test_jev_selects_once_in_order_and_llm_only_supplies_evidence(self):
        jev = FakeJev({QUESTION_NAME: "ordered_rule_2"})
        model = self.model('{"reason":"Matches the second condition","evidence":"text"}')
        rules = list(reversed(self.rules()))

        result = evaluate_rules(
            "text", rules, model, context={"post_title": "A title"}, jev_client=jev
        )

        self.assertEqual([match["rule"].name for match in result["matches"]], ["second"])
        self.assertFalse(result["decisions"]["first"]["matched"])
        self.assertTrue(result["decisions"]["second"]["matched"])
        self.assertIsNone(result["decisions"]["third"]["matched"])
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(len(jev.questions), 1)
        question = jev.questions[0][QUESTION_NAME]
        self.assertEqual(list(question["criteria"]), [
            "ordered_rule_1", "ordered_rule_2", "ordered_rule_3", NONE_LABEL
        ])
        self.assertIn("Does this match first?", question["criteria"]["ordered_rule_1"])
        self.assertEqual(jev.states[0], {
            "content": "text",
            "background": {"post_title": "A title"},
        })
        model.generate_content.assert_called_once()
        evidence_prompt = model.generate_content.call_args.args[0][0]["parts"][0]
        self.assertIn('"name": "second"', evidence_prompt)
        self.assertIn("Do not reconsider the classification", evidence_prompt)
        self.assertEqual(result["token_info"]["input_tokens"], 40)
        self.assertEqual(result["token_info"]["output_tokens"], 7)
        self.assertEqual(result["token_info"]["total_tokens"], 47)
        self.assertAlmostEqual(
            result["token_info"]["cost"],
            0.0001 + extract_token_info(model.generate_content.return_value)["cost"],
        )

    def test_jev_no_match_marks_all_rules_false_without_an_llm_call(self):
        jev = FakeJev({QUESTION_NAME: NONE_LABEL})
        model = Mock()

        result = evaluate_rules("text", self.rules(), model, jev_client=jev)

        self.assertFalse(result["matches"])
        self.assertTrue(all(not item["matched"] for item in result["decisions"].values()))
        self.assertEqual(result["token_info"]["input_tokens"], 17)
        model.generate_content.assert_not_called()
        self.assertEqual(jev.calls_used, 1)

    def test_missing_jev_does_not_fall_back_to_llm_classification(self):
        model = Mock()

        result = evaluate_rules("text", self.rules(), model)

        self.assertIsNone(result)
        model.generate_content.assert_not_called()

    def test_unavailable_or_exhausted_jev_is_retryable_without_llm_call(self):
        for ready, used, maximum in ((False, 0, 10), (True, 10, 10)):
            with self.subTest(ready=ready, used=used):
                jev = FakeJev({QUESTION_NAME: "ordered_rule_1"})
                jev.ready = ready
                jev.calls_used = used
                jev.max_calls = maximum
                model = Mock()

                self.assertIsNone(
                    evaluate_rules("text", self.rules(), model, jev_client=jev)
                )
                model.generate_content.assert_not_called()
                self.assertEqual(jev.calls_used, used)

    def test_missing_or_invalid_jev_choice_is_retryable(self):
        for answer in ({}, {QUESTION_NAME: "unknown_rule"}, {QUESTION_NAME: None}):
            with self.subTest(answer=answer):
                jev = FakeJev(answer)
                model = Mock()

                self.assertIsNone(
                    evaluate_rules("text", self.rules(), model, jev_client=jev)
                )
                model.generate_content.assert_not_called()

    def test_jev_failure_is_retryable_and_not_retried_in_the_same_evaluation(self):
        jev = FakeJev(TimeoutError("provider unavailable"))
        model = Mock()

        self.assertIsNone(evaluate_rules("text", self.rules(), model, jev_client=jev))

        self.assertEqual(jev.calls_used, 1)
        model.generate_content.assert_not_called()

    def test_llm_cannot_change_the_selected_rule_or_authorize_bad_evidence(self):
        invalid_responses = (
            '{"reason":"r","evidence":"invented"}',
            '{"reason":"r","evidence":"text","rule":"first"}',
            '{"reason":"","evidence":"text"}',
            '{"reason":"r","evidence":""}',
            '{"reason":"r","evidence":"text","evidence":"text"}',
        )
        for raw in invalid_responses:
            with self.subTest(raw=raw):
                jev = FakeJev({QUESTION_NAME: "ordered_rule_2"})
                model = self.model(raw)

                with self.assertRaises(ModerationEvidencePending):
                    evaluate_rules("text", self.rules(), model, jev_client=jev)
                model.generate_content.assert_called_once()
                self.assertEqual(jev.calls_used, 1)

    def test_evidence_parser_requires_an_exact_content_substring(self):
        self.assertEqual(
            parse_evidence_response('{"reason":"r","evidence":"exact words"}', "exact words here"),
            {"reason": "r", "evidence": "exact words"},
        )
        with self.assertRaises(ValueError):
            parse_evidence_response('{"reason":"r","evidence":"not present"}', "exact words")

    def test_evidence_quota_saves_the_jev_selection_for_an_explanation_only_retry(self):
        jev = FakeJev({QUESTION_NAME: "ordered_rule_1"})
        model = Mock(generate_content=Mock(side_effect=LLMQuotaExhausted("full")))

        with self.assertRaises(ModerationEvidencePending) as raised:
            evaluate_rules("text", self.rules(), model, jev_client=jev)

        self.assertEqual(jev.calls_used, 1)
        model.generate_content.assert_called_once()
        pending = raised.exception.pending_selection
        self.assertEqual(pending["selected_rule_names"], ["first"])

        retry_model = self.model('{"reason":"Matches the first condition","evidence":"text"}')
        result = evaluate_rules(
            "text", self.rules(), retry_model, jev_client=None,
            pending_selection=pending,
        )

        self.assertEqual([match["rule"].name for match in result["matches"]], ["first"])
        retry_model.generate_content.assert_called_once()

    def test_mixed_evidence_retry_keeps_overlapping_jev_selections_without_reclassifying(self):
        jev = FakeJev({
            "moderation_rule_1": MATCH_LABEL,
            "moderation_rule_2": MATCH_LABEL,
            "moderation_rule_3": NO_MATCH_LABEL,
            "moderation_rule_4": NO_MATCH_LABEL,
            "moderation_rule_5": NO_MATCH_LABEL,
        })
        quota_model = Mock(generate_content=Mock(side_effect=LLMQuotaExhausted("full")))
        with self.assertRaises(ModerationEvidencePending) as raised:
            evaluate_rules("text", self.mixed_rules(), quota_model, jev_client=jev)
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(
            raised.exception.pending_selection["selected_rule_names"],
            ["first_nonstop", "first_stop"],
        )

        retry_model = self.model(self.multi_evidence("first_nonstop", "first_stop"))
        result = evaluate_rules(
            "text", self.mixed_rules(), retry_model, jev_client=None,
            pending_selection=raised.exception.pending_selection,
        )
        self.assertEqual(
            [match["rule"].name for match in result["matches"]],
            ["first_nonstop", "first_stop"],
        )
        retry_model.generate_content.assert_called_once()

    def test_empty_or_oversize_content_does_not_spend_either_provider(self):
        jev = FakeJev({QUESTION_NAME: NONE_LABEL})
        model = Mock()

        self.assertEqual(evaluate_rules("", self.rules(), model, jev_client=jev)["matches"], [])
        self.assertIsNone(evaluate_rules("x" * 20001, self.rules(), model, jev_client=jev))
        self.assertEqual(jev.calls_used, 0)
        model.generate_content.assert_not_called()

    def test_mixed_rules_use_one_jev_request_and_keep_all_matches_before_first_stop(self):
        answers = {
            "moderation_rule_1": MATCH_LABEL,
            "moderation_rule_2": NO_MATCH_LABEL,
            "moderation_rule_3": MATCH_LABEL,
            "moderation_rule_4": MATCH_LABEL,
            # This answer is deliberately contradictory. The detector must not
            # consume or report a result after the first matching stop rule.
            "moderation_rule_5": MATCH_LABEL,
        }
        jev = FakeJev(answers)
        model = self.model(self.multi_evidence("first_nonstop", "later_nonstop", "later_stop"))

        result = evaluate_rules("text", self.mixed_rules(), model, jev_client=jev)

        self.assertEqual(
            [match["rule"].name for match in result["matches"]],
            ["first_nonstop", "later_nonstop", "later_stop"],
        )
        self.assertTrue(result["decisions"]["first_nonstop"]["matched"])
        self.assertFalse(result["decisions"]["first_stop"]["matched"])
        self.assertTrue(result["decisions"]["later_nonstop"]["matched"])
        self.assertTrue(result["decisions"]["later_stop"]["matched"])
        self.assertIsNone(result["decisions"]["unreached"]["matched"])
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(len(jev.questions), 1)
        self.assertEqual(list(jev.questions[0]), [
            "moderation_rule_1", "moderation_rule_2", "moderation_rule_3",
            "moderation_rule_4", "moderation_rule_5",
        ])
        self.assertIn("first_stop", jev.questions[0]["moderation_rule_3"]["instructions"])
        self.assertIn("not_reached_after_earlier_match", jev.questions[0]["moderation_rule_3"]["criteria"])
        evidence_prompt = model.generate_content.call_args.args[0][0]["parts"][0]
        self.assertIn('"name": "later_stop"', evidence_prompt)
        self.assertNotIn('"name": "unreached"', evidence_prompt)

    def test_mixed_rules_ignore_answers_after_first_stop_and_leave_tail_unknown(self):
        jev = FakeJev({
            "moderation_rule_1": NO_MATCH_LABEL,
            "moderation_rule_2": MATCH_LABEL,
        })
        model = self.model(self.multi_evidence("first_stop"))

        result = evaluate_rules("text", self.mixed_rules(), model, jev_client=jev)

        self.assertEqual([item["rule"].name for item in result["matches"]], ["first_stop"])
        self.assertFalse(result["decisions"]["first_nonstop"]["matched"])
        self.assertTrue(result["decisions"]["first_stop"]["matched"])
        self.assertTrue(all(
            result["decisions"][name]["matched"] is None
            for name in ("later_nonstop", "later_stop", "unreached")
        ))
        self.assertEqual(jev.calls_used, 1)
        model.generate_content.assert_called_once()

    def test_mixed_reachable_invalid_choice_is_retryable_without_evidence_call(self):
        for answers in (
            {"moderation_rule_1": MATCH_LABEL},
            {"moderation_rule_1": NOT_REACHED_LABEL},
            {"moderation_rule_1": "unknown"},
        ):
            with self.subTest(answers=answers):
                jev = FakeJev(answers)
                model = Mock()
                # These answers do not complete the ordered Jev result. A
                # positive answer for a non-stopping rule is not enough to
                # persist a final selection because later rules remain open.
                self.assertIsNone(
                    evaluate_rules("text", self.mixed_rules(), model, jev_client=jev)
                )
                self.assertEqual(jev.calls_used, 1)
                model.generate_content.assert_not_called()

    def test_mixed_multi_rule_evidence_must_cover_exact_selected_set(self):
        invalid = (
            '{"items":[{"rule":"first_nonstop","reason":"r","evidence":"text"},'
            '{"rule":"extra","reason":"r","evidence":"text"}]}',
            '{"items":[{"rule":"first_nonstop","reason":"r","evidence":"text"},'
            '{"rule":"first_nonstop","reason":"r","evidence":"text"}]}',
            '{"items":[{"rule":"first_nonstop","reason":"r","evidence":"invented"},'
            '{"rule":"first_stop","reason":"r","evidence":"text"}]}',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                jev = FakeJev({
                    "moderation_rule_1": MATCH_LABEL,
                    "moderation_rule_2": MATCH_LABEL,
                })
                model = self.model(raw)
                with self.assertRaises(ModerationEvidencePending):
                    evaluate_rules("text", self.mixed_rules(), model, jev_client=jev)
                model.generate_content.assert_called_once()
                self.assertEqual(jev.calls_used, 1)

    def test_mixed_all_no_match_returns_explicit_false_decisions(self):
        jev = FakeJev({
            f"moderation_rule_{index}": NO_MATCH_LABEL
            for index in range(1, 6)
        })
        model = Mock()

        result = evaluate_rules("text", self.mixed_rules(), model, jev_client=jev)

        self.assertFalse(result["matches"])
        self.assertTrue(all(not item["matched"] for item in result["decisions"].values()))
        model.generate_content.assert_not_called()
        self.assertEqual(jev.calls_used, 1)

    def test_multi_evidence_parser_rejects_added_or_missing_rules(self):
        selected = self.mixed_rules()[:2]
        self.assertEqual(
            set(parse_evidence_responses(
                self.multi_evidence("first_nonstop", "first_stop"), "text", selected
            )),
            {"first_nonstop", "first_stop"},
        )
        with self.assertRaises(ValueError):
            parse_evidence_responses(
                self.multi_evidence("first_nonstop"), "text", selected
            )


if __name__ == "__main__":
    unittest.main()
