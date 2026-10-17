"""Tests for the OpenRouter Jev structured-decision client."""

import json
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from jev_client import JevDecisionClient
from history_gate_jev import build_history_criteria_questions
from history_criteria_test_support import (
    ConfiguredHistoryCriteriaTestCase,
    synthetic_history_criteria_rules,
)


class _Response:
    def __init__(self, data):
        self.data = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.data


class TestJevDecisionClient(ConfiguredHistoryCriteriaTestCase):
    def test_posts_typed_noul_question_and_parses_probability_and_usage(self):
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _Response({
                "answers": {"needs_context": {"type": "noul", "noul": 0.81}},
                "usage": {"input_tokens": 44, "cost": 0.00001},
            })

        client = JevDecisionClient("test-key", opener=opener)
        result = client.probability_yes(
            question="Does the message need context?", state={"message": "Why?"}
        )
        self.assertEqual(result, 0.81)
        self.assertEqual(captured["url"], "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(captured["body"]["model"], "~typesafe/jev-latest")
        self.assertEqual(captured["body"]["questions"]["needs_context"]["type"], "noul")
        self.assertEqual(client.last_usage, {"input_tokens": 44, "cost": 0.00001})

    def test_missing_key_and_call_cap_fail_closed_to_no_decision(self):
        client = JevDecisionClient("", max_calls=1, opener=lambda *_a, **_k: None)
        self.assertIsNone(client.probability_yes(question="?", state={}))
        client = JevDecisionClient("key", max_calls=0, opener=lambda *_a, **_k: None)
        self.assertIsNone(client.probability_yes(question="?", state={}))

    def test_provider_error_returns_no_decision_without_exposing_request(self):
        client = JevDecisionClient("do-not-print-this-key", opener=lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError()))
        output = io.StringIO()
        with redirect_stdout(output):
            result = client.probability_yes(question="secret question", state={"text": "private"})
        self.assertIsNone(result)
        self.assertNotIn("do-not-print-this-key", output.getvalue())
        self.assertNotIn("private", output.getvalue())

    def test_http_error_records_status_without_logging_request_or_response_body(self):
        def opener(*_args, **_kwargs):
            raise HTTPError(
                "https://openrouter.ai/api/alpha/decisions",
                400,
                "Bad Request",
                {},
                io.BytesIO(b"private provider response details"),
            )

        client = JevDecisionClient("test-key", opener=opener)
        output = io.StringIO()
        with redirect_stdout(output):
            result = client.choose_many(
                state={"text": "private comment"},
                questions={"sample_check": {
                    "type": "choice", "instructions": "test",
                    "criteria": {"yes": "yes", "no": "no"},
                }},
            )

        self.assertEqual(result, {})
        self.assertEqual(client.last_error, {
            "type": "HTTPError", "status_code": 400, "reason": "Bad Request",
        })
        self.assertIn("HTTP 400 Bad Request", output.getvalue())
        self.assertNotIn("test-key", output.getvalue())
        self.assertNotIn("private comment", output.getvalue())
        self.assertNotIn("private provider response details", output.getvalue())

    def test_decide_many_sends_all_configured_history_criteria_in_one_request(self):
        captured = {}
        rules = synthetic_history_criteria_rules()
        questions = build_history_criteria_questions(rules)
        opener_calls = []

        def opener(request, timeout):
            opener_calls.append(1)
            captured["body"] = json.loads(request.data)
            return _Response({
                "model": "typesafe/jev-1.13",
                "answers": {
                    name: {"type": "noul", "noul": 0.93 if name == "sample_repeated_link_posting" else 0.04}
                    for name in questions
                },
                "usage": {"input_tokens": 88, "cost": 0.0001},
            })

        client = JevDecisionClient("test-key", opener=opener)
        results = client.decide_many(state={"reddit_history": [{"text": "synthetic"}]}, questions=questions)
        self.assertEqual(len(results), len(questions))
        self.assertEqual(results["sample_repeated_link_posting"], 0.93)
        self.assertEqual(client.calls_used, 1)
        self.assertEqual(opener_calls, [1])
        self.assertEqual(set(captured["body"]["questions"]), set(questions))
        self.assertTrue(all(question["type"] == "noul" for question in captured["body"]["questions"].values()))
        self.assertEqual(client.served_model, "typesafe/jev-1.13")

    def test_choose_many_returns_the_native_choice_label(self):
        captured = {}

        def opener(request, timeout):
            captured["body"] = json.loads(request.data)
            return _Response({
                "model": "typesafe/jev-1.13",
                "answers": {
                    "sample_check": {
                        "type": "choice",
                        "choice": "yes",
                        "probabilities": {"yes": 0.2, "no": 0.8},
                        "confidence": 0.8,
                    },
                    "negative_sample_check": {"type": "choice", "choice": "no"},
                    "invalid_label": {"type": "choice", "choice": "maybe"},
                },
                "usage": {"input_tokens": 55, "cost": 0.00002},
            })

        client = JevDecisionClient("test-key", opener=opener)
        questions = {
            "sample_check": {
                "type": "choice",
                "instructions": "Does the message include the requested detail?",
                "criteria": {"yes": "The detail is present", "no": "The detail is absent"},
            },
            "invalid_label": {
                "type": "choice",
                "instructions": "Choose yes or no.",
                "criteria": {"yes": "Yes", "no": "No"},
            },
            "negative_sample_check": {
                "type": "choice",
                "instructions": "Choose yes or no.",
                "criteria": {"yes": "Yes", "no": "No"},
            },
        }

        result = client.choose_many(state={"comments": ["example"]}, questions=questions)

        self.assertEqual(result, {"sample_check": "yes", "negative_sample_check": "no"})
        self.assertEqual(captured["body"]["questions"]["sample_check"]["type"], "choice")
        self.assertEqual(client.calls_used, 1)
        self.assertEqual(client.last_usage, {"input_tokens": 55, "cost": 0.00002})

    def test_choose_many_detailed_preserves_the_selected_probability_and_confidence(self):
        def opener(request, timeout):
            return _Response({
                "model": "typesafe/jev-1.13",
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": "none_of_the_above",
                        "probabilities": {
                            "category_a": 0.08,
                            "none_of_the_above": 0.92,
                            "unknown_option": 0.0,
                        },
                        "confidence": 0.84,
                    },
                    "invalid": {
                        "type": "choice",
                        "choice": "not_an_option",
                        "probabilities": {"yes": 0.99},
                        "confidence": 0.99,
                    },
                },
                "usage": {"input_tokens": 80, "cost": 0.00003},
            })

        client = JevDecisionClient("test-key", opener=opener)
        questions = {
            "category": {
                "type": "choice",
                "instructions": "Choose one category or none.",
                "criteria": {
                    "category_a": "Category A",
                    "none_of_the_above": "No category fits",
                },
            },
            "invalid": {
                "type": "choice",
                "instructions": "Choose yes or no.",
                "criteria": {"yes": "Yes", "no": "No"},
            },
        }

        result = client.choose_many_detailed(state={"comments": ["example"]}, questions=questions)

        self.assertEqual(result, {
            "category": {
                "choice": "none_of_the_above",
                "probabilities": {"category_a": 0.08, "none_of_the_above": 0.92},
                "confidence": 0.84,
            }
        })

    def test_choose_many_keeps_its_existing_simple_result_shape(self):
        class DetailClient(JevDecisionClient):
            def choose_many_detailed(self, *, state, questions):
                return {"category": {"choice": "none_of_the_above", "probabilities": {}}}

        client = DetailClient("test-key")
        result = client.choose_many(
            state={},
            questions={"category": {"type": "choice", "criteria": {"none_of_the_above": "None"}}},
        )
        self.assertEqual(result, {"category": "none_of_the_above"})




if __name__ == "__main__":
    unittest.main()
