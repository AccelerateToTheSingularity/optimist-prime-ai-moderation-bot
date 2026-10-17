"""Tests for the discrete rule-based moderation system."""

import json
import os
import tempfile
import unittest

from moderation_rules import (
    ModerationRule,
    RuleLoadError,
    load_rules,
    filter_rules,
)
from content_moderation import (
    parse_rule_response,
    evaluate_rules,
)


# ---------------------------------------------------------------------------
# ModerationRule dataclass
# ---------------------------------------------------------------------------

class TestModerationRule(unittest.TestCase):
    def test_basic_properties(self):
        rule = ModerationRule(
            name="test_rule",
            description="Is this a test?",
            conditions={"stop_on_match": True, "skip_mods": True, "skip_approved": False},
        )
        self.assertTrue(rule.stop_on_match)
        self.assertTrue(rule.skip_mods)
        self.assertFalse(rule.skip_approved)

    def test_defaults(self):
        rule = ModerationRule(name="r", description="d")
        self.assertTrue(rule.active)
        self.assertEqual(rule.order, 100)
        self.assertEqual(rule.target, "both")
        self.assertEqual(rule.actions, [])
        self.assertFalse(rule.stop_on_match)
        self.assertFalse(rule.skip_mods)
        self.assertFalse(rule.skip_approved)
        self.assertFalse(rule.use_vision)

    def test_applies_to(self):
        rule_both = ModerationRule(name="r", description="d", target="both")
        rule_posts = ModerationRule(name="r", description="d", target="posts")
        rule_comments = ModerationRule(name="r", description="d", target="comments")

        self.assertTrue(rule_both.applies_to("posts"))
        self.assertTrue(rule_both.applies_to("comments"))
        self.assertTrue(rule_posts.applies_to("posts"))
        self.assertFalse(rule_posts.applies_to("comments"))
        self.assertTrue(rule_comments.applies_to("comments"))
        self.assertFalse(rule_comments.applies_to("posts"))


# ---------------------------------------------------------------------------
# Rule loading and validation
# ---------------------------------------------------------------------------

class TestLoadRules(unittest.TestCase):
    def _write_rules(self, rules_list):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(rules_list, f)
        f.close()
        return f.name

    def test_load_valid_rules(self):
        path = self._write_rules([
            {"name": "rule_a", "description": "Is it A?", "order": 20},
            {"name": "rule_b", "description": "Is it B?", "order": 10},
        ])
        try:
            rules = load_rules(path)
            self.assertEqual(len(rules), 2)
            # Should be sorted by order
            self.assertEqual(rules[0].name, "rule_b")
            self.assertEqual(rules[1].name, "rule_a")
        finally:
            os.unlink(path)

    def test_missing_name_raises(self):
        path = self._write_rules([{"description": "No name"}])
        try:
            with self.assertRaises(RuleLoadError):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_missing_description_raises(self):
        path = self._write_rules([{"name": "rule_a"}])
        try:
            with self.assertRaises(RuleLoadError):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_space_in_name_raises(self):
        path = self._write_rules([{"name": "has space", "description": "d"}])
        try:
            with self.assertRaises(RuleLoadError):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_invalid_target_raises(self):
        path = self._write_rules([{"name": "r", "description": "d", "target": "images"}])
        try:
            with self.assertRaises(RuleLoadError):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_invalid_action_raises(self):
        path = self._write_rules([{"name": "r", "description": "d", "actions": ["explode"]}])
        try:
            with self.assertRaises(RuleLoadError):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_non_array_raises(self):
        path = self._write_rules({"name": "r", "description": "d"})
        try:
            with self.assertRaises(RuleLoadError):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(RuleLoadError):
            load_rules("/nonexistent/path/rules.json")

    def test_env_var_override(self):
        rules_json = json.dumps([
            {"name": "env_rule", "description": "From env"}
        ])
        os.environ["BOT_MODERATION_RULES_JSON"] = rules_json
        try:
            rules = load_rules()
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].name, "env_rule")
        finally:
            del os.environ["BOT_MODERATION_RULES_JSON"]

    def test_env_var_invalid_json_raises(self):
        os.environ["BOT_MODERATION_RULES_JSON"] = "not valid json"
        try:
            with self.assertRaises(RuleLoadError):
                load_rules()
        finally:
            del os.environ["BOT_MODERATION_RULES_JSON"]

    def test_inactive_rule_loaded(self):
        path = self._write_rules([
            {"name": "active_rule", "description": "A", "active": True, "actions": ["log"]},
            {"name": "inactive_rule", "description": "I", "active": False},
        ])
        try:
            rules = load_rules(path)
            self.assertEqual(len(rules), 2)
            active = [r for r in rules if r.active]
            inactive = [r for r in rules if not r.active]
            self.assertEqual(len(active), 1)
            self.assertEqual(len(inactive), 1)
        finally:
            os.unlink(path)

    def test_active_rule_without_explicit_action_is_rejected(self):
        path = self._write_rules([{
            "name": "example_rule",
            "description": "A synthetic operator-defined rule.",
            "active": True,
        }])
        try:
            with self.assertRaisesRegex(RuleLoadError, "Choose an action"):
                load_rules(path)
        finally:
            os.unlink(path)

    def test_operator_rule_can_start_inactive_without_actions_or_history_policy_flags(self):
        path = self._write_rules([{
            "name": "sample_rule",
            "description": "A rule written by the operator.",
            "active": False,
            "actions": [],
        }])
        try:
            rules = load_rules(path)
            self.assertEqual(len(rules), 1)
            self.assertFalse(rules[0].active)
            self.assertEqual(rules[0].actions, [])
            self.assertEqual(filter_rules(rules, "comments"), [])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Rule filtering
# ---------------------------------------------------------------------------

class TestFilterRules(unittest.TestCase):
    def _make_rules(self):
        return [
            ModerationRule(name="post_only", description="d", target="posts", active=True),
            ModerationRule(name="comment_only", description="d", target="comments", active=True),
            ModerationRule(name="both_active", description="d", target="both", active=True),
            ModerationRule(name="both_inactive", description="d", target="both", active=False),
        ]

    def test_filter_posts(self):
        rules = filter_rules(self._make_rules(), "posts")
        names = [r.name for r in rules]
        self.assertIn("post_only", names)
        self.assertIn("both_active", names)
        self.assertNotIn("comment_only", names)
        self.assertNotIn("both_inactive", names)

    def test_filter_comments(self):
        rules = filter_rules(self._make_rules(), "comments")
        names = [r.name for r in rules]
        self.assertIn("comment_only", names)
        self.assertIn("both_active", names)
        self.assertNotIn("post_only", names)

    def test_active_only(self):
        rules = filter_rules(self._make_rules(), "both", active_only=True)
        names = [r.name for r in rules]
        self.assertNotIn("both_inactive", names)

    def test_include_inactive(self):
        rules = filter_rules(self._make_rules(), "both", active_only=False)
        names = [r.name for r in rules]
        self.assertIn("both_inactive", names)

    def test_sorted_by_order(self):
        rules = [
            ModerationRule(name="c", description="d", order=30),
            ModerationRule(name="a", description="d", order=10),
            ModerationRule(name="b", description="d", order=20),
        ]
        filtered = filter_rules(rules, "both")
        self.assertEqual([r.name for r in filtered], ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseRuleResponse(unittest.TestCase):
    def _rules(self):
        return [
            ModerationRule(name="spam_detector", description="d"),
            ModerationRule(name="toxicity_detector", description="d"),
            ModerationRule(name="off_topic_detector", description="d"),
        ]

    def test_basic_parsing(self):
        response = "spam_detector: YES - obvious spam\noff_topic_detector: NO"
        results = parse_rule_response(response, self._rules())
        self.assertTrue(results["spam_detector"]["matched"])
        self.assertIn("obvious spam", results["spam_detector"]["reason"])
        self.assertFalse(results["toxicity_detector"]["matched"])
        self.assertFalse(results["off_topic_detector"]["matched"])

    def test_all_yes(self):
        response = (
            "spam_detector: YES\n"
            "toxicity_detector: YES\n"
            "off_topic_detector: YES"
        )
        results = parse_rule_response(response, self._rules())
        for rule in self._rules():
            self.assertTrue(results[rule.name]["matched"])

    def test_all_no(self):
        response = (
            "spam_detector: NO\n"
            "toxicity_detector: NO\n"
            "off_topic_detector: NO"
        )
        results = parse_rule_response(response, self._rules())
        for rule in self._rules():
            self.assertFalse(results[rule.name]["matched"])

    def test_case_insensitive(self):
        response = "SPAM_DETECTOR: Yes - spam"
        results = parse_rule_response(response, self._rules())
        self.assertTrue(results["spam_detector"]["matched"])

    def test_unmentioned_rules_default_no(self):
        response = "spam_detector: YES"
        results = parse_rule_response(response, self._rules())
        self.assertTrue(results["spam_detector"]["matched"])
        self.assertFalse(results["toxicity_detector"]["matched"])
        self.assertFalse(results["off_topic_detector"]["matched"])

    def test_empty_response(self):
        results = parse_rule_response("", self._rules())
        for rule in self._rules():
            self.assertFalse(results[rule.name]["matched"])

    def test_dashes_in_reason(self):
        response = "spam_detector: YES - this is spam - clearly"
        results = parse_rule_response(response, self._rules())
        self.assertTrue(results["spam_detector"]["matched"])
        self.assertIn("this is spam", results["spam_detector"]["reason"])

    def test_normalized_name_matching(self):
        response = "off-topic-detector: YES"
        results = parse_rule_response(response, self._rules())
        self.assertTrue(results["off_topic_detector"]["matched"])


# ---------------------------------------------------------------------------
# Jev-authoritative evaluation

class FakeJev:
    ready = True
    max_calls = 10
    last_usage = {"input_tokens": 17, "cost": 0.0001}

    def __init__(self, answers):
        self.answers = answers
        self.calls_used = 0
        self.questions = []

    def choose_many(self, *, state, questions):
        self.calls_used += 1
        self.questions.append(questions)
        return self.answers


class TestEvaluateRules(unittest.TestCase):
    def test_non_stop_rules_require_jev_and_never_fall_back_to_llm(self):
        class ForbiddenModel:
            def generate_content(self, contents, generation_config=None):
                raise AssertionError("LLM cannot classify moderation rules")

        rules = [ModerationRule(name="rule_a", description="Is it A?")]
        self.assertIsNone(evaluate_rules("content", rules, ForbiddenModel()))

    def test_non_stop_rules_use_one_jev_request_then_verified_evidence(self):
        jev = FakeJev({
            "moderation_rule_1": "rule_matches",
            "moderation_rule_2": "rule_does_not_match",
        })

        class CapturingModel:
            def __init__(self):
                self.calls = []

            def generate_content(self, contents, generation_config=None):
                self.calls.append((contents, generation_config))
                return type("Response", (), {
                    "text": '{"items":[{"rule":"rule_a","reason":"A applies",'
                    '"evidence":"exact words"}]}',
                    "usage_metadata": type("Usage", (), {
                        "prompt_token_count": 12,
                        "candidates_token_count": 4,
                    })(),
                })()

        model = CapturingModel()
        rules = [
            ModerationRule(name="rule_a", description="Is it A?", order=1),
            ModerationRule(name="rule_b", description="Is it B?", order=2),
        ]

        result = evaluate_rules("exact words here", rules, model, jev_client=jev)

        self.assertEqual([item["rule"].name for item in result["matches"]], ["rule_a"])
        self.assertTrue(result["decisions"]["rule_a"]["matched"])
        self.assertFalse(result["decisions"]["rule_b"]["matched"])
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(len(jev.questions), 1)
        self.assertEqual(len(jev.questions[0]), 2)
        self.assertEqual(len(model.calls), 1)
        evidence_prompt = model.calls[0][0][0]["parts"][0]
        self.assertIn("Do not reconsider, add, remove, or replace", evidence_prompt)
        self.assertIn('"name": "rule_a"', evidence_prompt)
        self.assertNotIn('"name": "rule_b"', evidence_prompt)

    def test_jev_no_match_skips_llm_and_empty_rules_need_no_jev(self):
        class ForbiddenModel:
            def generate_content(self, contents, generation_config=None):
                raise AssertionError("LLM cannot classify moderation rules")

        empty = evaluate_rules("content", [], ForbiddenModel())
        self.assertEqual(empty["matches"], [])
        self.assertEqual(empty["decisions"], {})

        jev = FakeJev({
            "moderation_rule_1": "rule_does_not_match",
        })
        rules = [ModerationRule(name="rule_a", description="Is it A?")]
        result = evaluate_rules("content", rules, ForbiddenModel(), jev_client=jev)
        self.assertFalse(result["matches"])
        self.assertFalse(result["decisions"]["rule_a"]["matched"])
        self.assertEqual(jev.calls_used, 1)


if __name__ == "__main__":
    unittest.main()
