"""Tests for the example and local community-rule reference."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from community_rules import (
    compare_live_rules,
    content_fingerprint,
    context_for_rules,
    load_reference,
    reference_status,
)


class TestCommunityRules(unittest.TestCase):
    def setUp(self):
        self.subreddit = "example_community"
        self.subreddit_env = patch.dict(
            os.environ,
            {
                "BOT_SUBREDDIT": self.subreddit,
                "BOT_COMMUNITY_RULES_REFERENCE_FILE": "data/community_rules.reference.example.json",
            },
            clear=False,
        )
        self.subreddit_env.start()
        self.addCleanup(self.subreddit_env.stop)

    def _synthetic_reference(self):
        reference = load_reference()
        reference["official_community_rules"] = [
            {
                "id": "community.example_rule",
                "name": "Example rule",
                "reddit_text": "Keep contributions relevant to the example community.",
            },
            {
                "id": "community.example_relevance",
                "name": "Relevance",
                "reddit_text": "Posts should relate to the sample community topic.",
            },
        ]
        reference["moderator_guidance"] = [
            {
                "id": "guidance.example_moderator_note",
                "name": "Example context",
                "description": "This fictional note is optional context for tests.",
            }
        ]
        reference["content_fingerprint"] = content_fingerprint(reference)
        return reference

    def test_public_example_reference_has_valid_fingerprint(self):
        reference = load_reference()
        status = reference_status(reference)
        self.assertTrue(status["fingerprintMatches"])
        self.assertEqual(status["computedFingerprint"], content_fingerprint(reference))

    def test_loading_a_reference_requires_an_explicit_subreddit(self):
        with patch.dict(
            os.environ,
            {"BOT_COMMUNITY_RULES_REFERENCE_FILE": "data/community_rules.reference.example.json"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "BOT_SUBREDDIT must be configured"):
                load_reference()

    def test_prompt_context_labels_guidance_as_not_official(self):
        context = json.loads(context_for_rules([], reference=self._synthetic_reference()))
        self.assertIn("official_rules", context)
        self.assertIn("moderator_guidance_not_official_rules", context)
        self.assertTrue(context["moderator_guidance_not_official_rules"])

    def test_prompt_context_uses_only_the_empty_public_example(self):
        context = json.loads(context_for_rules([]))
        model = context["moderation_model"]
        self.assertIn("No community-specific rules", model["description"])
        self.assertEqual(context["official_rules"], [])
        self.assertEqual(context["moderator_guidance_not_official_rules"], [])

    def test_live_rule_comparison_detects_change_without_updating_snapshot(self):
        original = load_reference()
        reference = self._synthetic_reference()
        live = [
            SimpleNamespace(short_name="Example rule", description="changed"),
            SimpleNamespace(
                short_name="Relevance",
                description="Posts should relate to the sample community topic.",
            ),
        ]
        result = compare_live_rules(reference, live)
        self.assertTrue(result["changed"])
        self.assertEqual(result["storedCount"], 2)
        self.assertEqual(result["liveCount"], 2)
        self.assertEqual(load_reference(), original)

    def test_live_rule_comparison_accepts_matching_rules(self):
        reference = load_reference()
        live = [
            SimpleNamespace(
                short_name=entry["name"],
                description=entry["reddit_text"],
            )
            for entry in reference["official_community_rules"]
        ]
        result = compare_live_rules(reference, live)
        self.assertFalse(result["changed"])


if __name__ == "__main__":
    unittest.main()
