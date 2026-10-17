"""Safety tests for operator-configured history criteria and durable retry state."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import config
from history_criteria_test_support import ConfiguredHistoryCriteriaTestCase
from history_criteria_enforcement import (
    apply_history_criteria_ban,
    deliver_review_modmail,
    format_history_criteria_matches,
    history_autoban_action_key,
    history_autoban_notification_key,
    load_history_assessment,
    queue_review_modmail,
    reconcile_pending_autoban_notices,
    save_history_assessment,
    triggering_history_criteria_scores,
)
from history_criteria_test_support import current_history_criteria_revisions


TARGET = "sample_repeated_link_posting"
REMOVED = "sample_direct_personal_attack"


def assessment(scores, *, complete=True):
    return SimpleNamespace(
        criterion_probabilities=dict(scores),
        matched_rules=tuple(name for name, score in scores.items() if score >= 0.67),
        rule_names=tuple(scores),
        match_threshold=0.67,
        scores_complete=complete,
        missing_criterion_ids=(),
        criterion_score_revisions=current_history_criteria_revisions(),
        evidence_truncated_items=0,
        served_model="fake-jev",
    )


class _BannedUsers:
    def __init__(self, existing=(), *, fail_add=0):
        self.existing = set(existing)
        self.lookups = []
        self.adds = []
        self.fail_add = fail_add

    def __call__(self, *, redditor, limit):
        self.lookups.append((redditor, limit))
        return iter([redditor] if redditor in self.existing else [])

    def add(self, redditor, **settings):
        self.adds.append((redditor, settings))
        if self.fail_add:
            self.fail_add -= 1
            raise RuntimeError("simulated write failure")
        self.existing.add(redditor)


class _Subreddit:
    def __init__(self, existing=(), *, fail_add=0):
        self.banned = _BannedUsers(existing, fail_add=fail_add)


class HistoryCriteriaEnforcementTests(ConfiguredHistoryCriteriaTestCase):
    def test_threshold_is_strict_and_scores_must_be_valid_and_active(self):
        scores = {
            TARGET: 0.87,
            "sample_contact_information": 0.871,
            "sample_direct_personal_attack": float("nan"),
            "retired_unused_criterion": 1.0,
        }
        with patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87):
            self.assertEqual(
                triggering_history_criteria_scores(assessment(scores)),
                (("sample_contact_information", 0.871),),
            )
        self.assertEqual(
            format_history_criteria_matches(assessment({"sample_repeated_link_posting": 1.0})),
            "Sample Repeated Link Posting (100%)",
        )

    def test_incomplete_score_set_does_not_trigger_a_ban(self):
        subreddit = _Subreddit()
        state = {}
        fake = assessment({TARGET: 0.951, "sample_contact_information": 0.1}, complete=False)
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87), \
             patch("reddit_observability.single_write_attempt", return_value=nullcontext()):
            result = apply_history_criteria_ban(
                subreddit, "example_user", fake, state, source_key="report:one"
            )
        self.assertEqual(result.status, "not_eligible")
        self.assertEqual(result.triggering_scores, ())
        self.assertEqual(subreddit.banned.lookups, [])
        self.assertEqual(subreddit.banned.adds, [])
        self.assertEqual(state, {})

    def test_exact_threshold_recommendation_does_not_ban(self):
        subreddit = _Subreddit()
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87):
            result = apply_history_criteria_ban(
                subreddit, "example_user", assessment({TARGET: 0.87}), {}, source_key="manual:one"
            )
        self.assertEqual(result.status, "not_eligible")
        self.assertEqual(subreddit.banned.lookups, [])

    def test_disabled_and_dry_run_never_check_or_write_bans(self):
        subreddit = _Subreddit()
        fake = assessment({TARGET: 0.99})
        with patch.object(config, "AUTOMATED_BAN_ENABLED", False):
            disabled = apply_history_criteria_ban(
                subreddit, "example_user", fake, {}, source_key="manual:disabled"
            )
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87):
            preview = apply_history_criteria_ban(
                subreddit, "example_user", fake, {}, source_key="manual:preview", dry_run=True
            )
        self.assertEqual(disabled.status, "disabled")
        self.assertEqual(preview.status, "would_ban")
        self.assertEqual(subreddit.banned.lookups, [])
        self.assertEqual(subreddit.banned.adds, [])

    def test_existing_ban_is_detected_without_a_second_write(self):
        subreddit = _Subreddit({"example_user"})
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87):
            result = apply_history_criteria_ban(
                subreddit, "example_user", assessment({TARGET: 0.95}), {}, source_key="saved:one"
            )
        self.assertEqual(result.status, "already_banned")
        self.assertEqual(subreddit.banned.adds, [])

    def test_failed_ban_can_retry_and_checks_existing_ban_first(self):
        subreddit = _Subreddit(fail_add=1)
        state = {}
        fake = assessment({TARGET: 0.95})
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87), \
             patch("reddit_observability.single_write_attempt", return_value=nullcontext()):
            first = apply_history_criteria_ban(
                subreddit, "example_user", fake, state, source_key="saved:one"
            )
            second = apply_history_criteria_ban(
                subreddit, "example_user", fake, state, source_key="saved:one"
            )
        self.assertEqual(first.status, "ban_failed")
        self.assertEqual(second.status, "banned")
        self.assertEqual(len(subreddit.banned.adds), 2)
        self.assertEqual(len(subreddit.banned.lookups), 2)

    def test_saved_scores_and_modmail_outbox_survive_retries(self):
        state = {}
        source_key = "reported:stable"
        saved = assessment({TARGET: 0.95})
        save_history_assessment(
            state, "example_user", source_key, saved,
            history_item_count=42, history_incomplete=False,
        )
        restored = load_history_assessment(state, "example_user", source_key)
        self.assertIsNotNone(restored)
        self.assertEqual(restored[0].criterion_probabilities[TARGET], 0.95)
        self.assertEqual(restored[1:], (42, False))
        self.assertIsNone(load_history_assessment(state, "bob", source_key))

        key = queue_review_modmail(
            state,
            notification_key="review-one",
            username="example_user",
            subject="History review [one]",
            body="A moderator-only result.",
        )
        self.assertIn(key, state["reported_user_reviews"]["pending_notifications"])

        conversations = []
        def find(**_kwargs):
            return iter(conversations)
        def create(*, subject, body, recipient):
            self.assertIsNone(recipient)
            conversation = SimpleNamespace(subject=subject, body=body, is_internal=True, id="one")
            conversations.append(conversation)
            return conversation
        subreddit = SimpleNamespace(modmail=SimpleNamespace(conversations=find, create=create))
        self.assertTrue(deliver_review_modmail(subreddit, state, key))
        self.assertEqual(state["reported_user_reviews"]["sent_notifications"][key]["subject"], "History review [one]")
        self.assertFalse(deliver_review_modmail(subreddit, state, key))

    def test_saved_scores_from_an_old_guide_revision_are_not_reused(self):
        state = {}
        source_key = "reported:before-disable"
        saved = assessment({TARGET: 1.0})
        saved.criterion_score_revisions = {}
        save_history_assessment(
            state, "example_user", source_key, saved,
            history_item_count=42, history_incomplete=False,
        )
        saved_record = next(iter(state["history_criteria_reviews"].values()))
        self.assertNotIn(
            TARGET,
            saved_record["criterion_probabilities"],
        )
        self.assertNotIn(
            TARGET,
            saved_record["matched_rules"],
        )
        # Simulate a record written before score revisions existed.
        saved_record["criterion_probabilities"][TARGET] = 1.0
        saved_record["matched_rules"].append(TARGET)
        saved_record["rule_names"].append(TARGET)

        restored, count, incomplete = load_history_assessment(
            state, "example_user", source_key
        )

        self.assertEqual(count, 42)
        self.assertFalse(incomplete)
        self.assertIn(TARGET, restored.rule_names)
        self.assertNotIn(TARGET, restored.criterion_probabilities)
        self.assertNotIn(TARGET, restored.matched_rules)
        self.assertIn(TARGET, restored.missing_criterion_ids)
        self.assertIn(TARGET, restored.outdated_criterion_ids)
        self.assertFalse(restored.scores_complete)

    def test_current_guide_scores_survive_save_load_and_can_trigger_policy(self):
        name = "sample_repeated_link_posting"
        current = assessment({name: 0.95})
        current.criterion_score_revisions = {name: 2}
        state = {}
        save_history_assessment(
            state, "example_user", "requested:new-definition", current,
            history_item_count=10, history_incomplete=False,
        )
        restored, _count, _incomplete = load_history_assessment(
            state, "example_user", "requested:new-definition"
        )
        self.assertEqual(restored.criterion_probabilities[name], 0.95)
        self.assertEqual(restored.criterion_score_revisions[name], 2)
        self.assertEqual(triggering_history_criteria_scores(restored, threshold_percent=87), ((name, 0.95),))
        self.assertIn("Repeated Link Posting", format_history_criteria_matches(restored))

    def test_invalid_cached_other_score_does_not_discard_a_valid_qualifying_score(self):
        state = {}
        save_history_assessment(
            state, "example_user", "requested:partial", assessment({TARGET: 0.95}),
            history_item_count=10, history_incomplete=False,
        )
        record = next(iter(state["history_criteria_reviews"].values()))
        record["criterion_probabilities"]["sample_contact_information"] = "invalid"
        restored, _count, _incomplete = load_history_assessment(state, "example_user", "requested:partial")
        self.assertEqual(triggering_history_criteria_scores(restored, threshold_percent=87), ((TARGET, 0.95),))
        self.assertFalse(restored.scores_complete)

    def test_old_guide_pending_ban_is_cancelled_without_reddit_access(self):
        source_key = "reported:old-definition-crash"
        key = history_autoban_action_key("example_user", source_key)
        for enabled in (True, False):
            with self.subTest(auto_ban_enabled=enabled):
                subreddit = _Subreddit({"example_user"})
                state = {"history_criteria_ban_actions": {key: {
                    "username": "example_user", "source_key": source_key,
                    "status": "ban_pending", "threshold_percent": 87,
                    "triggering_scores": [["sample_repeated_link_posting", 1.0]],
                }}}
                with patch.object(config, "AUTOMATED_BAN_ENABLED", enabled), \
                     patch.object(config, "SAFE_MODE", False), \
                     patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87):
                    self.assertEqual(reconcile_pending_autoban_notices(subreddit, state), 0)
                self.assertEqual(subreddit.banned.lookups, [])
                self.assertEqual(subreddit.banned.adds, [])
                self.assertEqual(state["history_criteria_ban_actions"][key]["status"], "cancelled_criterion_definition_changed")

    def test_restart_recovers_ban_and_sends_one_moderator_notice(self):
        subreddit = _Subreddit()
        conversations = []
        subreddit.modmail = SimpleNamespace(
            conversations=lambda **_kwargs: iter(conversations),
            create=lambda *, subject, body, recipient: conversations.append(
                SimpleNamespace(subject=subject, body=body, is_internal=True, id="one")
            ) or conversations[-1],
        )
        source_key = "reported:crash-window"
        action_key = history_autoban_action_key("example_user", source_key)
        state = {"history_criteria_ban_actions": {action_key: {
            "username": "example_user",
            "source_key": source_key,
            "status": "ban_pending",
            "threshold_percent": 87,
            "triggering_scores": [[TARGET, 0.95]],
            "criterion_score_revisions": {TARGET: current_history_criteria_revisions()[TARGET]},
            "notice_links": ["https://www.reddit.com/r/example_community/comments/post/title/comment/"],
            "moderator_notice_status": "pending",
        }}}
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87), \
             patch("reddit_observability.single_write_attempt", return_value=nullcontext()):
            delivered = reconcile_pending_autoban_notices(subreddit, state)
        self.assertEqual(delivered, 1)
        self.assertEqual(len(subreddit.banned.adds), 1)
        self.assertEqual(state["history_criteria_ban_actions"][action_key]["status"], "banned")
        self.assertEqual(state["history_criteria_ban_actions"][action_key]["moderator_notice_status"], "sent")
        self.assertEqual(len(conversations), 1)
        self.assertIn("95%", conversations[0].body)
        self.assertIn("was banned after a history review", conversations[0].body)
        self.assertIn("https://www.reddit.com/r/example_community/comments/post/title/comment/", conversations[0].body)

    def test_notice_retry_after_ban_uses_one_outbox_key(self):
        subreddit = _Subreddit({"example_user"})
        conversations = []
        subreddit.modmail = SimpleNamespace(
            conversations=lambda **_kwargs: iter(conversations),
            create=lambda *, subject, body, recipient: conversations.append(
                SimpleNamespace(subject=subject, body=body, is_internal=True, id="one")
            ) or conversations[-1],
        )
        source_key = "requested:timeout"
        action_key = history_autoban_action_key("example_user", source_key)
        notice_key = history_autoban_notification_key("example_user", source_key, "already_banned")
        state = {"history_criteria_ban_actions": {action_key: {
            "username": "example_user",
            "source_key": source_key,
            "status": "already_banned",
            "threshold_percent": 87,
            "triggering_scores": [[TARGET, 0.95]],
            "notification_key": notice_key,
            "moderator_notice_status": "pending",
        }}}
        with patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False):
            self.assertEqual(reconcile_pending_autoban_notices(subreddit, state), 1)
            self.assertEqual(reconcile_pending_autoban_notices(subreddit, state), 0)
        self.assertEqual(len(conversations), 1)

    def test_restart_does_not_retry_pending_ban_after_switch_is_turned_off(self):
        subreddit = _Subreddit()
        source_key = "reported:disabled-after-crash"
        action_key = history_autoban_action_key("example_user", source_key)
        state = {"history_criteria_ban_actions": {action_key: {
            "username": "example_user",
            "source_key": source_key,
            "status": "ban_pending",
            "threshold_percent": 87,
            "triggering_scores": [[TARGET, 0.95]],
            "criterion_score_revisions": {TARGET: current_history_criteria_revisions()[TARGET]},
            "moderator_notice_status": "pending",
        }}}
        with patch.object(config, "AUTOMATED_BAN_ENABLED", False):
            reconcile_pending_autoban_notices(subreddit, state)
        self.assertEqual(subreddit.banned.adds, [])
        self.assertEqual(state["history_criteria_ban_actions"][action_key]["status"], "cancelled_disabled")


if __name__ == "__main__":
    unittest.main()
