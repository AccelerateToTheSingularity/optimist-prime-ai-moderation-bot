import os
import unittest
from unittest.mock import patch

import config
from runtime_settings import apply_runtime_settings, resolve_runtime_settings

class TestRuntimeSettings(unittest.TestCase):
    def setUp(self):
        self._config_values = {
            name: getattr(config, name)
            for name in (
                "SUBREDDIT",
                "POST_TLDR_ENABLED",
                "COMMENT_TLDR_ENABLED",
                "COMMENT_SUMMARY_ENABLED",
                "CROSSPOST_ENABLED",
                "SUMMON_PATTERNS",
                "MAX_TLDR_PER_RUN",
                "MAX_REPLIES_PER_RUN",
                "MAX_TLDR_PER_DAY",
                "MAX_REPLIES_PER_DAY",
                "POST_SCAN_LIMIT",
                "COMMENT_SCAN_LIMIT",
                "MAX_AGE_HOURS",
                "POST_WORD_THRESHOLD",
                "SAFE_MODE",
                "SAME_USER_COOLDOWN_HOURS",
                "SAME_USER_REPLIES_BEFORE_COOLDOWN",
                "MODERATOR_REPLIES_BYPASS_LIMITS",
            )
            if hasattr(config, name)
        }

    def tearDown(self):
        for name, value in self._config_values.items():
            setattr(config, name, list(value) if name == "SUMMON_PATTERNS" else value)

    def test_default_settings_leave_optional_features_off(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = resolve_runtime_settings()
        self.assertFalse(settings.post_tldr_enabled)
        self.assertFalse(settings.comment_tldr_enabled)
        self.assertFalse(settings.summons_enabled)
        self.assertFalse(settings.crosspost_enabled)

    def test_default_scan_window_and_limits_are_bounded(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = resolve_runtime_settings()
        self.assertEqual(settings.post_scan_limit, 100)
        self.assertEqual(settings.comment_scan_limit, 100)
        self.assertEqual(settings.max_age_hours, 24)
        self.assertEqual(settings.max_tldr_per_day, 10)
        self.assertEqual(settings.max_replies_per_day, 10)

    def test_fresh_install_keeps_high_impact_actions_disabled(self):
        self.assertTrue(config.SAFE_MODE)
        self.assertFalse(config.CONTENT_MODERATION_ENABLED)
        self.assertFalse(config.AUTOMATED_BAN_ENABLED)
        self.assertIsNone(config.HISTORY_REVIEW_BAN_THRESHOLD_PERCENT)

    def test_invalid_safe_mode_value_uses_safe_default(self):
        with patch.dict(os.environ, {"BOT_SAFE_MODE": "treu"}, clear=False):
            self.assertTrue(config._env_on("BOT_SAFE_MODE", True))
            settings = resolve_runtime_settings()
        self.assertTrue(settings.safe_mode)

    def test_invalid_reply_mode_falls_back_to_local_logging(self):
        self.assertEqual(config._parse_engagement_action_mode("commment"), "log")
        self.assertEqual(config._parse_engagement_action_mode("comment"), "comment")

    def test_reply_defaults_are_bounded_and_generic(self):
        self.assertEqual(config.MAX_TLDR_PER_DAY, 10)
        self.assertEqual(config.MAX_REPLIES_PER_DAY, 10)
        self.assertEqual(config.SAME_USER_REPLIES_BEFORE_COOLDOWN, 5)
        self.assertFalse(config.MODERATOR_REPLIES_BYPASS_LIMITS)

    def test_apply_updates_config_module(self):
        with patch.dict(os.environ, {
            "BOT_SUBREDDIT": "example_community",
            "BOT_POST_TLDR_ENABLED": "true",
        }, clear=False):
            settings = resolve_runtime_settings(bot_username="random")
            apply_runtime_settings(settings)
        self.assertEqual(config.SUBREDDIT, "example_community")
        self.assertTrue(config.POST_TLDR_ENABLED)
        self.assertFalse(config.CROSSPOST_ENABLED)

    def test_explicit_limits_override_runtime_defaults(self):
        with patch.dict(os.environ, {
            "BOT_MAX_TLDR_PER_RUN": "4",
            "BOT_MAX_REPLIES_PER_RUN": "0",
            "BOT_MAX_TLDR_PER_DAY": "8",
            "BOT_MAX_REPLIES_PER_DAY": "6",
        }, clear=False):
            settings = resolve_runtime_settings()
            apply_runtime_settings(settings)
        self.assertEqual(config.MAX_TLDR_PER_RUN, 4)
        self.assertEqual(config.MAX_REPLIES_PER_RUN, 0)
        self.assertEqual(config.MAX_TLDR_PER_DAY, 8)
        self.assertEqual(config.MAX_REPLIES_PER_DAY, 6)

if __name__ == "__main__":
    unittest.main()
