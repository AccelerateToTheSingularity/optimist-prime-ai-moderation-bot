"""Tests for the short-lived verified-moderator rapid follow-up window."""

from __future__ import annotations

import json
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import moderator_activity


class ModeratorActivityTests(unittest.TestCase):
    def test_activity_survives_process_restart_and_expires_three_minutes_after_last_turn(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"BOT_RUNTIME_MODERATOR_ACTIVITY_FILE": os.path.join(directory, "activity.json")},
            clear=False,
        ):
            moderator_activity.record_moderator_activity(now=1000)
            # A second process can read the same tiny durable timestamp.
            self.assertTrue(moderator_activity.get_moderator_activity_status(now=1179)["active"])
            self.assertFalse(moderator_activity.get_moderator_activity_status(now=1180)["active"])

            moderator_activity.record_moderator_activity(now=1100)
            status = moderator_activity.get_moderator_activity_status(now=1279)
            self.assertTrue(status["active"])
            self.assertEqual(status["remaining_seconds"], 1)
            self.assertFalse(moderator_activity.get_moderator_activity_status(now=1280)["active"])

            saved = json.loads(moderator_activity.state_path().read_text(encoding="utf-8"))
            self.assertEqual(saved, {"version": 1, "last_interaction_at": 1100.0})
            self.assertNotIn("username", json.dumps(saved).casefold())

    def test_invalid_or_missing_activity_record_is_inactive(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"BOT_RUNTIME_MODERATOR_ACTIVITY_FILE": os.path.join(directory, "activity.json")},
            clear=False,
        ):
            self.assertFalse(moderator_activity.get_moderator_activity_status(now=1000)["active"])
            moderator_activity.state_path().write_text("not json", encoding="utf-8")
            self.assertFalse(moderator_activity.get_moderator_activity_status(now=1000)["active"])


if __name__ == "__main__":
    unittest.main()
