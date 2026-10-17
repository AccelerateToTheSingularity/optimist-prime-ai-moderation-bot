"""Tests for the short-lived fast-testing lease and worker wake-up."""

import json
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fast_test_mode
import local_worker


class TestFastTestLease(unittest.TestCase):
    def test_lease_uses_3_second_spacing_and_expires_after_2_minutes(self):
        with TemporaryDirectory() as folder, patch.dict(
            "os.environ",
            {"BOT_RUNTIME_FAST_TEST_MODE_FILE": str(Path(folder) / "lease.json")},
            clear=False,
        ):
            started = fast_test_mode.start_fast_test_mode(now=1000)
            self.assertTrue(started["active"])
            self.assertEqual(started["remaining_seconds"], 120)
            self.assertEqual(started["interval_seconds"], 3)
            payload = json.loads(fast_test_mode.state_path().read_text(encoding="utf-8"))
            self.assertEqual(payload["expires_at"] - payload["started_at"], 120)
            self.assertTrue(fast_test_mode.get_fast_test_mode(now=1119)["active"])
            self.assertFalse(fast_test_mode.get_fast_test_mode(now=1120)["active"])

    def test_restarting_lease_resets_its_two_minute_window(self):
        with TemporaryDirectory() as folder, patch.dict(
            "os.environ",
            {"BOT_RUNTIME_FAST_TEST_MODE_FILE": str(Path(folder) / "lease.json")},
            clear=False,
        ):
            fast_test_mode.start_fast_test_mode(now=1000)
            restarted = fast_test_mode.start_fast_test_mode(now=1050)
            self.assertEqual(restarted["remaining_seconds"], 120)
            self.assertTrue(fast_test_mode.get_fast_test_mode(now=1169)["active"])
            self.assertFalse(fast_test_mode.get_fast_test_mode(now=1170)["active"])

    def test_command_line_switch_uses_configured_owner_local_path(self):
        with TemporaryDirectory() as folder, patch.dict(
            "os.environ",
            {"BOT_RUNTIME_FAST_TEST_MODE_FILE": str(Path(folder) / "lease.json")},
            clear=False,
        ), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(fast_test_mode.main(["start"]), 0)
            self.assertTrue(fast_test_mode.get_fast_test_mode()["active"])
            self.assertIn('"interval_seconds": 3', output.getvalue())
            self.assertEqual(fast_test_mode.main(["stop"]), 0)
            self.assertFalse(fast_test_mode.get_fast_test_mode()["active"])

    def test_stop_removes_lease_and_malformed_lease_is_inactive(self):
        with TemporaryDirectory() as folder, patch.dict(
            "os.environ",
            {"BOT_RUNTIME_FAST_TEST_MODE_FILE": str(Path(folder) / "lease.json")},
            clear=False,
        ):
            fast_test_mode.start_fast_test_mode(now=1000)
            self.assertFalse(fast_test_mode.stop_fast_test_mode()["active"])
            self.assertFalse(fast_test_mode.state_path().exists())
            fast_test_mode.state_path().write_text("not json", encoding="utf-8")
            self.assertFalse(fast_test_mode.get_fast_test_mode(now=1001)["active"])


class TestFastTestWorkerWake(unittest.TestCase):
    def test_fast_mode_waits_after_cycle_instead_of_catching_up_back_to_back(self):
        self.assertEqual(
            local_worker._next_cycle_deadline(
                25,
                3,
                fast_test_active=True,
                now=40,
            ),
            43,
        )
        self.assertEqual(
            local_worker._next_cycle_deadline(
                25,
                60,
                fast_test_active=False,
                now=40,
            ),
            85,
        )

    def test_worker_wakes_early_when_fast_mode_is_turned_on(self):
        status = iter([{"active": True}])
        sleeps = []
        with patch.object(local_worker.time, "monotonic", side_effect=[10.0, 10.0, 11.0]), patch.object(
            local_worker.time, "sleep", side_effect=lambda seconds: sleeps.append(seconds)
        ):
            woke = local_worker._wait_for_next_cycle(
                60,
                status_reader=lambda: next(status),
                sleeper=lambda seconds: sleeps.append(seconds),
                mode_was_active=False,
            )
        self.assertEqual(woke, "activated")
        self.assertEqual(sleeps, [1.0])

    def test_stopping_fast_mode_interrupts_its_short_wait(self):
        status = iter([{"active": False}])
        sleeps = []
        with patch.object(local_worker.time, "monotonic", side_effect=[10.0, 10.0]):
            result = local_worker._wait_for_next_cycle(
                3,
                status_reader=lambda: next(status),
                sleeper=lambda seconds: sleeps.append(seconds),
                mode_was_active=True,
            )
        self.assertEqual(result, "deactivated")
        self.assertEqual(sleeps, [1.0])

    def test_normal_wait_reaches_deadline_without_fast_mode(self):
        sleeps = []
        with patch.object(local_worker.time, "monotonic", side_effect=[10.0, 10.0, 11.0]):
            result = local_worker._wait_for_next_cycle(
                0.5,
                status_reader=lambda: {"active": False},
                sleeper=lambda seconds: sleeps.append(seconds),
                mode_was_active=False,
            )
        self.assertEqual(result, "deadline")
        self.assertEqual(sleeps, [0.5])


if __name__ == "__main__":
    unittest.main()
