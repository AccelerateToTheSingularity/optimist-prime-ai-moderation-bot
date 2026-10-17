"""Tests for durable, non-destructive JSON state writes."""

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import file_lock
from file_lock import safe_json_load, safe_json_save


class TestSafeJsonSave(unittest.TestCase):
    def test_serialization_failure_preserves_existing_state_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            original = '{"active": true, "count": 3}'
            path.write_text(original, encoding="utf-8")
            cyclic = {}
            cyclic["self"] = cyclic

            with self.assertLogs("optimist_prime", level="ERROR") as captured:
                self.assertFalse(safe_json_save(str(path), cyclic))

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(
                safe_json_load(str(path), None),
                {"active": True, "count": 3},
            )
            self.assertEqual(
                captured.records[0].operation_stage,
                "serialize",
            )
            self.assertEqual(captured.records[0].error_type, "ValueError")

    def test_successful_save_replaces_state_atomically_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"old": true}', encoding="utf-8")

            self.assertTrue(safe_json_save(str(path), {"new": [1, 2, 3]}))

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": [1, 2, 3]})
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()), [
                "state.json",
            ])

    def test_transient_permission_error_retries_and_saves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"old": true}', encoding="utf-8")
            real_replace = file_lock.os.replace
            attempts = 0

            def replace_with_transient_lock(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("temporary sharing violation")
                return real_replace(source, destination)

            with mock.patch("file_lock.sys.platform", "win32"), mock.patch(
                "file_lock.os.replace", side_effect=replace_with_transient_lock
            ), self.assertLogs("optimist_prime", level="WARNING") as captured:
                self.assertTrue(safe_json_save(str(path), {"new": True}))

            self.assertEqual(attempts, 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": True})
            self.assertTrue(captured.records[0].retrying)
            self.assertEqual(captured.records[0].operation_stage, "write_or_replace")
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()), [
                "state.json",
            ])

    def test_persistent_permission_error_preserves_existing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            original = '{"old": true}'
            path.write_text(original, encoding="utf-8")

            with mock.patch("file_lock.sys.platform", "win32"), mock.patch(
                "file_lock.os.replace",
                side_effect=PermissionError("persistent sharing violation"),
            ) as replace, self.assertLogs("optimist_prime", level="WARNING") as captured:
                self.assertFalse(safe_json_save(str(path), {"new": True}))

            self.assertEqual(replace.call_count, 5)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(captured.records[-1].levelname, "ERROR")
            self.assertFalse(captured.records[-1].retrying)
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()), [
                "state.json",
            ])


if __name__ == "__main__":
    unittest.main()
