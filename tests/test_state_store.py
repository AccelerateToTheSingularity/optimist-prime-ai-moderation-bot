"""Tests for private runtime-state storage."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet

from state_store import (
    StateStoreError,
    load_state,
    migrate_plaintext_state,
    migrate_plaintext_state_to_local,
    save_state,
)


class TestStateStore(unittest.TestCase):
    def test_encrypted_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot_state.enc"
            with patch.dict(os.environ, {"BOT_STATE_ENCRYPTION_KEY": Fernet.generate_key().decode()}, clear=False):
                save_state({"username": "private", "count": 2}, path)
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn("private", raw)
                self.assertEqual(load_state(path), {"username": "private", "count": 2})

    def test_live_actions_refuse_plaintext_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true", "BOT_STATE_ENCRYPTION_KEY": ""},
            clear=False,
        ):
            with self.assertRaises(StateStoreError):
                load_state(Path(tmp) / "bot_state.enc", {})

    def test_dry_run_save_is_non_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot_state.json"
            save_state({"count": 1}, path, dry_run=True)
            self.assertFalse(path.exists())

    def test_existing_corrupt_plaintext_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot_state.json"
            path.write_text('{"replied_to_comments": [', encoding="utf-8")
            with self.assertRaises(StateStoreError):
                load_state(path, {"fresh": True})

    def test_existing_non_object_plaintext_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot_state.json"
            path.write_text('["old response IDs"]', encoding="utf-8")
            with self.assertRaises(StateStoreError):
                load_state(path, {"fresh": True})

    def test_dry_run_can_preview_without_encryption_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "BOT_STATE_BACKEND": "encrypted",
                "BOT_STATE_ENCRYPTION_KEY": "",
            },
            clear=False,
        ):
            path = Path(tmp) / "bot_state.enc"
            self.assertEqual(
                load_state(path, {"preview": True}, dry_run=True),
                {"preview": True},
            )

    def test_plaintext_migration_preserves_unknown_fields_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"BOT_STATE_ENCRYPTION_KEY": Fernet.generate_key().decode()},
            clear=False,
        ):
            source = Path(tmp) / "bot_state.json"
            target = Path(tmp) / "bot_state.enc"
            source.write_text(
                '{"future_field":{"keep":true},"action_keys":["post_tldr:c1"]}',
                encoding="utf-8",
            )
            migrated = migrate_plaintext_state(source, target)
            self.assertEqual(migrated["future_field"], {"keep": True})
            self.assertEqual(load_state(target), migrated)
            with self.assertRaises(StateStoreError):
                migrate_plaintext_state(source, target)

    def test_local_runtime_path_can_live_outside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "private" / "bot_state.json"
            with patch.dict(os.environ, {"BOT_RUNTIME_STATE_FILE": str(target)}, clear=False):
                save_state({"private": True})
                self.assertEqual(load_state(), {"private": True})

    def test_local_state_migration_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bot_state.json"
            target = Path(tmp) / "private" / "bot_state.json"
            source_contents = '{"future_field":{"keep":true}}'
            source.write_text(source_contents, encoding="utf-8")
            migrated = migrate_plaintext_state_to_local(source, target)
            self.assertEqual(migrated, {"future_field": {"keep": True}})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), migrated)
            self.assertEqual(source.read_text(encoding="utf-8"), source_contents)
            with self.assertRaises(StateStoreError):
                migrate_plaintext_state_to_local(source, target)


if __name__ == "__main__":
    unittest.main()
