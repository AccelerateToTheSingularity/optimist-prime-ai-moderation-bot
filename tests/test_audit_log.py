"""Tests for private, dynamically resolved action audit-log storage."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_log import audit_log_path, log_audit_event


class TestAuditLogPath(unittest.TestCase):
    def test_runtime_path_override_is_used_after_module_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private" / "audit.json"
            with patch.dict(os.environ, {"BOT_RUNTIME_AUDIT_LOG_FILE": str(path)}, clear=False):
                self.assertEqual(audit_log_path(), path)
                log_audit_event(
                    "test",
                    "item-1",
                    "test-user",
                    "fixture",
                    "logged",
                    True,
                )
            self.assertTrue(path.exists())
            self.assertFalse((Path("data") / "audit_log.json").resolve() == path.resolve())


if __name__ == "__main__":
    unittest.main()
