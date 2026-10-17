"""Tests for manage_rules CLI."""

import subprocess
import sys
import unittest
from pathlib import Path


class TestManageRules(unittest.TestCase):
    def test_validate_example_rules(self):
        root = Path(__file__).resolve().parents[1]
        example = root / "data" / "rules.example.json"
        result = subprocess.run(
            [sys.executable, str(root / "manage_rules.py"), "--file", str(example), "validate"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
