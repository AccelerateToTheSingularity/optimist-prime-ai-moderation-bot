"""Tests for the cross-process single-cycle lock."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import os
from pathlib import Path

from runtime_cycle_lock import RuntimeCycleLock, runtime_cycle_lock_path


class RuntimeCycleLockTests(unittest.TestCase):
    def test_lock_is_exclusive_and_reusable_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.cycle.lock"
            first = RuntimeCycleLock(path)
            second = RuntimeCycleLock(path)

            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.release()
            self.assertTrue(second.acquire())
            second.release()
            self.assertTrue(path.exists())

    def test_another_process_cannot_acquire_held_runtime_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.cycle.lock"
            lock = RuntimeCycleLock(path)
            self.assertTrue(lock.acquire())
            try:
                code = (
                    "from runtime_cycle_lock import RuntimeCycleLock; "
                    f"lock=RuntimeCycleLock({str(path)!r}); "
                    "print('acquired' if lock.acquire() else 'busy')"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "busy")
            finally:
                lock.release()

    def test_relative_and_absolute_state_paths_share_the_same_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "runtime.json"
            relative = os.path.relpath(state_path, Path.cwd())
            first = RuntimeCycleLock(runtime_cycle_lock_path(relative))
            second = RuntimeCycleLock(runtime_cycle_lock_path(state_path))

            self.assertEqual(first.path, second.path)
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.release()


if __name__ == "__main__":
    unittest.main()
