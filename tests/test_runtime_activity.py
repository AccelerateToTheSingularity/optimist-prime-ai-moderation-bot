"""Tests for the local activity snapshot used by the native console."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from runtime_activity import (
    read_activity_snapshot,
    read_cycle_summary,
    record_cycle_activity,
    write_cycle_summary,
)


class TestRuntimeActivity(unittest.TestCase):
    def test_cycle_items_are_bounded_deduplicated_and_readable_without_reddit(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "activity.json"
            comment = SimpleNamespace(
                id="c1",
                body="hello",
                author=SimpleNamespace(name="example_user"),
                subreddit=SimpleNamespace(display_name="example_community"),
                created_utc=2_000_000_000,
                score=3,
                permalink="/r/example_community/comments/c1",
                link_title="A post",
            )
            first = record_cycle_activity(
                [], [comment], subreddit="example_community", run_id="run-1", path=path
            )
            second = record_cycle_activity(
                [], [comment], subreddit="example_community", run_id="run-2", path=path
            )

            self.assertEqual(len(first["items"]), 1)
            self.assertEqual(len(second["items"]), 1)
            self.assertEqual(read_activity_snapshot(path)["items"][0]["runId"], "run-2")

    def test_cycle_summary_is_count_only_and_local(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "cycle_summary.json"
            write_cycle_summary(
                {"subreddit": "example_community", "eligibleComments": 12}, path
            )
            summary = read_cycle_summary(path)
            self.assertEqual(summary["eligibleComments"], 12)


if __name__ == "__main__":
    unittest.main()
