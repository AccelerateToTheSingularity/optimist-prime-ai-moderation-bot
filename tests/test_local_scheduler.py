"""Regression checks for the public local worker cycle and scheduler."""

import unittest
import os
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import local_worker


ROOT = Path(__file__).resolve().parents[1]


class TestLocalScheduler(unittest.TestCase):
    def test_hidden_worker_captures_output_and_user_agent_identity_is_configurable(self):
        worker = (ROOT / "local_worker.py").read_text(encoding="utf-8")
        self.assertIn("redirect_stdout", worker)
        self.assertIn("redirect_stderr", worker)
        self.assertIn("scheduler.cycle.started", worker)
        self.assertIn("scheduler.cycle.completed", worker)
        self.assertIn("duration_ms", worker)
        self.assertIn("run_id", worker)
        self.assertIn("BOT_SCHEDULER_CONTEXT", worker)
        for filename in ("bot_runner.py", "manage_rules.py", "obtain_refresh_token.py"):
            self.assertIn("REDDIT_DEVELOPER_USERNAME", (ROOT / filename).read_text(encoding="utf-8"))

    def test_one_cycle_records_terminal_scheduler_evidence(self):
        stream = StringIO()
        calls = []

        def runner():
            calls.append("ran")
            return 0

        with (
            patch.object(local_worker, "configure_logging", return_value=Path("structured.jsonl")),
            patch.object(local_worker, "log_event") as log_event,
        ):
            result = local_worker._run_cycle(runner, stream, cycle_number=1)

        self.assertEqual(result, 0)
        self.assertEqual(calls, ["ran"])
        self.assertIn("local cycle 1 started", stream.getvalue())
        self.assertIn("outcome=success", stream.getvalue())
        events = [call.args[0] for call in log_event.call_args_list]
        self.assertIn("scheduler.cycle.started", events)
        self.assertIn("scheduler.cycle.completed", events)

    def test_worker_interval_is_bounded(self):
        self.assertEqual(local_worker._interval_seconds(), 60)
        self.assertEqual(local_worker._interval_seconds(5), local_worker.MIN_INTERVAL_SECONDS)
        self.assertEqual(local_worker._interval_seconds(180), 180)
        self.assertEqual(local_worker._interval_seconds(9999), local_worker.MAX_INTERVAL_SECONDS)

    def test_adaptive_minimum_is_two_seconds(self):
        self.assertEqual(local_worker.MIN_ADAPTIVE_INTERVAL_SECONDS, 2)

    def test_saved_interval_migrates_without_legacy_comment_rate(self):
        state = local_worker._scheduler_state_from_summary(
            {
                "scheduler": {
                    "arrivalRatePerMinute": 99,
                    "lastIntervalSeconds": 816,
                    "samples": 12,
                    "lastEligibleComments": 30,
                }
            },
            default_interval=180,
            minimum_seconds=15,
            maximum_seconds=900,
        )
        self.assertEqual(state.last_interval_seconds, 816)
        self.assertEqual(state.samples, 12)
        self.assertEqual(state.last_item_count, 0)

    def test_adaptive_sample_requires_fresh_usable_enabled_scans(self):
        summary = {
            "runId": "current",
            "adaptiveMetricsVersion": 1,
            "postsScanEnabled": True,
            "postsScanComplete": True,
            "postsScanUsable": True,
            "eligiblePosts": 2,
            "commentsScanEnabled": True,
            "commentsScanComplete": True,
            "commentsScanUsable": True,
            "eligibleComments": 1,
            "privateChatScanEnabled": True,
            "privateChatScanComplete": True,
            "privateChatScanUsable": True,
            "privateChatMessagesHandled": 1,
        }
        self.assertEqual(
            local_worker._adaptive_sample_from_summary(summary, expected_run_id="current"),
            {
                "eligiblePosts": 2,
                "eligibleComments": 1,
                "privateChatMessagesHandled": 1,
            },
        )
        continuing_backlog = dict(summary, postsScanComplete=False)
        self.assertEqual(
            local_worker._adaptive_sample_from_summary(
                continuing_backlog, expected_run_id="current"
            )["eligiblePosts"],
            2,
        )
        incomplete = dict(summary, postsScanUsable=False)
        self.assertIsNone(
            local_worker._adaptive_sample_from_summary(incomplete, expected_run_id="current")
        )
        self.assertIsNone(
            local_worker._adaptive_sample_from_summary(summary, expected_run_id="older")
        )
        no_sources = dict(
            summary,
            postsScanEnabled=False,
            commentsScanEnabled=False,
            privateChatScanEnabled=False,
        )
        self.assertIsNone(
            local_worker._adaptive_sample_from_summary(no_sources, expected_run_id="current")
        )

    def test_adaptive_worker_counts_posts_comments_and_private_chat_together(self):
        summary = {"scheduler": {"lastIntervalSeconds": 180}}

        def runner():
            summary.update(
                {
                    "runId": local_worker.current_run_id(),
                    "adaptiveMetricsVersion": 1,
                    "postsScanEnabled": True,
                    "postsScanComplete": True,
                    "postsScanUsable": True,
                    "eligiblePosts": 2,
                    "commentsScanEnabled": True,
                    "commentsScanComplete": True,
                    "commentsScanUsable": True,
                    "eligibleComments": 1,
                    "privateChatScanEnabled": True,
                    "privateChatScanComplete": True,
                    "privateChatScanUsable": True,
                    "privateChatMessagesHandled": 1,
                }
            )
            return 0

        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "BOT_ADAPTIVE_SCHEDULING_ENABLED": "true",
                "BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN": "3",
                "BOT_ADAPTIVE_TARGET_COMMENTS_PER_RUN": "30",
                "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS": "180",
                "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS": "15",
                "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS": "900",
            },
            clear=False,
        ), patch.object(local_worker, "clear_inherited_runtime_environment"), patch.object(
            local_worker, "load_local_env"
        ), patch.object(
            local_worker, "console_log_path", return_value=Path(temp_dir) / "console.log"
        ), patch.object(
            local_worker, "configure_logging", return_value=Path(temp_dir) / "structured.jsonl"
        ), patch.object(local_worker, "log_event"), patch.object(
            local_worker, "read_cycle_summary", side_effect=lambda: summary
        ), patch.object(local_worker, "write_cycle_summary") as write_summary, patch.object(
            local_worker, "get_fast_test_mode", return_value={"active": False}
        ), patch.object(
            local_worker, "get_moderator_activity_status", return_value={"active": False}
        ), patch.object(local_worker, "_wait_for_next_cycle", return_value="deadline") as wait_cycle:
            result = local_worker.run_worker(max_cycles=2, interval_seconds=180, runner=runner)

        self.assertEqual(result, 0)
        self.assertAlmostEqual(wait_cycle.call_args.args[0], 153.0, delta=2.0)
        stored = write_summary.call_args.args[0]["scheduler"]
        self.assertEqual(stored["lastItemCount"], 4)
        self.assertEqual(stored["lastPrivateChatMessages"], 1)

    def test_incomplete_scan_holds_the_saved_adaptive_interval(self):
        summary = {
            "scheduler": {"lastIntervalSeconds": 153, "samples": 5},
            "runId": "old-run",
            "adaptiveMetricsVersion": 1,
            "postsScanEnabled": True,
            "postsScanComplete": False,
            "postsScanUsable": False,
            "eligiblePosts": 10,
            "commentsScanEnabled": False,
            "commentsScanComplete": True,
            "commentsScanUsable": True,
            "eligibleComments": 0,
            "privateChatScanEnabled": False,
            "privateChatScanComplete": True,
            "privateChatScanUsable": True,
            "privateChatMessagesHandled": 0,
        }

        def runner():
            summary["runId"] = local_worker.current_run_id()
            return 0

        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "BOT_ADAPTIVE_SCHEDULING_ENABLED": "true",
                "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS": "180",
                "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS": "15",
                "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS": "900",
            },
            clear=False,
        ), patch.object(local_worker, "clear_inherited_runtime_environment"), patch.object(
            local_worker, "load_local_env"
        ), patch.object(
            local_worker, "console_log_path", return_value=Path(temp_dir) / "console.log"
        ), patch.object(
            local_worker, "configure_logging", return_value=Path(temp_dir) / "structured.jsonl"
        ), patch.object(local_worker, "log_event"), patch.object(
            local_worker, "read_cycle_summary", return_value=summary
        ), patch.object(local_worker, "write_cycle_summary") as write_summary, patch.object(
            local_worker, "get_fast_test_mode", return_value={"active": False}
        ), patch.object(
            local_worker, "get_moderator_activity_status", return_value={"active": False}
        ), patch.object(local_worker, "_wait_for_next_cycle", return_value="deadline") as wait_cycle:
            result = local_worker.run_worker(max_cycles=2, interval_seconds=180, runner=runner)

        self.assertEqual(result, 0)
        self.assertAlmostEqual(wait_cycle.call_args.args[0], 153.0, delta=2.0)
        held_state = write_summary.call_args.args[0]["scheduler"]
        self.assertEqual(held_state["lastIntervalSeconds"], 153)
        self.assertEqual(held_state["samples"], 5)

    def test_stopping_fast_test_resumes_saved_adaptive_interval(self):
        summary = {"scheduler": {"lastIntervalSeconds": 153, "samples": 4}}
        statuses = iter([{"active": True}])
        waits = []
        runs = 0

        def runner():
            nonlocal runs
            runs += 1
            if runs == 1:
                summary.update(
                    {
                        "runId": local_worker.current_run_id(),
                        "adaptiveMetricsVersion": 1,
                        "postsScanEnabled": True,
                        "postsScanUsable": True,
                        "eligiblePosts": 1,
                        "commentsScanEnabled": True,
                        "commentsScanUsable": True,
                        "eligibleComments": 1,
                        "privateChatScanEnabled": True,
                        "privateChatScanUsable": True,
                        "privateChatMessagesHandled": 1,
                    }
                )
                return 0
            raise KeyboardInterrupt

        def wait_for_cycle(delay, *_args, **_kwargs):
            waits.append(delay)
            return "deactivated" if len(waits) == 1 else "deadline"

        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "BOT_ADAPTIVE_SCHEDULING_ENABLED": "true",
                "BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN": "3",
                "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS": "180",
                "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS": "15",
                "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS": "900",
            },
            clear=False,
        ), patch.object(local_worker, "clear_inherited_runtime_environment"), patch.object(
            local_worker, "load_local_env"
        ), patch.object(
            local_worker, "console_log_path", return_value=Path(temp_dir) / "console.log"
        ), patch.object(
            local_worker, "configure_logging", return_value=Path(temp_dir) / "structured.jsonl"
        ), patch.object(local_worker, "log_event"), patch.object(
            local_worker, "read_cycle_summary", side_effect=lambda: summary
        ), patch.object(local_worker, "write_cycle_summary"), patch.object(
            local_worker, "get_fast_test_mode", side_effect=lambda: next(statuses)
        ), patch.object(
            local_worker, "get_moderator_activity_status", return_value={"active": False}
        ), patch.object(
            local_worker, "_wait_for_next_cycle", side_effect=wait_for_cycle
        ):
            with self.assertRaises(KeyboardInterrupt):
                local_worker.run_worker(interval_seconds=180, runner=runner)

        self.assertEqual(len(waits), 2)
        self.assertEqual(waits[0], local_worker.FAST_INTERVAL_SECONDS)
        self.assertAlmostEqual(waits[1], 153.0, delta=1.0)

    def test_moderator_conversation_uses_three_second_wait_and_preserves_adaptive_interval(self):
        summary = {"scheduler": {"lastIntervalSeconds": 153, "samples": 4}}
        active = {
            "active": True,
            "source": "moderator_activity",
            "remaining_seconds": 120,
            "expires_at": 9999999999,
        }
        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "BOT_ADAPTIVE_SCHEDULING_ENABLED": "true",
                "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS": "180",
                "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS": "15",
                "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS": "900",
            },
            clear=False,
        ), patch.object(local_worker, "clear_inherited_runtime_environment"), patch.object(
            local_worker, "load_local_env"
        ), patch.object(
            local_worker, "console_log_path", return_value=Path(temp_dir) / "console.log"
        ), patch.object(
            local_worker, "configure_logging", return_value=Path(temp_dir) / "structured.jsonl"
        ), patch.object(local_worker, "log_event") as log_event, patch.object(
            local_worker, "read_cycle_summary", return_value=summary
        ), patch.object(local_worker, "write_cycle_summary") as write_summary, patch.object(
            local_worker, "get_fast_test_mode", return_value={"active": False}
        ), patch.object(
            local_worker, "get_moderator_activity_status", return_value=active
        ), patch.object(local_worker, "_wait_for_next_cycle", return_value="deadline") as wait_cycle:
            result = local_worker.run_worker(
                max_cycles=2,
                interval_seconds=180,
                runner=lambda: 0,
            )

        self.assertEqual(result, 0)
        self.assertEqual(wait_cycle.call_args.args[0], 3)
        write_summary.assert_not_called()
        self.assertEqual(summary["scheduler"]["lastIntervalSeconds"], 153)
        self.assertIn(
            "scheduler.moderator_activity.fast_cycles",
            [call.args[0] for call in log_event.call_args_list],
        )

    def test_adaptive_worker_holds_current_interval_after_failure_or_stale_summary(self):
        outcomes = iter([1, 0])
        prior_summary = {
            "scheduler": {"lastIntervalSeconds": 153, "samples": 4},
            "runId": "old-run",
            "adaptiveMetricsVersion": 1,
            "postsScanEnabled": False,
            "postsScanComplete": True,
            "postsScanUsable": True,
            "eligiblePosts": 0,
            "commentsScanEnabled": True,
            "commentsScanComplete": True,
            "commentsScanUsable": True,
            "eligibleComments": 45,
            "privateChatScanEnabled": False,
            "privateChatScanComplete": True,
            "privateChatScanUsable": True,
            "privateChatMessagesHandled": 0,
        }
        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "BOT_ADAPTIVE_SCHEDULING_ENABLED": "true",
                "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS": "180",
                "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS": "15",
                "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS": "900",
            },
            clear=False,
        ), patch.object(local_worker, "clear_inherited_runtime_environment"), patch.object(
            local_worker, "load_local_env"
        ), patch.object(
            local_worker, "console_log_path", return_value=Path(temp_dir) / "console.log"
        ), patch.object(
            local_worker, "configure_logging", return_value=Path(temp_dir) / "structured.jsonl"
        ), patch.object(local_worker, "log_event"), patch.object(
            local_worker,
            "read_cycle_summary",
            return_value=prior_summary,
        ), patch.object(local_worker, "write_cycle_summary") as write_summary, patch.object(
            local_worker, "get_fast_test_mode", return_value={"active": False}
        ), patch.object(
            local_worker, "get_moderator_activity_status", return_value={"active": False}
        ), patch.object(local_worker, "_wait_for_next_cycle", return_value="deadline") as wait_cycle:
            result = local_worker.run_worker(
                max_cycles=2,
                interval_seconds=180,
                runner=lambda: next(outcomes),
            )

        self.assertEqual(result, 0)
        # Failed cycles and stale summaries hold the last usable cadence.
        self.assertAlmostEqual(wait_cycle.call_args.args[0], 153.0, delta=2.0)
        self.assertEqual(
            write_summary.call_args.args[0]["scheduler"]["lastIntervalSeconds"],
            153,
        )

    def test_worker_clears_inherited_runtime_settings_before_loading_owner_env(self):
        with patch.dict(
            os.environ,
            {
                "BOT_SUBREDDIT": "wrong-subreddit",
                "OPENAI_API_KEY": "inherited-key",
                "REDDIT_REFRESH_TOKEN": "inherited-token",
                "UNRELATED_SETTING": "preserved",
            },
            clear=True,
        ):
            local_worker.clear_inherited_runtime_environment()
            self.assertNotIn("BOT_SUBREDDIT", os.environ)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("REDDIT_REFRESH_TOKEN", os.environ)
            self.assertEqual(os.environ["UNRELATED_SETTING"], "preserved")

    def test_owner_template_uses_placeholders_and_private_runtime_paths(self):
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("REDDIT_CLIENT_SECRET=replace_with_your_client_secret", env_example)
        self.assertIn("BOT_LLM_PROVIDER=", env_example)
        self.assertIn("LLM_MODEL=", env_example)
        self.assertIn("LLM_API_KEY=", env_example)
        self.assertIn("OPENROUTER_API_KEY=", env_example)
        self.assertIn("BOT_SAFE_MODE=true", env_example)
        self.assertIn("BOT_RUNTIME_STATE_FILE=data/bot_state.json", env_example)
        self.assertIn("BOT_RUNTIME_AUDIT_LOG_FILE=data/audit_log.json", env_example)
        self.assertNotIn("C:\\Users\\", env_example)


if __name__ == "__main__":
    unittest.main()
