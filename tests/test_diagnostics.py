"""Regression checks for structural versus operational diagnostics."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from diagnostics import dump_diagnostics


class TestDiagnostics(unittest.TestCase):
    def test_safe_mode_diagnostic_matches_the_fresh_install_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BOT_SAFE_MODE", None)
            payload = dump_diagnostics()

        self.assertEqual(payload["mode"]["safeMode"], "true")

    def test_moderator_fast_window_is_reported_without_identity_data(self):
        with patch(
            "diagnostics.get_moderator_activity_status",
            return_value={
                "active": True,
                "remaining_seconds": 120,
                "expires_at": 1234567890,
                "last_interaction_at": 1234567710,
            },
        ):
            payload = dump_diagnostics()

        status = payload["background"]["moderatorActivity"]
        self.assertTrue(status["active"])
        self.assertEqual(status["remaining_seconds"], 120)
        self.assertEqual(status["boostIntervalSeconds"], 3)
        self.assertEqual(status["quietPeriodSeconds"], 180)
        self.assertNotIn("username", json.dumps(status).casefold())

    def test_adaptive_scheduler_diagnostics_describe_combined_target(self):
        with patch.dict(
            os.environ,
            {
                "BOT_ADAPTIVE_SCHEDULING_ENABLED": "true",
                "BOT_ADAPTIVE_TARGET_ITEMS_PER_RUN": "3",
                "BOT_ADAPTIVE_MIN_INTERVAL_SECONDS": "2",
                "BOT_ADAPTIVE_NORMAL_INTERVAL_SECONDS": "60",
                "BOT_ADAPTIVE_MAX_INTERVAL_SECONDS": "60",
            },
            clear=False,
        ):
            payload = dump_diagnostics()

        adaptive = payload["background"]["adaptiveScheduling"]
        self.assertTrue(adaptive["enabled"])
        self.assertEqual(adaptive["targetItemsPerRun"], "3")
        self.assertEqual(adaptive["adjustmentPercent"], 15)
        self.assertIn("eligiblePosts", adaptive["countedSources"])
        self.assertIn("eligibleComments", adaptive["countedSources"])
        self.assertIn("privateChatMessagesHandled", adaptive["countedSources"])
        self.assertEqual(adaptive["minimumIntervalSeconds"], "2")
        self.assertEqual(adaptive["normalIntervalSeconds"], "60")
        self.assertEqual(adaptive["maximumIntervalSeconds"], "60")

    def test_history_review_ledger_metrics_are_exposed_without_user_data(self):
        metrics = {
            "ledger_available": True,
            "reviews_completed": 8,
            "skipped_duplicate_items": 41,
            "history_gaps": 1,
        }
        with patch("history_review_ledger.diagnostic_ledger_metrics", return_value=metrics):
            payload = dump_diagnostics()

        status = payload["background"]["historyReviewLedger"]
        self.assertTrue(status["available"])
        self.assertEqual(status["reviewsCompleted"], 8)
        self.assertEqual(status["skippedDuplicateItems"], 41)
        self.assertEqual(status["historyGaps"], 1)
        self.assertNotIn("username", json.dumps(status).casefold())
        self.assertNotIn("candidate", json.dumps(status).casefold())

    def test_scheduler_and_reddit_transport_evidence_is_summarized_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_dir.mkdir()
            timestamp = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
            records = [
                {
                    "timestamp": timestamp,
                    "level": "INFO",
                    "logger": "optimist_prime",
                    "message": "Local scheduled bot cycle started",
                    "eventName": "scheduler.cycle.started",
                    "correlationId": "cycle-1",
                    "component": "scheduler",
                },
                {
                    "timestamp": timestamp,
                    "level": "INFO",
                    "logger": "optimist_prime",
                    "message": "Reddit request attempt started",
                    "eventName": "reddit.request.started",
                    "correlationId": "cycle-1",
                    "requestId": "request-1",
                    "attempt": 1,
                    "requestMethod": "GET",
                    "requestPath": "/api/v1/me",
                    "timeoutSeconds": 11,
                },
                {
                    "timestamp": timestamp,
                    "level": "WARNING",
                    "logger": "optimist_prime",
                    "message": "Reddit request attempt failed",
                    "eventName": "reddit.request.retryable_failure",
                    "correlationId": "cycle-1",
                    "requestId": "request-1",
                    "attempt": 1,
                    "requestMethod": "GET",
                    "requestPath": "/api/v1/me",
                    "timeoutSeconds": 11,
                    "durationMs": 11000.0,
                    "retryable": True,
                    "errorType": "ReadTimeout",
                },
                {
                    "timestamp": timestamp,
                    "level": "WARNING",
                    "logger": "prawcore",
                    "message": "Reddit request retry scheduled",
                    "eventName": "reddit.retry",
                    "correlationId": "cycle-1",
                    "requestId": "request-1",
                    "attempt": 1,
                    "requestMethod": "GET",
                    "requestPath": "/api/v1/me",
                    "retryReason": "ReadTimeout",
                    "retryScheduled": True,
                    "retryable": True,
                },
            ]
            (log_dir / "optimist_prime.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "BOT_LOG_DIR": str(log_dir),
                    "BOT_RUNTIME_REQUEST_BUDGET_FILE": str(
                        Path(tmp) / "request-budget.sqlite3"
                    ),
                    "REDDIT_API_TIMEOUT_SECONDS": "11",
                },
                clear=False,
            ):
                payload = dump_diagnostics()

        scheduler = payload["background"]["scheduler"]
        reddit = payload["background"]["redditRequests"]
        self.assertTrue(scheduler["cycleInProgress"])
        self.assertEqual(scheduler["currentCycleRunId"], "cycle-1")
        self.assertEqual(scheduler["currentCycleRetryCount"], 1)
        self.assertEqual(scheduler["currentCycleTimeoutCount"], 1)
        self.assertEqual(reddit["configuredTimeoutSeconds"], 11)
        self.assertEqual(reddit["retryCount"], 1)
        self.assertEqual(reddit["timeoutCount"], 1)
        self.assertEqual(reddit["lastTimeout"]["path"], "/api/v1/me")
        self.assertEqual(reddit["lastTimeout"]["timeoutSeconds"], 11)
        budget = reddit["requestBudget"]
        self.assertEqual(budget["officialRequestsPerMinute"], 100)
        self.assertEqual(budget["safetyRequestsPerMinute"], 90)
        self.assertEqual(budget["windowSeconds"], 600)
        self.assertEqual(budget["maxRequestsPerWindow"], 900)
        self.assertEqual(budget["usedRequests"], 0)
        self.assertEqual(budget["remainingRequests"], 900)

    def test_public_stats_last_run_is_reported_as_completed_run(self):
        stats = {"last_run": "2026-10-15T00:00:00+00:00"}
        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            with patch.dict(os.environ, {"BOT_RUNTIME_STATS_FILE": str(stats_path)}):
                payload = dump_diagnostics()
        self.assertEqual(payload["background"]["lastCompletedRun"], stats.get("last_run"))
        self.assertEqual(payload["background"]["lastRunProven"], bool(stats.get("last_run")))
        self.assertIn(
            payload["background"]["operationalStatus"],
            {"recent completed run observed", "stale completed run observed", "no completed run observed"},
        )
        detail = payload["background"]["operationalStatusDetail"]
        self.assertEqual(
            set(detail),
            {
                "summary",
                "requested",
                "observed",
                "confirmation",
                "blocker",
                "nextAction",
                "nextActionAt",
                "userActionRequired",
                "severity",
                "incidentId",
            },
        )
        self.assertIn("reddit", detail["requested"].lower())
        self.assertTrue(detail["summary"])

    def test_public_response_diagnostics_count_uncertain_holds_without_exposing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            now = datetime.now(timezone.utc).timestamp()
            state_path.write_text(json.dumps({
                "public_response_ledger": {
                    "comment:private-id-1": {
                        "status": "uncertain", "created_at": now,
                        "thread_id": "private-thread-1", "tier": "ordinary",
                    },
                    "comment:private-id-2": {
                        "status": "pending", "created_at": now,
                        "thread_id": "private-thread-1", "tier": "moderator",
                    },
                    "comment:private-id-3": {
                        "status": "completed", "created_at": now - 7200,
                        "thread_id": "private-thread-2", "tier": "ordinary",
                    },
                }
            }), encoding="utf-8")
            with patch.dict(os.environ, {
                "BOT_RUNTIME_STATE_FILE": str(state_path),
                "BOT_STATE_BACKEND": "plaintext",
                "BOT_STATE_ENCRYPTION_KEY": "",
                "GITHUB_ACTIONS": "",
            }, clear=False):
                payload = dump_diagnostics()

        summary = payload["background"]["publicResponseSafety"]
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["uncertain"], 1)
        self.assertEqual(summary["countedLastHour"], 2)
        self.assertEqual(summary["ordinaryLastHour"], 1)
        self.assertEqual(summary["moderatorLastHour"], 1)
        self.assertNotIn("private-id", json.dumps(summary))


    def test_live_encrypted_state_requirement_is_visible_when_key_is_missing(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "BOT_STATE_BACKEND": "encrypted",
                "BOT_STATE_ENCRYPTION_KEY": "",
            },
            clear=False,
        ):
            payload = dump_diagnostics()
        self.assertEqual(payload["background"]["stateBackend"], "encrypted")
        self.assertFalse(payload["background"]["runtimeStateReadable"])
        self.assertEqual(
            payload["background"]["operationalStatusDetail"]["incidentId"],
            "runtime-state-unreadable",
        )

    def test_owner_runtime_paths_are_reported_as_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            stats_path = Path(tmp) / "stats.json"
            with patch.dict(
                os.environ,
                {
                    "BOT_RUNTIME_STATE_FILE": str(state_path),
                    "BOT_RUNTIME_STATS_FILE": str(stats_path),
                    "BOT_STATE_BACKEND": "",
                    "GITHUB_ACTIONS": "",
                },
                clear=False,
            ):
                payload = dump_diagnostics()
        self.assertTrue(payload["paths"]["stateFileOutsideRepository"])
        self.assertTrue(payload["paths"]["statsFileOutsideRepository"])
        self.assertTrue(payload["background"]["runtimeStateOutsideRepository"])

    def test_saved_flair_permission_pause_is_visible_in_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            state_path.write_text(
                json.dumps({
                    "reddit_flair_permission_pause": {
                        "blocked": True,
                        "reason": "Reddit denied required flair access (403 InsufficientScope)",
                        "since": "2026-09-30T00:00:00+00:00",
                    }
                }),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "BOT_RUNTIME_STATE_FILE": str(state_path),
                    "BOT_STATE_BACKEND": "",
                    "GITHUB_ACTIONS": "",
                },
                clear=False,
            ):
                payload = dump_diagnostics()
        self.assertTrue(payload["background"]["flairWrites"]["paused"])
        self.assertEqual(
            payload["background"]["flairWrites"]["reason"],
            "Reddit denied required flair access (403 InsufficientScope)",
        )

    def test_owner_audit_log_path_is_reported_and_can_stay_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            audit_path = Path(tmp) / "runtime" / "audit_log.json"
            env_path.write_text(
                "BOT_RUNTIME_AUDIT_LOG_FILE=" + str(audit_path) + "\n"
                "BOT_RUNTIME_STATE_FILE=" + str(Path(tmp) / "runtime" / "state.json") + "\n"
                "BOT_RUNTIME_STATS_FILE=" + str(Path(tmp) / "runtime" / "stats.json") + "\n"
                "BOT_LOG_DIR=" + str(Path(tmp) / "runtime" / "logs") + "\n"
                "OPENAI_API_KEY=\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                payload = dump_diagnostics(
                    load_environment=True,
                    environment_path=env_path,
                )
        self.assertEqual(payload["paths"]["auditLogFile"], str(audit_path))
        self.assertTrue(payload["paths"]["auditLogOutsideRepository"])

    def test_owner_diagnostics_does_not_treat_inherited_llm_key_as_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "BOT_LLM_PROVIDER=minimax\n"
                "OPENAI_API_KEY=\n"
                "REDDIT_CLIENT_ID=\n"
                "REDDIT_CLIENT_SECRET=\n"
                "REDDIT_REFRESH_TOKEN=\n"
                "BOT_EXPECTED_USERNAME=\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "inherited-only"}, clear=True):
                payload = dump_diagnostics(
                    load_environment=True,
                    environment_path=env_path,
                )
        self.assertFalse(payload["integrations"]["llmKeyConfigured"])
        self.assertEqual(payload["integrations"]["llmCredentialSource"], "missing")
        self.assertFalse(payload["integrations"]["localConfigurationReady"])

    def test_owner_diagnostics_uses_local_generic_provider_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "REDDIT_CLIENT_ID=local-id\n"
                "REDDIT_CLIENT_SECRET=local-secret\n"
                "REDDIT_REFRESH_TOKEN=local-refresh\n"
                "BOT_LLM_PROVIDER=openai\n"
                "OPENAI_API_KEY=local-provider-key\n"
                "LLM_MODEL=example-model\n"
                "LLM_BASE_URL=https://api.example.test/v1\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "BOT_LLM_PROVIDER": "claude",
                    "ANTHROPIC_API_KEY": "inherited-key",
                    "OPENAI_API_KEY": "inherited-key",
                    "REDDIT_CLIENT_ID": "inherited-id",
                },
                clear=True,
            ):
                payload = dump_diagnostics(
                    load_environment=True,
                    environment_path=env_path,
                )
        self.assertEqual(payload["integrations"]["activeLlmProvider"], "openai")
        self.assertEqual(payload["integrations"]["activeLlmModel"], "example-model")
        self.assertTrue(payload["integrations"]["llmConfigurationReady"])
        self.assertEqual(payload["integrations"]["llmCredentialSource"], "local .env")
        self.assertTrue(payload["integrations"]["localRedditConfigurationReady"])
        self.assertTrue(payload["integrations"]["localConfigurationReady"])

    def test_owner_diagnostics_restores_the_calling_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("BOT_SUBREDDIT=example_community\n", encoding="utf-8")
            with patch.dict(os.environ, {"BOT_SUBREDDIT": "caller_community"}, clear=True):
                dump_diagnostics(load_environment=True, environment_path=env_path)
                self.assertEqual(os.environ.get("BOT_SUBREDDIT"), "caller_community")

    def test_reddit_readiness_requires_client_pair_and_reports_credential_source(self):
        with patch.dict(
            os.environ,
            {
                "REDDIT_CLIENT_ID": "configured-id",
                "REDDIT_CLIENT_SECRET": "configured-secret",
                "REDDIT_REFRESH_TOKEN": "configured-refresh-token",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        ):
            payload = dump_diagnostics()
        self.assertTrue(payload["integrations"]["redditClientConfigured"])
        self.assertTrue(payload["integrations"]["redditRefreshTokenConfigured"])
        self.assertIn(
            payload["integrations"]["redditCredentialSource"],
            {"local .env", "process environment"},
        )


if __name__ == "__main__":
    unittest.main()
