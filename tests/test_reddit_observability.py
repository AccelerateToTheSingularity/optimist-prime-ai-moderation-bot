"""Regression tests for safe Reddit request and scheduler observability."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from requests.exceptions import ReadTimeout

import bot_logging
from bot_logging import OperatorLogFormatter
from prawcore.exceptions import RequestException
from reddit_observability import (
    ObservableRequestor,
    observable_requestor_settings,
    single_write_attempt,
)


def _close_root_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


class TestRedditObservability(unittest.TestCase):
    def tearDown(self) -> None:
        _close_root_handlers()
        bot_logging.clear_log_context()

    def test_settings_route_timeout_to_the_actual_requestor(self):
        settings = observable_requestor_settings(9)
        self.assertIs(settings["requestor_class"], ObservableRequestor)
        self.assertEqual(settings["requestor_kwargs"], {"timeout": 9})
        self.assertEqual(settings["timeout"], 9)

    def test_multiple_requestors_share_the_same_budget_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget_path = os.path.join(tmp, "reddit-budget.sqlite3")
            settings = {
                "request_budget_path": budget_path,
                "request_budget_limit": 4,
                "request_budget_window_seconds": 60,
                "request_budget_per_minute": 4,
            }
            first = ObservableRequestor(
                "script:optimist-prime:test", timeout=7, **settings
            )
            second = ObservableRequestor(
                "script:optimist-prime:test", timeout=7, **settings
            )
            first._http.request = Mock(return_value=SimpleNamespace(status_code=200))
            second._http.request = Mock(return_value=SimpleNamespace(status_code=200))

            first.request("GET", "https://oauth.reddit.com/api/v1/me")
            second.request("GET", "https://oauth.reddit.com/api/v1/me")

            snapshot = first._request_budget.snapshot()
            self.assertEqual(snapshot.used_requests, 2)
            self.assertEqual(snapshot.remaining_requests, 2)

    def test_retryable_attempts_share_id_and_show_effective_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("BOT_LOG_DIR")
            os.environ["BOT_LOG_DIR"] = tmp
            try:
                bot_logging.ensure_run_id("test-cycle")
                bot_logging.set_log_phase("reddit.auth")
                path = bot_logging.configure_logging(also_stderr=False)

                response = SimpleNamespace(status_code=200)
                requestor = ObservableRequestor(
                    "script:optimist-prime:test", timeout=7
                )
                requestor._http.request = Mock(
                    side_effect=[ReadTimeout("synthetic timeout"), response]
                )

                with self.assertRaises(RequestException):
                    requestor.request(
                        "GET",
                        "https://oauth.reddit.com/api/v1/me?private=value",
                    )
                # In a real PRAW call, its retry logger sets this marker
                # between the failed attempt and the recursive retry.
                bot_logging.mark_retry_pending()
                self.assertIs(
                    requestor.request(
                        "GET",
                        "https://oauth.reddit.com/api/v1/me?private=value",
                    ),
                    response,
                )
                self.assertEqual(
                    [call.kwargs["timeout"] for call in requestor._http.request.call_args_list],
                    [7, 7],
                )

                records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                started = [
                    record
                    for record in records
                    if record.get("eventName") == "reddit.request.started"
                ]
                self.assertEqual([record["attempt"] for record in started], [1, 2])
                self.assertEqual(started[0]["requestId"], started[1]["requestId"])
                self.assertEqual(started[0]["timeoutSeconds"], 7)
                self.assertEqual(started[0]["requestPath"], "/api/v1/me")

                failure = next(
                    record
                    for record in records
                    if record.get("eventName") == "reddit.request.retryable_failure"
                )
                self.assertEqual(failure["attempt"], 1)
                self.assertEqual(failure["errorType"], "ReadTimeout")
                self.assertTrue(failure["retryable"])
                self.assertIn("durationMs", failure)

                completed = next(
                    record
                    for record in records
                    if record.get("eventName") == "reddit.request.completed"
                )
                self.assertEqual(completed["attempt"], 2)
                self.assertEqual(completed["timeoutSeconds"], 7)
                self.assertEqual(completed["statusCode"], 200)
            finally:
                _close_root_handlers()
                bot_logging.clear_log_context()
                if previous is None:
                    os.environ.pop("BOT_LOG_DIR", None)
                else:
                    os.environ["BOT_LOG_DIR"] = previous

    def test_praw_retry_log_is_normalized_without_query_or_exception_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("BOT_LOG_DIR")
            os.environ["BOT_LOG_DIR"] = tmp
            try:
                bot_logging.ensure_run_id("retry-cycle")
                bot_logging.set_log_phase("content.post_scan")
                path = bot_logging.configure_logging(also_stderr=False)
                bot_logging.set_request_observation("request-1", 1, 13)

                logging.getLogger("prawcore").warning(
                    "Retrying due to ReadTimeout(ReadTimeoutError('hidden')) status: "
                    "GET https://oauth.reddit.com/api/v1/me?client_secret=hidden"
                )

                records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                retry = next(
                    record
                    for record in records
                    if record.get("eventName") == "reddit.retry"
                )
                self.assertEqual(retry["message"], "Reddit request retry scheduled")
                self.assertEqual(retry["requestId"], "request-1")
                self.assertEqual(retry["attempt"], 1)
                self.assertEqual(retry["requestPath"], "/api/v1/me")
                self.assertEqual(retry["timeoutSeconds"], 13)
                self.assertEqual(retry["retryReason"], "ReadTimeout")
                self.assertNotIn("client_secret", json.dumps(retry))
                self.assertNotIn("ReadTimeout(", json.dumps(retry))
            finally:
                _close_root_handlers()
                bot_logging.clear_log_context()
                if previous is None:
                    os.environ.pop("BOT_LOG_DIR", None)
                else:
                    os.environ["BOT_LOG_DIR"] = previous

    def test_exhausted_retryable_response_does_not_poison_next_request_number(self):
        requestor = ObservableRequestor("script:optimist-prime:test", timeout=7)
        requestor._http.request = Mock(
            side_effect=[
                SimpleNamespace(status_code=500),
                SimpleNamespace(status_code=200),
            ]
        )
        with self.assertLogs("optimist_prime", level="INFO"):
            requestor.request("GET", "https://oauth.reddit.com/api/v1/me")
            requestor.request("GET", "https://oauth.reddit.com/api/v1/me")
        self.assertEqual(
            [call.kwargs["timeout"] for call in requestor._http.request.call_args_list],
            [7, 7],
        )
        self.assertEqual(requestor._attempts, {})
        self.assertEqual(requestor._request_ids, {})

    def test_single_note_write_blocks_second_post_after_5xx_before_budget_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget_path = os.path.join(tmp, "reddit-budget.sqlite3")
            requestor = ObservableRequestor(
                "script:optimist-prime:test",
                timeout=7,
                request_budget_path=budget_path,
                request_budget_limit=10,
                request_budget_window_seconds=60,
                request_budget_per_minute=10,
            )
            response = SimpleNamespace(status_code=500)
            requestor._http.request = Mock(return_value=response)

            with single_write_attempt():
                self.assertIs(
                    requestor.request(
                        "POST", "https://oauth.reddit.com/api/mod/notes"
                    ),
                    response,
                )
                with self.assertRaisesRegex(RuntimeError, "repeated POST"):
                    requestor.request(
                        "POST", "https://oauth.reddit.com/api/mod/notes"
                    )

            self.assertEqual(requestor._http.request.call_count, 1)
            snapshot = requestor._request_budget.snapshot()
            self.assertEqual(snapshot.used_requests, 1)

    def test_single_note_write_blocks_second_post_after_transport_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget_path = os.path.join(tmp, "reddit-budget.sqlite3")
            requestor = ObservableRequestor(
                "script:optimist-prime:test",
                timeout=7,
                request_budget_path=budget_path,
                request_budget_limit=10,
                request_budget_window_seconds=60,
                request_budget_per_minute=10,
            )
            requestor._http.request = Mock(
                side_effect=[ReadTimeout("synthetic timeout"), SimpleNamespace(status_code=200)]
            )

            with single_write_attempt():
                with self.assertRaises(RequestException):
                    requestor.request(
                        "POST", "https://oauth.reddit.com/api/mod/notes"
                    )
                with self.assertRaisesRegex(RuntimeError, "repeated POST"):
                    requestor.request(
                        "POST", "https://oauth.reddit.com/api/mod/notes"
                    )

            self.assertEqual(requestor._http.request.call_count, 1)
            self.assertEqual(
                requestor._request_budget.snapshot().used_requests,
                1,
            )

    def test_single_note_write_allows_praw_auth_refresh_retry_after_401(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget_path = os.path.join(tmp, "reddit-budget.sqlite3")
            requestor = ObservableRequestor(
                "script:optimist-prime:test",
                timeout=7,
                request_budget_path=budget_path,
                request_budget_limit=10,
                request_budget_window_seconds=60,
                request_budget_per_minute=10,
            )
            unauthorized = SimpleNamespace(status_code=401)
            success = SimpleNamespace(status_code=200)
            requestor._http.request = Mock(side_effect=[unauthorized, success])

            with single_write_attempt():
                self.assertIs(
                    requestor.request(
                        "POST", "https://oauth.reddit.com/api/mod/notes"
                    ),
                    unauthorized,
                )
                # This models the second request PRAW makes after refreshing
                # the access token in response to the known 401 rejection.
                self.assertIs(
                    requestor.request(
                        "POST", "https://oauth.reddit.com/api/mod/notes"
                    ),
                    success,
                )

            self.assertEqual(requestor._http.request.call_count, 2)
            self.assertEqual(
                requestor._request_budget.snapshot().used_requests,
                2,
            )

    def test_separate_write_contexts_are_independent(self):
        requestor = ObservableRequestor("script:optimist-prime:test", timeout=7)
        response = SimpleNamespace(status_code=500)
        success = SimpleNamespace(status_code=200)
        requestor._http.request = Mock(side_effect=[response, success])
        url = "https://oauth.reddit.com/api/mod/notes"

        with single_write_attempt():
            self.assertIs(requestor.request("POST", url), response)
        with single_write_attempt():
            self.assertIs(requestor.request("POST", url), success)

        self.assertEqual(requestor._http.request.call_count, 2)

    def test_note_guard_does_not_block_get_retries(self):
        requestor = ObservableRequestor("script:optimist-prime:test", timeout=7)
        response = SimpleNamespace(status_code=500)
        success = SimpleNamespace(status_code=200)
        requestor._http.request = Mock(side_effect=[response, success])
        url = "https://oauth.reddit.com/api/mod/notes"

        with single_write_attempt():
            self.assertIs(requestor.request("GET", url), response)
            bot_logging.mark_retry_pending()
            self.assertIs(requestor.request("GET", url), success)

        self.assertEqual(requestor._http.request.call_count, 2)

    def test_operator_formatter_describes_request_progress(self):
        record = logging.LogRecord(
            "optimist_prime",
            logging.WARNING,
            __file__,
            1,
            "unused",
            (),
            None,
        )
        record.event_name = "reddit.request.retryable_failure"
        record.correlation_id = "cycle-1"
        record.phase = "reddit.auth"
        record.request_method = "GET"
        record.request_path = "/api/v1/me"
        record.attempt = 1
        record.timeout_seconds = 15
        record.duration_ms = 15000.2
        record.error_type = "ReadTimeout"

        rendered = OperatorLogFormatter().format(record)
        self.assertIn("GET /api/v1/me attempt 1", rendered)
        self.assertIn("failed retryably", rendered)
        self.assertIn("ReadTimeout", rendered)
        self.assertIn("15000.2ms", rendered)
        self.assertNotIn("unused", rendered)

    def test_operator_formatter_describes_budget_wait(self):
        record = logging.LogRecord(
            "optimist_prime",
            logging.WARNING,
            __file__,
            1,
            "unused",
            (),
            None,
        )
        record.event_name = "reddit.request_budget.waiting"
        record.correlation_id = "cycle-1"
        record.phase = "content.post_scan"
        record.request_method = "GET"
        record.request_path = "/r/example_community/comments/"
        record.wait_seconds = 42.0
        record.budget_remaining_requests = 0

        rendered = OperatorLogFormatter().format(record)

        self.assertIn("request safety budget is full", rendered)
        self.assertIn("42.0s", rendered)

    def test_cycle_action_metrics_are_serialized_as_safe_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("BOT_LOG_DIR")
            os.environ["BOT_LOG_DIR"] = tmp
            try:
                path = bot_logging.configure_logging(also_stderr=False)
                bot_logging.log_event(
                    "bot.cycle.completed",
                    "cycle finished",
                    tldrs_generated=1,
                    replies_sent=2,
                    summons_handled=3,
                    crossposts_made=5,
                    users_banned=0,
                    tokens_used=6,
                    cost=0.01,
                    catch_up_posts_fetched=7,
                    catch_up_comments_fetched=8,
                    catch_up_posts_complete=True,
                    catch_up_comments_complete=False,
                )
                record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            finally:
                _close_root_handlers()
                bot_logging.clear_log_context()
                if previous is None:
                    os.environ.pop("BOT_LOG_DIR", None)
                else:
                    os.environ["BOT_LOG_DIR"] = previous
        self.assertEqual(record["tldrsGenerated"], 1)
        self.assertEqual(record["repliesSent"], 2)
        self.assertEqual(record["catchUpCommentsFetched"], 8)
        self.assertFalse(record["catchUpCommentsComplete"])
        self.assertNotIn("username", json.dumps(record).lower())


if __name__ == "__main__":
    unittest.main()
