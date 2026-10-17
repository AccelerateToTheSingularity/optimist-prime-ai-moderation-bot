"""Tests for the native PySide6 moderator console boundary and models."""

from __future__ import annotations

import os
import json
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fast_test_mode

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import moderator_console as console_module  # noqa: E402

from moderator_console import (  # noqa: E402
    CommentFeedModel,
    CommentFilterProxyModel,
    FeedController,
    FeedWorker,
    LocalActivityController,
    ModeratorConsole,
    SettingsPanel,
    TaskController,
    TaskWorker,
    load_settings_for_ui,
    parse_args as parse_console_args,
    persist_settings_for_ui,
    read_log_tail,
)


_CONSOLE_TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="optimist-console-tests-")
_ORIGINAL_QSETTINGS_FORMAT = QSettings.defaultFormat()
_ORIGINAL_ACTIVITY_FILE = os.environ.get("BOT_RUNTIME_ACTIVITY_FILE")


def setUpModule():
    """Keep console tests away from the operator's heartbeat and UI settings."""
    root = Path(_CONSOLE_TEST_RUNTIME.name)
    runtime = root / "runtime"
    console_module.CONSOLE_RUNTIME_DIR = runtime
    console_module.CONSOLE_HEARTBEAT_PATH = runtime / "moderator_console_heartbeat.json"
    console_module.CONSOLE_LOCK_PATH = runtime / "moderator_console.lock"
    os.environ["BOT_RUNTIME_ACTIVITY_FILE"] = str(root / "activity.json")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(root / "settings"),
    )


def tearDownModule():
    QSettings.setDefaultFormat(_ORIGINAL_QSETTINGS_FORMAT)
    if _ORIGINAL_ACTIVITY_FILE is None:
        os.environ.pop("BOT_RUNTIME_ACTIVITY_FILE", None)
    else:
        os.environ["BOT_RUNTIME_ACTIVITY_FILE"] = _ORIGINAL_ACTIVITY_FILE
    _CONSOLE_TEST_RUNTIME.cleanup()


class FakeFeedService:
    account_name = "example_bot"

    def __init__(self):
        self.refresh_calls = 0
        self.paused = False
        self.stopped = False

    def refresh_now(self):
        self.refresh_calls += 1
        return True

    def pause(self):
        self.paused = True
        return True

    def resume(self):
        self.paused = False
        return True

    def stop(self):
        self.stopped = True

    def snapshot(self):
        return {
            "ok": True,
            "status": {
                "state": "paused" if self.paused else "connected",
                "message": "fake feed",
                "subreddit": "example_community",
                "accountName": self.account_name,
                "cachedCount": 0,
            },
            "comments": [],
        }


class TestModeratorConsoleModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["optimist-prime-tests"])

    def test_feed_model_replaces_rows_and_proxy_filters(self):
        model = CommentFeedModel()
        model.update_snapshot(
            {
                "comments": [
                    {
                        "id": "a",
                        "author": "example_user_a",
                        "body": "A useful post",
                        "linkTitle": "A thread",
                        "createdAt": "2026-09-10T00:00:00Z",
                        "createdUtc": 10,
                        "score": 3,
                        "processingState": "processed",
                    },
                    {
                        "id": "b",
                        "author": "example_user_b",
                        "body": "Another comment",
                        "linkTitle": "B thread",
                        "createdAt": "2026-09-10T00:01:00Z",
                        "createdUtc": 20,
                        "score": 1,
                        "processingState": "error",
                        "error": "temporary timeout",
                    },
                    {
                        "id": "c",
                        "author": "example_user_c",
                        "body": "Recent error",
                        "linkTitle": "C thread",
                        "createdAt": "2026-09-10T00:02:00Z",
                        "createdUtc": time.time() - 300,
                        "score": 2,
                        "processingState": "error",
                        "error": "temporary timeout",
                    },
                ]
            }
        )
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.data(model.index(0, 1)), "u/example_user_a")

        proxy = CommentFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.set_query("example_user_a")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.data(proxy.index(0, 1)), "u/example_user_a")
        proxy.set_query("not present")
        self.assertEqual(proxy.rowCount(), 0)
        proxy.set_query("")
        proxy.set_time_range("Last hour")
        self.assertEqual(proxy.rowCount(), 1)
        proxy.set_error_filter("Errors only")
        self.assertEqual(proxy.rowCount(), 1)
        proxy.set_processing_state("Error")
        self.assertEqual(proxy.rowCount(), 1)
        proxy.set_time_range("Any time")
        proxy.set_error_filter("All items")
        proxy.set_processing_state("All states")
        self.assertEqual(proxy.rowCount(), 3)

        model.update_snapshot({"comments": [{"id": "c", "author": "example_user", "body": "new"}]})
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.data(model.index(0, 1)), "u/example_user")

    def test_feed_worker_refresh_uses_service_without_gui_network_call(self):
        service = FakeFeedService()
        worker = FeedWorker(lambda: service, 1)
        snapshots = []
        worker.snapshot_ready.connect(snapshots.append)
        worker._service = service
        worker.refresh_now()
        self.assertEqual(service.refresh_calls, 1)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["status"]["accountName"], "example_bot")

    def test_task_worker_reports_result_and_error_without_hiding_failure(self):
        result = []
        errors = []
        worker = TaskWorker(lambda: {"ok": True})
        worker.succeeded.connect(result.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(result, [{"ok": True}])
        self.assertEqual(errors, [])

        failing = TaskWorker(lambda: (_ for _ in ()).throw(RuntimeError("temporary failure")))
        failing.failed.connect(errors.append)
        failing.run()
        self.assertTrue(errors)
        self.assertIn("temporary failure", errors[-1])

    def test_task_controller_runs_operation_on_a_qthread(self):
        result = []
        finished = []
        gui_thread_id = threading.get_ident()
        operation_thread_id = []

        def operation():
            operation_thread_id.append(threading.get_ident())
            return "done"

        controller = TaskController(operation)
        controller.succeeded.connect(result.append)
        controller.finished.connect(lambda: finished.append(True))
        self.assertTrue(controller.start())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not finished:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(result, ["done"])
        self.assertTrue(finished)
        self.assertEqual(len(operation_thread_id), 1)
        self.assertNotEqual(operation_thread_id[0], gui_thread_id)
        self.assertTrue(controller.wait(3000))

    def test_feed_controller_delivers_snapshot_and_stops_worker_cleanly(self):
        service = FakeFeedService()
        snapshots = []
        controller = FeedController(lambda: service, 1)
        controller.snapshot_ready.connect(snapshots.append)
        controller.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not snapshots:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(snapshots)
        self.assertTrue(controller.stop(3000))
        self.assertTrue(service.stopped)

    def test_console_can_start_without_reddit_and_close_cleanly(self):
        window = ModeratorConsole(
            start_feed=False,
            enable_tray=False,
            schedule_initial_refresh=False,
        )
        window.show()
        self.app.processEvents()
        self.assertIn("Optimist Prime", window.windowTitle())
        window.close()
        self.app.processEvents()

    def test_heartbeat_separates_qt_tray_registration_from_tray_area_availability(self):
        window = ModeratorConsole(
            start_activity=False,
            enable_tray=True,
            schedule_initial_refresh=False,
        )
        try:
            window._write_heartbeat()
            payload = json.loads(console_module.CONSOLE_HEARTBEAT_PATH.read_text(encoding="utf-8"))
            self.assertIsInstance(payload["systemTrayAvailable"], bool)
            self.assertIsInstance(payload["trayIconRegistered"], bool)
            self.assertEqual(payload["trayAvailable"], payload["trayIconRegistered"])
        finally:
            window._exiting = True
            window.close()
            self.app.processEvents()

    def test_show_existing_option_is_available_for_the_recovery_script(self):
        args = parse_console_args(["--show-existing"])
        self.assertTrue(args.show_existing)

    def test_show_existing_local_command_restores_window_without_another_console(self):
        name = "OptimistPrimeTest" + uuid.uuid4().hex
        restored = []
        window = SimpleNamespace(_restore_from_tray=lambda: restored.append(True))
        server = console_module._install_console_control_server(self.app, window, name)
        accepted = []
        server.newConnection.connect(lambda: accepted.append(True))
        client_result = []
        client = threading.Thread(
            target=lambda: client_result.append(
                console_module._request_existing_console_show(name)
            )
        )
        try:
            client.start()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and (client.is_alive() or not restored):
                self.app.processEvents()
                time.sleep(0.01)
            client.join(timeout=0.1)
            self.assertEqual(accepted, [True])
            self.assertEqual(restored, [True])
            self.assertEqual(client_result, [True])
        finally:
            server.close()
            server.removeServer(name)

    def test_console_feed_is_disabled_by_default(self):
        factory_called = False

        def service_factory():
            nonlocal factory_called
            factory_called = True
            return FakeFeedService()

        self.assertFalse(parse_console_args([]).enable_feed)
        window = ModeratorConsole(
            service_factory=service_factory,
            start_activity=False,
            enable_tray=False,
            schedule_initial_refresh=False,
        )
        self.assertIsNone(window._feed_controller)
        self.assertFalse(factory_called)
        window.close()
        self.app.processEvents()

    def test_console_exposes_action_contract_and_guards_destructive_action(self):
        window = ModeratorConsole(
            start_feed=False,
            enable_tray=False,
            schedule_initial_refresh=False,
        )
        actions = {item["id"]: item for item in window.available_actions()}
        self.assertIn("activity.refresh", actions)
        self.assertTrue(actions["activity.refresh"]["enabled"])
        rejected = window.invoke_action("worker.install")
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("unknown action", rejected["reason"].lower())
        self.assertFalse(any(action_id.startswith("scheduler.") for action_id in actions))

        # An offscreen Qt platform may start maximized when it has no saved
        # window geometry. Put the test into the exact state it intends to test.
        window.restore_console()
        self.assertFalse(window.isMaximized())
        maximized = window.invoke_action("window.maximize")
        self.assertEqual(maximized["status"], "requested")
        self.assertTrue(maximized["correlationId"])
        restored = window.invoke_action("window.restore")
        self.assertEqual(restored["status"], "requested")
        window.close()
        self.app.processEvents()

    def test_exit_explicitly_quits_console_after_tray_window_is_hidden(self):
        window = ModeratorConsole(
            start_feed=False,
            start_activity=False,
            enable_tray=False,
            schedule_initial_refresh=False,
        )
        window.hide()

        with patch.object(window, "_shutdown") as shutdown, \
             patch.object(console_module.QApplication, "quit") as quit_app:
            window._exit_application()

        shutdown.assert_called_once()
        quit_app.assert_called_once_with()
        self.app.processEvents()

    def test_fast_testing_control_uses_the_existing_worker_lease(self):
        with tempfile.TemporaryDirectory() as raw, patch.dict(
            os.environ,
            {"BOT_RUNTIME_FAST_TEST_MODE_FILE": str(Path(raw) / "fast-test.json")},
            clear=False,
        ):
            window = ModeratorConsole(
                start_feed=False,
                start_activity=False,
                enable_tray=False,
                schedule_initial_refresh=False,
            )
            try:
                action = next(
                    item for item in window.available_actions()
                    if item["id"] == "worker.fast_test"
                )
                self.assertTrue(action["enabled"])
                self.assertEqual(
                    window._fast_test_button.text(),
                    "Start 2-minute fast testing",
                )

                started = window.invoke_action(
                    "worker.fast_test", {"active": True}
                )
                self.assertEqual(started["status"], "requested")
                lease = fast_test_mode.get_fast_test_mode()
                self.assertTrue(lease["active"])
                self.assertEqual(lease["interval_seconds"], 3)
                self.assertIn("after each cycle", window._fast_test_status.text())
                self.assertEqual(window._fast_test_button.text(), "Stop fast testing")

                stopped = window.invoke_action(
                    "worker.fast_test", {"active": False}
                )
                self.assertEqual(stopped["status"], "requested")
                self.assertFalse(fast_test_mode.get_fast_test_mode()["active"])
            finally:
                window.close()
                self.app.processEvents()

    def test_local_activity_controller_reads_snapshot_without_reddit(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "activity.json"
            path.write_text(
                '{"version":1,"subreddit":"example_community","items":[{"id":"c1","contentType":"comment","author":"example_user","body":"hello"}]}',
                encoding="utf-8",
            )
            snapshots = []
            with patch.dict(os.environ, {"BOT_RUNTIME_ACTIVITY_FILE": str(path)}, clear=False):
                controller = LocalActivityController(1, "example_community")
                controller.snapshot_ready.connect(snapshots.append)
                controller.start()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not snapshots:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertTrue(snapshots)
                self.assertEqual(snapshots[0]["comments"][0]["id"], "c1")
                self.assertIn("no Reddit request", snapshots[0]["status"]["message"])
                self.assertTrue(controller.stop(3000))

    def test_local_activity_controller_uses_owner_path_from_env_file(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "activity.json"
            env_path = root / ".env"
            path.write_text(
                '{"version":1,"subreddit":"example_community","items":[{"id":"c2","contentType":"comment","author":"bob","body":"from operator path"}]}',
                encoding="utf-8",
            )
            env_path.write_text(f"BOT_RUNTIME_ACTIVITY_FILE={path}\n", encoding="utf-8")
            snapshots = []
            with patch.dict(os.environ, {"BOT_RUNTIME_ACTIVITY_FILE": ""}, clear=False):
                controller = LocalActivityController(1, "example_community", env_path=env_path)
                controller.snapshot_ready.connect(snapshots.append)
                controller.start()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not snapshots:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertTrue(snapshots)
                self.assertEqual(snapshots[0]["comments"][0]["id"], "c2")
                self.assertTrue(controller.stop(3000))


class TestModeratorConsolePersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["optimist-prime-tests-persistence"])

    def test_blank_password_editor_does_not_replace_stored_secret(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env_path = root / ".env"
            env_path.write_text(
                "DEEPSEEK_API_KEY=local-secret-value\nBOT_SAFE_MODE=true\n",
                encoding="utf-8",
            )
            panel = SettingsPanel(env_path)
            loaded_values, configured = load_settings_for_ui(env_path)
            panel.apply_loaded_values((loaded_values, configured))
            self.assertEqual(panel._values["DEEPSEEK_API_KEY"], "")
            self.assertTrue(panel._configured["DEEPSEEK_API_KEY"])
            requests = []
            panel.persist_requested.connect(lambda payload, path: requests.append((payload, path)))
            panel._queue_value("BOT_SAFE_MODE", "false")
            panel._persist_pending()
            self.assertEqual(requests[0][0], {"BOT_SAFE_MODE": "false"})
            persist_settings_for_ui(requests[0][0], Path(requests[0][1]))
            contents = env_path.read_text(encoding="utf-8")
            self.assertIn("DEEPSEEK_API_KEY=local-secret-value", contents)
            self.assertIn("BOT_SAFE_MODE=false", contents)

    def test_history_ban_threshold_is_optional_and_saves_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            env_path = Path(raw) / ".env"
            panel = SettingsPanel(env_path)
            key = "BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT"
            editor = panel._editors[key]
            self.assertEqual(editor.text(), "")
            requests = []
            panel.persist_requested.connect(lambda payload, path: requests.append((payload, path)))
            editor.setText("94")
            panel._persist_pending()
            self.assertEqual(requests[-1][0][key], "94")
            persist_settings_for_ui(requests[-1][0], Path(requests[-1][1]))
            self.assertIn(f"{key}=94", env_path.read_text(encoding="utf-8"))

    def test_log_reader_redacts_secret_loaded_from_local_env(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env_path = root / ".env"
            log_dir = root / "logs"
            log_dir.mkdir()
            env_path.write_text(
                f"BOT_LOG_DIR={log_dir}\nDEEPSEEK_API_KEY=local-secret-value\n",
                encoding="utf-8",
            )
            (log_dir / "optimist_prime.jsonl").write_text(
                '{"message":"accidental local-secret-value in old text"}\n',
                encoding="utf-8",
            )
            rendered = read_log_tail(env_path=env_path)
            self.assertIn("[redacted]", rendered)
            self.assertNotIn("local-secret-value", rendered)


if __name__ == "__main__":
    unittest.main()
