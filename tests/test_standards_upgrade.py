"""Tests for bot_logging, diagnostics, prefs, and expanded settings GUI APIs."""

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import bot_logging
import diagnostics
import settings_gui
import settings_gui_prefs
from env_file import load_env_file, save_env_file


class TestBotLogging(unittest.TestCase):
    def test_configure_creates_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            prev = bot_logging.os.environ.get("BOT_LOG_DIR")
            bot_logging.os.environ["BOT_LOG_DIR"] = tmp
            try:
                path = bot_logging.configure_logging(also_stderr=False)
                self.assertTrue(path.exists())
                bot_logging.log_event("test.event", "hello", component="test")
                text = path.read_text(encoding="utf-8")
                self.assertIn("test.event", text)
            finally:
                # Release TimedRotatingFileHandler lock on Windows
                root = bot_logging.logging.getLogger()
                for handler in list(root.handlers):
                    root.removeHandler(handler)
                    handler.close()
                if prev is None:
                    bot_logging.os.environ.pop("BOT_LOG_DIR", None)
                else:
                    bot_logging.os.environ["BOT_LOG_DIR"] = prev


class TestDiagnostics(unittest.TestCase):
    def test_dump_has_required_fields(self):
        payload = diagnostics.dump_diagnostics()
        self.assertTrue(payload["ok"])
        self.assertIn("version", payload)
        self.assertIn("pid", payload)
        self.assertIn("paths", payload)
        self.assertIn("logFile", payload["paths"])


class TestEnvAtomicSave(unittest.TestCase):
    def test_creates_bak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            save_env_file(path, {"A": "1"})
            save_env_file(path, {"A": "2"})
            self.assertEqual(load_env_file(path)["A"], "2")
            bak = Path(str(path) + ".bak")
            self.assertTrue(bak.exists(), f"expected backup at {bak}")
            self.assertIn("A=1", bak.read_text(encoding="utf-8"))


class TestPrefsRanking(unittest.TestCase):
    def test_frequency_rank_keeps_fixed(self):
        prefs = settings_gui_prefs.default_prefs()
        prefs["actionUseCounts"] = {"diagnostics.dump": 5, "rules.add": 1}
        ranked = settings_gui_prefs.ranked_action_ids(
            ["settings.done", "rules.add", "diagnostics.dump"],
            prefs,
            fixed={"settings.done"},
        )
        self.assertEqual(ranked[0], "settings.done")
        self.assertEqual(ranked[1], "diagnostics.dump")
        self.assertEqual(ranked[2], "rules.add")


class TestSettingsGUIExtras(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_env_path = settings_gui.ENV_PATH
        self._orig_repo = settings_gui.REPO_ROOT
        self._orig_ui_token = settings_gui.SettingsHandler.ui_token
        settings_gui.ENV_PATH = Path(self._tmpdir.name) / ".env"
        settings_gui.REPO_ROOT = Path(self._tmpdir.name)
        settings_gui.SettingsHandler.ui_token = "test-token"
        (settings_gui.REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
        settings_gui.save_env_file(
            settings_gui.ENV_PATH,
            {"BOT_MODERATION_RULES_FILE": "data/rules.json"},
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), settings_gui.SettingsHandler)
        settings_gui.SettingsHandler.http_server = self.server
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        settings_gui.ENV_PATH = self._orig_env_path
        settings_gui.REPO_ROOT = self._orig_repo
        settings_gui.SettingsHandler.ui_token = self._orig_ui_token
        self._tmpdir.cleanup()

    def test_actions_and_diagnostics_endpoints(self):
        with urlopen(f"{self.base}/api/actions") as resp:
            data = json.loads(resp.read().decode())
        self.assertTrue(data["ok"])
        ids = {a["id"] for a in data["actions"]}
        self.assertIn("diagnostics.dump", ids)
        self.assertIn("settings.done", ids)

        with urlopen(f"{self.base}/api/diagnostics") as resp:
            diag = json.loads(resp.read().decode())
        self.assertTrue(diag["ok"])
        self.assertEqual(diag["actionId"], "diagnostics.dump")

    def test_prefs_roundtrip(self):
        payload = json.dumps({
            "hiddenSettingIds": ["BOT_CROSSPOST_ENABLED"],
            "fontScale": 1.2,
            "showAdvanced": True,
        }).encode()
        req = Request(
            f"{self.base}/api/prefs",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Optimist-Prime-UI-Token": settings_gui.SettingsHandler.ui_token,
                "Origin": self.base,
            },
            method="POST",
        )
        with urlopen(req) as resp:
            result = json.loads(resp.read().decode())
        self.assertTrue(result["ok"])
        self.assertEqual(result["prefs"]["fontScale"], 1.2)
        self.assertIn("BOT_CROSSPOST_ENABLED", result["prefs"]["hiddenSettingIds"])

    def test_fields_include_visibility_flags(self):
        with urlopen(f"{self.base}/api/settings") as resp:
            data = json.loads(resp.read().decode())
        field = next(f for f in data["fields"] if f["key"] == "BOT_SAFE_MODE")
        self.assertFalse(field["hideable"])
        advanced = next(f for f in data["fields"] if f["key"] == "LLM_API_KEY")
        self.assertTrue(advanced["advanced"])


if __name__ == "__main__":
    unittest.main()
