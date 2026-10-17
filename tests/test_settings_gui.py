"""Tests for the local settings GUI HTTP API."""

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import settings_gui


class TestSettingsGUI(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_env_path = settings_gui.ENV_PATH
        self._orig_repo = settings_gui.REPO_ROOT
        self._orig_ui_token = settings_gui.SettingsHandler.ui_token
        settings_gui.ENV_PATH = Path(self._tmpdir.name) / ".env"
        settings_gui.REPO_ROOT = Path(self._tmpdir.name)
        settings_gui.SettingsHandler.ui_token = "test-token"
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

    def test_get_settings_and_save(self):
        with urlopen(f"{self.base}/api/settings") as resp:
            data = json.loads(resp.read().decode())
        self.assertTrue(data["ok"])
        self.assertIn("BOT_LLM_PROVIDER", data["values"])
        self.assertNotIn("BOT_CONTENT_MODERATION_RULES", data["values"])
        self.assertNotIn("BOT_CONTENT_MODERATION_ACTION", data["values"])

        payload = json.dumps({
            "BOT_LLM_PROVIDER": "claude",
            "BOT_SAFE_MODE": "true",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }).encode()
        req = Request(
            f"{self.base}/api/settings",
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

        saved = settings_gui.load_env_file(settings_gui.ENV_PATH)
        self.assertEqual(saved["BOT_LLM_PROVIDER"], "claude")
        self.assertEqual(saved["ANTHROPIC_API_KEY"], "sk-ant-test")

    def test_token_bearing_settings_page_cannot_be_framed(self):
        with urlopen(f"{self.base}/") as response:
            self.assertEqual(
                response.headers.get("Content-Security-Policy"),
                "frame-ancestors 'none'",
            )
            self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
            self.assertIn(b"test-token", response.read())

    def test_dns_rebinding_host_cannot_read_the_token_bearing_page(self):
        request = Request(f"{self.base}/", headers={"Host": "attacker.example"})
        with self.assertRaises(HTTPError) as context:
            urlopen(request)
        self.assertEqual(context.exception.code, 403)
        self.assertNotIn("test-token", context.exception.read().decode())

    def test_cross_origin_post_cannot_change_settings_even_with_session_token(self):
        request = Request(
            f"{self.base}/api/settings",
            data=json.dumps({"BOT_LLM_PROVIDER": "unexpected"}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Optimist-Prime-UI-Token": settings_gui.SettingsHandler.ui_token,
                "Origin": "http://attacker.example",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request)
        self.assertEqual(context.exception.code, 403)
        self.assertNotEqual(
            settings_gui.load_env_file(settings_gui.ENV_PATH).get("BOT_LLM_PROVIDER"),
            "unexpected",
        )

    def test_new_settings_default_to_safe_mode_and_no_reddit_actions(self):
        with urlopen(f"{self.base}/api/settings") as resp:
            data = json.loads(resp.read().decode())

        values = data["values"]
        self.assertEqual(values["BOT_SAFE_MODE"], "true")
        for key in (
            "BOT_CONTENT_MODERATION_ENABLED",
            "BOT_CONVERSATIONAL_REPLIES_ENABLED",
            "BOT_SUMMONS_ENABLED",
            "BOT_REPORTED_USER_REVIEW_ENABLED",
            "BOT_AUTOMATED_BAN_ENABLED",
            "BOT_MODMAIL_ASSISTANT_ENABLED",
            "BOT_CROSSPOST_ENABLED",
            "BOT_USER_CUSTOM_FLAIR_ENABLED",
        ):
            self.assertEqual(values[key], "false", key)
        self.assertEqual(values["BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT"], "")
        self.assertEqual(values["BOT_CONTENT_MODERATION_ENABLED"], "false")

    def test_invalid_safe_mode_value_keeps_safe_default_when_settings_are_loaded(self):
        settings_gui.save_env_file(
            settings_gui.ENV_PATH,
            {"BOT_SAFE_MODE": "treu", "BOT_LLM_PROVIDER": ""},
        )

        values = settings_gui._current_values()
        self.assertEqual(values["BOT_SAFE_MODE"], "true")
        saved = settings_gui._merge_save_payload(
            values | {"BOT_LLM_PROVIDER": "openai"},
            env_path=settings_gui.ENV_PATH,
        )
        self.assertEqual(saved["BOT_SAFE_MODE"], "true")
        self.assertEqual(saved["BOT_LLM_PROVIDER"], "openai")

    def test_history_ban_threshold_is_optional_and_bounded(self):
        with urlopen(f"{self.base}/api/settings") as resp:
            data = json.loads(resp.read().decode())
        fields = {field["key"]: field for field in data["fields"]}
        threshold_key = "BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT"
        self.assertEqual(fields[threshold_key]["field_type"], "int")
        self.assertEqual((fields[threshold_key]["minimum"], fields[threshold_key]["maximum"]), (1, 100))
        self.assertEqual(data["values"][threshold_key], "")
        enabled = settings_gui._merge_save_payload({
            "BOT_AUTOMATED_BAN_ENABLED": "true",
            threshold_key: "94",
        }, env_path=settings_gui.ENV_PATH)
        self.assertEqual(enabled["BOT_AUTOMATED_BAN_ENABLED"], "true")
        self.assertEqual(enabled[threshold_key], "94")
        disabled = settings_gui._merge_save_payload({threshold_key: ""}, env_path=settings_gui.ENV_PATH)
        self.assertEqual(disabled[threshold_key], "")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            settings_gui._merge_save_payload({threshold_key: "0"}, env_path=settings_gui.ENV_PATH)
        with self.assertRaisesRegex(ValueError, "at most 100"):
            settings_gui._merge_save_payload({threshold_key: "101"}, env_path=settings_gui.ENV_PATH)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            settings_gui._merge_save_payload({
                threshold_key: "90.5",
            }, env_path=settings_gui.ENV_PATH)

    def test_settings_api_masks_password_values(self):
        settings_gui.save_env_file(
            settings_gui.ENV_PATH,
            {"OPENAI_API_KEY": "operator-secret", "BOT_SAFE_MODE": "true"},
        )
        with urlopen(f"{self.base}/api/settings") as resp:
            data = json.loads(resp.read().decode())
        self.assertEqual(data["values"]["OPENAI_API_KEY"], "")
        self.assertTrue(data["configured"]["OPENAI_API_KEY"])

    def test_mutating_api_requires_session_token(self):
        req = Request(
            f"{self.base}/api/settings",
            data=b"{}",
            headers={"Content-Type": "application/json", "Origin": self.base},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req)
        self.assertEqual(ctx.exception.code, 403)

    def test_meta_lists_providers(self):
        with urlopen(f"{self.base}/api/meta") as resp:
            data = json.loads(resp.read().decode())
        self.assertIn("claude", data["providers"])
        self.assertIn("glm", data["providers"])
        self.assertEqual(
            data["mainPathSections"],
            ["Reddit", "AI providers", "Safety", "Features", "History review"],
        )

    def test_shutdown_payload_saves_pending_settings_and_rules_before_exit(self):
        settings_gui._persist_shutdown_payload({
            "values": {"BOT_SAFE_MODE": "false"},
            "rules": [{
                "name": "example_rule",
                "description": "A synthetic inactive example.",
                "active": False,
                "order": 10,
                "target": "both",
                "actions": [],
                "conditions": {},
            }],
        })

        saved_env = settings_gui.load_env_file(settings_gui.ENV_PATH)
        self.assertEqual(saved_env["BOT_SAFE_MODE"], "false")
        saved_rules = json.loads(
            (settings_gui.REPO_ROOT / "data" / "rules.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_rules[0]["name"], "example_rule")
        self.assertFalse(saved_rules[0]["active"])

    def test_rules_load_and_save(self):
        example = Path(self._tmpdir.name) / "data" / "rules.example.json"
        example.parent.mkdir(parents=True)
        example.write_text(
            json.dumps([{
                "name": "spam_detector",
                "description": "Is this spam?",
                "active": True,
                "order": 10,
                "target": "both",
                "actions": ["report"],
                "conditions": {},
            }]),
            encoding="utf-8",
        )
        # Point REPO_ROOT example - copy to repo root data
        settings_gui.REPO_ROOT = Path(self._tmpdir.name)
        (settings_gui.REPO_ROOT / "data").mkdir(exist_ok=True)
        (settings_gui.REPO_ROOT / "data" / "rules.example.json").write_text(
            example.read_text(encoding="utf-8"), encoding="utf-8"
        )

        with urlopen(f"{self.base}/api/rules") as resp:
            data = json.loads(resp.read().decode())
        self.assertTrue(data["ok"])
        self.assertEqual(data["rules"][0]["name"], "spam_detector")

        rules = data["rules"]
        rules[0]["description"] = "Updated question"
        payload = json.dumps({"rules": rules}).encode()
        req = Request(
            f"{self.base}/api/rules",
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

        rules_file = settings_gui.REPO_ROOT / "data" / "rules.json"
        saved = json.loads(rules_file.read_text(encoding="utf-8"))
        self.assertEqual(saved[0]["description"], "Updated question")

    def test_active_rule_without_action_is_rejected_without_overwriting_rules(self):
        rules_file = settings_gui.REPO_ROOT / "data" / "rules.json"
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        existing = [{
            "name": "existing_rule",
            "description": "A synthetic example rule.",
            "active": True,
            "order": 10,
            "target": "both",
            "actions": ["report"],
            "conditions": {},
        }]
        rules_file.write_text(json.dumps(existing), encoding="utf-8")
        payload = json.dumps({"rules": [{
            "name": "example_rule",
            "description": "A synthetic example rule.",
            "active": True,
            "order": 20,
            "target": "both",
            "actions": [],
            "conditions": {},
        }]}).encode()
        req = Request(
            f"{self.base}/api/rules",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Optimist-Prime-UI-Token": settings_gui.SettingsHandler.ui_token,
                "Origin": self.base,
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("needs at least one action", ctx.exception.read().decode())
        self.assertEqual(json.loads(rules_file.read_text(encoding="utf-8")), existing)

    def test_inactive_rule_can_be_saved_without_an_action(self):
        payload = json.dumps({"rules": [{
            "name": "example_rule",
            "description": "A synthetic example rule.",
            "active": False,
            "order": 20,
            "target": "both",
            "actions": [],
            "conditions": {},
        }]}).encode()
        req = Request(
            f"{self.base}/api/rules",
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
        rules_file = settings_gui.REPO_ROOT / "data" / "rules.json"
        saved = json.loads(rules_file.read_text(encoding="utf-8"))
        self.assertEqual(saved[0]["actions"], [])

    def test_shutdown_endpoint(self):
        req = Request(
            f"{self.base}/api/shutdown",
            data=b"",
            headers={
                "X-Optimist-Prime-UI-Token": settings_gui.SettingsHandler.ui_token,
                "Origin": self.base,
            },
            method="POST",
        )
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        self.assertTrue(data["ok"])
        self.server.shutdown()
        self.thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
