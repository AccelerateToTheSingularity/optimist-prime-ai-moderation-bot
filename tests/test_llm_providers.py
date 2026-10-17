"""Tests for multi-provider LLM configuration."""

import os
import unittest
from unittest.mock import patch

from llm_client import PROVIDER_PRESETS, create_llm_model, resolve_llm_config


class TestLLMProviders(unittest.TestCase):
    def test_all_presets_defined(self):
        for name in (
            "minimax", "openai", "claude", "gemini", "deepseek", "glm",
            "groq", "mistral", "together", "xai", "custom",
        ):
            self.assertIn(name, PROVIDER_PRESETS)

    def test_resolve_claude_preset(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "claude",
                "ANTHROPIC_API_KEY": "ant-test",
                "LLM_MODEL": "claude-test-model",
            },
            clear=False,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "claude")
        self.assertIn("anthropic.com", cfg.base_url)

    def test_resolve_glm_preset(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "glm",
                "GLM_API_KEY": "glm-test",
                "LLM_MODEL": "glm-test-model",
            },
            clear=False,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "glm")
        self.assertIn("bigmodel.cn", cfg.base_url)
        self.assertEqual(cfg.model, "glm-test-model")

    def test_resolve_groq_preset(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "groq",
                "GROQ_API_KEY": "gsk-test",
                "LLM_MODEL": "groq-test-model",
            },
            clear=False,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "groq")
        self.assertIn("groq.com", cfg.base_url)

    def test_resolve_openai_preset(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
                "LLM_MODEL": "gpt-4o-mini",
            },
            clear=False,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
        self.assertEqual(cfg.model, "gpt-4o-mini")

    def test_resolve_deepseek_preset(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "ds-test",
                "LLM_MODEL": "deepseek-test-model",
                "LLM_THINKING_ENABLED": "",
                "LLM_REASONING_EFFORT": "",
            },
            clear=True,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "deepseek")
        self.assertIn("deepseek.com", cfg.base_url)
        self.assertEqual(cfg.model, "deepseek-test-model")
        self.assertFalse(cfg.thinking_enabled)
        self.assertEqual(cfg.reasoning_effort, "low")

    def test_env_overrides_model_and_base_url(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
                "LLM_MODEL": "gpt-4o",
                "OPENAI_BASE_URL": "https://custom.example/v1",
            },
            clear=False,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertEqual(cfg.base_url, "https://custom.example/v1")

    def test_unknown_provider_fails_closed(self):
        with patch.dict(
            os.environ,
            {"BOT_LLM_PROVIDER": "not-a-real-provider", "LLM_API_KEY": "key"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                resolve_llm_config()

    def test_deepseek_does_not_fall_back_to_generic_key(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "",
                "LLM_API_KEY": "generic-key",
                "OPENAI_API_KEY": "openai-key",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                resolve_llm_config()

    def test_provider_and_model_must_be_selected_explicitly(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "LLM_MODEL": "gpt-4o-mini"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                resolve_llm_config()
        with patch.dict(
            os.environ,
            {"BOT_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Set LLM_MODEL"):
                resolve_llm_config()

    def test_explicit_provider_accepts_its_own_endpoint_override(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
                "LLM_MODEL": "example-model",
                "OPENAI_BASE_URL": "https://provider.example/v1",
            },
            clear=True,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.model, "example-model")
        self.assertEqual(cfg.base_url, "https://provider.example/v1")

    def test_deepseek_is_an_explicitly_selected_provider_not_a_default(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "ds-test",
                "LLM_MODEL": "deepseek-flash",
                "LLM_THINKING_ENABLED": "false",
                "LLM_REASONING_EFFORT": "low",
            },
            clear=True,
        ):
            cfg = resolve_llm_config()
        self.assertEqual(cfg.provider, "deepseek")
        self.assertEqual(cfg.model, "deepseek-flash")
        self.assertFalse(cfg.thinking_enabled)
        self.assertEqual(cfg.reasoning_effort, "low")

    def test_unrelated_provider_key_does_not_satisfy_selected_provider(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "minimax",
                "OPENAI_API_KEY": "",
                "LLM_API_KEY": "",
                "ANTHROPIC_API_KEY": "unrelated-provider-key",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                resolve_llm_config()

    @patch("llm_client.OpenAI")
    def test_create_llm_model_uses_provider(self, mock_openai):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "g-key",
                "LLM_MODEL": "gemini-test-model",
            },
            clear=True,
        ):
            create_llm_model()
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        self.assertEqual(call_kwargs["api_key"], "g-key")
        self.assertIn("generativelanguage.googleapis.com", call_kwargs["base_url"])


if __name__ == "__main__":
    unittest.main()
