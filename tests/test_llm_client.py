import json
import os
import unittest
from unittest.mock import MagicMock, patch

from bot_logging import JsonLineFormatter
from llm_client import (
    LLMIncompleteOutput,
    LLMQuotaExhausted,
    LLMModel,
    RateLimitedLLMModel,
    _prompt_from_contents,
    create_llm_model,
    resolve_api_key,
    resolve_api_timeout_seconds,
    resolve_llm_config,
)


class TestLLMClient(unittest.TestCase):
    def test_prompt_from_string(self):
        self.assertEqual(_prompt_from_contents("hello"), "hello")

    def test_prompt_from_gemini_style_parts(self):
        contents = [{"role": "user", "parts": ["part one", "part two"]}]
        self.assertEqual(_prompt_from_contents(contents), "part one\n\npart two")

    def test_resolve_api_key_prefers_openai(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "GEMINI_API_KEY": "gemini-key",
            },
            clear=False,
        ):
            self.assertEqual(resolve_api_key(), "openai-key")

    def test_generate_content_maps_openai_response(self):
        mock_client = MagicMock()
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_choice = MagicMock()
        mock_choice.message.content = "  summary text  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response

        model = LLMModel(mock_client, "MiniMax-M2.7")
        response = model.generate_content(
            [{"role": "user", "parts": ["Summarize this"]}],
            generation_config={"temperature": 0.3, "max_output_tokens": 100},
        )

        self.assertEqual(response.text, "summary text")
        self.assertEqual(response.usage_metadata.prompt_token_count, 10)
        self.assertEqual(response.usage_metadata.candidates_token_count, 5)
        mock_client.chat.completions.create.assert_called_once_with(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Summarize this"}],
            temperature=0.3,
            max_tokens=100,
        )

    @patch("llm_client.OpenAI")
    def test_create_llm_model_uses_env(self, mock_openai):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "minimax",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://api.minimax.io/v1",
                "LLM_MODEL": "MiniMax-M2.7",
            },
            clear=False,
        ):
            create_llm_model()
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            timeout=60.0,
        )

    def test_configured_llm_endpoint_requires_tls_except_for_loopback(self):
        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "custom",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "fixture-model",
                "OPENAI_BASE_URL": "http://provider.example.invalid/v1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must use HTTPS"):
                resolve_llm_config()

        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "custom",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "fixture-model",
                "OPENAI_BASE_URL": "http://127.0.0.1:1234/v1",
            },
            clear=True,
        ):
            self.assertEqual(resolve_llm_config().base_url, "http://127.0.0.1:1234/v1")

        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "custom",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "fixture-model",
                "OPENAI_BASE_URL": "https://user:password@provider.example.invalid/v1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must not contain credentials"):
                resolve_llm_config()

    def test_resolve_api_timeout_is_bounded_and_tolerates_invalid_values(self):
        with patch.dict(os.environ, {"LLM_API_TIMEOUT_SECONDS": "2"}, clear=False):
            self.assertEqual(resolve_api_timeout_seconds(), 5.0)
        with patch.dict(os.environ, {"LLM_API_TIMEOUT_SECONDS": "900"}, clear=False):
            self.assertEqual(resolve_api_timeout_seconds(), 600.0)
        with patch.dict(os.environ, {"LLM_API_TIMEOUT_SECONDS": "not-a-number"}, clear=False):
            self.assertEqual(resolve_api_timeout_seconds(), 60.0)

    @patch("llm_client.urlopen")
    def test_builtin_transport_posts_and_maps_chat_completion(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "  OK  ",
                            "reasoning_content": "private chain of thought that must not escape",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        with patch.dict(
            os.environ,
            {
                "BOT_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-key",
                "LLM_MODEL": "deepseek-flash",
                "LLM_THINKING_ENABLED": "true",
                "LLM_REASONING_EFFORT": "low",
                "OPENAI_BASE_URL": "",
                "LLM_BASE_URL": "",
                "LLM_API_TIMEOUT_SECONDS": "12",
            },
            clear=True,
        ):
            model = create_llm_model()
            response = model.generate_content("Reply with exactly OK.")

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 12.0)
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["model"], "deepseek-flash")
        self.assertEqual(request_body["thinking"], {"type": "enabled"})
        self.assertEqual(request_body["reasoning_effort"], "low")
        self.assertNotIn("temperature", request_body)
        self.assertEqual(response.text, "OK")
        self.assertNotIn("private chain of thought", response.text)
        self.assertEqual(response.usage_metadata.prompt_token_count, 4)
        self.assertEqual(response.usage_metadata.candidates_token_count, 2)

    def test_rate_limit_counts_failed_provider_attempts(self):
        inner = MagicMock()
        inner.generate_content.side_effect = RuntimeError("temporary provider failure")
        model = RateLimitedLLMModel(inner, 1)

        with self.assertRaises(RuntimeError):
            model.generate_content("first")
        with self.assertRaises(LLMQuotaExhausted):
            model.generate_content("second")
        self.assertEqual(inner.generate_content.call_count, 1)

    def test_empty_final_answer_is_rejected(self):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "   "
        response.choices[0].message.reasoning_content_length = 71
        response.choices[0].message.refusal_present = False
        response.choices[0].finish_reason = "length"
        response.choices[0].tool_call_count = 0
        response.usage = None
        client.chat.completions.create.return_value = response

        with self.assertLogs("optimist_prime", level="WARNING") as captured:
            with self.assertRaisesRegex(LLMIncompleteOutput, "empty final answer") as raised:
                LLMModel(client, "test-model").generate_content("hello")

        self.assertEqual(captured.records[0].event_name, "llm.response.empty")
        self.assertEqual(captured.records[0].finish_reason, "length")
        self.assertEqual(captured.records[0].reasoning_content_length, 71)
        self.assertEqual(raised.exception.reason, "empty_final_answer")

    def test_truncated_final_answer_is_rejected(self):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "This answer ends halfway"
        response.choices[0].message.reasoning_content_length = 0
        response.choices[0].finish_reason = "length"
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=512)
        client.chat.completions.create.return_value = response

        with self.assertLogs("optimist_prime", level="WARNING") as captured:
            with self.assertRaisesRegex(LLMIncompleteOutput, "output limit") as raised:
                LLMModel(client, "test-model").generate_content("hello")

        self.assertEqual(captured.records[0].event_name, "llm.response.truncated")
        self.assertEqual(captured.records[0].finish_reason, "length")
        self.assertEqual(captured.records[0].content_length, len("This answer ends halfway"))
        self.assertEqual(raised.exception.reason, "output_limit")
        log_record = json.loads(JsonLineFormatter().format(captured.records[0]))
        self.assertEqual(log_record["failureReason"], "output_limit")
        self.assertEqual(log_record["finishReason"], "length")
        self.assertNotIn("This answer ends halfway", str(log_record))

    def test_deepseek_thinking_can_be_disabled_for_one_generation(self):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Direct answer."
        response.usage = None
        client.chat.completions.create.return_value = response
        model = LLMModel(
            client,
            "deepseek-flash",
            provider="deepseek",
            thinking_enabled=True,
            reasoning_effort="low",
        )

        model.generate_content(
            "hello",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 512,
                "thinking_enabled": False,
            },
        )

        client.chat.completions.create.assert_called_once_with(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=512,
            thinking={"type": "disabled"},
            temperature=0.7,
        )

    def test_zero_rate_limit_allows_no_calls(self):
        inner = MagicMock()
        model = RateLimitedLLMModel(inner, 0)
        with self.assertRaises(LLMQuotaExhausted):
            model.generate_content("blocked")
        inner.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
