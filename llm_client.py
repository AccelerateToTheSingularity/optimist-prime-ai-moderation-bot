"""Centralized OpenAI-compatible LLM client for Optimist Prime.

The operator selects a supported provider and model in local configuration.
The wrapper keeps the ``generate_content`` interface used throughout feature
modules so they share request handling.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_REASONING_EFFORT = "low"
VALID_REASONING_EFFORTS = frozenset({"low", "high", "max"})
DEFAULT_API_TIMEOUT_SECONDS = 60.0
MIN_API_TIMEOUT_SECONDS = 5.0
MAX_API_TIMEOUT_SECONDS = 600.0


class _OpenAICompatibleCompletions:
    """Minimal chat-completions transport used by the bot.

    The bot only needs the common OpenAI-compatible chat-completions shape.
    Keeping this transport in the standard library avoids importing the large
    generated OpenAI SDK model tree on every short-lived scheduled cycle.
    """

    def __init__(self, api_key: str, base_url: str, timeout: float):
        self._api_key = api_key
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 1024,
        thinking: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> Any:
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # DeepSeek thinking requests must not contain temperature or other
        # sampling controls.  Optional fields are added only when the caller
        # explicitly asks for them, which also keeps the transport compatible
        # with providers that reject unknown request fields.
        if temperature is not None:
            request_payload["temperature"] = temperature
        if thinking is not None:
            request_payload["thinking"] = thinking
        if reasoning_effort is not None:
            request_payload["reasoning_effort"] = reasoning_effort

        payload = json.dumps(request_payload).encode("utf-8")
        request = Request(
            self._endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        choices = []
        for choice in body.get("choices", []):
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            choices.append(
                SimpleNamespace(
                    finish_reason=(
                        str(choice.get("finish_reason") or "")
                        if isinstance(choice, dict) else ""
                    ),
                    tool_call_count=(
                        len(message.get("tool_calls", []))
                        if isinstance(message, dict)
                        and isinstance(message.get("tool_calls", []), list)
                        else 0
                    ),
                    # Preserve only safe diagnostic metadata. DeepSeek's
                    # reasoning text is never posted, logged, or stored.
                    message=SimpleNamespace(
                        content=message.get("content") if isinstance(message, dict) else None,
                        reasoning_content_length=(
                            len(str(message.get("reasoning_content") or ""))
                            if isinstance(message, dict) else 0
                        ),
                        refusal_present=(
                            bool(message.get("refusal"))
                            if isinstance(message, dict) else False
                        ),
                    )
                )
            )
        usage_data = body.get("usage")
        usage = (
            SimpleNamespace(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            )
            if isinstance(usage_data, dict)
            else None
        )
        return SimpleNamespace(choices=choices, usage=usage)


class _OpenAICompatibleClient:
    """Small client surface matching the part of ``OpenAI`` the bot uses."""

    def __init__(self, *, api_key: str, base_url: str, timeout: float):
        self.chat = SimpleNamespace(
            completions=_OpenAICompatibleCompletions(api_key, base_url, timeout)
        )


# Keep the constructor patchable for the existing unit tests and contributor
# extensions while avoiding the slow third-party SDK import entirely.
OpenAI = _OpenAICompatibleClient

# Preset providers (BOT_LLM_PROVIDER env). API keys resolved per provider.
# All providers use the OpenAI-compatible chat completions API unless noted.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "api_key_envs": "OPENAI_API_KEY,LLM_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_envs": "OPENAI_API_KEY,LLM_API_KEY",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_envs": "ANTHROPIC_API_KEY,LLM_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_envs": "GEMINI_API_KEY,LLM_API_KEY",
    },
    "deepseek": {
        "base_url": DEEPSEEK_BASE_URL,
        # Avoid using a key configured for an unrelated provider.
        "api_key_envs": "DEEPSEEK_API_KEY",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_envs": "GLM_API_KEY,ZHIPU_API_KEY,LLM_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_envs": "GROQ_API_KEY,LLM_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_envs": "MISTRAL_API_KEY,LLM_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_envs": "TOGETHER_API_KEY,LLM_API_KEY",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_key_envs": "XAI_API_KEY,LLM_API_KEY",
    },
    "custom": {
        "base_url": "",
        "model": "",
        "api_key_envs": "LLM_API_KEY,OPENAI_API_KEY,GEMINI_API_KEY,ANTHROPIC_API_KEY",
    },
}


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    provider: str
    thinking_enabled: bool = False
    reasoning_effort: str = DEFAULT_LLM_REASONING_EFFORT


@dataclass
class UsageMetadata:
    prompt_token_count: int
    candidates_token_count: int


@dataclass
class GenerateContentResponse:
    text: str
    usage_metadata: UsageMetadata


def _first_env_key(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def resolve_api_key(env_list: str | None = None) -> str | None:
    """Return the first configured LLM API key from the environment."""
    if env_list:
        return _first_env_key(*(n.strip() for n in env_list.split(",") if n.strip()))
    return _first_env_key(
        "OPENAI_API_KEY",
        "LLM_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "GLM_API_KEY",
        "ZHIPU_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "XAI_API_KEY",
    )


def _validate_provider_base_url(base_url: str) -> None:
    """Keep provider credentials off unencrypted non-local endpoints."""
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        has_url_credentials = parsed.username is not None or parsed.password is not None
    except ValueError as exc:
        raise ValueError("LLM provider base URL is malformed.") from exc

    if not hostname or has_url_credentials or parsed.query or parsed.fragment:
        raise ValueError(
            "LLM provider base URL must have a host and must not contain credentials, query, or fragment."
        )

    if parsed.scheme.lower() == "https":
        return

    is_loopback = hostname.lower() == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme.lower() == "http" and is_loopback:
        return

    raise ValueError(
        "LLM provider base URL must use HTTPS; HTTP is allowed only for a loopback local model."
    )


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, not {raw!r}")


def resolve_llm_config(
    provider: str | None = None,
) -> LLMConfig:
    """
    Resolve provider preset + env overrides into connection settings.

    BOT_LLM_PROVIDER selects a supported provider preset. OPENAI_BASE_URL or
    LLM_BASE_URL can override its endpoint. The provider and model are always
    selected explicitly; credentials come from the matching key variable or
    LLM_API_KEY where supported.
    """
    raw_provider = (
        provider
        if provider is not None
        else os.environ.get("BOT_LLM_PROVIDER", "")
    ).strip().lower()
    if not raw_provider:
        available = ", ".join(PROVIDER_PRESETS)
        raise ValueError(f"Set BOT_LLM_PROVIDER to a supported provider: {available}.")
    if raw_provider not in PROVIDER_PRESETS:
        available = ", ".join(PROVIDER_PRESETS)
        raise ValueError(
            f"Unknown LLM provider '{raw_provider}'. Choose one of: {available}."
        )

    preset = PROVIDER_PRESETS[raw_provider]
    api_key = resolve_api_key(preset.get("api_key_envs"))
    if not api_key:
        raise ValueError(
            f"Missing API key for LLM provider '{raw_provider}'. "
            f"Set one of: {preset.get('api_key_envs', 'LLM_API_KEY')}."
        )

    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or preset.get("base_url")
    ).strip()

    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise ValueError("Set LLM_MODEL to a model supported by the selected provider.")

    thinking_enabled = _parse_bool_env(
        "LLM_THINKING_ENABLED",
        False,
    )
    reasoning_effort = (
        os.environ.get("LLM_REASONING_EFFORT", DEFAULT_LLM_REASONING_EFFORT)
        .strip()
        .lower()
        or DEFAULT_LLM_REASONING_EFFORT
    )
    if reasoning_effort not in VALID_REASONING_EFFORTS:
        allowed = ", ".join(sorted(VALID_REASONING_EFFORTS))
        raise ValueError(
            f"LLM_REASONING_EFFORT must be one of: {allowed}; got {reasoning_effort!r}."
        )

    if not base_url:
        raise ValueError(
            "Set OPENAI_BASE_URL or LLM_BASE_URL for the selected custom provider."
        )

    _validate_provider_base_url(base_url)

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        provider=raw_provider,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
    )


def resolve_api_timeout_seconds() -> float:
    """Return a bounded timeout for one provider API request.

    A provider outage or stalled network connection must not hold a bot cycle
    indefinitely.  Keep the setting forgiving of invalid local input while
    enforcing a finite lower and upper bound.
    """
    raw_timeout = os.environ.get("LLM_API_TIMEOUT_SECONDS", "").strip()
    try:
        timeout = float(raw_timeout) if raw_timeout else DEFAULT_API_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        timeout = DEFAULT_API_TIMEOUT_SECONDS
    if timeout != timeout:  # NaN is not a usable timeout.
        timeout = DEFAULT_API_TIMEOUT_SECONDS
    return min(MAX_API_TIMEOUT_SECONDS, max(MIN_API_TIMEOUT_SECONDS, timeout))


def _prompt_from_contents(contents: str | list[Any]) -> str:
    if isinstance(contents, str):
        return contents

    parts: list[str] = []
    for item in contents:
        if isinstance(item, dict):
            for part in item.get("parts", []):
                parts.append(str(part))
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


class LLMQuotaExhausted(Exception):
    """Raised when a per-run LLM call budget is exhausted."""


class LLMIncompleteOutput(ValueError):
    """A provider returned no usable final answer or stopped mid-output."""

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


class LLMModel:
    """Thin wrapper: callers use generate_content like the old Gemini SDK."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        provider: str = "generic",
        thinking_enabled: bool = False,
        reasoning_effort: str = DEFAULT_LLM_REASONING_EFFORT,
    ):
        self._client = client
        self._model = model
        self._provider = provider.strip().lower()
        self._thinking_enabled = thinking_enabled
        self._reasoning_effort = reasoning_effort

    def generate_content(
        self,
        contents: str | list[Any],
        generation_config: dict[str, Any] | None = None,
    ) -> GenerateContentResponse:
        generation_config = generation_config or {}
        prompt = _prompt_from_contents(contents)
        thinking_enabled = bool(
            generation_config.get("thinking_enabled", self._thinking_enabled)
        )

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": generation_config.get("max_output_tokens", 1024),
        }
        if self._provider == "deepseek" and thinking_enabled:
            request_kwargs["thinking"] = {"type": "enabled"}
            request_kwargs["reasoning_effort"] = self._reasoning_effort
        elif self._provider == "deepseek":
            request_kwargs["thinking"] = {"type": "disabled"}
            request_kwargs["temperature"] = generation_config.get("temperature", 0.3)
        else:
            request_kwargs["temperature"] = generation_config.get("temperature", 0.3)

        response = self._client.chat.completions.create(**request_kwargs)

        if not response.choices:
            usage = getattr(response, "usage", None)
            logging.getLogger("optimist_prime").warning(
                "LLM completion returned no choices",
                extra={
                    "event_name": "llm.response.empty",
                    "component": "llm",
                    "provider": self._provider,
                    "model_name": self._model,
                    "choice_count": 0,
                    "failure_reason": "no_choices",
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                },
            )
            raise LLMIncompleteOutput(
                "LLM returned empty response (no choices)", reason="no_choices"
            )

        choice = response.choices[0].message
        text = (choice.content or "").strip()
        if not text:
            usage = getattr(response, "usage", None)
            logging.getLogger("optimist_prime").warning(
                "LLM completion had no final answer text",
                extra={
                    "event_name": "llm.response.empty",
                    "component": "llm",
                    "provider": self._provider,
                    "model_name": self._model,
                    "choice_count": len(response.choices),
                    "finish_reason": str(
                        getattr(response.choices[0], "finish_reason", "") or ""
                    )[:80],
                    "content_length": len(str(choice.content or "")),
                    "reasoning_content_length": int(
                        getattr(choice, "reasoning_content_length", 0) or 0
                    ),
                    "refusal_present": bool(getattr(choice, "refusal_present", False)),
                    "tool_call_count": int(
                        getattr(response.choices[0], "tool_call_count", 0) or 0
                    ),
                    "failure_reason": "empty_final_answer",
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                },
            )
            raise LLMIncompleteOutput(
                "LLM returned an empty final answer", reason="empty_final_answer"
            )

        finish_reason = str(
            getattr(response.choices[0], "finish_reason", "") or ""
        ).casefold()
        if finish_reason in {"length", "max_tokens"}:
            usage = getattr(response, "usage", None)
            logging.getLogger("optimist_prime").warning(
                "LLM completion reached its output limit before finishing",
                extra={
                    "event_name": "llm.response.truncated",
                    "component": "llm",
                    "provider": self._provider,
                    "model_name": self._model,
                    "finish_reason": finish_reason,
                    "content_length": len(text),
                    "reasoning_content_length": int(
                        getattr(choice, "reasoning_content_length", 0) or 0
                    ),
                    "failure_reason": "output_limit",
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                },
            )
            raise LLMIncompleteOutput(
                "LLM response reached its output limit before completion",
                reason="output_limit",
            )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return GenerateContentResponse(
            text=text,
            usage_metadata=UsageMetadata(
                prompt_token_count=input_tokens,
                candidates_token_count=output_tokens,
            ),
        )


class RateLimitedLLMModel:
    """Wraps an LLMModel with a hard per-run call cap."""

    def __init__(self, inner: LLMModel, max_calls: int):
        self._inner = inner
        self._max_calls = max_calls
        self.calls_used = 0

    def generate_content(
        self,
        contents: str | list[Any],
        generation_config: dict[str, Any] | None = None,
    ) -> GenerateContentResponse:
        if self.calls_used >= self._max_calls:
            raise LLMQuotaExhausted(
                f"Per-run LLM limit reached ({self._max_calls} call(s))"
            )
        # Count attempts, not only successful responses.  A provider outage
        # must not let a loop retry indefinitely and exceed the advertised
        # per-run API budget.
        self.calls_used += 1
        return self._inner.generate_content(contents, generation_config)


def wrap_with_rate_limit(model: LLMModel, max_calls: int | None) -> LLMModel | RateLimitedLLMModel:
    if max_calls is None:
        return model
    return RateLimitedLLMModel(model, max(0, max_calls))


def model_without_per_run_cap(model: LLMModel | RateLimitedLLMModel) -> LLMModel:
    """Return the configured provider client without the ordinary cycle call cap.

    Use only after current subreddit moderator status has been verified.
    Provider-side rate limits and outages still apply.
    """
    while isinstance(model, RateLimitedLLMModel):
        model = model._inner
    return model


# Cost estimation (configurable via env vars, defaults approximate MiniMax pricing)
_COST_INPUT_PER_M = float(os.environ.get("LLM_COST_INPUT_PER_MILLION_TOKENS", "0.10"))
_COST_OUTPUT_PER_M = float(os.environ.get("LLM_COST_OUTPUT_PER_MILLION_TOKENS", "0.40"))


def estimate_cost(token_info: dict) -> float:
    """Estimate cost in USD from token counts."""
    input_tokens = token_info.get("input_tokens", 0)
    output_tokens = token_info.get("output_tokens", 0)
    return (input_tokens * _COST_INPUT_PER_M + output_tokens * _COST_OUTPUT_PER_M) / 1_000_000


def extract_token_info(response) -> dict:
    """Extract token counts from an LLM response into a standard dict."""
    token_info = {
        "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") else 0,
        "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") else 0,
    }
    token_info["total_tokens"] = token_info["input_tokens"] + token_info["output_tokens"]
    token_info["cost"] = estimate_cost(token_info)
    return token_info


def create_llm_model(
    provider: str | None = None,
) -> LLMModel:
    """Build the configured LLM client from environment variables."""
    cfg = resolve_llm_config(provider)
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=resolve_api_timeout_seconds(),
    )
    return LLMModel(
        client,
        cfg.model,
        provider=cfg.provider,
        thinking_enabled=cfg.thinking_enabled,
        reasoning_effort=cfg.reasoning_effort,
    )
