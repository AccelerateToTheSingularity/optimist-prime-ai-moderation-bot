"""Small OpenRouter Decisions API client for typed Jev decisions."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = "~typesafe/jev-latest"


class JevDecisionClient:
    """Issue typed Jev decisions without bringing in another SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_JEV_MODEL,
        max_calls: int = 30,
        timeout: float = 15.0,
        opener=None,
    ):
        self._api_key = (api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.model = model
        self.max_calls = max(0, int(max_calls))
        self.timeout = max(3.0, float(timeout))
        self.calls_used = 0
        self.last_usage = {"input_tokens": 0, "cost": 0.0}
        self.last_error: dict[str, Any] | None = None
        self.served_model: str | None = None
        self._opener = opener or urlopen

    @property
    def ready(self) -> bool:
        return bool(self._api_key)

    def probability_yes(
        self,
        *,
        question: str,
        state: dict[str, Any],
    ) -> float | None:
        """Return Jev's yes probability, or ``None`` if unavailable.

        ``None`` is intentionally not interpreted as a negative answer; callers
        should retain their useful context when the gate cannot decide.
        """
        result = self.decide_many(
            state=state,
            questions={
                "needs_context": {
                    "type": "noul",
                    "instructions": question,
                    "criteria": {
                        "true": "The message cannot be answered well without the surrounding post or discussion.",
                        "false": "The message can be answered well from the message itself.",
                    },
                }
            },
        )
        return result.get("needs_context")

    def _request_answers(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Submit typed questions once and return Jev's structured answers."""
        if not self.ready or self.calls_used >= self.max_calls or not questions:
            self.last_usage = {"input_tokens": 0, "cost": 0.0}
            self.last_error = None
            return {}
        self.calls_used += 1
        self.last_usage = {"input_tokens": 0, "cost": 0.0}
        self.last_error = None
        payload = {"model": self.model, "state": state, "questions": questions}
        request = Request(
            DECISIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            usage = result.get("usage", {})
            self.last_usage = {
                "input_tokens": max(0, int(usage.get("input_tokens", 0) or 0)),
                "cost": max(0.0, float(usage.get("cost", 0.0) or 0.0)),
            }
            self.served_model = result.get("model") or self.model
            answers = result.get("answers", {})
            return answers if isinstance(answers, dict) else {}
        except Exception as exc:
            # Never log request data or authorization headers. Callers treat
            # an unavailable structured decision as no usable result.
            error = {"type": type(exc).__name__}
            if isinstance(exc, HTTPError):
                error["status_code"] = int(exc.code)
                error["reason"] = " ".join(str(exc.reason or "").split())[:100]
            self.last_error = error
            status = (
                f", HTTP {error['status_code']} {error.get('reason', '')}".rstrip()
                if "status_code" in error else ""
            )
            print(f"    Jev decision unavailable ({type(exc).__name__}{status})")
        return {}

    def decide_many(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate Noul questions and return Jev's yes probabilities."""
        answers = self._request_answers(state=state, questions=questions)
        probabilities = {}
        for name, answer in answers.items():
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                continue
            try:
                probability = float(answer.get("noul"))
            except (TypeError, ValueError):
                continue
            if 0.0 <= probability <= 1.0:
                probabilities[name] = probability
        return probabilities

    def choose_many(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        """Evaluate Choice questions and return only their selected option labels."""
        return {
            name: detail["choice"]
            for name, detail in self.choose_many_detailed(state=state, questions=questions).items()
        }

    def choose_many_detailed(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Return each valid selected Choice label and its provider-reported evidence."""
        answers = self._request_answers(state=state, questions=questions)
        choices: dict[str, dict[str, Any]] = {}
        for name, answer in answers.items():
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                continue
            question = questions.get(name, {})
            if question.get("type") != "choice":
                continue
            criteria = question.get("criteria")
            if not isinstance(criteria, dict):
                continue
            choice = answer.get("choice")
            if isinstance(choice, str) and choice in criteria:
                probabilities = answer.get("probabilities", {})
                valid_probabilities = {}
                if isinstance(probabilities, dict):
                    for label, raw_probability in probabilities.items():
                        if label not in criteria:
                            continue
                        try:
                            probability = float(raw_probability)
                        except (TypeError, ValueError):
                            continue
                        if 0.0 <= probability <= 1.0:
                            valid_probabilities[label] = probability
                detail: dict[str, Any] = {
                    "choice": choice,
                    "probabilities": valid_probabilities,
                }
                try:
                    confidence = float(answer.get("confidence"))
                except (TypeError, ValueError):
                    confidence = None
                if confidence is not None and 0.0 <= confidence <= 1.0:
                    detail["confidence"] = confidence
                choices[name] = detail
        return choices
