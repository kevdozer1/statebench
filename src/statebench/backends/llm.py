"""OpenRouter chat completions with structured output over the same options.

Two registered backends, differing only in the model slug and (by
pre-registration) in how much of the grid they run:

``llm_cheap``   the most capable Haiku-class or Flash-class slug with
                ``structured_outputs`` support that fits the spend cap.
``llm_strong``  the Sonnet-class slug.

Both send the frozen preamble and the frozen instruction strings from
``questions.py``, byte-identical to what the Jev backend sends, and both are
constrained by a JSON schema whose enums are exactly the option lists. No
per-backend prompt tuning.

Cost is taken from OpenRouter's own ``usage.cost`` (requested with
``usage: {"include": true}``) and cross-checked against the catalogue list price
recorded at selection time.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import env as config_env
from ..questions import INSTRUCTIONS, NEXT_ACTION, SYSTEM_PREAMBLE, ACTION_CRITERIA
from .base import (
    Decision,
    DecisionBackend,
    answer_from,
    load_secret,
    register_backend,
)
from .http import post_json

DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
CREDENTIAL_VARS = ["OPENROUTER_API_KEY"]


def _schema(options: dict[str, list[str]]) -> dict[str, Any]:
    per_question = {}
    for name, opts in options.items():
        per_question[name] = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": list(opts)},
                "probabilities": {
                    "type": "object",
                    "properties": {o: {"type": "number"} for o in opts},
                    "required": list(opts),
                    "additionalProperties": False,
                },
            },
            "required": ["choice", "probabilities"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "object",
                "properties": per_question,
                "required": list(options),
                "additionalProperties": False,
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def build_user_message(
    state_text: str,
    options: dict[str, list[str]],
    instruction_override: str | None = None,
) -> str:
    """The frozen instructions plus the state, as plain text.

    The state is inserted verbatim rather than nested inside another JSON string.
    Embedding it as a JSON string value escaped every quote and newline and more
    than doubled the prompt-token count for the same content (3057 tokens against
    about 1400 on the same state). This is a transport fix; the preamble, the
    instruction strings and the option meanings are unchanged and identical
    across every backend.
    """
    lines = ["STATE", "-----", state_text, "", "QUESTIONS", "---------"]
    for name, opts in options.items():
        text = (
            instruction_override
            if (name == NEXT_ACTION and instruction_override)
            else INSTRUCTIONS[name]
        )
        lines.append(f"{name}: {text}")
        lines.append("  options: " + " | ".join(opts))
        if name == NEXT_ACTION:
            for option in opts:
                lines.append(f"    {option}: {ACTION_CRITERIA[option]}")
        lines.append("")
    lines.append(
        "ANSWER FORMAT: for every question give a probability for each option. The "
        "probabilities must be finite, between 0 and 1, and sum to 1, and `choice` "
        "must be an option with the highest probability."
    )
    return "\n".join(lines)


class OpenRouterBackend(DecisionBackend):
    name = "llm"
    remote = True
    model_env_var = "STATEBENCH_LLM_MODEL"
    default_model = "anthropic/claude-sonnet-5"

    def __init__(self, model: str | None = None):
        super().__init__(
            model=model or config_env(self.model_env_var, self.default_model)
        )
        self.url = config_env("STATEBENCH_LLM_URL", DEFAULT_URL)
        self.key = load_secret(CREDENTIAL_VARS, "OpenRouter")
        self.latencies: list[float] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def decide(
        self,
        state_text: str,
        questions: dict[str, list[str]],
        instruction_override: str | None = None,
    ) -> Decision:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1500,
            # Reasoning tokens are disabled for every OpenRouter backend alike.
            # Left on, gemini-3.8-flash spent 327 of 535 completion tokens on
            # reasoning and truncated the JSON on 19 of 24 calls in the Phase 0
            # smoke test. This is a transport setting, not prompt tuning: the
            # preamble and instruction strings are unchanged and identical
            # across backends.
            "reasoning": {"enabled": False},
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "statebench_decisions",
                    "strict": True,
                    "schema": _schema(questions),
                },
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PREAMBLE},
                {"role": "user", "content": build_user_message(
                    state_text, questions, instruction_override)},
            ],
        }
        result = post_json(self.url, body, self.key, timeout=120.0)
        if not result.ok:
            return self.account(
                Decision(
                    answers={}, raw={"attempts": result.attempt_log}, backend=self.name,
                    model=self.model, latency_seconds=result.latency_seconds,
                    error=result.error,
                    extra={"http_status": result.status, "retries": result.retries},
                )
            )
        raw = result.payload or {}
        usage = raw.get("usage") or {}
        answers: dict[str, Any] = {}
        error = None
        try:
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            for name, opts in questions.items():
                block = parsed["answers"][name]
                answers[name] = answer_from(
                    block["probabilities"], opts, block.get("choice")
                )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.latencies.append(result.latency_seconds)
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        return self.account(
            Decision(
                answers=answers,
                raw=raw,
                backend=self.name,
                model=raw.get("model", self.model),
                latency_seconds=result.latency_seconds,
                tokens_prompt=usage.get("prompt_tokens"),
                tokens_completion=usage.get("completion_tokens"),
                cost_usd=usage.get("cost"),
                error=error,
                extra={
                    "http_status": result.status,
                    "retries": result.retries,
                    "response_id": raw.get("id"),
                    "prompt_conditioned": instruction_override is not None,
                },
            )
        )

    def usage(self) -> dict[str, Any]:
        base = super().usage()
        base.update(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latencies=self.latencies,
        )
        return base


class CheapBackend(OpenRouterBackend):
    """Flash-class arm.

    Selection criterion, fixed before the grid: among catalogue slugs with
    ``structured_outputs`` support in the Haiku or Flash tier, take the
    highest-list-price one (price being the only capability proxy the catalogue
    offers) whose projected grid cost still leaves the full ``llm_strong`` arm
    inside the 4.50 USD cap. ``google/gemini-3.1-flash-lite`` projected at
    3.40 USD and would have forced dropping ``llm_strong``; this one projects
    well under.
    """

    name = "llm_cheap"
    model_env_var = "STATEBENCH_LLM_CHEAP_MODEL"
    default_model = "google/gemini-2.5-flash-lite"


class StrongBackend(OpenRouterBackend):
    name = "llm_strong"
    model_env_var = "STATEBENCH_LLM_STRONG_MODEL"
    default_model = "anthropic/claude-sonnet-5"


def smoke_request(backend: OpenRouterBackend) -> dict[str, Any]:
    """One real request with the same three-field state and the same question."""
    state_text = json.dumps({
        "gripper_opening_m": 0.089,
        "target_rel_gripper": {"dx": 0.10, "dy": -0.13, "dz": -0.22},
        "contact": "no_contact",
    })
    options = {"object_held": ["yes", "no"]}
    body = {
        "model": backend.model,
        "temperature": 0,
        "max_tokens": 400,
        "usage": {"include": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "statebench_smoke", "strict": True,
                            "schema": _schema(options)},
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PREAMBLE},
            {"role": "user", "content": build_user_message(state_text, options)},
        ],
    }
    started = time.perf_counter()
    result = post_json(backend.url, body, backend.key, timeout=120.0)
    elapsed = time.perf_counter() - started
    payload = result.payload or {}
    usage = payload.get("usage") or {}
    parsed = None
    try:
        parsed = json.loads(payload["choices"][0]["message"]["content"])
    except Exception:  # noqa: BLE001
        pass
    return {
        "backend": backend.name,
        "model_requested": backend.model,
        "model_returned": payload.get("model"),
        "http_status": result.status,
        "ok": result.ok,
        "latency_seconds": round(elapsed, 3),
        "attempts": result.attempts,
        "error": result.error,
        "parsed": parsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
    }


register_backend("llm", OpenRouterBackend)
register_backend("llm_cheap", CheapBackend)
register_backend("llm_strong", StrongBackend)
