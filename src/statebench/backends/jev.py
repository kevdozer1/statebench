"""TypeSafe Jev through the System One API.

Request shape confirmed against the TypeSafe docs (quickstart, and the ``choice``,
``noul`` and ``score`` primitive pages), not assumed::

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <key>
    {"model": "jev-latest", "state": <string or object>, "questions": {...}}

One request carries all three questions: ``next_action`` as a ``choice`` over the
seven primitives, ``object_held`` and ``step_complete`` as ``noul``.

Response handling follows the documented shapes:

* a ``choice`` answer returns ``choice``, ``probabilities`` (summing to 1) and
  ``confidence``;
* a ``noul`` answer returns a single float in [0, 1] and, per the docs, "there is
  no separate ``confidence`` value for a Noul". The float is the probability of
  the statement being true, so it maps onto our yes/no option pair as
  ``{"yes": noul, "no": 1 - noul}``.

``confidence`` is recorded next to the distribution but never treated as
calibrated: the docs define it as "a number from 0 to 1 computed from how
``probabilities`` is spread", which is a property of the distribution's shape and
needs application-specific validation before it means anything physical.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import env as config_env
from ..questions import (
    ACTION_CRITERIA,
    INSTRUCTIONS,
    NEXT_ACTION,
    NOUL_CRITERIA,
    OBJECT_HELD,
    STEP_COMPLETE,
    SYSTEM_PREAMBLE,
)
from .base import (
    Decision,
    DecisionBackend,
    answer_from,
    load_secret,
    register_backend,
)
from .http import post_json

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
FALLBACK_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "jev-latest"
FALLBACK_MODEL = "typesafe/jev-1.13"
CREDENTIAL_VARS = ["TYPESAFE_JEV_API_KEY", "TYPESAFE_API_KEY"]
FALLBACK_CREDENTIAL_VARS = ["OPENROUTER_API_KEY"]


def build_questions(
    options: dict[str, list[str]], instruction_override: str | None = None
) -> dict[str, Any]:
    """The three questions in TypeSafe primitive form.

    Wording is frozen except for ``instruction_override``, which the R4b
    prompt-conditioned arm uses to replace the ``next_action`` instruction with
    one that names the decision conditions. The two noul questions keep their
    frozen wording in every condition.
    """
    payload: dict[str, Any] = {}
    if NEXT_ACTION in options:
        next_action_text = instruction_override or INSTRUCTIONS[NEXT_ACTION]
        payload[NEXT_ACTION] = {
            "type": "choice",
            "instructions": f"{SYSTEM_PREAMBLE}\n\n{next_action_text}",
            "criteria": {o: ACTION_CRITERIA[o] for o in options[NEXT_ACTION]},
        }
    for name in (OBJECT_HELD, STEP_COMPLETE):
        if name in options:
            payload[name] = {
                "type": "noul",
                "instructions": f"{SYSTEM_PREAMBLE}\n\nStatement: {INSTRUCTIONS[name]}",
                "criteria": dict(NOUL_CRITERIA),
            }
    return payload


def _noul_to_distribution(value: float, options: list[str]) -> dict[str, float]:
    p = min(1.0, max(0.0, float(value)))
    yes, no = options[0], options[1]
    return {yes: p, no: 1.0 - p}


class JevBackend(DecisionBackend):
    """TypeSafe System One. Falls back to OpenRouter's decisions endpoint only if
    the primary endpoint refuses the credential or does not exist."""

    name = "jev"
    remote = True

    def __init__(self, model: str | None = None, *, state_as_object: bool = False):
        super().__init__(model=model or config_env("STATEBENCH_JEV_MODEL", DEFAULT_MODEL))
        self.url = config_env("STATEBENCH_JEV_URL", DEFAULT_URL)
        self.key = load_secret(CREDENTIAL_VARS, "TypeSafe Jev")
        self.state_as_object = state_as_object
        self.using_fallback = False
        self.latencies: list[float] = []
        self.input_tokens = 0
        self.output_tokens = 0

    # ------------------------------------------------------------------ state
    def _state_payload(self, state_text: str) -> Any:
        """A plain string by default.

        TypeSafe's docs say to "use an object for most requests so each part of
        the state has a descriptive name". Turn 2 keeps the string form for the
        main grid so that Jev and the OpenRouter backends receive byte-identical
        state, and tests the object form as a separate condition.
        """
        if not self.state_as_object:
            return state_text
        return {
            "robot_state": state_text,
            "format_note": (
                "The robot state above is the rendered representation under test. "
                "It is the only information about the scene."
            ),
        }

    def use_fallback(self) -> None:
        self.url = config_env("STATEBENCH_JEV_FALLBACK_URL", FALLBACK_URL)
        self.model = FALLBACK_MODEL
        self.key = load_secret(FALLBACK_CREDENTIAL_VARS, "OpenRouter (Jev fallback)")
        self.using_fallback = True

    # ----------------------------------------------------------------- decide
    def decide(
        self,
        state_text: str,
        questions: dict[str, list[str]],
        instruction_override: str | None = None,
    ) -> Decision:
        body = {
            "model": self.model,
            "state": self._state_payload(state_text),
            "questions": build_questions(questions, instruction_override),
        }
        result = post_json(self.url, body, self.key, timeout=90.0)
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
        confidences: dict[str, float] = {}
        error = None
        try:
            returned = raw["answers"]
            for name, opts in questions.items():
                block = returned[name]
                kind = block.get("type")
                if kind == "noul":
                    answers[name] = answer_from(
                        _noul_to_distribution(block["noul"], opts), opts
                    )
                else:
                    answers[name] = answer_from(
                        block["probabilities"], opts, block.get("choice")
                    )
                    if block.get("confidence") is not None:
                        confidences[name] = float(block["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        self.latencies.append(result.latency_seconds)
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        return self.account(
            Decision(
                answers=answers,
                raw=raw,
                backend=self.name,
                model=raw.get("model", self.model),
                latency_seconds=result.latency_seconds,
                tokens_prompt=input_tokens,
                tokens_completion=output_tokens,
                cost_usd=usage.get("cost"),
                error=error,
                extra={
                    "http_status": result.status,
                    "retries": result.retries,
                    "confidence": confidences,
                    "state_as_object": self.state_as_object,
                    "prompt_conditioned": instruction_override is not None,
                    "endpoint": self.url,
                },
            )
        )

    def usage(self) -> dict[str, Any]:
        base = super().usage()
        base.update(
            endpoint=self.url,
            using_fallback=self.using_fallback,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latencies=self.latencies,
        )
        return base


class JevObjectStateBackend(JevBackend):
    """Phase 4 variant: the same state passed as a named object."""

    name = "jev_object_state"

    def __init__(self, model: str | None = None):
        super().__init__(model=model, state_as_object=True)


def smoke_request(backend: JevBackend) -> dict[str, Any]:
    """One real request with a three-field state and a single noul question."""
    state = {
        "gripper_opening_m": 0.089,
        "target_rel_gripper": {"dx": 0.10, "dy": -0.13, "dz": -0.22},
        "contact": "no_contact",
    }
    body = {
        "model": backend.model,
        "state": state,
        "questions": {
            "object_held": {
                "type": "noul",
                "instructions": (
                    f"{SYSTEM_PREAMBLE}\n\nStatement: {INSTRUCTIONS[OBJECT_HELD]}"
                ),
                "criteria": dict(NOUL_CRITERIA),
            }
        },
    }
    started = time.perf_counter()
    result = post_json(backend.url, body, backend.key, timeout=90.0)
    elapsed = time.perf_counter() - started
    payload = result.payload or {}
    answer = (payload.get("answers") or {}).get("object_held") or {}
    return {
        "endpoint": backend.url,
        "model_requested": backend.model,
        "model_returned": payload.get("model"),
        "http_status": result.status,
        "ok": result.ok,
        "latency_seconds": round(elapsed, 3),
        "attempts": result.attempts,
        "error": result.error,
        "answer_type": answer.get("type"),
        "noul": answer.get("noul"),
        "noul_in_unit_interval": (
            isinstance(answer.get("noul"), (int, float)) and 0.0 <= float(answer["noul"]) <= 1.0
        ),
        "response_keys": sorted(payload.keys()),
        "answer_keys": sorted(answer.keys()),
        "usage": payload.get("usage"),
    }


register_backend("jev", JevBackend)
register_backend("jev_object_state", JevObjectStateBackend)
