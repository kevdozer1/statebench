"""Decision-backend abstraction.

Shaped after ``robolabel``'s provider layer: a name-to-class registry so adding a
backend is one file, a response record that carries measured latency and token
usage as first-class fields rather than a vendor claim, and a credential loader
that names the exact environment variable it wanted when it is missing.

The interface is::

    decide(state_text: str, questions: dict) -> dict

returning, for each question, a distribution over its options plus the raw
response. Backends see only the rendered representation text. Nothing else.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, env as config_env

NEXT_ACTION = "next_action"
OBJECT_HELD = "object_held"
STEP_COMPLETE = "step_complete"

YES_NO = ("yes", "no")

QUESTION_TEXT = {
    NEXT_ACTION: "Choose the next action for the robot from the options.",
    OBJECT_HELD: "The target object will stay with the gripper if the arm moves.",
    STEP_COMPLETE: "The current goal is satisfied.",
}


class MissingCredentialError(RuntimeError):
    """Raised when a backend's credential is absent; names the variable to set."""


def load_secret(names: list[str], label: str) -> str:
    for name in names:
        value = os.environ.get(name) or config_env(name)
        if value:
            return value
    raise MissingCredentialError(
        f"{label} credential not found. Set {names[0]} in the process environment "
        f"or add {names[0]}=... to {PROJECT_ROOT / '.env'}."
    )


@dataclass
class Answer:
    """One question's answer: a distribution over its options plus the choice."""

    choice: str
    probabilities: dict[str, float]

    def probability_of(self, option: str) -> float:
        return float(self.probabilities.get(option, 0.0))


@dataclass
class Decision:
    """One ``decide`` call: answers plus the measured cost of getting them."""

    answers: dict[str, Answer]
    raw: Any = None
    backend: str = ""
    model: str = ""
    latency_seconds: float = 0.0
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 4),
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "answers": {
                name: {"choice": a.choice, "probabilities": a.probabilities}
                for name, a in self.answers.items()
            },
            **self.extra,
        }


def validate_distribution(probabilities: dict[str, float], options: list[str]) -> dict[str, float]:
    if set(probabilities) != set(options):
        raise ValueError(f"distribution labels {sorted(probabilities)} != options {sorted(options)}")
    for value in probabilities.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("non-numeric probability")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("probability outside [0, 1]")
    total = sum(float(v) for v in probabilities.values())
    if abs(total - 1.0) > 0.025:
        raise ValueError(f"probabilities sum to {total:.4f}")
    return {k: float(v) / total for k, v in probabilities.items()}


def answer_from(probabilities: dict[str, float], options: list[str], choice: str | None = None) -> Answer:
    normalized = validate_distribution(probabilities, options)
    best = max(normalized, key=lambda k: normalized[k])
    if choice is None:
        choice = best
    if choice not in options:
        raise ValueError(f"choice {choice!r} not among options")
    if normalized[choice] + 1e-6 < normalized[best]:
        raise ValueError("choice does not match the distribution")
    return Answer(choice=choice, probabilities=normalized)


def point_mass(choice: str, options: list[str], confidence: float = 1.0) -> Answer:
    """A degenerate distribution, used by the deterministic rule backend."""
    confidence = min(1.0, max(1.0 / len(options), float(confidence)))
    spill = (1.0 - confidence) / max(1, len(options) - 1)
    probabilities = {o: (confidence if o == choice else spill) for o in options}
    return answer_from(probabilities, options, choice)


class DecisionBackend(ABC):
    """Answers the turn's three questions from a rendered state representation."""

    name: str = "backend"
    #: Backends that call a remote service report their per-call cost honestly.
    remote: bool = False

    def __init__(self, model: str | None = None):
        self.model = model or self.name
        self.calls = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0

    @abstractmethod
    def decide(
        self,
        state_text: str,
        questions: dict[str, list[str]],
        instruction_override: str | None = None,
    ) -> Decision:
        """Answer every question in ``questions`` from ``state_text`` alone.

        ``instruction_override`` replaces the frozen ``next_action`` instruction
        for the R4b prompt-conditioned arm only. It is None everywhere else, and
        every table that shows an R4b row labels it as prompt engineering.
        """

    def account(self, decision: Decision) -> Decision:
        self.calls += 1
        self.total_latency += decision.latency_seconds
        self.total_cost += decision.cost_usd or 0.0
        self.total_tokens_prompt += decision.tokens_prompt or 0
        self.total_tokens_completion += decision.tokens_completion or 0
        return decision

    def usage(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "calls": self.calls,
            "total_latency_seconds": round(self.total_latency, 3),
            "mean_latency_seconds": round(self.total_latency / self.calls, 4) if self.calls else None,
            "total_tokens_prompt": self.total_tokens_prompt or None,
            "total_tokens_completion": self.total_tokens_completion or None,
            "total_cost_usd": round(self.total_cost, 6) if self.remote else None,
        }


_REGISTRY: dict[str, type[DecisionBackend]] = {}


def register_backend(name: str, cls: type[DecisionBackend]) -> None:
    _REGISTRY[name.strip().lower()] = cls


def available_backends() -> list[str]:
    from . import frozen_head, jev, llm, rules  # noqa: F401  (registration)

    return sorted(_REGISTRY)


def build_backend(name: str, model: str | None = None) -> DecisionBackend:
    from . import jev, llm, rules  # noqa: F401

    try:  # optional: needs scikit-learn and a fitted bundle
        from . import frozen_head  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key](model=model)


def default_questions() -> dict[str, list[str]]:
    """Option lists come from the frozen ``questions`` module (Turn 2)."""
    from ..questions import options

    return options()


def receipts_dir(run: str = "turn1") -> Path:
    from ..config import run_dir

    path = run_dir(run) / "receipts"
    path.mkdir(parents=True, exist_ok=True)
    return path
