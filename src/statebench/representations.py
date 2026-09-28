"""Representations of COMPACT_TRUTH, and the ablations applied to them.

Two kinds of difference, kept apart on purpose:

* **R1 / R2 / R3 differ only in rendering.** The underlying state is identical
  unablated COMPACT_TRUTH, so the executor behaves identically and the contrast
  isolates what the decision model can read out of the text.
* **The five ablations degrade the underlying state.** Both the rendered text and
  the executor see the degraded state, because a field that was never estimated
  is not available to the motion either. That is what makes an ablation a
  representation *error* rather than a prompt-formatting change.

Every transform returns the ablated current state and the correspondingly
ablated history, so a primitive's documented fallback chain cannot recover a
field through the back door.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .observations import HistoryBuffer
from .schema import SCHEMA_VERSION

DELAY_SECONDS = 0.5
NOISE_SIGMA_M = 0.02


@dataclass(frozen=True)
class Condition:
    name: str
    representation: str  # R1 | R2 | R3 | R4
    ablation: str | None = None
    #: Turn 3: append the derived predicate block (R4 family).
    predicates: bool = False
    #: Turn 3: a second ablation applied after the base one, so R4 can be
    #: delayed or noised without inventing a combined ablation name.
    extra_ablation: str | None = None
    #: Turn 3/3b: which named-condition instruction replaces next_action.
    #: "b" is Turn 3's R4b wording, "c" is Turn 3b's corrected recovery rule.
    #: Prompt engineering, labelled as such everywhere it appears.
    prompt_variant: str | None = None

    @property
    def is_ablation(self) -> bool:
        return self.ablation is not None

    @property
    def prompt_conditions(self) -> bool:
        return self.prompt_variant is not None

    @property
    def ablation_chain(self) -> tuple[str, ...]:
        return tuple(a for a in (self.ablation, self.extra_ablation) if a)


CONDITIONS: tuple[Condition, ...] = (
    Condition("R1_caption", "R1"),
    Condition("R2_typed", "R2"),
    Condition("R3_typed_history", "R3"),
    Condition("R3_no_verifier_fields", "R3", "drop_verifier_fields"),
    Condition("R3_drop_contact", "R3", "drop_contact"),
    Condition("R3_drop_target_rel_gripper", "R3", "drop_target_rel_gripper"),
    Condition("R3_drop_events_recent", "R3", "drop_events_recent"),
    Condition("R3_noise_target_rel_gripper_2cm", "R3", "noise_target_rel_gripper"),
    Condition("R3_delay_0p5s", "R3", "delay_all_fields"),
    # ---- Turn 3: predicates, built on R3_no_verifier_fields ----
    Condition("R4_predicates", "R4", "drop_verifier_fields", predicates=True),
    Condition("R4b_conditions_in_prompt", "R4", "drop_verifier_fields",
              predicates=True, prompt_variant="b"),
    Condition("R4_delay_0p5s", "R4", "delay_all_fields", predicates=True,
              extra_ablation="drop_verifier_fields"),
    Condition("R4_noise_2cm", "R4", "drop_verifier_fields", predicates=True,
              extra_ablation="noise_target_rel_gripper"),
    # ---- Turn 3b ----
    Condition("R4c_conditions_in_prompt_fixed", "R4", "drop_verifier_fields",
              predicates=True, prompt_variant="c"),
    Condition("R4c_delay_0p5s", "R4", "delay_all_fields", predicates=True,
              extra_ablation="drop_verifier_fields", prompt_variant="c"),
    Condition("R5_predicates_only", "R5", "drop_verifier_fields", predicates=True),
)

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


# --------------------------------------------------------------------- ablate
def _drop_provenance(state: dict[str, Any], path: str) -> None:
    for key in ("source", "t_obs"):
        state.get("provenance", {}).get(key, {}).pop(path, None)


def apply_ablation(
    state: dict[str, Any],
    ablation: str | None,
    *,
    t: float,
    history: HistoryBuffer,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if ablation is None:
        return copy.deepcopy(state)

    if ablation == "delay_all_fields":
        # Turn 3b clarification: inspect waits for a fresh observation. The
        # trigger is events_recent, already in the declared observation set. The
        # executor charges the wait to the episode's 30 s budget.
        events = state.get("events_recent") or []
        if (
            events
            and events[-1].get("event") == "inspect_done"
            and float(events[-1].get("t", -99)) >= float(t) - 0.05
        ):
            fresh = copy.deepcopy(state)
            fresh.setdefault("uncertainty", {})["estimate_age_s"] = 0.0
            return fresh
        stale = history.at_or_before(t - DELAY_SECONDS)
        out = copy.deepcopy(stale if stale is not None else state)
        observed_at = out.get("provenance", {}).get("t_obs", {})
        newest = max(observed_at.values()) if observed_at else t
        out.setdefault("uncertainty", {})["estimate_age_s"] = round(float(t) - float(newest), 3)
        return out

    out = copy.deepcopy(state)
    if ablation == "drop_contact":
        out.pop("contact", None)
        _drop_provenance(out, "contact.gripper_target")
        out.setdefault("uncertainty", {})["contact_observed"] = False
    elif ablation == "drop_target_rel_gripper":
        out.get("geometry", {}).pop("target_rel_gripper", None)
        _drop_provenance(out, "geometry.target_rel_gripper")
        out.setdefault("uncertainty", {})["target_visible"] = False
    elif ablation == "drop_verifier_fields":
        # Turn 2: remove the two relations that overlap terms of the physical
        # verifier, so step_complete has to be inferred from geometry and events.
        relations = out.get("relations") or {}
        relations.pop("target_in_receptacle", None)
        relations.pop("target_supported_by", None)
        _drop_provenance(out, "relations.target_in_receptacle")
        _drop_provenance(out, "relations.target_supported_by")
    elif ablation == "drop_events_recent":
        out.pop("events_recent", None)
        _drop_provenance(out, "events_recent")
    elif ablation == "noise_target_rel_gripper":
        block = out.get("geometry", {}).get("target_rel_gripper")
        if isinstance(block, dict):
            for axis in ("dx", "dy", "dz"):
                block[axis] = round(float(block[axis]) + float(rng.normal(0.0, NOISE_SIGMA_M)), 4)
            out["provenance"]["source"]["geometry.target_rel_gripper"] = "estimated"
    else:
        raise ValueError(f"unknown ablation {ablation!r}")
    return out


def ablate_history(
    history: HistoryBuffer,
    ablation: str | None,
    *,
    t: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """The history the executor may use, degraded the same way as the state."""
    if ablation is None:
        return [snapshot for _, snapshot in history.snapshots()]
    if ablation == "delay_all_fields":
        return [s for stamp, s in history.snapshots() if stamp <= t - DELAY_SECONDS + 1e-9]
    out = []
    for _, snapshot in history.snapshots():
        out.append(
            apply_ablation(snapshot, ablation, t=t, history=history, rng=rng)
        )
    return out


# --------------------------------------------------------------------- render
def _cm(value: float) -> str:
    return f"{float(value) * 100:.1f} cm"


def render_r1_caption(state: dict[str, Any]) -> str:
    """A 2-4 sentence scene description, templated from the same fields.

    Phrasing is regular so that a rule-following reader can recover the facts;
    the numbers are rounded to the nearest 0.1 cm and 1 degree, which is what a
    caption costs relative to the typed fields.
    """
    goal = state.get("goal", {})
    geometry = state.get("geometry", {})
    relations = state.get("relations", {})
    contact = state.get("contact", {})
    uncertainty = state.get("uncertainty", {})
    target = goal.get("target_object_id", "target")
    receptacle = goal.get("receptacle_id", "receptacle")

    sentences: list[str] = [f"Task: {goal.get('instruction', '')}".strip()]

    rel = geometry.get("target_rel_gripper")
    if isinstance(rel, dict):
        yaw_deg = math.degrees(float(rel.get("dyaw_rad", 0.0)))
        sentences.append(
            f"The {target} is offset from the gripper by dx {_cm(rel['dx'])}, "
            f"dy {_cm(rel['dy'])}, dz {_cm(rel['dz'])} in the robot base frame, "
            f"and its faces are turned {yaw_deg:.0f} degrees from the gripper's finger axis."
        )
    else:
        sentences.append(
            f"The position of the {target} relative to the gripper is not available."
        )

    opening = geometry.get("gripper_opening_m")
    rel_r = geometry.get("target_rel_receptacle")
    where = relations.get("target_supported_by", "unknown")
    touch = contact.get("gripper_target", "unknown")
    moving = relations.get("target_moving_with_gripper", "unknown")
    above = relations.get("gripper_above_target")
    inside = relations.get("target_in_receptacle")
    piece = (
        f"The {target} is resting on the {where}" if where in ("table", "receptacle")
        else f"The {target} is supported by the {where}"
    )
    if touch == "unknown":
        piece += ", whether the gripper is touching it is not observed"
    else:
        piece += (
            ", the gripper is touching it" if touch == "contact"
            else ", the gripper is not touching it"
        )
    if moving == "unknown":
        piece += ", and whether it would move with the gripper is not yet established"
    else:
        piece += (
            ", and it is moving with the gripper" if moving
            else ", and it is not moving with the gripper"
        )
    if opening is not None:
        piece += f"; the gripper is open {_cm(opening)}"
    if above is not None:
        piece += f" and is {'directly above' if above else 'not above'} the {target}"
    sentences.append(piece + ".")

    tail = (
        f"The {target} is {'already inside' if inside else 'not yet inside'} the {receptacle}"
    )
    if isinstance(rel_r, dict):
        tail += (
            f", which lies dx {_cm(rel_r['dx'])}, dy {_cm(rel_r['dy'])}, "
            f"dz {_cm(rel_r['dz'])} from the {target}"
        )
    age = uncertainty.get("estimate_age_s")
    if age:
        tail += f"; these readings are {float(age):.1f} s old"
    events = state.get("events_recent")
    if events:
        recent = ", ".join(f"{e['event']} at t={e['t']:.1f}s" for e in events[-4:])
        tail += f". Recently: {recent}"
    sentences.append(tail + ".")
    sentences.append(
        "Available actions: " + ", ".join(state.get("actions_available", [])) + "."
    )
    return " ".join(s for s in sentences if s)


def render_r2_typed(state: dict[str, Any]) -> str:
    """Geometry and relations only: no events, no uncertainty, no contact block."""
    trimmed = {
        "goal": state.get("goal", {}),
        "geometry": state.get("geometry", {}),
        "relations": state.get("relations", {}),
        "actions_available": state.get("actions_available", []),
        "provenance": {
            "source": {
                k: v
                for k, v in state.get("provenance", {}).get("source", {}).items()
                if k.startswith(("geometry.", "relations."))
            },
            "t_obs": {
                k: v
                for k, v in state.get("provenance", {}).get("t_obs", {}).items()
                if k.startswith(("geometry.", "relations."))
            },
            "schema_version": SCHEMA_VERSION,
        },
    }
    return json.dumps(trimmed, indent=2, sort_keys=True)


def render_r3_typed_history(state: dict[str, Any]) -> str:
    """The full schema, including events_recent and uncertainty."""
    return json.dumps(state, indent=2, sort_keys=True)


def render_r4_predicates(state: dict[str, Any]) -> str:
    """R3 JSON plus the derived predicate values.

    The predicate block's ``formulas`` and ``derived_from`` entries are static -
    byte-identical on every call - so they are kept in the state dict for the
    provenance check but trimmed from the rendered text rather than re-sent
    thousands of times. What the model sees is the values plus ``source:
    derived``.
    """
    trimmed = copy.deepcopy(state)
    block = trimmed.get("predicates")
    if isinstance(block, dict):
        trimmed["predicates"] = {
            "source": block.get("source", "derived"),
            **(block.get("values") or {}),
        }
    return json.dumps(trimmed, indent=2, sort_keys=True)


def render_r5_predicates_only(state: dict[str, Any]) -> str:
    """R4 with the numeric ``geometry`` block dropped from the TEXT only.

    The executor still receives the full R4 state (``prepare`` returns the state
    and the text separately), because R5 asks what Jev reads, not what the arm
    can physically do. Dropping geometry from the executor as well would make
    every primitive fall back to the task prior and would measure the executor's
    fallback policy instead.
    """
    trimmed = copy.deepcopy(state)
    trimmed.pop("geometry", None)
    for key in ("source", "t_obs"):
        block = (trimmed.get("provenance") or {}).get(key) or {}
        for path in [k for k in block if k.startswith("geometry.")]:
            block.pop(path, None)
    predicates = trimmed.get("predicates")
    if isinstance(predicates, dict):
        trimmed["predicates"] = {
            "source": predicates.get("source", "derived"),
            **(predicates.get("values") or {}),
        }
    return json.dumps(trimmed, indent=2, sort_keys=True)


RENDERERS = {
    "R1": render_r1_caption,
    "R2": render_r2_typed,
    "R3": render_r3_typed_history,
    "R4": render_r4_predicates,
    "R5": render_r5_predicates_only,
}


def render(condition: Condition, state: dict[str, Any]) -> str:
    return RENDERERS[condition.representation](state)


def prepare(
    condition: Condition,
    compact: dict[str, Any],
    history: HistoryBuffer,
    t: float,
    rng: np.random.Generator,
    proprio: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Return (state the executor may use, history it may use, text for the model)."""
    state = compact
    for ablation in condition.ablation_chain:
        state = apply_ablation(state, ablation, t=t, history=history, rng=rng)
    if not condition.ablation_chain:
        state = apply_ablation(state, None, t=t, history=history, rng=rng)

    hist = ablate_history(history, condition.ablation, t=t, rng=rng)
    if condition.extra_ablation:
        hist = [
            apply_ablation(s, condition.extra_ablation, t=t, history=history, rng=rng)
            for s in hist
        ]

    if condition.predicates:
        from .predicates import build_predicates

        state["predicates"] = build_predicates(state, proprio, hist)
        state.setdefault("provenance", {}).setdefault("source", {})["predicates"] = "derived"
        state["provenance"].setdefault("t_obs", {})["predicates"] = round(float(t), 3)
    return state, hist, render(condition, state)
