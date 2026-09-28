"""Deterministic rule backend: the sanity check on the schema.

It reads the rendered representation text exactly as a model would - JSON for R2
and R3, the templated prose for R1 - recovers what it can, and applies the same
explicit conditions as the reference controller, expressed in the fields the
compact schema actually provides.

That last clause is the point of the check. The reference controller has full
simulator state; this backend has only the schema. Anywhere the reference can
decide and this backend cannot, the schema is missing something.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..verify import RELEASE_OPENING_M
from .base import (
    NEXT_ACTION,
    OBJECT_HELD,
    STEP_COMPLETE,
    Answer,
    Decision,
    DecisionBackend,
    point_mass,
    register_backend,
)

ALIGN_Z_M = 0.015
ALIGN_YAW_RAD = 0.10
OVER_RECEPTACLE_M = 0.030
PLACE_CLEARANCE_M = 0.060

_NUM = r"(-?\d+(?:\.\d+)?)"


def _cm_to_m(text: str | None) -> float | None:
    return None if text is None else round(float(text) / 100.0, 5)


def parse_caption(text: str) -> dict[str, Any]:
    """Recover the schema facts from the R1 template. Missing means missing."""
    out: dict[str, Any] = {}
    m = re.search(
        rf"offset from the gripper by dx {_NUM} cm, dy {_NUM} cm, dz {_NUM} cm", text
    )
    if m:
        out["dx"], out["dy"], out["dz"] = (_cm_to_m(g) for g in m.groups())
    m = re.search(rf"turned {_NUM} degrees from the gripper's finger axis", text)
    if m:
        out["dyaw_rad"] = round(float(m.group(1)) * 3.141592653589793 / 180.0, 4)
    m = re.search(
        rf"which lies dx {_NUM} cm, dy {_NUM} cm, dz {_NUM} cm from the", text
    )
    if m:
        out["rdx"], out["rdy"], out["rdz"] = (_cm_to_m(g) for g in m.groups())
    m = re.search(rf"the gripper is open {_NUM} cm", text)
    if m:
        out["gripper_opening_m"] = _cm_to_m(m.group(1))
    m = re.search(r"is (?:resting on|supported by) the (\w+)", text)
    if m:
        out["supported_by"] = m.group(1)
    if "whether the gripper is touching it is not observed" in text:
        out["contact"] = "unknown"
    elif "the gripper is not touching it" in text:
        out["contact"] = "no_contact"
    elif "the gripper is touching it" in text:
        out["contact"] = "contact"
    if "whether it would move with the gripper is not yet established" in text:
        out["moving"] = "unknown"
    elif "it is not moving with the gripper" in text:
        out["moving"] = False
    elif "it is moving with the gripper" in text:
        out["moving"] = True
    if "is directly above the" in text:
        out["above"] = True
    elif "is not above the" in text:
        out["above"] = False
    if "already inside the" in text:
        out["in_receptacle"] = True
    elif "not yet inside the" in text:
        out["in_receptacle"] = False
    m = re.search(r"Recently: (.+?)\.\s*Available actions", text, re.S)
    out["events"] = m.group(1) if m else None
    m = re.search(r"Available actions: (.+?)\.", text)
    out["actions"] = [a.strip() for a in m.group(1).split(",")] if m else []
    return out


def parse_typed(text: str) -> dict[str, Any]:
    """Recover the schema facts from the R2/R3 JSON."""
    state = json.loads(text)
    geometry = state.get("geometry") or {}
    relations = state.get("relations") or {}
    contact = state.get("contact") or {}
    out: dict[str, Any] = {}
    rel = geometry.get("target_rel_gripper")
    if isinstance(rel, dict):
        out["dx"], out["dy"], out["dz"] = rel.get("dx"), rel.get("dy"), rel.get("dz")
        out["dyaw_rad"] = rel.get("dyaw_rad")
    rel_r = geometry.get("target_rel_receptacle")
    if isinstance(rel_r, dict):
        out["rdx"], out["rdy"], out["rdz"] = rel_r.get("dx"), rel_r.get("dy"), rel_r.get("dz")
    out["gripper_opening_m"] = geometry.get("gripper_opening_m")
    out["supported_by"] = relations.get("target_supported_by")
    moving = relations.get("target_moving_with_gripper")
    out["moving"] = moving
    out["above"] = relations.get("gripper_above_target")
    out["in_receptacle"] = relations.get("target_in_receptacle")
    out["contact"] = contact.get("gripper_target", "unknown") if contact else "unknown"
    events = state.get("events_recent")
    out["events"] = (
        ", ".join(str(e.get("event")) for e in events) if isinstance(events, list) else None
    )
    out["actions"] = state.get("actions_available") or []
    return out


def parse_state_text(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return parse_typed(text)
    return parse_caption(text)


#: A closed gripper with the fingers this far apart has something between them.
#: Closed on nothing settles at about 12 mm; closed on the 40 mm cube at about
#: 38 mm. The backend uses this to recover the contact fact when the ``contact``
#: block is absent (R2, and the drop-contact ablation), so that those conditions
#: measure what the representation costs rather than how naive the reader is.
FINGERS_ON_SOMETHING_M = 0.020


def _touching(facts: dict[str, Any]) -> bool:
    """Best estimate of finger-target contact from whatever the state carries."""
    stated = facts.get("contact")
    if stated in ("contact", "no_contact"):
        return stated == "contact"
    opening = facts.get("gripper_opening_m")
    if opening is None:
        return False
    return FINGERS_ON_SOMETHING_M < float(opening) <= RELEASE_OPENING_M


def _held(facts: dict[str, Any]) -> bool:
    supported = facts.get("supported_by")
    if supported == "gripper":
        return True
    if supported in ("table", "receptacle"):
        return False
    if facts.get("moving") is True:
        return True
    return bool(_touching(facts) and supported in (None, "none"))


def _aligned_for_close(facts: dict[str, Any]) -> bool:
    above = facts.get("above")
    if above is False:
        return False
    dz = facts.get("dz")
    if dz is None:
        # No relative geometry at all: being above the target is the only
        # evidence left, and the approach primitive always ends at grasp height.
        return bool(above)
    if abs(float(dz)) >= ALIGN_Z_M:
        return False
    dyaw = facts.get("dyaw_rad")
    if dyaw is not None and abs(float(dyaw)) >= ALIGN_YAW_RAD:
        return False
    return bool(above) if above is not None else True


def _over_receptacle(facts: dict[str, Any]) -> bool:
    rdx, rdy, rdz = facts.get("rdx"), facts.get("rdy"), facts.get("rdz")
    if rdx is None or rdy is None:
        return False
    horizontal = (float(rdx) ** 2 + float(rdy) ** 2) ** 0.5
    if horizontal >= OVER_RECEPTACLE_M:
        return False
    return rdz is None or float(rdz) < PLACE_CLEARANCE_M


def choose_action(facts: dict[str, Any]) -> tuple[str, str]:
    """Return (primitive, the condition that fired)."""
    opening = facts.get("gripper_opening_m")
    open_enough = opening is None or float(opening) > RELEASE_OPENING_M
    touching = _touching(facts)
    inside = facts.get("in_receptacle")

    if inside and not touching and open_enough:
        events = facts.get("events") or ""
        if "retreat_done" in events:
            return "inspect", "placed and released; already retreated"
        return "retreat", "placed and released; clear the receptacle"
    if _held(facts):
        if _over_receptacle(facts):
            return "release", "held and over the receptacle at placing height"
        return "transport", "held; not yet over the receptacle"
    if touching:
        return "test_lift", "touching the target but it is still on a surface"
    if not open_enough:
        return "release", "fingers closed on nothing; open before retrying"
    if _aligned_for_close(facts):
        return "close", "open, lined up on the grasp pose"
    return "approach", "open, not lined up on the grasp pose"


class RuleBackend(DecisionBackend):
    """No model call. Deterministic, zero latency worth reporting, zero cost."""

    name = "rules"
    remote = False

    def __init__(self, model: str | None = None):
        super().__init__(model=model or "explicit-conditions-v1")

    def decide(
        self,
        state_text: str,
        questions: dict[str, list[str]],
        instruction_override: str | None = None,
    ) -> Decision:
        import time

        started = time.perf_counter()
        try:
            facts = parse_state_text(state_text)
            parse_error = None
        except Exception as exc:  # noqa: BLE001
            facts, parse_error = {}, f"{type(exc).__name__}: {exc}"

        action, condition = choose_action(facts)
        held = _held(facts)
        opening = facts.get("gripper_opening_m")
        open_enough = opening is None or float(opening) > RELEASE_OPENING_M
        complete = bool(facts.get("in_receptacle") and not _touching(facts) and open_enough)
        answers: dict[str, Answer] = {}
        if NEXT_ACTION in questions:
            options = questions[NEXT_ACTION]
            answers[NEXT_ACTION] = point_mass(
                action if action in options else options[0], options, 0.97
            )
        if OBJECT_HELD in questions:
            moving = facts.get("moving")
            confidence = 0.95 if moving in (True, False) else 0.70
            answers[OBJECT_HELD] = point_mass(
                "yes" if held else "no", questions[OBJECT_HELD], confidence
            )
        if STEP_COMPLETE in questions:
            answers[STEP_COMPLETE] = point_mass(
                "yes" if complete else "no", questions[STEP_COMPLETE], 0.95
            )
        decision = Decision(
            answers=answers,
            raw={"parsed_facts": facts, "condition": condition, "parse_error": parse_error},
            backend=self.name,
            model=self.model,
            latency_seconds=time.perf_counter() - started,
            error=parse_error,
            extra={"condition": condition, "recovered_fields": sorted(k for k, v in facts.items() if v is not None)},
        )
        return self.account(decision)


register_backend("rules", RuleBackend)
