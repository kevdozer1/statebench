"""``rules_v2``: the R4c decision table as an explicit first-match lookup.

The comparator for every named-policy condition from v0.2 on. ``backends/rules.py``
(Turn 1) is unchanged: it reads ``target_in_receptacle`` / ``target_supported_by``,
which R4 removes, so it scored 0/50 on R4 for a reason unrelated to R4.

It reads only the predicate block of an R4 packet (JSON or prose), and applies
the rules of Turn 3b's R4c instruction in this order, first match wins:

1. inspect   if estimate_stale
2. release   if gripper_closed_empty
3. retreat   if released_at_receptacle
4. release   if object_at_receptacle and gripper_closed_on_object
5. transport if gripper_closed_on_object and object_moving_with_gripper is true
6. test_lift if gripper_closed_on_object and object_moving_with_gripper is unknown
7. close     if aligned_for_grasp and the gripper is open
8. approach  otherwise

"The gripper is open" is not a predicate. It is read as: neither
gripper_closed_on_object nor gripper_closed_empty is true, which on this gripper
is the same as the last command being ``open`` (see ``predicates._grip``).

The two noul answers are point masses from the same block, declared so the judge
channel is populated: step_complete = released_at_receptacle, object_held =
gripper_closed_on_object and object_moving_with_gripper is true.
"""

from __future__ import annotations

import json
import time
from typing import Any

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

TABLE_ORDER = (
    ("inspect", "estimate_stale"),
    ("release", "gripper_closed_empty"),
    ("retreat", "released_at_receptacle"),
    ("release", "object_at_receptacle and gripper_closed_on_object"),
    ("transport", "gripper_closed_on_object and object_moving_with_gripper is true"),
    ("test_lift", "gripper_closed_on_object and object_moving_with_gripper is unknown"),
    ("close", "aligned_for_grasp and the gripper is open"),
    ("approach", "otherwise"),
)


def predicate_block(text: str) -> dict[str, Any]:
    """The predicate values from an R4 packet, JSON or prose."""
    if text.lstrip().startswith("{"):
        state = json.loads(text)
    else:
        from ..representations_v2 import parse_prose

        state = parse_prose(text)
    block = state.get("predicates") or {}
    return block.get("values", block) if isinstance(block, dict) else {}


def choose(p: dict[str, Any]) -> tuple[str, str]:
    """(primitive, the rule that fired)."""
    on_object = p.get("gripper_closed_on_object") is True
    empty = p.get("gripper_closed_empty") is True
    moving = p.get("object_moving_with_gripper")
    if p.get("estimate_stale") is True:
        return TABLE_ORDER[0]
    if empty:
        return TABLE_ORDER[1]
    if p.get("released_at_receptacle") is True:
        return TABLE_ORDER[2]
    if p.get("object_at_receptacle") is True and on_object:
        return TABLE_ORDER[3]
    if on_object and moving is True:
        return TABLE_ORDER[4]
    if on_object and moving == "unknown":
        return TABLE_ORDER[5]
    if p.get("aligned_for_grasp") is True and not on_object and not empty:
        return TABLE_ORDER[6]
    return TABLE_ORDER[7]


class RulesV2Backend(DecisionBackend):
    name = "rules_v2"
    remote = False

    def __init__(self, model: str | None = None):
        super().__init__(model=model or "r4c-lookup-v2")

    def decide(self, state_text, questions, instruction_override=None) -> Decision:
        started = time.perf_counter()
        error = None
        try:
            p = predicate_block(state_text)
        except Exception as exc:  # noqa: BLE001
            p, error = {}, f"{type(exc).__name__}: {exc}"
        action, rule = choose(p)
        answers: dict[str, Answer] = {}
        if NEXT_ACTION in questions:
            answers[NEXT_ACTION] = point_mass(action, questions[NEXT_ACTION], 0.97)
        if STEP_COMPLETE in questions:
            done = p.get("released_at_receptacle") is True
            answers[STEP_COMPLETE] = point_mass("yes" if done else "no", questions[STEP_COMPLETE], 0.95)
        if OBJECT_HELD in questions:
            held = p.get("gripper_closed_on_object") is True and p.get("object_moving_with_gripper") is True
            answers[OBJECT_HELD] = point_mass("yes" if held else "no", questions[OBJECT_HELD], 0.95)
        return self.account(Decision(
            answers=answers, raw={"rule": rule, "predicates": p}, backend=self.name,
            model=self.model, latency_seconds=time.perf_counter() - started, error=error,
            extra={"rule": rule},
        ))


register_backend("rules_v2", RulesV2Backend)
