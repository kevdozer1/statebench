"""FROZEN question wording. Identical across every backend and every condition.

Turn 2 pre-registers these strings. No per-backend prompt tuning: the Jev
backend and the OpenRouter backends send the same preamble and the same three
instruction strings, and only the transport differs (TypeSafe ``choice``/``noul``
primitives versus a JSON-schema-constrained chat completion).

The option descriptions are the primitives' contracts as the fixed executor
implements them. They describe what each primitive does, never which one to
pick in a given state.
"""

from __future__ import annotations

from .schema import ACTIONS

NEXT_ACTION = "next_action"
OBJECT_HELD = "object_held"
STEP_COMPLETE = "step_complete"

YES = "yes"
NO = "no"
YES_NO = (YES, NO)

SYSTEM_PREAMBLE = (
    "You are choosing actions for a robot arm with a parallel-jaw gripper. The arm is "
    "driven by seven fixed scripted skill primitives; you select which primitive runs "
    "next, and the primitive reads the same state you were given. You never command "
    "joint angles or positions directly. Answer every question from the supplied state "
    "alone: it is the only information about the scene that exists for you, and any "
    "field it does not contain is genuinely unknown rather than merely omitted. A closed "
    "gripper does not imply the object is held; contact and moving-with-the-gripper are "
    "separate facts, and test_lift is the primitive that establishes the second one. "
    "Your probabilities are your own assessments, not calibrated physical success rates."
)

INSTRUCTIONS = {
    NEXT_ACTION: (
        "Choose the single primitive the robot should run next in order to finish the "
        "task described in the state's goal block."
    ),
    OBJECT_HELD: (
        "The target object will stay with the gripper if the arm moves."
    ),
    STEP_COMPLETE: (
        "The current goal is satisfied."
    ),
}

ACTION_CRITERIA = {
    "approach": (
        "Move the gripper to the grasp pose on the target object: across to the target's "
        "position at travel height, then down to grasp height, turning the finger axis to "
        "line up with the object. Does not change the fingers."
    ),
    "close": (
        "Close the fingers where the gripper currently is. Does not move the arm."
    ),
    "test_lift": (
        "Raise the gripper straight up to travel height without changing the fingers, so "
        "that whether the object comes with it becomes observable."
    ),
    "transport": (
        "Carry the gripper across to the receptacle at travel height and lower it to "
        "placing height, keeping the fingers as they are."
    ),
    "release": (
        "Open the fingers. This is also how a gripper that closed on nothing is reopened "
        "before another attempt."
    ),
    "inspect": (
        "Hold still and observe the state again without moving the arm or the fingers."
    ),
    "retreat": (
        "Raise the gripper to travel height and withdraw it to the home pose, clear of "
        "the receptacle."
    ),
}

NOUL_CRITERIA = {
    "true": "The statement is true of the physical state described.",
    "false": "The statement is not true of the physical state described.",
}


def options() -> dict[str, list[str]]:
    """The option list per question. Fixed for the whole turn."""
    return {
        NEXT_ACTION: list(ACTIONS),
        OBJECT_HELD: list(YES_NO),
        STEP_COMPLETE: list(YES_NO),
    }


def frozen_text() -> dict[str, str]:
    """Everything printed in the wrap-up, so the wording is on the record."""
    return {
        "system_preamble": SYSTEM_PREAMBLE,
        "next_action": INSTRUCTIONS[NEXT_ACTION],
        "object_held": INSTRUCTIONS[OBJECT_HELD],
        "step_complete": INSTRUCTIONS[STEP_COMPLETE],
    }
