"""Turn 13: the one prompt every LLM reader gets (frozen and hashed before any LLM result).

Every reader (the two local models, Jev, Sonnet and the Astra probe) receives the text ``build_prompt`` returns,
byte for byte, at temperature 0. The prompt has five parts and nothing else:

1. the goal;
2. the skill menu, one line per skill (what the skill does, never when to use it);
3. a glossary of only the fields present in this condition's state;
4. the last 3 actions, each with the completion events the robot reported for its own commands;
5. the state, then the output instruction.

There is no decision table and there are no recovery hints. The glossary is source-neutral: the same field has the
same line whether its value comes from tracking rules, from truth, late or flipped, so the conditions differ only in
the state's values and in which fields are present.

Transport differences, declared (the text is the same):
* local models (Ollama): one user message; output constrained by a JSON schema whose ``skill`` enum is the skill
  list; thinking off.
* OpenRouter (Sonnet, Astra): one user message; ``response_format`` strict JSON schema with the same enum;
  reasoning off where the model allows it.
* Jev (TypeSafe System One ``choice``): ``state`` = the full prompt text (byte-identical to the other readers'
  message), ``instructions`` = the prompt's first line, ``criteria`` = the same skill lines keyed by skill. The
  options are the skill list.
"""

from __future__ import annotations

import json

PROMPT_VERSION = "statebench/prompt/v13"

SKILLS = ("approach", "close", "test_lift", "transport", "release", "inspect", "retreat")

GOAL = "Pick up the red cube from the table and place it in the blue receptacle."

SKILL_LINES = {
    "approach": "move the gripper above the cube's estimated position at travel height, then down to grasp height, "
                "turning the fingers to the cube's estimated yaw; the fingers are not changed.",
    "close": "close the fingers where the gripper is; the arm does not move.",
    "test_lift": "raise the gripper straight up to travel height; the fingers are not changed.",
    "transport": "carry the gripper to the receptacle at travel height, then lower it to placing height; the "
                 "fingers are not changed.",
    "release": "open the fingers; the arm does not move.",
    "inspect": "hold still for a moment and observe again.",
    "retreat": "raise the gripper to travel height and move it to the home pose.",
}

GLOSSARY = {
    # T: tracking geometry and the robot's own state
    "target_rel_gripper_m": "estimated position of the red cube minus the gripper's grasp point, in metres, robot "
                            "base frame (dx forward, dy left, dz up), from camera tracking.",
    "target_yaw_rel_gripper_rad": "estimated rotation of the cube about the vertical axis relative to the fingers, "
                                  "in radians, from camera tracking.",
    "target_rel_receptacle_m": "estimated position of the red cube minus the blue receptacle's centre, in metres, "
                               "robot base frame, from camera tracking.",
    "gripper_opening_m": "measured distance between the fingers, in metres.",
    "gripper_command": "the last command sent to the fingers: open or closed.",
    # M: meaning fields
    "contact": "whether a finger touches the cube: contact, no_contact or unknown.",
    "grip": "the fingers' state: open, on_object (closed on the cube), empty (closed on nothing) or unknown.",
    "object_moving_with_gripper": "whether the cube moves with the gripper: true, false or unknown.",
    "object_at_receptacle": "whether the cube is at the receptacle: true or false.",
    "released_at_receptacle": "whether the cube has been released at the receptacle: true or false.",
    # P: procedural fields
    "aligned_for_grasp": "whether the gripper is at the grasp pose on the cube with the fingers open: true or false.",
    "estimate_stale": "whether the tracking estimate is out of date: true or false.",
    "close_attempts_since_lift": "number of close commands since the last test_lift.",
}

FIELDS_T = ("target_rel_gripper_m", "target_yaw_rel_gripper_rad", "target_rel_receptacle_m", "gripper_opening_m",
            "gripper_command")
FIELDS_M = ("contact", "grip", "object_moving_with_gripper", "object_at_receptacle", "released_at_receptacle")
FIELDS_P = ("aligned_for_grasp", "estimate_stale", "close_attempts_since_lift")

#: events the robot reports for its own commands (object-motion events are never shown)
EVENT_LINES = "each with the completion events the robot reported for its own commands"

OUTPUT_INSTRUCTION = ('Answer with a JSON object {"skill": S}, where S is exactly one of: '
                      + ", ".join(SKILLS) + ".")

HISTORY_LEN = 3


def output_schema() -> dict:
    return {"type": "object", "properties": {"skill": {"type": "string", "enum": list(SKILLS)}},
            "required": ["skill"], "additionalProperties": False}


def state_block(view: dict) -> str:
    return json.dumps(view, indent=1, sort_keys=False)


def history_lines(last_actions: list[dict]) -> list[str]:
    recent = last_actions[-HISTORY_LEN:]
    if not recent:
        return ["(none yet)"]
    return [f"{i + 1}. {a['skill']}: " + (", ".join(a["events"]) if a["events"] else "no event reported")
            for i, a in enumerate(recent)]


def build_prompt(view: dict, last_actions: list[dict]) -> str:
    """The full prompt text. ``view`` holds only the fields present in the condition."""
    lines = ["You choose the next skill for a robot arm with a two-finger gripper.", "",
             f"GOAL: {GOAL}", "", "SKILLS:"]
    lines += [f"- {s}: {SKILL_LINES[s]}" for s in SKILLS]
    lines += ["", "FIELDS IN THE STATE:"]
    lines += [f"- {k}: {GLOSSARY[k]}" for k in view]
    lines += ["", f"LAST {HISTORY_LEN} ACTIONS (oldest first), {EVENT_LINES}:"]
    lines += history_lines(last_actions)
    lines += ["", "STATE:", state_block(view), "", OUTPUT_INSTRUCTION]
    return "\n".join(lines)


def prompt_hash() -> str:
    from pathlib import Path

    from .hashing import normalized_sha256

    return normalized_sha256(Path(__file__))
