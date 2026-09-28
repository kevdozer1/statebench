"""Turn 14: prompt v14, the one prompt every LLM reader gets from now on (frozen and hashed before any Phase 2 result).

The text is built exactly as ``prompt_v13`` builds it (role line, goal, skill menu with one line per skill, a
glossary of only the fields present, the last 3 actions with the robot's own completion events, the state, the output
instruction), with one change for pick and place: **the goal line states the verifier's condition** ("Pick up the red
cube, place it in the blue receptacle, then move the gripper clear of the receptacle."). v13 remains a historical
condition only.

The builder is task-generic (``Task``): the button task (Phase 3) is declared in its own module with the same
recipe and frozen separately, before any LLM reads a button state.

The glossary adds the T+H fields (Phase 2): a numeric history of the tracking the state builder used, the robot's own
gripper position, and the calibrated constants the meaning rules use, as numbers. They are described by what they
measure, never by what to do with them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import prompt_v13 as V13

PROMPT_VERSION = "statebench/prompt/v14"
HISTORY_LEN = 3


@dataclass(frozen=True)
class Task:
    name: str
    goal: str
    skills: tuple
    skill_lines: dict
    glossary: dict
    role: str = "You choose the next skill for a robot arm with a two-finger gripper."
    event_lines: str = V13.EVENT_LINES
    extra: dict = field(default_factory=dict)

    def output_instruction(self) -> str:
        return 'Answer with a JSON object {"skill": S}, where S is exactly one of: ' + ", ".join(self.skills) + "."

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"skill": {"type": "string", "enum": list(self.skills)}},
                "required": ["skill"], "additionalProperties": False}


PP_GLOSSARY = dict(V13.GLOSSARY)
PP_GLOSSARY.update({
    "gripper_tcp_m": "the robot's own measured position of the gripper's grasp point, in metres, robot base frame "
                     "(x forward, y left, z up).",
    "tracking_history": "earlier samples of the tracked quantities, each with its time relative to now (t_rel_s, "
                        "seconds, negative): target_rel_gripper_m, target_rel_receptacle_m, gripper_opening_m and "
                        "gripper_tcp_m, as defined above.",
    "decision_history": "the same quantities at the start of each of the last 3 actions, oldest first, each with its "
                        "time relative to now and the skill that followed.",
    "calibration": "calibrated constants of the tracking and gripper, as numbers (metres, seconds, radians), named by "
                   "the quantity they describe.",
    "estimate_age_s": "time since the tracking estimate in the state was made, in seconds.",
})

PICK_PLACE_V14 = Task(
    name="pick_and_place",
    goal="Pick up the red cube, place it in the blue receptacle, then move the gripper clear of the receptacle.",
    skills=V13.SKILLS,
    skill_lines=dict(V13.SKILL_LINES),
    glossary=PP_GLOSSARY,
)


def history_lines(last_actions: list[dict]) -> list[str]:
    recent = last_actions[-HISTORY_LEN:]
    if not recent:
        return ["(none yet)"]
    return [f"{i + 1}. {a['skill']}: " + (", ".join(a["events"]) if a["events"] else "no event reported")
            for i, a in enumerate(recent)]


def build_prompt(task: Task, view: dict, last_actions: list[dict]) -> str:
    lines = [task.role, "", f"GOAL: {task.goal}", "", "SKILLS:"]
    lines += [f"- {s}: {task.skill_lines[s]}" for s in task.skills]
    lines += ["", "FIELDS IN THE STATE:"]
    lines += [f"- {k}: {task.glossary[k]}" for k in view]
    lines += ["", f"LAST {HISTORY_LEN} ACTIONS (oldest first), {task.event_lines}:"]
    lines += history_lines(last_actions)
    lines += ["", "STATE:", json.dumps(view, indent=1, sort_keys=False), "", task.output_instruction()]
    return "\n".join(lines)


def check_only_goal_changed() -> bool:
    """For every Turn 13 field set, v14 text = v13 text with the goal line replaced."""
    view = {k: 0.0 for k in V13.FIELDS_T + V13.FIELDS_M + V13.FIELDS_P}
    acts = [{"skill": "approach", "events": ["approach_done"]}]
    for n in (5, 10, 13):
        v = dict(list(view.items())[:n])
        a, b = V13.build_prompt(v, acts), build_prompt(PICK_PLACE_V14, v, acts)
        if a.replace(f"GOAL: {V13.GOAL}", f"GOAL: {PICK_PLACE_V14.goal}") != b:
            return False
    return True


def prompt_hash() -> str:
    from pathlib import Path

    from .hashing import normalized_sha256

    return normalized_sha256(Path(__file__))
