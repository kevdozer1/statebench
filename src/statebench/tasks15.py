"""Turn 15: four tasks for skill selectors, the intent conditions, the prompt, and the episode runners.

Tasks:
* ``block``: Turn 14 pick and place (``runner_v14``), state condition T (tracking only), forced miss as Turn 13-14.
* ``button``: Turn 14 button (``button_task_v14``), state condition T, forced miss as Turn 14.
* ``drawing``: the playground drawing scene at planner level (``external/playground_v15/draw_scene.py``, a JSON-lines
  server in the playground venv), our targets (3 public-domain painting outlines and the word ABC).
* ``writing``: the playground whiteboard-writing scene at planner level (``external/playground_v15/write_scene.py``),
  a Fibonacci-style sequence per seed, written in binary (the scene's own glyphs 0 and 1; see write_scene.py for why).

The forced failure is scheduled on every task by ``scene.perturbation_fires(seed, 0.3)``.

**Prompt (v15)** = prompt v14's layout (role line, intent block, skills, fields present, last 3 actions with the robot's
own events, state, output instruction). The intent block replaces the goal line:
* a goal intent is printed as ``GOAL: <text>``;
* chunks are printed as ``STEPS (in order; each step is done when its condition holds):`` then ``1. <step>: done
  when <condition>`` per chunk;
* I5 prints the STEPS block without a GOAL line.
For block and button the v14 builder is replaced at runtime by a dispatcher that prints this intent block for an
``IntentTask`` and is byte-identical to v14 for any other task (checked by ``check_v14_unchanged``).

**Chunk judge** (diagnostic): at every decision of a Jev I4 run, one Jev request carries one noul question per chunk
("Is this step complete: <step>: done when <condition>"), with the same state text as the decision. The judge is
asked about every chunk at every decision (a superset of the open chunks). Truth per chunk comes from the task state
(``chunk_truth``); the naive baseline calls a chunk complete once its skill has returned (``naive_complete``).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import prompt_v14 as PV
from .scene import perturbation_fires
from .config import SCENES, playground_python

PLAYGROUND_PY = str(playground_python())
V15_DIR = SCENES
COVERAGE_DONE = 0.9
INTENTS = ("I0", "I1", "I2", "I3", "I4", "I5", "I6")


@dataclass(frozen=True)
class IntentTask(PV.Task):
    intent_goal: str | None = None
    intent_chunks: tuple = ()   # ((step, done condition), ...)


def intent_block(goal: str | None, chunks) -> list[str]:
    lines = []
    if goal:
        lines += [f"GOAL: {goal}"]
    if chunks:
        if goal:
            lines += [""]
        lines += ["STEPS (in order; each step is done when its condition holds):"]
        lines += [f"{i + 1}. {s}: done when {c}" for i, (s, c) in enumerate(chunks)]
    return lines


_V14_BUILD = PV.build_prompt


def build_prompt_v15(task, view: dict, last_actions: list[dict]) -> str:
    if not isinstance(task, IntentTask):
        return _V14_BUILD(task, view, last_actions)
    lines = [task.role, ""] + intent_block(task.intent_goal, task.intent_chunks) + ["", "SKILLS:"]
    lines += [f"- {s}: {task.skill_lines[s]}" for s in task.skills]
    lines += ["", "FIELDS IN THE STATE:"]
    lines += [f"- {k}: {task.glossary[k]}" for k in view]
    lines += ["", f"LAST {PV.HISTORY_LEN} ACTIONS (oldest first), {task.event_lines}:"]
    lines += PV.history_lines(last_actions)
    lines += ["", "STATE:", json.dumps(view, indent=1, sort_keys=False), "", task.output_instruction()]
    return "\n".join(lines)


PV.build_prompt = build_prompt_v15  # runtime dispatcher (declared); v14 text unchanged for plain tasks


def check_v14_unchanged() -> bool:
    view = {"target_rel_gripper_m": {"dx": 0.0, "dy": 0.0, "dz": 0.0}, "gripper_command": "open"}
    acts = [{"skill": "approach", "events": ["approach_done"]}]
    return build_prompt_v15(PV.PICK_PLACE_V14, view, acts) == _V14_BUILD(PV.PICK_PLACE_V14, view, acts)


# ====================================================================== intents
def block_intents() -> dict:
    goal3 = ("Pick up the red cube and place it in the blue receptacle so that it rests inside it, open the gripper to "
             "release it, then move the gripper clear of the receptacle.")
    chunks = (("Grasp the cube", "the gripper is closed on the cube and the cube moves with the gripper when it is lifted"),
              ("Carry it to the receptacle", "the cube is held above the receptacle"),
              ("Release it", "the gripper is open and the cube rests inside the receptacle"),
              ("Move clear", "the gripper is clear of the receptacle"))
    return {"I0": {"goal": "pick and place"}, "I1": {"goal": "Put the block away."},
            "I2": {"goal": "Pick up the red cube from the table and place it in the blue receptacle."},
            "I3": {"goal": goal3}, "I4": {"goal": goal3, "chunks": chunks}, "I5": {"chunks": chunks},
            "I6": {"goal": ("Pick up the red cube and place it in the blue receptacle so that it rests inside it, open the "
                            "gripper to release it, then keep the gripper just above the receptacle.")},
            "I6_detail": "the final condition is wrong: keep the gripper over the receptacle instead of moving it clear"}


def button_intents() -> dict:
    goal3 = "Press the button until it activates, then release it and move the gripper clear."
    chunks = (("Line up over the button", "the gripper is over the centre of the button's cap, above it"),
              ("Press", "the button is activated"),
              ("Release", "the button has been activated and then released"),
              ("Move clear", "the gripper is clear of the button"))
    return {"I0": {"goal": "button press"}, "I1": {"goal": "Press the button."},
            "I2": {"goal": "Press the button until it activates."}, "I3": {"goal": goal3},
            "I4": {"goal": goal3, "chunks": chunks}, "I5": {"chunks": chunks},
            "I6": {"goal": "Press the button until it activates, then release it and keep the gripper resting on it."},
            "I6_detail": "the final condition is wrong: keep the gripper on the button instead of moving it clear"}


def drawing_intents(n_strokes: int) -> dict:
    goal3 = (f"Draw all {n_strokes} strokes of the target outline so that each stroke is inked along at least 90% of its "
             "path with no ink off the path, then lift the pen off the paper.")
    chunks = tuple((f"Draw stroke {k}", f"stroke {k}'s tracked coverage is at least 0.9") for k in range(n_strokes))
    chunks += (("Lift the pen", "the pen is up after the last stroke"),)
    return {"I0": {"goal": "drawing"}, "I1": {"goal": "Draw the picture."},
            "I2": {"goal": "Draw the strokes of the target outline."}, "I3": {"goal": goal3},
            "I4": {"goal": goal3, "chunks": chunks}, "I5": {"chunks": chunks},
            "I6": {"goal": (f"Draw all {n_strokes} strokes of the target outline so that each stroke is inked along at least "
                            "90% of its path with no ink off the path, then leave the pen resting on the paper.")},
            "I6_detail": "the final condition is wrong: leave the pen on the paper instead of lifting it"}


def writing_intents(task: dict) -> dict:
    a, b = task["start"]
    n, bins = task["n_terms"], task["terms_binary"]
    goal3 = (f"Write, in binary, the first {n} terms of the sequence that starts {a}, {b}, where each term is the sum of "
             "the two before it, on one line, separated by single spaces, then lift the marker away from the board.")
    chunks = []
    for i, t in enumerate(bins):
        prefix = " ".join(bins[: i + 1])
        chunks.append((f"Write {t}" + (" and a space" if i < n - 1 else ""), f'the board reads "{prefix}"'))
    chunks.append(("Lift the marker", "the marker is clear of the board"))
    wrong_n = n - 1
    return {"I0": {"goal": "fibonacci writing"}, "I1": {"goal": "Write the Fibonacci numbers on the board."},
            "I2": {"goal": f"Write the Fibonacci sequence starting {a}, {b} on the board, in binary."}, "I3": {"goal": goal3},
            "I4": {"goal": goal3, "chunks": tuple(chunks)}, "I5": {"chunks": tuple(chunks)},
            "I6": {"goal": (f"Write, in binary, the first {wrong_n} terms of the sequence that starts {a}, {b}, where each term "
                            "is the sum of the two before it, on one line, separated by single spaces, then lift the marker "
                            "away from the board.")},
            "I6_detail": f"the count is wrong: {wrong_n} terms instead of {n}"}


# ====================================================================== drawing and writing task definitions
DRAW_GLOSSARY = {
    "strokes": ("the target's strokes in order, each with its index k, its length in millimetres, the share of its path "
                "the tracked pen tip has covered while touching the paper (tracked_coverage, 0 to 1), and how many times "
                "it has been drawn."),
    "pen": "whether the pen is up or down.",
}
WRITE_GLOSSARY = {
    "recognized_text": "the characters on the board so far, as recognized from the ink, with spaces for blank positions "
                       "and ? for an unreadable glyph.",
    "cursor": "the position where the next glyph or space goes, counted from 0 at the start of the line.",
    "marker": "near board or clear of board.",
}


def drawing_task(n_strokes: int, intent: dict) -> IntentTask:
    skills = tuple(f"draw({k})" for k in range(n_strokes)) + tuple(f"redraw({k})" for k in range(n_strokes)) + ("lift", "finish")
    lines = {f"draw({k})": (f"move to stroke {k}'s start (raising the pen first if it is down), put the pen down and draw "
                            f"stroke {k}; the pen stays down at its end.") for k in range(n_strokes)}
    lines.update({f"redraw({k})": f"draw stroke {k} again in the same way." for k in range(n_strokes)})
    lines["lift"] = "raise the pen 20 mm off the paper."
    lines["finish"] = "end the task."
    return IntentTask(name="drawing", goal=intent.get("goal") or "", skills=skills, skill_lines=lines,
                      glossary=DRAW_GLOSSARY, role="You choose the next skill for a robot arm holding a pencil over paper.",
                      intent_goal=intent.get("goal"), intent_chunks=tuple(intent.get("chunks", ())))


def writing_task(intent: dict) -> IntentTask:
    digits = "01"
    skills = tuple(f"write_glyph({c})" for c in digits) + ("space",) + tuple(f"rewrite({c})" for c in digits) + ("lift", "finish")
    lines = {f"write_glyph({c})": f"write the digit {c} at the cursor, then move the cursor one position right." for c in digits}
    lines["space"] = "move the cursor one position right without writing."
    lines.update({f"rewrite({c})": f"write the digit {c} again over the last written glyph; the cursor does not move."
                  for c in digits})
    lines["lift"] = "move the marker away from the board."
    lines["finish"] = "end the task."
    return IntentTask(name="writing", goal=intent.get("goal") or "", skills=skills, skill_lines=lines,
                      glossary=WRITE_GLOSSARY, role="You choose the next skill for a humanoid robot writing with a marker on a whiteboard.",
                      intent_goal=intent.get("goal"), intent_chunks=tuple(intent.get("chunks", ())),
                      event_lines=PV.PICK_PLACE_V14.event_lines)


def block_task(intent: dict) -> IntentTask:
    t = PV.PICK_PLACE_V14
    return IntentTask(name="pick_and_place", goal=intent.get("goal") or "", skills=t.skills, skill_lines=t.skill_lines,
                      glossary=t.glossary, intent_goal=intent.get("goal"), intent_chunks=tuple(intent.get("chunks", ())))


def button_task(intent: dict) -> IntentTask:
    from .button_task_v14 import BUTTON_TASK as t

    return IntentTask(name="button", goal=intent.get("goal") or "", skills=t.skills, skill_lines=t.skill_lines,
                      glossary=t.glossary, intent_goal=intent.get("goal"), intent_chunks=tuple(intent.get("chunks", ())))


# ====================================================================== scene servers (drawing, writing)
class SceneServer:
    """One playground-venv process running a scene server; JSON lines over stdin/stdout."""

    def __init__(self, script: str):
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        import os

        full = dict(os.environ, **env)
        full.pop("MUJOCO_GL", None)
        self.p = subprocess.Popen([PLAYGROUND_PY, str(V15_DIR / script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1, env=full, cwd=str(V15_DIR))
        self.lock = threading.Lock()

    def call(self, msg: dict) -> dict:
        with self.lock:
            self.p.stdin.write(json.dumps(msg) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("scene server ended")
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.p.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        self.p.kill()


def _strip_hidden(state: dict, task_name: str) -> dict:
    if task_name == "drawing":
        return {"strokes": [{k: s[k] for k in ("k", "length_mm", "tracked_coverage", "times_drawn")} for s in state["strokes"]],
                "pen": state["pen"]}
    return {"recognized_text": state["recognized_text"], "cursor": state["cursor"], "marker": state["marker"]}


def run_server_episode(server: SceneServer, task_name: str, seed: int, intent_name: str, intents_fn, reader,
                       keep_prompts: bool = False, judge=None) -> dict:
    """One drawing or writing episode through the scene server."""
    started = time.perf_counter()
    fired = perturbation_fires(seed, 0.3)
    r0 = server.call({"cmd": "reset", "seed": seed, "forced": fired})
    st = r0["state"]
    if task_name == "drawing":
        n = st["n_strokes"]
        intents = intents_fn(n)
        max_dec = 2 * n + 6
    else:
        intents = intents_fn(r0["task"])
        max_dec = 2 * len(r0["task"]["text"]) + 6
    intent = intent_name if isinstance(intent_name, dict) else intents[intent_name]
    task = (drawing_task(st["n_strokes"], intent) if task_name == "drawing" else writing_task(intent))
    if hasattr(reader, "text") and task_name == "writing":
        reader.text = r0["task"]["text"]
    if hasattr(reader, "task"):
        reader.task = task  # the episode's skill list (drawing: per target)
    last_actions, steps = [], []
    recent = []
    loop_stopped = False
    for index in range(max_dec):
        view = _strip_hidden(st, task_name)
        prompt = build_prompt_v15(task, view, last_actions)
        jud = judge(prompt, task) if judge is not None else None
        dec = reader.decide(prompt, view)
        skill = dec["skill"]
        if skill in task.skills:
            r = server.call({"cmd": "do", "skill": skill})
        else:  # an invalid answer: no action this decision (declared)
            skill = "none"
            r = {"state": st, "events": ["no action (invalid answer)"], "sim_dt": 0.0, "wall_s": 0.0}
        last_actions.append({"skill": skill, "events": [e for e in r["events"] if not e.startswith("rejected")] or r["events"]})
        rec = {"index": index, "chosen": dec["skill"], "executed": skill, "valid": dec["valid"], "raw_skill": dec["raw_skill"],
               "error": dec["error"], "tokens_in": dec["tokens_in"], "tokens_out": dec["tokens_out"],
               "latency_s": dec["latency_s"], "cost_usd": dec["cost_usd"], "events": r["events"], "view": view,
               "state_after": r["state"], "judge": jud, "sim_dt": r.get("sim_dt"), "server_wall_s": r.get("wall_s")}
        if keep_prompts:
            rec["prompt"] = prompt
        steps.append(rec)
        st = r["state"]
        recent.append((skill, json.dumps(_strip_hidden(st, task_name), sort_keys=True)))
        recent = recent[-6:]
        if len(recent) == 6 and len(set(recent)) == 1:
            loop_stopped = True
            break
        if skill == "finish" or st.get("error"):
            break
    v = server.call({"cmd": "verify"})
    lat = [s["latency_s"] for s in steps]
    return {"seed": seed, "task": task_name, "condition": intent_name if isinstance(intent_name, str) else "custom",
            "reader": getattr(reader, "name", "?"), "success": bool(v["success"]), "verifier": v,
            "perturbation_fired": fired, "loop_stopped": loop_stopped, "decision_steps": len(steps),
            "task_def": r0.get("task") or {"target": st.get("target"), "n_strokes": st.get("n_strokes")},
            "invalid_answers": sum(1 for s in steps if not s["valid"]),
            "tokens_in": sum(s["tokens_in"] or 0 for s in steps), "tokens_out": sum(s["tokens_out"] or 0 for s in steps),
            "cost_usd": round(sum(s["cost_usd"] or 0 for s in steps), 6),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "wall_seconds": round(time.perf_counter() - started, 3), "steps": steps}


# ====================================================================== rules (solvability references; no intent read)
class RulesDraw:
    name = "rules"

    def decide(self, prompt, view):
        todo = [s for s in view["strokes"] if s["tracked_coverage"] < COVERAGE_DONE]
        if todo:
            s = todo[0]
            sk = ("redraw" if s["times_drawn"] else "draw") + f"({s['k']})"
        elif view["pen"] == "down":
            sk = "lift"
        else:
            sk = "finish"
        return {"skill": sk, "valid": True, "raw_skill": sk, "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                "cost_usd": 0.0, "error": None, "raw": None}


class RulesWrite:
    """Knows the task's text from the task definition (set per episode by ``set_task``)."""

    name = "rules"

    def __init__(self):
        self.text = None

    def decide(self, prompt, view):
        tgt, rec, cur = self.text, view["recognized_text"], view["cursor"]
        if cur > 0 and tgt[cur - 1] != " " and (len(rec) < cur or rec[cur - 1] != tgt[cur - 1]):
            sk = f"rewrite({tgt[cur - 1]})"
        elif cur < len(tgt):
            sk = "space" if tgt[cur] == " " else f"write_glyph({tgt[cur]})"
        elif view["marker"] != "clear of board":
            sk = "lift"
        else:
            sk = "finish"
        return {"skill": sk, "valid": True, "raw_skill": sk, "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                "cost_usd": 0.0, "error": None, "raw": None}
