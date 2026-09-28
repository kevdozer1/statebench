"""Turn 17: instruction variants rebuilt from saved replies, contrasting goals with their verifiers, and execution
controls. Nothing here changes a pinned file; runtime patches are declared.

**Instruction variants** (Phase 1; deterministic rewrites of what Turn 16 saved; no new vision call):
* ``C``: the compiled goal and chunks exactly as run in Turn 16 (``runs/turn16/iv_intents.json``; for v1 on dev seeds
  that file holds the Turn 15 intents, as Turn 16 ran them).
* ``O``: the outcome sentence alone, no chunks. v2 (Sol, cheap): the reply's ``outcome`` field. v1 (the Turn 15
  prompt) has no separate outcome field: its ``goal`` sentence is its instruction-level outcome, so O is that goal.
* ``O+S`` (``OS``): O plus the chunk steps in order, with each step's done condition removed. The prompt shows them
  under "STEPS (in order):" as "n. step" (``intent_block_v17``).
* ``E``: the goal line "No goal is given." and no chunks.

**Contrasting goals, same scene and skills** (Phase 2; declared before any run):
* button ``B-release``: "Press the button until it activates, then release it and move the gripper clear." Verifier:
  ``env_button.ButtonVerifier`` (a completed press cycle, button off, no contact, TCP 30 mm above the cap's rest top,
  held 0.5 s).
* button ``B-hold``: "Press the button until it activates, and keep it pressed at the end." Verifier (``hold_button``):
  on the final state, the button is latched ON and the cap is pressed past the activation point.
* block ``K-place``: the Turn 15 block I3 goal (place in the receptacle, release, move clear). Verifier: the Turn 14
  verifier (``verify.Verifier``).
* block ``K-hold``: "Lift the red cube and hold it above the blue receptacle." Verifier (``hold_block``): on the final
  state, the fingers touch the cube and the gripper command is closed; the cube centre is inside the receptacle's
  inner footprint (|dx|, |dy| <= 0.079 - 0.020 m from its centre); and the cube touches neither the receptacle nor
  the table.
* How an episode ends (unchanged runners): when the task's own Turn 14/15 verifier resolves (block: success after 1 s
  stable, failure or timeout; button: success after 0.5 s stable, failure at 20 s), when the loop stop fires (6
  identical primitives with no change in the verifier's conditions), or at the decision cap. Both verifiers of a pair
  are then evaluated on the final state (the task's own verifier by its recorded status).

**Wording pair** (Phase 3): one fixed instruction naming the target, one slot:
"Press the red button on the dark base until {slot}, then release it and move the gripper clear."
``W-contact``: slot = "the gripper touches the button"; ``W-active``: slot = "the button activates".

**Execution controls** (declared runtime patches):
* the forced failure: ``schedule`` (the usual ``scene.perturbation_fires(seed, 0.3)``), ``on`` (always) or ``off``
  (never), set per process by replacing the name ``perturbation_fires`` in ``runner_v14``, ``button_task_v14`` and
  ``tasks15``;
* final-state capture: the block and button environments are subclassed so that ``close()`` stores the truth needed
  by the hold verifiers (per thread).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from . import tasks15 as T
from .config import RUNS

R16 = RUNS / "turn16"
R15 = RUNS / "turn15"
E_GOAL = "No goal is given."
GOALS = {
    "B-release": "Press the button until it activates, then release it and move the gripper clear.",
    "B-hold": "Press the button until it activates, and keep it pressed at the end.",
    "K-hold": "Lift the red cube and hold it above the blue receptacle.",
    "W-contact": "Press the red button on the dark base until the gripper touches the button, then release it and move the gripper clear.",
    "W-active": "Press the red button on the dark base until the button activates, then release it and move the gripper clear.",
}
TRAY_C, TRAY_INNER, CUBE_HALF = (0.44, 0.20), 0.079, 0.020


# ------------------------------------------------------------------ prompt: steps without done conditions
def intent_block_v17(goal, chunks) -> list[str]:
    if not chunks or all(c is not None for _, c in chunks):
        return _ORIG_BLOCK(goal, chunks)
    lines = [f"GOAL: {goal}"] if goal else []
    if goal:
        lines += [""]
    lines += ["STEPS (in order):"] + [f"{i + 1}. {s}" for i, (s, _) in enumerate(chunks)]
    return lines


_ORIG_BLOCK = T.intent_block
T.intent_block = intent_block_v17


# ------------------------------------------------------------------ variants from saved replies
_CACHE: dict = {}


def _iv16() -> dict:
    if "iv" not in _CACHE:
        _CACHE["iv"] = json.loads((R16 / "iv_intents.json").read_text())
        out = {}
        for line in (R16 / "iv2_raw.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                out[(r["model_tag"], r["task"], r["seed"])] = r
        _CACHE["raw"] = out
    return _CACHE["iv"]


def variant(kind: str, source: str, task: str, seed: int) -> dict:
    """kind C, O, OS or E; source v1, v2-sol or v2-cheap."""
    if kind == "E":
        return {"goal": E_GOAL}
    it = _iv16()[source][task][str(seed)]
    if kind == "C":
        return {"goal": it["goal"], "chunks": tuple(tuple(c) for c in it["chunks"])}
    if source == "v1":
        outcome = it["goal"]
    else:
        outcome = str(_CACHE["raw"][(source.split("-")[1], task, seed)]["reply"]["outcome"]).strip()
    if kind == "O":
        return {"goal": outcome}
    if kind == "OS":
        return {"goal": outcome, "chunks": tuple((c[0], None) for c in it["chunks"])}
    raise ValueError(kind)


def block_i3() -> dict:
    return T.block_intents()["I3"]


# ------------------------------------------------------------------ forced failure control
def set_forced(mode: str) -> None:
    from . import button_task_v14, runner_v14
    from .scene import perturbation_fires as orig

    fn = {"schedule": orig, "on": lambda seed, p: True, "off": lambda seed, p: False}[mode]
    runner_v14.perturbation_fires = fn
    button_task_v14.perturbation_fires = fn
    T.perturbation_fires = fn


# ------------------------------------------------------------------ final-state capture and hold verifiers
FINAL = threading.local()


def install_capture() -> None:
    from . import button_task_v14, runner_v14
    from .env_button import ButtonEnv
    from .env_v8 import EnvV8

    if getattr(runner_v14.EnvV8, "_t17", False):
        return

    class CapBlock(EnvV8):
        _t17 = True

        def close(self):
            c = self.truth_cube_contacts()
            FINAL.block = {"cube_m": [round(float(v), 4) for v in self.truth_cube_position()],
                           "fingers": sorted(c["fingers"]), "receptacle": bool(c["receptacle"]), "table": bool(c["table"]),
                           "grip_command": self.proprio_grip_command(), "tcp_m": [round(float(v), 4) for v in self.proprio_tcp()]}
            super().close()

    class CapButton(ButtonEnv):
        _t17 = True

        def close(self):
            FINAL.button = {**self.truth_labels(), "on_m": self.button["on_m"],
                            "tcp_m": [round(float(v), 4) for v in self.proprio_tcp()]}
            super().close()

    runner_v14.EnvV8 = CapBlock
    button_task_v14.ButtonEnv = CapButton


def hold_block(f: dict) -> bool:
    x, y, z = f["cube_m"]
    inside = abs(x - TRAY_C[0]) <= TRAY_INNER - CUBE_HALF and abs(y - TRAY_C[1]) <= TRAY_INNER - CUBE_HALF
    return bool(f["fingers"]) and f["grip_command"] == "closed" and inside and not f["receptacle"] and not f["table"]


def hold_button(f: dict) -> bool:
    return bool(f["activated"]) and bool(f["pressed_past_activation"])
