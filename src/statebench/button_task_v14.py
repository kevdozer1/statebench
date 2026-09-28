"""Turn 14 Phase 3: the button task for skill selectors, with an explicit observation contract.

Designed from the task definition (``env_button``) and the truth controller, and frozen and hashed before any LLM
reads a button state.

**Skills and executor** (each skill's one-line description is what the prompt shows):
* ``approach_button``: close the fingers if they are open (0.45 s), move above the button's estimated position at
  travel height (1.2 s), then down to the hover height, 30 mm above the estimated touch height (0.8 s).
* ``press``: move the gripper down to PRESS_DEPTH_M = 20 mm below the estimated touch height at its current x, y
  (0.8 s), and hold 0.3 s. The touch height is the estimated cap top plus the fingers' reach below the grasp point
  (FINGER_BELOW_TCP_M = 28 mm, measured in Turn 13).
* ``release_press``: move the gripper straight up to the hover height (0.6 s).
* ``inspect``: hold 0.5 s.
* ``retreat``: rise to travel height (1.0 s) and move to the home pose (1.0 s).
Each skill reports its own completion event (approach_done, press_done, release_press_done, inspect_done,
retreat_done).

**Forced failure:** on seeds where ``scene.perturbation_fires(seed, 0.3)`` fires, the first press is displaced by
FORCED_OFFSET_M = 45 mm along +y before it goes down. The fingers come down beside the button and do not activate
it.

**Observation contract** (what a deployed observer is assumed to have):
* *Known by calibration*, per button unit: its travel, its activation point (60% of travel) and its release point (20%
  of travel), measured once by pressing the unit with a displacement gauge. In simulation they are read from the
  button's definition: **simulator truth, justified as a one-time per-unit calibration**. Also the cap's half-width
  (16 mm) and the fingers' reach below the grasp point (28 mm, measured).
* *From L2 tracking* (the pick-and-place convention): the cap top's centre, with a 5 mm persistent bias in a random
  direction per seed plus 2 mm per-axis jitter at every 50 ms tick. The gripper's own position, opening and command
  come from proprioception (exact).
* *History the state builder keeps:*
  - the rest height of the cap top, the mean of its tracked height over the episode's first 0.5 s, before any
    press; travel is estimated as rest height minus the mean of the last 3 tracked heights (150 ms), so the
    persistent bias cancels;
  - a hysteresis latch on the estimated travel (ON at the activation point, OFF at the release point), updated at
    every tick, with a count of completed ON-OFF cycles;
  - the count of press skills since the last estimated activation.
  The hysteresis means current geometry alone may not decide "activated", so this history is part of the contract.

**Conditions** (same recipe as pick and place; fields listed in CONDITIONS):
* T: button_rel_gripper_m (tracked cap top minus the grasp point), gripper_opening_m, gripper_command; the last 3
  actions with the robot's own events.
* T+H: T + gripper_tcp_m, estimate_age_s, tracking_history (button_rel_gripper_m, button_top_z_m, gripper_tcp_m and
  gripper_opening_m at t_rel 0, -0.25, -0.5, -1.0, -2.0 s), decision_history (the same at the start of each of the
  last 3 actions, with the skill that followed), tracking_reference (the rest height of the cap top) and calibration.
* T+M: T + contact, pressed_past_activation, activated, released (by the declared rules from tracking and history).
* T+M+P: T+M + aligned_for_press, estimate_stale, press_attempts_since_activation.

**Declared rules:**
* contact: the fingertips (grasp point minus 28 mm) are within 3 mm of the tracked cap top or below it, and the
  grasp point is within cap half-width + 4 mm of the tracked cap centre laterally;
* pressed_past_activation: estimated travel >= activation point;
* activated: the hysteresis latch;
* released: at least one completed latch cycle and the latch OFF;
* aligned_for_press: lateral distance to the tracked cap centre < 6 mm and fingertips above the tracked cap top;
* estimate_stale: estimate age > 0.3 s;
* press_attempts_since_activation: the builder's count.

Goal line (the verifier's condition): "Press the button until it activates, then release it and move the gripper
clear." Verifier: ``env_button.ButtonVerifier``, with a 20 s limit. Loop conventions as pick and place: at most
MAX_DECISIONS; the loop stop fires after 6 identical primitives with no change in the verifier's state (cycles,
activated).
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from . import prompt_v14 as PV
from .env_button import BUTTON, ButtonEnv, ButtonVerifier, cap_rest_top_z
from .scene import HOME_TCP, TRAVEL_Z, perturbation_fires

FINGER_BELOW_TCP_M = 0.028
HOVER_ABOVE_TOUCH_M = 0.030
PRESS_DEPTH_M = 0.020
FORCED_OFFSET_M = (0.0, 0.045, 0.0)
BIAS_M, JITTER_M, TICK_S = 0.005, 0.002, 0.05
REST_WINDOW_S = 0.5
TRAVEL_AVG_N = 3
CONTACT_GAP_M = 0.003
CONTACT_XY_EXTRA_M = 0.004
ALIGN_XY_M = 0.006
STALE_AGE_S = 0.3
MAX_DECISIONS = 16
LOOP_STOP_REPEATS = 6
HISTORY_T_REL = (0.0, -0.25, -0.5, -1.0, -2.0)
OWN_EVENTS = ("approach_done", "press_done", "release_press_done", "inspect_done", "retreat_done")

SKILLS = ("approach_button", "press", "release_press", "inspect", "retreat")
SKILL_LINES = {
    "approach_button": "close the fingers if they are open, move above the button's estimated position at travel "
                       "height, then down to a hover height 30 mm above the estimated touch height.",
    "press": "move the gripper straight down to 20 mm below the estimated touch height at its current position, "
             "and hold briefly.",
    "release_press": "move the gripper straight up to the hover height.",
    "inspect": "hold still for a moment and observe again.",
    "retreat": "raise the gripper to travel height and move it to the home pose.",
}
GLOSSARY = {
    "button_rel_gripper_m": "estimated position of the centre of the button's cap top minus the gripper's grasp "
                            "point, in metres, robot base frame (dx forward, dy left, dz up), from camera tracking.",
    "gripper_opening_m": "measured distance between the fingers, in metres.",
    "gripper_command": "the last command sent to the fingers: open or closed.",
    "gripper_tcp_m": "the robot's own measured position of the gripper's grasp point, in metres, robot base frame.",
    "estimate_age_s": "time since the tracking estimate in the state was made, in seconds.",
    "tracking_history": "earlier samples of the tracked quantities, each with its time relative to now (t_rel_s, "
                        "seconds, negative): button_rel_gripper_m, button_top_z_m (tracked height of the cap top, "
                        "metres), gripper_tcp_m and gripper_opening_m.",
    "decision_history": "the same quantities at the start of each of the last 3 actions, oldest first, each with its "
                        "time relative to now and the skill that followed.",
    "tracking_reference": "the tracked height of the cap top at rest, averaged over the first 0.5 s of the episode, "
                          "in metres.",
    "calibration": "calibrated constants of the button, tracking and gripper, as numbers (metres, seconds), named by "
                   "the quantity they describe.",
    "contact": "whether a finger touches the button's cap: contact or no_contact.",
    "pressed_past_activation": "whether the cap is now pressed at or past its activation point: true or false.",
    "activated": "whether the button is activated: true or false.",
    "released": "whether the button has been activated and then released: true or false.",
    "aligned_for_press": "whether the gripper is over the cap's centre, above it: true or false.",
    "estimate_stale": "whether the tracking estimate is out of date: true or false.",
    "press_attempts_since_activation": "number of press actions since the button was last activated (or since the "
                                       "start).",
}
BUTTON_TASK = PV.Task(name="button", goal="Press the button until it activates, then release it and move the gripper clear.",
                      skills=SKILLS, skill_lines=SKILL_LINES, glossary=GLOSSARY)

T = ("button_rel_gripper_m", "gripper_opening_m", "gripper_command")
H = ("gripper_tcp_m", "estimate_age_s", "tracking_history", "decision_history", "tracking_reference", "calibration")
M = ("contact", "pressed_past_activation", "activated", "released")
P = ("aligned_for_press", "estimate_stale", "press_attempts_since_activation")
CONDITIONS = {"T": T, "T+H": T + H, "T+M": T + M, "T+M+P": T + M + P}


def _xyz(v) -> dict:
    return {"dx": round(float(v[0]), 4), "dy": round(float(v[1]), 4), "dz": round(float(v[2]), 4)}


def _pt(v) -> dict:
    return {k: round(float(x), 4) for k, x in zip(("x", "y", "z"), v)}


class Tracker:
    """L2 tracking of the cap top and the state builder's history (rest height, latch, cycles)."""

    def __init__(self, env: ButtonEnv, seed: int):
        self.env = env
        self.rng = np.random.default_rng(5_000_011 + seed)
        d = self.rng.normal(size=3)
        self.bias = BIAS_M * d / np.linalg.norm(d)
        self.samples: list[dict] = []
        self.rest_z = None
        self.latch, self.cycles, self.events = False, 0, []
        self.b = env.button

    def cap_top_true(self) -> np.ndarray:
        return np.array([self.b["x"], self.b["y"], cap_rest_top_z() - float(self.env.truth_labels()["travel_m"])])

    def tick(self, env) -> None:
        est = self.cap_top_true() + self.bias + self.rng.normal(0.0, JITTER_M, size=3)
        s = {"t": round(env.t, 3), "cap_m": est.tolist(), "tcp_m": [float(v) for v in env.proprio_tcp()],
             "opening_m": float(env.proprio_gripper_opening())}
        self.samples.append(s)
        del self.samples[:-80]
        if self.rest_z is None and env.t >= REST_WINDOW_S - 1e-9:
            early = [x["cap_m"][2] for x in self.samples if x["t"] <= REST_WINDOW_S + 1e-9]
            self.rest_z = float(np.mean(early))
        tr = self.travel()
        if tr is not None:
            if not self.latch and tr >= self.b["on_m"]:
                self.latch = True
                self.events.append({"t": s["t"], "event": "est_on"})
            elif self.latch and tr <= self.b["off_m"]:
                self.latch = False
                self.cycles += 1
                self.events.append({"t": s["t"], "event": "est_off"})

    def travel(self) -> float | None:
        if self.rest_z is None or not self.samples:
            return None
        return float(self.rest_z - np.mean([x["cap_m"][2] for x in self.samples[-TRAVEL_AVG_N:]]))

    def at(self, t: float) -> dict | None:
        best = None
        for x in self.samples:
            if x["t"] <= t + 1e-9:
                best = x
        return best


class ButtonExecutor:
    def __init__(self, env: ButtonEnv, events: list, tracker: Tracker, forced: bool):
        self.env, self.events, self.tr, self.forced, self.presses = env, events, tracker, forced, 0

    def _emit(self, name):
        self.events.append({"t": round(self.env.t, 3), "event": name})

    def _touch_z(self) -> float:
        return float(self.tr.samples[-1]["cap_m"][2]) + FINGER_BELOW_TCP_M

    def execute(self, skill: str) -> None:
        e = self.env
        if skill == "approach_button":
            if e.proprio_grip_command() != "closed":
                e.set_grip(True, 0.45)
            cap = self.tr.samples[-1]["cap_m"]
            e.move_to([cap[0], cap[1], TRAVEL_Z], yaw=0.0, seconds=1.2)
            e.move_to([cap[0], cap[1], self._touch_z() + HOVER_ABOVE_TOUCH_M], seconds=0.8)
            self._emit("approach_done")
        elif skill == "press":
            tcp = e.proprio_tcp()
            xy = np.array(tcp[:2], dtype=float)
            self.presses += 1
            if self.forced and self.presses == 1:
                xy = xy + np.array(FORCED_OFFSET_M[:2])
                e.move_to([xy[0], xy[1], tcp[2]], seconds=0.3)
            e.move_to([xy[0], xy[1], self._touch_z() - PRESS_DEPTH_M], seconds=0.8)
            e.hold(0.3)
            self._emit("press_done")
        elif skill == "release_press":
            tcp = e.proprio_tcp()
            e.move_to([tcp[0], tcp[1], self._touch_z() + HOVER_ABOVE_TOUCH_M], seconds=0.6)
            self._emit("release_press_done")
        elif skill == "inspect":
            e.hold(0.5)
            self._emit("inspect_done")
        elif skill == "retreat":
            tcp = e.proprio_tcp()
            e.move_to([tcp[0], tcp[1], TRAVEL_Z], seconds=1.0)
            e.move_to(HOME_TCP, yaw=0.0, seconds=1.0)
            self._emit("retreat_done")
        else:
            e.hold(0.5)


def calibration(b: dict) -> dict:
    return {"button_travel_m": round(b["travel_m"], 5), "activation_travel_m": round(b["on_m"], 5),
            "release_travel_m": round(b["off_m"], 5), "cap_half_width_m": BUTTON["cap_half"][0],
            "finger_below_grasp_point_m": FINGER_BELOW_TCP_M, "contact_max_gap_m": CONTACT_GAP_M,
            "contact_lateral_extra_m": CONTACT_XY_EXTRA_M, "align_max_xy_m": ALIGN_XY_M,
            "travel_average_samples": TRAVEL_AVG_N, "tracking_period_s": TICK_S, "estimate_max_age_s": STALE_AGE_S}


def rules_fields(tr: Tracker, tcp: np.ndarray, cap_now: np.ndarray, presses_since_on: int) -> dict:
    fing = float(tcp[2]) - FINGER_BELOW_TCP_M
    lat = float(np.linalg.norm(tcp[:2] - cap_now[:2]))
    travel = tr.travel()
    return {"contact": "contact" if (fing <= cap_now[2] + CONTACT_GAP_M and lat < BUTTON["cap_half"][0] + CONTACT_XY_EXTRA_M)
            else "no_contact",
            "pressed_past_activation": bool(travel is not None and travel >= tr.b["on_m"]),
            "activated": bool(tr.latch), "released": bool(tr.cycles >= 1 and not tr.latch),
            "aligned_for_press": bool(lat < ALIGN_XY_M and fing > cap_now[2] + CONTACT_GAP_M),
            "estimate_stale": False, "press_attempts_since_activation": int(presses_since_on)}


def truth_fields(env: ButtonEnv, tcp: np.ndarray, presses_since_true_on: int) -> dict:
    lab = env.truth_labels()
    cap = np.array([env.button["x"], env.button["y"], cap_rest_top_z() - lab["travel_m"]])
    fing = float(tcp[2]) - FINGER_BELOW_TCP_M
    return {"contact": "contact" if lab["contact"] else "no_contact",
            "pressed_past_activation": bool(lab["pressed_past_activation"]), "activated": bool(lab["activated"]),
            "released": bool(lab["cycles"] >= 1 and not lab["activated"]),
            "aligned_for_press": bool(float(np.linalg.norm(tcp[:2] - cap[:2])) < ALIGN_XY_M and fing > cap[2] + CONTACT_GAP_M),
            "estimate_stale": False, "press_attempts_since_activation": int(presses_since_true_on)}


def _sample(s: dict, t_now: float) -> dict:
    cap, tcp = np.asarray(s["cap_m"]), np.asarray(s["tcp_m"])
    return {"t_rel_s": round(s["t"] - t_now, 3), "button_rel_gripper_m": _xyz(cap - tcp),
            "button_top_z_m": round(float(cap[2]), 4), "gripper_tcp_m": _pt(tcp),
            "gripper_opening_m": round(float(s["opening_m"]), 4)}


def run_episode(seed: int, condition: str, reader, *, task=BUTTON_TASK, keep_prompts: bool = False,
                truth_selector: bool = False) -> dict:
    started = time.perf_counter()
    fields = CONDITIONS[condition]
    env = ButtonEnv(seed)
    tr = Tracker(env, seed)
    ver = ButtonVerifier(env)
    tick = {"next": 0.0}

    def hook(e):
        if e.t + 1e-9 >= tick["next"]:
            tick["next"] = e.t + TICK_S
            tr.tick(e)

    env.step_hooks += [ver.update, hook]
    hook(env)
    env.hold(REST_WINDOW_S + 0.05)  # the rest-height window, before the first decision
    fired = perturbation_fires(seed, 0.3)
    events: list[dict] = []
    ex = ButtonExecutor(env, events, tr, fired)
    steps, recent, last_actions, dsamples = [], [], [], []
    presses_since_on, presses_since_true_on = 0, 0
    last_est_on, last_true_cycles_on = 0, False
    loop_stopped = False
    for index in range(MAX_DECISIONS):
        if ver.status != "running":
            break
        tcp = np.asarray(env.proprio_tcp(), dtype=float)
        cap_now = np.asarray(tr.samples[-1]["cap_m"])
        rf = rules_fields(tr, tcp, cap_now, presses_since_on)
        tf = truth_fields(env, tcp, presses_since_true_on)
        view = {"button_rel_gripper_m": _xyz(cap_now - tcp), "gripper_opening_m": round(float(env.proprio_gripper_opening()), 4),
                "gripper_command": env.proprio_grip_command()}
        for k in M + P:
            if k in fields:
                view[k] = rf[k]
        if "tracking_history" in fields:
            view["gripper_tcp_m"] = _pt(tcp)
            view["estimate_age_s"] = 0.0
            hist = [tr.at(tr.samples[-1]["t"] + dt) for dt in HISTORY_T_REL]
            view["tracking_history"] = [_sample(s, env.t) for s in hist if s is not None]
            view["decision_history"] = [dict(_sample(s, env.t), skill=sk) for s, sk in dsamples[-3:]]
            view["tracking_reference"] = round(float(tr.rest_z), 4)
            view["calibration"] = calibration(env.button)
        view = {k: view[k] for k in fields}
        prompt = PV.build_prompt(task, view, last_actions)
        dec = reader.decide(prompt, view) if not truth_selector else reader.decide(prompt, tf)
        skill = dec["skill"]
        n_ev = len(events)
        dsamples.append((dict(tr.samples[-1]), skill))
        ex.execute(skill)
        if skill == "press":
            presses_since_on += 1
            presses_since_true_on += 1
        if tr.latch and not last_est_on:
            presses_since_on = 0
        last_est_on = tr.latch
        if env.on and not last_true_cycles_on:
            presses_since_true_on = 0
        last_true_cycles_on = env.on
        own = [e["event"] for e in events[n_ev:] if e["event"] in OWN_EVENTS]
        last_actions.append({"skill": skill, "events": own})
        rec = {"index": index, "t": round(env.t, 3), "chosen": skill, "executed": skill, "valid": dec["valid"],
               "raw_skill": dec["raw_skill"], "error": dec["error"], "tokens_in": dec["tokens_in"],
               "tokens_out": dec["tokens_out"], "latency_s": dec["latency_s"], "cost_usd": dec["cost_usd"],
               "loop_exit": None, "view": {k: v for k, v in view.items() if k not in ("tracking_history", "decision_history", "calibration")},
               "rules_fields": rf, "truth_fields": tf, "own_events": own,
               "verifier_after": {"status": ver.status, "cycles": env.cycles, "activated": env.on}}
        if keep_prompts:
            rec["prompt"] = prompt
        steps.append(rec)
        recent.append((skill, env.cycles, env.on))
        recent = recent[-LOOP_STOP_REPEATS:]
        if len(recent) == LOOP_STOP_REPEATS and len(set(recent)) == 1:
            ver.status, ver.reason = "failure", "loop: the same primitive 6 times with no change in the button state"
            loop_stopped = True
            break
    while ver.status == "running" and env.t < 20.0 and steps and steps[-1]["executed"] == "retreat":
        env.hold(0.1)
    if ver.status == "running":
        ver.status, ver.reason = "failure", "decision loop ended without a verified press cycle"
    rep = ver.report()
    env.close()
    lat = [s["latency_s"] for s in steps]
    return {"seed": seed, "condition": condition, "reader": getattr(reader, "name", "?"), "task": "button",
            "runner_version": "statebench-button/v14", "prompt_sha256": PV.prompt_hash(), "button": env.button,
            "success": rep["status"] == "success", "verifier": rep, "time_to_success_s": rep["t_success"],
            "loop_stopped": loop_stopped, "perturbation_fired": fired, "decision_steps": len(steps), "loop_exits": [],
            "invalid_answers": sum(1 for s in steps if not s["valid"]),
            "tokens_in": sum(s["tokens_in"] or 0 for s in steps), "tokens_out": sum(s["tokens_out"] or 0 for s in steps),
            "cost_usd": round(sum(s["cost_usd"] or 0 for s in steps), 6),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "wall_seconds": round(time.perf_counter() - started, 3), "steps": steps}


# ------------------------------------------------------------------ deterministic selectors
def _table(p: dict) -> tuple[str, str]:
    if p["released"]:
        return "retreat", "released"
    if p["activated"]:
        return "release_press", "activated"
    if p["aligned_for_press"]:
        return "press", "aligned_for_press"
    return "approach_button", "otherwise"


class RulesButton:
    """The table on T+M+P fields (or on truth fields, as the truth controller)."""

    name = "rules"

    def __init__(self, task):
        self.task = task

    def decide(self, prompt, view):
        t0 = time.perf_counter()
        skill, rule = _table(view)
        return {"skill": skill, "valid": True, "raw_skill": skill, "tokens_in": 0, "tokens_out": 0,
                "latency_s": round(time.perf_counter() - t0, 4), "cost_usd": 0.0, "error": None, "raw": {"rule": rule}}


def derive_button_from_th(view: dict) -> dict:
    """The table's inputs recomputed from the T+H numbers only (the latch over the samples provided)."""
    c = view["calibration"]
    rest = view["tracking_reference"]
    samples = sorted(view["decision_history"] + view["tracking_history"], key=lambda s: s["t_rel_s"])
    latch, cycles = False, 0
    for s in samples:
        travel = rest - s["button_top_z_m"]
        if not latch and travel >= c["activation_travel_m"]:
            latch = True
        elif latch and travel <= c["release_travel_m"]:
            latch, cycles = False, cycles + 1
    rg = view["button_rel_gripper_m"]
    lat = math.hypot(rg["dx"], rg["dy"])
    fing_above = (-rg["dz"]) - c["finger_below_grasp_point_m"]
    return {"released": bool(cycles >= 1 and not latch), "activated": bool(latch),
            "aligned_for_press": bool(lat < c["align_max_xy_m"] and fing_above > c["contact_max_gap_m"])}


class RulesButtonTH:
    name = "rules"

    def __init__(self, task):
        self.task = task

    def decide(self, prompt, view):
        t0 = time.perf_counter()
        skill, rule = _table(derive_button_from_th(view))
        return {"skill": skill, "valid": True, "raw_skill": skill, "tokens_in": 0, "tokens_out": 0,
                "latency_s": round(time.perf_counter() - t0, 4), "cost_usd": 0.0, "error": None, "raw": {"rule": rule}}


def make_rules(task, condition):
    return RulesButtonTH(task) if condition == "T+H" else RulesButton(task)


def field_accuracy(rows: list[dict]) -> dict:
    """Per M and P field: share of decisions where the rules value equals the truth value."""
    out = {}
    for k in M + P:
        n = ok = 0
        for r in rows:
            for s in r["steps"]:
                n += 1
                ok += s["rules_fields"][k] == s["truth_fields"][k]
        out[k] = {"agree": ok, "n": n, "accuracy": round(ok / n, 4) if n else None}
    return out


if __name__ == "__main__":
    import sys

    # the truth controller: the same table on truth fields, through the same executor
    lo, hi = (int(v) for v in sys.argv[1].split(":"))
    rows = [run_episode(s, "T+M+P", RulesButton(BUTTON_TASK), truth_selector=True) for s in range(lo, hi)]
    for r in rows:
        print(r["seed"], r["success"], r["perturbation_fired"], [s["executed"] for s in r["steps"]], r["verifier"]["reason"],
              round(r["wall_seconds"], 2))
    print(json.dumps({"success": sum(r["success"] for r in rows), "n": len(rows),
                      "forced": f"{sum(r['success'] for r in rows if r['perturbation_fired'])}/{sum(r['perturbation_fired'] for r in rows)}",
                      "accuracy": field_accuracy(rows)}, indent=0))
