"""Turn 14: pick and place with a skill selector reading one condition's state view (prompt v14).

The episode loop is ``runner_v13``'s, unchanged (L2 tracking: 5 mm persistent bias plus 2 mm per-axis jitter per
50 ms tick; forced miss on perturbed seeds; loop exit and loop stop; at most 24 decisions; the executor always acts
on the same rules-derived L2 state). Changes:
* the prompt is ``prompt_v14`` (the goal line states the verifier's condition);
* new conditions T+H, T+M+P minus one field, T+M plus one field (``CONDITIONS``);
* every decision logs the view the reader saw (Turn 13 did not log the P fields);
* the tracking stream also records the gripper opening, for the T+H history.

Conditions (every field listed in ``CONDITIONS`` and in the notes before any run):
* ``T``: as Turn 13.
* ``T+H`` (memory and calibration without interpretation): T plus ``gripper_tcp_m`` (the robot's own grasp-point
  position), ``estimate_age_s``, ``tracking_history`` (the tracked target_rel_gripper_m, target_rel_receptacle_m,
  gripper_opening_m and gripper_tcp_m at t_rel 0, -0.25, -0.5, -1.0 and -2.0 s, from the same 50 ms tracking
  stream the state builder reads; the rules' widest windows are 0.25 s of motion and the 2 s event horizon),
  ``decision_history`` (the same quantities at the start of each of the last 3 actions, with the skill that
  followed) and ``calibration`` (every constant and threshold the meaning and procedure rules use, as numbers).
  No derived booleans.
* ``T+M``, ``T+M+P``: as Turn 13.
* ``T+M+P-A`` / ``-S`` / ``-C``: T+M+P without aligned_for_grasp / estimate_stale / close_attempts_since_lift.
* ``T+M+A`` / ``+S`` / ``+C``: T+M plus exactly that one field.
"""

from __future__ import annotations

import time

import numpy as np

from . import prompt_v13 as V13
from . import prompt_v14 as PV
from . import rules as prereg1
from .env_v8 import EnvV8
from .executor_v6 import ExecutorV6
from .observation_e1 import (CONTACT_XY_M, CONTACT_Z_M, FINGERS_ON_SOMETHING_M, MOTION_MATCH_M, MOTION_MIN_M,
                             MOTION_WINDOW_S, NO_CONTACT_M, RELEASE_OPENING_M, build_estimated_state, estimated_events)
from .observation_e1_1 import lift_diverged_interp, moving_interp
from .observations import HistoryBuffer, build_compact_truth, build_proprio
from .perception_e1 import CONSTANTS
from .predicates import STALE_AGE_S
from .predicates_v3 import STALE_MOTION_M
from .predicates_v4 import CLOSED_ON_NOTHING_M, HELD_OFFSET_M, build_predicates_v4
from .reference import ALIGN_XY_M, ALIGN_YAW_RAD, ALIGN_Z_M, OVER_RECEPTACLE_M
from .reference_v8 import heights
from .representations_v2 import CONDITIONS_V2_BY_NAME, prepare_v2
from .runner_meaning_v10 import GEOMETRY, TICK_S, read_fields
from .runner_v2 import completion_labels
from .runner_v6 import LOOP_REPEATS, LOOP_TCP_M, MISS_OFFSET
from .runner_v13 import M_NAMES, OWN_EVENTS, _j, build_view
from .scene import perturbation_fires
from .scene_v8 import VARIANTS
from .verify import RUNNING, SUCCESS, Verifier

RUNNER_VERSION = "statebench-runner/v14"
LOOP_STOP_REPEATS = 6
PLACE_CLEARANCE_M = 0.060
T, M, P = V13.FIELDS_T, V13.FIELDS_M, V13.FIELDS_P
H = ("gripper_tcp_m", "estimate_age_s", "tracking_history", "decision_history", "calibration")
CONDITIONS = {
    "T": T, "T+H": T + H, "T+M": T + M, "T+M+P": T + M + P,
    "T+M+P-A": T + M + ("estimate_stale", "close_attempts_since_lift"),
    "T+M+P-S": T + M + ("aligned_for_grasp", "close_attempts_since_lift"),
    "T+M+P-C": T + M + ("aligned_for_grasp", "estimate_stale"),
    "T+M+A": T + M + ("aligned_for_grasp",), "T+M+S": T + M + ("estimate_stale",),
    "T+M+C": T + M + ("close_attempts_since_lift",),
}
HISTORY_T_REL = (0.0, -0.25, -0.5, -1.0, -2.0)


def calibration(pconst: dict) -> dict:
    return {"object_size_m": pconst["object_size_m"], "grasp_height_offset_m": pconst["grasp_offset_m"],
            "gripper_opening_closed_on_nothing_m": CLOSED_ON_NOTHING_M,
            "gripper_opening_held_offset_m": HELD_OFFSET_M, "release_opening_m": RELEASE_OPENING_M,
            "fingers_on_something_opening_m": FINGERS_ON_SOMETHING_M, "contact_max_xy_m": CONTACT_XY_M,
            "contact_max_z_m": CONTACT_Z_M, "no_contact_min_distance_m": NO_CONTACT_M,
            "motion_window_s": MOTION_WINDOW_S, "motion_min_gripper_travel_m": MOTION_MIN_M,
            "motion_match_m": MOTION_MATCH_M, "align_max_xy_m": ALIGN_XY_M, "align_max_z_error_m": ALIGN_Z_M,
            "align_max_yaw_rad": ALIGN_YAW_RAD, "receptacle_max_xy_m": OVER_RECEPTACLE_M,
            "receptacle_max_dz_m": PLACE_CLEARANCE_M, "estimate_max_age_s": STALE_AGE_S,
            "stale_arm_motion_m": STALE_MOTION_M}


def _xyz(v) -> dict:
    return {"dx": round(float(v[0]), 4), "dy": round(float(v[1]), 4), "dz": round(float(v[2]), 4)}


def _sample(entry: dict, floor: np.ndarray, t_now: float) -> dict:
    cube, tcp = np.asarray(entry["cube_m"]), np.asarray(entry["tcp_m"])
    return {"t_rel_s": round(entry["t"] - t_now, 3), "target_rel_gripper_m": _xyz(cube - tcp),
            "target_rel_receptacle_m": _xyz(cube - floor), "gripper_opening_m": round(float(entry["opening_m"]), 4),
            "gripper_tcp_m": {k: round(float(v), 4) for k, v in zip(("x", "y", "z"), tcp)}}


def _interp(track: list[dict], t: float) -> dict | None:
    if not track or t < track[0]["t"] - 1e-9:
        return None
    for a, b in zip(track, track[1:]):
        if a["t"] - 1e-9 <= t <= b["t"] + 1e-9:
            w = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            return {"t": t, **{k: ((1 - w) * np.asarray(a[k]) + w * np.asarray(b[k])).tolist()
                               for k in ("cube_m", "tcp_m")}, "opening_m": (1 - w) * a["opening_m"] + w * b["opening_m"]}
    return dict(track[-1]) if t >= track[-1]["t"] - 1e-9 else None


def run_episode(seed: int, condition: str, reader, *, task=PV.PICK_PLACE_V14, variant: str = "V0",
                geometry: str = "L2", max_steps: int = prereg1.MAX_DECISION_STEPS, keep_prompts: bool = False) -> dict:
    started = time.perf_counter()
    fields = CONDITIONS[condition]
    env = EnvV8(seed, variant=variant)
    events: list[dict] = []
    diverged: list[dict] = []
    verifier = Verifier(env)
    cfg = VARIANTS[variant]
    constants = dict(CONSTANTS, cube_edge_m=2 * cfg.cube_half)
    floor = np.array([*constants["tray_center_xy_m"], constants["tray_floor_top_m"]])
    grasp_z, place_z = heights(variant)
    pconst = {"grasp_offset_m": round(grasp_z - cfg.cube_half, 4), "object_size_m": 2 * cfg.cube_half}
    bias_m, jitter_m = GEOMETRY[geometry]
    grng = np.random.default_rng(5_000_011 + seed)
    d = grng.normal(size=3)
    bias = bias_m * d / np.linalg.norm(d)
    erng = np.random.default_rng(9_000_013 + seed)
    history, est_history = HistoryBuffer(), HistoryBuffer()
    track: list[dict] = []
    rcond = CONDITIONS_V2_BY_NAME["R4j"]
    tick = {"next": 0.0}

    def truth_fields() -> dict:
        fingers = env.truth_cube_contacts()["fingers"]
        closed = env.proprio_grip_command() == "closed"
        compact = build_compact_truth(env, events)
        preds = build_predicates_v4(compact, build_proprio(env), [s for _, s in history.snapshots()], pconst)["values"]
        return {"contact": "contact" if fingers else "no_contact",
                "grip": ("on_object" if fingers else "empty") if closed else "open",
                "held": compact["relations"]["target_moving_with_gripper"],
                "released": preds["released_at_receptacle"], "at_receptacle": preds["object_at_receptacle"]}

    def noisy_cube():
        return np.asarray(env.truth_cube_position(), dtype=float) + bias + grng.normal(0.0, jitter_m, size=3)

    def hook(env_):
        if env_.t + 1e-9 >= tick["next"]:
            tick["next"] = env_.t + TICK_S
            history.push(env_.t, build_compact_truth(env_, events))
            track.append({"t": round(env_.t, 3), "cube_m": noisy_cube().tolist(),
                          "tcp_m": [float(v) for v in env_.proprio_tcp()],
                          "opening_m": float(env_.proprio_gripper_opening())})
            del track[:-60]

    env.step_hooks += [verifier.update, hook]
    hook(env)
    fired = perturbation_fires(seed, prereg1.PERTURBATION_PROBABILITY)
    offset = np.array([0.0, 0.0, MISS_OFFSET.get(variant, 0.035)]) if fired else None
    executor = ExecutorV6(env, events, offset, grasp_z, place_z)
    rng = np.random.default_rng(seed + 40_000)
    steps, loop_exits, recent, loop_run, last_actions, decision_samples = [], [], [], [], [], []
    loop_stopped = False
    for index in range(max_steps):
        if verifier.status != RUNNING:
            break
        proprio = build_proprio(env)
        compact = build_compact_truth(env, events)
        history.push(env.t, compact)
        est = {"cube_m": noisy_cube().tolist(), "yaw_mod90_rad": float(env.truth_cube_yaw()) % (np.pi / 2),
               "t_est": env.t, "fresh": True, "visible_fraction": 1.0}
        base = build_estimated_state(est, proprio, proprio, estimated_events(events, diverged), track, constants)
        if track:
            base["relations"]["target_moving_with_gripper"] = moving_interp(track, track[-1]["t"])
        est_history.push(env.t, base)
        state, h, _ = prepare_v2(rcond, base, est_history, env.t, rng, proprio=proprio)
        block = build_predicates_v4(state, proprio, h, pconst)
        preds = block["values"]
        state["predicates"] = block
        truth = truth_fields()
        rules_m = read_fields(state, preds)
        erng.uniform(size=len(M_NAMES))  # the Turn 13 per-decision draws, kept so the streams stay aligned
        meaning = rules_m if any(f in fields for f in M) else None
        full = build_view("T+M+P", state, preds, proprio, rules_m)
        view = {k: full[k] for k in fields if k in full}
        tcp_now = np.asarray(proprio["tcp_position_m"], dtype=float)
        now_sample = {"t": env.t, "cube_m": est["cube_m"], "tcp_m": tcp_now.tolist(),
                      "opening_m": float(proprio["gripper_opening_m"])}
        if "tracking_history" in fields:
            view["gripper_tcp_m"] = {k: round(float(v), 4) for k, v in zip(("x", "y", "z"), tcp_now)}
            view["estimate_age_s"] = round(float(env.t - est["t_est"]), 3)
            t_last = track[-1]["t"] if track else env.t
            hist = [_interp(track, t_last + dt) for dt in HISTORY_T_REL]
            view["tracking_history"] = [_sample(e, floor, env.t) for e in hist if e is not None]
            view["decision_history"] = [dict(_sample(e, floor, env.t), skill=sk) for e, sk in decision_samples[-3:]]
            view["calibration"] = calibration(pconst)
            view = {k: view[k] for k in fields}
        prompt = PV.build_prompt(task, view, last_actions)
        dec = reader.decide(prompt, view)
        chosen = dec["skill"]
        primitive = chosen
        vkey = PV.json.dumps(view, sort_keys=True)
        recent.append((chosen, vkey, tcp_now))
        recent = recent[-LOOP_REPEATS:]
        exit_here = None
        if (len(recent) == LOOP_REPEATS and chosen != "approach" and len({r[0] for r in recent}) == 1
                and len({r[1] for r in recent}) == 1
                and max(float(np.linalg.norm(r[2] - recent[0][2])) for r in recent) < LOOP_TCP_M):
            primitive = "approach"
            exit_here = {"index": index, "repeated": chosen}
            loop_exits.append(exit_here)
            recent = []
        t0 = env.t
        n_ev = len(events)
        executor.execute(primitive, state, h)
        decision_samples.append((now_sample, primitive))
        if primitive == "test_lift":
            if lift_diverged_interp(track, t0, env.t):
                diverged.append({"t": round(env.t, 3), "event": "target_motion_diverged"})
        own = [e["event"] for e in events[n_ev:] if e.get("event") in OWN_EVENTS]
        last_actions.append({"skill": primitive, "events": own})
        after = completion_labels(verifier)
        rec = {"index": index, "t": round(t0, 3), "chosen": chosen, "executed": primitive, "valid": dec["valid"],
               "raw_skill": dec["raw_skill"], "error": dec["error"], "tokens_in": dec["tokens_in"],
               "tokens_out": dec["tokens_out"], "latency_s": dec["latency_s"], "cost_usd": dec["cost_usd"],
               "loop_exit": exit_here, "view": view if "tracking_history" not in fields else {k: view[k] for k in view if k not in ("tracking_history", "decision_history", "calibration")},
               "meaning_rules": {f: _j(rules_m[f]) for f in M_NAMES}, "truth": {f: _j(truth[f]) for f in M_NAMES},
               "p_rules": {k: _j(full[k]) for k in P}, "own_events": own,
               "verifier_after": {"status": verifier.status, "conditions": after["conditions"]}}
        if keep_prompts:
            rec["prompt"] = prompt
        steps.append(rec)
        loop_run.append((primitive, tuple(sorted(after["conditions"].items()))))
        loop_run = loop_run[-LOOP_STOP_REPEATS:]
        if len(loop_run) == LOOP_STOP_REPEATS and len(set(loop_run)) == 1:
            verifier.finish("loop: the same primitive 6 times with no change in the verifier conditions")
            loop_stopped = True
            break
        if verifier.status != RUNNING:
            break
    if verifier.status == RUNNING:
        verifier.finish("decision loop ended without a verified placement")
    report = verifier.report()
    env.close()
    success = report["status"] == SUCCESS
    lat = [s["latency_s"] for s in steps]
    return {"seed": seed, "condition": condition, "reader": getattr(reader, "name", "?"), "task": task.name,
            "variant": variant, "geometry": geometry, "runner_version": RUNNER_VERSION,
            "prompt_sha256": PV.prompt_hash(), "success": success, "verifier": report,
            "time_to_success_s": report["t_success"] if success else None, "loop_stopped": loop_stopped,
            "perturbation_fired": fired, "decision_steps": len(steps), "loop_exits": loop_exits,
            "invalid_answers": sum(1 for s in steps if not s["valid"]),
            "tokens_in": sum(s["tokens_in"] or 0 for s in steps), "tokens_out": sum(s["tokens_out"] or 0 for s in steps),
            "cost_usd": round(sum(s["cost_usd"] or 0 for s in steps), 6),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "wall_seconds": round(time.perf_counter() - started, 3), "steps": steps}
