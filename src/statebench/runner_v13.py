"""Turn 13 Phase 2: pick and place with an LLM (or rules) planner reading one condition's state view.

The episode loop is ``runner_meaning_v10``'s (Turn 8 L2 executor convention): L2 tracking geometry (5 mm persistent
bias in a random direction per seed plus 2 mm per-axis jitter per 50 ms tick; yaw exact), the forced miss on
perturbed seeds (``scene.perturbation_fires``, p 0.3), the loop exit (the same non-approach choice 3 times on the
same view with the gripper still is replaced by approach) and the loop stop (the same primitive 6 times with no
change in the verifier conditions ends the episode), at most 24 decisions.

The executor always acts on the same rules-derived L2 state; conditions change only what the planner reads. The
planner is asked once per decision (no counterfactual re-decision this turn).

Conditions (fields in ``prompt_v13``):
* ``T``: geometry numbers from L2 tracking, the measured gripper opening, the commanded gripper state, and the
  completion events of the robot's own commands (in the last-actions list). Nothing from truth or object motion.
* ``T+M``: plus the 5 meaning fields by the v0.4 rules from the same L2 tracking.
* ``T+M+P``: plus aligned_for_grasp, estimate_stale and close attempts since the last lift (v0.4 predicates).
* ``T+M late``: T+M with every meaning field as the same rules computed it 0.4 s earlier (the pipeline is run at
  every 50 ms tick on the tracking stream with its own history and RNG, so decision-time streams are unchanged).
* ``T+M flip``: T+M with every meaning field flipped with probability 0.1 per decision (one draw per field from a
  per-episode RNG; ``unknown`` and an open grip are never flipped).
* ``T+M truth``: T+M with the meaning fields from simulator truth.
"""

from __future__ import annotations

import time

import numpy as np

from . import prompt_v13 as PV
from . import rules as prereg1
from .env_v8 import EnvV8
from .executor_v6 import ExecutorV6
from .observation_e1 import build_estimated_state, estimated_events
from .observation_e1_1 import lift_diverged_interp, moving_interp
from .observations import HistoryBuffer, build_compact_truth, build_proprio
from .perception_e1 import CONSTANTS
from .predicates_v4 import build_predicates_v4
from .reference_v8 import heights
from .representations_v2 import CONDITIONS_V2_BY_NAME, prepare_v2
from .runner_meaning_v10 import GEOMETRY, TICK_S, apply_error, read_fields
from .runner_v2 import completion_labels
from .runner_v6 import LOOP_REPEATS, LOOP_TCP_M, MISS_OFFSET
from .scene import perturbation_fires
from .scene_v8 import VARIANTS
from .verify import RUNNING, SUCCESS, Verifier

RUNNER_VERSION = "statebench-runner/v13"
LOOP_STOP_REPEATS = 6
LATE_S = 0.4
FLIP_P = 0.1
CONDITIONS = {
    "T": {"fields": PV.FIELDS_T, "meaning": None},
    "T+M": {"fields": PV.FIELDS_T + PV.FIELDS_M, "meaning": "rules"},
    "T+M+P": {"fields": PV.FIELDS_T + PV.FIELDS_M + PV.FIELDS_P, "meaning": "rules"},
    "T+M late": {"fields": PV.FIELDS_T + PV.FIELDS_M, "meaning": "late"},
    "T+M flip": {"fields": PV.FIELDS_T + PV.FIELDS_M, "meaning": "flip"},
    "T+M truth": {"fields": PV.FIELDS_T + PV.FIELDS_M, "meaning": "truth"},
}
#: runner field name -> view field name
M_NAMES = {"contact": "contact", "grip": "grip", "held": "object_moving_with_gripper",
           "at_receptacle": "object_at_receptacle", "released": "released_at_receptacle"}
OWN_EVENTS = ("approach_done", "close_attempted", "lift_started", "lift_done", "transport_done", "release_done",
              "inspect_done", "retreat_done", "primitive_rejected")


def _j(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def build_view(condition: str, state: dict, preds: dict, proprio: dict, meaning: dict | None) -> dict:
    g = state["geometry"]
    tg, tr = g.get("target_rel_gripper") or {}, g.get("target_rel_receptacle") or {}
    view = {"target_rel_gripper_m": {k: round(float(tg[k]), 4) for k in ("dx", "dy", "dz") if k in tg},
            "target_yaw_rel_gripper_rad": round(float(tg["dyaw_rad"]), 3) if tg.get("dyaw_rad") is not None else None,
            "target_rel_receptacle_m": {k: round(float(tr[k]), 4) for k in ("dx", "dy", "dz") if k in tr},
            "gripper_opening_m": round(float(g["gripper_opening_m"]), 4),
            "gripper_command": proprio["gripper_command"]}
    fields = CONDITIONS[condition]["fields"]
    if meaning is not None:
        for f, name in M_NAMES.items():
            view[name] = _j(meaning[f])
    if "aligned_for_grasp" in fields:
        view["aligned_for_grasp"] = _j(preds.get("aligned_for_grasp"))
        view["estimate_stale"] = _j(preds.get("estimate_stale"))
        view["close_attempts_since_lift"] = int(preds.get("close_attempts_since_last_lift") or 0)
    return {k: view[k] for k in fields}


def run_episode(seed: int, condition: str, reader, *, variant: str = "V0", geometry: str = "L2",
                max_steps: int = prereg1.MAX_DECISION_STEPS, keep_prompts: bool = False) -> dict:
    started = time.perf_counter()
    cond = CONDITIONS[condition]
    env = EnvV8(seed, variant=variant)
    events: list[dict] = []
    diverged: list[dict] = []
    verifier = Verifier(env)
    cfg = VARIANTS[variant]
    constants = dict(CONSTANTS, cube_edge_m=2 * cfg.cube_half)
    grasp_z, place_z = heights(variant)
    pconst = {"grasp_offset_m": round(grasp_z - cfg.cube_half, 4), "object_size_m": 2 * cfg.cube_half}
    bias_m, jitter_m = GEOMETRY[geometry]
    grng = np.random.default_rng(5_000_011 + seed)
    d = grng.normal(size=3)
    bias = bias_m * d / np.linalg.norm(d)
    erng = np.random.default_rng(9_000_013 + seed)
    history, est_history, late_history = HistoryBuffer(), HistoryBuffer(), HistoryBuffer()
    track: list[dict] = []
    late_buf: list[tuple[float, dict]] = []
    late_rng = np.random.default_rng(7_000_003 + seed)
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

    def rules_meaning_at_tick(env_):
        """The decision-time rules pipeline on the tracking stream's latest sample (late condition only)."""
        pr = build_proprio(env_)
        est = {"cube_m": track[-1]["cube_m"], "yaw_mod90_rad": float(env_.truth_cube_yaw()) % (np.pi / 2),
               "t_est": env_.t, "fresh": True, "visible_fraction": 1.0}
        base = build_estimated_state(est, pr, pr, estimated_events(events, diverged), track, constants)
        base["relations"]["target_moving_with_gripper"] = moving_interp(track, track[-1]["t"])
        late_history.push(env_.t, base)
        st, h, _ = prepare_v2(rcond, base, late_history, env_.t, late_rng, proprio=pr)
        return read_fields(st, build_predicates_v4(st, pr, h, pconst)["values"])

    def hook(env_):
        if env_.t + 1e-9 >= tick["next"]:
            tick["next"] = env_.t + TICK_S
            history.push(env_.t, build_compact_truth(env_, events))
            track.append({"t": round(env_.t, 3), "cube_m": noisy_cube().tolist(),
                          "tcp_m": [float(v) for v in env_.proprio_tcp()]})
            del track[:-60]
            if cond["meaning"] == "late":
                late_buf.append((env_.t, rules_meaning_at_tick(env_)))
                del late_buf[:-40]

    env.step_hooks += [verifier.update, hook]
    hook(env)
    fired = perturbation_fires(seed, prereg1.PERTURBATION_PROBABILITY)
    offset = np.array([0.0, 0.0, MISS_OFFSET.get(variant, 0.035)]) if fired else None
    executor = ExecutorV6(env, events, offset, grasp_z, place_z)
    rng = np.random.default_rng(seed + 40_000)
    steps, loop_exits, recent, loop_run, last_actions = [], [], [], [], []
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
        u = {f: float(erng.uniform()) for f in M_NAMES}  # drawn every decision in every condition (paired streams)
        m = cond["meaning"]
        if m is None:
            meaning = None
        elif m == "rules":
            meaning = rules_m
        elif m == "truth":
            meaning = truth
        elif m == "flip":
            meaning = {f: apply_error(f, rules_m[f], "flicker", FLIP_P, u[f]) for f in M_NAMES}
        elif m == "late":
            past = [mm for t_, mm in late_buf if t_ <= env.t - LATE_S + 1e-9]
            meaning = past[-1] if past else (late_buf[0][1] if late_buf else rules_m)
        view = build_view(condition, state, preds, proprio, meaning)
        prompt = PV.build_prompt(view, last_actions)
        dec = reader.decide(prompt, view)
        chosen = dec["skill"]
        primitive = chosen
        tcp_now = np.asarray(proprio["tcp_position_m"])
        vkey = PV.state_block(view)
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
        if primitive == "test_lift":
            if lift_diverged_interp(track, t0, env.t):
                diverged.append({"t": round(env.t, 3), "event": "target_motion_diverged"})
        own = [e["event"] for e in events[n_ev:] if e.get("event") in OWN_EVENTS]
        last_actions.append({"skill": primitive, "events": own})
        after = completion_labels(verifier)
        rec = {"index": index, "t": round(t0, 3), "chosen": chosen, "executed": primitive, "valid": dec["valid"],
               "raw_skill": dec["raw_skill"], "error": dec["error"], "tokens_in": dec["tokens_in"],
               "tokens_out": dec["tokens_out"], "latency_s": dec["latency_s"], "cost_usd": dec["cost_usd"],
               "loop_exit": exit_here, "meaning_delivered": None if meaning is None else {f: _j(meaning[f]) for f in M_NAMES},
               "meaning_rules": {f: _j(rules_m[f]) for f in M_NAMES}, "truth": {f: _j(truth[f]) for f in M_NAMES},
               "own_events": own, "verifier_after": {"status": verifier.status, "conditions": after["conditions"]}}
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
    return {"seed": seed, "condition": condition, "reader": getattr(reader, "name", "?"), "variant": variant,
            "geometry": geometry, "runner_version": RUNNER_VERSION, "prompt_sha256": PV.prompt_hash(),
            "success": success, "verifier": report, "time_to_success_s": report["t_success"] if success else None,
            "loop_stopped": loop_stopped, "perturbation_fired": fired, "decision_steps": len(steps),
            "loop_exits": loop_exits, "invalid_answers": sum(1 for s in steps if not s["valid"]),
            "tokens_in": sum(s["tokens_in"] or 0 for s in steps), "tokens_out": sum(s["tokens_out"] or 0 for s in steps),
            "cost_usd": round(sum(s["cost_usd"] or 0 for s in steps), 6),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "wall_seconds": round(time.perf_counter() - started, 3), "steps": steps}
