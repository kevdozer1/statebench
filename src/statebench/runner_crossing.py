"""Turn 5 episode loop: the oracle/estimated crossing. ``runner_v2.py`` is unchanged.

Cells (first letter: the packet the consumer reads; second: the geometry the
executor uses): OO, EO, OE, EE. Both packets are built at every decision and
logged side by side; each reader only ever receives its own.

* Oracle packet: exactly ``runner_v2`` (COMPACT_TRUTH, R4 predicates, v0.2 executor).
* Estimated packet: E1 on the camera frame, written into the schema by
  ``observation_e1``; predicates by the unchanged predicate code from the
  estimated packet, PROPRIO and the estimated history.

Declared simplification: frames are fresh. The camera is rendered at every 10 Hz
recorder tick (E1 needs a track for moving-with-gripper) and at every decision,
and the decision's estimate is made from the frame at decision time. There is no
pipeline latency this turn.

``patch`` replaces one estimated field by its oracle value, in the text and in
the executor, in the current state and in the estimated history, with every other
field still estimated (Phase 5.3).
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from . import rules as prereg1
from .backends.base import NEXT_ACTION, OBJECT_HELD, STEP_COMPLETE, default_questions
from .env import StateBenchEnv
from .executor_v2 import ExecutorV2
from .frames_v2 import render_rgbd, visibility
from .observation_e1 import build_estimated_state, estimated_events, lift_diverged
from .observations import HistoryBuffer, build_compact_truth, build_proprio
from .perception_e1 import CONSTANTS, E1_VERSION, estimate, new_memory
from .render_frames import camera_model
from .representations_v2 import CONDITIONS_V2_BY_NAME, prepare_v2
from .runner_v2 import completion_labels
from .scene import RECEPTACLE_CENTER, perturbation_fires, perturbation_offset
from .verify import RUNNING, SUCCESS, Verifier, cube_moved_with_gripper

RUNNER_VERSION = "statebench-runner-crossing/v1"
RECORD_DT = 0.1
MAX_BACKEND_ERRORS = 3
CLAIM_THRESHOLD = 0.5
CELLS = ("OO", "EO", "OE", "EE")

#: Estimated fields that can be patched with their oracle value. The two
#: verifier-overlap relations are omitted: the R4 packet drops them.
PATCH_FIELDS = {
    "target_rel_gripper": ("geometry", "target_rel_gripper"),
    "target_rel_receptacle": ("geometry", "target_rel_receptacle"),
    "target_moving_with_gripper": ("relations", "target_moving_with_gripper"),
    "gripper_above_target": ("relations", "gripper_above_target"),
    "contact": ("contact", None),
    "uncertainty": ("uncertainty", None),
    "target_motion_diverged": ("events_recent", None),
}


def patch_state(est: dict, oracle: dict, field: str | None) -> dict:
    if field is None:
        return est
    out = copy.deepcopy(est)
    block, leaf = PATCH_FIELDS[field]
    path = f"{block}.{leaf}" if leaf else block
    if field == "target_motion_diverged":
        out["events_recent"] = copy.deepcopy(oracle.get("events_recent") or [])
        return out
    if leaf is None:
        out[block] = copy.deepcopy(oracle.get(block))
        prov_path = "contact.gripper_target" if block == "contact" else block
    else:
        if leaf in (oracle.get(block) or {}):
            out.setdefault(block, {})[leaf] = copy.deepcopy(oracle[block][leaf])
        else:
            (out.get(block) or {}).pop(leaf, None)
        prov_path = path
    for key in ("source", "t_obs"):
        value = (oracle.get("provenance") or {}).get(key, {}).get(prov_path)
        if value is not None:
            out["provenance"][key][prov_path] = value
    return out


class _Perception:
    """E1 at 10 Hz and at decisions, the estimated history, and truth-side logging."""

    def __init__(self, env, events: list, patch: str | None):
        self.env = env
        self.events = events
        self.patch = patch
        self.memory = new_memory()
        self.camera: dict[str, Any] | None = None
        self.proprio_by_t: dict[float, dict] = {}
        self.est_history = HistoryBuffer()
        self.diverged: list[dict] = []
        self.last_estimate: dict | None = None
        self.next_t = 0.0

    def est_events(self) -> list[dict]:
        return estimated_events(self.events, self.diverged)

    def observe(self, oracle_compact: dict) -> tuple[dict, dict]:
        env = self.env
        rgb, depth = render_rgbd(env)
        if self.camera is None:
            self.camera = camera_model(env)
        proprio = build_proprio(env)
        t = round(env.t, 3)
        self.proprio_by_t[t] = proprio
        est, self.memory = estimate(rgb, depth, self.camera, proprio, CONSTANTS, self.memory)
        self.last_estimate = est
        t_est = None if est.get("t_est") is None else round(float(est["t_est"]), 3)
        at_est = self.proprio_by_t.get(t_est, proprio)
        state = build_estimated_state(est, at_est, proprio, self.est_events(), self.memory["track"], CONSTANTS)
        state = patch_state(state, oracle_compact, self.patch)
        self.est_history.push(env.t, state)
        return state, est

    def tick(self, oracle_compact_fn):
        if self.env.t + 1e-9 < self.next_t:
            return
        self.next_t = self.env.t + RECORD_DT
        self.observe(oracle_compact_fn())


def _true_target_error(execution: dict, env_truth: dict) -> dict | None:
    """Commanded target against the true one for approach (cube) and transport (tray)."""
    target = execution.get("commanded_target")
    if not target:
        return None
    if execution["primitive"] == "approach":
        truth = np.array(env_truth["cube_m"][:2])
        err = {"xy_m": round(float(np.linalg.norm(np.asarray(target[:2]) - truth)), 4)}
        if execution.get("commanded_yaw") is not None:
            d = float(execution["commanded_yaw"]) - float(env_truth["cube_yaw_rad"])
            err["yaw_mod90_rad"] = round(abs((d + np.pi / 4) % (np.pi / 2) - np.pi / 4), 4)
        return err
    if execution["primitive"] == "transport":
        return {"xy_m": round(float(np.linalg.norm(np.asarray(target[:2]) - RECEPTACLE_CENTER)), 4)}
    return None


def run_crossing_episode(seed: int, cell: str, consumer: str, backend, *, patch: str | None = None,
                         max_steps: int = prereg1.MAX_DECISION_STEPS) -> dict[str, Any]:
    from .backends.rules_v2 import choose as rules_v2_choose

    assert cell in CELLS
    started_wall = time.perf_counter()
    env = StateBenchEnv(seed, render=True)
    events: list[dict] = []
    history = HistoryBuffer()
    verifier = Verifier(env)
    perception = _Perception(env, events, patch)
    ticks: list[dict] = []
    oracle_fn = lambda: build_compact_truth(env, events)  # noqa: E731

    def recorder(env_):
        if env_.t + 1e-9 >= perception.next_t:
            compact = oracle_fn()
            history.push(env_.t, compact)
            labels = completion_labels(verifier)
            perception.tick(lambda: compact)
            ticks.append({"t": round(env_.t, 3), "complete_now": labels["complete_now"],
                          "placed": labels["placed"]})

    env.step_hooks.append(verifier.update)
    env.step_hooks.append(recorder)
    fired = perturbation_fires(seed, prereg1.PERTURBATION_PROBABILITY)
    offset = perturbation_offset(seed, prereg1.PERTURBATION_OFFSET_M) if fired else None
    executor = ExecutorV2(env, events, offset)
    rng = np.random.default_rng(seed + 40_000)
    questions = default_questions()
    condition = CONDITIONS_V2_BY_NAME["R4c" if consumer == "jev_r4c" else "R4j"]
    override = None
    if condition.prompt_variant == "c":
        from .rules_turn3b import R4C_NEXT_ACTION_INSTRUCTION as override

    steps: list[dict] = []
    backend_errors = 0
    calibration = []
    recorder(env)
    stop_reason = "verifier"
    for index in range(max_steps):
        if verifier.status != RUNNING:
            break
        compact = oracle_fn()
        history.push(env.t, compact)
        est_compact, est = perception.observe(compact)
        proprio = build_proprio(env)
        before = completion_labels(verifier)
        o_state, o_hist, o_text = prepare_v2(condition, compact, history, env.t, rng, proprio=proprio)
        e_state, e_hist, e_text = prepare_v2(condition, est_compact, perception.est_history, env.t, rng,
                                             proprio=proprio)
        text = o_text if cell[0] == "O" else e_text
        exec_state, exec_hist = (o_state, o_hist) if cell[1] == "O" else (e_state, e_hist)
        decision = backend.decide(text, questions, instruction_override=override)
        record = decision.to_record()
        judge_p = held_p = None
        if decision.error or NEXT_ACTION not in decision.answers:
            backend_errors += 1
            primitive = "inspect"
            record["status"] = "BACKEND_ERROR"
        else:
            primitive = decision.answers[NEXT_ACTION].choice
            if STEP_COMPLETE in decision.answers:
                judge_p = decision.answers[STEP_COMPLETE].probability_of("yes")
            if OBJECT_HELD in decision.answers:
                held_p = decision.answers[OBJECT_HELD].probability_of("yes")
        truth = {"cube_m": [float(v) for v in env.truth_cube_position()],
                 "cube_yaw_rad": float(env.truth_cube_yaw()),
                 # Truth-only, for attribution; no reader sees it.
                 "visible_fraction": visibility(env, perception.camera)["visible_fraction"]}
        rv2_oracle = rules_v2_choose(o_state["predicates"]["values"])[0]
        rv2_est = rules_v2_choose(e_state["predicates"]["values"])[0]
        t_start = env.t
        execution = executor.execute(primitive, exec_state, exec_hist)
        if primitive == "test_lift":
            div = lift_diverged(perception.memory["track"], t_start, env.t)
            if div:
                perception.diverged.append({"t": round(env.t, 3), "event": "target_motion_diverged"})
        after = completion_labels(verifier)
        if held_p is not None:
            calibration.append({"t": round(t_start, 3), "predicted_yes": round(float(held_p), 4)})
        steps.append({
            "index": index, "t": round(t_start, 3),
            "oracle_text": o_text, "estimated_text": e_text, "consumer_read": cell[0],
            "executor_used": cell[1],
            "oracle_predicates": o_state["predicates"]["values"],
            "estimated_predicates": e_state["predicates"]["values"],
            "e1": {k: est.get(k) for k in ("fresh", "cube_m", "yaw_mod90_rad", "visible_fraction",
                                            "red_px", "age_s", "yaw_source")},
            "truth": truth,
            "decision": record,
            "rules_v2_on_oracle": rv2_oracle, "rules_v2_on_estimated": rv2_est,
            "judge_p_complete": None if judge_p is None else round(float(judge_p), 6),
            "labels_before": {"complete_now": before["complete_now"], "placed": before["placed"]},
            "execution": execution,
            "target_error": _true_target_error(execution, truth),
            "verifier_after": {"status": verifier.status, "conditions": after["conditions"],
                               "complete_now": after["complete_now"], "placed": after["placed"]},
        })
        if verifier.status != RUNNING:
            break
    else:
        stop_reason = "step budget"
    if verifier.status == RUNNING:
        verifier.finish(f"decision loop ended ({stop_reason}) without a verified placement")
    for row in calibration:
        row["label"] = cube_moved_with_gripper(env, row["t"], 0.5)
    report = verifier.report()
    success = report["status"] == SUCCESS
    for s in steps:
        s["labels_before"]["eventual_success"] = success
    for tk in ticks:
        tk["eventual_success"] = success
    first_claim = next((s for s in steps if (s["judge_p_complete"] or 0) > CLAIM_THRESHOLD), None)
    result = {
        "seed": seed, "cell": cell, "consumer": consumer, "patch": patch,
        "backend": backend.name, "model": backend.model, "runner_version": RUNNER_VERSION,
        "e1_version": E1_VERSION, "success": success, "verifier": report,
        "time_to_success_s": report["t_success"] if success else None,
        "sim_seconds": round(env.t, 3), "wall_seconds": round(time.perf_counter() - started_wall, 3),
        "perturbation_fired": fired, "decision_steps": len(steps), "backend_errors": backend_errors,
        "excluded": backend_errors > MAX_BACKEND_ERRORS,
        "latencies": [float(s["decision"].get("latency_seconds") or 0.0) for s in steps],
        "cost_usd": 0.0,
        "first_claim": None if first_claim is None else {
            "index": first_claim["index"], "p": first_claim["judge_p_complete"],
            "complete_now": first_claim["labels_before"]["complete_now"],
            "placed": first_claim["labels_before"]["placed"], "eventual_success": success},
        "claim_then_eventual_failure": first_claim is not None and not success,
        "calibration": calibration, "ticks": ticks,
        "primitive_counts": _counts(s["execution"]["primitive"] for s in steps),
        "steps": steps,
    }
    env.close()
    return result


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))

