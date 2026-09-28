"""Turn 6 episode loop. ``runner_crossing.py`` and ``runner_v2.py`` are unchanged.

``runner_crossing`` plus: scene variants (V0, V1, V2), sensor-informed synthetic
perturbations and coherent corrections (``pipeline_v6``), the E1 or E1.1 motion
relation, the R4d consumer, the OO-semantic oracle cell, and the loop-exit rule.

**Loop-exit rule (declared executor rule).** When the consumer chooses the same
primitive for the third decision in a row, the predicate block it read is
identical across those three decisions, and the TCP moved less than 5 mm over
them, the executor runs ``approach`` instead and logs a loop exit. It never
triggers on ``approach`` itself.

With the variant V0, the ``clean`` condition, no corrections, E1, the Jev R4c
consumer and no loop exit, the loop reproduces ``runner_crossing`` (checked
against Turn 5's EE packs).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import rules as prereg1
from .backends.base import NEXT_ACTION, OBJECT_HELD, STEP_COMPLETE, default_questions
from .env_v6 import EnvV6
from .executor_v6 import ExecutorV6
from .frames_v2 import visibility
from .observations import HistoryBuffer, build_compact_truth, build_proprio
from .perception_e1 import CONSTANTS
from .pipeline_v6 import Pipeline, semantic_predicates
from .representations_v2 import CONDITIONS_V2_BY_NAME, prepare_v2, render
from .runner_crossing import _true_target_error, patch_state
from .runner_v2 import completion_labels
from .scene import perturbation_fires
from .scene_v6 import VARIANTS, draw_scene_params
from .verify import RUNNING, SUCCESS, Verifier, cube_moved_with_gripper

RUNNER_VERSION = "statebench-runner-v6/v1"
RECORD_DT = 0.1
MAX_BACKEND_ERRORS = 3
CLAIM_THRESHOLD = 0.5
LOOP_REPEATS = 3
LOOP_TCP_M = 0.005
#: Measured this turn (notes/turn6.md): V1's grasp height and forced-miss offset.
V1_GRASP_Z = 0.024
MISS_OFFSET = {"V0": 0.035, "V1": 0.015, "V2": 0.035}


def variant_heights(variant: str) -> tuple[float, float]:
    cfg = VARIANTS[variant]
    return (V1_GRASP_Z if variant == "V1" else cfg.grasp_z), cfg.place_z


def run_episode_v6(seed: int, cell: str, consumer: str, backend, *, variant: str = "V0",
                   condition: str = "clean", corrections: tuple[str, ...] = (), e1_version: str = "E1",
                   loop_exit: bool = False, oracle_semantic: bool = False, patch: str | None = None,
                   max_steps: int = prereg1.MAX_DECISION_STEPS) -> dict[str, Any]:
    from .backends.rules_v2 import choose as rules_v2_choose

    started_wall = time.perf_counter()
    cfg = VARIANTS[variant]
    scene = None
    if condition in ("rgb", "combined"):
        scene = draw_scene_params(np.random.default_rng(3_000_017 + seed))
    env = EnvV6(seed, render=True, variant=variant, scene=scene)
    events: list[dict] = []
    history = HistoryBuffer()
    verifier = Verifier(env)
    constants = dict(CONSTANTS, cube_edge_m=2 * cfg.cube_half)
    pipe = Pipeline(env, condition, corrections, seed, e1_version=e1_version, constants=constants)
    est_history = HistoryBuffer()
    diverged: list[dict] = []
    ticks: list[dict] = []
    state = {"next_t": 0.0}

    def est_events():
        from .observation_e1 import estimated_events

        return estimated_events(events, diverged)

    def observe(oracle_compact):
        proprio = build_proprio(env)
        est_state, est, frame = pipe.observe(proprio, est_events())
        est_state = patch_state(est_state, oracle_compact, patch)
        est_history.push(env.t, est_state)
        return est_state, est, frame

    def recorder(env_):
        pipe.frame_hook(env_, lambda: build_proprio(env_))
        if env_.t + 1e-9 >= state["next_t"]:
            state["next_t"] = env_.t + RECORD_DT
            compact = build_compact_truth(env_, events)
            history.push(env_.t, compact)
            labels = completion_labels(verifier)
            observe(compact)
            ticks.append({"t": round(env_.t, 3), "complete_now": labels["complete_now"],
                          "placed": labels["placed"]})

    env.step_hooks.append(verifier.update)
    env.step_hooks.append(recorder)
    fired = perturbation_fires(seed, prereg1.PERTURBATION_PROBABILITY)
    offset = np.array([0.0, 0.0, MISS_OFFSET[variant]]) if fired else None
    grasp_z, place_z = variant_heights(variant)
    executor = ExecutorV6(env, events, offset, grasp_z, place_z)
    rng = np.random.default_rng(seed + 40_000)
    questions = default_questions()
    condition_name = "R4c" if consumer == "jev_r4c" else "R4j"
    rcond = CONDITIONS_V2_BY_NAME[condition_name]
    #: The optional paid consumer reads the R3nv packet as prose: no predicates, no table.
    read_cond = CONDITIONS_V2_BY_NAME["R3nvp"] if consumer == "llm_strong" else None
    override = None
    if consumer == "jev_r4c":
        from .rules_turn3b import R4C_NEXT_ACTION_INSTRUCTION as override
    elif consumer == "jev_r4d":
        from .instructions_v6 import R4D_NEXT_ACTION_INSTRUCTION as override

    steps: list[dict] = []
    backend_errors = 0
    calibration = []
    loop_exits = []
    recent: list[tuple] = []
    recorder(env)
    stop_reason = "verifier"
    for index in range(max_steps):
        if verifier.status != RUNNING:
            break
        compact = build_compact_truth(env, events)
        history.push(env.t, compact)
        est_compact, est, frame = observe(compact)
        proprio = build_proprio(env)
        before = completion_labels(verifier)
        o_state, o_hist, o_text = prepare_v2(rcond, compact, history, env.t, rng, proprio=proprio)
        e_state, e_hist, e_text = prepare_v2(rcond, est_compact, est_history, env.t, rng, proprio=proprio)
        if "semantic" in corrections:
            e_state["predicates"]["values"] = semantic_predicates(e_state["predicates"]["values"], env)
            e_text = render(rcond, e_state)
        if oracle_semantic:
            o_state["predicates"]["values"] = semantic_predicates(o_state["predicates"]["values"], env)
            o_text = render(rcond, o_state)
        read_state = o_state if cell[0] == "O" else e_state
        text = o_text if cell[0] == "O" else e_text
        if read_cond is not None:
            bare = {k: v for k, v in read_state.items() if k != "predicates"}
            bare["provenance"] = {**bare["provenance"],
                                  "source": {k: v for k, v in bare["provenance"]["source"].items() if k != "predicates"},
                                  "t_obs": {k: v for k, v in bare["provenance"]["t_obs"].items() if k != "predicates"}}
            text = render(read_cond, bare)
        exec_state, exec_hist = (o_state, o_hist) if cell[1] == "O" else (e_state, e_hist)
        decision = backend.decide(text, questions, instruction_override=override)
        record = decision.to_record()
        judge_p = held_p = None
        if decision.error or NEXT_ACTION not in decision.answers:
            backend_errors += 1
            chosen = "inspect"
            record["status"] = "BACKEND_ERROR"
        else:
            chosen = decision.answers[NEXT_ACTION].choice
            if STEP_COMPLETE in decision.answers:
                judge_p = decision.answers[STEP_COMPLETE].probability_of("yes")
            if OBJECT_HELD in decision.answers:
                held_p = decision.answers[OBJECT_HELD].probability_of("yes")
        primitive = chosen
        tcp_now = np.asarray(proprio["tcp_position_m"])
        recent.append((chosen, tuple(sorted((k, str(v)) for k, v in read_state["predicates"]["values"].items())),
                       tcp_now))
        recent = recent[-LOOP_REPEATS:]
        loop_exit_here = None
        if (loop_exit and len(recent) == LOOP_REPEATS and chosen != "approach"
                and len({r[0] for r in recent}) == 1 and len({r[1] for r in recent}) == 1
                and max(float(np.linalg.norm(r[2] - recent[0][2])) for r in recent) < LOOP_TCP_M):
            primitive = "approach"
            loop_exit_here = {"index": index, "repeated": chosen, "executed": "approach"}
            loop_exits.append(loop_exit_here)
            recent = []
        truth = {"cube_m": [float(v) for v in env.truth_cube_position()],
                 "cube_yaw_rad": float(env.truth_cube_yaw()),
                 "visible_fraction": visibility(env, pipe.true_camera)["visible_fraction"]}
        t_start = env.t
        execution = executor.execute(primitive, exec_state, exec_hist)
        if primitive == "test_lift":
            if pipe.lift_diverged(t_start, env.t):
                diverged.append({"t": round(env.t, 3), "event": "target_motion_diverged"})
        after = completion_labels(verifier)
        if held_p is not None:
            calibration.append({"t": round(t_start, 3), "predicted_yes": round(float(held_p), 4)})
        steps.append({
            "index": index, "t": round(t_start, 3),
            "oracle_text": o_text, "estimated_text": e_text, "consumer_text": text if read_cond is not None else None, "consumer_read": cell[0], "executor_used": cell[1],
            "oracle_predicates": o_state["predicates"]["values"],
            "estimated_predicates": e_state["predicates"]["values"],
            "e1": {k: est.get(k) for k in ("fresh", "cube_m", "yaw_mod90_rad", "visible_fraction", "red_px",
                                            "age_s", "yaw_source")},
            "frame_age_s": round(t_start - frame.t, 3),
            "truth": truth, "decision": record, "chosen": chosen, "loop_exit": loop_exit_here,
            "rules_v2_on_oracle": rules_v2_choose(o_state["predicates"]["values"])[0],
            "rules_v2_on_estimated": rules_v2_choose(e_state["predicates"]["values"])[0],
            "judge_p_complete": None if judge_p is None else round(float(judge_p), 6),
            "labels_before": {"complete_now": before["complete_now"], "placed": before["placed"]},
            "execution": execution, "target_error": _true_target_error(execution, truth),
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
    first_claim = next((s for s in steps if (s["judge_p_complete"] or 0) > CLAIM_THRESHOLD), None)
    result = {
        "seed": seed, "cell": cell, "consumer": consumer, "variant": variant, "condition": condition,
        "corrections": list(corrections), "e1_version": e1_version, "loop_exit_rule": loop_exit,
        "oracle_semantic": oracle_semantic, "patch": patch,
        "scene_params": None if scene is None else scene.__dict__,
        "extrinsic_perturbation": pipe.extrinsic_info, "exposure": pipe.exposure,
        "white_balance": [round(float(v), 4) for v in pipe.wb],
        "backend": backend.name, "model": backend.model, "runner_version": RUNNER_VERSION,
        "success": success, "verifier": report,
        "time_to_success_s": report["t_success"] if success else None,
        "sim_seconds": round(env.t, 3), "wall_seconds": round(time.perf_counter() - started_wall, 3),
        "perturbation_fired": fired, "decision_steps": len(steps), "backend_errors": backend_errors,
        "excluded": backend_errors > MAX_BACKEND_ERRORS, "loop_exits": loop_exits,
        "latencies": [float(s["decision"].get("latency_seconds") or 0.0) for s in steps],
        "cost_usd": round(sum(float(s["decision"].get("cost_usd") or 0.0) for s in steps), 8),
        "first_claim": None if first_claim is None else {
            "index": first_claim["index"], "p": first_claim["judge_p_complete"],
            "complete_now": first_claim["labels_before"]["complete_now"],
            "placed": first_claim["labels_before"]["placed"], "eventual_success": success},
        "claim_then_eventual_failure": first_claim is not None and not success,
        "calibration": calibration, "ticks": ticks,
        "steps": steps,
    }
    env.close()
    return result
