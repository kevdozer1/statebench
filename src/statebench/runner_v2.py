"""Episode loop v0.2. ``runner.py`` is unchanged and still reproduces v0.1 runs.

Differences from v0.1, each declared in ``notes/turn4.md``:

1. ``executor_v2.ExecutorV2`` registers each displacement at its own ``t_obs``.
2. A completion claim is p(step_complete = yes) **above** 0.5, the same threshold
   ``step_complete_bins`` always used (v0.1's runner counted p >= 0.5).
3. Three completion targets from simulator truth, never shown to any reader:

   ``complete_now``      all 5 verifier conditions hold at this instant;
   ``placed``            cube centre in the tray volume, cube touching the tray,
                         cube touching neither finger pad, gripper anywhere;
   ``eventual_success``  the verifier's verdict, filled in at episode end.

   They are logged at every decision *before* the primitive runs and at every
   10 Hz recorder tick. The verifier's condition dict is logged both before and
   after the primitive (v0.1 mixed a before value, ``says_complete``, with an
   after value, ``conditions``).
4. The old claim statistic is kept and renamed "claim then eventual failure".
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import rules as prereg
from .backends.base import NEXT_ACTION, OBJECT_HELD, STEP_COMPLETE, DecisionBackend, default_questions
from .env import StateBenchEnv
from .executor_v2 import EXECUTOR_VERSION, ExecutorV2
from .observations import HistoryBuffer, build_compact_truth, build_proprio
from .representations_v2 import ConditionV2, prepare_v2
from .scene import perturbation_fires, perturbation_offset
from .schema import validate_state
from .verify import RUNNING, SUCCESS, Verifier, cube_moved_with_gripper

RUNNER_VERSION = "statebench-runner/v0.2"
STATE_RECORD_DT = 0.1
CALIBRATION_WINDOW_S = 0.5
MAX_BACKEND_ERRORS = 3
CLAIM_THRESHOLD = 0.5  # a claim is p > CLAIM_THRESHOLD


def completion_labels(verifier: Verifier) -> dict[str, Any]:
    """complete_now and placed at this instant, from simulator truth."""
    conditions = verifier._conditions()  # read-only use of the frozen verifier
    placed = bool(
        conditions["cube_center_in_volume"]
        and conditions["cube_touches_receptacle"]
        and conditions["cube_free_of_fingers"]
    )
    return {
        "complete_now": bool(all(conditions.values())),
        "placed": placed,
        "conditions": {k: bool(v) for k, v in conditions.items()},
    }


class _Recorder:
    """10 Hz: pushes COMPACT_TRUTH into the history buffer and logs the labels."""

    def __init__(self, history: HistoryBuffer, events: list, verifier: Verifier):
        self.history = history
        self.events = events
        self.verifier = verifier
        self.next_t = 0.0
        self.ticks: list[dict[str, Any]] = []

    def __call__(self, env) -> None:
        if env.t + 1e-9 >= self.next_t:
            self.next_t = env.t + STATE_RECORD_DT
            self.history.push(env.t, build_compact_truth(env, self.events))
            labels = completion_labels(self.verifier)
            self.ticks.append({"t": round(env.t, 3), "complete_now": labels["complete_now"],
                               "placed": labels["placed"]})


def run_episode_v2(
    seed: int,
    condition: ConditionV2,
    backend: DecisionBackend,
    *,
    max_steps: int = prereg.MAX_DECISION_STEPS,
    audit_reads: bool = False,
    questions: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    started_wall = time.perf_counter()
    env = StateBenchEnv(seed)
    events: list[dict[str, Any]] = []
    history = HistoryBuffer()
    verifier = Verifier(env)
    recorder = _Recorder(history, events, verifier)
    env.step_hooks.append(verifier.update)
    env.step_hooks.append(recorder)

    fired = perturbation_fires(seed, prereg.PERTURBATION_PROBABILITY)
    offset = perturbation_offset(seed, prereg.PERTURBATION_OFFSET_M) if fired else None
    executor = ExecutorV2(env, events, offset)
    rng = np.random.default_rng(seed + 40_000)
    questions = questions or default_questions()
    override = None
    if condition.prompt_variant == "c":
        from .rules_turn3b import R4C_NEXT_ACTION_INSTRUCTION as override
    elif condition.prompt_variant is not None:
        raise ValueError(f"unknown prompt variant {condition.prompt_variant!r}")

    steps: list[dict[str, Any]] = []
    schema_problems: list[str] = []
    read_audit: list | None = [] if audit_reads else None
    backend_errors = 0
    calibration: list[dict[str, Any]] = []
    stop_reason = "verifier"

    recorder(env)
    for index in range(max_steps):
        if verifier.status != RUNNING:
            break
        compact = build_compact_truth(env, events)
        history.push(env.t, compact)
        schema_problems.extend(validate_state(compact))
        t_decision = env.t
        before = completion_labels(verifier)
        proprio = build_proprio(env)
        state, hist, text = prepare_v2(condition, compact, history, env.t, rng,
                                       proprio=proprio, read_audit=read_audit)
        decision = backend.decide(text, questions, instruction_override=override)
        record = decision.to_record()
        judge_p = held_p = None
        if decision.error or NEXT_ACTION not in decision.answers:
            backend_errors += 1
            primitive = "inspect"
            record["fallback_primitive"] = primitive
            record["status"] = "BACKEND_ERROR"
        else:
            primitive = decision.answers[NEXT_ACTION].choice
            if STEP_COMPLETE in decision.answers:
                judge_p = decision.answers[STEP_COMPLETE].probability_of("yes")
            if OBJECT_HELD in decision.answers:
                held_p = decision.answers[OBJECT_HELD].probability_of("yes")

        execution = executor.execute(primitive, state, hist)
        after = completion_labels(verifier)
        if held_p is not None:
            calibration.append({"t": round(t_decision, 3), "predicted_yes": round(float(held_p), 4)})
        steps.append({
            "index": index,
            "t": round(t_decision, 3),
            "state_text": text,
            "state_text_chars": len(text),
            "estimate_age_s": (state.get("uncertainty") or {}).get("estimate_age_s"),
            "proprio": {k: proprio[k] for k in ("gripper_opening_m", "gripper_command",
                                                "tcp_position_m", "tcp_yaw_rad")},
            "decision": record,
            "judge_p_complete": None if judge_p is None else round(float(judge_p), 6),
            "judge_claims_complete": judge_p is not None and judge_p > CLAIM_THRESHOLD,
            "labels_before": {"complete_now": before["complete_now"], "placed": before["placed"]},
            "verifier_before": before["conditions"],
            "execution": execution,
            "verifier_after": {
                "status": verifier.status,
                "stable_seconds": round(verifier.state.stable_seconds, 3),
                "conditions": after["conditions"],
                "complete_now": after["complete_now"],
                "placed": after["placed"],
            },
        })
        if verifier.status != RUNNING:
            break
    else:
        stop_reason = "step budget"

    if verifier.status == RUNNING:
        verifier.finish(f"decision loop ended ({stop_reason}) without a verified placement")
    for row in calibration:
        row["label"] = cube_moved_with_gripper(env, row["t"], CALIBRATION_WINDOW_S)

    report = verifier.report()
    success = report["status"] == SUCCESS
    for step in steps:
        step["labels_before"]["eventual_success"] = success
    for tick in recorder.ticks:
        tick["eventual_success"] = success

    first_claim = next((s for s in steps if s["judge_claims_complete"]), None)
    result = {
        "seed": seed,
        "condition": condition.name,
        "representation": condition.representation,
        "packet": condition.packet,
        "rendering": condition.rendering,
        "prompt_variant": condition.prompt_variant,
        "backend": backend.name,
        "model": backend.model,
        "runner_version": RUNNER_VERSION,
        "executor_version": EXECUTOR_VERSION,
        "success": success,
        "verifier": report,
        "time_to_success_s": report["t_success"] if success else None,
        "sim_seconds": round(env.t, 3),
        "wall_seconds": round(time.perf_counter() - started_wall, 3),
        "perturbation_fired": fired,
        "perturbation_applied": executor.perturbation_fired,
        "decision_steps": len(steps),
        "backend_errors": backend_errors,
        "excluded": backend_errors > MAX_BACKEND_ERRORS,
        "retries": sum(int(s["decision"].get("retries") or 0) for s in steps),
        "latencies": [float(s["decision"].get("latency_seconds") or 0.0) for s in steps],
        "cost_usd": round(sum(float(s["decision"].get("cost_usd") or 0.0) for s in steps), 8),
        "tokens_prompt": sum(int(s["decision"].get("tokens_prompt") or 0) for s in steps),
        "tokens_completion": sum(int(s["decision"].get("tokens_completion") or 0) for s in steps),
        "inspect_calls": sum(1 for s in steps if s["execution"]["primitive"] == "inspect"),
        "schema_problems": sorted(set(schema_problems)),
        "judge_scored_steps": sum(1 for s in steps if s["judge_p_complete"] is not None),
        "first_claim": None if first_claim is None else {
            "index": first_claim["index"], "t": first_claim["t"],
            "p": first_claim["judge_p_complete"],
            "complete_now": first_claim["labels_before"]["complete_now"],
            "placed": first_claim["labels_before"]["placed"],
            "eventual_success": success,
        },
        # Legacy statistic, renamed: first claim at p > 0.5, then the episode failed.
        "claim_then_eventual_failure": first_claim is not None and not success,
        "judge_complete_verifier_not": sum(
            1 for s in steps if s["judge_claims_complete"] and not s["labels_before"]["complete_now"]),
        "verifier_complete_judge_not": sum(
            1 for s in steps if s["judge_p_complete"] is not None
            and s["labels_before"]["complete_now"] and not s["judge_claims_complete"]),
        "calibration": calibration,
        "ticks": recorder.ticks,
        "read_audit": read_audit,
        "registrations": executor.registrations,
        "primitive_counts": _counts(s["execution"]["primitive"] for s in steps),
        "steps": steps,
    }
    env.close()
    return result


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))
