"""TURN 3b PRE-REGISTRATION. Jev-only follow-ups. Hashed before any 3b run.

Three questions left open by Turn 3:

1. **R4c** - Turn 3's R4b named the wrong recovery rule ("If gripper_closed_empty,
   choose approach"), and Jev complied with it at p=1.00, which is precisely why
   recovery stayed at 0/12. R4c changes that one sentence to "Choose release if
   gripper_closed_empty." and nothing else.
2. **The delay condition measures the executor, not the policy.** Established
   from code and logs before this file was hashed (see INSPECT_FINDING). The
   clarification below changes ``inspect`` so that a decision policy can in
   principle succeed, and is applied to every condition.
3. **R5** - is Jev a pure boolean reader? R5 shows it the predicates and
   relations without the numeric geometry.

No OpenRouter backend runs in Turn 3b. Expected spend: 0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .rules_turn3 import R4B_NEXT_ACTION_INSTRUCTION

SCHEMA_VERSION = "statebench/state/v0.1"
TURN3_PREREG_SHA256 = "c578fe153855999a72f1a16d07e02d4331caaa6ec6cf70942b76267961ccbb68"

# ------------------------------------------------------------ the R4c change
#: R4b's final sentence, and its replacement. Everything before it is verbatim.
R4B_FINAL_SENTENCE = "If gripper_closed_empty, choose approach."
R4C_FINAL_SENTENCE = "Choose release if gripper_closed_empty."

assert R4B_NEXT_ACTION_INSTRUCTION.endswith(R4B_FINAL_SENTENCE)
R4C_NEXT_ACTION_INSTRUCTION = (
    R4B_NEXT_ACTION_INSTRUCTION[: -len(R4B_FINAL_SENTENCE)] + R4C_FINAL_SENTENCE
)

# ------------------------------------------------- the inspect finding + fix
INSPECT_FINDING = (
    "Established from executor.py and the Turn 2/3 logs BEFORE this file was "
    "hashed. `_do_inspect` holds for DURATIONS['inspect'] = 0.30 s of sim time, "
    "emits inspect_done, and returns. It does not request or await a new "
    "observation. Independently, the delay ablation recomputes "
    "history.at_or_before(t - 0.5) at every decision, so the observation age is a "
    "constant 0.5 s pipeline latency no matter what the policy does. Only two "
    "inspect calls exist in all delay logs: turn2/R3_delay seed 152 step 10 "
    "(age 0.5 s at the decision, 0.5 s at the following step) and seed 176 step 0 "
    "(0.0 s, then 0.3 s, rising toward the 0.5 s steady state). inspect therefore "
    "does not return a fresh observation, and under the delay condition no "
    "decision policy can reduce the age of what it sees. The condition as run in "
    "Turns 1-3 measured the executor, not the policy."
)

INSPECT_CLARIFICATION = (
    "inspect waits until a fresh observation is available (estimate_age_s below "
    "0.1 s of sim time) and returns it, with the wait counted against the 30 s "
    "timeout. Implemented as: (a) DURATIONS['inspect'] rises from 0.30 s to 0.50 s, "
    "equal to the pipeline latency, so the wait is real sim time charged to the "
    "episode budget; (b) the delay ablation returns the current, undelayed state "
    "with estimate_age_s = 0.0 when the most recent entry of events_recent is an "
    "inspect_done at or after t - 0.05 s. The trigger uses only events_recent, a "
    "field already in the declared observation set. Applies to every condition."
)
FRESH_AGE_THRESHOLD_S = 0.1
INSPECT_DURATION_BEFORE_S = 0.30
INSPECT_DURATION_AFTER_S = 0.50

#: executor.py before the clarification (frozen across Turns 1-3).
EXECUTOR_SHA256_BEFORE = "a4411d74bb66af7dc2e80bc1285b971fcbb8b6216d657b02f55e42b73c4abcb3"
#: Filled in by tools after the edit; every Turn 3b table is labelled with it.
EXECUTOR_SHA256_AFTER = "a814ec3ab3f30b457816f2a63c1bea4d8b3a6617754a86e7211f4e014bb10d4f"

EXECUTOR_CHANGE_SCOPE = (
    "This is a change to a file frozen since Turn 1. Every result in Turns 1, 2 "
    "and 3 was produced under EXECUTOR_SHA256_BEFORE and still stands under it. "
    "Every Turn 3b table is labelled with EXECUTOR_SHA256_AFTER. The spot check "
    "below exists to show the change is inert outside the delay conditions."
)

SPOT_CHECK = (
    "rerun rules on R3_typed_history for 20 episodes, seeds 0-19 (the contiguous "
    "band, not the stride-4 grid), and compare success and the full primitive "
    "sequence against the Turn 1 logs. Expect 20/20 and identical sequences."
)
SPOT_CHECK_SEEDS = tuple(range(0, 20))

# ---------------------------------------------------------------------- R5
R5_SPEC = (
    "R5 is R4 with the numeric geometry block removed from the TEXT SENT TO THE "
    "BACKEND only. The executor continues to receive the full R4 state. This is a "
    "rendering-only difference, in the same sense that R1/R2/R3 differ only in "
    "rendering, because the question R5 asks is what Jev reads, not what the arm "
    "can physically do. Removing geometry from the executor's state as well would "
    "leave every primitive falling back to the task prior and would measure the "
    "executor's fallback policy instead of Jev's dependence on numbers."
)

# ------------------------------------------------------------------- the runs
JEV = "jev"
RULES = "rules"

#: (backend, condition, n, seed_set)
GRID: tuple[tuple[str, str, int, str], ...] = (
    (JEV, "R4c_conditions_in_prompt_fixed", 50, "dev"),
    (JEV, "R5_predicates_only", 50, "dev"),
    (RULES, "R3_delay_0p5s", 50, "dev"),
    (JEV, "R4_delay_0p5s", 50, "dev"),
    (JEV, "R4c_delay_0p5s", 50, "dev"),
)
#: Reruns that must overwrite their Turn 3 packs because the executor changed.
RERUN_UNDER_NEW_EXECUTOR = ("R3_delay_0p5s", "R4_delay_0p5s", "R4c_delay_0p5s")

DEV_SEED_POOL = tuple(range(0, 200))
SEED_STRIDE = 4
EPISODE_SEEDS = tuple(DEV_SEED_POOL[::SEED_STRIDE])[:50]
PERTURBED_SEEDS = (12, 24, 28, 88, 104, 120, 148, 160, 164, 180, 184, 192)

MAX_DECISION_STEPS = 24
TIMEOUT_SECONDS = 30.0
DELAY_AGE_SPLIT_PREDICATE = "estimate_stale"

# -------------------------------------------------------------------- budget
LLM_BACKENDS_RUN = ()
EXPECTED_SPEND_USD = 0.0
DO_NOT_TOUCH = "the llm_cheap and llm_strong queues started in Turn 3 run undisturbed"

# ------------------------------------------------------------------ readouts
READOUT_R4C = (
    "recovery on the 12 perturbed seeds and overall success, beside R4 and R4b, "
    "with seed-paired discordant counts. If recovery moves off 0/12, name the "
    "seeds. If it stays at 0/12, print the next_action distribution at the first "
    "gripper_closed_empty step for 5 perturbed episodes."
)
READOUT_DELAY = (
    "success, inspect count per episode split by the estimate_stale predicate, "
    "and whether inspect is followed by progress (a non-inspect primitive that "
    "changes the predicate vector within two steps)."
)
READOUT_R5 = (
    "success, recovery and seed-paired discordance against R4. If R5 equals R4 at "
    "38/50 with the same 12 failures, the numbers contribute nothing to Jev's "
    "decisions on this task; if worse, name the steps that changed."
)

OUTCOME_CLEAR = "CLEAR DIFFERENCE"
OUTCOME_NO_CLEAR = "NO CLEAR DIFFERENCE"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
MIN_COMPLETED_EPISODES = 30


def preregistration_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def executor_hash_now() -> str:
    return hashlib.sha256(
        (Path(__file__).resolve().parent / "executor.py").read_bytes()
    ).hexdigest()


def summary() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "turn3_prereg_sha256": TURN3_PREREG_SHA256,
        "r4b_instruction": R4B_NEXT_ACTION_INSTRUCTION,
        "r4c_instruction": R4C_NEXT_ACTION_INSTRUCTION,
        "r4b_final_sentence": R4B_FINAL_SENTENCE,
        "r4c_final_sentence": R4C_FINAL_SENTENCE,
        "inspect_finding": INSPECT_FINDING,
        "inspect_clarification": INSPECT_CLARIFICATION,
        "fresh_age_threshold_s": FRESH_AGE_THRESHOLD_S,
        "inspect_duration_before_s": INSPECT_DURATION_BEFORE_S,
        "inspect_duration_after_s": INSPECT_DURATION_AFTER_S,
        "executor_sha256_before": EXECUTOR_SHA256_BEFORE,
        "executor_sha256_after_declared": EXECUTOR_SHA256_AFTER,
        "executor_sha256_now": executor_hash_now(),
        "executor_change_scope": EXECUTOR_CHANGE_SCOPE,
        "spot_check": SPOT_CHECK,
        "spot_check_seeds": list(SPOT_CHECK_SEEDS),
        "r5_spec": R5_SPEC,
        "grid": [list(row) for row in GRID],
        "rerun_under_new_executor": list(RERUN_UNDER_NEW_EXECUTOR),
        "episode_seeds": list(EPISODE_SEEDS),
        "perturbed_seeds": list(PERTURBED_SEEDS),
        "expected_spend_usd": EXPECTED_SPEND_USD,
        "llm_backends_run": list(LLM_BACKENDS_RUN),
        "readout_r4c": READOUT_R4C,
        "readout_delay": READOUT_DELAY,
        "readout_r5": READOUT_R5,
        "sha256": preregistration_hash(),
    }
