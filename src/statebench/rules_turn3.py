"""TURN 3 PRE-REGISTRATION. Hashed before the grid; verified unchanged at the end.

Turn 2 established that Jev carries the numbers but does not act on them: it never
released on numeric arrival, never escaped a repeated ``close_attempted``, never
reopened a closed empty gripper, and never reacted to ``estimate_age_s``. This
turn tests whether the failure is in deriving the condition rather than in
choosing the action, by having the state encoder pre-compute the conditions as
predicates (R4), and separately by naming the conditions in the prompt (R4b).

It also adds a frozen closed-form head - multinomial logistic regression on the
structured fields, no text, no network - as the cheapest possible backend, so
there is a floor to read Jev against on held-out seeds.

The task, executor, verifier, schema and the three question strings are
unchanged. ``executor.py``, ``verify.py``, ``schema.py`` and ``questions.py`` are
byte-identical to Turn 2 and their hashes are recorded below.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA_VERSION = "statebench/state/v0.1"

# ---------------------------------------------------------------- fixed files
FIXED_FILE_SHA256 = {
    "executor.py": "a4411d74bb66af7dc2e80bc1285b971fcbb8b6216d657b02f55e42b73c4abcb3",
    "verify.py": "75b5bde219b737541f6e8bcce4d40508d5d580c0d937b4ba0ebe8bf9b50f9903",
    "schema.py": "c4f5b2fb508cd060b4f85105af7a806af17b0bc4ffcb0e7ef8105fbbca93cc94",
    "questions.py": "0fd6e9149073886de784bf83a85d310b0eecf7e90790e53c39ded9794b31206e",
}
TURN1_PREREG_SHA256 = "072f60d0e23f9b0a05c1efeeb2d0c94b11d86f06a5fbe1ae5b589ea70f259915"
TURN2_PREREG_SHA256 = "75d42b5f3d64368b435b5cb201c6af87d9f36238047e606cc045fcdb385a264a"

# ---------------------------------------------------------------------- seeds
DEV_SEED_POOL = tuple(range(0, 200))
#: Turn 3 opens the held-out band for the frozen-head comparison only.
FROZEN_HEAD_SEED_POOL = tuple(range(200, 300))
RESERVED_SEED_POOL = tuple(range(300, 1000))
SEED_STRIDE = 4
EPISODE_SEEDS = tuple(DEV_SEED_POOL[::SEED_STRIDE])[:50]
#: Held-out comparison set: stride 2 through 200-299 gives 50 seeds.
HELDOUT_SEEDS = tuple(range(200, 300, 2))[:50]

MAX_DECISION_STEPS = 24
TIMEOUT_SECONDS = 30.0

# ------------------------------------------------------------------- backends
COMPARATOR_BACKEND = "rules"
JEV = "jev"
JEV_OBJECT = "jev_object_state"
LLM_CHEAP = "llm_cheap"
LLM_STRONG = "llm_strong"
FROZEN_HEAD = "frozen_head"
BACKENDS = (JEV, JEV_OBJECT, LLM_CHEAP, LLM_STRONG, FROZEN_HEAD)
OPENROUTER_BACKENDS = (LLM_CHEAP, LLM_STRONG)

# ------------------------------------------------------- R4b prompt condition
#: The ONLY permitted replacement for the next_action instruction, verbatim.
#: This is prompt engineering and every table that shows an R4b row says so.
R4B_NEXT_ACTION_INSTRUCTION = (
    "Choose approach if not aligned_for_grasp and not gripper_closed_on_object. "
    "Choose close if aligned_for_grasp and the gripper is open. "
    "Choose test_lift if gripper_closed_on_object and object_moving_with_gripper is "
    "unknown. "
    "Choose transport if gripper_closed_on_object and object_moving_with_gripper. "
    "Choose release if object_at_receptacle and gripper_closed_on_object. "
    "Choose retreat if released_at_receptacle. "
    "Choose inspect if estimate_stale. "
    "If gripper_closed_empty, choose approach."
)

# ------------------------------------------------------------------ the grid
#: (backend, condition, n, seed_set). seed_set is "dev" or "heldout".
GRID: tuple[tuple[str, str, int, str], ...] = (
    # Jev first: free, uncapped, logged.
    (JEV, "R4_predicates", 50, "dev"),
    (JEV, "R4b_conditions_in_prompt", 50, "dev"),
    (JEV, "R4_delay_0p5s", 50, "dev"),
    (JEV, "R4_noise_2cm", 50, "dev"),
    (JEV_OBJECT, "R4_predicates", 50, "dev"),
    (JEV, "R4_predicates", 50, "heldout"),
    (FROZEN_HEAD, "R4_predicates", 50, "heldout"),
    # Then the paid arms.
    (LLM_CHEAP, "R4_predicates", 50, "dev"),
    (LLM_CHEAP, "R4b_conditions_in_prompt", 50, "dev"),
    (LLM_STRONG, "R4_predicates", 30, "dev"),
    (LLM_STRONG, "R4b_conditions_in_prompt", 30, "dev"),
)

#: If the cap is hit, these are dropped in this order.
DROP_ORDER = (
    (LLM_STRONG, "R4b_conditions_in_prompt"),
    (LLM_STRONG, "R4_predicates"),
)

# --------------------------------------------------------------------- budget
OPENROUTER_BALANCE_USD = 11.64
LLM_SPEND_CAP_USD = 8.00
JEV_UNCAPPED = True

# ------------------------------------------------------------- call mechanics
RETRY_MAX_ATTEMPTS = 3
RETRY_ON_STATUSES = (429, 500, 502, 503, 504, 520, 522, 524)
MAX_BACKEND_ERRORS_PER_EPISODE = 3
QUESTION_WORDING_FROZEN_IN = "src/statebench/questions.py"

# ---------------------------------------------------------------- frozen head
FROZEN_HEAD_SPEC = (
    "Three independent scikit-learn LogisticRegression heads, L2, default "
    "settings, no tuning, no text embedding and no neural network. Features are "
    "every numeric field in R4 (geometry deltas, gripper opening, estimate age), "
    "every boolean in relations and predicates (one-hot over true/false/unknown), "
    "and close_attempts_since_last_lift. Fitted on Turn 1 and Turn 2 logs from "
    "seeds 0-199 only. next_action imitates the action the rules backend took at "
    "that state; object_held uses Turn 1's operational label; step_complete uses "
    "the verifier's final outcome for that step's episode. Evaluated closed-loop "
    "on seeds 200-299 with the same executor and verifier."
)
FROZEN_HEAD_TRAIN_RUNS = ("turn1", "turn2")
FROZEN_HEAD_TRAIN_SEEDS = "0-199 only"

# -------------------------------------------------------------------- metrics
PRIMARY_METRIC = "success rate, as judged by verify.py, with a Wilson 95% interval"
PAIRED_COMPARISON = (
    "against the Turn 1 rules backend on the same seeds and condition family: "
    "discordant counts b (model succeeded, rules failed) and c (the reverse), "
    "with an exact-binomial sign test on b+c"
)
SECONDARY_METRICS = (
    "time to success over successful episodes only",
    "recovery rate on episodes where the forced-miss perturbation fired",
    "judge-versus-verifier disagreement, both directions",
    "BACKEND_ERROR step count and excluded-episode count",
    "latency p50 and p95 per backend",
    "spend per condition, printed as a running total after each",
)

DELAY_AGE_SPLIT_S = 0.3
DELAY_PROBE = (
    "on the delay conditions, inspect counts split by the estimate_stale "
    "predicate rather than by the raw age, since R4 exposes the predicate"
)

# ---------------------------------------------------------------- calibration
CALIBRATION_HOLDOUT_UNIT = "episode"
CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONFIDENCE_DECILES = tuple(i / 10 for i in range(11))
CONFIDENCE_VS_CORRECTNESS = (
    "for jev next_action on R3_typed_history and on R4_predicates: bucket calls by "
    "returned confidence decile and report the rate at which the chosen action "
    "matched what the rules backend would have chosen from the same state. Reports "
    "whether confidence separates right from wrong at all."
)

# ------------------------------------------------------------------- outcomes
OUTCOME_CLEAR = "CLEAR DIFFERENCE"
OUTCOME_NO_CLEAR = "NO CLEAR DIFFERENCE"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
PERMITTED_OUTCOMES = (OUTCOME_CLEAR, OUTCOME_NO_CLEAR, OUTCOME_INCONCLUSIVE)
MIN_COMPLETED_EPISODES = 30

#: (backend_a, condition_a, run_a, backend_b, condition_b, run_b)
CONTRASTS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (JEV, "R4_predicates", "turn3", JEV, "R3_no_verifier_fields", "turn2"),
    (JEV, "R4_predicates", "turn3", JEV, "R3_typed_history", "turn2"),
    (JEV, "R4b_conditions_in_prompt", "turn3", JEV, "R4_predicates", "turn3"),
    (JEV, "R4_delay_0p5s", "turn3", JEV, "R3_delay_0p5s", "turn2"),
    (JEV, "R4_predicates@heldout", "turn3", FROZEN_HEAD, "R4_predicates@heldout", "turn3"),
    (LLM_STRONG, "R4_predicates", "turn3", LLM_STRONG, "R3_typed_history", "turn2"),
    (LLM_CHEAP, "R4_predicates", "turn3", LLM_CHEAP, "R3_typed_history", "turn2"),
)

# ------------------------------------------------------------------- figures
HEADLINE_FIGURE = "headline_success.png"
RECOVERY_FIGURE = "recovery.png"
FIGURE_REPRESENTATION_ORDER = (
    "R1_caption", "R2_typed", "R3_typed_history", "R3_no_verifier_fields",
    "R4_predicates", "R4b_conditions_in_prompt",
)
FIGURE_BACKEND_ORDER = ("rules", "jev", "llm_cheap", "llm_strong", "frozen_head")


def preregistration_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fixed_files_ok() -> dict[str, dict[str, object]]:
    here = Path(__file__).resolve().parent
    out: dict[str, dict[str, object]] = {}
    for name, expected in FIXED_FILE_SHA256.items():
        actual = hashlib.sha256((here / name).read_bytes()).hexdigest()
        out[name] = {
            "expected": expected,
            "actual": actual,
            "match": (actual == expected) if expected != "PENDING" else None,
        }
    return out


def seeds_for(seed_set: str) -> tuple[int, ...]:
    return HELDOUT_SEEDS if seed_set == "heldout" else EPISODE_SEEDS


def cell_key(backend: str, condition: str, seed_set: str) -> str:
    return f"{backend}__{condition}" + ("@heldout" if seed_set == "heldout" else "")


def summary() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "turn1_prereg_sha256": TURN1_PREREG_SHA256,
        "turn2_prereg_sha256": TURN2_PREREG_SHA256,
        "fixed_files": fixed_files_ok(),
        "grid": [list(row) for row in GRID],
        "drop_order": [list(row) for row in DROP_ORDER],
        "dev_seeds": list(EPISODE_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "reserved_seeds": f"{RESERVED_SEED_POOL[0]}-{RESERVED_SEED_POOL[-1]} (untouched)",
        "r4b_instruction": R4B_NEXT_ACTION_INSTRUCTION,
        "openrouter_balance_usd": OPENROUTER_BALANCE_USD,
        "llm_spend_cap_usd": LLM_SPEND_CAP_USD,
        "jev_uncapped": JEV_UNCAPPED,
        "frozen_head_spec": FROZEN_HEAD_SPEC,
        "frozen_head_train_runs": list(FROZEN_HEAD_TRAIN_RUNS),
        "frozen_head_train_seeds": FROZEN_HEAD_TRAIN_SEEDS,
        "primary_metric": PRIMARY_METRIC,
        "paired_comparison": PAIRED_COMPARISON,
        "secondary_metrics": list(SECONDARY_METRICS),
        "delay_probe": DELAY_PROBE,
        "confidence_vs_correctness": CONFIDENCE_VS_CORRECTNESS,
        "permitted_outcomes": list(PERMITTED_OUTCOMES),
        "min_completed_episodes": MIN_COMPLETED_EPISODES,
        "contrasts": [list(c) for c in CONTRASTS],
        "figures": [HEADLINE_FIGURE, RECOVERY_FIGURE],
        "sha256": preregistration_hash(),
    }
