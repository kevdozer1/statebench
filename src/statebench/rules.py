"""PRE-REGISTRATION. Written before any run; its SHA-256 is printed in the wrap-up.

Nothing in this file may be edited after the hash is recorded. It fixes the
conditions, the episode seeds, the metrics and the permitted outcome words, so
that the analysis cannot be chosen after seeing the numbers.

Seed assignment. Seeds 0-999 are reserved for this task; 0-199 are development
and 200-999 are held out and are not touched this turn. The 50 episodes per
condition are drawn round-robin through the development pool with stride 4
(0, 4, 8, ..., 196), and the *same* 50 seeds are used for every condition. This
is what makes the design paired: the executor, the verifier, the seeds and the
perturbation schedule are identical across conditions, so a contrast between two
conditions is not confounded by which initial conditions each one happened to
get.

The perturbation schedule is a pure function of the seed
(``scene.perturbation_fires``), so it too is identical across conditions by
construction rather than by bookkeeping.

The ``test_lift`` primitive is available in every condition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA_VERSION = "statebench/state/v0.1"

# --------------------------------------------------------------------- design
DEV_SEED_POOL = tuple(range(0, 200))
HELD_OUT_SEED_POOL = tuple(range(200, 1000))
SEED_STRIDE = 4
N_PER_CONDITION = 50
EPISODE_SEEDS = tuple(DEV_SEED_POOL[::SEED_STRIDE])[:N_PER_CONDITION]

MAX_DECISION_STEPS = 24
TIMEOUT_SECONDS = 30.0
PERTURBATION_PROBABILITY = 0.3
#: 3.5 cm vertical, not the 2 cm first specified. 2 cm does not miss with this
#: gripper: see the measurement recorded in ``scene.perturbation_offset``.
PERTURBATION_OFFSET_M = 0.035

CONDITION_NAMES = (
    "R1_caption",
    "R2_typed",
    "R3_typed_history",
    "R3_drop_contact",
    "R3_drop_target_rel_gripper",
    "R3_drop_events_recent",
    "R3_noise_target_rel_gripper_2cm",
    "R3_delay_0p5s",
)

ACTIONS_AVAILABLE_EVERY_CONDITION = (
    "approach", "close", "test_lift", "transport", "release", "inspect", "retreat",
)

# --------------------------------------------------------------------- metrics
PRIMARY_METRIC = "success rate, as judged by verify.py, with a Wilson 95% interval"
SECONDARY_METRICS = (
    "time to success: sim seconds from episode start to the verifier's success, "
    "over successful episodes only",
    "recovery rate: success rate restricted to episodes where the forced-miss "
    "perturbation fired",
)
JUDGE_CHANNEL = (
    "the backend's step_complete probability, logged per step and never used to "
    "declare success"
)
PHYSICAL_CHANNEL = "verify.py, the only thing that may declare success"

# ------------------------------------------------------------------- outcomes
OUTCOME_CLEAR = "CLEAR DIFFERENCE"
OUTCOME_NO_CLEAR = "NO CLEAR DIFFERENCE"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
PERMITTED_OUTCOMES = (OUTCOME_CLEAR, OUTCOME_NO_CLEAR, OUTCOME_INCONCLUSIVE)
MIN_COMPLETED_EPISODES = 30

#: (condition_a, condition_b). Read as "a versus b".
CONTRASTS = (
    ("R2_typed", "R1_caption"),
    ("R3_typed_history", "R2_typed"),
    ("R3_typed_history", "R1_caption"),
    ("R3_drop_contact", "R3_typed_history"),
    ("R3_drop_target_rel_gripper", "R3_typed_history"),
    ("R3_drop_events_recent", "R3_typed_history"),
    ("R3_noise_target_rel_gripper_2cm", "R3_typed_history"),
    ("R3_delay_0p5s", "R3_typed_history"),
)

# ---------------------------------------------------------------- calibration
CALIBRATION_QUESTION = "object_held"
CALIBRATION_LABEL = (
    "the cube's displacement matched the TCP's to within 10 mm over the next 0.5 s of "
    "sim time, with the TCP moving at least 10 mm; decisions where the TCP moved less "
    "than that are excluded because the relation is not observable"
)
CALIBRATION_HOLDOUT_UNIT = "episode"
CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ------------------------------------------------------------- reference gate
REFERENCE_CEILING_REQUIRED = 0.90
REFERENCE_GATE = (
    "If the reference controller on RICH_TRUTH scores below 90% over 50 episodes, "
    "the task or the executor is fixed before anything else is run, and the change "
    "is recorded."
)


def preregistration_hash() -> str:
    """SHA-256 of this file's bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def summary() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "conditions": list(CONDITION_NAMES),
        "episode_seeds": list(EPISODE_SEEDS),
        "n_per_condition": N_PER_CONDITION,
        "held_out_seeds": f"{HELD_OUT_SEED_POOL[0]}-{HELD_OUT_SEED_POOL[-1]} (not used)",
        "max_decision_steps": MAX_DECISION_STEPS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "perturbation_probability": PERTURBATION_PROBABILITY,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "permitted_outcomes": list(PERMITTED_OUTCOMES),
        "min_completed_episodes": MIN_COMPLETED_EPISODES,
        "contrasts": [list(c) for c in CONTRASTS],
        "calibration_label": CALIBRATION_LABEL,
        "reference_gate": REFERENCE_GATE,
        "sha256": preregistration_hash(),
    }
