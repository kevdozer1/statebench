"""Turn 8 reference controller runs on RICH_TRUTH (no rendering): solvability and V3's slip rate.

``reference_v6.decide`` (unchanged) chooses each primitive from RICH_TRUTH; the Turn 6
executor carries it out at the variant's grasp and place heights; the forced miss is on
for perturbed seeds as in every closed-loop cell. **First-grasp slip:** the first
``transport`` that starts with both fingers on the cube; it slipped if, at the end of
that transport, the cube is more than 30 mm from the TCP or no finger touches it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import rules as prereg1
from .env_v8 import EnvV8
from .executor_v6 import ExecutorV6
from .observations import build_compact_truth, build_rich_truth
from .reference_v6 import decide
from .runner_v6 import MISS_OFFSET
from .scene import perturbation_fires
from .scene_v8 import VARIANTS
from .verify import RUNNING, SUCCESS, Verifier

SLIP_DISTANCE_M = 0.030


def heights(variant: str) -> tuple[float, float]:
    from .runner_v6 import V1_GRASP_Z

    cfg = VARIANTS[variant]
    return (V1_GRASP_Z if variant == "V1" else cfg.grasp_z), cfg.place_z


def run_reference(seed: int, variant: str, friction=None, mass=None,
                  max_steps: int = prereg1.MAX_DECISION_STEPS) -> dict[str, Any]:
    env = EnvV8(seed, variant=variant, friction=friction, mass=mass)
    events: list = []
    verifier = Verifier(env)
    env.step_hooks.append(verifier.update)
    fired = perturbation_fires(seed, prereg1.PERTURBATION_PROBABILITY)
    offset = np.array([0.0, 0.0, MISS_OFFSET.get(variant, 0.035)]) if fired else None
    gz, pz = heights(variant)
    ex = ExecutorV6(env, events, offset, gz, pz)
    first = None
    slips = 0
    transports = 0
    steps = []
    for _ in range(max_steps):
        if verifier.status != RUNNING:
            break
        rich = build_rich_truth(env)
        p = decide(rich, gz, pz)
        held_before = len(rich["cube_finger_contacts"]) >= 2
        ex.execute(p, build_compact_truth(env, events), [])
        if p == "transport" and held_before:
            transports += 1
            gap = float(np.linalg.norm(env.truth_cube_position() - env.proprio_tcp()))
            slipped = gap > SLIP_DISTANCE_M or not env.truth_cube_contacts()["fingers"]
            slips += int(slipped)
            if first is None:
                first = slipped
        steps.append(p)
    if verifier.status == RUNNING:
        verifier.finish("reference: step budget")
    report = verifier.report()
    env.close()
    return {"seed": seed, "variant": variant, "success": report["status"] == SUCCESS, "perturbed": fired,
            "first_grasp_slip": first, "transports_held": transports, "slips": slips, "steps": steps}
