"""Phase 3 state generator. A data source, never an evaluated policy.

It follows the reference controller (``reference.decide`` on RICH_TRUTH, the v0.1
executor's primitives, no forced-miss perturbation) and perturbs only the
placement. One run per (seed, variant):

``normal``               the reference placement, then retreat.
``high_release``         release 5 cm higher than normal over the tray centre.
``offset_release``       release with the cube centre 6-9 cm off the tray centre
                         toward one wall, from above the wall top, so it lands
                         inside, on the rim or outside.
``table_release``        release 12 cm off the tray centre, just above the table.
``gripped_stop``         stop with the cube inside the tray volume, still gripped.
``released_no_retreat``  normal placement and release, then stop with the gripper
                         still inside the tray.

After a perturbed release the reference controller continues (it may re-grasp and
succeed); the two ``stop`` variants hold 2.0 s and end. Every run keeps the
benchmark budget (24 decisions, 30 s).

At every 10 Hz recorder tick the run logs the COMPACT_TRUTH state (the R3
packet), the predicate block computed exactly as the closed loop computes it for
R4 (``prepare_v2`` with PROPRIO and the ablated history), the three completion
labels, and a simulator-truth sidecar used only to define strata. No reader ever
sees the labels or the sidecar.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import reference
from .env import StateBenchEnv
from .executor import DURATIONS, Executor
from .observations import HistoryBuffer, build_compact_truth, build_proprio, build_rich_truth
from .representations_v2 import CONDITIONS_V2_BY_NAME, prepare_v2
from .runner_v2 import completion_labels
from .scene import (
    CUBE_HALF,
    PLACE_Z,
    RECEPTACLE_CENTER,
    RECEPTACLE_FLOOR_TOP,
    RECEPTACLE_INNER_HALF,
    RECEPTACLE_WALL_TOP,
    TRAVEL_Z,
)
from .verify import RUNNING, SUCCESS, Verifier

GENERATOR_VERSION = "statebench-judge-generator/v1"
VARIANTS = ("normal", "high_release", "offset_release", "table_release",
            "gripped_stop", "released_no_retreat")
WALLS = (("+x", (1.0, 0.0)), ("-x", (-1.0, 0.0)), ("+y", (0.0, 1.0)), ("-y", (0.0, -1.0)))
OFFSET_RANGE_M = (0.06, 0.09)
TABLE_OFFSET_M = 0.12
HIGH_RELEASE_DZ_M = 0.05
#: Release height over a wall: cube bottom 15 mm above the wall top.
OVER_WALL_TCP_Z = RECEPTACLE_WALL_TOP + CUBE_HALF + 0.015
#: Release height beside the tray: cube bottom 10 mm above the table.
TABLE_TCP_Z = CUBE_HALF + 0.010
STOP_HOLD_S = 2.0
MAX_DECISIONS = 24
RECORD_DT = 0.1
NEAR_MISS_OUTSIDE_M = 0.03


def variant_params(seed: int, variant: str) -> dict[str, Any]:
    """Placement parameters: a pure function of (seed, variant)."""
    rng = np.random.default_rng(70_000 + 16 * int(seed) + VARIANTS.index(variant))
    cx, cy = RECEPTACLE_CENTER
    if variant == "high_release":
        return {"release_xy": [cx, cy], "release_z": PLACE_Z + HIGH_RELEASE_DZ_M}
    if variant in ("offset_release", "table_release"):
        name, (ux, uy) = WALLS[int(rng.integers(0, 4))]
        d = float(rng.uniform(*OFFSET_RANGE_M)) if variant == "offset_release" else TABLE_OFFSET_M
        z = OVER_WALL_TCP_Z if variant == "offset_release" else TABLE_TCP_Z
        return {"wall": name, "offset_m": round(d, 4),
                "release_xy": [cx + ux * d, cy + uy * d], "release_z": z}
    return {}


class _GenExecutor(Executor):
    """The v0.1 executor with one transport target overridden, once."""

    placement: tuple[list[float], float] | None = None

    def _do_transport(self, state, history):
        if self.placement is None:
            return super()._do_transport(state, history)
        (x, y), z = self.placement
        self.placement = None
        across, lower = DURATIONS["transport"]
        moved_a = self.env.move_to([x, y, TRAVEL_Z], yaw=0.0, seconds=across)
        moved_b = self.env.move_to([x, y, z], yaw=0.0, seconds=lower)
        return {"receptacle_source": "generator_placement",
                "commanded_target": [round(float(v), 4) for v in (x, y, z)],
                "accepted": bool(moved_a and moved_b)}


def truth_sidecar(env) -> dict[str, Any]:
    cube = env.truth_cube_position()
    contacts = env.truth_cube_contacts()
    dx = abs(float(cube[0]) - RECEPTACLE_CENTER[0])
    dy = abs(float(cube[1]) - RECEPTACLE_CENTER[1])
    z = float(cube[2])
    ex = max(0.0, dx - RECEPTACLE_INNER_HALF)
    ey = max(0.0, dy - RECEPTACLE_INNER_HALF)
    ez = max(0.0, RECEPTACLE_FLOOR_TOP - z, z - RECEPTACLE_WALL_TOP)
    return {
        "cube_m": [round(float(v), 5) for v in cube],
        "cube_yaw_rad": round(float(env.truth_cube_yaw()), 5),
        "fingers": sorted(contacts["fingers"]),
        "touches_receptacle": bool(contacts["receptacle"]),
        "touches_table": bool(contacts["table"]),
        "tcp_m": [round(float(v), 5) for v in env.proprio_tcp()],
        "gripper_opening_m": round(float(env.proprio_gripper_opening()), 5),
        "distance_outside_volume_m": round(float(np.sqrt(ex * ex + ey * ey + ez * ez)), 5),
    }


def near_miss_kind(sidecar: dict[str, Any], labels: dict[str, Any]) -> str | None:
    """Near-miss class of a not-placed state, or None."""
    if labels["placed"]:
        return None
    in_volume = labels["conditions"]["cube_center_in_volume"]
    if in_volume and sidecar["fingers"]:
        return "inside_still_gripped"
    if not in_volume and sidecar["touches_receptacle"]:
        return "on_rim_or_against_wall"
    if not in_volume and sidecar["distance_outside_volume_m"] <= NEAR_MISS_OUTSIDE_M:
        return "just_outside_volume"
    return None


class _TickRecorder:
    def __init__(self, env, history, events, verifier, frames_hook=None):
        self.env = env
        self.history = history
        self.events = events
        self.verifier = verifier
        self.next_t = 0.0
        self.ticks: list[dict[str, Any]] = []
        self.frames_hook = frames_hook
        self.phase = "pre_placement"
        self.rng = np.random.default_rng(0)

    def __call__(self, env) -> None:
        if env.t + 1e-9 < self.next_t:
            return
        self.next_t = env.t + RECORD_DT
        compact = build_compact_truth(env, self.events)
        self.history.push(env.t, compact)
        labels = completion_labels(self.verifier)
        r4, _, _ = prepare_v2(CONDITIONS_V2_BY_NAME["R4j"], compact, self.history, env.t,
                              self.rng, proprio=build_proprio(env))
        sidecar = truth_sidecar(env)
        tick = {
            "t": round(env.t, 3),
            "phase": self.phase,
            "r3": compact,
            "predicates": r4["predicates"]["values"],
            "labels": {"complete_now": labels["complete_now"], "placed": labels["placed"]},
            "conditions": labels["conditions"],
            "truth": sidecar,
            "near_miss": near_miss_kind(sidecar, labels),
        }
        if self.frames_hook is not None:
            tick["frame"] = self.frames_hook(env, len(self.ticks))
        self.ticks.append(tick)


def generate_run(seed: int, variant: str, frames_hook=None, render: bool = False) -> dict[str, Any]:
    env = StateBenchEnv(seed, render=render)
    events: list[dict[str, Any]] = []
    history = HistoryBuffer()
    verifier = Verifier(env)
    recorder = _TickRecorder(env, history, events, verifier, frames_hook)
    env.step_hooks.append(verifier.update)
    env.step_hooks.append(recorder)
    executor = _GenExecutor(env, events, None)
    params = variant_params(seed, variant)
    placement_done = False
    primitives: list[str] = []
    stopped = None
    recorder(env)
    for _ in range(MAX_DECISIONS):
        if verifier.status != RUNNING:
            break
        compact = build_compact_truth(env, events)
        history.push(env.t, compact)
        snapshots = [s for _, s in history.snapshots()]
        primitive = reference.decide(build_rich_truth(env))
        if primitive == "transport" and not placement_done:
            placement_done = True
            recorder.phase = "placement"
            if "release_xy" in params:
                executor.placement = (params["release_xy"], params["release_z"])
                executor.execute("transport", compact, snapshots)
                primitives.append("transport*")
                compact = build_compact_truth(env, events)
                history.push(env.t, compact)
                executor.execute("release", compact, [s for _, s in history.snapshots()])
                primitives.append("release*")
                recorder.phase = "post_placement"
                continue
            executor.execute("transport", compact, snapshots)
            primitives.append("transport")
            if variant == "gripped_stop":
                recorder.phase = "stopped"
                env.hold(STOP_HOLD_S)
                stopped = "gripped_stop"
                break
            continue
        executor.execute(primitive, compact, snapshots)
        primitives.append(primitive)
        if placement_done and primitive == "release":
            recorder.phase = "post_placement"
            if variant == "released_no_retreat":
                recorder.phase = "stopped"
                env.hold(STOP_HOLD_S)
                stopped = "released_no_retreat"
                break
    if verifier.status == RUNNING:
        verifier.finish("generator run ended without a verified placement")
    success = verifier.status == SUCCESS
    for tick in recorder.ticks:
        tick["labels"]["eventual_success"] = success
    env.close()
    return {
        "generator": GENERATOR_VERSION, "seed": int(seed), "variant": variant,
        "params": params, "primitives": primitives, "stopped": stopped,
        "success": success, "verifier": verifier.report(), "sim_seconds": round(env.t, 3),
        "ticks": recorder.ticks,
    }


def run_seed(args: tuple[int, str]) -> dict[str, Any]:
    """Pool worker: all six variants of one seed, packed into one gzip JSONL file."""
    seed, out_dir = args
    path = Path(out_dir) / f"seed{int(seed):04d}.jsonl.gz"
    summary = []
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for variant in VARIANTS:
            run = generate_run(seed, variant)
            handle.write(json.dumps(run, separators=(",", ":")) + "\n")
            summary.append({"variant": variant, "success": run["success"],
                            "ticks": len(run["ticks"]), "primitives": run["primitives"],
                            "params": run["params"]})
    return {"seed": int(seed), "path": str(path), "runs": summary}


def load_seed(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
