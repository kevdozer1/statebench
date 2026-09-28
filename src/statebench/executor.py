"""Scripted skill primitives. FIXED across every condition.

A primitive's *motion* is computed only from the state dict the decision model
saw, plus proprioception. Proprioception is always available because it is the
robot's own joint encoders and its own kinematic model - it is not privileged
information, and without it the arm could not be commanded at all. What varies
by condition is the object and relational state.

Event emission is a separate matter and is deliberately not part of the control
path: ``record_primitive_events`` is an observation-layer function (schema source
``history``) that watches what physically happened and appends to the event
stream, which the schema then exposes as ``events_recent`` and which the
"drop events_recent" ablation removes. The executor never reads that outcome to
decide where to move.

Declared fallback policy, applied identically in every condition. When a
primitive needs a geometry field that the current state does not carry, it uses,
in order:

1. the most recent history snapshot that does carry the field,
2. the task prior (the centre of the declared cube distribution for the target;
   the receptacle's fixed nominal pose for the receptacle),

and emits ``field_missing_fallback``. This makes a dropped field measurably worse
rather than trivially fatal, which is the point of the ablation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .observations import (
    RECEPTACLE_PRIOR,
    TABLE_CENTER_PRIOR,
)
from .scene import GRASP_Z, HOME_TCP, PLACE_Z, TRAVEL_Z
from .verify import cube_moved_with_gripper

DURATIONS = {
    "approach": (1.00, 0.80),
    "close": 0.55,
    "test_lift": 0.80,
    "transport": (1.20, 0.80),
    "release": 0.45,
    # Turn 3b clarification: inspect waits for a fresh observation rather than
    # merely pausing. 0.50 s equals the delay condition's pipeline latency, so
    # the wait is real sim time and is charged against the 30 s timeout. Before
    # Turn 3b this was 0.30 s and inspect returned an equally stale observation,
    # which made the delay condition a measurement of the executor.
    "inspect": 0.50,
    "retreat": (0.70, 1.00),
}
PERTURBATION_MOVE_S = 0.35


def _relative(block: Any) -> np.ndarray | None:
    if not isinstance(block, dict):
        return None
    try:
        return np.array([float(block["dx"]), float(block["dy"]), float(block["dz"])])
    except (KeyError, TypeError, ValueError):
        return None


def _yaw(block: Any) -> float | None:
    if not isinstance(block, dict) or "dyaw_rad" not in block:
        return None
    try:
        return float(block["dyaw_rad"])
    except (TypeError, ValueError):
        return None


class Executor:
    """Runs one primitive per call and logs exactly what it read."""

    def __init__(self, env, events: list[dict[str, Any]], perturbation_offset: np.ndarray | None):
        self.env = env
        self.events = events
        self.perturbation_offset = perturbation_offset
        self.perturbation_fired = False
        self.close_count = 0
        self.log: list[dict[str, Any]] = []

    # ------------------------------------------------------------- resolution
    def _resolve_target(
        self, state: dict[str, Any], history: list[dict[str, Any]]
    ) -> tuple[np.ndarray, float | None, str]:
        """World-frame target position and desired gripper yaw, plus the source used."""
        tcp = self.env.proprio_tcp()
        tcp_yaw = self.env.proprio_tcp_yaw()
        block = (state.get("geometry") or {}).get("target_rel_gripper")
        rel = _relative(block)
        if rel is not None:
            dyaw = _yaw(block)
            return tcp + rel, (tcp_yaw + dyaw) if dyaw is not None else None, "state"
        for snapshot in reversed(history):
            block = (snapshot.get("geometry") or {}).get("target_rel_gripper")
            rel = _relative(block)
            if rel is not None:
                dyaw = _yaw(block)
                return tcp + rel, (tcp_yaw + dyaw) if dyaw is not None else None, "history"
        self._emit("field_missing_fallback")
        return TABLE_CENTER_PRIOR.copy(), None, "prior"

    def _resolve_receptacle(
        self, state: dict[str, Any], history: list[dict[str, Any]]
    ) -> tuple[np.ndarray, str]:
        """Receptacle position, chained through the two declared geometry fields."""
        tcp = self.env.proprio_tcp()
        for source, candidate in (("state", state), *(("history", s) for s in reversed(history))):
            geometry = candidate.get("geometry") or {}
            rel_g = _relative(geometry.get("target_rel_gripper"))
            rel_r = _relative(geometry.get("target_rel_receptacle"))
            if rel_g is not None and rel_r is not None:
                return tcp + rel_g - rel_r, source
        # The receptacle is fixed and named in the goal block: its nominal pose
        # is a task constant, not per-episode state.
        return RECEPTACLE_PRIOR.copy(), "prior"

    def _emit(self, name: str, **extra: Any) -> None:
        self.events.append({"t": round(self.env.t, 3), "event": name, **extra})

    # ------------------------------------------------------------- primitives
    def execute(
        self, primitive: str, state: dict[str, Any], history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        t_start = self.env.t
        record: dict[str, Any] = {
            "primitive": primitive,
            "t_start": round(t_start, 3),
            "tcp_at_start": [round(float(v), 4) for v in self.env.proprio_tcp()],
        }
        handler = getattr(self, f"_do_{primitive}", None)
        if handler is None:
            self._emit("primitive_rejected", primitive=primitive)
            record.update(accepted=False, reason=f"unknown primitive {primitive!r}")
            self.env.hold(DURATIONS["inspect"])
            record["t_end"] = round(self.env.t, 3)
            self.log.append(record)
            return record
        record.update(handler(state, history))
        record["t_end"] = round(self.env.t, 3)
        record["tcp_at_end"] = [round(float(v), 4) for v in self.env.proprio_tcp()]
        record.setdefault("accepted", True)
        record_primitive_events(self.env, primitive, t_start, self.events, record)
        self.log.append(record)
        return record

    def _do_approach(self, state, history) -> dict[str, Any]:
        target, yaw, source = self._resolve_target(state, history)
        above, descend = DURATIONS["approach"]
        moved_a = self.env.move_to([target[0], target[1], TRAVEL_Z], yaw=yaw, seconds=above)
        moved_b = self.env.move_to([target[0], target[1], GRASP_Z], yaw=yaw, seconds=descend)
        return {
            "target_source": source,
            "commanded_target": [round(float(v), 4) for v in target],
            "commanded_yaw": None if yaw is None else round(float(yaw), 4),
            "accepted": bool(moved_a and moved_b),
        }

    def _do_close(self, state, history) -> dict[str, Any]:
        self.close_count += 1
        offset_applied = None
        if (
            self.close_count == 1
            and self.perturbation_offset is not None
            and not self.perturbation_fired
        ):
            self.perturbation_fired = True
            offset_applied = [round(float(v), 4) for v in self.perturbation_offset]
            self.env.move_to(
                self.env.proprio_tcp() + self.perturbation_offset, seconds=PERTURBATION_MOVE_S
            )
        self._emit("close_attempted")
        self.env.set_grip(True, DURATIONS["close"])
        return {"perturbation_offset_m": offset_applied, "close_index": self.close_count}

    def _do_test_lift(self, state, history) -> dict[str, Any]:
        self._emit("lift_started")
        tcp = self.env.proprio_tcp()
        moved = self.env.move_to([tcp[0], tcp[1], TRAVEL_Z], seconds=DURATIONS["test_lift"])
        return {"accepted": bool(moved)}

    def _do_transport(self, state, history) -> dict[str, Any]:
        receptacle, source = self._resolve_receptacle(state, history)
        across, lower = DURATIONS["transport"]
        moved_a = self.env.move_to(
            [receptacle[0], receptacle[1], TRAVEL_Z], yaw=0.0, seconds=across
        )
        moved_b = self.env.move_to(
            [receptacle[0], receptacle[1], PLACE_Z], yaw=0.0, seconds=lower
        )
        return {
            "receptacle_source": source,
            "commanded_target": [round(float(v), 4) for v in receptacle],
            "accepted": bool(moved_a and moved_b),
        }

    def _do_release(self, state, history) -> dict[str, Any]:
        self.env.set_grip(False, DURATIONS["release"])
        self._emit("release_done")
        return {}

    def _do_inspect(self, state, history) -> dict[str, Any]:
        self.env.hold(DURATIONS["inspect"])
        self._emit("inspect_done")
        return {}

    def _do_retreat(self, state, history) -> dict[str, Any]:
        tcp = self.env.proprio_tcp()
        up, back = DURATIONS["retreat"]
        self.env.move_to([tcp[0], tcp[1], TRAVEL_Z], seconds=up)
        self.env.move_to(HOME_TCP, yaw=0.0, seconds=back)
        self._emit("retreat_done")
        return {}


def record_primitive_events(
    env, primitive: str, t_start: float, events: list[dict[str, Any]], record: dict[str, Any]
) -> None:
    """Observation-layer event emission (schema source ``history``).

    Watches the physical outcome of the primitive just executed. Not part of the
    executor's control path.
    """
    t = round(env.t, 3)
    if primitive == "approach":
        events.append({"t": t, "event": "approach_done"})
    elif primitive == "test_lift":
        window = max(0.2, env.t - t_start)
        moved = cube_moved_with_gripper(env, t_start, window=window, motion_m=0.008)
        events.append({"t": t, "event": "lift_done"})
        if moved is False:
            events.append({"t": t, "event": "target_motion_diverged"})
        record["cube_tracked_gripper"] = moved
    elif primitive == "transport":
        events.append({"t": t, "event": "transport_done"})
    if record.get("accepted") is False:
        events.append({"t": t, "event": "primitive_rejected"})
