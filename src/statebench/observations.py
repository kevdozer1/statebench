"""The observation interface: the only route from simulator state to a controller.

Three declared observation sets:

``PROPRIO``
    Joint positions, gripper opening, gripper actuator force, sim time. Nothing
    privileged. The executor always has this, because it is the robot's own
    encoders plus its own kinematic model.

``RICH_TRUTH``
    Full simulator state: every body pose, every contact pair, joint and object
    velocities. Reference condition only.

``COMPACT_TRUTH``
    The ``statebench/state/v0.1`` schema, populated from simulator truth.

Plus a 2 s history buffer of COMPACT_TRUTH snapshots, used by representations
that read history and by the 0.5 s delay ablation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .ik import wrap_to_half_pi
from .scene import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    RECEPTACLE_CENTER,
    RECEPTACLE_FLOOR_TOP,
    in_receptacle_volume,
)
from .schema import ACTIONS, SCHEMA_VERSION, empty_state

PROPRIO = "PROPRIO"
RICH_TRUTH = "RICH_TRUTH"
COMPACT_TRUTH = "COMPACT_TRUTH"

INSTRUCTION = "Pick up the red cube from the table and place it in the blue receptacle."
TARGET_OBJECT_ID = "red_cube"
RECEPTACLE_ID = "blue_receptacle"

HISTORY_SECONDS = 2.0
MOTION_WINDOW_S = 0.25
MOTION_MATCH_M = 0.006
MOTION_MIN_M = 0.006

# Task priors: constants of the task statement, not per-episode state. The
# executor falls back to these when a needed geometry field is absent.
TABLE_CENTER_PRIOR = np.array(
    [sum(CUBE_X_RANGE) / 2.0, sum(CUBE_Y_RANGE) / 2.0, 0.020]
)
RECEPTACLE_PRIOR = np.array(
    [RECEPTACLE_CENTER[0], RECEPTACLE_CENTER[1], RECEPTACLE_FLOOR_TOP]
)


def build_proprio(env) -> dict[str, Any]:
    return {
        "joint_positions_rad": env.proprio_joint_positions(),
        "gripper_opening_m": round(env.proprio_gripper_opening(), 4),
        "gripper_force_n": env.proprio_gripper_force(),
        "gripper_command": env.proprio_grip_command(),
        "tcp_position_m": [round(float(v), 4) for v in env.proprio_tcp()],
        "tcp_yaw_rad": round(env.proprio_tcp_yaw(), 4),
        "sim_time_s": round(env.t, 3),
        "provenance": {"source": "proprio", "t_obs": round(env.t, 3)},
    }


def build_rich_truth(env) -> dict[str, Any]:
    contacts = env.truth_cube_contacts()
    return {
        "proprio": build_proprio(env),
        "body_positions_m": env.truth_bodies(),
        "contact_pairs": env.truth_contact_pairs(),
        "cube_position_m": [round(float(v), 5) for v in env.truth_cube_position()],
        "cube_yaw_rad": round(env.truth_cube_yaw(), 5),
        "cube_velocity_mps": [round(float(v), 5) for v in env.truth_cube_velocity()],
        "cube_finger_contacts": sorted(contacts["fingers"]),
        "cube_touches_receptacle": bool(contacts["receptacle"]),
        "cube_touches_table": bool(contacts["table"]),
        "qpos": [round(float(v), 6) for v in env.data.qpos],
        "qvel": [round(float(v), 6) for v in env.data.qvel],
        "provenance": {"source": "sim_truth", "t_obs": round(env.t, 3)},
    }


def _moving_with_gripper(env) -> bool | None:
    t0 = env.t - MOTION_WINDOW_S
    if t0 < 0:
        return None
    pair = env.displacement_over(t0, MOTION_WINDOW_S)
    if pair is None:
        return None
    cube, tcp = pair
    if float(np.linalg.norm(tcp)) < MOTION_MIN_M:
        return None
    return bool(float(np.linalg.norm(cube - tcp)) < MOTION_MATCH_M)


def build_compact_truth(env, events: list[dict[str, Any]]) -> dict[str, Any]:
    """The v0.1 schema populated from simulator truth."""
    t = round(env.t, 3)
    tcp = env.proprio_tcp()
    cube = env.truth_cube_position()
    cube_yaw = env.truth_cube_yaw()
    contacts = env.truth_cube_contacts()
    fingers = contacts["fingers"]
    opening = env.proprio_gripper_opening()
    force = env.proprio_gripper_force()

    if contacts["receptacle"]:
        supported_by = "receptacle"
    elif contacts["table"]:
        supported_by = "table"
    elif fingers:
        supported_by = "gripper"
    else:
        supported_by = "none"

    moving = _moving_with_gripper(env)
    rel_gripper = cube - tcp
    rel_receptacle = cube - np.array([RECEPTACLE_CENTER[0], RECEPTACLE_CENTER[1], RECEPTACLE_FLOOR_TOP])

    if fingers:
        evidence = (
            f"{len(fingers)} finger pad(s) in contact with {TARGET_OBJECT_ID}; "
            f"gripper opening {opening:.3f} m, actuator force {force:.1f} N"
        )
    else:
        evidence = (
            f"no finger pad in contact with {TARGET_OBJECT_ID}; "
            f"gripper opening {opening:.3f} m, actuator force {force:.1f} N"
        )

    state = empty_state()
    state["goal"] = {
        "target_object_id": TARGET_OBJECT_ID,
        "receptacle_id": RECEPTACLE_ID,
        "instruction": INSTRUCTION,
    }
    state["geometry"] = {
        "target_rel_gripper": {
            "dx": round(float(rel_gripper[0]), 4),
            "dy": round(float(rel_gripper[1]), 4),
            "dz": round(float(rel_gripper[2]), 4),
            "dyaw_rad": round(wrap_to_half_pi(cube_yaw - env.proprio_tcp_yaw()), 4),
            "frame": "robot_base",
            "units": "m",
        },
        "target_rel_receptacle": {
            "dx": round(float(rel_receptacle[0]), 4),
            "dy": round(float(rel_receptacle[1]), 4),
            "dz": round(float(rel_receptacle[2]), 4),
            "frame": "robot_base",
            "units": "m",
        },
        "gripper_opening_m": round(opening, 4),
    }
    state["relations"] = {
        "target_supported_by": supported_by,
        "target_in_receptacle": bool(in_receptacle_volume(cube)),
        "target_moving_with_gripper": moving if moving is not None else "unknown",
        "gripper_above_target": bool(
            float(np.linalg.norm(rel_gripper[:2])) < 0.015 and tcp[2] > cube[2]
        ),
    }
    state["contact"] = {
        "gripper_target": "contact" if fingers else "no_contact",
        "evidence": evidence,
    }
    state["events_recent"] = [
        dict(e) for e in events if e["t"] >= t - HISTORY_SECONDS - 1e-9
    ]
    state["uncertainty"] = {
        "target_visible": True,
        "estimate_age_s": 0.0,
        "contact_observed": True,
    }
    state["actions_available"] = list(ACTIONS)
    source = {
        "geometry.target_rel_gripper": "sim_truth",
        "geometry.target_rel_receptacle": "sim_truth",
        "geometry.gripper_opening_m": "proprio",
        "relations.target_supported_by": "sim_truth",
        "relations.target_in_receptacle": "sim_truth",
        "relations.target_moving_with_gripper": "sim_truth",
        "relations.gripper_above_target": "sim_truth",
        "contact.gripper_target": "sim_truth",
        "events_recent": "history",
        "uncertainty": "sim_truth",
    }
    state["provenance"] = {
        "source": source,
        "t_obs": {path: t for path in source},
        "schema_version": SCHEMA_VERSION,
    }
    return state


class HistoryBuffer:
    """Last ``HISTORY_SECONDS`` of COMPACT_TRUTH snapshots, newest last."""

    def __init__(self, seconds: float = HISTORY_SECONDS):
        self.seconds = seconds
        self._items: deque[tuple[float, dict[str, Any]]] = deque()

    def push(self, t: float, state: dict[str, Any]) -> None:
        self._items.append((round(float(t), 3), state))
        while self._items and self._items[0][0] < t - self.seconds - 1e-9:
            self._items.popleft()

    def at_or_before(self, t: float) -> dict[str, Any] | None:
        """The newest snapshot at or before ``t``; None when the buffer is empty."""
        chosen = None
        for stamp, state in self._items:
            if stamp <= t + 1e-9:
                chosen = state
        if chosen is None and self._items:
            chosen = self._items[0][1]
        return chosen

    def snapshots(self) -> list[tuple[float, dict[str, Any]]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
