"""Turn 5 frames: RGB-D from the existing camera, plus truth-only sidecars.

The camera, resolution and archive format are Turn 4's (``render_frames.py``).
Added per frame: PROPRIO (so an estimator can be run offline exactly as online),
the oracle R3 state and R4 predicates, and a truth-rendered visibility figure
that no estimator reads:

``visible_fraction`` = cube pixels in a truth segmentation render divided by the
area of the convex hull of the true cube's 8 projected corners (its unoccluded
silhouette; exact for a convex box under a pinhole camera).

``JitteredEnv`` adds a small, seeded start-pose jitter to the cube without
touching ``env.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .env import StateBenchEnv
from .render_frames import camera_model
from .scene import CUBE_HALF

JITTER_XY_M = 0.010
JITTER_YAW_RAD = math.radians(5.0)


class JitteredEnv(StateBenchEnv):
    """StateBenchEnv whose cube start pose is offset by a seeded jitter (index 0 = none)."""

    def __init__(self, seed: int, jitter_index: int = 0, render: bool = False):
        self.jitter_index = int(jitter_index)
        super().__init__(seed, render=render)

    def reset(self) -> None:
        super().reset()
        if not self.jitter_index:
            return
        rng = np.random.default_rng(90_000 + 64 * self.seed + self.jitter_index)
        dx, dy = rng.uniform(-JITTER_XY_M, JITTER_XY_M, size=2)
        dyaw = rng.uniform(-JITTER_YAW_RAD, JITTER_YAW_RAD)
        q = self.data.qpos
        a = self.cube_qpos
        q[a] += dx
        q[a + 1] += dy
        yaw = 2.0 * math.atan2(q[a + 6], q[a + 3]) + dyaw
        q[a + 3: a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.trajectory.clear()
        self._last_log_t = -1.0
        self.hold(0.30)
        self.start_time = float(self.data.time)
        self.trajectory.clear()
        self._last_log_t = -1.0
        self._log_trajectory()
        self.jitter = {"dx": round(float(dx), 5), "dy": round(float(dy), 5), "dyaw": round(float(dyaw), 5)}


# ---------------------------------------------------------------- rendering
def render_rgbd(env) -> tuple[np.ndarray, np.ndarray]:
    r = env.renderer
    r.update_scene(env.data, camera=env.camera)
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(env.data, camera=env.camera)
    depth = r.render().copy()
    r.disable_depth_rendering()
    far = float(env.model.stat.extent * env.model.vis.map.zfar)
    depth_mm = np.where(depth >= far * 0.999, 0, np.round(depth * 1000.0)).clip(0, 65535).astype(np.uint16)
    return rgb, depth_mm


def truth_cube_mask(env) -> np.ndarray:
    r = env.renderer
    r.enable_segmentation_rendering()
    r.update_scene(env.data, camera=env.camera)
    seg = r.render().copy()
    r.disable_segmentation_rendering()
    return (seg[..., 0] == env.cube_geom) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))


def project(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """World points (N,3) to pixels (N,2) and camera-frame depth (N,)."""
    K = np.asarray(camera["K"])
    R = np.asarray(camera["camera_to_world"]["R"])
    t = np.asarray(camera["camera_to_world"]["t"])
    pc = (np.asarray(points) - t) @ R
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    return uv, pc[:, 2]


def _hull_area(pts: np.ndarray) -> float:
    pts = sorted(map(tuple, np.asarray(pts, dtype=float)))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def true_cube_corners(env) -> np.ndarray:
    c = env.truth_cube_position()
    R = env.data.xmat[env.cube_body].reshape(3, 3)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return c + (signs * CUBE_HALF) @ R.T


def visibility(env, camera: dict[str, Any]) -> dict[str, float]:
    mask = truth_cube_mask(env)
    uv, _ = project(true_cube_corners(env), camera)
    area = _hull_area(uv)
    visible = int(mask.sum())
    return {"visible_px": visible, "silhouette_px": round(area, 1),
            "visible_fraction": round(min(1.0, visible / area), 4) if area > 0 else 0.0}


def bucket(frac: float) -> str:
    if frac <= 0.0:
        return "0"
    if frac < 0.25:
        return "<25%"
    if frac <= 0.75:
        return "25-75%"
    return ">75%"


def closed_not_held(proprio: dict[str, Any], sidecar: dict[str, Any]) -> bool:
    """Gripper commanded closed and the cube not in bilateral finger contact."""
    return proprio["gripper_command"] == "closed" and len(sidecar["fingers"]) < 2


def proprio_for_meta(proprio: dict[str, Any]) -> dict[str, Any]:
    return {k: proprio[k] for k in ("joint_positions_rad", "gripper_opening_m", "gripper_force_n",
                                    "gripper_command", "tcp_position_m", "tcp_yaw_rad", "sim_time_s")}


def save_archive(path: Path, rgb: list, depth: list, t: list, meta: dict) -> int:
    np.savez_compressed(path, rgb=np.stack(rgb), depth_mm=np.stack(depth), t=np.asarray(t),
                        meta=np.asarray(json.dumps(meta)))
    return path.stat().st_size


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        return z["rgb"], z["depth_mm"], z["t"], json.loads(str(z["meta"]))
