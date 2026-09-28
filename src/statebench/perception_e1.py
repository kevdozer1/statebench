"""Estimator E1: the cube's pose from one RGB-D frame. It learns nothing.

``estimate(rgb, depth_mm, camera, proprio, constants, memory)`` is a pure
function of exactly these arguments:

* the RGB-D frame from the existing camera at decision time,
* the camera intrinsics and pose (``camera``, as ``render_frames.camera_model``),
* PROPRIO (TCP pose, finger opening, gripper command, sim time),
* task constants (``CONSTANTS``: cube edge, tray pose and size, table height),
* its own previous estimates (``memory``, returned updated; the input is not mutated).

It never receives the environment or any simulator truth. ``tests/test_perception_e1.py``
checks the signature and the module source, and ``provenance_audit_e1`` records
every key it reads from the dict arguments.

Method, tuned on dev frames of seeds 0-199 only and frozen before seeds 400-498:

1. Red mask: ``r >= 80 and r >= 2 g and r >= 2 b``. The floor of 80 drops the
   dark mixed pixels where the cube meets a black finger, whose depth is the
   finger's.
2. Back-project masked pixels with valid depth (pixel centres) through K and the
   camera pose into world points; drop points farther than 45 mm from the
   per-axis median.
3. Top face: points within 4 mm of the 95th-percentile height; the cube centre
   height is the top face's median height minus half an edge.
4. Yaw modulo 90 degrees, first rule that applies:
   a. *grasp constraint*: gripper commanded closed, finger opening 20-55 mm and
      the points within 30 mm of the TCP: the fingers force the cube's faces onto
      the finger axis, so yaw = TCP yaw (PROPRIO) modulo 90;
   b. *top-face box*: the TCP more than 80 mm from the points (the hand cannot
      occlude): the angle minimising the area of the box around the top-face
      points (1 degree grid, 0.1 degree refinement);
   c. *silhouette*: otherwise, the angle whose projected cube silhouette best
      matches the red mask, ignoring pixels in front of the cube (occluders); and
      if E1's last top-face-box estimate, re-anchored at this frame, lies within
      10 mm, the cube has not moved and keeps that yaw (*static memory*).
5. Centre: in the cube frame, anchored on the camera-facing extent of each
   horizontal axis (that face is seen to its edge), clamped so the cube contains
   every point (edge 40 mm).
6. Visible fraction: red pixels over the hull area of the estimated cube's 8
   projected corners.
7. Too little seen (fewer than 25 red pixels or 8 top-face points): no new
   estimate; the previous one is returned with its own time, ``fresh = False``,
   and the caller reports ``estimate_age_s = now - t_est``.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

E1_VERSION = "statebench-perception-e1/v1"

CONSTANTS = {
    "cube_edge_m": 0.040,
    "tray_center_xy_m": (0.44, 0.20),
    "tray_inner_half_m": 0.079,
    "tray_outer_half_m": 0.085,
    "tray_floor_top_m": 0.012,
    "tray_wall_top_m": 0.057,
    "table_z_m": 0.0,
}

RED_MIN = 80
RED_RATIO = 2.0
OUTLIER_M = 0.045
TOP_BAND_M = 0.004
MIN_PX = 25
MIN_TOP = 8
PIXEL_CENTER_OFFSET = 0.5
YAW_FROM = "all"  # "top" | "all"
YAW_MODE = "hybrid"  # "silhouette" | "hybrid": top-face box when the hand is far
HAND_FAR_M = 0.08  # the hand cannot occlude the cube beyond this TCP distance
CENTRE_NEAR_HAND = "near_face"  # anchoring when the hand is within HAND_FAR_M
STATIC_YAW_MEMORY = True
STATIC_M = 0.010  # a cube within this of its last confident estimate has not moved
CENTRE_FROM = "near_face"  # "midpoint" | "near_face"

#: Keys E1 may read from its dict arguments. Anything else fails the audit.
DECLARED_READS = frozenset({
    "camera.K", "camera.camera_to_world.R", "camera.camera_to_world.t",
    "proprio.tcp_position_m", "proprio.tcp_yaw_rad", "proprio.gripper_opening_m",
    "proprio.gripper_command", "proprio.sim_time_s",
    "constants.cube_edge_m", "constants.tray_center_xy_m",
    "memory.last", "memory.track", "memory.confident",
})


def new_memory() -> dict[str, Any]:
    return {"last": None, "track": [], "confident": None}


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r >= RED_MIN) & (r * 10 >= int(RED_RATIO * 10) * g) & (r * 10 >= int(RED_RATIO * 10) * b)


def _backproject(us, vs, z, camera) -> np.ndarray:
    K = np.asarray(camera["K"], dtype=float)
    R = np.asarray(camera["camera_to_world"]["R"], dtype=float)
    t = np.asarray(camera["camera_to_world"]["t"], dtype=float)
    x = (us + PIXEL_CENTER_OFFSET - K[0, 2]) / K[0, 0] * z
    y = (vs + PIXEL_CENTER_OFFSET - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], axis=1)
    return pc @ R.T + t


def _project(points: np.ndarray, camera) -> np.ndarray:
    K = np.asarray(camera["K"], dtype=float)
    R = np.asarray(camera["camera_to_world"]["R"], dtype=float)
    t = np.asarray(camera["camera_to_world"]["t"], dtype=float)
    pc = (points - t) @ R
    return (pc @ K.T)[:, :2] / pc[:, 2:3]


def _box_area(xy: np.ndarray, theta: float) -> float:
    c, s = math.cos(theta), math.sin(theta)
    u = xy[:, 0] * c + xy[:, 1] * s
    v = -xy[:, 0] * s + xy[:, 1] * c
    return float((u.max() - u.min()) * (v.max() - v.min()))


def _yaw_mod90(xy: np.ndarray) -> float:
    grid = np.deg2rad(np.arange(0.0, 90.0, 1.0))
    best = min(grid, key=lambda th: _box_area(xy, th))
    fine = best + np.deg2rad(np.arange(-1.0, 1.01, 0.1))
    best = min(fine, key=lambda th: _box_area(xy, th))
    return float(best % (math.pi / 2))


def _hull_area(pts: np.ndarray) -> float:
    pts = sorted(map(tuple, pts))

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
    return abs(sum(hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
                   for i in range(len(hull)))) / 2.0


def _corners(center: np.ndarray, yaw: float, half: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    signs = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], dtype=float)
    return center + (signs * half) @ R.T


GRASP_OPEN_MIN_M = 0.020  # the finger band that means "closed on something"
GRASP_OPEN_MAX_M = 0.055
GRASP_NEAR_XY_M = 0.030
OCCLUDER_MARGIN_M = 0.004


def _centre_for_yaw(pts: np.ndarray, yaw: float, top_face: float, half: float, camera,
                    mode: str | None = None) -> np.ndarray:
    """Centre in the cube frame: anchored on the camera-facing extent of each axis
    (``near_face``), the far extent (``far_face``) or the midpoint, clamped so the
    cube contains every point; height from the top face."""
    mode = mode or CENTRE_FROM
    c, s = math.cos(yaw), math.sin(yaw)
    u = pts[:, 0] * c + pts[:, 1] * s
    v = -pts[:, 0] * s + pts[:, 1] * c
    cam_t = np.asarray(camera["camera_to_world"]["t"], dtype=float)
    cam_uv = (cam_t[0] * c + cam_t[1] * s, -cam_t[0] * s + cam_t[1] * c)
    out = []
    for axis, cam_axis in zip((u, v), cam_uv, strict=True):
        lo, hi = float(np.percentile(axis, 1)), float(np.percentile(axis, 99))
        mid = (lo + hi) / 2.0
        if mode == "near_face":
            mid = hi - half if cam_axis > mid else lo + half
        elif mode == "far_face":
            mid = lo + half if cam_axis > mid else hi - half
        a, b = hi - half, lo + half
        out.append(min(max(mid, min(a, b)), max(a, b)))
    return np.array([out[0] * c - out[1] * s, out[0] * s + out[1] * c, top_face - half])


def _grasp_constrained_yaw(pts: np.ndarray, proprio) -> float | None:
    """Closed on something with the cube at the fingers: the fingers force the cube's
    faces onto the finger axis, so its yaw modulo 90 is the TCP yaw (PROPRIO)."""
    if proprio.get("gripper_command") != "closed":
        return None
    opening = proprio.get("gripper_opening_m")
    if opening is None or not (GRASP_OPEN_MIN_M < float(opening) <= GRASP_OPEN_MAX_M):
        return None
    tcp = np.asarray(proprio["tcp_position_m"], dtype=float)
    if float(np.linalg.norm(np.median(pts, axis=0)[:2] - tcp[:2])) > GRASP_NEAR_XY_M:
        return None
    return float(float(proprio["tcp_yaw_rad"]) % (math.pi / 2))


def _silhouette_yaw(pts, top_face, half, mask, depth_mm, camera) -> float:
    """Yaw in [0, 90) degrees whose projected silhouette best matches the red mask,
    ignoring pixels that lie in front of the cube (occluders such as the fingers)."""
    vs, us = np.nonzero(mask)
    pad = 12
    u0, u1 = max(0, us.min() - pad), min(mask.shape[1], us.max() + pad + 1)
    v0, v1 = max(0, vs.min() - pad), min(mask.shape[0], vs.max() + pad + 1)
    sub_mask = mask[v0:v1, u0:u1]
    sub_depth = depth_mm[v0:v1, u0:u1].astype(float) / 1000.0
    near = float(np.percentile(depth_mm[vs, us], 2)) / 1000.0
    occluder = (sub_depth > 0) & (sub_depth < near - OCCLUDER_MARGIN_M) & ~sub_mask
    gv, gu = np.mgrid[v0:v1, u0:u1]
    px = np.stack([gu.ravel() + PIXEL_CENTER_OFFSET, gv.ravel() + PIXEL_CENTER_OFFSET], axis=1)
    red = sub_mask.ravel()
    occ = occluder.ravel()

    def score(theta: float) -> float:
        centre = _centre_for_yaw(pts, theta, top_face, half, camera)
        hull = _convex(_project(_corners(centre, theta, half), camera))
        inside = _inside(px, hull)
        tp = np.sum(inside & red)
        denom = np.sum(inside & ~occ) + np.sum(red & ~inside)
        return tp / denom if denom else 0.0

    grid = np.deg2rad(np.arange(0.0, 90.0, 2.0))
    best = max(grid, key=score)
    fine = best + np.deg2rad(np.arange(-2.0, 2.01, 0.25))
    best = max(fine, key=score)
    return float(best % (math.pi / 2))


def _convex(pts: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, pts))

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
    return np.asarray(lower[:-1] + upper[:-1])


def _inside(px: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Pixels inside a counter-clockwise convex polygon."""
    ok = np.ones(len(px), dtype=bool)
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ok &= (b[0] - a[0]) * (px[:, 1] - a[1]) - (b[1] - a[1]) * (px[:, 0] - a[0]) >= 0
    return ok


def estimate(rgb, depth_mm, camera, proprio, constants, memory):
    """One E1 step. Returns (estimate dict, updated memory)."""
    memory = copy.deepcopy(memory) if memory is not None else new_memory()
    t_now = float(proprio["sim_time_s"])
    tcp = np.asarray(proprio["tcp_position_m"], dtype=float)
    half = float(constants["cube_edge_m"]) / 2.0

    mask = _red_mask(rgb) & (depth_mm > 0)
    vs, us = np.nonzero(mask)
    n_px = int(len(us))
    result: dict[str, Any] | None = None
    if n_px >= MIN_PX:
        pts = _backproject(us.astype(float), vs.astype(float), depth_mm[vs, us].astype(float) / 1000.0, camera)
        med = np.median(pts, axis=0)
        keep = np.all(np.abs(pts - med) <= OUTLIER_M, axis=1)
        pts = pts[keep]
        if len(pts) >= MIN_PX:
            top_z = float(np.percentile(pts[:, 2], 95))
            top = pts[pts[:, 2] >= top_z - TOP_BAND_M]
            if len(top) >= MIN_TOP:
                top_face = float(np.median(top[:, 2]))
                grasp_yaw = _grasp_constrained_yaw(pts, proprio)
                far = float(np.linalg.norm(np.median(pts, axis=0)
                                           - np.asarray(proprio["tcp_position_m"], dtype=float))) > HAND_FAR_M
                confident = memory.get("confident")
                if grasp_yaw is not None:
                    yaw, yaw_source = grasp_yaw, "grasp_constraint"
                elif YAW_MODE == "hybrid" and far:
                    yaw, yaw_source = _yaw_mod90(top[:, :2]), "top_face_box"
                else:
                    yaw, yaw_source = _silhouette_yaw(pts, top_face, half, mask, depth_mm, camera), "silhouette"
                    if confident is not None and STATIC_YAW_MEMORY:
                        # The open hand occludes the faces; a cube that has not moved
                        # keeps the yaw seen when the hand was far.
                        probe = _centre_for_yaw(pts, float(confident["yaw_mod90_rad"]), top_face, half,
                                                camera, mode=CENTRE_NEAR_HAND)
                        if float(np.linalg.norm(probe - np.asarray(confident["cube_m"]))) < STATIC_M:
                            yaw, yaw_source = float(confident["yaw_mod90_rad"]), "static_memory"
                centre = _centre_for_yaw(pts, yaw, top_face, half, camera,
                                         mode=CENTRE_FROM if far else CENTRE_NEAR_HAND)
                if yaw_source == "top_face_box":
                    memory["confident"] = {"cube_m": [float(v_) for v_ in centre], "yaw_mod90_rad": yaw,
                                           "t_est": t_now}
                area = _hull_area(_project(_corners(centre, yaw, half), camera))
                result = {
                    "t_est": t_now, "fresh": True,
                    "cube_m": [float(v_) for v_ in centre], "yaw_mod90_rad": yaw,
                    "visible_fraction": float(min(1.0, n_px / area)) if area > 0 else 0.0,
                    "red_px": n_px, "top_px": int(len(top)), "yaw_source": yaw_source,
                }
    if result is None:
        last = memory.get("last")
        if last is None:
            result = {"t_est": None, "fresh": False, "cube_m": None, "yaw_mod90_rad": None,
                      "visible_fraction": 0.0, "red_px": n_px, "top_px": 0}
        else:
            result = dict(last, fresh=False, red_px=n_px, visible_fraction=0.0 if n_px == 0
                          else float(last["visible_fraction"]) * n_px / max(1, int(last["red_px"])))
    result["t"] = t_now
    result["age_s"] = None if result["t_est"] is None else round(t_now - float(result["t_est"]), 3)
    if result["fresh"]:
        memory["last"] = {k: result[k] for k in ("t_est", "cube_m", "yaw_mod90_rad", "visible_fraction",
                                                  "red_px", "top_px")}
        memory["track"].append({"t": t_now, "cube_m": result["cube_m"], "tcp_m": [float(x) for x in tcp]})
        memory["track"] = [e for e in memory["track"] if e["t"] >= t_now - 2.0 - 1e-9]
    return result, memory
