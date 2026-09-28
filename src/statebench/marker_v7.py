"""Turn 7 calibration marker: a declared hardware change, and the camera-side functions that use it.

**Hardware (declared).** A sphere of radius 12.5 mm, colour rgb (0.10, 0.70, 0.25),
rigidly mounted on the gripper base (``xarm_gripper_base_link``) at local position
(-0.040, -0.070, 0.045) m. It is visual only (no contacts, zero mass), so it does
not change the dynamics. In the TCP's yaw frame its centre sits at
``MARKER_OFFSET_YAW_FRAME`` = (+0.040, -0.070, +0.100) m from the TCP: the gripper
base frame is Rz(yaw) * diag(-1, 1, -1) in the world (checked on the model), and
the TCP is at local (0, 0, 0.145).

**Functions** (none receives the env, a Frame or any truth; each takes arrays,
the camera intrinsics or pose as a dict, and PROPRIO):

* ``detect_marker(rgb, depth_mm, K)``: green pixels (g >= 60, g >= 1.5 r, g >= 1.5 b),
  largest 4-connected blob; the viewing ray through the blob's centroid (colour only,
  so depth noise does not move it), and the distance = median over valid-depth blob
  pixels of (pixel depth + the sphere's depth profile r * sqrt(1 - rho^2) at that
  pixel's normalised distance rho from the centroid). Camera frame, metres.
* ``marker_world_from_proprio(proprio)``: TCP + Rz(yaw) * offset.
* ``solve_camera_pose(points_cam, points_world)``: Kabsch; camera-to-world R, t.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MARKER_RADIUS_M = 0.0125
MARKER_RGBA = (0.10, 0.70, 0.25, 1.0)
MARKER_BODY = "xarm_gripper_base_link"
MARKER_LOCAL_POS = (-0.040, -0.070, 0.045)
TCP_LOCAL_Z = 0.145
MARKER_OFFSET_YAW_FRAME = (-MARKER_LOCAL_POS[0], MARKER_LOCAL_POS[1], -(MARKER_LOCAL_POS[2] - TCP_LOCAL_Z))
GREEN_MIN = 60
GREEN_RATIO = 1.5
MIN_MARKER_PX = 12
PIXEL_CENTER_OFFSET = 0.5

DECLARED_READS = frozenset({"proprio.tcp_position_m", "proprio.tcp_yaw_rad", "camera.K",
                            "camera.camera_to_world.R", "camera.camera_to_world.t"})


def marker_geom_xml_attrs() -> dict[str, str]:
    return {"name": "calibration_marker", "type": "sphere", "size": f"{MARKER_RADIUS_M}",
            "pos": " ".join(f"{v}" for v in MARKER_LOCAL_POS),
            "rgba": " ".join(f"{v}" for v in MARKER_RGBA),
            "contype": "0", "conaffinity": "0", "mass": "0", "group": "1"}


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def marker_world_from_proprio(proprio: dict[str, Any]) -> np.ndarray:
    tcp = np.asarray(proprio["tcp_position_m"], dtype=float)
    return tcp + rz(float(proprio["tcp_yaw_rad"])) @ np.asarray(MARKER_OFFSET_YAW_FRAME)


def gripper_from_marker(marker_world: np.ndarray, tcp_yaw_rad: float) -> np.ndarray:
    """The TCP position implied by a marker position and the (proprio) yaw."""
    return np.asarray(marker_world, dtype=float) - rz(float(tcp_yaw_rad)) @ np.asarray(MARKER_OFFSET_YAW_FRAME)


def _green_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (g >= GREEN_MIN) & (g * 10 >= int(GREEN_RATIO * 10) * r) & (g * 10 >= int(GREEN_RATIO * 10) * b)


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected component (small images, simple flood fill)."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return mask
    seen = np.zeros_like(mask)
    best: list[tuple[int, int]] = []
    for v0, u0 in zip(vs, us):
        if seen[v0, u0]:
            continue
        stack, comp = [(v0, u0)], []
        seen[v0, u0] = True
        while stack:
            v, u = stack.pop()
            comp.append((v, u))
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = v + dv, u + du
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask)
    for v, u in best:
        out[v, u] = True
    return out


def detect_marker(rgb: np.ndarray, depth_mm: np.ndarray, K) -> dict[str, Any] | None:
    """The marker centre in the camera frame (x right, y down, z forward), or None."""
    colour = _green_mask(rgb)
    if int(colour.sum()) < MIN_MARKER_PX:
        return None
    blob = _largest_blob(colour)
    valid = blob & (depth_mm > 0)
    vs, us = np.nonzero(valid)
    if len(us) < MIN_MARKER_PX // 2:
        return None
    K = np.asarray(K, dtype=float)
    # direction: the colour blob's centroid (no depth involved)
    bv, bu = np.nonzero(blob)
    uc, vc = float(bu.mean()) + PIXEL_CENTER_OFFSET, float(bv.mean()) + PIXEL_CENTER_OFFSET
    # distance: each valid pixel's depth plus the sphere's profile at that pixel, then the median
    z = depth_mm[vs, us].astype(float) / 1000.0
    # each valid pixel's ray, and its measured distance along the ray
    rays = np.stack([(us + PIXEL_CENTER_OFFSET - K[0, 2]) / K[0, 0], (vs + PIXEL_CENTER_OFFSET - K[1, 2]) / K[1, 1],
                     np.ones_like(z)], axis=1)
    s_meas = z * np.linalg.norm(rays, axis=1)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    dc = np.array([(uc - K[0, 2]) / K[0, 0], (vc - K[1, 2]) / K[1, 1], 1.0])
    dc /= np.linalg.norm(dc)

    def residuals(lam: float) -> np.ndarray:
        c = lam * dc
        b = rays @ c
        disc = np.clip(b * b - lam * lam + MARKER_RADIUS_M ** 2, 0.0, None)
        return (b - np.sqrt(disc)) - s_meas

    # the centre lies along the centroid ray; its distance fits every pixel's ray-sphere intersection
    lam0 = float(np.median(s_meas)) + 0.7 * MARKER_RADIUS_M
    grid = lam0 + np.arange(-0.010, 0.010 + 1e-9, 0.0002)
    costs = [float(np.sum(np.minimum(residuals(g) ** 2, (0.01) ** 2))) for g in grid]
    lam = float(grid[int(np.argmin(costs))])
    for step in (0.0001, 0.00002, 0.000004):
        cand = lam + np.arange(-5, 6) * step
        lam = float(cand[int(np.argmin([np.sum(np.minimum(residuals(g) ** 2, 0.01 ** 2)) for g in cand]))])
    c = lam * dc
    res = residuals(lam)
    return {"center_cam_m": [float(v) for v in c], "px": int(blob.sum()), "valid_px": int(len(us)),
            "fit_rms_m": float(np.sqrt(np.mean(np.minimum(res ** 2, 0.01 ** 2))))}


def cam_to_world(point_cam, camera: dict[str, Any]) -> np.ndarray:
    R = np.asarray(camera["camera_to_world"]["R"], dtype=float)
    t = np.asarray(camera["camera_to_world"]["t"], dtype=float)
    return R @ np.asarray(point_cam, dtype=float) + t


def solve_camera_pose(points_cam: np.ndarray, points_world: np.ndarray) -> dict[str, Any]:
    """Kabsch: R, t minimising |R p_cam + t - p_world| over the correspondences."""
    pc, pw = np.asarray(points_cam, dtype=float), np.asarray(points_world, dtype=float)
    cc, cw = pc.mean(axis=0), pw.mean(axis=0)
    H = (pc - cc).T @ (pw - cw)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cw - R @ cc
    resid = np.linalg.norm((pc @ R.T + t) - pw, axis=1)
    return {"R": R.tolist(), "t": t.tolist(), "fit_rms_m": float(np.sqrt(np.mean(resid ** 2))),
            "n": int(len(pc))}


def pose_error(camera_a: dict[str, Any], camera_b: dict[str, Any]) -> dict[str, float]:
    """Rotation angle (deg) and translation distance (mm) between two camera-to-world poses."""
    Ra = np.asarray(camera_a["camera_to_world"]["R"], dtype=float)
    Rb = np.asarray(camera_b["camera_to_world"]["R"], dtype=float)
    ta = np.asarray(camera_a["camera_to_world"]["t"], dtype=float)
    tb = np.asarray(camera_b["camera_to_world"]["t"], dtype=float)
    M = Ra.T @ Rb
    sin = np.linalg.norm([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / 2.0
    cos = (np.trace(M) - 1.0) / 2.0
    return {"rotation_deg": float(math.degrees(math.atan2(sin, cos))),
            "translation_mm": float(np.linalg.norm(ta - tb) * 1000.0)}
