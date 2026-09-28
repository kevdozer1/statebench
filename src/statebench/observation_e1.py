"""The estimated observation: E1's cube estimate written into the v0.1 schema.

``observations.build_compact_truth`` and ``executor.record_primitive_events`` are
unchanged. This module is their estimated counterpart and reads only E1's output,
PROPRIO, the executor's event list and task constants.

Declared rules:

* ``target_rel_gripper`` = cube estimate minus the PROPRIO TCP **at the estimate's
  time** (so the v0.2 executor registers it correctly); ``dyaw_rad`` =
  (estimated yaw modulo 90) minus the TCP yaw, wrapped into [-45, 45) degrees.
  The oracle wraps into [-90, 90); for a four-fold-symmetric cube both are
  graspable, and errors are compared modulo 90.
* ``target_rel_receptacle`` = cube estimate minus the constant tray floor centre.
* ``target_moving_with_gripper`` from E1's own track against the TCP track, with
  ``observations.py``'s window (0.25 s) and thresholds (6 mm match, 6 mm minimum
  TCP motion); ``unknown`` when the track does not reach back to the window start.
* ``gripper_above_target``: horizontal distance under 15 mm and TCP above the cube.
* **Contact rule.** ``contact`` when the finger opening is in the closed-on-something
  band (20-55 mm) and the estimate is within 25 mm of the TCP horizontally and 30 mm
  vertically; ``no_contact`` when the estimate is farther than 45 mm from the TCP,
  or the opening is at or below 20 mm (fingers closed on nothing); ``unknown``
  otherwise, including when there is no estimate yet.
* **Support rule.** ``receptacle`` when the estimate lies inside the tray's inner
  footprint and within 8 mm of resting height on the tray floor; ``table`` when it
  lies outside the tray's outer footprint and within 8 mm of resting height on the
  table; ``gripper`` when contact is ``contact`` and the estimate is more than 8 mm
  above either resting height; ``unknown`` otherwise.
* ``uncertainty``: ``target_visible`` = visible fraction at least 0.10;
  ``estimate_age_s`` = frame time minus the estimate's time (above 0 whenever E1
  fell back to its last estimate); ``contact_observed`` = contact is not ``unknown``.
* Provenance: ``estimated`` for every field E1 fills, ``proprio`` for the finger
  opening, ``history`` for events; ``t_obs`` = the estimate's time.
* ``target_motion_diverged`` after ``test_lift``: E1's cube displacement against the
  TCP displacement over the lift window, with the legacy thresholds (8 mm TCP
  motion, 10 mm match).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import ACTIONS, SCHEMA_VERSION, empty_state

MOTION_WINDOW_S = 0.25
MOTION_MATCH_M = 0.006
MOTION_MIN_M = 0.006
LIFT_MOTION_M = 0.008
LIFT_MATCH_M = 0.010
VISIBLE_MIN = 0.10
CONTACT_XY_M = 0.025
CONTACT_Z_M = 0.030
NO_CONTACT_M = 0.045
REST_TOL_M = 0.008
FINGERS_ON_SOMETHING_M = 0.020
RELEASE_OPENING_M = 0.055

INSTRUCTION = "Pick up the red cube from the table and place it in the blue receptacle."


def _wrap45(a: float) -> float:
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def _track_at(track: list[dict], t: float) -> dict | None:
    chosen = None
    for e in track:
        if e["t"] <= t + 1e-9:
            chosen = e
    return chosen


def moving_with_gripper(track: list[dict], t_now: float) -> bool | str:
    if not track:
        return "unknown"
    now = _track_at(track, t_now)
    then = _track_at(track, t_now - MOTION_WINDOW_S)
    if now is None or then is None or then is now or track[0]["t"] > t_now - MOTION_WINDOW_S + 1e-9:
        return "unknown"
    cube = np.asarray(now["cube_m"]) - np.asarray(then["cube_m"])
    tcp = np.asarray(now["tcp_m"]) - np.asarray(then["tcp_m"])
    if float(np.linalg.norm(tcp)) < MOTION_MIN_M:
        return "unknown"
    return bool(float(np.linalg.norm(cube - tcp)) < MOTION_MATCH_M)


def lift_diverged(track: list[dict], t_start: float, t_end: float) -> bool | None:
    """E1's counterpart of verify.cube_moved_with_gripper over a test_lift, negated."""
    a = _track_at(track, t_start)
    b = _track_at(track, t_end)
    if a is None or b is None or a is b:
        return None
    cube = np.asarray(b["cube_m"]) - np.asarray(a["cube_m"])
    tcp = np.asarray(b["tcp_m"]) - np.asarray(a["tcp_m"])
    if float(np.linalg.norm(tcp)) < LIFT_MOTION_M:
        return None
    return bool(float(np.linalg.norm(cube - tcp)) >= LIFT_MATCH_M)


def contact_rule(cube: np.ndarray | None, tcp: np.ndarray, opening: float) -> str:
    if opening <= FINGERS_ON_SOMETHING_M:
        return "no_contact"
    if cube is None:
        return "unknown"
    dxy = float(np.linalg.norm(cube[:2] - tcp[:2]))
    dz = abs(float(cube[2] - tcp[2]))
    if FINGERS_ON_SOMETHING_M < opening <= RELEASE_OPENING_M and dxy < CONTACT_XY_M and dz < CONTACT_Z_M:
        return "contact"
    if float(np.linalg.norm(cube - tcp)) > NO_CONTACT_M:
        return "no_contact"
    return "unknown"


def support_rule(cube: np.ndarray | None, contact: str, constants: dict) -> str:
    if cube is None:
        return "unknown"
    half = constants["cube_edge_m"] / 2.0
    cx, cy = constants["tray_center_xy_m"]
    dx, dy = abs(cube[0] - cx), abs(cube[1] - cy)
    floor_rest = constants["tray_floor_top_m"] + half
    table_rest = constants["table_z_m"] + half
    inside = dx < constants["tray_inner_half_m"] and dy < constants["tray_inner_half_m"]
    outside = dx > constants["tray_outer_half_m"] or dy > constants["tray_outer_half_m"]
    if inside and abs(cube[2] - floor_rest) <= REST_TOL_M:
        return "receptacle"
    if outside and abs(cube[2] - table_rest) <= REST_TOL_M:
        return "table"
    if contact == "contact" and cube[2] > min(floor_rest, table_rest) + REST_TOL_M:
        return "gripper"
    return "unknown"


def build_estimated_state(est: dict[str, Any], proprio_at_est: dict[str, Any] | None,
                          proprio_now: dict[str, Any], events: list[dict], track: list[dict],
                          constants: dict[str, Any]) -> dict[str, Any]:
    """The v0.1 schema filled from E1. ``proprio_at_est`` is PROPRIO at the estimate's time."""
    t_now = round(float(proprio_now["sim_time_s"]), 3)
    tcp_now = np.asarray(proprio_now["tcp_position_m"], dtype=float)
    opening = float(proprio_now["gripper_opening_m"])
    cube = None if est.get("cube_m") is None else np.asarray(est["cube_m"], dtype=float)
    t_est = t_now if est.get("t_est") is None else round(float(est["t_est"]), 3)
    ref = proprio_at_est or proprio_now
    tcp_ref = np.asarray(ref["tcp_position_m"], dtype=float)

    state = empty_state()
    state["goal"] = {"target_object_id": "red_cube", "receptacle_id": "blue_receptacle",
                     "instruction": INSTRUCTION}
    geometry: dict[str, Any] = {"gripper_opening_m": round(opening, 4)}
    relations: dict[str, Any] = {}
    contact = contact_rule(cube, tcp_now, opening)
    if cube is not None:
        rel = cube - tcp_ref
        geometry["target_rel_gripper"] = {
            "dx": round(float(rel[0]), 4), "dy": round(float(rel[1]), 4), "dz": round(float(rel[2]), 4),
            "dyaw_rad": round(_wrap45(float(est["yaw_mod90_rad"]) - float(ref["tcp_yaw_rad"])), 4),
            "frame": "robot_base", "units": "m"}
        floor = np.array([*constants["tray_center_xy_m"], constants["tray_floor_top_m"]])
        rr = cube - floor
        geometry["target_rel_receptacle"] = {
            "dx": round(float(rr[0]), 4), "dy": round(float(rr[1]), 4), "dz": round(float(rr[2]), 4),
            "frame": "robot_base", "units": "m"}
        relations["target_supported_by"] = support_rule(cube, contact, constants)
        cx, cy = constants["tray_center_xy_m"]
        inner = constants["tray_inner_half_m"]
        relations["target_in_receptacle"] = bool(
            abs(cube[0] - cx) < inner and abs(cube[1] - cy) < inner
            and constants["tray_floor_top_m"] <= cube[2] <= constants["tray_wall_top_m"])
        relations["gripper_above_target"] = bool(
            float(np.linalg.norm(cube[:2] - tcp_now[:2])) < 0.015 and tcp_now[2] > cube[2])
    else:
        relations["target_supported_by"] = "unknown"
        relations["target_in_receptacle"] = "unknown"
        relations["gripper_above_target"] = "unknown"
    relations["target_moving_with_gripper"] = moving_with_gripper(track, t_now)
    state["geometry"] = geometry
    state["relations"] = relations
    state["contact"] = {
        "gripper_target": contact,
        "evidence": (f"estimated from camera (E1): visible fraction {float(est.get('visible_fraction') or 0.0):.2f}, "
                     f"gripper opening {opening:.3f} m"),
    }
    state["events_recent"] = [dict(e) for e in events if e["t"] >= t_now - 2.0 - 1e-9]
    age = 0.0 if est.get("t_est") is None else round(t_now - float(est["t_est"]), 3)
    state["uncertainty"] = {
        "target_visible": bool(float(est.get("visible_fraction") or 0.0) >= VISIBLE_MIN),
        "estimate_age_s": max(0.0, age),
        "contact_observed": contact != "unknown",
    }
    state["actions_available"] = list(ACTIONS)
    source, t_obs = {}, {}
    for path in ("geometry.target_rel_gripper", "geometry.target_rel_receptacle",
                 "relations.target_supported_by", "relations.target_in_receptacle",
                 "relations.target_moving_with_gripper", "relations.gripper_above_target",
                 "contact.gripper_target", "uncertainty"):
        block, _, leaf = path.partition(".")
        if leaf and leaf not in (state.get(block) or {}):
            continue
        source[path] = "estimated"
        t_obs[path] = t_est
    source["geometry.gripper_opening_m"] = "proprio"
    t_obs["geometry.gripper_opening_m"] = t_now
    source["events_recent"] = "history"
    t_obs["events_recent"] = t_now
    state["provenance"] = {"source": source, "t_obs": t_obs, "schema_version": SCHEMA_VERSION}
    return state


def estimated_events(oracle_events: list[dict], diverged_estimated: list[dict]) -> list[dict]:
    """Executor facts as they are; target_motion_diverged replaced by E1's version."""
    base = [dict(e) for e in oracle_events if e.get("event") != "target_motion_diverged"]
    out = []
    for e in base:
        out.append(e)
        if e.get("event") == "lift_done":
            out += [dict(d) for d in diverged_estimated if abs(float(d["t"]) - float(e["t"])) < 1e-6]
    return out
