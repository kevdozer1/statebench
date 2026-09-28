"""v0.3 predicates (Turn 7). ``predicates.py`` (v0.2) is unchanged and still used by stack-off cells.

Same names and value types as v0.2. Two rules change; every other predicate is
v0.2's code, called unchanged.

**R-sem, aligned_for_grasp.** v0.2 required |dz| < 15 mm, a window just below the
cube centre, which a small cube can never satisfy (Turn 6: 0 of 800 V1 decisions).
v0.3 compares the TCP with the grasp pose computed from the estimated object pose
and its size, the rule ``reference_v6.decide`` uses (|TCP z - grasp z| < 15 mm):

    grasp pose = estimated cube centre + (0, 0, grasp_offset_m),
    grasp_offset_m = the executor's declared grasp height for the object minus its half-size
                   (V0: 0.022 - 0.020 = +0.002 m; V1: 0.024 - 0.008 = +0.016 m)
    aligned = |dx| < 0.012 and |dy| < 0.012 and |dz + grasp_offset_m| < 0.015 and |dyaw| < 0.10

(dz = cube minus TCP, so TCP z - grasp z = -(dz + grasp_offset_m).)

**R-time, estimate_stale.** v0.2: estimate_age_s > 0.3 s. v0.3: true when the object
is in contact or held and the arm has moved more than 5 mm since ``t_obs``
(``uncertainty.arm_motion_since_obs_m``, from PROPRIO); in contact or held =
``contact.gripper_target`` is ``contact`` or gripper_closed_on_object. False
otherwise; unknown when the motion field is missing.

``rules`` selects which change is on ({"sem", "time"}; v0.3 = both), for development.
"""

from __future__ import annotations

from typing import Any

from .predicates import ALLOWED_PATHS, DERIVED_FROM, FORMULAS, _aligned_for_grasp, build_predicates
from .provenance_audit import RecordingDict, _allowed
from .reference import ALIGN_XY_M, ALIGN_YAW_RAD, ALIGN_Z_M

STALE_MOTION_M = 0.005
PREDICATE_VERSION = "v0.3"
ALLOWED_PATHS_V3 = ALLOWED_PATHS | frozenset({"uncertainty.arm_motion_since_obs_m", "constants.grasp_offset_m"})

FORMULAS_V3 = dict(FORMULAS)
FORMULAS_V3["aligned_for_grasp"] = (
    f"|target_rel_gripper.dx| < {ALIGN_XY_M} and |dy| < {ALIGN_XY_M} and |dz + grasp_offset_m| < {ALIGN_Z_M} "
    f"and |dyaw_rad| < {ALIGN_YAW_RAD}; grasp_offset_m = declared grasp height minus the object's half-size")
FORMULAS_V3["estimate_stale"] = (
    f"(contact.gripper_target == 'contact' or gripper_closed_on_object) and "
    f"uncertainty.arm_motion_since_obs_m > {STALE_MOTION_M}")
DERIVED_FROM_V3 = {k: list(v) for k, v in DERIVED_FROM.items()}
DERIVED_FROM_V3["aligned_for_grasp"] = DERIVED_FROM["aligned_for_grasp"] + ["constants.grasp_offset_m"]
DERIVED_FROM_V3["estimate_stale"] = ["uncertainty.arm_motion_since_obs_m", "contact.gripper_target",
                                     "proprio.gripper_command", "geometry.gripper_opening_m"]


def _aligned_v3(state: dict[str, Any], grasp_offset_m: float) -> bool | str:
    block = (state.get("geometry") or {}).get("target_rel_gripper")
    if not isinstance(block, dict):
        return "unknown"
    try:
        dx, dy, dz = float(block["dx"]), float(block["dy"]), float(block["dz"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if abs(dx) >= ALIGN_XY_M or abs(dy) >= ALIGN_XY_M or abs(dz + grasp_offset_m) >= ALIGN_Z_M:
        return False
    dyaw = block.get("dyaw_rad")
    if dyaw is not None and abs(float(dyaw)) >= ALIGN_YAW_RAD:
        return False
    return True


def _stale_v3(state: dict[str, Any], closed_on_object: Any) -> bool | str:
    motion = (state.get("uncertainty") or {}).get("arm_motion_since_obs_m")
    if motion is None:
        return "unknown"
    contact = (state.get("contact") or {}).get("gripper_target") == "contact"
    return bool((contact or closed_on_object is True) and float(motion) > STALE_MOTION_M)


def build_predicates_v3(state: dict[str, Any], proprio: dict[str, Any] | None = None,
                        history: list[dict[str, Any]] | None = None, constants: dict[str, Any] | None = None,
                        rules: frozenset[str] = frozenset({"sem", "time"})) -> dict[str, Any]:
    block = build_predicates(state, proprio, history)
    values = dict(block["values"])
    if "sem" in rules:
        values["aligned_for_grasp"] = _aligned_v3(state, float((constants or {})["grasp_offset_m"]))
    if "time" in rules:
        values["estimate_stale"] = _stale_v3(state, values["gripper_closed_on_object"])
    formulas = dict(FORMULAS)
    derived = {k: list(v) for k, v in DERIVED_FROM.items()}
    for name, flag in (("aligned_for_grasp", "sem"), ("estimate_stale", "time")):
        if flag in rules:
            formulas[name] = FORMULAS_V3[name]
            derived[name] = list(DERIVED_FROM_V3[name])
    return {"values": values, "source": "derived", "version": PREDICATE_VERSION, "rules": sorted(rules),
            "derived_from": derived, "formulas": formulas}


def audited_build_predicates_v3(state, proprio, history, constants, rules=frozenset({"sem", "time"})):
    """Run the v0.3 build under read recording. Returns (block, paths read, undeclared paths)."""
    log: set[str] = set()
    block = build_predicates_v3(RecordingDict(state, "", log),
                                None if proprio is None else RecordingDict(proprio, "proprio", log),
                                None if history is None else [RecordingDict(s, "history", log, collapse="history")
                                                              for s in history],
                                RecordingDict(constants or {}, "constants", log), rules)
    plain = build_predicates_v3(state, proprio, history, constants, rules)
    if plain["values"] != block["values"]:
        raise AssertionError("recording wrapper changed predicate values")
    undeclared = {p for p in log if not _allowed(p, ALLOWED_PATHS_V3)}
    return plain, log, undeclared


_ = _aligned_for_grasp  # v0.2 rule kept importable for comparison
