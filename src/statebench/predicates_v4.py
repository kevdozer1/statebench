"""v0.4 predicates (Turn 8): v0.3's R-sem plus R-grip. ``predicates.py`` and ``predicates_v3.py`` are unchanged.

**R-grip, size-aware grip band.** v0.2-v0.3 call the gripper closed on the object when
the finger opening is in a fixed 20-55 mm band. A 16 mm cube held between the fingers
reads 16-20 mm and is called ``gripper_closed_empty`` (Turn 7: ``semantic`` in every
rules_v2 V1 failure label). v0.4 measures the opening relative to the object size:

    expected held opening = object size - 1.7 mm   (measured on dev: V0 38.4 mm for the 40 mm cube,
                                                     V1 14.2 mm for the 16 mm cube after the lift)
    threshold = (closed-on-nothing opening + expected held opening) / 2
              = (12.5 mm + size - 1.7 mm) / 2   (V0: 25.4 mm, V1: 13.4 mm)
    gripper_closed_on_object = commanded closed and threshold < opening <= 55 mm
    gripper_closed_empty     = commanded closed and opening <= threshold

The closed-on-nothing opening (12.5 mm) and the held offset (1.7 mm) are declared
gripper constants, measured on dev seeds (25 of 25 grasps each size; closed on nothing
12.5 mm every time); the object size is the declared size E1 uses
(``constants.object_size_m``). The fixed 20 mm band is retired. Dev tuning: a first
version used (12.5 mm + size) / 2 = 14.25 mm for V1, which the held-and-lifted cube
(14.2 mm) falls below.

Everything else is v0.3 with the sem rule only (``aligned_for_grasp`` size-aware;
``estimate_stale`` stays v0.2's age rule).
"""

from __future__ import annotations

from typing import Any

from .predicates_v3 import ALLOWED_PATHS_V3, build_predicates_v3
from .provenance_audit import RecordingDict, _allowed
from .verify import RELEASE_OPENING_M

PREDICATE_VERSION = "v0.4"
CLOSED_ON_NOTHING_M = 0.0125
HELD_OFFSET_M = 0.0017
ALLOWED_PATHS_V4 = ALLOWED_PATHS_V3 | frozenset({"constants.object_size_m"})
FORMULAS_V4 = {
    "gripper_closed_on_object": (f"proprio.gripper_command == 'closed' and (0.0125 + object_size_m - 0.0017) / 2 < "
                                 f"gripper_opening_m <= {RELEASE_OPENING_M}"),
    "gripper_closed_empty": ("proprio.gripper_command == 'closed' and "
                             "gripper_opening_m <= (0.0125 + object_size_m - 0.0017) / 2"),
}


def grip_threshold_m(object_size_m: float) -> float:
    return (CLOSED_ON_NOTHING_M + float(object_size_m) - HELD_OFFSET_M) / 2.0


def grip_state(command: str | None, opening: float | None, object_size_m: float) -> tuple[Any, Any]:
    if command is None or opening is None:
        return "unknown", "unknown"
    closed = command == "closed"
    th = grip_threshold_m(object_size_m)
    return bool(closed and th < opening <= RELEASE_OPENING_M), bool(closed and opening <= th)


def build_predicates_v4(state: dict[str, Any], proprio: dict[str, Any] | None = None,
                        history: list[dict[str, Any]] | None = None,
                        constants: dict[str, Any] | None = None) -> dict[str, Any]:
    block = build_predicates_v3(state, proprio, history, constants, frozenset({"sem"}))
    command = (proprio or {}).get("gripper_command")
    opening = (state.get("geometry") or {}).get("gripper_opening_m")
    if opening is None:
        opening = (proprio or {}).get("gripper_opening_m")
    opening = None if opening is None else float(opening)
    if command is None and opening is not None:
        command = "closed" if opening <= RELEASE_OPENING_M else "open"
    on, empty = grip_state(command, opening, float((constants or {})["object_size_m"]))
    block["values"]["gripper_closed_on_object"] = on
    block["values"]["gripper_closed_empty"] = empty
    for name in ("gripper_closed_on_object", "gripper_closed_empty"):
        block["formulas"][name] = FORMULAS_V4[name]
        block["derived_from"][name] = ["proprio.gripper_command", "geometry.gripper_opening_m",
                                       "constants.object_size_m"]
    block["version"] = PREDICATE_VERSION
    block["rules"] = ["grip", "sem"]
    return block


def audited_build_predicates_v4(state, proprio, history, constants):
    log: set[str] = set()
    block = build_predicates_v4(RecordingDict(state, "", log),
                                None if proprio is None else RecordingDict(proprio, "proprio", log),
                                None if history is None else [RecordingDict(s, "history", log, collapse="history")
                                                              for s in history],
                                RecordingDict(constants or {}, "constants", log))
    plain = build_predicates_v4(state, proprio, history, constants)
    if plain["values"] != block["values"]:
        raise AssertionError("recording wrapper changed predicate values")
    return plain, log, {p for p in log if not _allowed(p, ALLOWED_PATHS_V4)}
