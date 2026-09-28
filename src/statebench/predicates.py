"""R4 predicates: the decision conditions, pre-computed by the state encoder.

Turn 2 found that Jev carries the numbers but does not act on them. It never
released on numeric arrival, never escaped a repeated ``close_attempted``, never
reopened a closed empty gripper and never reacted to ``estimate_age_s``. This
module tests the obvious hypothesis: that the failure is in deriving the
condition, not in choosing the action once the condition is stated.

**Every predicate is computed from fields already present in the R3 state or in
PROPRIO. None reads simulator truth.** ``check_provenance`` enforces that
mechanically: a predicate declares the field paths it consumed, and the check
fails if any path is outside the declared observation set.

Thresholds are exactly the ones ``reference.py`` and ``backends/rules.py``
already use for their own conditions, imported rather than restated, so the
predicates cannot silently drift from the controller they are meant to mirror:

* ``ALIGN_XY_M`` 0.012, ``ALIGN_Z_M`` 0.015, ``ALIGN_YAW_RAD`` 0.10
* ``OVER_RECEPTACLE_M`` 0.030, ``PLACE_CLEARANCE_M`` 0.060
* ``FINGERS_ON_SOMETHING_M`` 0.020, ``RELEASE_OPENING_M`` 0.055

One note on the grip band. The turn brief describes the on-object band as "about
30 to 45 mm"; the constants already in the repo give ``0.020 < opening <= 0.055``.
The measured opening closed on the 40 mm cube is 38.4 mm and closed on nothing is
about 12 mm, so both descriptions separate the two cases identically on this
task. The existing constants are used, because the instruction to reuse
``reference.py``'s thresholds is the binding one and a fresh pair of magic
numbers would be a second source of truth.
"""

from __future__ import annotations

from typing import Any

from .backends.rules import FINGERS_ON_SOMETHING_M, PLACE_CLEARANCE_M
from .reference import ALIGN_XY_M, ALIGN_YAW_RAD, ALIGN_Z_M, OVER_RECEPTACLE_M
from .verify import RELEASE_OPENING_M

STALE_AGE_S = 0.3

#: Field paths a predicate is allowed to read. Anything else fails the check.
ALLOWED_PATHS = frozenset({
    "geometry.target_rel_gripper.dx",
    "geometry.target_rel_gripper.dy",
    "geometry.target_rel_gripper.dz",
    "geometry.target_rel_gripper.dyaw_rad",
    "geometry.target_rel_receptacle.dx",
    "geometry.target_rel_receptacle.dy",
    "geometry.target_rel_receptacle.dz",
    "geometry.gripper_opening_m",
    "relations.target_moving_with_gripper",
    "relations.gripper_above_target",
    "contact.gripper_target",
    "events_recent",
    "uncertainty.estimate_age_s",
    "history",
    "proprio.gripper_command",
    "proprio.gripper_opening_m",
})

#: Human-readable formulas, printed in the wrap-up.
FORMULAS = {
    "aligned_for_grasp":
        f"|target_rel_gripper.dx| < {ALIGN_XY_M} and |dy| < {ALIGN_XY_M} and "
        f"|dz| < {ALIGN_Z_M} and |dyaw_rad| < {ALIGN_YAW_RAD}",
    "gripper_closed_on_object":
        f"proprio.gripper_command == 'closed' and "
        f"{FINGERS_ON_SOMETHING_M} < gripper_opening_m <= {RELEASE_OPENING_M}",
    "gripper_closed_empty":
        f"proprio.gripper_command == 'closed' and "
        f"gripper_opening_m <= {FINGERS_ON_SOMETHING_M}",
    "object_moving_with_gripper":
        "relations.target_moving_with_gripper, passed through unchanged "
        "(true / false / unknown)",
    "object_at_receptacle":
        f"|target_rel_receptacle.dx| < {OVER_RECEPTACLE_M} and "
        f"|dy| < {OVER_RECEPTACLE_M} and 0 <= dz < {PLACE_CLEARANCE_M}",
    "released_at_receptacle":
        "the most recent entry of events_recent is release_done, and "
        "object_at_receptacle evaluated on the history snapshot nearest that "
        "event's timestamp was true",
    "estimate_stale":
        f"uncertainty.estimate_age_s > {STALE_AGE_S}",
    "close_attempts_since_last_lift":
        "count of close_attempted entries in events_recent that occur after the "
        "last lift_started or lift_done entry (0 if none); the window is the "
        "schema's 2 s event horizon",
}

#: Field paths each predicate consumes, checked by check_provenance.
DERIVED_FROM = {
    "aligned_for_grasp": [
        "geometry.target_rel_gripper.dx", "geometry.target_rel_gripper.dy",
        "geometry.target_rel_gripper.dz", "geometry.target_rel_gripper.dyaw_rad",
    ],
    "gripper_closed_on_object": ["proprio.gripper_command", "geometry.gripper_opening_m"],
    "gripper_closed_empty": ["proprio.gripper_command", "geometry.gripper_opening_m"],
    "object_moving_with_gripper": ["relations.target_moving_with_gripper"],
    "object_at_receptacle": [
        "geometry.target_rel_receptacle.dx", "geometry.target_rel_receptacle.dy",
        "geometry.target_rel_receptacle.dz",
    ],
    "released_at_receptacle": ["events_recent", "history"],
    "estimate_stale": ["uncertainty.estimate_age_s"],
    "close_attempts_since_last_lift": ["events_recent"],
}

PREDICATE_NAMES = tuple(FORMULAS)

_LIFT_EVENTS = ("lift_started", "lift_done")


def _rel(state: dict[str, Any], key: str) -> dict[str, Any] | None:
    block = (state.get("geometry") or {}).get(key)
    return block if isinstance(block, dict) else None


def _object_at_receptacle(state: dict[str, Any]) -> bool | str:
    block = _rel(state, "target_rel_receptacle")
    if block is None:
        return "unknown"
    try:
        dx, dy, dz = float(block["dx"]), float(block["dy"]), float(block["dz"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    return bool(
        abs(dx) < OVER_RECEPTACLE_M
        and abs(dy) < OVER_RECEPTACLE_M
        and 0.0 <= dz < PLACE_CLEARANCE_M
    )


def _aligned_for_grasp(state: dict[str, Any]) -> bool | str:
    block = _rel(state, "target_rel_gripper")
    if block is None:
        return "unknown"
    try:
        dx, dy, dz = float(block["dx"]), float(block["dy"]), float(block["dz"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if abs(dx) >= ALIGN_XY_M or abs(dy) >= ALIGN_XY_M or abs(dz) >= ALIGN_Z_M:
        return False
    dyaw = block.get("dyaw_rad")
    if dyaw is not None and abs(float(dyaw)) >= ALIGN_YAW_RAD:
        return False
    return True


def _grip(state: dict[str, Any], proprio: dict[str, Any] | None) -> tuple[str | None, float | None]:
    """The last gripper command and the finger opening.

    When PROPRIO is not supplied - which happens only when recomputing predicates
    offline from a stored log, never in the closed loop - the command is derived
    from the opening instead. On this gripper the two are equivalent: open rests
    at 89 mm, closed on the 40 mm cube at 38 mm and closed on nothing at 12 mm,
    so ``opening <= RELEASE_OPENING_M`` and "the last command was close" pick out
    exactly the same states. Still only declared fields.
    """
    command = (proprio or {}).get("gripper_command")
    opening = (state.get("geometry") or {}).get("gripper_opening_m")
    if opening is None:
        opening = (proprio or {}).get("gripper_opening_m")
    opening = None if opening is None else float(opening)
    if command is None and opening is not None:
        command = "closed" if opening <= RELEASE_OPENING_M else "open"
    return command, opening


def _close_attempts_since_last_lift(events: list[dict[str, Any]] | None) -> int:
    if not events:
        return 0
    last_lift = -1
    for i, event in enumerate(events):
        if event.get("event") in _LIFT_EVENTS:
            last_lift = i
    return sum(
        1 for event in events[last_lift + 1:] if event.get("event") == "close_attempted"
    )


def _released_at_receptacle(
    state: dict[str, Any], history: list[dict[str, Any]] | None
) -> bool:
    events = state.get("events_recent") or []
    if not events or events[-1].get("event") != "release_done":
        return False
    t_release = events[-1].get("t")
    if t_release is None or not history:
        # No history to check against: fall back to the current geometry, which
        # is the same instant for a release that just happened.
        return _object_at_receptacle(state) is True
    best = None
    best_gap = None
    for snapshot in history:
        stamps = (snapshot.get("provenance") or {}).get("t_obs") or {}
        if not stamps:
            continue
        t_snapshot = max(stamps.values())
        gap = abs(float(t_snapshot) - float(t_release))
        if best_gap is None or gap < best_gap:
            best, best_gap = snapshot, gap
    if best is None:
        return _object_at_receptacle(state) is True
    return _object_at_receptacle(best) is True


def build_predicates(
    state: dict[str, Any],
    proprio: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the predicate block for an R3-family state.

    Returns the block that R4 appends to the state, including per-predicate
    provenance (``source: derived`` plus the field paths consumed).
    """
    command, opening = _grip(state, proprio)
    closed = command == "closed"
    events = state.get("events_recent")
    moving = (state.get("relations") or {}).get("target_moving_with_gripper", "unknown")
    age = (state.get("uncertainty") or {}).get("estimate_age_s")

    values: dict[str, Any] = {
        "aligned_for_grasp": _aligned_for_grasp(state),
        "gripper_closed_on_object": (
            "unknown" if opening is None or command is None
            else bool(closed and FINGERS_ON_SOMETHING_M < opening <= RELEASE_OPENING_M)
        ),
        "gripper_closed_empty": (
            "unknown" if opening is None or command is None
            else bool(closed and opening <= FINGERS_ON_SOMETHING_M)
        ),
        "object_moving_with_gripper": moving,
        "object_at_receptacle": _object_at_receptacle(state),
        "released_at_receptacle": _released_at_receptacle(state, history),
        "estimate_stale": (
            "unknown" if age is None else bool(float(age) > STALE_AGE_S)
        ),
        "close_attempts_since_last_lift": _close_attempts_since_last_lift(events),
    }
    return {
        "values": values,
        "source": "derived",
        "derived_from": {name: list(paths) for name, paths in DERIVED_FROM.items()},
        "formulas": dict(FORMULAS),
    }


def check_provenance(
    block: dict[str, Any], allowed: frozenset[str] = ALLOWED_PATHS
) -> list[str]:
    """Return violations: predicate field paths outside the declared set."""
    problems: list[str] = []
    if block.get("source") != "derived":
        problems.append(f"predicates.source is {block.get('source')!r}, expected 'derived'")
    declared = block.get("derived_from") or {}
    for name in PREDICATE_NAMES:
        if name not in (block.get("values") or {}):
            problems.append(f"missing predicate value: {name}")
        if name not in declared:
            problems.append(f"missing derived_from for: {name}")
            continue
        for path in declared[name]:
            if path not in allowed:
                problems.append(f"{name} reads undeclared field path: {path}")
    return problems
