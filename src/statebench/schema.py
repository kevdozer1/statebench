"""Typed state schema, version ``statebench/state/v0.1``.

Every field carries a provenance entry: ``source`` (one of ``sim_truth``,
``proprio``, ``estimated``, ``history``) and ``t_obs``, the sim time at which the
value was observed. A field written without provenance is a bug, and
:func:`validate_state` fails on it.

One addition to the originally specified field list, declared here rather than
made silently: ``geometry.target_rel_gripper.dyaw_rad``. The cube's yaw is
sampled per episode over +/- 45 degrees and the gripper's finger axis is a line,
so a gripper with fixed yaw cannot grasp every sampled pose. Without this field
no representation could express a graspable pose and the task would be unwinnable
for reasons unrelated to what the benchmark measures.

Two fields of the schema overlap with terms of the physical verifier:
``relations.target_in_receptacle`` and ``relations.target_supported_by``. That is
deliberate and is stated in ``notes/turn1.md``: in turn 1 every representation is
populated from simulator truth by construction, so the quantity of interest is
the contrast between representations, not the absolute success rate. No field
exposes the verifier's stability timer or its verdict, and the model's own
``step_complete`` answer (the judge channel) never declares success.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "statebench/state/v0.1"

SOURCES = ("sim_truth", "proprio", "estimated", "history")

ACTIONS = (
    "approach",
    "close",
    "test_lift",
    "transport",
    "release",
    "inspect",
    "retreat",
)

EVENT_NAMES = (
    "approach_done",
    "close_attempted",
    "lift_started",
    "lift_done",
    "target_motion_diverged",
    "transport_done",
    "release_done",
    "inspect_done",
    "retreat_done",
    "primitive_rejected",
    "field_missing_fallback",
)

# Leaf paths that must appear in provenance when the containing block is present.
PROVENANCED_PATHS = (
    "geometry.target_rel_gripper",
    "geometry.target_rel_receptacle",
    "geometry.gripper_opening_m",
    "relations.target_supported_by",
    "relations.target_in_receptacle",
    "relations.target_moving_with_gripper",
    "relations.gripper_above_target",
    "contact.gripper_target",
    "events_recent",
    "uncertainty",
)


def empty_state() -> dict[str, Any]:
    return {
        "goal": {"target_object_id": None, "receptacle_id": None, "instruction": None},
        "geometry": {},
        "relations": {},
        "contact": {},
        "events_recent": [],
        "uncertainty": {},
        "actions_available": list(ACTIONS),
        "provenance": {"source": {}, "t_obs": {}, "schema_version": SCHEMA_VERSION},
    }


def validate_state(state: dict[str, Any]) -> list[str]:
    """Return a list of schema violations. Empty list means valid."""
    problems: list[str] = []
    for key in ("goal", "actions_available", "provenance"):
        if key not in state:
            problems.append(f"missing required block: {key}")
    provenance = state.get("provenance", {})
    if provenance.get("schema_version") != SCHEMA_VERSION:
        problems.append("provenance.schema_version is not " + SCHEMA_VERSION)
    sources = provenance.get("source", {})
    times = provenance.get("t_obs", {})
    for path in PROVENANCED_PATHS:
        block, _, leaf = path.partition(".")
        present = block in state and (not leaf or leaf in (state.get(block) or {}))
        if not present:
            continue
        if path not in sources:
            problems.append(f"present field without provenance.source: {path}")
        elif sources[path] not in SOURCES:
            problems.append(f"unknown source for {path}: {sources[path]!r}")
        if path not in times:
            problems.append(f"present field without provenance.t_obs: {path}")
    for action in state.get("actions_available", []):
        if action not in ACTIONS:
            problems.append(f"unknown action offered: {action!r}")
    for event in state.get("events_recent", []) or []:
        if not isinstance(event, dict) or "t" not in event or "event" not in event:
            problems.append(f"malformed event entry: {event!r}")
        elif event["event"] not in EVENT_NAMES:
            problems.append(f"unknown event name: {event['event']!r}")
    return problems
