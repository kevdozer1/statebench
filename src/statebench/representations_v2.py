"""v0.2 canonical packets, each rendered twice (JSON and prose).

Three packets:

``R3``    the full schema (``R3_typed_history``'s state).
``R3nv``  R3 without ``relations.target_in_receptacle`` and
          ``relations.target_supported_by`` (``R3_no_verifier_fields``'s state).
``R4``    R3nv plus the predicate block (``R4_predicates``'s state).

Each packet is rendered as JSON (``R3j``, ``R3nvj``, ``R4j``), byte-identical to
the legacy renderers' text, and as prose (``R3p``, ``R3nvp``, ``R4p``). The prose
renderer takes the *parsed JSON packet* as its only input, so it cannot add or
lose information relative to the JSON by construction, and ``parse_prose``
recovers the packet exactly: every field at its stored precision (4 decimals in
metres and radians, 3 in seconds), with units, the full event list with
timestamps, the uncertainty block, every provenance source and observation time,
and every predicate by name and value. Relations are phrased in their stored
direction: ``target_rel_receptacle`` is the cube minus the receptacle floor
centre, and ``target_rel_gripper`` is the cube minus the tool centre point.

The legacy renderers in ``representations.py`` are untouched.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .observations import HistoryBuffer
from .representations import (
    ablate_history,
    apply_ablation,
    render_r3_typed_history,
    render_r4_predicates,
)

REPRESENTATIONS_VERSION = "statebench-representations/v0.2"


# ------------------------------------------------------------------ conditions
@dataclass(frozen=True)
class ConditionV2:
    name: str
    packet: str  # R3 | R3nv | R4
    rendering: str  # json | prose
    #: an extra state ablation on top of the packet's own (none this turn)
    ablation: str | None = None
    #: named-condition instruction replacing next_action ("c" = Turn 3b R4c).
    #: Prompt engineering, labelled as such everywhere it appears.
    prompt_variant: str | None = None

    @property
    def predicates(self) -> bool:
        return self.packet == "R4"

    @property
    def ablation_chain(self) -> tuple[str, ...]:
        chain = []
        if self.packet in ("R3nv", "R4"):
            chain.append("drop_verifier_fields")
        if self.ablation:
            chain.append(self.ablation)
        return tuple(chain)

    @property
    def representation(self) -> str:
        return self.packet + ("j" if self.rendering == "json" else "p")


CONDITIONS_V2: tuple[ConditionV2, ...] = (
    ConditionV2("R3j", "R3", "json"),
    ConditionV2("R3nvj", "R3nv", "json"),
    ConditionV2("R4j", "R4", "json"),
    ConditionV2("R3p", "R3", "prose"),
    ConditionV2("R3nvp", "R3nv", "prose"),
    ConditionV2("R4p", "R4", "prose"),
    ConditionV2("R4c", "R4", "json", prompt_variant="c"),
)
CONDITIONS_V2_BY_NAME = {c.name: c for c in CONDITIONS_V2}

#: v0.2 packets and the legacy conditions whose JSON they reproduce byte for byte.
LEGACY_JSON_EQUIVALENT = {
    "R3j": "R3_typed_history",
    "R3nvj": "R3_no_verifier_fields",
    "R4j": "R4_predicates",
    "R4c": "R4c_conditions_in_prompt_fixed",
}


# ----------------------------------------------------------------- JSON packets
def render_json(packet: str, state: dict[str, Any]) -> str:
    """The JSON rendering; ``state`` is already ablated for R3nv/R4."""
    if packet == "R4":
        return render_r4_predicates(state)
    return render_r3_typed_history(state)


def packet_of(packet: str, state: dict[str, Any]) -> dict[str, Any]:
    """The canonical packet: exactly what the JSON rendering carries."""
    return json.loads(render_json(packet, state))


# ------------------------------------------------------------------ prose render
def _m(v: Any) -> str:
    return f"{float(v):.4f}"


def _s(v: Any) -> str:
    return f"{float(v):.3f}"


def _val(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    raise ValueError(f"cannot render value {v!r}")


def _prov(packet: dict[str, Any], path: str) -> str:
    prov = packet.get("provenance") or {}
    src = (prov.get("source") or {}).get(path)
    t = (prov.get("t_obs") or {}).get(path)
    if src is None and t is None:
        return ""
    if src is None or t is None:
        raise ValueError(f"half-present provenance for {path}")
    return f" (source {src}, observed at t = {_s(t)} s)"


_UNCERTAINTY_KEYS = ("target_visible", "estimate_age_s", "contact_observed")
_RELATION_PHRASES = {
    "target_supported_by": "The {target} is supported by: {value}",
    "target_in_receptacle": "The {target} is inside the {receptacle}: {value}",
    "target_moving_with_gripper": "The {target} is moving with the gripper: {value}",
    "gripper_above_target": "The gripper is directly above the {target}: {value}",
}


def render_prose(packet: dict[str, Any]) -> str:
    """Prose for a canonical packet dict. Every field, same precision, stored direction."""
    known = {"goal", "geometry", "relations", "contact", "events_recent", "uncertainty",
             "actions_available", "provenance", "predicates"}
    unknown = set(packet) - known
    if unknown:
        raise ValueError(f"packet carries blocks the prose renderer does not know: {unknown}")
    goal = packet.get("goal") or {}
    target = goal.get("target_object_id") or "target"
    receptacle = goal.get("receptacle_id") or "receptacle"
    lines: list[str] = []
    lines.append(
        f"Goal: the target object is {goal.get('target_object_id')}, the receptacle is "
        f"{goal.get('receptacle_id')}, and the instruction is \"{goal.get('instruction')}\"."
    )

    geometry = packet.get("geometry")
    if geometry is not None:
        lines.append("Geometry:" if geometry else "Geometry: none recorded.")
        rel = geometry.get("target_rel_gripper")
        if rel is not None:
            yaw = (f", and the {target}'s yaw minus the gripper's yaw is "
                   f"{_m(rel['dyaw_rad'])} rad" if "dyaw_rad" in rel else "")
            lines.append(
                f"The {target}'s centre minus the gripper's tool centre point is "
                f"dx {_m(rel['dx'])} m, dy {_m(rel['dy'])} m, dz {_m(rel['dz'])} m{yaw} "
                f"[frame {rel['frame']}, units {rel['units']}]"
                f"{_prov(packet, 'geometry.target_rel_gripper')}."
            )
        rel_r = geometry.get("target_rel_receptacle")
        if rel_r is not None:
            lines.append(
                f"The {target}'s centre minus the {receptacle}'s floor centre is "
                f"dx {_m(rel_r['dx'])} m, dy {_m(rel_r['dy'])} m, dz {_m(rel_r['dz'])} m "
                f"[frame {rel_r['frame']}, units {rel_r['units']}]"
                f"{_prov(packet, 'geometry.target_rel_receptacle')}."
            )
        if "gripper_opening_m" in geometry:
            lines.append(
                f"The gripper opening is {_m(geometry['gripper_opening_m'])} m"
                f"{_prov(packet, 'geometry.gripper_opening_m')}."
            )
        extra = set(geometry) - {"target_rel_gripper", "target_rel_receptacle", "gripper_opening_m"}
        if extra:
            raise ValueError(f"unknown geometry fields {extra}")

    relations = packet.get("relations")
    if relations is not None:
        lines.append("Relations:" if relations else "Relations: none recorded.")
        for key, phrase in _RELATION_PHRASES.items():
            if key in relations:
                lines.append(
                    phrase.format(target=target, receptacle=receptacle,
                                  value=_val(relations[key]))
                    + _prov(packet, f"relations.{key}") + "."
                )
        extra = set(relations) - set(_RELATION_PHRASES)
        if extra:
            raise ValueError(f"unknown relations {extra}")

    contact = packet.get("contact")
    if contact is not None:
        if not contact:
            lines.append("Contact: none recorded.")
        else:
            lines.append(
                f"Contact: gripper-to-{target} contact is {contact['gripper_target']}, with "
                f"evidence \"{contact['evidence']}\""
                f"{_prov(packet, 'contact.gripper_target')}."
            )

    events = packet.get("events_recent")
    if events is not None:
        lines.append(
            f"Recent events, oldest first, {len(events)} in total"
            f"{_prov(packet, 'events_recent')}:"
        )
        for event in events:
            extras = "".join(
                f" ({k}: {event[k]})" for k in sorted(event) if k not in ("t", "event")
            )
            lines.append(f"At t = {_s(event['t'])} s: {event['event']}{extras}.")

    uncertainty = packet.get("uncertainty")
    if uncertainty is not None:
        extra = set(uncertainty) - set(_UNCERTAINTY_KEYS)
        if extra:
            raise ValueError(f"unknown uncertainty fields {extra}")
        parts = []
        if "target_visible" in uncertainty:
            parts.append(f"target visible is {_val(uncertainty['target_visible'])}")
        if "estimate_age_s" in uncertainty:
            parts.append(f"estimate age is {_s(uncertainty['estimate_age_s'])} s")
        if "contact_observed" in uncertainty:
            parts.append(f"contact observed is {_val(uncertainty['contact_observed'])}")
        lines.append("Uncertainty: " + (", ".join(parts) or "none recorded")
                     + _prov(packet, "uncertainty") + ".")

    predicates = packet.get("predicates")
    if predicates is not None:
        names = sorted(k for k in predicates if k != "source")
        lines.append(
            f"Derived predicates, block source {predicates.get('source')}"
            f"{_prov(packet, 'predicates')}: "
            + "; ".join(f"{n} is {_val(predicates[n])}" for n in names) + "."
        )

    lines.append("Available actions: " + ", ".join(packet.get("actions_available", [])) + ".")
    prov = packet.get("provenance") or {}
    lines.append(f"Schema version: {prov.get('schema_version')}.")
    return "\n".join(lines)


# ------------------------------------------------------------------ prose parse
_NUM = r"(-?\d+\.\d+)"
_PROV = r"(?: \(source (\S+), observed at t = " + _NUM + r" s\))?"


def _v(text: str) -> Any:
    if text == "true":
        return True
    if text == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def parse_prose(text: str) -> dict[str, Any]:
    """Recover the canonical packet from ``render_prose`` output. Exact or it raises."""
    out: dict[str, Any] = {}
    source: dict[str, str] = {}
    t_obs: dict[str, float] = {}

    def prov(path: str, src: str | None, t: str | None) -> None:
        if src is not None:
            source[path] = src
            t_obs[path] = float(t)

    lines = text.split("\n")
    i = 0
    in_events = False
    while i < len(lines):
        line = lines[i]
        i += 1
        m = re.fullmatch(r'Goal: the target object is (\S+), the receptacle is (\S+), '
                         r'and the instruction is "(.*)"\.', line)
        if m:
            out["goal"] = {"target_object_id": m.group(1), "receptacle_id": m.group(2),
                           "instruction": m.group(3)}
            continue
        if line in ("Geometry:", "Geometry: none recorded."):
            out["geometry"] = {}
            in_events = False
            continue
        m = re.fullmatch(
            r"The (\S+)'s centre minus the gripper's tool centre point is "
            rf"dx {_NUM} m, dy {_NUM} m, dz {_NUM} m"
            rf"(?:, and the \S+'s yaw minus the gripper's yaw is {_NUM} rad)? "
            r"\[frame (\S+), units (\S+)\]" + _PROV + r"\.", line)
        if m:
            block = {"dx": float(m.group(2)), "dy": float(m.group(3)), "dz": float(m.group(4))}
            if m.group(5) is not None:
                block["dyaw_rad"] = float(m.group(5))
            block["frame"], block["units"] = m.group(6), m.group(7)
            out.setdefault("geometry", {})["target_rel_gripper"] = block
            prov("geometry.target_rel_gripper", m.group(8), m.group(9))
            continue
        m = re.fullmatch(
            r"The (\S+)'s centre minus the (\S+)'s floor centre is "
            rf"dx {_NUM} m, dy {_NUM} m, dz {_NUM} m "
            r"\[frame (\S+), units (\S+)\]" + _PROV + r"\.", line)
        if m:
            out.setdefault("geometry", {})["target_rel_receptacle"] = {
                "dx": float(m.group(3)), "dy": float(m.group(4)), "dz": float(m.group(5)),
                "frame": m.group(6), "units": m.group(7)}
            prov("geometry.target_rel_receptacle", m.group(8), m.group(9))
            continue
        m = re.fullmatch(rf"The gripper opening is {_NUM} m" + _PROV + r"\.", line)
        if m:
            out.setdefault("geometry", {})["gripper_opening_m"] = float(m.group(1))
            prov("geometry.gripper_opening_m", m.group(2), m.group(3))
            continue
        if line in ("Relations:", "Relations: none recorded."):
            out["relations"] = {}
            continue
        matched = False
        for key, phrase in _RELATION_PHRASES.items():
            pattern = (re.escape(phrase).replace(r"\{target\}", r"\S+")
                       .replace(r"\{receptacle\}", r"\S+").replace(r"\{value\}", r"(\S+?)"))
            m = re.fullmatch(pattern + _PROV + r"\.", line)
            if m:
                out.setdefault("relations", {})[key] = _v(m.group(1))
                prov(f"relations.{key}", m.group(2), m.group(3))
                matched = True
                break
        if matched:
            continue
        if line == "Contact: none recorded.":
            out["contact"] = {}
            continue
        m = re.fullmatch(r'Contact: gripper-to-\S+ contact is (\S+), with evidence "(.*)"'
                         + _PROV + r"\.", line)
        if m:
            out["contact"] = {"gripper_target": m.group(1), "evidence": m.group(2)}
            prov("contact.gripper_target", m.group(3), m.group(4))
            continue
        m = re.fullmatch(r"Recent events, oldest first, (\d+) in total" + _PROV + ":", line)
        if m:
            n = int(m.group(1))
            prov("events_recent", m.group(2), m.group(3))
            events = []
            for _ in range(n):
                em = re.fullmatch(rf"At t = {_NUM} s: (\S+?)((?: \(\S+: [^)]*\))*)\.", lines[i])
                if not em:
                    raise ValueError(f"bad event line: {lines[i]!r}")
                i += 1
                event: dict[str, Any] = {"event": em.group(2), "t": float(em.group(1))}
                for k, v in re.findall(r" \((\S+): ([^)]*)\)", em.group(3)):
                    event[k] = v
                events.append(event)
            out["events_recent"] = events
            continue
        m = re.fullmatch(r"Uncertainty: (.*?)" + _PROV + r"\.", line)
        if m:
            block: dict[str, Any] = {}
            body = m.group(1)
            if body != "none recorded":
                for part in body.split(", "):
                    pm = re.fullmatch(rf"estimate age is {_NUM} s", part)
                    if pm:
                        block["estimate_age_s"] = float(pm.group(1))
                        continue
                    pm = re.fullmatch(r"(target visible|contact observed) is (\S+)", part)
                    if not pm:
                        raise ValueError(f"bad uncertainty part {part!r}")
                    block[pm.group(1).replace(" ", "_")] = _v(pm.group(2))
            out["uncertainty"] = block
            prov("uncertainty", m.group(2), m.group(3))
            continue
        m = re.fullmatch(r"Derived predicates, block source (\S+?)" + _PROV + r": (.*)\.", line)
        if m:
            block = {"source": m.group(1)}
            for part in m.group(4).split("; "):
                pm = re.fullmatch(r"(\S+) is (\S+)", part)
                if not pm:
                    raise ValueError(f"bad predicate part {part!r}")
                block[pm.group(1)] = _v(pm.group(2))
            out["predicates"] = block
            prov("predicates", m.group(2), m.group(3))
            continue
        m = re.fullmatch(r"Available actions: (.*)\.", line)
        if m:
            out["actions_available"] = [a for a in m.group(1).split(", ") if a]
            continue
        m = re.fullmatch(r"Schema version: (\S+)\.", line)
        if m:
            out["provenance"] = {"source": source, "t_obs": t_obs,
                                 "schema_version": m.group(1)}
            continue
        raise ValueError(f"unparsed prose line: {line!r}")
    _ = in_events
    return out


def render(condition: ConditionV2, state: dict[str, Any]) -> str:
    text = render_json(condition.packet, state)
    if condition.rendering == "json":
        return text
    return render_prose(json.loads(text))


# ------------------------------------------------------------------- prepare
def prepare_v2(
    condition: ConditionV2,
    compact: dict[str, Any],
    history: HistoryBuffer,
    t: float,
    rng: np.random.Generator,
    proprio: dict[str, Any] | None = None,
    read_audit: list | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """(state the executor may use, history it may use, text for the model).

    Same ablation mechanics as ``representations.prepare``. When ``read_audit`` is
    a list, the predicate build runs under the access-recording check and appends
    ``(paths_read, undeclared_paths)`` to it.
    """
    state = compact
    chain = condition.ablation_chain
    for ablation in chain:
        state = apply_ablation(state, ablation, t=t, history=history, rng=rng)
    if not chain:
        state = apply_ablation(state, None, t=t, history=history, rng=rng)

    first = chain[0] if chain else None
    hist = ablate_history(history, first, t=t, rng=rng)
    for extra in chain[1:]:
        hist = [apply_ablation(s, extra, t=t, history=history, rng=rng) for s in hist]

    if condition.predicates:
        from .predicates import build_predicates

        if read_audit is not None:
            from .provenance_audit import audited_build_predicates

            block, read, undeclared = audited_build_predicates(state, proprio, hist)
            read_audit.append((sorted(read), sorted(undeclared)))
        else:
            block = build_predicates(state, proprio, hist)
        state["predicates"] = block
        state.setdefault("provenance", {}).setdefault("source", {})["predicates"] = "derived"
        state["provenance"].setdefault("t_obs", {})["predicates"] = round(float(t), 3)
    return state, hist, render(condition, state)


def deepcopy_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)
