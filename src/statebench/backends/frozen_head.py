"""Frozen closed-form head: the floor to read the decision models against.

Three independent scikit-learn ``LogisticRegression`` heads over the structured
fields of R4. No text embedding, no neural network, no tuning, no network call at
decision time. Fitted once on Turn 1 and Turn 2 logs from seeds 0-199 and then
frozen; evaluated closed-loop on seeds 200-299 with the same executor and the
same verifier as every other backend.

The point is not that logistic regression is a good controller. It is to say how
far a two-minute fit on already-structured state gets, so that a frontier
decision model's score on the same held-out seeds can be read against something
rather than against nothing.

Labels:

* ``next_action``   - the primitive the ``rules`` backend took at that state
  (imitation of the reference on the development seeds).
* ``object_held``   - Turn 1's operational label (the cube's displacement matched
  the TCP's over the next 0.5 s, TCP moving at least 10 mm).
* ``step_complete`` - the verifier's final outcome for that step's episode.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

from ..config import run_dir
from ..questions import NEXT_ACTION, OBJECT_HELD, STEP_COMPLETE
from ..schema import ACTIONS, EVENT_NAMES
from .base import Answer, Decision, DecisionBackend, answer_from, register_backend

MODEL_FILENAME = "frozen_head.pkl"

_TRISTATE = ("true", "false", "unknown")
_SUPPORTED_BY = ("table", "receptacle", "gripper", "none", "absent")
_CONTACT = ("contact", "no_contact", "unknown")

_PREDICATE_TRISTATE = (
    "aligned_for_grasp",
    "gripper_closed_on_object",
    "gripper_closed_empty",
    "object_moving_with_gripper",
    "object_at_receptacle",
    "released_at_receptacle",
    "estimate_stale",
)


def _tri(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _onehot(value: str, options: tuple[str, ...]) -> list[float]:
    return [1.0 if value == o else 0.0 for o in options]


def feature_names() -> list[str]:
    names: list[str] = []
    for axis in ("dx", "dy", "dz", "dyaw_rad"):
        names.append(f"rel_gripper.{axis}")
    names.append("rel_gripper.present")
    for axis in ("dx", "dy", "dz"):
        names.append(f"rel_receptacle.{axis}")
    names.append("rel_receptacle.present")
    names += ["gripper_opening_m", "estimate_age_s", "target_visible", "contact_observed"]
    for base, options in (
        ("moving_with_gripper", _TRISTATE),
        ("gripper_above_target", _TRISTATE),
        ("target_in_receptacle", _TRISTATE),
    ):
        names += [f"rel.{base}={o}" for o in options]
    names += [f"rel.supported_by={o}" for o in _SUPPORTED_BY]
    names += [f"contact.gripper_target={o}" for o in _CONTACT]
    for predicate in _PREDICATE_TRISTATE:
        names += [f"pred.{predicate}={o}" for o in _TRISTATE]
    names.append("pred.close_attempts_since_last_lift")
    names += [f"event_count.{e}" for e in EVENT_NAMES]
    return names


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)


def featurize(state: dict[str, Any], proprio: dict[str, Any] | None = None) -> list[float]:
    """Fixed-length numeric vector from an R3/R4 state dict.

    Tolerates missing blocks: an absent field contributes its "absent"/"unknown"
    encoding rather than raising, so the same extractor works on every condition.
    """
    geometry = state.get("geometry") or {}
    relations = state.get("relations") or {}
    contact = state.get("contact") or {}
    uncertainty = state.get("uncertainty") or {}
    predicates = (state.get("predicates") or {})
    values = predicates.get("values", predicates) if isinstance(predicates, dict) else {}

    out: list[float] = []
    rel_g = geometry.get("target_rel_gripper")
    if isinstance(rel_g, dict):
        for axis in ("dx", "dy", "dz", "dyaw_rad"):
            try:
                out.append(float(rel_g.get(axis, 0.0)))
            except (TypeError, ValueError):
                out.append(0.0)
        out.append(1.0)
    else:
        out += [0.0, 0.0, 0.0, 0.0, 0.0]

    rel_r = geometry.get("target_rel_receptacle")
    if isinstance(rel_r, dict):
        for axis in ("dx", "dy", "dz"):
            try:
                out.append(float(rel_r.get(axis, 0.0)))
            except (TypeError, ValueError):
                out.append(0.0)
        out.append(1.0)
    else:
        out += [0.0, 0.0, 0.0, 0.0]

    opening = geometry.get("gripper_opening_m")
    if opening is None and proprio:
        opening = proprio.get("gripper_opening_m")
    out.append(float(opening) if opening is not None else 0.0)
    age = uncertainty.get("estimate_age_s")
    out.append(float(age) if age is not None else 0.0)
    out.append(1.0 if uncertainty.get("target_visible") else 0.0)
    out.append(1.0 if uncertainty.get("contact_observed") else 0.0)

    out += _onehot(_tri(relations.get("target_moving_with_gripper")), _TRISTATE)
    out += _onehot(_tri(relations.get("gripper_above_target")), _TRISTATE)
    out += _onehot(_tri(relations.get("target_in_receptacle")), _TRISTATE)
    supported = relations.get("target_supported_by")
    out += _onehot(supported if supported in _SUPPORTED_BY else "absent", _SUPPORTED_BY)
    out += _onehot(contact.get("gripper_target", "unknown"), _CONTACT)

    for predicate in _PREDICATE_TRISTATE:
        out += _onehot(_tri(values.get(predicate)), _TRISTATE)
    try:
        out.append(float(values.get("close_attempts_since_last_lift", 0) or 0))
    except (TypeError, ValueError):
        out.append(0.0)

    events = state.get("events_recent") or []
    counts = {name: 0.0 for name in EVENT_NAMES}
    for event in events:
        name = event.get("event") if isinstance(event, dict) else None
        if name in counts:
            counts[name] += 1.0
    out += [counts[name] for name in EVENT_NAMES]
    assert len(out) == N_FEATURES, f"{len(out)} != {N_FEATURES}"
    return out


# --------------------------------------------------------------------- model
class FrozenHeadBackend(DecisionBackend):
    """Loads the fitted bundle and answers from it. No network, no text model."""

    name = "frozen_head"
    remote = False

    def __init__(self, model: str | None = None, bundle_path: Path | None = None):
        super().__init__(model=model or "logreg-l2-v1")
        self.bundle_path = bundle_path or (run_dir("turn3") / MODEL_FILENAME)
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"frozen head not fitted yet: {self.bundle_path}. "
                "Run `python -m statebench.turn3 fit-head` first."
            )
        with self.bundle_path.open("rb") as handle:
            self.bundle = pickle.load(handle)
        self.latencies: list[float] = []

    def _distribution(self, head: str, x: list[float], options: list[str]) -> Answer:
        model = self.bundle["models"].get(head)
        if model is None:
            raise KeyError(head)
        import numpy as np

        probs = model.predict_proba(np.asarray([x], dtype=float))[0]
        classes = [str(c) for c in model.classes_]
        mapping = {c: float(p) for c, p in zip(classes, probs, strict=True)}
        # Options the head never saw in training get zero mass.
        full = {o: mapping.get(o, 0.0) for o in options}
        total = sum(full.values())
        if total <= 0:
            full = {o: 1.0 / len(options) for o in options}
        else:
            full = {o: v / total for o, v in full.items()}
        return answer_from(full, options)

    def decide(
        self,
        state_text: str,
        questions: dict[str, list[str]],
        instruction_override: str | None = None,
    ) -> Decision:
        started = time.perf_counter()
        error = None
        answers: dict[str, Answer] = {}
        try:
            state = json.loads(state_text)
            x = featurize(state)
            for name, options in questions.items():
                answers[name] = self._distribution(name, x, options)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        self.latencies.append(latency)
        return self.account(
            Decision(
                answers=answers,
                raw={"n_features": N_FEATURES},
                backend=self.name,
                model=self.model,
                latency_seconds=latency,
                error=error,
                extra={"fit_seconds": self.bundle.get("fit_seconds"),
                       "trained_on": self.bundle.get("trained_on")},
            )
        )

    def usage(self) -> dict[str, Any]:
        base = super().usage()
        base.update(latencies=self.latencies, n_features=N_FEATURES,
                    fit_seconds=self.bundle.get("fit_seconds"))
        return base


def _iter_turn1_rules_steps():
    """(state_dict, primitive, held_label, episode_success) from Turn 1 rule logs."""
    root = run_dir("turn1") / "episodes"
    for directory in sorted(root.glob("rules__*")):
        for path in sorted(directory.glob("*.json")):
            episode = json.loads(path.read_text(encoding="utf-8"))
            if int(episode["seed"]) > 199:
                continue
            held_by_t = {
                round(float(row["t"]), 3): row.get("label")
                for row in episode.get("calibration", [])
            }
            for step in episode["steps"]:
                text = step["state_text"]
                if not text.lstrip().startswith("{"):
                    continue  # R1 captions are prose; the head reads structure only
                try:
                    state = json.loads(text)
                except json.JSONDecodeError:
                    continue
                yield (
                    state,
                    step["execution"]["primitive"],
                    held_by_t.get(round(float(step["t"]), 3)),
                    bool(episode["success"]),
                )


def _iter_turn2_steps():
    """Same, from Turn 2 packed logs (any backend), seeds 0-199 only.

    Turn 2 supplies extra state coverage - states Jev drove into that the rules
    backend never visited - but the next_action label still comes from what the
    rules backend would do, so it is never imitating a model.
    """
    root = run_dir("turn2") / "episodes"
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload.get("episodes", []):
            if int(episode["seed"]) > 199:
                continue
            held_by_t = {
                round(float(row["t"]), 3): row.get("label")
                for row in episode.get("calibration", [])
            }
            for step in episode["steps"]:
                text = step["state_text"]
                if not text.lstrip().startswith("{"):
                    continue
                try:
                    state = json.loads(text)
                except json.JSONDecodeError:
                    continue
                yield state, None, held_by_t.get(round(float(step["t"]), 3)), bool(episode["success"])


def fit(out_path: Path | None = None) -> dict[str, Any]:
    """Fit the three heads and freeze them. Returns a report."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from ..backends.rules import choose_action, parse_typed
    from ..predicates import build_predicates

    out_path = out_path or (run_dir("turn3") / MODEL_FILENAME)
    rows_x: list[list[float]] = []
    rows_action: list[str] = []
    rows_held_x: list[list[float]] = []
    rows_held_y: list[int] = []
    rows_done_x: list[list[float]] = []
    rows_done_y: list[int] = []

    def add(state: dict[str, Any], primitive: str | None, held: Any, success: bool) -> None:
        # Recompute predicates offline so the training features match R4 exactly.
        if "predicates" not in state:
            state = dict(state)
            state["predicates"] = build_predicates(state, None, None)
        x = featurize(state)
        # next_action label is always what the RULES backend would choose from
        # this state, never what a model actually chose.
        label = primitive
        if label is None:
            try:
                label, _ = choose_action(parse_typed(json.dumps(state)))
            except Exception:  # noqa: BLE001
                return
        rows_x.append(x)
        rows_action.append(label)
        if held in (True, False):
            rows_held_x.append(x)
            rows_held_y.append(1 if held else 0)
        rows_done_x.append(x)
        rows_done_y.append(1 if success else 0)

    for state, primitive, held, success in _iter_turn1_rules_steps():
        add(state, primitive, held, success)
    for state, primitive, held, success in _iter_turn2_steps():
        add(state, primitive, held, success)

    started = time.perf_counter()
    models: dict[str, Any] = {}
    models[NEXT_ACTION] = LogisticRegression(max_iter=2000).fit(
        np.asarray(rows_x, dtype=float), np.asarray(rows_action)
    )
    if len(set(rows_held_y)) > 1:
        models[OBJECT_HELD] = LogisticRegression(max_iter=2000).fit(
            np.asarray(rows_held_x, dtype=float),
            np.asarray(["yes" if v else "no" for v in rows_held_y]),
        )
    if len(set(rows_done_y)) > 1:
        models[STEP_COMPLETE] = LogisticRegression(max_iter=2000).fit(
            np.asarray(rows_done_x, dtype=float),
            np.asarray(["yes" if v else "no" for v in rows_done_y]),
        )
    fit_seconds = time.perf_counter() - started

    bundle = {
        "models": models,
        "feature_names": FEATURE_NAMES,
        "fit_seconds": fit_seconds,
        "trained_on": "turn1+turn2 logs, seeds 0-199",
        "n_rows_next_action": len(rows_x),
        "n_rows_object_held": len(rows_held_x),
        "n_rows_step_complete": len(rows_done_x),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return {
        "path": str(out_path),
        "n_features": N_FEATURES,
        "fit_seconds": round(fit_seconds, 3),
        "n_rows_next_action": len(rows_x),
        "n_rows_object_held": len(rows_held_x),
        "n_rows_step_complete": len(rows_done_x),
        "action_label_counts": {
            a: rows_action.count(a) for a in sorted(set(rows_action))
        },
        "heads": sorted(models),
    }


register_backend("frozen_head", FrozenHeadBackend)
