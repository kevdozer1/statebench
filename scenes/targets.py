"""Turn 10 pen targets: every target is a plain list of polylines in paper coordinates (metres, paper x/y), stored
as JSON ``{"name": ..., "strokes": [[[x, y], ...], ...]}`` and loaded by ``load_target``, so an outline traced from
an image can be dropped in later without changing anything else. Shapes are our own (circle, square, A, B, C, and
the word ABC); nothing artwork-derived. Seeds choose the target (``default_rng(1_950_010 + seed)``), its size and its
position (``default_rng(1_900_010 + seed)``) inside the drawing area the playground's demo uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from shapes import unit_strokes  # noqa: E402
import paths  # noqa: E402

TARGETS = ("circle", "square", "A", "B", "C", "ABC")


def _place(unit: list, size: float, cx: float, cy: float) -> list[list[list[float]]]:
    # shape-up -> paper +x, shape-right -> paper -y (as Turn 9's shapes.placed)
    return [[[round(cx + size * (y - 0.5), 6), round(cy + size * (0.5 - x), 6)] for x, y in s] for s in unit]


def make_target(seed: int) -> dict:
    name = TARGETS[int(np.random.default_rng(1_950_010 + seed).integers(len(TARGETS)))]
    return make_named(name, seed)


def make_named(name: str, seed: int) -> dict:
    rng = np.random.default_rng(1_900_010 + seed)
    if name == "ABC":
        size = float(rng.uniform(0.025, 0.032))
        cx, cy = float(rng.uniform(0.42, 0.48)), float(rng.uniform(-0.01, 0.01))
        gap = size * 1.25
        strokes = []
        for i, letter in enumerate("ABC"):
            strokes += _place(unit_strokes(letter), size, cx, cy - (i - 1) * gap)
    else:
        size = float(rng.uniform(0.03, 0.06))
        cx, cy = float(rng.uniform(0.40, 0.50)), float(rng.uniform(-0.05, 0.05))
        strokes = _place(unit_strokes(name), size, cx, cy)
    return {"name": name, "seed": seed, "size_m": round(size, 4), "strokes": strokes}


def target_path(seed: int, name: str | None = None) -> Path:
    d = paths.DATA / "runs" / "turn10" / "pen_targets"
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"{seed}.json" if name is None else f"{seed}_{name}.json")


def write_target(seed: int, name: str | None = None) -> Path:
    p = target_path(seed, name)
    if not p.exists():
        p.write_text(json.dumps(make_target(seed) if name is None else make_named(name, seed)))
    return p


def load_target(path: Path) -> tuple[str, list[np.ndarray]]:
    d = json.loads(Path(path).read_text())
    return d["name"], [np.asarray(s, dtype=float) for s in d["strokes"]]
