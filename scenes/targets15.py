"""Turn 15 drawing targets: our own letters (the word ABC, from Turn 10's ``targets``; single-stroke shapes are left out so the task stays long) plus outlines traced from 3 public-domain
paintings (``trace_paintings.py``). Seeds choose the target uniformly from TARGETS (``default_rng(1_950_015 + seed)``),
and its size and position (``default_rng(1_900_015 + seed)``) inside the drawing area the Turn 10 targets use.
Paintings are placed with their longer side 0.075 m (the traced scale), centre x 0.44-0.46 m, y -0.01 to 0.01 m.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import targets as T10  # noqa: E402
import paths  # noqa: E402

PAINT_DIR = paths.ASSETS / "paintings"
PAINTINGS = ("starry_night", "great_wave", "mona_lisa")
TARGETS = PAINTINGS + ("ABC",)


def make_target(seed: int) -> dict:
    name = TARGETS[int(np.random.default_rng(1_950_015 + seed).integers(len(TARGETS)))]
    if name in PAINTINGS:
        t = json.loads((PAINT_DIR / f"{name}_trace.json").read_text())
        rng = np.random.default_rng(1_900_015 + seed)
        size = t["method"]["scale_m"]
        cx, cy = float(rng.uniform(0.44, 0.46)), float(rng.uniform(-0.01, 0.01))
        strokes = [[[round(cx + size * (y - 0.5), 6), round(cy + size * (0.5 - x), 6)] for x, y in s]
                   for s in t["strokes_unit"]]
        return {"name": name, "seed": seed, "size_m": size, "strokes": strokes}
    return T10.make_named(name, seed)
