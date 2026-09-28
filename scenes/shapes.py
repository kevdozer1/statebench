"""Turn 9 Phase 1: simple shapes in the llm-robotics-playground writing and drawing scenes (feasibility only).

Runs inside the playground's own venv (never the statebench venv), from the playground
checkout: ``<playground python> shapes.py <scene> <shape> <seed> [--inject flicker]``.
Uses only our own shapes (circle, square, A, B, C); nothing artwork-derived.

The controllers are imported unchanged. The only override is the optional injection
test: a subclass that replaces the method the controller reads contact through
(writing: the force ``Writer.step`` returns and stores; drawing:
``PhysicalWriter.contact_force``), leaving the controller's logic alone.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import paths  # noqa: E402

PLAYGROUND = paths.PLAYGROUND
SHAPES = ("circle", "square", "A", "B", "C")


def unit_strokes(name: str) -> list[list[tuple[float, float]]]:
    """Shapes in a unit box (x right, y up, 0-1)."""
    if name == "circle":
        return [[(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)]]
    if name == "square":
        return [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
    if name == "A":
        return [[(0, 0), (0.5, 1), (1, 0)], [(0.22, 0.42), (0.78, 0.42)]]
    if name == "B":
        return [[(0.1, 0), (0.1, 1), (0.65, 1), (0.85, 0.87), (0.85, 0.63), (0.65, 0.52), (0.1, 0.52),
                 (0.7, 0.52), (0.92, 0.38), (0.92, 0.14), (0.7, 0), (0.1, 0)]]
    if name == "C":
        return [[(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
                 for a in np.linspace(math.radians(45), math.radians(315), 19)]]
    raise ValueError(name)


def placed(scene: str, name: str, seed: int) -> tuple[list[np.ndarray], dict]:
    """Size and position drawn from the seed, inside the area the scene's own demo uses."""
    rng = np.random.default_rng(1_900_000 + seed)
    size = float(rng.uniform(0.03, 0.06))
    if scene == "writing":  # whiteboard, board coordinates (y, z); y decreases to the right
        cy, cz = float(rng.uniform(-0.30, -0.08)), float(rng.uniform(0.98, 1.09))
        strokes = [np.array([(cy + size * (0.5 - x), cz + size * (y - 0.5)) for x, y in s]) for s in unit_strokes(name)]
    else:  # paper, (x, y)
        cx, cy = float(rng.uniform(0.40, 0.50)), float(rng.uniform(-0.05, 0.05))
        strokes = [np.array([(cx + size * (y - 0.5), cy + size * (0.5 - x)) for x, y in s]) for s in unit_strokes(name)]
    return strokes, {"size_m": round(size, 4), "centre": [round(float(v), 4) for v in strokes[0].mean(axis=0)]}


def run_writing(name: str, seed: int, inject: str | None = None) -> dict:
    os.chdir(PLAYGROUND / "experiments" / "fibonacci-writing")
    sys.path.insert(0, os.getcwd())
    from writer import Writer

    class Injected(Writer):
        """Replaces only the sensed tip force the controller reads (self.force, filtered_force)."""
        def step(self):
            before = self.filtered_force
            true_force = super().step()
            if inject == "flicker" and np.random.random() < 0.1:
                self.force = 0.0
                self.filtered_force = 0.95 * before + 0.05 * self.force
            return self.force

    strokes, meta = placed("writing", name, seed)
    w = (Injected if inject else Writer)()
    t0 = time.monotonic()
    w.goto([0.410, *strokes[0][0]], 2)
    for s in strokes:
        w.draw(s, 0.018)
    a = np.array(w.track)
    err = np.linalg.norm(a[:, 1:3] - a[:, 4:6], axis=1) * 1000 if len(a) else np.array([np.nan])
    return {"scene": "writing", "shape": name, "seed": seed, **meta, "inject": inject,
            "wall_s": round(time.monotonic() - t0, 1), "sim_s": round(float(w.d.time), 2), "marks": len(w.marks),
            "median_tracking_error_mm": round(float(np.median(err)), 3), "strokes": len(strokes)}


def run_drawing(name: str, seed: int, inject: str | None = None) -> dict:
    os.chdir(PLAYGROUND / "experiments" / "dove-drawing")
    sys.path.insert(0, os.getcwd())
    from controller import Controller
    from run import RecordingPhysical
    from writer import PhysicalWriter

    class Injected(PhysicalWriter):
        """Replaces only the contact reading the controller uses; ink is still recorded from true contact."""
        def contact_force(self):
            f = super().contact_force()
            return 0.0 if (inject == "flicker" and np.random.random() < 0.1) else f

    strokes, meta = placed("drawing", name, seed)
    f = RecordingPhysical()
    c = Controller(f)
    w = (Injected if inject else PhysicalWriter)(c)
    t0 = time.monotonic()
    c.command({}, 1)
    w.goto([*strokes[0][0], 0.794], 2.5)
    for s in strokes:
        w.draw(s, 0.008)
    a = np.array(w.track)
    err = np.linalg.norm(a[:, :2] - a[:, 2:4], axis=1) * 1000 if len(a) else np.array([np.nan])
    return {"scene": "drawing", "shape": name, "seed": seed, **meta, "inject": inject,
            "wall_s": round(time.monotonic() - t0, 1), "sim_s": round(float(f.data.time), 2), "marks": len(f.marks),
            "median_tracking_error_mm": round(float(np.median(err)), 3), "strokes": len(strokes)}


if __name__ == "__main__":
    scene, shape, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    inject = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--inject" else None
    np.random.seed(seed)
    try:
        out = (run_writing if scene == "writing" else run_drawing)(shape, seed, inject)
    except Exception as ex:  # report, do not hide
        out = {"scene": scene, "shape": shape, "seed": seed, "error": f"{type(ex).__name__}: {ex}"}
    print(json.dumps(out), flush=True)
