"""Turn 9 Phase 4: the drawing scene built the statebench way (exploratory; no confirmation this turn).

Runs inside the playground's own venv (never the statebench venv):
``<playground python> pen_bench.py grid <out.jsonl> [--processes N]`` or
``<playground python> pen_bench.py one <seed> <source> [<level>] [--keep <dir>]``.

* Seeds randomize the shape (circle, square, A, B, C; ``default_rng(1_950_000 + seed)``) and its size
  and position (``shapes.placed``, ``default_rng(1_900_000 + seed)``). Only our own shapes.
* The controller (``controller.Controller``, ``writer.PhysicalWriter``) is imported unchanged. Its one
  read of the world during a stroke is ``PhysicalWriter.contact_force()`` (every 20 ms): the pen-touching
  label is injected there and nowhere else. Ink is still deposited by the unchanged firmware from true
  contact, so the verifier reads what really happened.
* Meaning labels for the pen, logged from truth at every read: touching (true tip-paper normal force above
  0.015 N, the firmware's own ink threshold), pressing (force within PRESS_BAND), stroke done (coverage of
  the current stroke's path reaches COVERAGE_MIN), shape done (every stroke done).
* Delivered touching label -> the reading the controller gets: the true force when the label is right;
  0 N when it says "not touching" while the pen touches; FORCE_TARGET (0.15 N, the controller's own
  target, i.e. "touching, no correction") when it says "touching" while the pen is in the air. The
  sensor delivers its own force value; the geometry rules deliver FORCE_TARGET or 0.
* Verifier from the ink record: the target path sampled every 0.25 mm is covered where a mark lies within
  COVER_TOL_M; off-path ink = fraction of marks farther than OFF_PATH_M from the path. Success =
  the episode completed, coverage >= COVERAGE_MIN and off-path ink <= OFF_PATH_MAX.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes import PLAYGROUND, SHAPES, placed  # noqa: E402

TOUCH_N = 0.015
PRESS_BAND = (0.05, 0.40)
FORCE_TARGET = 0.15
COVER_TOL_M = 0.0010
OFF_PATH_M = 0.0015
COVERAGE_MIN = 0.90
OFF_PATH_MAX = 0.05
SENSOR = {"noise_sd_n": 0.01, "touch_threshold_n": 0.03}
GEOMETRY = {"L1": {"bias_mm": 2.0, "jitter_mm_sd": 1.0}, "L2": {"bias_mm": 5.0, "jitter_mm_sd": 2.0}}
RULE_MARGIN_M = 0.000125  # rules: touching when the estimated tip height is at most contact height + 0.125 mm
LEVELS = {"delay": (0.05, 0.2, 0.5), "flicker": (0.05, 0.2, 0.5), "fnr": (0.05, 0.2, 0.5)}
DEV_SEEDS = tuple(range(20))


def shape_for(seed: int) -> str:
    return SHAPES[int(np.random.default_rng(1_950_000 + seed).integers(len(SHAPES)))]


def densify(stroke: np.ndarray, step: float = 0.00025) -> np.ndarray:
    pts = [stroke[0]]
    for a, b in zip(stroke[:-1], stroke[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return np.array(pts)


def seg_dist(p: np.ndarray, strokes: list[np.ndarray]) -> np.ndarray:
    """Distance from each point to the nearest segment of any stroke."""
    best = np.full(len(p), np.inf)
    for s in strokes:
        for a, b in zip(s[:-1], s[1:]):
            ab = b - a
            t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
            best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def coverage(path_pts: np.ndarray, ink: np.ndarray) -> float:
    if len(ink) == 0:
        return 0.0
    d = np.min(np.linalg.norm(path_pts[:, None, :] - ink[None, :, :], axis=2), axis=1)
    return float(np.mean(d <= COVER_TOL_M))


class Source:
    """Delivered touching label and the reading the controller gets, from truth at each read."""

    def __init__(self, kind: str, level: float | None, seed: int, contact_z: float):
        self.kind, self.level = kind, level
        self.rng = np.random.default_rng(7_000_000 + seed)
        self.buf: list[tuple[float, float]] = []
        self.contact_z = contact_z
        if kind in GEOMETRY:
            g = GEOMETRY[kind]
            r = np.random.default_rng(5_000_011 + seed)
            v = r.normal(size=3)
            self.bias_z = float(v[2] / np.linalg.norm(v)) * g["bias_mm"] / 1000
            self.jitter = g["jitter_mm_sd"] / 1000

    def read(self, t: float, force: float, tip_z: float) -> tuple[bool, float]:
        touching = force > TOUCH_N
        k = self.kind
        if k == "truth":
            return touching, force
        if k == "delay":
            self.buf.append((t, force))
            past = [f for t_, f in self.buf if t_ <= t - self.level + 1e-9]
            f = past[-1] if past else self.buf[0][1]
            return f > TOUCH_N, f
        if k == "flicker":
            if self.rng.uniform() < self.level:
                return (not touching), (0.0 if touching else FORCE_TARGET)
            return touching, force
        if k == "fnr":
            if touching and self.rng.uniform() < self.level:
                return False, 0.0
            return touching, force
        if k == "sensor":
            r = max(0.0, force + float(self.rng.normal(0, SENSOR["noise_sd_n"])))
            return r > SENSOR["touch_threshold_n"], r
        if k in GEOMETRY:
            z = tip_z + self.bias_z + float(self.rng.normal(0, self.jitter))
            lab = z <= self.contact_z + RULE_MARGIN_M
            return lab, (FORCE_TARGET if lab else 0.0)
        raise ValueError(k)


def run_one(seed: int, kind: str = "truth", level: float | None = None, keep: str | None = None) -> dict:
    os.chdir(PLAYGROUND / "experiments" / "dove-drawing")
    sys.path.insert(0, os.getcwd())
    from controller import Controller
    from firmware import PAPER_Z
    from run import RecordingPhysical
    from writer import PhysicalWriter

    contact_z = PAPER_Z + CONTACT_OFFSET_M
    shape = shape_for(seed)
    strokes, meta = placed("drawing", shape, seed)
    f = RecordingPhysical()
    c = Controller(f)
    src = Source(kind, level, seed, contact_z)
    log: list[list] = []
    stroke_idx = [-1]

    class Injected(PhysicalWriter):
        def contact_force(self):
            force = super().contact_force()
            tip = self.tip()
            lab, reading = src.read(float(f.data.time), force, float(tip[2]))
            log.append([round(float(f.data.time), 3), stroke_idx[0], round(force, 4), int(force > TOUCH_N),
                        int(PRESS_BAND[0] <= force <= PRESS_BAND[1]), int(lab), round(reading, 4), len(f.marks),
                        round(float(tip[2]) - PAPER_Z, 5)])
            return reading

    w = Injected(c)
    t0 = time.monotonic()
    completed, error = True, None
    wall = [None]
    mark_start = []
    try:
        c.command({}, 1)
        w.goto([*strokes[0][0], 0.794], 2.5)
        for i, s in enumerate(strokes):
            stroke_idx[0] = i
            mark_start.append(len(f.marks))
            w.draw(s, 0.008)
        stroke_idx[0] = len(strokes)
        mark_start.append(len(f.marks))
        w.goto(w.tip() + [0, 0, 0.02], 1)
    except Exception as ex:  # the episode stops; recorded, not hidden
        completed, error = False, f"{type(ex).__name__}: {ex}"
        mark_start.append(len(f.marks))
    wall[0] = round(time.monotonic() - t0, 1)
    ink = np.array([p[:2] for p, _, _ in f.marks]) if f.marks else np.zeros((0, 2))
    dense = [densify(s) for s in strokes]
    cov = coverage(np.vstack(dense), ink)
    off = float(np.mean(seg_dist(ink, strokes) > OFF_PATH_M)) if len(ink) else 0.0
    per_stroke = []
    for i, d in enumerate(dense):
        a = mark_start[i] if i < len(mark_start) else len(ink)
        b = mark_start[i + 1] if i + 1 < len(mark_start) else len(ink)
        per_stroke.append({"stroke": i, "marks": int(b - a), "coverage": round(coverage(d, ink), 3)})
    L = np.array(log) if log else np.zeros((0, 9))
    label_acc = float(np.mean(L[:, 5] == L[:, 3])) if len(L) else None
    # stroke done / shape done from truth: the first read at which the stroke's coverage reached COVERAGE_MIN
    done_t = []
    for i, d in enumerate(dense):
        t_done = None
        if len(ink):
            near = np.linalg.norm(d[:, None, :] - ink[None, :, :], axis=2) <= COVER_TOL_M
            first = np.where(near.any(axis=1), near.argmax(axis=1), len(ink) + 1)  # first mark covering each point
            for row in L[L[:, 1] == i] if len(L) else []:
                if np.mean(first < row[7]) >= COVERAGE_MIN:
                    t_done = float(row[0])
                    break
        done_t.append(t_done)
    success = bool(completed and cov >= COVERAGE_MIN and off <= OFF_PATH_MAX)
    out = {"seed": seed, "shape": shape, **meta, "source": kind, "level": level, "success": success,
           "completed": completed, "error": error, "coverage": round(cov, 4), "off_path_ink": round(off, 4),
           "marks": int(len(ink)), "sim_s": round(float(f.data.time), 2), "wall_s": wall[0],
           "touching_label_accuracy": None if label_acc is None else round(label_acc, 4),
           "reads": int(len(L)), "reads_touching_truth": int(L[:, 3].sum()) if len(L) else 0,
           "reads_pressing_truth": int(L[:, 4].sum()) if len(L) else 0,
           "median_force_touching": round(float(np.median(L[L[:, 3] == 1, 2])), 4) if len(L) and L[:, 3].any() else None,
           "per_stroke": per_stroke, "stroke_done_t": done_t,
           "shape_done_t": max(done_t) if all(x is not None for x in done_t) else None}
    if keep:
        kd = Path(keep)
        kd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(kd / f"pen_{seed}_{kind}_{level}.npz", times=np.array([r[0] for r in f.frames]),
                            qpos=np.array([r[1] for r in f.frames]), mark_counts=np.array([r[2] for r in f.frames]),
                            marks=np.array([[*p, *q, v] for p, q, v in f.marks]) if f.marks else np.zeros((0, 7)),
                            reads=L, strokes=np.array(json.dumps([s.tolist() for s in strokes])))
        (kd / f"pen_{seed}_{kind}_{level}.json").write_text(json.dumps(out, indent=1))
    return out


# Tip-site height above the paper at true contact, measured once from the truth run on dev seed 0 (median
# 0.60 mm at reads with true contact, 5-95% 0.50-0.85 mm; hovering reads sit at 0.85 mm) and declared here
# before any rules cell ran. The rule threshold 0.60 + 0.125 = 0.725 mm is midway to the hover height.
CONTACT_OFFSET_M = 0.0006


def cells() -> list[tuple[str, float | None]]:
    out = [("truth", None), ("sensor", None), ("L1", None), ("L2", None)]
    out += [(k, lv) for k, lvs in LEVELS.items() for lv in lvs]
    return out


def _job(args):
    seed, kind, level = args
    try:
        return run_one(seed, kind, level)
    except Exception as ex:  # noqa: BLE001
        return {"seed": seed, "source": kind, "level": level, "success": False, "error": f"crash {ex}"}


def grid(out_path: str, processes: int = 10, only: list[str] | None = None) -> None:
    from multiprocessing import Pool

    jobs = [(s, k, lv) for k, lv in cells() if not only or k in only for s in DEV_SEEDS]
    done = set()
    if Path(out_path).exists():
        for line in Path(out_path).read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["source"], r["level"]))
    jobs = [j for j in jobs if j not in done]
    print(json.dumps({"grid": "phase 4 pen", "runs": len(jobs), "expected_runtime_min":
                      round(len(jobs) * 6.0 / processes / 60, 1)}), flush=True)
    with Pool(processes) as pool, open(out_path, "a") as fh:
        for r in pool.imap_unordered(_job, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "grid":
        procs = int(sys.argv[sys.argv.index("--processes") + 1]) if "--processes" in sys.argv else 10
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
        grid(sys.argv[2], procs, only)
    elif cmd == "one":
        seed, kind = int(sys.argv[2]), sys.argv[3]
        level = float(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else None
        keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
        print(json.dumps(run_one(seed, kind, level, keep)), flush=True)
