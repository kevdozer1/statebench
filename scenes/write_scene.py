"""Turn 15 Phase 0: the writing scene at planner level, as a JSON-lines server (playground venv).

The playground's fibonacci-writing scene (Unitree G1 holding a marker at a whiteboard), with the scene's own writer
(``writer.Writer.draw``: approach, press with force control, follow the path, lift 15 mm) and glyph layout (one row,
15 mm per column, glyph height 35 mm).

Task per seed (``task_for``; its own RNG, 1_970_015 + seed): write a Fibonacci-style sequence **in binary**, each term
the sum of the two before, terms separated by single spaces, e.g. "1 1 10 11 101". The seed picks one of TASKS (start
pair and number of terms), all at most 14 characters on one line.
Why binary and short (declared, Phase 0): the scene's own glyph set has only the digits 0 and 1. Our glyphs for 2-9
lost ink contact at some board columns (a 3 failed at columns 7, 9, 17 and 18), and on lines longer than about 14
characters even the scene's own 0 and 1 drifted (misread glyphs from about column 15, and one lost grip at 169 s).
Columns 0-12 wrote 0, 1 and 10 patterns cleanly.

Skills (declared from the writer API):
* ``write_glyph(c)``: write digit c (0 or 1) at the cursor, then move the cursor one position right;
* ``space``: move the cursor one position right without writing;
* ``rewrite(c)``: write digit c again at the previous position (over the last written glyph); the cursor stays.
  Declared addition to the brief's list: without it a badly written glyph cannot be corrected;
* ``lift``: withdraw the marker from the board (tip about 60 mm away);
* ``finish``: end the episode.

Forced failure: when ``forced`` is set by the caller (p 0.3), the third write_glyph call writes only the first 25% of
the glyph's first stroke (a badly written glyph; it reads as ?, and a rewrite over it reads correctly).

Glyph check (declared): the ink marks in each column's cell are expressed in glyph units and compared with each digit
template (densified at 0.02 units) by the symmetric mean nearest-point (chamfer) distance; the cell reads as the
nearest digit if that distance is at most MATCH_MAX, "?" if not, and blank if it holds fewer than MIN_MARKS marks.
The recognized text is the cells from the first to the last non-blank one, blanks read as spaces.

Verifier: the recognized text equals the task's text exactly (every glyph right, in order, nothing extra), the marker
is lifted (tip at least 40 mm from the board) and ``finish`` was called, with no controller error.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import paths  # noqa: E402

HERE = Path(__file__).resolve().parent
PLAY = paths.PLAYGROUND / "experiments" / "fibonacci-writing"
MATCH_MAX = 0.12
MIN_MARKS = 15  # blank below this: a stray smear left 8-14 marks, a full glyph about 100-230, a bad (25%) glyph 22-33
BOARD_X = 0.4298
LIFTED_MIN_M = 0.040
FORCED_GLYPH_INDEX = 2
FORCED_FRACTION = 0.25
TASKS = (((0, 1), 4), ((0, 1), 5), ((1, 1), 4), ((1, 1), 5), ((1, 2), 3), ((1, 2), 4), ((2, 3), 3), ((2, 3), 4))
ALPHABET = "01"


def task_for(seed: int) -> dict:
    rng = np.random.default_rng(1_970_015 + seed)
    (a, b), n = TASKS[int(rng.integers(len(TASKS)))]
    terms = [a, b]
    while len(terms) < n:
        terms.append(terms[-1] + terms[-2])
    text = " ".join(format(t, "b") for t in terms)
    assert len(text) <= 14
    return {"start": [a, b], "n_terms": n, "terms": terms, "terms_binary": [format(t, "b") for t in terms], "text": text}


def cell_to_board(col: int, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return np.c_[-0.015 - (col + xy[:, 0]) * 0.015, 1.115 + xy[:, 1] * 0.035]


def densify(poly, step=0.02) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * (i + 1) / n for i in range(n)]
    return np.asarray(out)


class WriteScene:
    def __init__(self):
        os.chdir(PLAY)
        sys.path.insert(0, str(PLAY))
        sys.path.insert(0, str(HERE))
        import digits15

        self.glyphs = digits15.digit_glyphs()
        self.glyphs = {c: g for c, g in self.glyphs.items() if c in ALPHABET}
        self.templates = {c: np.vstack([densify(s) for s in strokes]) for c, strokes in self.glyphs.items()}

    def reset(self, seed: int, forced: bool) -> dict:
        from writer import Writer

        self.seed, self.forced = seed, bool(forced)
        self.task = task_for(seed)
        self.w = Writer()
        self.w.goto([0.410, -0.015, 1.15], 2)
        self.cursor, self.writes, self.finished, self.error = 0, 0, False, None
        self.marker = "near board"
        return self.state()

    def _write(self, col: int, ch: str, partial: bool) -> None:
        strokes = self.glyphs[ch]
        if partial:
            s = np.asarray(strokes[0])
            strokes = [s[: max(2, int(np.ceil(len(s) * FORCED_FRACTION)))]]
        for st in strokes:
            self.w.draw(cell_to_board(col, st), 0.018)
        self.marker = "near board"

    def cells(self) -> list[str]:
        marks = np.array(self.w.marks)[:, 1:3] if self.w.marks else np.zeros((0, 2))
        out = []
        n_cols = max(self.cursor, 1)
        for col in range(n_cols + 1):
            gx = (-0.015 - marks[:, 0]) / 0.015 - col
            gy = (marks[:, 1] - 1.115) / 0.035
            sel = (gx >= -0.1) & (gx <= 0.9) & (gy >= -0.35) & (gy <= 1.2)
            pts = np.c_[gx[sel], gy[sel]]
            if len(pts) < MIN_MARKS:
                out.append(" ")
                continue
            best, bd = "?", 1e9
            for c, tpl in self.templates.items():
                d1 = np.min(np.linalg.norm(pts[:, None] - tpl[None], axis=2), axis=1).mean()
                d2 = np.min(np.linalg.norm(tpl[:, None] - pts[None], axis=2), axis=1).mean()
                if (d1 + d2) / 2 < bd:
                    best, bd = c, (d1 + d2) / 2
            out.append(best if bd <= MATCH_MAX else "?")
        return out

    def recognized(self) -> str:
        return "".join(self.cells()).strip(" ")

    def do(self, skill: str) -> dict:
        t0 = self.w.d.time
        events: list[str] = []
        try:
            if skill.startswith("write_glyph(") and skill.endswith(")"):
                ch = skill[len("write_glyph("):-1]
                if ch in self.glyphs:
                    self.writes += 1
                    self._write(self.cursor, ch, self.forced and self.writes == FORCED_GLYPH_INDEX + 1)
                    self.cursor += 1
                    events = ["glyph_done"]
                else:
                    events = ["rejected: unknown glyph"]
            elif skill.startswith("rewrite(") and skill.endswith(")"):
                ch = skill[len("rewrite("):-1]
                if ch in self.glyphs and self.cursor > 0:
                    self._write(self.cursor - 1, ch, False)
                    events = ["rewrite_done"]
                else:
                    events = ["rejected"]
            elif skill == "space":
                self.cursor += 1
                events = ["space_done"]
            elif skill == "lift":
                tip = self.w.tip()
                self.w.goto([0.365, tip[1], tip[2]], 0.8)
                self.marker = "clear of board"
                events = ["lift_done"]
            elif skill == "finish":
                self.finished = True
                events = ["finish_done"]
            else:
                events = ["rejected: unknown skill"]
        except Exception as ex:  # noqa: BLE001
            self.error = f"controller error: {type(ex).__name__}: {ex}"[:200]
            events = ["error"]
        return {"state": self.state(), "events": events, "sim_dt": round(self.w.d.time - t0, 2)}

    def state(self) -> dict:
        return {"recognized_text": self.recognized(), "cursor": self.cursor, "marker": self.marker,
                "sim_time_s": round(float(self.w.d.time), 2), "error": self.error}

    def verify(self) -> dict:
        tip = self.w.tip()
        lifted = bool(BOARD_X - tip[0] >= LIFTED_MIN_M)
        rec = self.recognized()
        success = bool(self.finished and self.error is None and rec == self.task["text"] and lifted)
        return {"success": success, "recognized": rec, "target": self.task["text"], "lifted": lifted,
                "tip_distance_mm": round(float(BOARD_X - tip[0]) * 1000, 1), "finished": self.finished, "error": self.error,
                "sim_time_s": round(float(self.w.d.time), 2), "cells": self.cells()}


def render_frame(model, data, marks_xyz, path, lookat, distance, azimuth, elevation, radius=0.0006):
    """512x384 RGB frame from a fixed free camera, ink stamped as small dark ellipsoids; saved as PNG."""
    import mujoco
    from PIL import Image

    global _RENDERER
    try:
        _RENDERER
    except NameError:
        _RENDERER = {}
    key = id(model)
    if key not in _RENDERER:
        _RENDERER.clear()
        _RENDERER[key] = mujoco.Renderer(model, 384, 512)
    r = _RENDERER[key]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    r.update_scene(data, camera=cam)
    for p in marks_xyz[-3000:]:
        if r.scene.ngeom >= r.scene.maxgeom:
            break
        g = r.scene.geoms[r.scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, radius, radius]), np.asarray(p[:3], dtype=float),
                            np.eye(3).ravel(), np.array([0.1, 0.1, 0.12, 1.0], dtype=np.float32))
        r.scene.ngeom += 1
    Image.fromarray(r.render()).save(path)
    return str(path)


def ink_inset(points_2d, x_range, y_range, flip_x=False, radius_px=2):
    """384 x 384 top-down image of the ink: points_2d (N, 2) in the given coordinate ranges."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (384, 384), (247, 245, 236))
    dr = ImageDraw.Draw(im)
    (x0, x1), (y0, y1) = x_range, y_range
    for a, b in points_2d:
        u = (a - x0) / (x1 - x0)
        v = (b - y0) / (y1 - y0)
        if flip_x:
            u = 1 - u
        px, py = u * 383, (1 - v) * 383
        dr.ellipse((px - radius_px, py - radius_px, px + radius_px, py + radius_px), fill=(30, 30, 36))
    return im


def compose(view_png, inset_img, path):
    from PIL import Image

    v = Image.open(view_png).convert("RGB")
    W = Image.new("RGB", (v.width + inset_img.width, max(v.height, inset_img.height)), (255, 255, 255))
    W.paste(v, (0, 0))
    W.paste(inset_img, (v.width, 0))
    W.save(path)
    return str(path)


def main() -> None:
    scene = WriteScene()
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "reset":
            out = {"state": scene.reset(int(msg["seed"]), bool(msg.get("forced"))), "task": scene.task}
        elif msg["cmd"] == "do":
            out = scene.do(msg["skill"])
        elif msg["cmd"] == "frame":
            marks = [m[:3] for m in scene.w.marks]
            view = render_frame(scene.w.m, scene.w.d, marks, msg["path"], [0.30, -0.12, 1.05], 1.10, 20.0, -8.0, 0.0012)
            # the written line (board y -0.35 to -0.01 m, z 1.08 to 1.17 m), seen from the robot's side (y leftward)
            pts = [(m[1], m[2]) for m in marks]
            out = {"path": compose(view, ink_inset(pts, (-0.35, -0.01), (1.00, 1.25), flip_x=True, radius_px=1), msg["path"])}
        elif msg["cmd"] == "verify":
            out = scene.verify()
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
