"""Turn 16 Phase 2: the robot draws a model's strokes (JSON-lines server; playground venv).

The Turn 15 drawing scene (``draw_scene.DrawScene``, unchanged: Kinova + Shadow hand + pencil, L1 tracking, the
declared force sensor for contact) with the model's strokes in place of a target. The canvas is a 90 x 90 mm square
on the paper, centred on the Turn 15 drawing area (x 0.45 m, y 0 m), seen from above as in the Turn 15 ink inset:
canvas x to the right is paper -y, canvas y up is paper +x.

Protocol:
* {"cmd": "reset", "seed": s, "strokes_mm": [[[x, y], ...], ...]} -> {"state"}: the strokes (canvas mm, already
  clipped and capped by the caller) become the scene's strokes; ``seed`` sets the tracking noise; no forced failure.
* {"cmd": "do", "skill": "draw(k)" | "lift" | "finish"}: the Turn 15 skills. draw(k) raises the pen if it is down,
  moves above stroke k's start, puts the pen down by the force sensor, draws the stroke at 8 mm/s; the pen stays
  down at its end.
* {"cmd": "ink"} -> {"ink_mm": [[x, y], ...]} (true ink marks, canvas mm); {"cmd": "verify"} -> draw_scene's verdict
  (pen_lifted, finished, error, in_time are the fields used); {"cmd": "frame", "path"}; {"cmd": "quit"}.
"""
from __future__ import annotations

import json
import sys
import time

import draw_scene as D
import targets15 as TG

CANVAS_MM = 90.0
CX, CY = 0.45, 0.0


def to_world(x_mm: float, y_mm: float) -> list[float]:
    return [round(CX + (y_mm - CANVAS_MM / 2) / 1000.0, 6), round(CY - (x_mm - CANVAS_MM / 2) / 1000.0, 6)]


def to_canvas(wx: float, wy: float) -> list[float]:
    return [round((CY - wy) * 1000.0 + CANVAS_MM / 2, 3), round((wx - CX) * 1000.0 + CANVAS_MM / 2, 3)]


def main() -> None:
    scene = D.DrawScene()
    box: dict = {}
    TG.make_target = lambda seed: {"name": "model", "seed": seed, "strokes": box["strokes"]}
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "reset":
            box["strokes"] = [[to_world(x, y) for x, y in s] for s in msg["strokes_mm"]]
            out = {"state": scene.reset(int(msg["seed"]), False)}
        elif msg["cmd"] == "do":
            out = scene.do(msg["skill"])
        elif msg["cmd"] == "ink":
            out = {"ink_mm": [to_canvas(float(m[0][0]), float(m[0][1])) for m in scene.f.marks]}
        elif msg["cmd"] == "verify":
            out = scene.verify()
        elif msg["cmd"] == "frame":
            marks = [m[0] for m in scene.f.marks]
            view = D.render_frame(scene.f.model, scene.f.data, marks, msg["path"], [0.45, 0.0, 0.80], 0.70, 150.0, -40.0)
            pts = [(-m[1], m[0]) for m in marks]
            out = {"path": D.compose(view, D.ink_inset(pts, (-0.06, 0.06), (0.39, 0.51)), msg["path"])}
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
