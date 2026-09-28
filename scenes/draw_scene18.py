"""Turn 18: the model-draws scene, with strokes appended round by round (JSON-lines server; playground venv).

The Turn 15 drawing scene (``draw_scene.DrawScene``, unchanged: Kinova + Shadow hand + pencil, L1 tracking, the declared
force sensor) with:
* the Turn 16 canvas (``model_draw_scene16``: 90 x 90 mm, canvas mm to paper metres);
* {"cmd": "reset", "seed": s}: a fresh page; the pen starts 30 mm above the canvas centre; no strokes yet;
* {"cmd": "append", "strokes_mm": [...]}: add strokes (canvas mm) to the page's stroke list; returns their indices;
* {"cmd": "do", "skill": "draw(k)" | "lift" | "finish"}: the Turn 15 skills; ``finish`` is **finish-lift** (Turn 17,
  ``draw_scene17`` mode ``lift_home``): lift the pen 20 mm if it is down, move at that height to the home point, rise;
* {"cmd": "ink"}: the true ink marks in canvas mm; {"cmd": "verify"}: draw_scene's verdict (pen_lifted is used);
* {"cmd": "record", "dir", "every_s", "cameras"} / {"cmd": "recorded"}: frames for videos (``existing``: the Turn 15/16
  frame, the 3D view beside the top-down ink inset; ``side``: the Turn 17 side camera). Recording is used only when
  replaying finished programs for videos, never during a scored run.
There is no simulated-time budget for appended strokes (the Turn 15 budget belongs to a fixed target).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import draw_scene as D
import draw_scene16 as D16
import draw_scene17 as D17
import model_draw_scene16 as M
import probe_scene17 as P17
import targets15 as TG

CENTRE_MM = (45.0, 45.0)


def frame(scene, cam: str, path: str) -> str:
    return D16._frame(scene, path) if cam == "existing" else P17.side_frame(scene, path)


class Scene18(D17.Scene17):
    finish_mode = "lift_home"


def main() -> None:
    scene = Scene18()
    placeholder = [M.to_world(*CENTRE_MM), M.to_world(CENTRE_MM[0] + 0.5, CENTRE_MM[1])]
    TG.make_target = lambda seed: {"name": "model", "seed": seed, "strokes": [placeholder]}
    rec = None
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "reset":
            scene.reset(int(msg["seed"]), False)
            scene.strokes, scene.dense, scene.lengths, scene.hits, scene.done_calls = [], [], [], [], {}
            scene.budget_s = 1e9
            scene.marks_before_finish = None
            rec = None
            out = {"ok": True, "sim_time_s": round(float(scene.f.data.time), 3)}
        elif msg["cmd"] == "append":
            idx = []
            for s in msg["strokes_mm"]:
                w = np.asarray([M.to_world(x, y) for x, y in s], dtype=float)
                scene.strokes.append(w)
                scene.dense.append(D.PB.densify(w))
                scene.lengths.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()))
                scene.hits.append([])
                scene.done_calls[len(scene.strokes) - 1] = 0
                idx.append(len(scene.strokes) - 1)
            out = {"indices": idx}
        elif msg["cmd"] == "do":
            r = scene.do(msg["skill"])
            out = {"events": r["events"], "sim_time_s": round(float(scene.f.data.time), 3), "error": scene.error,
                   "pen": scene.pen}
        elif msg["cmd"] == "ink":
            out = {"ink_mm": [M.to_canvas(float(m[0][0]), float(m[0][1])) for m in scene.f.marks]}
        elif msg["cmd"] == "verify":
            v = scene.verify() if scene.strokes else {"pen_lifted": True, "error": scene.error, "finished": scene.finished}
            out = {k: v.get(k) for k in ("pen_lifted", "error", "finished", "tip_height_mm", "sim_time_s")}
        elif msg["cmd"] == "record":
            d = Path(msg["dir"])
            d.mkdir(parents=True, exist_ok=True)
            rec = {"frames": [], "dir": d, "cams": msg["cameras"]}
            f = scene.f
            orig = f.step
            nxt = {"t": float(f.data.time)}
            every = float(msg["every_s"])

            def step(orig=orig, nxt=nxt, every=every, rec=rec, f=f):
                o = orig()
                t = float(f.data.time)
                if t >= nxt["t"]:
                    nxt["t"] = t + every
                    i = len(rec["frames"])
                    rec["frames"].append({"t": round(t, 3), **{c: frame(scene, c, str(rec["dir"] / f"{c}_{i:05d}.png"))
                                                                 for c in rec["cams"]}})
                return o

            f.step = step
            out = {"ok": True}
        elif msg["cmd"] == "recorded":
            i = len(rec["frames"])
            rec["frames"].append({"t": round(float(scene.f.data.time), 3),
                                  **{c: frame(scene, c, str(rec["dir"] / f"{c}_{i:05d}.png")) for c in rec["cams"]}})
            out = {"frames": rec["frames"]}
        elif msg["cmd"] == "hold":
            scene.c.command({}, float(msg["s"]))
            out = {"ok": True}
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
