"""Turn 16: the Turn 15 drawing scene server (``draw_scene``, unchanged) plus read-only commands.

* {"cmd": "ink"} -> {"ink_xy": [[x, y], ...] (true ink marks on the paper, m), "target": [[[x, y], ...], ...] (the
  target strokes, m), "target_name"}.
* {"cmd": "record", "dir": d, "every_s": dt} (after reset): from now on, save a frame (the Turn 15 frame: the 3D view
  beside the top-down ink inset) whenever simulated time has advanced ``dt`` since the last one, as
  ``d/raw_XXXX.png``. {"cmd": "recorded"} -> {"frames": [[path, sim_time_s], ...]}; it also saves one final frame.
  Recording only reads the simulation; it does not change it.
Everything else is ``draw_scene.main`` as it was.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import draw_scene as D


def _frame(scene, path: str) -> str:
    marks = [m[0] for m in scene.f.marks]
    view = D.render_frame(scene.f.model, scene.f.data, marks, path, [0.45, 0.0, 0.80], 0.70, 150.0, -40.0)
    pts = [(-m[1], m[0]) for m in marks]
    return D.compose(view, D.ink_inset(pts, (-0.06, 0.06), (0.39, 0.51)), path)


def start_recording(scene, out_dir: str, every_s: float) -> list:
    rec: list = []
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = scene.f
    orig = f.step
    nxt = {"t": float(f.data.time)}

    def step():
        out = orig()
        t = float(f.data.time)
        if t >= nxt["t"]:
            nxt["t"] = t + every_s
            p = str(d / f"raw_{len(rec):04d}.png")
            rec.append([_frame(scene, p), round(t, 3)])
        return out

    f.step = step
    return rec


def main() -> None:
    scene = D.DrawScene()
    rec = None
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "reset":
            out = {"state": scene.reset(int(msg["seed"]), bool(msg.get("forced")))}
            rec = None
        elif msg["cmd"] == "do":
            out = scene.do(msg["skill"])
        elif msg["cmd"] == "ink":
            out = {"ink_xy": [[round(float(m[0][0]), 5), round(float(m[0][1]), 5)] for m in scene.f.marks],
                   "target": [s.round(5).tolist() for s in scene.strokes], "target_name": scene.target_name}
        elif msg["cmd"] == "record":
            rec = start_recording(scene, msg["dir"], float(msg["every_s"]))
            out = {"ok": True}
        elif msg["cmd"] == "recorded":
            d = Path(msg.get("dir") or Path(rec[0][0]).parent)
            rec.append([_frame(scene, str(d / f"raw_{len(rec):04d}.png")), round(float(scene.f.data.time), 3)])
            out = {"frames": rec}
        elif msg["cmd"] == "frame":
            out = {"path": _frame(scene, msg["path"])}
        elif msg["cmd"] == "verify":
            out = scene.verify()
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
