"""Turn 17 Phase 4: terminal drawing scenes for the visibility probe (JSON-lines server; playground venv).

The Turn 15 drawing scene with custom strokes (``model_draw_scene16``'s mapping: canvas mm, 90 x 90 mm, centred on the
drawing area). Commands:
* {"cmd": "reset", "seed": s, "strokes_mm": [...]}: the strokes become the scene's strokes (no forced failure). Without
  ``strokes_mm`` the seed's own Turn 15 target is used, with the given ``forced`` flag (the frames arm re-renders the
  Turn 16 demonstrations this way);
* {"cmd": "do", "skill": "draw(k)" | "lift"}: the Turn 15 skills;
* {"cmd": "hold", "s": t}: hold the arm still for t s of simulated time;
* {"cmd": "record", "dir": d, "every_s": dt, "cameras": [...]}: from now on save a frame from each named camera every dt
  of simulated time; {"cmd": "recorded"} returns them (and saves one final frame per camera);
* {"cmd": "pen"}: the true tip height above the paper (mm) and the tip-paper contact force (N).
Cameras (declared before any probe call):
* ``existing``: the Turn 15/16 frame: the 3D view (lookat 0.45, 0, 0.80; distance 0.70; azimuth 150; elevation -40)
  beside the top-down ink inset (896 x 384);
* ``side``: a fixed camera at paper height looking across the paper: lookat (0.45, 0.0, paper + 15 mm), distance 0.45 m,
  azimuth SIDE_AZIMUTH, elevation -2 degrees (512 x 384).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import draw_scene as D
import draw_scene16 as D16
import model_draw_scene16 as M
import targets15 as TG

SIDE_AZIMUTH = 90.0


def side_frame(scene, path: str) -> str:
    marks = [m[0] for m in scene.f.marks]
    return D.render_frame(scene.f.model, scene.f.data, marks, path, [0.45, 0.0, scene.PAPER_Z + 0.015], 0.45,
                          SIDE_AZIMUTH, -2.0)


def frame(scene, cam: str, path: str) -> str:
    return D16._frame(scene, path) if cam == "existing" else side_frame(scene, path)


def main() -> None:
    scene = D.DrawScene()
    box: dict = {}
    orig_target = TG.make_target
    TG.make_target = lambda seed: (orig_target(seed) if box["strokes"] is None
                                   else {"name": "probe", "seed": seed, "strokes": box["strokes"]})
    rec = None
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "reset":
            sm = msg.get("strokes_mm")
            box["strokes"] = None if sm is None else [[M.to_world(x, y) for x, y in s] for s in sm]
            out = {"state": scene.reset(int(msg["seed"]), bool(msg.get("forced")))}
            rec = None
        elif msg["cmd"] == "do":
            out = scene.do(msg["skill"])
        elif msg["cmd"] == "hold":
            scene.c.command({}, float(msg["s"]))
            out = {"ok": True}
        elif msg["cmd"] == "pen":
            out = {"tip_height_mm": round(float(scene.pw.tip()[2] - scene.PAPER_Z) * 1000, 2),
                   "force_n": round(float(scene.pw.contact_force()), 4), "pen": scene.pen}
        elif msg["cmd"] == "record":
            d = Path(msg["dir"])
            d.mkdir(parents=True, exist_ok=True)
            rec = {"frames": [], "dir": d, "cams": msg["cameras"]}
            f = scene.f
            orig = f.step
            nxt = {"t": float(f.data.time)}
            every = float(msg["every_s"])

            def step(orig=orig, nxt=nxt, every=every, rec=rec):
                o = orig()
                t = float(f.data.time)
                if t >= nxt["t"]:
                    nxt["t"] = t + every
                    i = len(rec["frames"])
                    rec["frames"].append({"t": round(t, 3), **{c: frame(scene, c, str(rec["dir"] / f"{c}_{i:04d}.png"))
                                                                 for c in rec["cams"]}})
                return o

            f.step = step
            out = {"ok": True}
        elif msg["cmd"] == "recorded":
            i = len(rec["frames"])
            rec["frames"].append({"t": round(float(scene.f.data.time), 3),
                                  **{c: frame(scene, c, str(rec["dir"] / f"{c}_{i:04d}.png")) for c in rec["cams"]}})
            out = {"frames": rec["frames"]}
        elif msg["cmd"] == "frame":
            out = {"path": frame(scene, msg.get("camera", "existing"), msg["path"])}
        elif msg["cmd"] == "verify":
            out = scene.verify()
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
