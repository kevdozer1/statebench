"""Turn 19 article figure f04 (playground venv): false "not touching" against false "touching", same seed.

``contact19.py <seed> <out_dir>``: re-runs the Turn 12 confirmation cells ``fnr`` and ``fpr`` at level 0.5 on one
confirmation seed with controller v11.1 (``pen_v111.run_one``, keep), checks each outcome against the logged
confirmation line (passed in by the caller), and renders 10 Hz frames from the kept joint trajectories at 960 x 720 with
the Turn 9 close camera (``pen_demo.render_run``'s camera and ink stamps; only the offscreen size is larger). Writes
``<out_dir>/<source>/f_#####.png`` and ``<out_dir>/meta.json``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_bench as PB  # noqa: E402
import pen_v111 as Q  # noqa: E402


def render(npz: Path, out: Path, W: int = 960, H: int = 720) -> list:
    import mujoco
    from PIL import Image

    os.chdir(PB.PLAYGROUND / "experiments" / "dove-drawing")
    sys.path.insert(0, os.getcwd())
    from firmware import Firmware

    a = np.load(npz)
    strokes = [np.array(s) for s in json.loads(str(a["strokes"]))]
    fw = Firmware("scene.xml", "initial_state.json")
    m, d = fw.model, fw.data
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
    r = mujoco.Renderer(m, H, W, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera()
    centre = np.vstack(strokes).mean(axis=0)
    cam.lookat[:] = [centre[0], centre[1], 0.78]
    cam.distance, cam.azimuth, cam.elevation = 0.32, 150, -42
    marks, times = a["marks"], a["times"]
    out.mkdir(parents=True, exist_ok=True)
    ts = []
    for j, k in enumerate(range(0, len(times), 3)):  # 30 Hz recording -> 10 Hz
        d.qpos[:] = a["qpos"][k]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam, scene_option=opt)
        sc = r.scene
        for row in marks[: int(a["mark_counts"][k])][-6000:]:
            if sc.ngeom >= sc.maxgeom:
                break
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([0.0003, 0.0003, 0.000015]),
                                row[:3], np.eye(3).ravel(), np.array([0.1, 0.1, 0.1, 1.0]))
            sc.ngeom += 1
        Image.fromarray(r.render()).save(out / f"f_{j:05d}.png")
        ts.append(round(float(times[k]), 3))
    return ts


def main(seed: int, od: str) -> None:
    od = Path(od)
    keep = od / "runs"
    meta = {}
    for src in ("fnr", "fpr"):
        res = Q.run_one(seed, src, 0.5, keep=str(keep))
        npz = keep / f"pen111_{seed}_seed_{src}_0.5.npz"
        ts = render(npz, od / src)
        meta[src] = {k: res.get(k) for k in ("success", "failure", "coverage", "redescents", "strokes_started_in_air", "sim_s", "marks",
                                             "dislodged_ever")}
        meta[src]["frame_times"] = ts
    (od / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: {x: y for x, y in v.items() if x != "frame_times"} for k, v in meta.items()}))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
