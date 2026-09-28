"""Turn 11 Phase 3: ink from pixels (time-boxed, dev only). Playground venv.

A declared fixed scene camera (the scene's only camera is the wrist camera, which sees the inside of the hand):
look-at the paper centre (0.46, 0, 0.764), azimuth 270 deg, elevation -55 deg, distance 0.40 m, 640 x 480, 30 Hz,
chosen for visibility of the pencil tip before any accuracy result. Graphite is drawn as the firmware's marks
rendered into the image (image formation), exactly as the playground renderer draws them.

The detector reads only the rendered frames and the L2 tip estimate (Turn 10's L2 bias and jitter), projected into
the image. It never receives mark counts, mark coordinates or contact events. Per frame, inside a disc of
REGION_M around the projected estimate:
* paper pixels: min channel >= PAPER_MIN; dark neutral pixels: max channel <= DARK_MAX and channel spread <= 30;
* observable when paper + dark neutral pixels cover at least OBS_MIN of the region (else "unknown": hand or pencil);
* touching when at least NEW_MIN pixels are dark neutral now and in the previous frame, and were paper REF_FRAMES
  frames ago (newly visible ink); not touching otherwise.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pen_v11 as P  # noqa: E402
import paths  # noqa: E402

CAM = {"lookat": (0.46, 0.0, 0.764), "azimuth": 270.0, "elevation": -55.0, "distance": 0.40, "W": 640, "H": 480}
# tuned on dev seed 1 open loop (tuning log in notes/turn11.md), then frozen
DET = {"REGION_M": 0.008, "PAPER_MIN": 170, "DARK_MAX": 210, "OBS_MIN": 0.5, "NEW_MIN": 1, "REF_FRAMES": 15}


class Camera:
    def __init__(self, model):
        import mujoco

        self.mj = mujoco
        self.m = model
        self.r = mujoco.Renderer(model, CAM["H"], CAM["W"], max_geom=20000)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = CAM["lookat"]
        self.cam.azimuth, self.cam.elevation, self.cam.distance = CAM["azimuth"], CAM["elevation"], CAM["distance"]
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0

    def render(self, data, marks) -> np.ndarray:
        mj = self.mj
        self.r.update_scene(data, camera=self.cam, scene_option=self.opt)
        sc = self.r.scene
        for row in marks:
            if sc.ngeom >= sc.maxgeom:
                break
            mj.mjv_initGeom(sc.geoms[sc.ngeom], mj.mjtGeom.mjGEOM_ELLIPSOID, np.array([0.0003, 0.0003, 0.000015]),
                            np.asarray(row[0]), np.eye(3).ravel(), np.array([0.1, 0.1, 0.1, 1.0]))
            sc.ngeom += 1
        c0, c1 = sc.camera[0], sc.camera[1]  # the two stereo eyes; the image is rendered from their midpoint
        self.gl = {"pos": (np.array(c0.pos) + np.array(c1.pos)) / 2, "forward": np.array(c0.forward),
                   "up": np.array(c0.up)}
        return self.r.render().copy()

    def project(self, p) -> tuple[float, float, float]:
        """World point -> pixel (u, v) and depth: a pinhole with the model's vertical field of view, posed by the
        rendered scene's GL camera."""
        g = self.gl
        pos, fwd, up = g["pos"], g["forward"], g["up"]
        right = np.cross(fwd, up)
        d = np.asarray(p) - pos
        x, y, z = float(d @ right), float(d @ up), float(d @ fwd)
        f = (CAM["H"] / 2) / np.tan(np.radians(self.m.vis.global_.fovy) / 2)
        return CAM["W"] / 2 + f * x / z, CAM["H"] / 2 - f * y / z, z


class PixelInk:
    def __init__(self):
        self.frames: list[np.ndarray] = []

    def label(self, rgb: np.ndarray, cam: Camera, est_xyz) -> tuple[int, float]:
        """(label, observable fraction); label 1 touching, 0 not touching, -1 unknown."""
        u, v, _ = cam.project(est_xyz)
        u2, _, _ = cam.project(np.asarray(est_xyz) + np.array([DET["REGION_M"], 0, 0]))
        u3, v3, _ = cam.project(np.asarray(est_xyz) + np.array([0, DET["REGION_M"], 0]))
        rad = max(3.0, float(np.hypot(u2 - u, 0)), float(np.hypot(u3 - u, v3 - v)))
        H, W = rgb.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        region = (xx - u) ** 2 + (yy - v) ** 2 <= rad ** 2
        self.frames.append(rgb)
        self.frames = self.frames[-(DET["REF_FRAMES"] + 1):]
        if not region.any():
            return -1, 0.0

        def classes(img):
            mn, mx = img.min(axis=2), img.max(axis=2)
            paper = mn >= DET["PAPER_MIN"]
            dark = (mx <= DET["DARK_MAX"]) & ((mx.astype(int) - mn.astype(int)) <= 30)
            return paper, dark

        paper, dark = classes(rgb)
        obs = float((paper | dark)[region].mean())
        if obs < DET["OBS_MIN"]:
            return -1, obs
        if len(self.frames) <= DET["REF_FRAMES"]:
            return 0, obs
        p_ref, _ = classes(self.frames[0])
        _, d_prev = classes(self.frames[-2])
        new = region & dark & d_prev & p_ref
        return int(new.sum() >= DET["NEW_MIN"]), obs


def open_loop(npz_path: str, seed: int) -> dict:
    """Replay a kept truth run: render every 30 Hz frame, run the detector, score against true contact."""
    os.chdir(P.PLAYGROUND / "experiments" / "dove-drawing")
    sys.path.insert(0, os.getcwd())
    import mujoco
    from firmware import Firmware

    a = np.load(npz_path)
    fw = Firmware("scene.xml", "initial_state.json")
    m, d = fw.model, fw.data
    cam = Camera(m)
    det = PixelInk()
    L = a["log30"]  # t, tip x y z, grip z, truth, force, stroke, marks
    marks = a["marks"]
    times = a["times"]
    v = np.random.default_rng(5_000_011 + seed).normal(size=3)
    bias = v / np.linalg.norm(v) * 0.005
    jr = np.random.default_rng(7_300_000 + seed)
    labs, obs = [], []
    for row in L:
        k = int(np.argmin(np.abs(times - row[0])))
        d.qpos[:] = a["qpos"][k]
        mujoco.mj_forward(m, d)
        rgb = cam.render(d, [(mk[:3],) for mk in marks[: int(row[8])]])
        est = np.array(row[1:4]) + bias + jr.normal(0, 0.002, size=3)
        lab, o = det.label(rgb, cam, est)
        labs.append(lab)
        obs.append(o)
    labs = np.array(labs)
    truth = L[:, 5].astype(int)
    return {"seed": seed, "frames": len(labs), "accuracy": float((labs == truth).mean()),
            "observable": float((labs != -1).mean()),
            "accuracy_observable": float((labs[labs != -1] == truth[labs != -1]).mean()) if (labs != -1).any() else None,
            "labels": labs.tolist(), "truth": truth.tolist(), "t": L[:, 0].tolist()}


class PixelSource(P.Source):
    """Closed loop: every new 30 Hz frame is rendered from the declared camera and read by the detector."""

    def __init__(self, kind, level, seed, paper_z):
        super().__init__("truth", level, seed, paper_z)
        self.kind = "pixink"
        v = np.random.default_rng(5_000_011 + seed).normal(size=3)
        self.bias = v / np.linalg.norm(v) * 0.005
        self.jr = np.random.default_rng(7_300_000 + seed)
        self.cam = None
        self.det = PixelInk()
        self.frame_t = -1.0
        self.cur = -1
        self.obs: list[float] = []

    def read(self, t, force, tip, grip_z, fw, first_touch_done):
        if self.cam is None:
            self.cam = Camera(fw.model)
        if t + 1e-9 >= self.frame_t + P.FRAME_DT:
            self.frame_t = t
            rgb = self.cam.render(fw.data, [(m[0],) for m in fw.marks])
            est = np.asarray(tip) + self.bias + self.jr.normal(0, 0.002, size=3)
            self.cur, o = self.det.label(rgb, self.cam, est)
            self.obs.append(o)
        return self.cur


def closed_loop(seed: int, keep: str | None = None, target_name: str | None = None) -> dict:
    orig = P.Source
    P.Source = lambda kind, level, seed_, paper_z: PixelSource(kind, level, seed_, paper_z)
    try:
        r = P.run_one(seed, "pixink", None, keep=keep, target_name=target_name)
    finally:
        P.Source = orig
    return r


def _ol(a):
    path, seed = a
    r = open_loop(path, seed)
    t, lab, tr = np.array(r["t"]), np.array(r["labels"]), np.array(r["truth"])
    lat = []
    for i in range(1, len(t)):
        if tr[i - 1] == 0 and tr[i] == 1:
            w = np.where((t >= t[i]) & (t <= t[i] + 3.0) & (lab == 1))[0]
            lat.append(None if len(w) == 0 else float(t[w[0]] - t[i]))
    return {k: r[k] for k in ("seed", "frames", "accuracy", "observable", "accuracy_observable")} | {"onset_latency": lat}


def _cl(seed):
    try:
        return closed_loop(seed)
    except Exception as ex:  # noqa: BLE001
        return {"seed": seed, "source": "pixink", "success": False, "error": f"crash {ex}"}


if __name__ == "__main__":
    import json
    from multiprocessing import Pool

    cmd = sys.argv[1]
    if cmd == "open":
        files = sorted((paths.DATA / "runs" / "turn11" / "truth_logs").glob("pen11_*_seed_truth_None.npz"))
        jobs = [(str(f), int(f.name.split("_")[1])) for f in files]
        print(json.dumps({"grid": "pixel ink open loop", "runs": len(jobs), "expected_runtime_min":
                          round(len(jobs) * 45 / 6 / 60, 1)}), flush=True)
        with Pool(6) as pool:
            rows = pool.map(_ol, jobs)
        (paths.DATA / "runs" / "turn11" / "pixel_open_loop.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(json.dumps(r))
    elif cmd == "closed":
        seeds = range(int(sys.argv[2]), int(sys.argv[3]))
        print(json.dumps({"grid": "pixel ink closed loop", "runs": len(seeds), "expected_runtime_min":
                          round(len(seeds) * 60 / 6 / 60, 1)}), flush=True)
        with Pool(6) as pool, open(paths.DATA / "runs" / "turn11" / "pen11_pixink_dev.jsonl", "a") as fh:
            for r in pool.imap_unordered(_cl, seeds):
                fh.write(json.dumps(r) + chr(10))
                fh.flush()
