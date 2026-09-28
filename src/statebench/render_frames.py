"""Phase 6: RGB-D frames of the Phase 3 generator runs, for the perception turn.

Re-runs the (pinned, deterministic) generator for a seed with rendering on, and at
every 10 Hz tick captures an RGB frame and a metric depth frame from the existing
env camera (640x480 free camera, ``env.camera``). One archive per run:
``<band>/seed<NNNN>__<variant>.npz`` holding ``rgb`` (N,480,640,3 uint8),
``depth_mm`` (N,480,640 uint16, metres * 1000, 0 = no hit beyond the far plane),
``t`` (N,), and a JSON string ``meta`` with intrinsics, extrinsics and the
per-frame labels and truth sidecar. Packed per run because D: is exFAT with 512 KB
clusters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import data_root

VARIANTS_ALL = None  # filled from generator at import time of run_seed


class FrameGrabber:
    def __init__(self):
        self.rgb: list[np.ndarray] = []
        self.depth: list[np.ndarray] = []
        self.camera: dict[str, Any] | None = None

    def __call__(self, env, index: int) -> dict[str, Any]:
        renderer = env.renderer
        renderer.update_scene(env.data, camera=env.camera)
        rgb = renderer.render().copy()
        if self.camera is None:
            self.camera = camera_model(env)
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=env.camera)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        far = float(env.model.stat.extent * env.model.vis.map.zfar)
        depth_mm = np.where(depth >= far * 0.999, 0, np.round(depth * 1000.0)).clip(0, 65535)
        self.rgb.append(rgb)
        self.depth.append(depth_mm.astype(np.uint16))
        return {"frame_index": len(self.rgb) - 1}


def camera_model(env) -> dict[str, Any]:
    """Pinhole intrinsics and camera-to-world extrinsics of the rendered view."""
    renderer = env.renderer
    height, width = renderer.height, renderer.width
    gl = renderer.scene.camera
    pos = (np.asarray(gl[0].pos) + np.asarray(gl[1].pos)) / 2.0
    forward = np.asarray(gl[0].forward, dtype=float)
    up = np.asarray(gl[0].up, dtype=float)
    right = np.cross(forward, up)
    fovy = float(env.model.vis.global_.fovy)
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    # Camera frame: x right, y down, z forward (OpenCV convention).
    rotation = np.stack([right, -up, forward], axis=1)
    return {
        "width": width, "height": height, "fovy_deg": fovy,
        "K": [[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        "camera_to_world": {"R": rotation.tolist(), "t": pos.tolist(),
                            "convention": "OpenCV: x right, y down, z forward; world = robot base"},
        "mujoco_free_camera": {"lookat": list(map(float, env.camera.lookat)),
                               "distance": float(env.camera.distance),
                               "azimuth": float(env.camera.azimuth),
                               "elevation": float(env.camera.elevation)},
        "depth": "depth_mm = metres along the optical axis * 1000, uint16; 0 = beyond far plane",
    }


def render_seed(args: tuple[int, str]) -> dict[str, Any]:
    from .generator import VARIANTS, generate_run

    seed, out_dir = args
    out = Path(out_dir)
    written, nbytes, frames = [], 0, 0
    for variant in VARIANTS:
        path = out / f"seed{int(seed):04d}__{variant}.npz"
        if path.exists():
            written.append(path.name)
            nbytes += path.stat().st_size
            continue
        grab = FrameGrabber()
        run = generate_run(seed, variant, frames_hook=grab, render=True)
        meta = {
            "seed": seed, "variant": variant, "params": run["params"], "success": run["success"],
            "camera": grab.camera,
            "frames": [{"t": tick["t"], "phase": tick["phase"], "labels": tick["labels"],
                        "conditions": tick["conditions"], "near_miss": tick["near_miss"],
                        "truth": tick["truth"], "predicates": tick["predicates"]}
                       for tick in run["ticks"]],
        }
        np.savez_compressed(path, rgb=np.stack(grab.rgb), depth_mm=np.stack(grab.depth),
                            t=np.asarray([tick["t"] for tick in run["ticks"]]),
                            meta=np.asarray(json.dumps(meta)))
        written.append(path.name)
        nbytes += path.stat().st_size
        frames += len(grab.rgb)
    return {"seed": int(seed), "files": len(written), "bytes": nbytes, "frames": frames}


def render_band(seeds: list[int], processes: int = 10) -> dict[str, Any]:
    from multiprocessing import Pool

    out = data_root() / "renders" / "turn4_judge_train"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total = {"files": 0, "bytes": 0, "frames": 0}
    with Pool(processes=processes) as pool:
        for i, r in enumerate(pool.imap_unordered(render_seed, [(s, str(out)) for s in seeds]), 1):
            for k in total:
                total[k] += r[k]
            if i % 10 == 0:
                print(f"  rendered {i}/{len(seeds)} seeds, {total['bytes'] / 1e9:.2f} GB", flush=True)
    return {"dir": str(out), **total, "wall_s": round(time.perf_counter() - started, 1)}
