"""Turn 6 estimation pipeline: sensor-informed synthetic perturbations, frozen E1, coherent corrections.

**Sensor-informed synthetic perturbations** (never called realistic), applied to
the rendered frame or the camera model before E1 sees it, from a fixed seed per
episode:

* ``depth``: per-pixel Gaussian noise with sigma_z = z^2 * s / (f * B),
  s = 0.08 px subpixel disparity RMS, B = 50 mm baseline, f from the render
  intrinsics (579.41 px), following Intel's RealSense D400-series depth-tuning
  guidance for depth RMS error; a 2 px invalid band at depth discontinuities
  (adjacent pixels differing by more than 20 mm, a stated convention); 1 mm
  quantization. It matches RMS magnitude only: no spatial correlation, bias or
  holes of the real sensor are modelled.
* ``rgb``: pixel noise sigma = 3/255, exposure gain U(0.7, 1.3) and per-channel
  white balance U(0.9, 1.1) drawn once per episode; a randomized key light and one
  of 5 table textures (``scene_v6``), drawn once per episode.
* ``extrinsics``: the rendering camera is displaced by 5 mm (random direction) and
  rotated by 1 degree (random direction in azimuth/elevation), drawn once per
  seed; E1 keeps the nominal pose. 1 degree at 1.1 m is about 19 mm laterally.
* ``latency``: E1 receives the most recent frame at least 150 ms old (frames
  rendered at 20 Hz, so ages are 150-200 ms); the estimate carries that frame's
  time as ``t_obs`` and the v0.2 executor registers it there.
* ``combined``: all four.

**Coherent corrections** replace one underlying input or estimate with truth and
recompute everything that depends on it (both relative positions,
gripper_above_target, the motion relation and the track-derived event, contact
and support by E1's own rules, and, in the runner, the predicates):

* ``pose``: E1's cube position and yaw replaced by truth at the frame's time.
* ``track_timing``: the motion relation and lift event computed from the track
  interpolated at the exact window endpoints.
* ``identity``: E1 receives the true target mask (a truth segmentation; the frame
  is recoloured so E1's own red test selects exactly those pixels).
* ``calibration``: E1 receives the true (rendering) camera pose.
* ``latency``: E1 receives the fresh frame.
* ``semantic``: contact and the two closed-gripper predicates from truth (pad
  contact) instead of the opening rule; applied by the runner after the
  unchanged predicate code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .frames_v2 import truth_cube_mask
from .observation_e1 import build_estimated_state, support_rule
from .observation_e1_1 import lift_diverged_interp, moving_interp  # E1.1's motion layer
from .perception_e1 import CONSTANTS, estimate, new_memory
from .render_frames import camera_model

DEPTH_SUBPIXEL_PX = 0.08
DEPTH_BASELINE_M = 0.050
DEPTH_DISCONTINUITY_M = 0.020
DEPTH_INVALID_BAND_PX = 2
RGB_NOISE_SIGMA = 3.0 / 255.0
EXPOSURE_RANGE = (0.7, 1.3)
WHITE_BALANCE_RANGE = (0.9, 1.1)
EXTRINSIC_TRANSLATION_M = 0.005
EXTRINSIC_ROTATION_DEG = 1.0
LATENCY_S = 0.150
LATENCY_FRAME_DT = 0.05
CONDITIONS = {
    "clean": (),
    "depth": ("depth",),
    "rgb": ("rgb",),
    "extrinsics": ("extrinsics",),
    "latency": ("latency",),
    "combined": ("depth", "rgb", "extrinsics", "latency"),
}
CORRECTIONS = ("pose", "track_timing", "identity", "calibration", "latency", "semantic")


def applicable_corrections(condition: str) -> tuple[str, ...]:
    parts = CONDITIONS[condition]
    out = ["pose", "track_timing", "identity"]
    if "extrinsics" in parts:
        out.append("calibration")
    if "latency" in parts:
        out.append("latency")
    out.append("semantic")
    return tuple(out)


def depth_sigma_m(z_m: np.ndarray, f_px: float) -> np.ndarray:
    return z_m ** 2 * DEPTH_SUBPIXEL_PX / (f_px * DEPTH_BASELINE_M)


def perturb_depth(depth_mm: np.ndarray, f_px: float, rng: np.random.Generator) -> np.ndarray:
    z = depth_mm.astype(float) / 1000.0
    valid = depth_mm > 0
    noisy = z + rng.normal(0.0, 1.0, size=z.shape) * depth_sigma_m(z, f_px)
    edge = np.zeros_like(valid)
    dx = np.abs(np.diff(z, axis=1)) > DEPTH_DISCONTINUITY_M
    dy = np.abs(np.diff(z, axis=0)) > DEPTH_DISCONTINUITY_M
    edge[:, :-1] |= dx
    edge[:, 1:] |= dx
    edge[:-1, :] |= dy
    edge[1:, :] |= dy
    band = edge.copy()
    for _ in range(DEPTH_INVALID_BAND_PX):
        grown = band.copy()
        grown[1:, :] |= band[:-1, :]
        grown[:-1, :] |= band[1:, :]
        grown[:, 1:] |= band[:, :-1]
        grown[:, :-1] |= band[:, 1:]
        band = grown
    out = np.round(noisy * 1000.0)
    out[~valid | band] = 0
    return out.clip(0, 65535).astype(np.uint16)


def perturb_rgb(rgb: np.ndarray, exposure: float, wb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = rgb.astype(float) / 255.0 * exposure * wb[None, None, :]
    img += rng.normal(0.0, RGB_NOISE_SIGMA, size=img.shape)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _forward(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def perturbed_camera(nominal: mujoco.MjvCamera, rng: np.random.Generator) -> tuple[mujoco.MjvCamera, dict]:
    """A free camera whose pose is the nominal pose moved 5 mm and turned 1 degree."""
    fwd = _forward(nominal.azimuth, nominal.elevation)
    pos = np.asarray(nominal.lookat) - nominal.distance * fwd
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    pos2 = pos + EXTRINSIC_TRANSLATION_M * direction
    phi = rng.uniform(0, 2 * math.pi)
    az2 = nominal.azimuth + EXTRINSIC_ROTATION_DEG * math.cos(phi) / max(1e-6, math.cos(math.radians(nominal.elevation)))
    el2 = nominal.elevation + EXTRINSIC_ROTATION_DEG * math.sin(phi)
    fwd2 = _forward(az2, el2)
    cam = mujoco.MjvCamera()
    cam.distance = nominal.distance
    cam.azimuth, cam.elevation = az2, el2
    cam.lookat[:] = pos2 + nominal.distance * fwd2
    angle = math.degrees(math.acos(float(np.clip(np.dot(fwd, fwd2), -1, 1))))
    return cam, {"translation_m": [round(float(v), 5) for v in pos2 - pos], "rotation_deg": round(angle, 4),
                 "lateral_at_1.1m_mm": round(1100 * math.tan(math.radians(angle)), 2)}


@dataclass
class Frame:
    t: float
    rgb: np.ndarray
    depth: np.ndarray
    proprio: dict
    truth_cube: list
    truth_yaw: float
    mask: np.ndarray | None


class Pipeline:
    def __init__(self, env, condition: str, corrections: tuple[str, ...], seed: int,
                 e1_version: str = "E1", constants: dict | None = None, variant_cfg=None):
        self.env = env
        self.parts = CONDITIONS[condition]
        self.corrections = set(corrections)
        self.e1_version = e1_version
        self.constants = dict(constants or CONSTANTS)
        self.rng = np.random.default_rng(1_000_003 * 7 + int(seed))
        self.exposure = float(self.rng.uniform(*EXPOSURE_RANGE)) if "rgb" in self.parts else 1.0
        self.wb = self.rng.uniform(*WHITE_BALANCE_RANGE, size=3) if "rgb" in self.parts else np.ones(3)
        self.nominal_camera = self._camera_for(env.camera)
        self.extrinsic_info = None
        if "extrinsics" in self.parts:
            cam, self.extrinsic_info = perturbed_camera(env.camera, self.rng)
            env.camera = cam
        self.true_camera = self._camera_for(env.camera)
        self.memory = new_memory()
        self.track: list[dict] = []
        self.frames: list[Frame] = []
        self.need_mask = "identity" in self.corrections
        self.proprio_by_t: dict[float, dict] = {}
        self.last_frame_t = -1.0

    def _camera_for(self, cam) -> dict:
        self.env.renderer.update_scene(self.env.data, camera=cam)
        return camera_model(self.env)

    # --------------------------------------------------------------- frames
    def capture(self, proprio: dict) -> Frame:
        from .frames_v2 import render_rgbd

        env = self.env
        rgb, depth = render_rgbd(env)
        mask = truth_cube_mask(env) if self.need_mask else None
        if "rgb" in self.parts:
            rgb = perturb_rgb(rgb, self.exposure, self.wb, self.rng)
        if "depth" in self.parts:
            depth = perturb_depth(depth, float(self.nominal_camera["K"][0][0]), self.rng)
        frame = Frame(round(env.t, 3), rgb, depth, proprio, [float(v) for v in env.truth_cube_position()],
                      float(env.truth_cube_yaw()), mask)
        self.frames.append(frame)
        self.frames = [f for f in self.frames if f.t >= env.t - 1.0]
        self.proprio_by_t[frame.t] = proprio
        self.last_frame_t = env.t
        return frame

    def frame_hook(self, env, proprio_fn) -> None:
        """Latency conditions render at 20 Hz so a 150 ms old frame exists."""
        if "latency" in self.parts and env.t + 1e-9 >= self.last_frame_t + LATENCY_FRAME_DT:
            self.capture(proprio_fn())

    def select(self, fresh: Frame) -> Frame:
        if "latency" not in self.parts or "latency" in self.corrections:
            return fresh
        cutoff = fresh.t - LATENCY_S + 1e-9
        old = [f for f in self.frames if f.t <= cutoff]
        return old[-1] if old else self.frames[0]

    # ------------------------------------------------------------- estimate
    def observe(self, proprio_now: dict, events: list[dict]) -> tuple[dict, dict, Frame]:
        fresh = self.capture(proprio_now)
        frame = self.select(fresh)
        rgb = frame.rgb
        if "identity" in self.corrections and frame.mask is not None:
            rgb = np.where(frame.mask[..., None], np.array([255, 0, 0], np.uint8), np.array([128, 128, 128], np.uint8))
        camera = self.true_camera if "calibration" in self.corrections else self.nominal_camera
        est, self.memory = estimate(rgb, frame.depth, camera, frame.proprio, self.constants, self.memory)
        est = dict(est)
        if "pose" in self.corrections:
            est.update(cube_m=list(frame.truth_cube), yaw_mod90_rad=float(frame.truth_yaw) % (math.pi / 2),
                       t_est=frame.t, fresh=True)
        if est.get("fresh"):
            self.track.append({"t": frame.t, "cube_m": list(est["cube_m"]),
                               "tcp_m": [float(v) for v in frame.proprio["tcp_position_m"]]})
            self.track = [e for e in self.track if e["t"] >= proprio_now["sim_time_s"] - 2.0 - 1e-9]
        at_est = frame.proprio if est.get("t_est") is not None and abs(float(est["t_est"]) - frame.t) < 1e-6 \
            else self._proprio_at(est.get("t_est"), proprio_now)
        state = build_estimated_state(est, at_est, proprio_now, events, self.track, self.constants)
        if self.e1_version == "E1.1" or "track_timing" in self.corrections:
            t_end = self.track[-1]["t"] if self.track else float(proprio_now["sim_time_s"])
            state["relations"]["target_moving_with_gripper"] = moving_interp(self.track, t_end)
        if "semantic" in self.corrections:
            state = semantic_contact(state, self.env, est, proprio_now, self.constants)
        return state, est, frame

    def _proprio_at(self, t: float | None, fallback: dict) -> dict:
        if t is None:
            return fallback
        return self.proprio_by_t.get(round(float(t), 3), fallback)

    def lift_diverged(self, t0: float, t1: float) -> bool | None:
        from .observation_e1 import lift_diverged

        if self.e1_version == "E1.1" or "track_timing" in self.corrections:
            return lift_diverged_interp(self.track, t0, t1)
        return lift_diverged(self.track, t0, t1)


def truth_grip(env) -> dict[str, bool]:
    """Semantic facts from truth: pad contact with the target, and the commanded grip."""
    fingers = env.truth_cube_contacts()["fingers"]
    closed = env.proprio_grip_command() == "closed"
    return {"contact": bool(fingers), "closed_on_object": bool(closed and fingers),
            "closed_empty": bool(closed and not fingers)}


def semantic_contact(state: dict, env, est: dict, proprio_now: dict, constants: dict) -> dict:
    """Contact (and support by E1's rule) recomputed with the truth contact fact."""
    grip = truth_grip(env)
    state["contact"]["gripper_target"] = "contact" if grip["contact"] else "no_contact"
    state["contact"]["evidence"] += "; contact from truth (semantic correction)"
    cube = None if est.get("cube_m") is None else np.asarray(est["cube_m"], dtype=float)
    if "target_supported_by" in state["relations"]:
        state["relations"]["target_supported_by"] = support_rule(cube, state["contact"]["gripper_target"], constants)
    state["uncertainty"]["contact_observed"] = True
    state["provenance"]["source"]["contact.gripper_target"] = "sim_truth"
    return state


def semantic_predicates(values: dict, env) -> dict:
    """Replace the two closed-gripper predicates with their truth-based meaning."""
    grip = truth_grip(env)
    out = dict(values)
    out["gripper_closed_on_object"] = grip["closed_on_object"]
    out["gripper_closed_empty"] = grip["closed_empty"]
    return out
