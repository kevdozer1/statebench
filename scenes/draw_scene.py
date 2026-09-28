"""Turn 15 Phase 0: the drawing scene at planner level, as a JSON-lines server (playground venv).

The playground's drawing scene (Kinova + Shadow hand + pencil, paper) with our own targets (``targets15``): simple
shapes and letters and outlines traced from 3 public-domain paintings. The dove artwork is not used.

Skills (the planner's actions):
* ``draw(k)``: if the pen is down, raise it to the hover height; move above stroke k's start; put the pen down
  (descend until the declared force sensor reads contact, then the 1.0 mm preload, as controller v11.1); draw the
  stroke at 8 mm/s, advancing while the sensor reads contact and re-descending after 3 missed reads. The pen stays
  down at the stroke's end.
* ``redraw(k)``: the same motion on stroke k again.
* ``lift``: raise the pen 20 mm.
* ``finish``: end the episode (the verifier decides).
Contact inside the skills comes from the declared force sensor (Turn 9: force + N(0, 0.01 N) > 0.03 N), because this
turn studies intent and keeps contact reliable.

Tracking (L1: 2 mm persistent bias in a random direction per seed plus 1 mm per-axis jitter per 30 Hz frame).
Per-stroke tracked coverage = the share of the stroke's densified path within COVER_RADIUS_M of a tracked tip
position recorded while the sensor read contact.

Forced failure: when ``forced`` is set by the caller (the Turn 13 schedule, p 0.3), the first draw of the declared
stroke (the longest) is interrupted at half its path: the pen stops advancing and the skill returns as usual.

Verifier: Turn 9's ink verifier (total coverage of the target by true ink >= COVERAGE_MIN and off-path share <=
OFF_PATH_MAX) plus the pen lifted at the end (true tip at least 5 mm above the paper and no tip-paper force),
checked when ``finish`` is called; a sim-time budget of 3 x the drawing time + 60 s.

Protocol (one JSON object per line): {"cmd": "reset", "seed": s, "forced": bool} -> {"state"}; {"cmd": "do", "skill":
"draw(3)"} -> {"state", "events"}; {"cmd": "verify"} -> the verdict; {"cmd": "quit"}.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_v111 as Q  # noqa: E402
import targets15 as TG  # noqa: E402

PB = Q.PB
COVER_RADIUS_M = 0.003
LIFT_M = 0.020
MISS_READS = 3
L1_BIAS_M, L1_JITTER_M = 0.002, 0.001


class DrawScene:
    def __init__(self):
        os.chdir(Q.PLAYGROUND / "experiments" / "dove-drawing")
        sys.path.insert(0, os.getcwd())

    def reset(self, seed: int, forced: bool) -> dict:
        from controller import Controller
        from firmware import PAPER_Z
        from run import RecordingPhysical
        from writer import PhysicalWriter

        self.seed, self.forced, self.PAPER_Z = seed, bool(forced), PAPER_Z
        tgt = TG.make_target(seed)
        self.target_name = tgt["name"]
        self.strokes = [np.asarray(s, dtype=float) for s in tgt["strokes"]]
        self.dense = [PB.densify(s) for s in self.strokes]
        self.lengths = [float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in self.strokes]
        self.forced_k = int(np.argmax(self.lengths))
        self.forced_used = False
        self.P = dict(Q.PARAMS)
        rng = np.random.default_rng(15_000_011 + seed)
        d = rng.normal(size=3)
        self.bias = L1_BIAS_M * d / np.linalg.norm(d)
        self.jrng = np.random.default_rng(15_100_000 + seed)
        self.srng = np.random.default_rng(15_200_000 + seed)
        self.hits: list[list[np.ndarray]] = [[] for _ in self.strokes]
        self.f = RecordingPhysical()
        self.c = Controller(self.f)
        self.pw = PhysicalWriter(self.c)
        self.c.command({}, 1)
        self.pw.goto([*self.strokes[0][0], PAPER_Z + 0.030], 2.5)
        self.pen = "up"
        self.done_calls = {k: 0 for k in range(len(self.strokes))}
        self.budget_s = sum(self.lengths) / self.P["SPEED_M_S"] * 3.0 + 60.0
        self.finished = False
        self.error = None
        return self.state()

    # ------------------------------------------------------------ sensing
    def sensor(self, force: float) -> int:
        return int(max(0.0, force + float(self.srng.normal(0, PB.SENSOR["noise_sd_n"]))) > PB.SENSOR["touch_threshold_n"])

    def track(self, k: int, contact: int) -> None:
        if contact:
            est = self.pw.tip() + self.bias + self.jrng.normal(0, L1_JITTER_M, size=3)
            self.hits[k].append(est[:2].copy())

    def coverage(self, k: int) -> float:
        if not self.hits[k]:
            return 0.0
        h = np.asarray(self.hits[k])
        pts = self.dense[k]
        dmin = np.min(np.linalg.norm(pts[:, None, :] - h[None, :, :], axis=2), axis=1)
        return float(np.mean(dmin <= COVER_RADIUS_M))

    # ------------------------------------------------------------ skills
    def _raise(self, dz: float, seconds: float) -> None:
        self.pw.goto(self.pw.tip() + [0, 0, dz], seconds)

    def _stroke(self, k: int) -> list[str]:
        P, PAPER_Z, c, pw = self.P, self.PAPER_Z, self.c, self.pw
        pts = self.strokes[k]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        total = float(cum[-1])
        stop_at = 0.5 * total if (self.forced and k == self.forced_k and not self.forced_used) else total
        if stop_at < total:
            self.forced_used = True

        def point(s):
            s = min(max(s, 0.0), total)
            j = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            u = (s - cum[j]) / max(seg[j], 1e-12)
            return pts[j] + u * (pts[j + 1] - pts[j])

        if self.pen == "down":
            self._raise(P["HOVER_M"] + 0.002, 0.6)
        pw.goto([*pts[0], PAPER_Z + P["HOVER_M"] + Q.CONTACT_OFFSET_M], 1.2)
        grip, _ = c.pose()
        seedq = self.f.data.qpos[:9].copy()
        dt = P["DT_S"]
        s, phase, preload, miss = 0.0, "down", 0.0, 0
        t_end = self.f.data.time + max(total, 0.005) / P["SPEED_M_S"] * P["STROKE_TIMEOUT_FACTOR"] + 10.0
        force = pw.contact_force()
        while s < stop_at - 1e-9:
            if self.f.data.time > t_end:
                self.pen = "down"
                return ["stroke_timeout"]
            tip = pw.tip()
            lab = self.sensor(force)
            self.track(k, lab)
            desired = point(s)
            if phase == "down":
                if lab:
                    phase, preload = "preload", P["PRELOAD_M"]
                else:
                    grip[2] -= P["DOWN_M_S"] * dt
            if phase == "preload":
                d = min(P["DOWN_M_S"] * dt, preload)
                grip[2] -= d
                preload -= d
                if preload <= 1e-12:
                    phase, miss = "draw", 0
            elif phase == "draw":
                if lab:
                    miss = 0
                    new = point(s + P["SPEED_M_S"] * dt)
                    grip[:2] += new - desired
                    s += P["SPEED_M_S"] * dt
                    desired = new
                else:
                    miss += 1
                    if miss >= MISS_READS:
                        phase = "down"
            grip[:2] += np.clip(0.15 * (desired - tip[:2]), -0.00015, 0.00015)
            q, _ = c.solve(grip, pw.R, seedq)
            seedq = q
            c.command({self.f.names[i]: float(q[i]) for i in range(9)}, dt)
            force = pw.contact_force()
            if force > P["FORCE_LIMIT_N"]:
                raise Q.PressedTooHard(f"true tip force {force:.2f} N")
        self.pen = "down"
        return ["draw_done"]

    def do(self, skill: str) -> dict:
        t0 = self.f.data.time
        events: list[str] = []
        try:
            if skill.startswith("draw(") or skill.startswith("redraw("):
                k = int(skill[skill.index("(") + 1: skill.index(")")])
                if 0 <= k < len(self.strokes):
                    events = [e.replace("draw_done", "redraw_done" if skill.startswith("redraw") else "draw_done")
                              for e in self._stroke(k)]
                    self.done_calls[k] += 1
                else:
                    events = ["rejected: no such stroke"]
            elif skill == "lift":
                if self.pen == "down":
                    self._raise(LIFT_M, 0.8)
                self.pen = "up"
                events = ["lift_done"]
            elif skill == "finish":
                self.finished = True
                events = ["finish_done"]
            else:
                events = ["rejected: unknown skill"]
        except Q.PressedTooHard as ex:
            self.error = f"pressed too hard: {ex}"
            events = ["error"]
        except Exception as ex:  # noqa: BLE001
            self.error = f"controller error: {type(ex).__name__}: {ex}"
            events = ["error"]
        return {"state": self.state(), "events": events, "sim_dt": round(self.f.data.time - t0, 2)}

    def state(self) -> dict:
        ink = np.array([m[0][:2] for m in self.f.marks]) if self.f.marks else np.zeros((0, 2))
        return {"target": self.target_name, "n_strokes": len(self.strokes),
                "true_coverage": [round(PB.coverage(self.dense[k], ink), 3) for k in range(len(self.strokes))],
                "pen_height_mm": round(float(self.pw.tip()[2] - self.PAPER_Z) * 1000, 1),
                "strokes": [{"k": k, "length_mm": round(self.lengths[k] * 1000, 1),
                             "start_xy_mm": [round(v * 1000, 1) for v in self.strokes[k][0]],
                             "tracked_coverage": round(self.coverage(k), 3), "times_drawn": self.done_calls[k]}
                            for k in range(len(self.strokes))],
                "pen": self.pen, "sim_time_s": round(float(self.f.data.time), 2), "budget_s": round(self.budget_s, 1),
                "error": self.error}

    def verify(self) -> dict:
        ink = np.array([m[0][:2] for m in self.f.marks]) if self.f.marks else np.zeros((0, 2))
        cov = PB.coverage(np.vstack(self.dense), ink)
        off = float(np.mean(PB.seg_dist(ink, self.strokes) > PB.OFF_PATH_M)) if len(ink) else 0.0
        tip_h = float(self.pw.tip()[2] - self.PAPER_Z)
        lifted = bool(tip_h >= 0.005 and self.pw.contact_force() < 0.015)
        per = [round(PB.coverage(self.dense[k], ink), 3) for k in range(len(self.strokes))]
        in_time = self.f.data.time <= self.budget_s + 1e-9
        success = bool(self.finished and self.error is None and cov >= PB.COVERAGE_MIN and off <= PB.OFF_PATH_MAX
                       and lifted and in_time)
        return {"success": success, "coverage": round(cov, 4), "off_path": round(off, 4), "pen_lifted": lifted,
                "tip_height_mm": round(tip_h * 1000, 1), "per_stroke_true_coverage": per, "finished": self.finished,
                "error": self.error, "in_time": in_time, "sim_time_s": round(float(self.f.data.time), 2),
                "forced_stroke": self.forced_k if self.forced else None}


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
    scene = DrawScene()
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "reset":
            out = {"state": scene.reset(int(msg["seed"]), bool(msg.get("forced")))}
        elif msg["cmd"] == "do":
            out = scene.do(msg["skill"])
        elif msg["cmd"] == "frame":
            marks = [m[0] for m in scene.f.marks]
            view = render_frame(scene.f.model, scene.f.data, marks, msg["path"], [0.45, 0.0, 0.80], 0.70, 150.0, -40.0)
            # the paper region the targets use (x 0.39-0.51 m, y -0.06 to 0.06 m), seen from above: paper x up, paper y left
            pts = [(-m[1], m[0]) for m in marks]
            out = {"path": compose(view, ink_inset(pts, (-0.06, 0.06), (0.39, 0.51)), msg["path"])}
        elif msg["cmd"] == "verify":
            out = scene.verify()
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
