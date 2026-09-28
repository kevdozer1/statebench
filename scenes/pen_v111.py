"""Turn 12: controller v11.1 = controller v11 (Turn 11, unchanged in its decisions) with a corrected clock and the
Turn 12 additions. Playground venv only. v11 (external/playground_v11/pen_v11.py) is preserved as the original.

Clock contract (the Turn 11 defect: frame_t was reset to the control time, so a "30 Hz" stream ran at 25 Hz):
* observation frames are taken inside the physics step at k/30 s on the simulation clock (the first 2 ms physics
  step at or after k/30), never reset to the control time, and each carries its own timestamp;
* at every 20 ms control read, a source consumes every frame that arrived since the last read, in order;
* every filter, error process and window updates from the elapsed time between frame timestamps.

Label schema (every read in the log): kind (state or event source), value (1 / 0 / -1 unknown), source, t_evidence
(the time of the evidence it rests on) and the evidence age when used. Event sources declare when the absence of the
event counts as evidence (ABSENCE_RULE).

Additions: G4.1 (a G3 decision expires EXPIRE_G3_S after the grip last moved, then G2 is used); the shared ink
adapter (ideal and pixel ink); shadow sources (evaluated and logged, never controlling); scripted perturbations
(pauses and lifts) for adapter tests; scene variants (tilted paper, stiffer grip); the tool-dislodged early stop.
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
import pen_bench as PB  # noqa: E402
import pen_v11 as V11  # noqa: E402
from targets import load_target, write_target  # noqa: E402

PLAYGROUND = PB.PLAYGROUND
PARAMS = dict(V11.PARAMS)  # the frozen v11 controller parameters, unchanged
FRAME_DT = 1.0 / 30.0
TOUCH_N = PB.TOUCH_N
CONTACT_OFFSET_M = PB.CONTACT_OFFSET_M
SCALES, STRUCTURES = V11.SCALES, V11.STRUCTURES
EST = {k: dict(v) for k, v in V11.EST.items()}
EXPIRE_G3_S = 2.0  # G4.1: tuned on dev (0.5, 1.0, 2.0 s; pooled success, ties by lower mean time), frozen
ADAPTER = {"ADVANCE_M": 0.0015, "EXPIRE_S": 2.5}  # expiry longer than the longest declared pause (2.0 s)
DISLODGED = {"TILT_DEG": 20.0, "SLIDE_M": 0.008, "HOLD_S": 1.0, "EARLY_STOP": False}
ABSENCE_RULE = {"ink": ("no new ink counts as no contact only after the pen was commanded to advance ADVANCE_M "
                        "without a new ink event; otherwise the state holds for up to EXPIRE_S after the last event, "
                        "then becomes unknown; unknown at stroke start"),
                "ink_v11": "Turn 11 adapter: no new ink in the last frame counts as no contact (for the record)"}


# ------------------------------------------------------------------ error process, true elapsed-time updates
class ErrorProcess:
    def __init__(self, structure: str, scale: str, seed: int, tau_s: float = V11.AR_TAU_S):
        self.st, self.sig, self.tau = STRUCTURES[structure], SCALES[scale], tau_s
        self.rng = np.random.default_rng(8_100_000 + seed)
        s2 = self.sig ** 2
        shared = self.rng.normal(0, math.sqrt(self.st["shared_bias"] * s2), size=3)
        self.bias_tip = shared + self.rng.normal(0, math.sqrt(self.st["own_bias"] * s2), size=3)
        self.bias_paper = shared[2] + self.rng.normal(0, math.sqrt(self.st["own_bias"] * s2))
        self.ar_tip = self.rng.normal(0, math.sqrt(self.st["ar"] * s2), size=3)
        self.ar_paper = self.rng.normal(0, math.sqrt(self.st["ar"] * s2))
        self.last_t = None

    def frame(self, t: float):
        s2 = self.sig ** 2
        if self.st["ar"] > 0 and self.last_t is not None:
            phi = math.exp(-(t - self.last_t) / self.tau)
            k = math.sqrt(self.st["ar"] * s2 * (1 - phi ** 2))
            self.ar_tip = phi * self.ar_tip + self.rng.normal(0, k, size=3)
            self.ar_paper = phi * self.ar_paper + self.rng.normal(0, k)
        self.last_t = t
        j = math.sqrt(self.st["jitter"] * s2)
        return self.bias_tip + self.ar_tip + self.rng.normal(0, j, size=3), \
            float(self.bias_paper + self.ar_paper + self.rng.normal(0, j))


# ------------------------------------------------------------------ estimators, elapsed-time updates
class Estimator:
    def __init__(self, name: str, expire_g3_s: float | None = None):
        self.name = name
        self.p = EST.get(name if name != "G4.1" else "G4", {})
        self.state, self.z_ref, self.ema, self.last_t, self.n = -1, None, None, None, 0
        self.buf: list[tuple[float, float, float]] = []
        self.g3_fresh_t = None  # time of the last G3 decision made while the grip moved
        self.expire = expire_g3_s
        if name in ("G4", "G4.1"):
            self.g2, self.g3 = Estimator("G2"), Estimator("G3")
        self.trace = {}

    def update(self, t: float, tip_z: float, paper_z: float, grip_z: float) -> int:
        n = self.name
        if n in ("G4", "G4.1"):
            a = self.g2.update(t, tip_z, paper_z, grip_z)
            b = self.g3.update(t, tip_z, paper_z, grip_z)
            age = None if self.g3.g3_fresh_t is None else t - self.g3.g3_fresh_t
            held = b != -1 and self.g3.trace.get("held", False)
            use_g3 = b != -1 and (n == "G4" or age is None or age <= self.expire)
            self.state = b if use_g3 else a
            self.trace = {"g2": a, "g3": b, "g3_held": held, "g3_age": age, "used_g3": use_g3}
            return self.state
        if n == "G0":
            if self.z_ref is None:
                self.z_ref = paper_z
            self.state = int(tip_z - self.z_ref <= CONTACT_OFFSET_M + self.p["margin_m"])
            return self.state
        r = tip_z - paper_z - CONTACT_OFFSET_M
        if n == "G2":
            if self.ema is None:
                self.ema = r
            else:
                a = 1 - math.exp(-(t - self.last_t) / self.p["tau_s"])
                self.ema += a * (r - self.ema)
            self.last_t = t
            self.n += 1
            if self.n < self.p["min_frames"]:
                return -1
            r = self.ema
        if n in ("G1", "G2"):
            if r <= self.p["on_m"]:
                self.state = 1
            elif r >= self.p["off_m"]:
                self.state = 0
            return self.state
        if n == "G3":
            self.buf.append((t, tip_z, grip_z))
            w = self.p["window_s"]
            self.buf = [x for x in self.buf if x[0] >= t - w - 1e-9]
            self.trace = {"held": False}
            if len(self.buf) < 5:
                self.trace["held"] = self.state != -1
                return self.state
            T = np.array([x[0] for x in self.buf])
            Z = np.array([x[1] for x in self.buf])
            G = np.array([x[2] for x in self.buf])
            Tc = T - T.mean()
            sxx = float((Tc ** 2).sum())
            gs = float((Tc * (G - G.mean())).sum() / sxx)
            if gs > self.p["grip_min_m_s"]:
                self.state, self.g3_fresh_t = 0, t
                return self.state
            if gs > -self.p["grip_min_m_s"]:
                self.trace["held"] = self.state != -1  # the grip is still: the last decision is held
                return self.state
            zs = float((Tc * (Z - Z.mean())).sum() / sxx)
            res = Z - Z.mean() - zs * Tc
            se = float(np.sqrt((res ** 2).sum() / max(len(Z) - 2, 1) / sxx))
            if se > self.p["max_se_ratio"] * abs(gs):
                self.state = -1
                return self.state
            ratio = zs / gs
            self.state = 1 if ratio < self.p["stall_ratio"] else (0 if ratio > self.p["follow_ratio"] else -1)
            if self.state != -1:
                self.g3_fresh_t = t
            return self.state
        raise ValueError(n)


# ------------------------------------------------------------------ ink adapter (shared by ideal and pixel ink)
class InkAdapter:
    def __init__(self):
        self.state, self.t_evidence, self.adv_at_event = -1, None, None

    def reset_stroke(self):
        self.state, self.t_evidence, self.adv_at_event = -1, None, None

    def frame(self, t: float, event: bool, advanced_m: float) -> int:
        if event:
            self.state, self.t_evidence, self.adv_at_event = 1, t, advanced_m
            return self.state
        if self.t_evidence is None:
            return self.state  # before any event in the stroke: unknown
        if advanced_m - self.adv_at_event >= ADAPTER["ADVANCE_M"]:
            self.state = 0  # commanded to advance without new ink: evidence of no contact
            return self.state
        if t - self.t_evidence > ADAPTER["EXPIRE_S"]:
            self.state = -1
        return self.state


# ------------------------------------------------------------------ sources
EVENT_KINDS = ("ink", "pixink", "ink_v11", "pixink_v11")


class Source:
    """truth | sensor | fpr | fnr | flicker | delay | flicker_descent | flicker_drawing (state sources read at the
    control rate from the true force) | geo:<G>:<structure>:<scale>[:tau=<s>][:plane=flat|fit] | ink | pixink |
    ink_v11 (frame sources: consume the 30 Hz frames)."""

    def __init__(self, kind: str, level, seed: int, paper_z: float, scene: dict):
        self.kind, self.level, self.seed = kind, level, seed
        self.rng = np.random.default_rng(7_200_000 + seed)
        self.buf: list = []
        self.paper_z = paper_z
        self.scene = scene
        self.cur, self.t_evidence, self.next_frame = -1, None, 0
        base = kind.split(":")[0]
        self.base = base
        self.label_kind = "event" if base in EVENT_KINDS else "state"
        if base in ("ink", "pixink", "ink_v11", "pixink_v11"):
            v = np.random.default_rng(5_000_011 + seed).normal(size=3)
            self.bias = v / np.linalg.norm(v) * 0.005
            self.jr = np.random.default_rng(7_300_000 + seed)
            self.adapter = InkAdapter()
            self.last_marks = None
        if base == "pixink":
            import pixel_ink as X

            self.X = X
            self.det = X.PixelInk()
            self.cam = None
        if base == "geo":
            parts = kind.split(":")
            g, st, sc = parts[1], parts[2], parts[3]
            opts = dict(p.split("=") for p in parts[4:])
            self.est = Estimator(g, EXPIRE_G3_S if g == "G4.1" else None)
            self.err = ErrorProcess(st, sc, seed, float(opts.get("tau", V11.AR_TAU_S)))
            self.plane = opts.get("plane", "flat")
            self.stroke_start_xy = None

    def new_stroke(self, start_xy):
        if hasattr(self, "adapter"):
            self.adapter.reset_stroke()
        if self.base == "geo":
            self.stroke_start_xy = np.asarray(start_xy)

    def paper_height(self, xy) -> float:
        tilt = self.scene.get("tilt_deg", 0.0)
        return self.paper_z + (xy[0] - 0.46) * math.tan(math.radians(tilt)) if tilt else self.paper_z

    def read(self, t: float, force: float, tip, grip_z: float, fw, first_touch_done: bool) -> tuple[int, float]:
        """(label, t_evidence)."""
        tr = int(force > TOUCH_N)
        k, lv = self.base, self.level
        if k in ("truth", "sensor", "fpr", "fnr", "flicker", "flicker_descent", "flicker_drawing"):
            if k == "truth":
                v = tr
            elif k == "sensor":
                v = int(max(0.0, force + float(self.rng.normal(0, PB.SENSOR["noise_sd_n"]))) > PB.SENSOR["touch_threshold_n"])
            elif k == "fpr":
                v = 1 if (tr == 0 and self.rng.uniform() < lv) else tr
            elif k == "fnr":
                v = 0 if (tr == 1 and self.rng.uniform() < lv) else tr
            elif k == "flicker":
                v = 1 - tr if self.rng.uniform() < lv else tr
            elif k == "flicker_descent":
                v = 1 - tr if (not first_touch_done and self.rng.uniform() < lv) else tr
            else:
                v = 1 - tr if (first_touch_done and self.rng.uniform() < lv) else tr
            return v, t
        if k == "delay":
            self.buf.append((t, tr))
            past = [(t_, x) for t_, x in self.buf if t_ <= t - lv + 1e-9]
            return (past[-1][1], past[-1][0]) if past else (self.buf[0][1], self.buf[0][0])
        frames = fw.obs
        while self.next_frame < len(frames) and frames[self.next_frame]["t"] <= t + 1e-9:
            fr = frames[self.next_frame]
            self.next_frame += 1
            self._frame(fr, fw)
        return self.cur, (self.t_evidence if self.t_evidence is not None else t)

    def _frame(self, fr: dict, fw):
        t = fr["t"]
        if self.base == "geo":
            te, pe = self.err.frame(t)
            tip = np.asarray(fr["tip"])
            ref_xy = tip[:2] if self.plane == "fit" else (self.stroke_start_xy if self.stroke_start_xy is not None
                                                           else tip[:2])
            paper_hat = self.paper_height(ref_xy) + pe
            self.cur = self.est.update(t, float(tip[2] + te[2]), paper_hat, fr["grip_z"])
            self.t_evidence = t
            return
        est = np.asarray(fr["tip"]) + self.bias + self.jr.normal(0, 0.002, size=3)
        if self.base in ("ink", "ink_v11"):
            a = fr["marks_prev_frame"]
            b = fr["marks"]
            event = False
            if b > a:
                new = np.array([m[0][:2] for m in fw.marks[a:b]])
                event = bool((np.linalg.norm(new - est[:2], axis=1) <= V11.PARAMS["INK_RADIUS_M"]).any())
        else:
            event = fr.get("pixel_event") == 1
            if fr.get("pixel_event") == -1 and self.adapter.state == -1:
                pass
        if self.base == "pixink_v11":  # Turn 11's pixel source semantics: the detector's frame label as the state
            self.cur = int(fr.get("pixel_event", -1))
            self.t_evidence = t
            return
        if self.base == "ink_v11":
            self.cur = int(event)
        else:
            self.cur = self.adapter.frame(t, event, fr["s_cmd_total"])
        if event:
            self.t_evidence = t


# ------------------------------------------------------------------ one run
class PressedTooHard(Exception):
    pass


class StrokeTimeout(Exception):
    pass


class ToolDislodged(Exception):
    pass


def apply_scene(f, scene: dict):
    """Tilt: the paper is a static world geom, whose pose MuJoCo fixes at compile time (changing geom_quat on the
    compiled model does nothing), so the scene is recompiled through MjSpec with the paper rotated about the robot
    y axis through its centre, and the firmware's model and data are replaced before the controller is built.
    Stiffer grip: friction is read at every step, so it is scaled on the compiled model."""
    import mujoco

    if scene.get("tilt_deg"):
        spec = mujoco.MjSpec.from_file(str(Path(os.getcwd()) / "scene.xml"))
        th = math.radians(scene["tilt_deg"])
        q = [math.cos(th / 2), 0.0, -math.sin(th / 2), 0.0]  # surface rises with x at slope tan(th)
        R = np.array([[math.cos(th), 0.0, -math.sin(th)], [0.0, 1.0, 0.0], [math.sin(th), 0.0, math.cos(th)]])
        pivot = np.array(spec.geom("paper").pos, dtype=float)
        for gname in ("paper", "pad"):  # the paper and the pad under it tilt together, as one writing board
            g = spec.geom(gname)
            g.pos = (pivot + R @ (np.array(g.pos, dtype=float) - pivot)).tolist()
            g.quat = q
        m2 = spec.compile()
        assert m2.nq == f.model.nq and m2.ngeom == f.model.ngeom
        f.model, f.data = m2, mujoco.MjData(m2)
        mujoco.mj_forward(f.model, f.data)
    m = f.model
    if scene.get("grip_friction_scale"):
        import mujoco

        pb = m.body("pencil").id
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] == pb and m.geom(gid).name != "graphite_tip":
                m.geom_friction[gid, 0] *= scene["grip_friction_scale"]
        for gid in range(m.ngeom):
            if m.body(m.geom_bodyid[gid]).name.startswith("rh_"):
                m.geom_friction[gid, 0] *= scene["grip_friction_scale"]
        mujoco.mj_forward(m, mujoco.MjData(m))


def run_one(seed: int, kind: str = "truth", level=None, keep: str | None = None, target_name: str | None = None,
            params: dict | None = None, scene: dict | None = None, shadows: tuple = (), script: dict | None = None,
            render_pixels: bool = False) -> dict:
    os.chdir(PLAYGROUND / "experiments" / "dove-drawing")
    sys.path.insert(0, os.getcwd())
    from controller import Controller
    from firmware import PAPER_Z
    from run import RecordingPhysical
    from writer import PhysicalWriter, relative

    scene = scene or {}
    P = dict(PARAMS, **(params or {}))
    name, strokes = load_target(write_target(seed, target_name))
    need_pix = render_pixels or kind in ("pixink", "pixink_v11") or any(s in ("pixink", "pixink_v11") for s in shadows)

    class ObsPhysical(RecordingPhysical):
        def __init__(self):
            super().__init__()
            self.obs: list[dict] = []
            self.k_next = 0
            self.grip_z, self.s_cmd_total, self.stroke = 0.0, 0.0, -1
            self.render_cb = None

        def step(self):
            force = super().step()
            tnow = float(self.data.time)
            if tnow + 1e-9 >= self.k_next * FRAME_DT:
                self.k_next = int(math.floor(tnow / FRAME_DT + 1e-9)) + 1
                prev = self.obs[-1]["marks"] if self.obs else 0
                fr = {"t": round(tnow, 5), "tip": self.data.site_xpos[self.tip_site].copy().tolist(),
                      "force": float(force), "marks": len(self.marks), "marks_prev_frame": prev,
                      "grip_z": float(self.grip_z), "s_cmd_total": float(self.s_cmd_total), "stroke": self.stroke}
                if self.render_cb is not None:
                    fr["pixel_event"], fr["observable"] = self.render_cb(self, fr)
                self.obs.append(fr)
            return force

    f = ObsPhysical()
    apply_scene(f, scene)
    c = Controller(f)
    pw = PhysicalWriter(c)
    src = Source(kind, level, seed, PAPER_Z, scene)
    shadow_src = [Source(s, None, seed, PAPER_Z, scene) for s in shadows]
    if need_pix:
        import pixel_ink as X

        cam = X.Camera(f.model)
        det = X.PixelInk()
        v = np.random.default_rng(5_000_011 + seed).normal(size=3)
        pbias = v / np.linalg.norm(v) * 0.005
        pjr = np.random.default_rng(7_300_000 + seed)

        def render_cb(fw, fr):
            rgb = cam.render(fw.data, [(m[0],) for m in fw.marks])
            est = np.asarray(fr["tip"]) + pbias + pjr.normal(0, 0.002, size=3)
            lab, o = det.label(rgb, cam, est)
            return lab, o

        f.render_cb = render_cb
    dt = P["DT_S"]
    log: list[list] = []
    peak = [0.0]
    events: list[dict] = []
    ref_pose: dict = {}
    dis_since = [None]

    def pose_check(t):
        pos, ax = relative(f)
        if not ref_pose:
            ref_pose.update(p=pos.copy(), a=ax / np.linalg.norm(ax))
        a0 = ref_pose["a"]
        tilt = math.degrees(math.acos(float(np.clip(ax @ a0 / np.linalg.norm(ax), -1, 1))))
        slide = float((pos - ref_pose["p"]) @ a0)
        dis = tilt > DISLODGED["TILT_DEG"] or abs(slide) > DISLODGED["SLIDE_M"]
        if dis:
            dis_since[0] = t if dis_since[0] is None else dis_since[0]
            if DISLODGED["EARLY_STOP"] and t - dis_since[0] >= DISLODGED["HOLD_S"]:
                raise ToolDislodged(f"tool dislodged since t = {dis_since[0]:.2f} s")
        else:
            dis_since[0] = None
        return tilt, slide, dis

    def step(grip, seedq):
        f.grip_z = float(grip[2])
        q, _ = c.solve(grip, pw.R, seedq)
        c.command({f.names[i]: float(q[i]) for i in range(9)}, dt)
        force = pw.contact_force()
        peak[0] = max(peak[0], force)
        if force > P["FORCE_LIMIT_N"]:
            raise PressedTooHard(f"true tip force {force:.2f} N at t = {f.data.time:.2f} s")
        return q, force

    def draw_stroke(si, pts):
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        total = float(cum[-1])

        def point(s):
            s = min(max(s, 0.0), total)
            j = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            u = (s - cum[j]) / max(seg[j], 1e-12)
            return pts[j] + u * (pts[j + 1] - pts[j])

        pw.goto([*pts[0], PAPER_Z + P["HOVER_M"] + CONTACT_OFFSET_M], 1.2)
        grip, _ = c.pose()
        f.grip_z, f.stroke = float(grip[2]), si
        for s_ in [src] + shadow_src:
            s_.new_stroke(pts[0])
        seedq = f.data.qpos[:9].copy()
        phase, s, miss, preload_left = "down", 0.0, 0, 0.0
        first_touch = False
        pause_left, lift_left, lift_rise, lift_done = 0.0, 0.0, 0.0, set()
        t_end = f.data.time + total / P["SPEED_M_S"] * P["STROKE_TIMEOUT_FACTOR"] + 10.0
        force = pw.contact_force()
        while True:
            if f.data.time > t_end:
                raise StrokeTimeout(f"stroke {si} not done by t = {f.data.time:.1f} s")
            t = float(f.data.time)
            tip = pw.tip()
            tr = int(force > TOUCH_N)
            first_touch = first_touch or bool(tr)
            lab, t_ev = src.read(t, force, tip, float(grip[2]), f, first_touch)
            sh = [x.read(t, force, tip, float(grip[2]), f, first_touch) for x in shadow_src]
            tilt, slide, dis = pose_check(t)
            extra = []
            if src.base == "geo" and src.est.name in ("G4", "G4.1"):
                tr_ = src.est.trace
                extra = [tr_.get("g2", -9), tr_.get("g3", -9), int(bool(tr_.get("g3_held"))),
                         -1.0 if tr_.get("g3_age") is None else round(tr_["g3_age"], 3), int(bool(tr_.get("used_g3")))]
            log.append([round(t, 3), si, {"down": 0, "preload": 1, "draw": 2}[phase], round(force, 4), tr, lab,
                        round(t - t_ev, 4), len(f.marks), round(float(tip[2]) - PAPER_Z, 5), round(s / total, 4),
                        round(tilt, 2), round(slide, 5), int(dis)] + [x[0] for x in sh] + extra)
            desired = point(s)
            # scripted perturbations (adapter tests): a pause holds the pen still in contact; a lift raises it
            if script and phase == "draw":
                for i, (frac, kind_, dur) in enumerate(script.get("events", [])):
                    if (si, i) not in lift_done and s / total >= frac:
                        lift_done.add((si, i))
                        events.append({"t": round(t, 3), "stroke": si, "event": f"script_{kind_}", "dur": dur})
                        if kind_ == "pause":
                            pause_left = dur
                        else:
                            lift_left, lift_rise = dur, 0.0
            if pause_left > 0:
                pause_left -= dt
            elif lift_left > 0:  # a real contact loss: the pen rises 10 mm/s while the path keeps advancing
                grip[2] += 0.01 * dt if lift_rise < 0.003 else 0.0
                lift_rise += 0.01 * dt
                new = point(s + P["SPEED_M_S"] * dt)
                grip[:2] += (new - desired)
                s += P["SPEED_M_S"] * dt
                f.s_cmd_total += P["SPEED_M_S"] * dt
                desired = new
                lift_left -= dt
                if lift_left <= 0:
                    phase = "down"
                    events.append({"t": round(t, 3), "stroke": si, "event": "script_lift_end"})
            elif phase == "down":
                if lab == 1:
                    phase, preload_left = "preload", P["PRELOAD_M"]
                    events.append({"t": round(t, 3), "stroke": si, "event": "touch_declared", "truth_touching": tr})
                else:
                    grip[2] -= P["DOWN_M_S"] * dt
            if pause_left <= 0 and lift_left <= 0:
                if phase == "preload":
                    d = min(P["DOWN_M_S"] * dt, preload_left)
                    grip[2] -= d
                    preload_left -= d
                    if preload_left <= 1e-12:
                        phase, miss = "draw", 0
                        events.append({"t": round(t, 3), "stroke": si, "event": "stroke_start",
                                       "truth_touching": tr, "started_in_air": not tr})
                elif phase == "draw":
                    if lab == 1:
                        miss = 0
                        new = point(s + P["SPEED_M_S"] * dt)
                        grip[:2] += (new - desired)
                        s += P["SPEED_M_S"] * dt
                        f.s_cmd_total += P["SPEED_M_S"] * dt
                        desired = new
                    elif lab == 0:
                        miss += 1
                        if miss >= P["MISS_READS"]:
                            phase = "down"
                            events.append({"t": round(t, 3), "stroke": si, "event": "lost_contact_redescent",
                                           "truth_touching": tr})
            grip[:2] += np.clip(0.15 * (desired - tip[:2]), -0.00015, 0.00015)
            seedq, force = step(grip, seedq)
            if s >= total - 1e-9:
                break
        pw.goto(pw.tip() + [0, 0, P["HOVER_M"]], 0.6)
        f.stroke = -1

    t0 = time.monotonic()
    completed, error, cause = True, None, None
    mark_start = []
    try:
        c.command({}, 1)
        pw.goto([*strokes[0][0], 0.794], 2.5)
        for si, pts in enumerate(strokes):
            mark_start.append(len(f.marks))
            draw_stroke(si, pts)
        mark_start.append(len(f.marks))
        pw.goto(pw.tip() + [0, 0, 0.02], 1)
    except PressedTooHard as ex:
        completed, error, cause = False, str(ex), "pressed too hard"
    except StrokeTimeout as ex:
        completed, error, cause = False, str(ex), "stroke timeout"
    except ToolDislodged as ex:
        completed, error, cause = False, str(ex), "tool dislodged (early stop)"
    except Exception as ex:  # noqa: BLE001
        completed, error, cause = False, f"{type(ex).__name__}: {ex}", "controller error"
    wall = round(time.monotonic() - t0, 1)
    while len(mark_start) < len(strokes) + 1:
        mark_start.append(len(f.marks))
    ink = np.array([m[0][:2] for m in f.marks]) if f.marks else np.zeros((0, 2))
    dense = [PB.densify(s) for s in strokes]
    cov = PB.coverage(np.vstack(dense), ink)
    off = float(np.mean(PB.seg_dist(ink, strokes) > PB.OFF_PATH_M)) if len(ink) else 0.0
    L = np.array(log, dtype=float) if log else np.zeros((0, 13))
    known = L[:, 5] != -1 if len(L) else np.zeros(0, bool)
    success = bool(completed and cov >= PB.COVERAGE_MIN and off <= PB.OFF_PATH_MAX)
    dislodged_ever = bool(L[:, 12].any()) if len(L) else False
    if success:
        failure = None
    elif cause in ("stroke timeout",) and dislodged_ever and L[-1, 12]:
        failure = "tool dislodged"
    elif cause == "stroke timeout":
        failure = "stroke timeout, tool held"
    else:
        failure = cause or ("coverage" if cov < PB.COVERAGE_MIN else "off-path ink")
    ft = np.diff([o["t"] for o in f.obs]) if len(f.obs) > 1 else np.zeros(0)
    in_stroke = np.array([o["stroke"] >= 0 for o in f.obs[1:]]) if len(f.obs) > 1 else np.zeros(0, bool)
    out = {"seed": seed, "target": name, "source": kind, "level": level, "scene": scene, "version": "v11.1",
           "label_kind": src.label_kind, "absence_rule": ABSENCE_RULE.get("ink") if src.base in ("ink", "pixink") else None,
           "success": success, "completed": completed, "failure": failure, "cause_raw": cause, "error": error,
           "coverage": round(cov, 4), "off_path_ink": round(off, 4), "marks": int(len(ink)),
           "peak_force_n": round(peak[0], 3), "sim_s": round(float(f.data.time), 2), "wall_s": wall,
           "strokes": len(strokes), "redescents": sum(1 for e in events if e["event"] == "lost_contact_redescent"),
           "strokes_started_in_air": sum(1 for e in events if e["event"] == "stroke_start" and e["started_in_air"]),
           "read_accuracy": round(float(np.mean(L[:, 5] == L[:, 4])), 4) if len(L) else None,
           "unknown_rate": round(float(np.mean(~known)), 4) if len(L) else None,
           "evidence_age_median_s": round(float(np.median(L[:, 6])), 4) if len(L) else None,
           "dislodged_ever": dislodged_ever, "tilt_max_deg": round(float(L[:, 10].max()), 2) if len(L) else None,
           "frame_dt_in_strokes": {"n": int(in_stroke.sum()), "min": round(float(ft[in_stroke].min()), 5) if in_stroke.any() else None,
                                   "max": round(float(ft[in_stroke].max()), 5) if in_stroke.any() else None,
                                   "mean": round(float(ft[in_stroke].mean()), 5) if in_stroke.any() else None},
           "per_stroke": [{"stroke": i, "marks": int(mark_start[i + 1] - mark_start[i]),
                           "coverage": round(PB.coverage(d, ink), 3)} for i, d in enumerate(dense)]}
    if keep:
        kd = Path(keep)
        kd.mkdir(parents=True, exist_ok=True)
        tag = f"pen111_{seed}_{target_name or 'seed'}_{kind.replace(':', '-').replace('=', '')}_{level}" + \
              ("_" + "_".join(f"{k}{v}" for k, v in scene.items()) if scene else "") + ("_script" if script else "")
        np.savez_compressed(kd / f"{tag}.npz", times=np.array([r[0] for r in f.frames]),
                            qpos=np.array([r[1] for r in f.frames]), mark_counts=np.array([r[2] for r in f.frames]),
                            marks=np.array([[*p, *q, v] for p, q, v in f.marks]) if f.marks else np.zeros((0, 7)),
                            reads=L, obs=np.array(json.dumps(f.obs)),
                            strokes=np.array(json.dumps([s.tolist() for s in strokes])),
                            events=np.array(json.dumps(events)), shadows=np.array(json.dumps(list(shadows))))
        (kd / f"{tag}.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ grid
def _job(args):
    seed, cell = args
    try:
        return run_one(seed, cell["source"], cell.get("level"), scene=cell.get("scene"))
    except Exception as ex:  # noqa: BLE001
        return {"seed": seed, "source": cell["source"], "level": cell.get("level"), "scene": cell.get("scene"),
                "success": False, "error": f"crash {type(ex).__name__}: {ex}"}


def cell_key(r):
    return (r["seed"], r["source"], r.get("level"), json.dumps(r.get("scene") or {}, sort_keys=True))


def grid(out_path: str, cells: list[dict], seeds, processes: int = 16, s_per_run: float = 30.0) -> None:
    from multiprocessing import Pool

    done = set()
    if Path(out_path).exists():
        done = {cell_key(json.loads(x)) for x in Path(out_path).read_text().splitlines()}
    jobs = [(s, c) for c in cells for s in seeds
            if (s, c["source"], c.get("level"), json.dumps(c.get("scene") or {}, sort_keys=True)) not in done]
    print(json.dumps({"grid": f"pen v11.1 {Path(out_path).stem}", "runs": len(jobs),
                      "expected_runtime_min": round(len(jobs) * s_per_run / processes / 60, 1)}), flush=True)
    t0 = time.monotonic()
    with Pool(processes) as pool, open(out_path, "a") as fh:
        for r in pool.imap_unordered(_job, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(json.dumps({"wall_min": round((time.monotonic() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    if cmd == "grid":
        a, b, st = (int(x) for x in arg("--seeds", "0:20:1").split(":"))
        grid(sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), range(a, b, st), int(arg("--processes", "16")),
             float(arg("--s-per-run", "30")))
    elif cmd == "one":
        lv = arg("--level")
        print(json.dumps(run_one(int(sys.argv[2]), sys.argv[3], float(lv) if lv else None, arg("--keep"),
                                 arg("--target"))), flush=True)
