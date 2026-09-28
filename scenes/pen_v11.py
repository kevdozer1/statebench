"""Turn 11: controller v11 (one Boolean touching interface for every source), label sources, declared joint error
structures for the tip and the paper plane, and geometry estimators G0-G4. Playground venv only.

``<playground python> pen_v11.py one <seed> <source> [--level L] [--target NAME] [--keep DIR]``
``<playground python> pen_v11.py grid <out.jsonl> <cells.json> [--seeds a:b:step] [--processes N]``

The study is contact-source substitution inside an otherwise privileged controller: the path-following correction
reads the exact simulator tip (as PhysicalWriter.draw does). The playground's Controller, PhysicalWriter (goto,
tip, true tip-paper force), firmware and RecordingPhysical are imported unchanged.

Controller v11 (every 20 ms control step; PARAMS):
* descent: from HOVER_M above contact height, descend at DOWN_M_S on "not touching" or "unknown"; on "touching"
  start the stroke.
* preload: on "touching" (the first of a stroke, and again after every re-descent) descend a further PRELOAD_M at
  DOWN_M_S, ignoring the label, then hold that height (no force regulation).
* during a stroke: advance along the path only on "touching"; hold position on "unknown" (no progress, no descent)
  and on "not touching" reads short of MISS_READS consecutive; after MISS_READS consecutive "not touching" reads,
  stop and descend again.
* pen up at stroke done (truth: path progress reached the stroke end); a true force above FORCE_LIMIT_N fails the
  run ("pressed too hard"); a stroke not done within STROKE_TIMEOUT_FACTOR x nominal + 10 s fails ("stroke timeout").

Every source returns touching (1), not touching (0) or unknown (-1). Onset records carry t_event (truth contact
onset), t_evidence (the source's first "touching" at or after it) and online=True (closed loop is always online).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_bench as PB  # noqa: E402
from targets import load_target, write_target  # noqa: E402

PLAYGROUND = PB.PLAYGROUND
PARAMS = {"HOVER_M": 0.003, "DOWN_M_S": 0.002, "PRELOAD_M": 0.001, "SPEED_M_S": 0.008, "MISS_READS": 3,
          "FORCE_LIMIT_N": 2.0, "DT_S": 0.02, "STROKE_TIMEOUT_FACTOR": 3.0, "CAMERA_HZ": 30.0,
          "INK_RADIUS_M": 0.008}
TOUCH_N = PB.TOUCH_N
CONTACT_OFFSET_M = PB.CONTACT_OFFSET_M  # tip-site height above the paper at true contact (Turn 9, 0.60 mm)

# ------------------------------------------------------------------ error structures (per axis, per object)
SCALES = {"L1": 0.0015, "L2": 0.0035}  # per-axis RMS error of each object's position estimate (m)
STRUCTURES = {
    # fractions of the per-object variance sigma^2
    "S": {"shared_bias": 0.9, "own_bias": 0.0, "ar": 0.0, "jitter": 0.1},
    "D": {"shared_bias": 0.0, "own_bias": 0.9, "ar": 0.0, "jitter": 0.1},
    "J": {"shared_bias": 0.0, "own_bias": 0.1, "ar": 0.0, "jitter": 0.9},
    "C": {"shared_bias": 0.0, "own_bias": 0.0, "ar": 0.9, "jitter": 0.1},
}
AR_TAU_S = 0.5
FRAME_DT = 1.0 / 30.0


class ErrorProcess:
    """Per-episode draws for the tip (3 axes) and the paper plane (height), 30 Hz frames."""

    def __init__(self, structure: str, scale: str, seed: int):
        self.st, self.sig = STRUCTURES[structure], SCALES[scale]
        self.rng = np.random.default_rng(8_100_000 + seed)
        s2 = self.sig ** 2
        shared = self.rng.normal(0, np.sqrt(self.st["shared_bias"] * s2), size=3)
        self.bias_tip = shared + self.rng.normal(0, np.sqrt(self.st["own_bias"] * s2), size=3)
        self.bias_paper = shared[2] + self.rng.normal(0, np.sqrt(self.st["own_bias"] * s2))
        self.ar_tip = self.rng.normal(0, np.sqrt(self.st["ar"] * s2), size=3)
        self.ar_paper = self.rng.normal(0, np.sqrt(self.st["ar"] * s2))
        self.phi = float(np.exp(-FRAME_DT / AR_TAU_S))

    def frame(self) -> tuple[np.ndarray, float]:
        s2 = self.sig ** 2
        if self.st["ar"] > 0:
            k = np.sqrt(self.st["ar"] * s2 * (1 - self.phi ** 2))
            self.ar_tip = self.phi * self.ar_tip + self.rng.normal(0, k, size=3)
            self.ar_paper = self.phi * self.ar_paper + self.rng.normal(0, k)
        j = np.sqrt(self.st["jitter"] * s2)
        tip_err = self.bias_tip + self.ar_tip + self.rng.normal(0, j, size=3)
        paper_err = self.bias_paper + self.ar_paper + self.rng.normal(0, j)
        return tip_err, float(paper_err)


# ------------------------------------------------------------------ geometry estimators (30 Hz frames)
EST = {
    "G0": {"margin_m": 0.000125},
    "G1": {"on_m": 0.0003, "off_m": 0.0010},
    "G2": {"tau_s": 0.2, "on_m": 0.0003, "off_m": 0.0010, "min_frames": 3},
    "G3": {"window_s": 1.5, "grip_min_m_s": 0.001, "stall_ratio": 0.4, "follow_ratio": 0.7, "max_se_ratio": 0.6},
    "G4": {"rule": "G3 when it is definite, else G2"},
}


class Estimator:
    """Online by default. Reads only noisy tip and paper observations and the commanded grip height."""

    def __init__(self, name: str):
        self.name, self.p = name, EST[name]
        self.state = -1
        self.z_ref = None
        self.ema = None
        self.n = 0
        self.buf: list[tuple[float, float, float]] = []  # (t, tip_z_hat, grip_z)
        self.g2 = Estimator("G2") if name == "G4" else None
        self.g3 = Estimator("G3") if name == "G4" else None

    def update(self, t: float, tip_z: float, paper_z: float, grip_z: float) -> int:
        n = self.name
        if n == "G4":
            a = self.g2.update(t, tip_z, paper_z, grip_z)
            b = self.g3.update(t, tip_z, paper_z, grip_z)
            self.state = b if b != -1 else a
            return self.state
        if n == "G0":
            if self.z_ref is None:
                self.z_ref = paper_z  # a one-time calibration of the paper height, then static
            self.state = int(tip_z - self.z_ref <= CONTACT_OFFSET_M + self.p["margin_m"])
            return self.state
        r = tip_z - paper_z - CONTACT_OFFSET_M
        if n == "G2":
            a = 1 - np.exp(-FRAME_DT / self.p["tau_s"])
            self.ema = r if self.ema is None else self.ema + a * (r - self.ema)
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
            self.buf = [b for b in self.buf if b[0] >= t - w - 1e-9]
            if len(self.buf) < 5:
                return self.state
            T = np.array([b[0] for b in self.buf])
            Z = np.array([b[1] for b in self.buf])
            G = np.array([b[2] for b in self.buf])
            Tc = T - T.mean()
            sxx = float((Tc ** 2).sum())
            gs = float((Tc * (G - G.mean())).sum() / sxx)
            if gs > self.p["grip_min_m_s"]:
                self.state = 0  # the grip is rising: the pen is being lifted
                return self.state
            if gs > -self.p["grip_min_m_s"]:
                return self.state  # grip not moving: hold the last decision
            zs = float((Tc * (Z - Z.mean())).sum() / sxx)
            res = Z - Z.mean() - zs * Tc
            se = float(np.sqrt((res ** 2).sum() / max(len(Z) - 2, 1) / sxx))
            if se > self.p["max_se_ratio"] * abs(gs):
                self.state = -1  # too noisy to tell stall from following
                return self.state
            ratio = zs / gs
            if ratio < self.p["stall_ratio"]:
                self.state = 1
            elif ratio > self.p["follow_ratio"]:
                self.state = 0
            else:
                self.state = -1
            return self.state
        raise ValueError(n)


# ------------------------------------------------------------------ sources
class Source:
    """kind: truth | sensor | static_L1 | static_L2 | ink | fpr | fnr | flicker | delay | flicker_descent |
    flicker_drawing | geo:<G>:<structure>:<scale>"""

    def __init__(self, kind: str, level: float | None, seed: int, paper_z: float):
        self.kind, self.level = kind, level
        self.rng = np.random.default_rng(7_200_000 + seed)
        self.buf: list[tuple[float, int]] = []
        self.paper_z = paper_z
        self.frame_t = -1.0
        self.cur = -1
        if kind in ("static_L1", "static_L2", "ink"):
            g = PB.GEOMETRY["L1" if kind == "static_L1" else "L2"]
            v = np.random.default_rng(5_000_011 + seed).normal(size=3)
            self.bias = v / np.linalg.norm(v) * g["bias_mm"] / 1000
            self.jitter = g["jitter_mm_sd"] / 1000
        if kind.startswith("geo:"):
            _, g, st, sc = kind.split(":")
            self.est = Estimator(g)
            self.err = ErrorProcess(st, sc, seed)

    def read(self, t: float, force: float, tip: np.ndarray, grip_z: float, fw, first_touch_done: bool) -> int:
        tr = int(force > TOUCH_N)
        k, lv = self.kind, self.level
        if k == "truth":
            return tr
        if k == "sensor":
            return int(max(0.0, force + float(self.rng.normal(0, PB.SENSOR["noise_sd_n"]))) > PB.SENSOR["touch_threshold_n"])
        if k == "fpr":
            return 1 if (tr == 0 and self.rng.uniform() < lv) else tr
        if k == "fnr":
            return 0 if (tr == 1 and self.rng.uniform() < lv) else tr
        if k == "flicker":
            return 1 - tr if self.rng.uniform() < lv else tr
        if k == "flicker_descent":  # flips only before the first true touch of the stroke
            return 1 - tr if (not first_touch_done and self.rng.uniform() < lv) else tr
        if k == "flicker_drawing":  # flips only after the first true touch of the stroke
            return 1 - tr if (first_touch_done and self.rng.uniform() < lv) else tr
        if k == "delay":
            self.buf.append((t, tr))
            past = [x for t_, x in self.buf if t_ <= t - lv + 1e-9]
            return past[-1] if past else self.buf[0][1]
        if k in ("static_L1", "static_L2"):  # Turn 10's rule unchanged (it reads the exact paper height)
            z = float(tip[2]) + float(self.bias[2]) + float(self.rng.normal(0, self.jitter))
            return int(z <= self.paper_z + CONTACT_OFFSET_M + PB.RULE_MARGIN_M)
        if k == "ink":  # Turn 10's ideal ink source unchanged
            est = np.asarray(tip[:2]) + self.bias[:2] + self.rng.normal(0, self.jitter, size=2)
            fr = [x for x in fw.frames if x[0] <= t + 1e-9]
            if len(fr) >= 2:
                a, b = int(fr[-2][2]), int(fr[-1][2])
                if b > a:
                    new = np.array([m[0][:2] for m in fw.marks[a:b]])
                    return int((np.linalg.norm(new - est, axis=1) <= PARAMS["INK_RADIUS_M"]).any())
            return 0
        if k.startswith("geo:"):
            if t + 1e-9 >= self.frame_t + FRAME_DT:  # a new 30 Hz observation frame
                self.frame_t = t
                te, pe = self.err.frame()
                self.cur = self.est.update(t, float(tip[2] + te[2]), self.paper_z + pe, grip_z)
            return self.cur
        raise ValueError(k)


# ------------------------------------------------------------------ one run
class PressedTooHard(Exception):
    pass


class StrokeTimeout(Exception):
    pass


def run_one(seed: int, kind: str = "truth", level: float | None = None, keep: str | None = None,
            target_name: str | None = None, params: dict | None = None, frame_hook=None) -> dict:
    os.chdir(PLAYGROUND / "experiments" / "dove-drawing")
    sys.path.insert(0, os.getcwd())
    from controller import Controller
    from firmware import PAPER_Z
    from run import RecordingPhysical
    from writer import PhysicalWriter

    P = dict(PARAMS, **(params or {}))
    name, strokes = load_target(write_target(seed, target_name))
    f = RecordingPhysical()
    c = Controller(f)
    pw = PhysicalWriter(c)
    src = Source(kind, level, seed, PAPER_Z)
    dt = P["DT_S"]
    log: list[list] = []
    log30: list[list] = []
    peak = [0.0]
    events: list[dict] = []
    onsets: list[dict] = []
    last30 = [-1.0]

    def step(grip, seedq):
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
        seedq = f.data.qpos[:9].copy()
        phase, s, miss, preload_left = "down", 0.0, 0, 0.0
        first_touch = False
        pending_onset = None
        prev_truth = 0
        t_end = f.data.time + total / P["SPEED_M_S"] * P["STROKE_TIMEOUT_FACTOR"] + 10.0
        force = pw.contact_force()
        while True:
            if f.data.time > t_end:
                raise StrokeTimeout(f"stroke {si} not done by t = {f.data.time:.1f} s")
            t = float(f.data.time)
            tip = pw.tip()
            tr = int(force > TOUCH_N)
            if tr and not first_touch:
                first_touch = True
            if tr and not prev_truth:
                pending_onset = {"stroke": si, "t_event": round(t, 3), "t_evidence": None, "online": True}
                onsets.append(pending_onset)
            prev_truth = tr
            lab = src.read(t, force, tip, float(grip[2]), f, first_touch)
            if pending_onset is not None and lab == 1 and pending_onset["t_evidence"] is None:
                pending_onset["t_evidence"] = round(t, 3)
                pending_onset = None
            if t + 1e-9 >= last30[0] + FRAME_DT:
                last30[0] = t
                log30.append([round(t, 4), float(tip[0]), float(tip[1]), float(tip[2]), float(grip[2]), tr,
                              round(force, 4), si, len(f.marks)])
                if frame_hook is not None:
                    frame_hook(f, t)
            log.append([round(t, 3), si, {"down": 0, "preload": 1, "draw": 2}[phase], round(force, 4), tr, lab,
                        len(f.marks), round(float(tip[2]) - PAPER_Z, 5), round(s / total, 4)])
            desired = point(s)
            if phase == "down":
                if lab == 1:
                    phase, preload_left = "preload", P["PRELOAD_M"]
                    events.append({"t": round(t, 3), "stroke": si, "event": "touch_declared", "truth_touching": tr})
                else:
                    grip[2] -= P["DOWN_M_S"] * dt
            if phase == "preload":
                d = min(P["DOWN_M_S"] * dt, preload_left)
                grip[2] -= d
                preload_left -= d
                if preload_left <= 1e-12:
                    phase, miss = "draw", 0
                    events.append({"t": round(t, 3), "stroke": si, "event": "stroke_start", "truth_touching": tr,
                                   "started_in_air": not tr})
            elif phase == "draw":
                if lab == 1:
                    miss = 0
                    new = point(s + P["SPEED_M_S"] * dt)
                    grip[:2] += (new - desired)
                    s += P["SPEED_M_S"] * dt
                    desired = new
                elif lab == 0:
                    miss += 1
                    if miss >= P["MISS_READS"]:
                        phase = "down"
                        events.append({"t": round(t, 3), "stroke": si, "event": "lost_contact_redescent",
                                       "truth_touching": tr})
                # lab == -1: hold position
            grip[:2] += np.clip(0.15 * (desired - tip[:2]), -0.00015, 0.00015)
            seedq, force = step(grip, seedq)
            if s >= total - 1e-9:
                break
        pw.goto(pw.tip() + [0, 0, P["HOVER_M"]], 0.6)

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
    except Exception as ex:  # noqa: BLE001
        completed, error, cause = False, f"{type(ex).__name__}: {ex}", "controller error"
    wall = round(time.monotonic() - t0, 1)
    while len(mark_start) < len(strokes) + 1:
        mark_start.append(len(f.marks))
    ink = np.array([m[0][:2] for m in f.marks]) if f.marks else np.zeros((0, 2))
    dense = [PB.densify(s) for s in strokes]
    cov = PB.coverage(np.vstack(dense), ink)
    off = float(np.mean(PB.seg_dist(ink, strokes) > PB.OFF_PATH_M)) if len(ink) else 0.0
    L = np.array(log, dtype=float) if log else np.zeros((0, 9))
    known = L[:, 5] != -1 if len(L) else np.zeros(0, bool)
    acc = float(np.mean(L[:, 5] == L[:, 4])) if len(L) else None
    acc_known = float(np.mean(L[known, 5] == L[known, 4])) if known.any() else None
    success = bool(completed and cov >= PB.COVERAGE_MIN and off <= PB.OFF_PATH_MAX)
    in_air = sum(1 for e in events if e["event"] == "stroke_start" and e["started_in_air"])
    lat = [o["t_evidence"] - o["t_event"] for o in onsets if o["t_evidence"] is not None]
    out = {"seed": seed, "target": name, "source": kind, "level": level, "success": success, "completed": completed,
           "failure": None if success else (cause or ("coverage" if cov < PB.COVERAGE_MIN else "off-path ink")),
           "error": error, "coverage": round(cov, 4), "off_path_ink": round(off, 4), "marks": int(len(ink)),
           "peak_force_n": round(peak[0], 3), "pressed_too_hard": cause == "pressed too hard",
           "sim_s": round(float(f.data.time), 2), "wall_s": wall,
           "read_accuracy": None if acc is None else round(acc, 4),
           "read_accuracy_known": None if acc_known is None else round(acc_known, 4),
           "unknown_rate": round(float(np.mean(~known)), 4) if len(L) else None, "reads": int(len(L)),
           "strokes_started_in_air": in_air,
           "redescents": sum(1 for e in events if e["event"] == "lost_contact_redescent"),
           "onset_latency_median_s": round(float(np.median(lat)), 3) if lat else None, "onsets": onsets[:40],
           "per_stroke": [{"stroke": i, "marks": int(mark_start[i + 1] - mark_start[i]),
                           "coverage": round(PB.coverage(d, ink), 3)} for i, d in enumerate(dense)]}
    if keep:
        kd = Path(keep)
        kd.mkdir(parents=True, exist_ok=True)
        tag = f"pen11_{seed}_{target_name or 'seed'}_{kind.replace(':', '-')}_{level}"
        np.savez_compressed(kd / f"{tag}.npz", times=np.array([r[0] for r in f.frames]),
                            qpos=np.array([r[1] for r in f.frames]), mark_counts=np.array([r[2] for r in f.frames]),
                            marks=np.array([[*p, *q, v] for p, q, v in f.marks]) if f.marks else np.zeros((0, 7)),
                            reads=L, log30=np.array(log30, dtype=float) if log30 else np.zeros((0, 9)),
                            strokes=np.array(json.dumps([s.tolist() for s in strokes])),
                            events=np.array(json.dumps(events)))
        (kd / f"{tag}.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ grid
def _job(args):
    seed, kind, level = args
    try:
        return run_one(seed, kind, level)
    except Exception as ex:  # noqa: BLE001
        return {"seed": seed, "source": kind, "level": level, "success": False, "error": f"crash {ex}"}


def grid(out_path: str, cells: list, seeds, processes: int = 10, s_per_run: float = 15.0) -> None:
    from multiprocessing import Pool

    jobs = [(s, k, lv) for k, lv in cells for s in seeds]
    done = set()
    if Path(out_path).exists():
        for line in Path(out_path).read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["source"], r["level"]))
    jobs = [j for j in jobs if j not in done]
    print(json.dumps({"grid": f"pen v11 {Path(out_path).stem}", "runs": len(jobs),
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
        cells = [tuple(c) for c in json.loads(Path(sys.argv[3]).read_text())]
        grid(sys.argv[2], cells, range(a, b, st), int(arg("--processes", "10")), float(arg("--s-per-run", "15")))
    elif cmd == "one":
        lv = arg("--level")
        print(json.dumps(run_one(int(sys.argv[2]), sys.argv[3], float(lv) if lv else None, arg("--keep"),
                                 arg("--target"))), flush=True)
