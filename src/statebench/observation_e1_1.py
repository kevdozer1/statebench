"""E1.1: E1 with its track interpolated at the exact motion-window endpoints.

A pure measurement correction, adopted in Turn 6 Phase 2.2 because 2,870 of E1's
3,711 dev-frame motion disagreements with the oracle were shared by truth sampled
at E1's own 10 Hz timestamps (a sampling artifact, not an estimation error).

E1.1 = ``perception_e1`` (unchanged, e37dbc33) + ``observation_e1`` (unchanged)
with the motion relation and the lift event taken from this module: E1's track
(one entry per fresh estimate) is linearly interpolated at exactly t - 0.25 s and t
(t = the newest estimate's time), then ``observations.py``'s thresholds apply
(6 mm minimum TCP motion, 6 mm match; lift: 8 mm and 10 mm). The relation is
``unknown`` when the track does not reach back to t - 0.25.
"""

from __future__ import annotations

import numpy as np

from .observation_e1 import LIFT_MATCH_M, LIFT_MOTION_M, MOTION_MATCH_M, MOTION_MIN_M, MOTION_WINDOW_S

E1_1_VERSION = "statebench-perception-e1.1/v1 (e1 e37dbc33 + interpolated track)"


def interp_track(track: list[dict], t: float, key: str) -> np.ndarray | None:
    if not track or t < track[0]["t"] - 1e-9:
        return None
    for a, b in zip(track, track[1:]):
        if a["t"] - 1e-9 <= t <= b["t"] + 1e-9:
            w = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            return (1 - w) * np.asarray(a[key]) + w * np.asarray(b[key])
    return np.asarray(track[-1][key]) if t >= track[-1]["t"] - 1e-9 else None


def moving_interp(track: list[dict], t_end: float) -> bool | str:
    c0, c1 = interp_track(track, t_end - MOTION_WINDOW_S, "cube_m"), interp_track(track, t_end, "cube_m")
    p0, p1 = interp_track(track, t_end - MOTION_WINDOW_S, "tcp_m"), interp_track(track, t_end, "tcp_m")
    if c0 is None or c1 is None or t_end > track[-1]["t"] + 1e-9:
        return "unknown"
    tcp = p1 - p0
    if float(np.linalg.norm(tcp)) < MOTION_MIN_M:
        return "unknown"
    return bool(float(np.linalg.norm((c1 - c0) - tcp)) < MOTION_MATCH_M)


def lift_diverged_interp(track: list[dict], t0: float, t1: float) -> bool | None:
    c0, c1 = interp_track(track, t0, "cube_m"), interp_track(track, t1, "cube_m")
    p0, p1 = interp_track(track, t0, "tcp_m"), interp_track(track, t1, "tcp_m")
    if c0 is None or c1 is None:
        return None
    tcp = p1 - p0
    if float(np.linalg.norm(tcp)) < LIFT_MOTION_M:
        return None
    return bool(float(np.linalg.norm((c1 - c0) - tcp)) >= LIFT_MATCH_M)
