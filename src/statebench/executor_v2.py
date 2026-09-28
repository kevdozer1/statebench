"""Executor v0.2: stale coordinates are registered with the pose that measured them.

The v0.1 executor (``executor.py``, unchanged) turns a relative displacement into a
world target by adding it to the gripper pose *now*::

    target = tcp(now) + rel            yaw = yaw(now) + dyaw

``rel`` was measured at the field's own ``t_obs``. Whenever ``t_obs`` is not now
- under the delay ablation, or on the history fallback path - the arm has moved
since, and the sum mixes two instants. v0.2 keeps a declared proprio history of
TCP position and yaw (source ``proprio``, 2 s horizon, one entry per physics
step) and reconstructs::

    target = tcp(t_obs) + rel          yaw = yaw(t_obs) + dyaw

using each field's own ``t_obs`` from provenance. Outside delay ``t_obs`` equals
the decision time and nothing has moved since, so v0.1 and v0.2 command the same
targets bit for bit. That is the spot check.

Proprioception is the robot's own encoder record, so remembering 2 s of it is not
privileged information. Every other primitive, duration and the fallback policy
are inherited unchanged from ``executor.Executor``.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Any

import numpy as np

from .executor import Executor, _relative, _yaw
from .observations import RECEPTACLE_PRIOR, TABLE_CENTER_PRIOR

EXECUTOR_VERSION = "statebench-executor/v0.2"
PROPRIO_HISTORY_S = 2.0
TARGET_PATH = "geometry.target_rel_gripper"
RECEPTACLE_PATH = "geometry.target_rel_receptacle"


class ProprioHistory:
    """TCP position and yaw per physics step over the last ``seconds``. Source: proprio."""

    source = "proprio"

    def __init__(self, env, seconds: float = PROPRIO_HISTORY_S):
        self.env = env
        self.seconds = float(seconds)
        self._t: deque[float] = deque()
        self._pose: deque[tuple[np.ndarray, float]] = deque()
        self(env)

    def __call__(self, env) -> None:
        t = float(env.t)
        self._t.append(t)
        self._pose.append((env.proprio_tcp(), float(env.proprio_tcp_yaw())))
        # Keep one entry older than the horizon so a lookup at exactly
        # now - seconds still has something at or before it.
        while len(self._t) > 2 and self._t[1] < t - self.seconds - 1e-9:
            self._t.popleft()
            self._pose.popleft()

    def at(self, t_obs: float) -> tuple[np.ndarray, float, str]:
        """Pose at the newest entry at or before ``t_obs``, and how it was found."""
        times = list(self._t)
        i = bisect_right(times, float(t_obs) + 1e-6) - 1
        if i < 0:
            tcp, yaw = self._pose[0]
            return tcp.copy(), yaw, "older_than_horizon"
        tcp, yaw = self._pose[i]
        return tcp.copy(), yaw, "proprio_history"


def _t_obs(state: dict[str, Any], path: str) -> float | None:
    stamps = (state.get("provenance") or {}).get("t_obs") or {}
    value = stamps.get(path)
    return None if value is None else float(value)


class ExecutorV2(Executor):
    """v0.1 primitives with registration of each displacement at its own ``t_obs``."""

    def __init__(self, env, events, perturbation_offset):
        super().__init__(env, events, perturbation_offset)
        self.proprio_history = ProprioHistory(env)
        env.step_hooks.append(self.proprio_history)
        self.registrations: list[dict[str, Any]] = []

    def _pose_for(self, t_obs: float | None) -> tuple[np.ndarray, float, dict[str, Any]]:
        now_tcp = self.env.proprio_tcp()
        now_yaw = float(self.env.proprio_tcp_yaw())
        if t_obs is None:
            return now_tcp, now_yaw, {"t_obs": None, "how": "no_t_obs_use_now", "shift_m": 0.0}
        tcp, yaw, how = self.proprio_history.at(t_obs)
        info = {
            "t_obs": round(t_obs, 3),
            "age_s": round(float(self.env.t) - t_obs, 3),
            "how": how,
            "shift_m": round(float(np.linalg.norm(tcp - now_tcp)), 5),
        }
        return tcp, yaw, info

    def _resolve_target(self, state, history):
        for source, candidate in (("state", state), *(("history", s) for s in reversed(history))):
            block = (candidate.get("geometry") or {}).get("target_rel_gripper")
            rel = _relative(block)
            if rel is None:
                continue
            tcp, tcp_yaw, info = self._pose_for(_t_obs(candidate, TARGET_PATH))
            dyaw = _yaw(block)
            self.registrations.append({"field": TARGET_PATH, "source": source, **info})
            self._last_registration = info
            return tcp + rel, (tcp_yaw + dyaw) if dyaw is not None else None, source
        self._emit("field_missing_fallback")
        return TABLE_CENTER_PRIOR.copy(), None, "prior"

    def _resolve_receptacle(self, state, history):
        for source, candidate in (("state", state), *(("history", s) for s in reversed(history))):
            geometry = candidate.get("geometry") or {}
            rel_g = _relative(geometry.get("target_rel_gripper"))
            rel_r = _relative(geometry.get("target_rel_receptacle"))
            if rel_g is None or rel_r is None:
                continue
            # Both relations are measured at the same instant in every state this
            # project produces; register at target_rel_gripper's t_obs.
            tcp, _, info = self._pose_for(_t_obs(candidate, TARGET_PATH))
            info["t_obs_receptacle_field"] = _t_obs(candidate, RECEPTACLE_PATH)
            self.registrations.append({"field": RECEPTACLE_PATH, "source": source, **info})
            self._last_registration = info
            return tcp + rel_g - rel_r, source
        return RECEPTACLE_PRIOR.copy(), "prior"

    def execute(self, primitive, state, history):
        self._last_registration = None
        record = super().execute(primitive, state, history)
        if self._last_registration is not None:
            record["registration"] = self._last_registration
        return record
