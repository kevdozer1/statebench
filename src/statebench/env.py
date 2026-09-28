"""Simulation wrapper. This is the ONLY module that touches MuJoCo state directly.

Everything downstream (observations, executor, backends) reads through
``observations.py``, which decides per condition which of these accessors is
allowed. Keeping every privileged read behind one class is what makes the leak
audit in ``notes/turn1.md`` checkable for our own code.

The accessors are grouped and named by provenance:

* ``proprio_*``  - derivable from joint encoders and the robot's own model.
* ``truth_*``    - privileged simulator state.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .ik import HOME, ArmKinematics, wrap_to_half_pi
from .scene import (
    CUBE_HALF,
    GRASP_Z,
    HOME_TCP,
    clamp_to_workspace,
    sample_initial_conditions,
    scene_xml,
)

GRIP_OPEN_CTRL = 0.0
GRIP_CLOSED_CTRL = 235.0
TRAJECTORY_DT = 0.01


class StateBenchEnv(ArmKinematics):
    """One episode of the pick-and-place task."""

    def __init__(self, seed: int, render: bool = False):
        self.seed = int(seed)
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)

        self.tcp = self.model.site("link_tcp").id
        self.qids = np.array(
            [self.model.jnt_qposadr[self.model.joint(f"joint{i}").id] for i in range(1, 8)]
        )
        self.dofs = np.array(
            [self.model.jnt_dofadr[self.model.joint(f"joint{i}").id] for i in range(1, 8)]
        )
        self.ranges = np.array(
            [self.model.jnt_range[self.model.joint(f"joint{i}").id] for i in range(1, 8)]
        )
        self.cube_body = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube_geom").id
        self.cube_qpos = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
        self.cube_dof = self.model.jnt_dofadr[self.model.joint("cube_joint").id]
        self.pad_geoms = {
            side: [
                self.model.geom(f"{side}_finger_pad_{i}").id for i in (1, 2)
            ]
            for side in ("left", "right")
        }
        self.receptacle_geoms = {
            self.model.geom(name).id
            for name in (
                "receptacle_floor",
                "receptacle_wall_px",
                "receptacle_wall_nx",
                "receptacle_wall_py",
                "receptacle_wall_ny",
            )
        }
        self.table_geom = self.model.geom("table").id

        self.command_grip = 0.0
        self.commanded_yaw = 0.0
        self.step_hooks: list = []
        self.trajectory: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._last_log_t = -1.0
        self.renderer = mujoco.Renderer(self.model, 480, 640) if render else None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.38, 0.02, 0.10]
        self.camera.distance = 1.20
        self.camera.azimuth = 132
        self.camera.elevation = -28
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        conditions = sample_initial_conditions(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = HOME
        self.data.ctrl[:7] = HOME
        self.data.ctrl[7] = GRIP_OPEN_CTRL
        self.command_grip = 0.0
        q = self.solve_ik(HOME_TCP, yaw=0.0)
        self.data.qpos[self.qids] = q
        self.data.ctrl[:7] = q
        self.commanded_yaw = 0.0

        yaw = conditions["cube_yaw"]
        self.data.qpos[self.cube_qpos : self.cube_qpos + 3] = [
            conditions["cube_x"],
            conditions["cube_y"],
            CUBE_HALF,
        ]
        self.data.qpos[self.cube_qpos + 3 : self.cube_qpos + 7] = [
            math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
        ]
        mujoco.mj_forward(self.model, self.data)
        self.trajectory.clear()
        self._last_log_t = -1.0
        self.hold(0.30)
        self.start_time = float(self.data.time)
        self.trajectory.clear()
        self._last_log_t = -1.0
        self._log_trajectory()

    # ------------------------------------------------------------- simulation
    @property
    def t(self) -> float:
        return float(self.data.time) - getattr(self, "start_time", 0.0)

    def _log_trajectory(self) -> None:
        if self.t - self._last_log_t >= TRAJECTORY_DT - 1e-9:
            self._last_log_t = self.t
            self.trajectory.append((self.t, self.truth_cube_position(), self.proprio_tcp()))

    def _step(self) -> None:
        self.data.qfrc_applied[self.dofs] = self.data.qfrc_bias[self.dofs]
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Physics diverged")
        self._log_trajectory()
        for hook in self.step_hooks:
            hook(self)

    def hold(self, seconds: float) -> None:
        for _ in range(max(0, round(seconds / self.model.opt.timestep))):
            self._step()

    def move_to(self, target, yaw: float | None = None, seconds: float = 1.0) -> bool:
        """Interpolate the arm to ``target`` with a smooth profile.

        Returns False (and holds still for the same duration) when the target is
        unreachable, so a rejected motion still costs the episode its time.
        """
        goal = clamp_to_workspace(target)
        want_yaw = self.commanded_yaw if yaw is None else wrap_to_half_pi(yaw)
        try:
            q = self.solve_ik(goal, yaw=want_yaw)
        except RuntimeError:
            self.hold(seconds)
            return False
        initial = self.data.qpos[self.qids].copy()
        n = max(1, round(seconds / self.model.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            blend = s * s * s * (10 + s * (-15 + 6 * s))
            self.data.ctrl[:7] = initial + (q - initial) * blend
            self._step()
        self.commanded_yaw = want_yaw
        return True

    def set_grip(self, closed: bool, seconds: float = 0.45) -> None:
        self.command_grip = 1.0 if closed else 0.0
        self.data.ctrl[7] = GRIP_CLOSED_CTRL if closed else GRIP_OPEN_CTRL
        self.hold(seconds)

    # -------------------------------------------------------------- proprio_*
    def proprio_tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp].copy()

    def proprio_tcp_yaw(self) -> float:
        rot = self.data.site_xmat[self.tcp].reshape(3, 3)
        return wrap_to_half_pi(math.atan2(rot[1, 0], rot[0, 0]))

    def proprio_joint_positions(self) -> list[float]:
        return [round(float(v), 5) for v in self.data.qpos[self.qids]]

    def proprio_gripper_opening(self) -> float:
        """Inner gap between the finger pads, from the finger joint encoders."""
        left = self.data.geom_xpos[self.pad_geoms["left"][0]]
        right = self.data.geom_xpos[self.pad_geoms["right"][0]]
        pad_half_thickness = float(self.model.geom_size[self.pad_geoms["left"][0]][1])
        return max(0.0, float(np.linalg.norm(left - right)) - 2 * pad_half_thickness)

    def proprio_gripper_force(self) -> float:
        """Magnitude of the gripper tendon actuator force (a real, if crude, sensor)."""
        return round(abs(float(self.data.actuator_force[7])), 3)

    def proprio_grip_command(self) -> str:
        return "closed" if self.command_grip > 0.5 else "open"

    # ---------------------------------------------------------------- truth_*
    def truth_cube_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def truth_cube_yaw(self) -> float:
        rot = self.data.xmat[self.cube_body].reshape(3, 3)
        return wrap_to_half_pi(math.atan2(rot[1, 0], rot[0, 0]))

    def truth_cube_velocity(self) -> np.ndarray:
        return self.data.qvel[self.cube_dof : self.cube_dof + 3].copy()

    def truth_cube_contacts(self) -> dict[str, bool | set[str]]:
        """Which named surfaces the cube currently touches."""
        fingers: set[str] = set()
        receptacle = False
        table = False
        for c in self.data.contact[: self.data.ncon]:
            if c.dist > 0.001 or self.cube_geom not in (c.geom1, c.geom2):
                continue
            other = int(c.geom2 if c.geom1 == self.cube_geom else c.geom1)
            for side, ids in self.pad_geoms.items():
                if other in ids:
                    fingers.add(side)
            if other in self.receptacle_geoms:
                receptacle = True
            if other == self.table_geom:
                table = True
        return {"fingers": fingers, "receptacle": receptacle, "table": table}

    def truth_bodies(self) -> dict[str, list[float]]:
        """Every body pose. RICH_TRUTH only."""
        out: dict[str, list[float]] = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            out[name] = [round(float(v), 5) for v in self.data.xpos[i]]
        return out

    def truth_contact_pairs(self) -> list[list[str]]:
        """Every active contact pair by geom name. RICH_TRUTH only."""
        pairs: list[list[str]] = []
        for c in self.data.contact[: self.data.ncon]:
            if c.dist > 0.001:
                continue
            a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or "?"
            b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or "?"
            pairs.append(sorted([a, b]))
        return pairs

    def truth_grasp_target(self) -> np.ndarray:
        """Where the TCP must be to grasp the cube."""
        p = self.truth_cube_position()
        return np.array([p[0], p[1], GRASP_Z])

    # ------------------------------------------------------------------ misc.
    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def displacement_over(self, t0: float, window: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Cube and TCP displacement over ``[t0, t0 + window]`` of sim time.

        Returns None when the logged trajectory does not cover the window, which
        is what happens at decisions near the end of an episode.
        """
        if not self.trajectory or self.trajectory[-1][0] < t0 + window - 1e-6:
            return None
        times = np.array([row[0] for row in self.trajectory])
        i0 = int(np.searchsorted(times, t0 - 1e-9))
        i1 = int(np.searchsorted(times, t0 + window - 1e-9))
        i0 = min(i0, len(self.trajectory) - 1)
        i1 = min(i1, len(self.trajectory) - 1)
        cube = self.trajectory[i1][1] - self.trajectory[i0][1]
        tcp = self.trajectory[i1][2] - self.trajectory[i0][2]
        return cube, tcp
