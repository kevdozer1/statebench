"""Turn 6 environment: ``StateBenchEnv`` on a variant scene. ``env.py`` is unchanged.

The constructor mirrors ``StateBenchEnv.__init__`` except for the XML it loads,
and ``reset`` mirrors ``StateBenchEnv.reset`` except that the cube rests at its
own half-height and a V2 distractor is placed by a seeded draw at least 70 mm
from the target. Every accessor is inherited, so ``truth_*`` still reads only the
target cube.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .env import GRIP_OPEN_CTRL, StateBenchEnv
from .ik import HOME
from .scene import CUBE_X_RANGE, CUBE_Y_RANGE, HOME_TCP, sample_initial_conditions
from .scene_v6 import DISTRACTOR_MIN_SEPARATION_M, VARIANTS, SceneParams, VariantConfig, variant_xml


class EnvV6(StateBenchEnv):
    def __init__(self, seed: int, render: bool = False, variant: str = "V0",
                 scene: SceneParams | None = None, jitter: int = 0):
        self.seed = int(seed)
        self.cfg: VariantConfig = VARIANTS[variant]
        self.scene_params = scene
        self.jitter_index = int(jitter)
        self.model = mujoco.MjModel.from_xml_string(variant_xml(self.cfg, scene))
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        self.tcp = self.model.site("link_tcp").id
        self.qids = np.array([self.model.jnt_qposadr[self.model.joint(f"joint{i}").id] for i in range(1, 8)])
        self.dofs = np.array([self.model.jnt_dofadr[self.model.joint(f"joint{i}").id] for i in range(1, 8)])
        self.ranges = np.array([self.model.jnt_range[self.model.joint(f"joint{i}").id] for i in range(1, 8)])
        self.cube_body = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube_geom").id
        self.cube_qpos = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
        self.cube_dof = self.model.jnt_dofadr[self.model.joint("cube_joint").id]
        self.distractor_qpos = (self.model.jnt_qposadr[self.model.joint("distractor_joint").id]
                                if self.cfg.distractor else None)
        self.distractor_geom = self.model.geom("distractor_geom").id if self.cfg.distractor else None
        self.pad_geoms = {side: [self.model.geom(f"{side}_finger_pad_{i}").id for i in (1, 2)]
                          for side in ("left", "right")}
        self.receptacle_geoms = {self.model.geom(n).id for n in (
            "receptacle_floor", "receptacle_wall_px", "receptacle_wall_nx",
            "receptacle_wall_py", "receptacle_wall_ny")}
        self.table_geom = self.model.geom("table").id
        self.command_grip = 0.0
        self.commanded_yaw = 0.0
        self.step_hooks = []
        self.trajectory = []
        self._last_log_t = -1.0
        self.renderer = mujoco.Renderer(self.model, 480, 640) if render else None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.38, 0.02, 0.10]
        self.camera.distance = 1.20
        self.camera.azimuth = 132
        self.camera.elevation = -28
        self.reset()

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
        x, y, yaw = conditions["cube_x"], conditions["cube_y"], conditions["cube_yaw"]
        if self.jitter_index:
            rng = np.random.default_rng(90_000 + 64 * self.seed + self.jitter_index)
            dx, dy = rng.uniform(-0.010, 0.010, size=2)
            x, y, yaw = x + dx, y + dy, yaw + rng.uniform(-math.radians(5), math.radians(5))
        h = self.cfg.cube_half
        self.data.qpos[self.cube_qpos: self.cube_qpos + 3] = [x, y, h]
        self.data.qpos[self.cube_qpos + 3: self.cube_qpos + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        if self.cfg.distractor:
            rng = np.random.default_rng(80_000 + self.seed)
            for _ in range(200):
                dx_ = float(rng.uniform(*CUBE_X_RANGE))
                dy_ = float(rng.uniform(CUBE_Y_RANGE[0] - 0.04, CUBE_Y_RANGE[1] + 0.04))
                if math.hypot(dx_ - x, dy_ - y) >= DISTRACTOR_MIN_SEPARATION_M:
                    break
            dyaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
            a = self.distractor_qpos
            self.data.qpos[a: a + 3] = [dx_, dy_, h]
            self.data.qpos[a + 3: a + 7] = [math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)]
            self.distractor_start = [round(dx_, 4), round(dy_, 4)]
        mujoco.mj_forward(self.model, self.data)
        self.trajectory.clear()
        self._last_log_t = -1.0
        self.hold(0.30)
        self.start_time = float(self.data.time)
        self.trajectory.clear()
        self._last_log_t = -1.0
        self._log_trajectory()
