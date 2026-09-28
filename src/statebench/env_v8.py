"""Turn 8 environment: ``EnvV7`` built from ``scene_v8`` (adds the V3 slip variant). ``env_v7.py`` is unchanged."""

from __future__ import annotations

import mujoco
import numpy as np

from .env_v7 import EnvV7
from .scene_v8 import VARIANTS, variant_xml_v8


class EnvV8(EnvV7):
    def __init__(self, seed: int, render: bool = False, variant: str = "V0", scene=None, jitter: int = 0,
                 friction=None, mass=None):
        self.seed = int(seed)
        self.cfg = VARIANTS[variant]
        self.variant = variant
        self.scene_params = scene
        self.jitter_index = int(jitter)
        self.has_marker = False
        self.model = mujoco.MjModel.from_xml_string(variant_xml_v8(variant, scene, friction=friction, mass=mass))
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

    def pad_normal_forces(self) -> dict[str, float]:
        """Normal force (N) between each finger's pads and the target cube: the fingertip sensor's quantity."""
        out = {"left": 0.0, "right": 0.0}
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.cube_geom not in (c.geom1, c.geom2):
                continue
            other = int(c.geom2 if c.geom1 == self.cube_geom else c.geom1)
            for side, ids in self.pad_geoms.items():
                if other in ids:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    out[side] += abs(float(f6[0]))
        return out
