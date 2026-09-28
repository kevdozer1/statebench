"""Turn 7 environment: ``EnvV6`` with the optional declared calibration marker. ``env_v6.py`` is unchanged.

With ``marker=False`` the model is built from exactly the XML ``EnvV6`` builds, so
the episode is identical. With ``marker=True`` one visual-only, zero-mass sphere is
added to the gripper base (``marker_v7``); the dynamics are unchanged (checked:
identical trajectories on dev seeds, see notes/turn7.md).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env_v6 import EnvV6
from .marker_v7 import MARKER_BODY, marker_geom_xml_attrs
from .scene_v6 import VARIANTS, SceneParams, VariantConfig, variant_xml


def variant_xml_v7(cfg: VariantConfig, scene: SceneParams | None, marker: bool) -> str:
    xml = variant_xml(cfg, scene)
    if not marker:
        return xml
    root = ET.fromstring(xml)
    body = root.find(f".//body[@name='{MARKER_BODY}']")
    if body is None:
        raise RuntimeError(f"marker body {MARKER_BODY} not found in the scene XML")
    ET.SubElement(body, "geom", **marker_geom_xml_attrs())
    return ET.tostring(root, encoding="unicode")


class EnvV7(EnvV6):
    def __init__(self, seed: int, render: bool = False, variant: str = "V0",
                 scene: SceneParams | None = None, jitter: int = 0, marker: bool = False):
        self.seed = int(seed)
        self.cfg: VariantConfig = VARIANTS[variant]
        self.scene_params = scene
        self.jitter_index = int(jitter)
        self.has_marker = bool(marker)
        self.model = mujoco.MjModel.from_xml_string(variant_xml_v7(self.cfg, scene, marker))
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

    def truth_marker_position(self) -> np.ndarray:
        """Truth, for tests and diagnostics only; no remedy reads it."""
        return self.data.geom_xpos[self.model.geom("calibration_marker").id].copy()
