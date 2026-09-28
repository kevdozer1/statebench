"""Turn 13 Phase 7: the button scene (build only; nothing is claimed from it this turn).

The pick-and-place arm and table (``scene_v8`` V0) with a push button on the table; the cube is parked out of reach
(the receptacle stays, fixed).

Button (declared in BUTTON):
* a static housing (50 x 50 x 20 mm) and a cap (32 x 32 x 10 mm) on a vertical slide joint;
* travel per seed (joint range -travel to 0);
* a spring (joint stiffness, spring reference 0) returns the cap up; damping 2 N s/m;
* activation with hysteresis: the button latches ON when the cap has travelled ON_FRACTION = 0.6 of its travel and
  goes OFF only when it has come back to OFF_FRACTION = 0.2 of its travel or less; each ON then OFF is one completed
  press cycle.

Per seed (``button_draw``; its own RNG, 31_000 + seed): the button's position on the table (x in 0.32-0.54 m, y in
-0.24 to 0.02 m, clear of the receptacle and of the parked cube), its travel (8-12 mm), and the activation force at
the ON point (2.4-3.6 N; stiffness = force / ON point).

Truth labels (``truth_labels``): contact (any gripper geom touching the cap), travel_m, pressed_past_activation (the
cap is at or past the ON point now), activated (the latched state), cycles (completed press cycles).

Verifier (``ButtonVerifier``): success when at least one press cycle has completed, the button is OFF, no gripper
geom touches the cap, and the TCP is at least 30 mm above the cap's rest top, all held for 0.5 s of simulated time;
failure at 20 s of simulated time.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .env_v8 import EnvV8
from .scene_v8 import VARIANTS, variant_xml_v8

BUTTON = {"housing_half": (0.025, 0.025, 0.010), "cap_half": (0.016, 0.016, 0.005), "on_fraction": 0.6,
          "off_fraction": 0.2, "damping": 2.0}
TRAVEL_RANGE_M = (0.008, 0.012)
ACTIVATION_FORCE_RANGE_N = (2.4, 3.6)
BUTTON_X_RANGE = (0.32, 0.54)
BUTTON_Y_RANGE = (-0.24, 0.02)
CUBE_PARK = (0.24, 0.30)
TIMEOUT_S = 20.0
STABLE_S = 0.5
CLEAR_ABOVE_M = 0.030


def button_draw(seed: int) -> dict:
    rng = np.random.default_rng(31_000 + seed)
    x, y = float(rng.uniform(*BUTTON_X_RANGE)), float(rng.uniform(*BUTTON_Y_RANGE))
    travel = float(rng.uniform(*TRAVEL_RANGE_M))
    force = float(rng.uniform(*ACTIVATION_FORCE_RANGE_N))
    on_m = BUTTON["on_fraction"] * travel
    return {"x": x, "y": y, "travel_m": travel, "on_m": on_m, "off_m": BUTTON["off_fraction"] * travel,
            "activation_force_n": force, "stiffness": force / on_m}


def cap_rest_top_z() -> float:
    return 2 * BUTTON["housing_half"][2] + 2 * BUTTON["cap_half"][2]


def button_xml(seed: int) -> str:
    b = button_draw(seed)
    root = ET.fromstring(variant_xml_v8("V0"))
    world, asset = root.find("worldbody"), root.find("asset")
    ET.SubElement(asset, "material", name="button_housing_skin", rgba=".25 .25 .28 1")
    ET.SubElement(asset, "material", name="button_cap_skin", rgba=".85 .12 .10 1")
    hh, ch = BUTTON["housing_half"], BUTTON["cap_half"]
    ET.SubElement(world, "geom", name="button_housing", type="box", pos=f"{b['x']} {b['y']} {hh[2]}",
                  size=f"{hh[0]} {hh[1]} {hh[2]}", material="button_housing_skin")
    cap = ET.SubElement(world, "body", name="button_cap", pos=f"{b['x']} {b['y']} {2 * hh[2] + ch[2]}")
    ET.SubElement(cap, "joint", name="button_slide", type="slide", axis="0 0 1", limited="true",
                  range=f"{-b['travel_m']} 0", stiffness=f"{b['stiffness']}", springref="0",
                  damping=f"{BUTTON['damping']}")
    ET.SubElement(cap, "geom", name="button_cap_geom", type="box", size=f"{ch[0]} {ch[1]} {ch[2]}",
                  material="button_cap_skin", mass="0.02", friction="1.0 .01 .002", condim="3")
    return ET.tostring(root, encoding="unicode")


class ButtonEnv(EnvV8):
    def __init__(self, seed: int, render: bool = False):
        """EnvV8's constructor with the button scene's model (every id is taken from that model)."""
        self.seed = int(seed)
        self.button = button_draw(seed)
        self.on, self.cycles, self.button_events = False, 0, []
        self.cfg = VARIANTS["V0"]
        self.variant, self.scene_params, self.jitter_index, self.has_marker = "V0", None, 0, False
        self.model = mujoco.MjModel.from_xml_string(button_xml(seed))
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        m = self.model
        self.tcp = m.site("link_tcp").id
        self.qids = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.dofs = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.ranges = np.array([m.jnt_range[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.cube_body = m.body("cube").id
        self.cube_geom = m.geom("cube_geom").id
        self.cube_qpos = m.jnt_qposadr[m.joint("cube_joint").id]
        self.cube_dof = m.jnt_dofadr[m.joint("cube_joint").id]
        self.distractor_qpos, self.distractor_geom = None, None
        self.pad_geoms = {side: [m.geom(f"{side}_finger_pad_{i}").id for i in (1, 2)] for side in ("left", "right")}
        self.receptacle_geoms = {m.geom(n).id for n in ("receptacle_floor", "receptacle_wall_px", "receptacle_wall_nx",
                                                          "receptacle_wall_py", "receptacle_wall_ny")}
        self.table_geom = m.geom("table").id
        self.slide_q = m.jnt_qposadr[m.joint("button_slide").id]
        self.cap_geom = m.geom("button_cap_geom").id
        self.gripper_geoms = {g for g in range(m.ngeom) if self._is_gripper(g)}
        self.command_grip = 0.0
        self.commanded_yaw = 0.0
        self.step_hooks = [ButtonEnv._button_logic]
        self.trajectory = []
        self._last_log_t = -1.0
        self.renderer = mujoco.Renderer(m, 480, 640) if render else None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.38, 0.02, 0.10]
        self.camera.distance = 1.20
        self.camera.azimuth = 132
        self.camera.elevation = -28
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.data.qpos[self.cube_qpos:self.cube_qpos + 3] = [*CUBE_PARK, self.cfg.cube_half]
        self.data.qpos[self.cube_qpos + 3:self.cube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.hold(0.2)
        self.on, self.cycles, self.button_events = False, 0, []
        self.start_time = float(self.data.time)
        self.trajectory.clear()
        self._last_log_t = -1.0

    def _is_gripper(self, g: int) -> bool:
        b = int(self.model.geom_bodyid[g])
        names = []
        while b > 0:
            names.append(self.model.body(b).name)
            b = int(self.model.body_parentid[b])
        return any(("finger" in n or "gripper" in n or "hand" in n) for n in names)

    @staticmethod
    def _button_logic(env) -> None:
        travel = -float(env.data.qpos[env.slide_q])
        if not env.on and travel >= env.button["on_m"]:
            env.on = True
            env.button_events.append({"event": "button_on", "t": round(env.t, 3)})
        elif env.on and travel <= env.button["off_m"]:
            env.on = False
            env.cycles += 1
            env.button_events.append({"event": "button_off", "t": round(env.t, 3)})

    def truth_labels(self) -> dict:
        contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if self.cap_geom in pair and (pair - {self.cap_geom}) & self.gripper_geoms:
                contact = True
                break
        travel = -float(self.data.qpos[self.slide_q])
        return {"contact": contact, "travel_m": round(travel, 5), "pressed_past_activation": travel >= self.button["on_m"],
                "activated": self.on, "cycles": self.cycles}


class ButtonVerifier:
    def __init__(self, env: ButtonEnv):
        self.env, self.status, self.since, self.reason, self.t_success = env, "running", None, None, None

    def update(self, env=None) -> None:
        if self.status != "running":
            return
        e = self.env
        lab = e.truth_labels()
        ok = (lab["cycles"] >= 1 and not lab["activated"] and not lab["contact"]
              and float(e.proprio_tcp()[2]) >= cap_rest_top_z() + CLEAR_ABOVE_M)
        if ok:
            self.since = e.t if self.since is None else self.since
            if e.t - self.since >= STABLE_S:
                self.status, self.reason, self.t_success = "success", "press cycle completed and gripper clear", round(e.t, 3)
        else:
            self.since = None
        if self.status == "running" and e.t >= TIMEOUT_S:
            self.status, self.reason = "failure", f"reached {TIMEOUT_S:.0f} s of sim time"

    def report(self) -> dict:
        return {"status": self.status, "reason": self.reason, "t_success": self.t_success,
                "cycles": self.env.cycles, "events": list(self.env.button_events)}
