"""E1 receives only its declared inputs. Run: python -m unittest tests.test_perception_e1 -v"""

import inspect
import re
import unittest
from pathlib import Path

import numpy as np

from statebench import perception_e1 as E

SOURCE = Path(E.__file__).read_text(encoding="utf-8")


def _camera():
    return {"K": [[579.41, 0.0, 320.0], [0.0, 579.41, 240.0], [0.0, 0.0, 1.0]],
            "camera_to_world": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}}


def _proprio(t=1.0):
    return {"tcp_position_m": [0.3, -0.02, 0.24], "tcp_yaw_rad": 0.0, "gripper_opening_m": 0.089,
            "gripper_command": "open", "sim_time_s": t, "joint_positions_rad": [0.0] * 7}


class TestE1Inputs(unittest.TestCase):
    def test_signature_is_exactly_the_declared_inputs(self):
        params = list(inspect.signature(E.estimate).parameters)
        self.assertEqual(params, ["rgb", "depth_mm", "camera", "proprio", "constants", "memory"])

    def test_module_never_touches_the_simulator(self):
        code = re.sub(r'""".*?"""', "", SOURCE, flags=re.S)
        for banned in ("mujoco", "StateBenchEnv", "truth_", ".data.", "env.", "import env",
                       "observations", "verify", "sim_truth"):
            self.assertNotIn(banned, code, banned)

    def test_callable_with_only_its_arguments(self):
        rgb = np.full((480, 640, 3), 245, dtype=np.uint8)
        depth = np.full((480, 640), 1500, dtype=np.uint16)
        rgb[230:250, 310:330] = (180, 30, 25)  # a red patch in front of the camera
        est, mem = E.estimate(rgb, depth, _camera(), _proprio(), E.CONSTANTS, E.new_memory())
        self.assertTrue(est["fresh"])
        self.assertEqual(len(mem["track"]), 1)
        # Nothing visible: falls back to the previous estimate, with age above 0.
        est2, _ = E.estimate(np.full_like(rgb, 245), depth, _camera(), _proprio(1.4), E.CONSTANTS, mem)
        self.assertFalse(est2["fresh"])
        self.assertGreater(est2["age_s"], 0.0)



if __name__ == "__main__":
    unittest.main()
