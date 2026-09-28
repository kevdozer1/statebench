"""Turn 6 reference controller: ``reference.decide`` with the variant's grasp and place heights.

``reference.py`` is unchanged; this repeats its rules with the two heights as
parameters, for the solvability controls on RICH_TRUTH.
"""

from __future__ import annotations

import numpy as np

from .ik import wrap_to_half_pi
from .reference import ALIGN_XY_M, ALIGN_YAW_RAD, ALIGN_Z_M, OVER_RECEPTACLE_M
from .scene import RECEPTACLE_CENTER, in_receptacle_volume
from .verify import RELEASE_OPENING_M


def decide(rich: dict, grasp_z: float, place_z: float) -> str:
    cube = np.asarray(rich["cube_position_m"], dtype=float)
    cube_yaw = float(rich["cube_yaw_rad"])
    fingers = set(rich["cube_finger_contacts"])
    on_receptacle = bool(rich["cube_touches_receptacle"])
    on_table = bool(rich["cube_touches_table"])
    proprio = rich["proprio"]
    tcp = np.asarray(proprio["tcp_position_m"], dtype=float)
    tcp_yaw = float(proprio["tcp_yaw_rad"])
    open_enough = float(proprio["gripper_opening_m"]) > RELEASE_OPENING_M
    inside = in_receptacle_volume(cube)
    over = float(np.linalg.norm(tcp[:2] - RECEPTACLE_CENTER)) < OVER_RECEPTACLE_M
    if inside and not fingers and open_enough:
        return "inspect" if tcp[2] > 0.145 else "retreat"
    if fingers and not on_table and not on_receptacle:
        return "release" if over and tcp[2] <= place_z + 0.015 else "transport"
    if fingers:
        return "test_lift"
    if not open_enough:
        return "release"
    aligned = (abs(float(cube[0] - tcp[0])) < ALIGN_XY_M and abs(float(cube[1] - tcp[1])) < ALIGN_XY_M
               and abs(float(tcp[2] - grasp_z)) < ALIGN_Z_M
               and abs(wrap_to_half_pi(cube_yaw - tcp_yaw)) < ALIGN_YAW_RAD)
    return "close" if aligned else "approach"
