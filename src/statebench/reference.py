"""Reference controller: a scripted geometric policy on RICH_TRUTH.

It chooses the next primitive by explicit conditions on full simulator state,
with no hidden state machine - every branch is a predicate on the current
observation, so the same observation always yields the same primitive. Its
success rate is the ceiling for this task with this executor.
"""

from __future__ import annotations

import numpy as np

from .ik import wrap_to_half_pi
from .scene import GRASP_Z, PLACE_Z, RECEPTACLE_CENTER, in_receptacle_volume
from .verify import RELEASE_OPENING_M

ALIGN_XY_M = 0.012
ALIGN_Z_M = 0.015
ALIGN_YAW_RAD = 0.10
OVER_RECEPTACLE_M = 0.030


def decide(rich: dict) -> str:
    """The next primitive, from RICH_TRUTH."""
    cube = np.asarray(rich["cube_position_m"], dtype=float)
    cube_yaw = float(rich["cube_yaw_rad"])
    fingers = set(rich["cube_finger_contacts"])
    on_receptacle = bool(rich["cube_touches_receptacle"])
    on_table = bool(rich["cube_touches_table"])
    proprio = rich["proprio"]
    tcp = np.asarray(proprio["tcp_position_m"], dtype=float)
    tcp_yaw = float(proprio["tcp_yaw_rad"])
    opening = float(proprio["gripper_opening_m"])

    open_enough = opening > RELEASE_OPENING_M
    inside = in_receptacle_volume(cube)
    over_receptacle = float(np.linalg.norm(tcp[:2] - RECEPTACLE_CENTER)) < OVER_RECEPTACLE_M

    # 1. Placed and let go: clear the receptacle so the verifier can settle.
    if inside and not fingers and open_enough:
        return "inspect" if tcp[2] > 0.145 else "retreat"

    # 2. Supported by nothing but the gripper: it is genuinely held.
    if fingers and not on_table and not on_receptacle:
        if over_receptacle and tcp[2] <= PLACE_Z + 0.015:
            return "release"
        return "transport"

    # 3. Touching the object but it is still supported by a surface: prove the grasp.
    if fingers:
        return "test_lift"

    # 4. Fingers closed on nothing: open before trying again.
    if not open_enough:
        return "release"

    # 5. Open and not touching: close if lined up on the grasp pose, else approach.
    aligned = (
        abs(float(cube[0] - tcp[0])) < ALIGN_XY_M
        and abs(float(cube[1] - tcp[1])) < ALIGN_XY_M
        and abs(float(tcp[2] - GRASP_Z)) < ALIGN_Z_M
        and abs(wrap_to_half_pi(cube_yaw - tcp_yaw)) < ALIGN_YAW_RAD
    )
    return "close" if aligned else "approach"
