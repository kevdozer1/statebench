"""The physical verifier. FIXED. Nothing that optimizes may edit this file.

Success requires, held continuously for 1.0 s of sim time:

* the cube's centre inside the receptacle's inner volume,
* the cube in contact with the receptacle and in contact with neither finger pad,
* the gripper open past the release threshold (this is what "after release"
  means physically - the verifier never asks the executor whether it released),
* the gripper clear of the receptacle.

The verifier is the only thing that may declare success. The decision model's
``step_complete`` answer is a separate, non-authoritative judge channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import gripper_clear_of_receptacle, in_receptacle_volume

STABLE_SECONDS_REQUIRED = 1.0
RELEASE_OPENING_M = 0.055
TIMEOUT_SECONDS = 30.0
CUBE_LOST_Z = -0.05

RUNNING = "running"
SUCCESS = "success"
FAILURE = "failure"
TIMEOUT = "timeout"


@dataclass
class VerifierState:
    status: str = RUNNING
    stable_seconds: float = 0.0
    max_stable_seconds: float = 0.0
    t_success: float | None = None
    reason: str | None = None
    conditions: dict[str, bool] = field(default_factory=dict)


class Verifier:
    """Attach to an env via ``env.on_step``; it accrues on every physics step."""

    def __init__(self, env):
        self.env = env
        self.state = VerifierState()

    def _conditions(self) -> dict[str, bool]:
        cube = self.env.truth_cube_position()
        contacts = self.env.truth_cube_contacts()
        return {
            "cube_center_in_volume": in_receptacle_volume(cube),
            "cube_touches_receptacle": bool(contacts["receptacle"]),
            "cube_free_of_fingers": not contacts["fingers"],
            "gripper_open_past_release": self.env.proprio_gripper_opening() > RELEASE_OPENING_M,
            "gripper_clear_of_receptacle": gripper_clear_of_receptacle(self.env.proprio_tcp()),
        }

    def update(self, _env=None) -> None:
        if self.state.status != RUNNING:
            return
        conditions = self._conditions()
        self.state.conditions = conditions
        if all(conditions.values()):
            self.state.stable_seconds += float(self.env.model.opt.timestep)
        else:
            self.state.stable_seconds = 0.0
        self.state.max_stable_seconds = max(
            self.state.max_stable_seconds, self.state.stable_seconds
        )
        if self.state.stable_seconds >= STABLE_SECONDS_REQUIRED:
            self.state.status = SUCCESS
            self.state.t_success = round(self.env.t, 3)
            self.state.reason = "all placement conditions held for 1.0 s of sim time"
            return
        if float(self.env.truth_cube_position()[2]) < CUBE_LOST_Z:
            self.state.status = FAILURE
            self.state.reason = "cube left the table"
            return
        if self.env.t >= TIMEOUT_SECONDS:
            self.state.status = TIMEOUT
            self.state.reason = f"reached {TIMEOUT_SECONDS:.0f} s of sim time"

    def finish(self, reason: str) -> None:
        """Close out an episode that ended for a non-physical reason."""
        if self.state.status == RUNNING:
            self.state.status = FAILURE
            self.state.reason = reason

    @property
    def status(self) -> str:
        return self.state.status

    def report(self) -> dict:
        return {
            "status": self.state.status,
            "reason": self.state.reason,
            "t_success": self.state.t_success,
            "max_stable_seconds": round(self.state.max_stable_seconds, 3),
            "final_conditions": self.state.conditions,
        }


def verifier_says_complete(verifier: Verifier) -> bool:
    """The physical channel's current answer to 'is the goal satisfied?'.

    Used only for the judge-versus-verifier disagreement accounting, never to
    end an episode.
    """
    return verifier.status == SUCCESS or all(
        verifier.state.conditions.get(k, False)
        for k in (
            "cube_center_in_volume",
            "cube_touches_receptacle",
            "cube_free_of_fingers",
            "gripper_open_past_release",
            "gripper_clear_of_receptacle",
        )
    )


def cube_moved_with_gripper(
    env, t0: float, window: float = 0.5, match_m: float = 0.010, motion_m: float = 0.010
) -> bool | None:
    """Operational label for ``object_held``.

    True when, over the next ``window`` seconds of sim time, the cube's
    displacement matches the TCP's to within ``match_m``. Returns None when the
    TCP moved less than ``motion_m`` over the window (the relation is then not
    observable) or the trajectory does not cover the window.
    """
    pair = env.displacement_over(t0, window)
    if pair is None:
        return None
    cube, tcp = pair
    if float(np.linalg.norm(tcp)) < motion_m:
        return None
    return bool(float(np.linalg.norm(cube - tcp)) < match_m)
