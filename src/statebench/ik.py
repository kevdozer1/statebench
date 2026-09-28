"""Damped least-squares IK for the xArm7 with a downward tool and free yaw.

Derived from the fixed-orientation solver in OpenRoboto's ``jev-robot-control``
(``simulation.py``, BSD-3-Clause, UFACTORY xArm7 description). The change here is
that the desired tool orientation is parameterised by a yaw angle about world +Z
instead of being a constant matrix, because our target is a cube whose yaw is
sampled per episode and a fixed-yaw gripper cannot grasp it at every angle.

This is kinematics, not state: it uses the robot's own model and the commanded
target, never the simulator's knowledge of any object.
"""

from __future__ import annotations

import mujoco
import numpy as np

HOME = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])


def desired_rotation(yaw: float) -> np.ndarray:
    """Tool frame pointing down (-Z) with its finger axis rotated by ``yaw``.

    The base orientation is ``diag(-1, 1, -1)``: tool +Z maps to world -Z. A yaw
    rotation about world +Z is applied on the left.
    """
    base = np.diag([-1.0, 1.0, -1.0])
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ base


class ArmKinematics:
    """Mixin providing ``solve_ik`` for a model with ``tcp``/``qids``/``dofs``."""

    def solve_ik(self, target, yaw: float = 0.0, initial=None) -> np.ndarray:
        d = self.ikdata
        d.qpos[:] = self.data.qpos
        d.qpos[self.qids] = self.data.qpos[self.qids] if initial is None else initial
        desired = desired_rotation(yaw)
        jp = np.zeros((3, self.model.nv))
        jr = jp.copy()
        residual = np.inf
        for _ in range(220):
            mujoco.mj_forward(self.model, d)
            rot = d.site_xmat[self.tcp].reshape(3, 3)
            ep = np.asarray(target, dtype=float) - d.site_xpos[self.tcp]
            er = 0.5 * sum(
                (np.cross(rot[:, i], desired[:, i]) for i in range(3)), np.zeros(3)
            )
            residual = float(np.linalg.norm(ep))
            if residual < 0.00015 and np.linalg.norm(er) < 0.002:
                return d.qpos[self.qids].copy()
            mujoco.mj_jacSite(self.model, d, jp, jr, self.tcp)
            jac = np.vstack([jp[:, self.dofs], jr[:, self.dofs] * 0.35])
            err = np.r_[ep, er * 0.35]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.00004, err)
            delta *= min(1.0, 0.12 / (np.max(np.abs(delta)) + 1e-12))
            d.qpos[self.qids] = np.clip(
                d.qpos[self.qids] + delta, self.ranges[:, 0] + 0.01, self.ranges[:, 1] - 0.01
            )
        raise RuntimeError(
            f"IK did not converge for {np.round(np.asarray(target, dtype=float), 3)} "
            f"yaw={yaw:.3f}; position residual={residual:.4f} m"
        )


def wrap_to_half_pi(angle: float) -> float:
    """Map a yaw onto (-pi/4, pi/4], the equivalence class of a square prism.

    A cube's graspable finger axis repeats every 90 degrees, so the gripper never
    needs to rotate more than 45 degrees to line up with a face pair.
    """
    a = (float(angle) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(a)
