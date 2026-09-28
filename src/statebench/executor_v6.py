"""Turn 6 executor: ``ExecutorV2`` with the grasp and place heights of the variant's cube.

``executor.py`` and ``executor_v2.py`` are unchanged. For V0 the heights equal
``scene.GRASP_Z`` and ``scene.PLACE_Z`` exactly, so V0 behaviour is identical
(checked against Turn 5's EE packs). Registration at ``t_obs`` is inherited.
"""

from __future__ import annotations

from .executor import DURATIONS
from .executor_v2 import ExecutorV2
from .scene import TRAVEL_Z


class ExecutorV6(ExecutorV2):
    def __init__(self, env, events, perturbation_offset, grasp_z: float, place_z: float):
        super().__init__(env, events, perturbation_offset)
        self.grasp_z = float(grasp_z)
        self.place_z = float(place_z)

    def _do_approach(self, state, history):
        target, yaw, source = self._resolve_target(state, history)
        above, descend = DURATIONS["approach"]
        moved_a = self.env.move_to([target[0], target[1], TRAVEL_Z], yaw=yaw, seconds=above)
        moved_b = self.env.move_to([target[0], target[1], self.grasp_z], yaw=yaw, seconds=descend)
        return {"target_source": source, "commanded_target": [round(float(v), 4) for v in target],
                "commanded_yaw": None if yaw is None else round(float(yaw), 4),
                "accepted": bool(moved_a and moved_b)}

    def _do_transport(self, state, history):
        receptacle, source = self._resolve_receptacle(state, history)
        across, lower = DURATIONS["transport"]
        moved_a = self.env.move_to([receptacle[0], receptacle[1], TRAVEL_Z], yaw=0.0, seconds=across)
        moved_b = self.env.move_to([receptacle[0], receptacle[1], self.place_z], yaw=0.0, seconds=lower)
        return {"receptacle_source": source, "commanded_target": [round(float(v), 4) for v in receptacle],
                "accepted": bool(moved_a and moved_b)}
