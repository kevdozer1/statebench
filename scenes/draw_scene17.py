"""Turn 17 Phase 6: the Turn 15 drawing scene (``draw_scene.DrawScene``, unchanged) with a declared finish mode.

{"cmd": "mode", "finish": m} sets how the ``finish`` skill behaves for the rest of the process:
* ``plain``: as Turn 15 (the episode ends; the arm does not move);
* ``lift_home`` (**finish-lift**): if the pen is down, raise it 20 mm (the lift skill's motion, 0.8 s); then move at that
  height to above the home point (HOME_XY, 2.0 s), then rise to HOME_UP_M above the paper (1.0 s); the episode ends;
* ``home`` (**finish-home**): the same motion without the lift: move at the current tip height to above the home point
  (2.0 s), then rise (1.0 s). A pen left down drags along the paper on the way.
The home point is on the paper, at a corner of the Turn 15 drawing area (x 0.42 m, y 0.04 m), so drag ink stays on the
paper. Everything else (reset, draw, redraw, lift, verify, frame) is ``draw_scene``'s. {"cmd": "ink"} returns the true
ink marks and how many were added by the finish skill.
"""
from __future__ import annotations

import json
import sys
import time

import draw_scene as D

HOME_XY = (0.42, 0.04)
HOME_UP_M = 0.06


class Scene17(D.DrawScene):
    finish_mode = "plain"
    marks_before_finish = None

    def do(self, skill: str) -> dict:
        if skill != "finish" or self.finish_mode == "plain":
            return super().do(skill)
        t0 = self.f.data.time
        self.marks_before_finish = len(self.f.marks)
        events = []
        try:
            if self.finish_mode == "lift_home" and self.pen == "down":
                self._raise(D.LIFT_M, 0.8)
                self.pen = "up"
                events.append("lift_done")
            tip = self.pw.tip()
            self.pw.goto([HOME_XY[0], HOME_XY[1], float(tip[2])], 2.0)
            self.pw.goto([HOME_XY[0], HOME_XY[1], self.PAPER_Z + HOME_UP_M], 1.0)
            self.pen = "up"
            events.append("home_done")
        except D.Q.PressedTooHard as ex:
            self.error = f"pressed too hard: {ex}"
            events.append("error")
        except Exception as ex:  # noqa: BLE001
            self.error = f"controller error: {type(ex).__name__}: {ex}"
            events.append("error")
        self.finished = True
        events.append("finish_done")
        return {"state": self.state(), "events": events, "sim_dt": round(self.f.data.time - t0, 2)}


def main() -> None:
    scene = Scene17()
    for line in sys.stdin:
        msg = json.loads(line)
        t0 = time.monotonic()
        if msg["cmd"] == "mode":
            Scene17.finish_mode = msg["finish"]
            out = {"finish_mode": Scene17.finish_mode}
        elif msg["cmd"] == "reset":
            scene.marks_before_finish = None
            out = {"state": scene.reset(int(msg["seed"]), bool(msg.get("forced")))}
        elif msg["cmd"] == "do":
            out = scene.do(msg["skill"])
        elif msg["cmd"] == "ink":
            ink = [[round(float(m[0][0]), 5), round(float(m[0][1]), 5)] for m in scene.f.marks]
            nb = scene.marks_before_finish
            out = {"ink_xy": ink, "n_marks": len(ink), "marks_before_finish": nb,
                   "added_by_finish": None if nb is None else len(ink) - nb}
        elif msg["cmd"] == "verify":
            out = scene.verify()
        else:
            break
        out["wall_s"] = round(time.monotonic() - t0, 2)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
