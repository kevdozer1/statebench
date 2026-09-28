"""Turn 15 Phase 1: audit of the video-derived intents against I3 (declared keyword rubric, applied to the lower-cased
goal and, for IV-chunked, the goal plus the steps).

Per task, the verifier conditions I3 states, and the rubric that counts a condition as stated:
* block: (in) the cube ends inside the receptacle: a container word (tray, receptacle, box, container, bin, basket)
  with in / inside / into; (released) release, let go, open the gripper, drop, no longer held; (clear) clear, away,
  withdraw, retreat, move back, home, raise the gripper, lift the gripper.
* button: (activated) press, push, activate, depress; (released) release, let go, raise, lift, no longer touching,
  pops back; (clear) clear, away, withdraw, retreat, home.
* drawing: (all strokes) all, every, complete, entire, full, whole; (pen lifted) lift, raise, away, no longer
  touching, off the paper.
* writing: (content) the task's binary terms appear in order in the text (as separate numbers, or run together);
  (spacing) the exact text with single spaces appears, or "space" / "separated"; (lift) lift, away, clear, withdraw,
  step back, remove the marker.
"Wrong": block or button text describing the other task (button words in a block intent, tray / pick words in a
button intent); writing text whose binary digits differ from the task's (the digits written, in order, ignoring
spaces); drawing text calling the target letters or writing is not counted as wrong (the target of the word ABC is
letters).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from .config import RUNS, SCENES

R = RUNS / "turn15"

RUBRIC = {
    "block": {"in": r"(tray|receptacle|box|container|bin|basket)", "in2": r"\b(in|inside|into)\b",
              "released": r"(releas|let go|open the gripper|open its gripper|drop|no longer (held|grip))",
              "clear": r"(clear|away|withdraw|retreat|move back|home|raise the gripper|lift the gripper)"},
    "button": {"activated": r"(press|push|activat|depress)", "released": r"(releas|let go|raise|lift|no longer touch|pops? back)",
               "clear": r"(clear|away|withdraw|retreat|home)"},
    "drawing": {"all": r"\b(all|every|complete|entire|full|whole)\b", "lifted": r"(lift|raise|away|no longer touch|off the paper)"},
    "writing": {"lift": r"(lift|away|clear|withdraw|step back|remove the marker)"},
}


def audit_one(task: str, text: str, task_def: dict | None) -> dict:
    t = text.lower()
    out = {}
    if task == "block":
        out["in"] = bool(re.search(RUBRIC["block"]["in"], t) and re.search(RUBRIC["block"]["in2"], t))
        out["released"] = bool(re.search(RUBRIC["block"]["released"], t))
        out["clear"] = bool(re.search(RUBRIC["block"]["clear"], t))
        out["wrong"] = bool(re.search(r"\bbutton\b", t))
    elif task == "button":
        for k in ("activated", "released", "clear"):
            out[k] = bool(re.search(RUBRIC["button"][k], t))
        out["wrong"] = bool(re.search(r"(tray|pick up|place it|block)", t))
    elif task == "drawing":
        out["all"] = bool(re.search(RUBRIC["drawing"]["all"], t))
        out["lifted"] = bool(re.search(RUBRIC["drawing"]["lifted"], t))
        out["wrong"] = False
    elif task == "writing":
        bins = task_def["terms_binary"]
        digits_task = "".join(bins)
        found = re.findall(r"[01]+", t)
        digits_text = "".join(found)
        out["content"] = digits_task in digits_text
        out["spacing"] = (" ".join(bins) in t) or bool(re.search(r"(space|separated)", t))
        out["lift"] = bool(re.search(RUBRIC["writing"]["lift"], t))
        out["wrong"] = bool(found) and digits_task not in digits_text
    return out


def audit(seeds=None) -> dict:
    import sys

    sys.path.insert(0, str(SCENES))
    import write_scene as W  # noqa: E402

    vi = json.loads((R / "video_intents.json").read_text())
    res = {}
    for task, per in vi.items():
        rows = []
        for s, v in per.items():
            if seeds is not None and int(s) not in seeds:
                continue
            td = W.task_for(int(s)) if task == "writing" else None
            goal_only = audit_one(task, v["goal"], td)
            chunked = audit_one(task, v["goal"] + " " + " ".join(f"{a} {b}" for a, b in v["chunks"]), td)
            rows.append({"seed": int(s), "goal": goal_only, "chunked": chunked, "n_steps": v["n_steps"]})
        keys = [k for k in (rows[0]["goal"] if rows else {})]
        res[task] = {"n": len(rows), "no_steps": sum(r["n_steps"] == 0 for r in rows),
                     "IV-goal states": {k: sum(r["goal"][k] for r in rows) for k in keys},
                     "IV-chunked states": {k: sum(r["chunked"][k] for r in rows) for k in keys}, "rows": rows}
    return res


if __name__ == "__main__":
    import sys

    rng = None
    if len(sys.argv) > 1:
        a, b, c = (int(x) for x in sys.argv[1].split(":"))
        rng = set(range(a, b, c))
    o = audit(rng)
    (R / f"video_intents_audit{'_' + sys.argv[1].replace(':', '-') if len(sys.argv) > 1 else ''}.json").write_text(json.dumps(o, indent=1))
    for t, v in o.items():
        print(t, "n", v["n"], "no steps", v["no_steps"], "| IV-goal", v["IV-goal states"], "| IV-chunked", v["IV-chunked states"])
