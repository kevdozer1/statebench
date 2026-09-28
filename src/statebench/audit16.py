"""Turn 16 Phase 1: audit of every video-derived intent (Turn 15 v1, v2 Sol, v2 cheap) on block, button and drawing.

Each intent is audited in its full form (the goal plus the chunks, the form the planner sees; ``chunked``) and as the
goal alone (``goal``):
* the Turn 15 rubric (``audit15.audit_one``, unchanged): block in / released / clear / wrong; button activated /
  released / clear / wrong; drawing all / lifted;
* the **end condition** per task: block clear, button clear, drawing lifted. Its omission rate is the share of intents
  that do not state it;
* the **right-target check** (added this turn, declared after one smoke reply had called the button "a red block on a
  dark base"):
  - block: the intent names a cube or block and a container (tray, receptacle, box, container, bin, basket), and not a
    button;
  - button: the intent names a button, or the object on the dark base (the button sits on a dark base; the parked
    cube does not), and describes no pick-and-place (pick up, place it or the, put ... in or into, into or onto the
    tray, in the tray);
  - drawing: the intent says to draw, trace, sketch, write or outline, and on seeds whose target is the word ABC it
    names ABC or letters. The paintings' coarse outlines cannot be named from the ink, so no name is required.
* the **button wrong-task rate**: button intents that fail the right-target check. The Turn 15 keyword rule (``wrong``:
  tray, pick up, place it or block anywhere in the text) is reported beside it for comparison with Turn 15.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .audit15 import audit_one
from .config import RUNS, SCENES

R = RUNS / "turn16"
END = {"block": "clear", "button": "clear", "drawing": "lifted"}
CONTAINER = r"(tray|receptacle|box|container|bin|basket)"
PICKPLACE = r"(pick (it )?up|place (it|the)|put [^.]* (in|into) |into the tray|onto the tray|in the tray)"


def drawing_target(seed: int) -> str:
    sys.path.insert(0, str(SCENES))
    import targets15 as TG

    return TG.TARGETS[int(__import__("numpy").random.default_rng(1_950_015 + seed).integers(len(TG.TARGETS)))]


def right_target(task: str, text: str, seed: int) -> bool:
    t = text.lower()
    if task == "block":
        return bool(re.search(r"\b(cube|block)s?\b", t) and re.search(CONTAINER, t) and not re.search(r"\bbutton", t))
    if task == "button":
        return bool(re.search(r"(\bbutton|dark base|black base)", t)) and not re.search(PICKPLACE, t)
    if task == "drawing":
        act = bool(re.search(r"(draw|trac|sketch|writ|outlin)", t))
        if drawing_target(seed) == "ABC":
            return act and bool(re.search(r"(\babc\b|letter)", t))
        return act
    raise ValueError(task)


def text_of(intent: dict, full: bool) -> str:
    return intent["goal"] + (" " + " ".join(f"{a} {b}" for a, b in intent.get("chunks", [])) if full else "")


def audit(seeds: set[int], sources=("v1", "v2-sol", "v2-cheap")) -> dict:
    iv = json.loads((R / "iv_intents.json").read_text())
    out: dict = {}
    for src in sources:
        for task in ("block", "button", "drawing"):
            rows = []
            for s, it in sorted(iv.get(src, {}).get(task, {}).items(), key=lambda x: int(x[0])):
                if int(s) not in seeds:
                    continue
                row = {"seed": int(s)}
                for form, full in (("goal", False), ("chunked", True)):
                    tx = text_of(it, full)
                    a = audit_one(task, tx, None)
                    a["end_condition"] = a[END[task]]
                    a["right_target"] = right_target(task, tx, int(s))
                    row[form] = a
                row["n_chunks"] = len(it.get("chunks", []))
                rows.append(row)
            n = len(rows)
            keys = list(rows[0]["chunked"]) if rows else []
            out[f"{src}|{task}"] = {
                "n": n,
                "states_chunked": {k: sum(r["chunked"][k] for r in rows) for k in keys},
                "states_goal": {k: sum(r["goal"][k] for r in rows) for k in keys},
                "end_condition_omitted_chunked": sum(not r["chunked"]["end_condition"] for r in rows),
                "end_condition_omitted_goal": sum(not r["goal"]["end_condition"] for r in rows),
                "wrong_target_chunked": sum(not r["chunked"]["right_target"] for r in rows),
                "rows": rows}
    return out


if __name__ == "__main__":
    a, b, c = (int(x) for x in sys.argv[1].split(":"))
    o = audit(set(range(a, b, c)))
    name = sys.argv[2] if len(sys.argv) > 2 else f"iv_audit_{a}-{b}-{c}.json"
    (R / name).write_text(json.dumps(o, indent=1))
    for k, v in o.items():
        print(k.ljust(18), "n", v["n"], "| end omitted (full / goal)", v["end_condition_omitted_chunked"], "/",
              v["end_condition_omitted_goal"], "| wrong target", v["wrong_target_chunked"], "|", v["states_chunked"])
