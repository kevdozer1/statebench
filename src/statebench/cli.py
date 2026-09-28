"""Three quickstarts: one Sol drawing, one Jev pick-and-place episode, and the field check.

    statebench draw --painting mona_lisa --way one-shot
    statebench pick-place --seed 1600 --fields T --goal complete
    statebench fields --seeds 1600:1610

Model calls need keys (``OPENROUTER_API_KEY`` for Sol, ``TYPESAFE_JEV_API_KEY`` for Jev). Without one, ``draw`` prints the
prompt Sol would get and stops, and the Jev commands run the scripted rules reader instead. Each command says which.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import assets_dir, data_root, env, playground_dir, playground_python

WAYS = {"one-shot": "A1", "rounds": "A2", "rounds-by-part": "A3", "coarse-to-fine": "A4", "parts-with-layout": "A5",
        "rounds-with-coordinates": "A2c", "draft-revise": "R"}
PAINTINGS = ("mona_lisa", "starry_night", "great_wave")
GOALS = ("complete", "no-ending", "none")
FIELDS = ("T", "T+M", "T+M+P")


def _out_dir():
    d = data_root() / "runs" / "quickstart"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ one Sol drawing
def first_prompt(painting: str, cond: str) -> str:
    from . import draw18 as A
    from . import draw18a5 as A5
    from . import draw19 as Q

    if cond == "A1":
        return A.prompt_a1(painting)
    if cond == "A5":
        return A5.prompt(painting, 1, None)
    if cond in ("A2c", "R"):
        return Q.prompt(painting, cond, 1, [])
    return A.prompt_round(painting, cond, 1)


def draw(a) -> None:
    cond = WAYS[a.way]
    if not env("OPENROUTER_API_KEY"):
        print(first_prompt(a.painting, cond))
        print("\nNo OPENROUTER_API_KEY set: above is the first prompt Sol would get. Skipped the paid call.")
        return
    if not playground_python().exists() or not (playground_dir() / "experiments" / "dove-drawing").exists():
        print(f"The drawing robot runs in llm-robotics-playground, which was not found at {playground_dir()} "
              f"(Python {playground_python()}). See the README's install section.")
        return
    from . import draw18 as A
    from . import draw18a5 as A5
    from . import draw19 as Q
    from . import spend_guard_v13 as G
    from . import transport_v13b  # noqa: F401
    from .spend_guard import LedgerGuard

    out = _out_dir()
    lifetime = LedgerGuard(cap_usd=1e9).lifetime()
    guard = G.SpendGuardV13(cap_usd=lifetime + a.cap_usd, calls_path=out / "spend_calls.jsonl")
    print(f"One drawing, {a.way}, of {a.painting}. Spend cap for this run: ${a.cap_usd:.2f}. The robot is slow: "
          "expect a few minutes.", flush=True)
    seed = 1_000 + PAINTINGS.index(a.painting)
    if cond == "A5":
        rec = A5.run_drawing(a.painting, 0, guard, seed, "quickstart")
    elif cond in ("A2c", "R"):
        rec = Q.run_drawing(a.painting, cond, 0, guard, seed, "quickstart")
    else:
        rec = A.run_drawing(a.painting, cond, 0, guard, seed, "quickstart")
    stem = out / f"drawing_{a.painting}_{a.way}"
    stem.with_suffix(".json").write_text(json.dumps(rec))
    A.canvas_image(rec["ink_mm"]).save(stem.with_suffix(".png"))
    s = rec["scores"]
    print(f"{rec['strokes_used']} strokes in {rec['calls']} call(s), ${rec['cost_usd']:.3f}; pen lifted at the end: "
          f"{s['pen_lifted']}; chamfer {s.get('chamfer')}, coverage {s.get('coverage')}")
    print(f"ink: {stem.with_suffix('.png')}")


# ------------------------------------------------------------------ pick and place with Jev
def _task(goal: str):
    from . import prompt_v13 as V13
    from . import prompt_v14 as PV

    if goal == "complete":
        return PV.PICK_PLACE_V14
    return replace(PV.PICK_PLACE_V14, goal=V13.GOAL if goal == "no-ending" else "No goal is given.")


def _reader_name(asked: str) -> str:
    if asked == "jev" and not (env("TYPESAFE_JEV_API_KEY") or env("TYPESAFE_API_KEY")):
        print("No TYPESAFE_JEV_API_KEY set: running the scripted rules reader instead of Jev. It ignores the goal and "
              "needs the meaning labels and procedure fields (T+M+P), so it checks the scene and the loop, not the "
              "findings.", flush=True)
        return "rules"
    return asked


def _assets_ok() -> bool:
    if (assets_dir() / "xarm7.xml").exists():
        return True
    print(f"The xArm7 model was not found at {assets_dir()}. See the README's install section.")
    return False


def _episode(seed: int, fields: str, goal: str, reader_name: str) -> dict:
    from . import readers_v14 as RD
    from . import runner_v14 as RV

    task = _task(goal)
    return RV.run_episode(seed, fields, RD.make_reader(reader_name, task, fields), task=task)


def pick_place(a) -> None:
    if not _assets_ok():
        return
    name = _reader_name(a.reader)
    if name == "rules" and a.fields != "T+M+P":
        print(f"The rules reader cannot act on {a.fields}: using T+M+P.")
        a.fields = "T+M+P"
    r = _episode(a.seed, a.fields, a.goal, name)
    print(f"GOAL: {_task(a.goal).goal}")
    print(" -> ".join(str(s["executed"]) for s in r["steps"]))
    print(f"success: {r['success']} after {r['decision_steps']} decisions ({name}, fields {a.fields}, seed {a.seed})")


def fields(a) -> None:
    if not _assets_ok():
        return
    from concurrent.futures import ThreadPoolExecutor

    name = _reader_name(a.reader)
    lo, hi = (int(x) for x in a.seeds.split(":"))
    jobs = [(g, f, s) for g in GOALS for f in FIELDS for s in range(lo, hi)]
    print(f"{len(jobs)} episodes ({name}); fields T = positions, M = meaning labels, P = procedure fields", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda j: _episode(j[2], j[1], j[0], name)["success"], jobs))
    ok = {}
    for (g, f, _), r in zip(jobs, res):
        ok[(g, f)] = ok.get((g, f), 0) + int(r)
    n = hi - lo
    print(f"\n{'goal':<12}" + "".join(f"{f:>9}" for f in FIELDS))
    for g in GOALS:
        print(f"{g:<12}" + "".join(f"{ok[(g, f)]:>6}/{n:<2}" for f in FIELDS))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="statebench", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="Sol draws a painting; the robot draws Sol's strokes")
    d.add_argument("--painting", choices=PAINTINGS, default="mona_lisa")
    d.add_argument("--way", choices=list(WAYS), default="one-shot")
    d.add_argument("--cap-usd", type=float, default=1.0)
    d.set_defaults(fn=draw)
    pp = sub.add_parser("pick-place", help="one pick-and-place episode")
    pp.add_argument("--seed", type=int, default=1600)
    pp.add_argument("--fields", choices=FIELDS, default="T")
    pp.add_argument("--goal", choices=GOALS, default="complete")
    pp.add_argument("--reader", choices=("jev", "rules"), default="jev")
    pp.set_defaults(fn=pick_place)
    f = sub.add_parser("fields", help="success by goal and state fields, over a range of seeds")
    f.add_argument("--seeds", default="1600:1610")
    f.add_argument("--reader", choices=("jev", "rules"), default="jev")
    f.add_argument("--threads", type=int, default=4)
    f.set_defaults(fn=fields)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
