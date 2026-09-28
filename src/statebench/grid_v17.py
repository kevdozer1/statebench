"""Turn 17 grid driver: Jev on block, button and drawing with the Turn 17 conditions.

    python -m statebench.grid_v17 TASK OUT CONDS a:b:s [--forced schedule|on|off] [--goal-offset N] [--scene NAME]
                                  [--threads N] [--s-per-episode S]

Conditions:
* ``I3``; ``E``; ``C:<src>``, ``O:<src>``, ``OS:<src>`` with src v1, v2-sol or v2-cheap (``tasks17.variant``);
* ``B-release``, ``B-hold`` (button), ``K-place`` (block I3), ``K-hold`` (block), ``W-contact``, ``W-active``;
* ``V3-req``, ``V3-all`` (and ``V3F-req``, ``V3F-all``, the frames arm), read from ``runs/turn17/v3_goals.json``.
``--goal-offset N``: the goal is built from seed (execution seed - N) (transfer: N = 1). ``--forced``: the forced
failure (``tasks17.set_forced``). ``--scene``: the drawing server script (default ``draw_scene.py``; Phase 6 uses
``draw_scene17.py`` with its finish mode).

Execution is the Turn 16 setting: Jev (``grid_v15.make_reader``), state condition T for block and button, the Turn 15
stroke planner for drawing. Each record adds the condition, the goal text, the goal seed, the forced mode and, for
block and button, the final state with both hold verifiers. The resume key is (task, condition, seed, forced mode,
goal offset, scene). Before it runs it prints the expected runtime (Jev and the scenes cost no OpenRouter money).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from . import grid_v15 as G  # noqa: E402
from . import tasks15 as T  # noqa: E402
from . import tasks17 as T17  # noqa: E402
from . import transport_v13b  # noqa: E402
from .config import RUNS

V3 = RUNS / "turn17" / "v3_goals.json"


def intent(task: str, cond: str, goal_seed: int, n_strokes: int | None = None) -> dict:
    if cond == "I3" or cond == "K-place":
        if task == "drawing":
            return T.drawing_intents(n_strokes)["I3"]
        return (T.block_intents() if task == "block" else T.button_intents())["I3"]
    if cond == "E":
        return T17.variant("E", "", task, goal_seed)
    if cond in T17.GOALS:
        return {"goal": T17.GOALS[cond]}
    if ":" in cond:
        kind, src = cond.split(":")
        return T17.variant(kind, src, task, goal_seed)
    if cond.startswith("V3"):
        v = json.loads(V3.read_text())[cond][task][str(goal_seed)]
        return {"goal": v["goal"]}
    raise ValueError(cond)


def run_one(task: str, cond: str, seed: int, goal_seed: int, server, forced: str, scene: str) -> dict:
    reader_name = "jev"
    if task == "drawing":
        rd = G.make_reader(reader_name, T.drawing_task(1, {"goal": "x"}), cond, None)
        if cond == "I3":  # built from the episode's own stroke count, as in Turn 15
            rec = T.run_server_episode(server, "drawing", seed, "I3", T.drawing_intents, rd)
            it = T.drawing_intents(rec["task_def"]["n_strokes"])["I3"]
        else:
            it = intent(task, cond, goal_seed)
            rec = T.run_server_episode(server, "drawing", seed, it, T.drawing_intents, rd)
    else:
        from . import button_task_v14, runner_v14

        it = intent(task, cond, goal_seed)
        tk = T.block_task(it) if task == "block" else T.button_task(it)
        rd = G.make_reader(reader_name, tk, cond, None)
        if task == "block":
            rec = runner_v14.run_episode(seed, "T", rd, task=tk)
            fin = T17.FINAL.block
            rec["final_state"] = fin
            rec["hold"] = {"K-place": bool(rec["success"]), "K-hold": T17.hold_block(fin)}
        else:
            rec = button_task_v14.run_episode(seed, "T", rd, task=tk)
            fin = T17.FINAL.button
            rec["final_state"] = fin
            rec["hold"] = {"B-release": bool(rec["success"]), "B-hold": T17.hold_button(fin)}
    rec.update(task=task, condition=cond, reader=reader_name, goal=it.get("goal"),
               goal_chunks=[list(c) for c in it.get("chunks", ())], goal_seed=goal_seed, forced_mode=forced, scene=scene)
    return rec


def run_grid(task: str, out: str, conds: list[str], seeds, forced: str = "schedule", goal_offset: int = 0,
             scene: str = "draw_scene.py", threads: int = 6, s_per_episode: float = 12.0) -> dict:
    T17.set_forced(forced)
    T17.install_capture()
    p = Path(out)
    done = set()
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["condition"], r["seed"], r["forced_mode"], r["seed"] - r["goal_seed"], r["scene"]))
    jobs = [(c, s) for c in conds for s in seeds if (task, c, s, forced, goal_offset, scene) not in done]
    print(json.dumps({"grid": f"turn17 {task} jev {','.join(conds)}", "episodes": len(jobs), "forced": forced,
                      "goal_offset": goal_offset, "scene": scene, "threads": threads,
                      "expected_runtime_min": round(len(jobs) * s_per_episode / threads / 60, 1),
                      "expected_openrouter_usd": 0.0, "rate_limit": "Jev: none observed"}), flush=True)
    t0 = time.monotonic()
    lock, servers, tls = threading.Lock(), [], threading.local()

    def work(job):
        c, s = job
        srv = None
        if task == "drawing":
            if not hasattr(tls, "server"):
                tls.server = T.SceneServer(scene)
                if scene != "draw_scene.py":
                    tls.server.call({"cmd": "mode", "finish": os.environ.get("T17_FINISH_MODE", "plain")})
                with lock:
                    servers.append(tls.server)
            srv = tls.server
        return run_one(task, c, s, s - goal_offset, srv, forced, scene)

    n, err = 0, None
    with ThreadPoolExecutor(max_workers=threads) as pool, open(p, "a") as fh:
        futs = {pool.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as ex:  # noqa: BLE001
                err = f"{type(ex).__name__}: {ex}"
                print(json.dumps({"job_error": err, "job": futs[f]}), flush=True)
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
    for s in servers:
        s.close()
    summary = {"wall_min": round((time.monotonic() - t0) / 60, 2), "episodes_written": n, "errors": err,
               "transport": dict(transport_v13b.STATS)}
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    a = sys.argv
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    lo, hi, st = (int(x) for x in a[4].split(":"))
    run_grid(a[1], a[2], a[3].split(","), range(lo, hi, st), arg("--forced", "schedule"), int(arg("--goal-offset", "0")),
             arg("--scene", "draw_scene.py"), int(arg("--threads", "6")), float(arg("--s-per-episode", "12")))
