"""Turn 14 grid driver (both tasks). One process, a thread pool; paid readers share one ``SpendGuardV13``.

Before it runs it prints the expected runtime and, for paid readers, the expected spend upper bound. The runtime
estimate includes Sonnet's global spacing (3.2 s per call across threads): for Sonnet it is the larger of the
threaded estimate and episodes x decisions x 3.2 s. Results are appended per episode; (reader, condition, seed)
triples already present are skipped. Every record carries the task name.

Usage::

    python -m statebench.grid_v14 TASK OUT.jsonl READER "COND1,COND2" SEEDS [--threads N] [--s-per-decision S]
        [--decisions-per-episode D] [--keep-prompts]

TASK is ``pick_and_place`` or ``button``; SEEDS is ``a:b:step``.
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

from . import transport_v13b  # noqa: E402
from .readers_v13 import PAID  # noqa: E402
from .readers_v14 import make_reader  # noqa: E402

TYPICAL_PROMPT_CHARS = {"pick_and_place": 3500, "button": 3500}


def task_and_runner(task_name: str):
    if task_name == "pick_and_place":
        from . import prompt_v14, runner_v14

        return prompt_v14.PICK_PLACE_V14, runner_v14.run_episode
    from . import button_task_v14, button_task_v14b  # noqa: F401  (v14b registers the minus-one conditions)

    return button_task_v14.BUTTON_TASK, button_task_v14.run_episode


def run_grid(task_name: str, out: str, reader_name: str, conditions: list[str], seeds, threads: int = 4,
             s_per_decision: float = 1.0, decisions_per_episode: float = 12.0, keep_prompts: bool = False) -> dict:
    task, run_episode = task_and_runner(task_name)
    guard = None
    if reader_name in PAID:
        from .spend_guard_v13 import SpendGuardV13

        guard = SpendGuardV13()
    done = set()
    p = Path(out)
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            done.add((r["reader"], r["condition"], r["seed"]))
    jobs = [(c, s) for c in conditions for s in seeds if (reader_name, c, s) not in done]
    est_min = len(jobs) * decisions_per_episode * s_per_decision / max(1, threads) / 60 + len(jobs) * 1.0 / 60
    if reader_name == "sonnet":
        est_min = max(est_min, len(jobs) * decisions_per_episode * transport_v13b.SONNET_MIN_INTERVAL_S / 60)
    est = {"grid": f"turn14 {task_name} {reader_name} {','.join(conditions)}", "episodes": len(jobs),
           "threads": threads, "expected_runtime_min": round(est_min, 1)}
    if guard is not None:
        rd = make_reader(reader_name, task, None, guard)
        per_call = rd.estimate("x" * TYPICAL_PROMPT_CHARS[task_name] * (2 if any("H" in c for c in conditions) else 1))
        est.update(expected_spend_upper_usd=round(len(jobs) * decisions_per_episode * per_call, 4),
                   headroom_usd=round(guard.headroom(), 4))
    print(json.dumps(est), flush=True)
    t0 = time.monotonic()
    lock = threading.Lock()
    pools: dict = {}

    def work(job):
        cond, seed = job
        with lock:
            free = pools.setdefault(cond, [])
            rd = free.pop() if free else make_reader(reader_name, task, cond, guard)
        try:
            return run_episode(seed, cond, rd, keep_prompts=keep_prompts)
        finally:
            with lock:
                pools[cond].append(rd)

    stopped, n = None, 0
    with ThreadPoolExecutor(max_workers=threads) as pool, open(p, "a") as fh:
        futs = {pool.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as ex:  # noqa: BLE001  (a spend-cap refusal ends the grid)
                stopped = f"{type(ex).__name__}: {ex}"
                for g in futs:
                    g.cancel()
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
    summary = {"wall_min": round((time.monotonic() - t0) / 60, 2), "episodes_written": n, "stopped": stopped,
               "transport": dict(transport_v13b.STATS)}
    if guard is not None:
        summary["headroom_after_usd"] = round(guard.headroom(), 4)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    a = sys.argv
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    lo, hi, st = (int(x) for x in a[5].split(":"))
    run_grid(a[1], a[2], a[3], a[4].split(","), range(lo, hi, st), int(arg("--threads", "4")),
             float(arg("--s-per-decision", "1.0")), float(arg("--decisions-per-episode", "12")), "--keep-prompts" in a)
