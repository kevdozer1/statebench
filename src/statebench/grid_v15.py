"""Turn 15 grid driver: one task, one reader, several intent conditions, a seed range.

* block and button run through their Turn 14 runners (state condition T for every reader except rules, which uses its
  own T+H reference condition); the intent replaces the goal line (``tasks15.IntentTask``).
* drawing and writing run through a playground-venv scene server per worker thread.
* ``--judge`` (Jev, I4 cells): at every decision one Jev request with one noul question per chunk; the answers and
  the chunk truth are stored per step (diagnostic; the planner does not see them).
* Intent conditions I0-I6 come from ``tasks15``; IV-goal and IV-chunked come from the frozen video-derived intents
  file (``runs/turn15/video_intents.json``), per seed.
Before it runs it prints the expected runtime and, for paid readers, the expected spend and the reader's rate limit.
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

from . import tasks15 as T  # noqa: E402
from . import transport_v13b  # noqa: E402
from .readers_v13 import PAID, PRICES  # noqa: E402
from .config import RUNS

PAID15 = dict(PAID, sol="openai/gpt-6-sol")
PRICES.setdefault("openai/gpt-6-sol", (2e-6, 1e-5))  # listed price (OpenRouter catalogue, 2026-09-24), for the reservations
RATE_LIMITS = {"sonnet": "20 requests per minute (account limit, spaced 3.2 s)", "sol": "not stated; calls not spaced, 429 retried with backoff",
               "jev": "none observed", "local-mid": "local (one GPU)", "rules": "local"}
VIDEO_INTENTS = RUNS / "turn15" / "video_intents.json"


def intent_for(task_name: str, cond: str, seed: int, n_or_task=None) -> dict:
    if cond.startswith("IV"):
        vi = json.loads(VIDEO_INTENTS.read_text())[task_name][str(seed)]
        return {"goal": vi["goal"]} if cond == "IV-goal" else {"goal": vi["goal"], "chunks": tuple(tuple(c) for c in vi["chunks"])}
    if task_name == "block":
        return T.block_intents()[cond]
    if task_name == "button":
        return T.button_intents()[cond]
    raise ValueError(task_name)


class JevJudge:
    """One Jev request per decision, one noul question per chunk."""

    def __init__(self):
        from . import jev_client_v6  # noqa: F401
        from .backends import jev

        self.b = jev.JevBackend()
        self.jev = jev

    def __call__(self, prompt: str, task) -> dict | None:
        chunks = task.intent_chunks
        if not chunks:
            return None
        qs = {f"c{i}": {"type": "noul", "instructions": f"Is this step complete: {s}: done when {c}"}
              for i, (s, c) in enumerate(chunks)}
        res = self.jev.post_json(self.b.url, {"model": self.b.model, "state": prompt, "questions": qs}, self.b.key, timeout=90.0)
        if not res.ok:
            return {"error": f"http {res.status}"}
        ans = (res.payload or {}).get("answers") or {}
        u = (res.payload or {}).get("usage") or {}
        return {"p_complete": [ans.get(f"c{i}", {}).get("noul") for i in range(len(chunks))],
                "tokens_in": u.get("input_tokens"), "latency_s": round(res.latency_seconds, 3)}


class JudgedReader:
    """Wraps a reader for block/button: asks the judge on the same prompt, then decides."""

    def __init__(self, reader, judge, task):
        self.reader, self.judge, self.task, self.log = reader, judge, task, []
        self.name = reader.name

    def decide(self, prompt, view):
        self.log.append(self.judge(prompt, self.task))
        return self.reader.decide(prompt, view)


def make_reader(name: str, task, condition: str, guard):
    from .readers_v14 import JevReader, OllamaReader, OpenRouterReader

    if name == "rules":
        if task.name == "drawing":
            return T.RulesDraw()
        if task.name == "writing":
            return T.RulesWrite()
        from .readers_v14 import make_reader as mk

        return mk("rules", task, "T+H")
    if name in ("local-mid", "local-small"):
        return OllamaReader(name, task)
    if name == "jev":
        return JevReader(task)
    if name in ("sonnet", "sol"):
        r = OpenRouterReader.__new__(OpenRouterReader)
        from .backends.base import load_secret

        r.name, r.model, r.task, r.guard, r.bucket, r.max_tokens = name, PAID15[name], task, guard, "main", 64
        r.key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
        return r
    raise ValueError(name)


def run_one(task_name: str, cond: str, seed: int, reader_name: str, guard, server, judge, keep_prompts: bool) -> dict:
    if task_name in ("drawing", "writing"):
        fn = T.drawing_intents if task_name == "drawing" else T.writing_intents
        if cond.startswith("IV"):
            vi = json.loads(VIDEO_INTENTS.read_text())[task_name][str(seed)]
            intent = {"goal": vi["goal"]} if cond == "IV-goal" else {"goal": vi["goal"], "chunks": tuple(tuple(c) for c in vi["chunks"])}
        else:
            intent = None
        placeholder = T.drawing_task(1, {"goal": "x"}) if task_name == "drawing" else T.writing_task({"goal": "x"})
        rd = make_reader(reader_name, placeholder, cond, guard)
        rec = T.run_server_episode(server, task_name, seed, intent if intent is not None else cond, fn, rd,
                                   keep_prompts=keep_prompts, judge=judge)
        rec["condition"] = cond
        return rec
    from . import button_task_v14, runner_v14

    intent = intent_for(task_name, cond, seed)
    task = T.block_task(intent) if task_name == "block" else T.button_task(intent)
    rd = make_reader(reader_name, task, cond, guard)
    state_cond = "T+H" if reader_name == "rules" else "T"
    if judge is not None:
        rd = JudgedReader(rd, judge, task)
    if task_name == "block":
        rec = runner_v14.run_episode(seed, state_cond, rd, task=task, keep_prompts=keep_prompts)
    else:
        rec = button_task_v14.run_episode(seed, state_cond, rd, task=task, keep_prompts=keep_prompts)
    if judge is not None:
        for s, j in zip(rec["steps"], rd.log):
            s["judge"] = j
    rec.update(task=task_name, condition=cond, state_condition=state_cond, reader=reader_name)
    return rec


def run_grid(task_name: str, out: str, reader_name: str, conditions: list[str], seeds, threads: int = 4,
             s_per_episode: float = 20.0, calls_per_episode: float = 10.0, judge: bool = False,
             keep_prompts: bool = False) -> dict:
    guard = None
    if reader_name in PAID15:
        from .spend_guard_v13 import SpendGuardV13

        guard = SpendGuardV13()
    done = set()
    p = Path(out)
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            done.add((r.get("task"), r["reader"], r["condition"], r["seed"]))
    jobs = [(c, s) for c in conditions for s in seeds if (task_name, reader_name, c, s) not in done]
    est_min = len(jobs) * s_per_episode / max(1, threads) / 60
    if reader_name in ("sonnet", "sol"):
        est_min = max(est_min, len(jobs) * calls_per_episode * transport_v13b.SONNET_MIN_INTERVAL_S / 60)
    est = {"grid": f"turn15 {task_name} {reader_name} {','.join(conditions)}", "episodes": len(jobs), "threads": threads,
           "expected_runtime_min": round(est_min, 1), "rate_limit": RATE_LIMITS.get(reader_name, "-")}
    if guard is not None:
        pin, pout = PRICES.get(PAID15[reader_name], (2e-6, 1e-5))
        per_call = (3500 / 2.5 + 50) * pin + 64 * pout
        est.update(expected_spend_upper_usd=round(len(jobs) * calls_per_episode * per_call, 3), headroom_usd=round(guard.headroom(), 3),
                   slug=PAID15[reader_name], price_per_token=[pin, pout])
    print(json.dumps(est), flush=True)
    t0 = time.monotonic()
    lock = threading.Lock()
    servers: list = []
    tls = threading.local()
    jd = JevJudge() if judge else None

    def work(job):
        cond, seed = job
        if task_name in ("drawing", "writing"):
            if not hasattr(tls, "server"):
                tls.server = T.SceneServer("draw_scene.py" if task_name == "drawing" else "write_scene.py")
                with lock:
                    servers.append(tls.server)
            srv = tls.server
        else:
            srv = None
        return run_one(task_name, cond, seed, reader_name, guard, srv, jd, keep_prompts)

    stopped, n = None, 0
    with ThreadPoolExecutor(max_workers=threads) as pool, open(p, "a") as fh:
        futs = {pool.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as ex:  # noqa: BLE001
                stopped = f"{type(ex).__name__}: {ex}"
                print(json.dumps({"job_error": stopped, "job": futs[f]}), flush=True)
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
    for s in servers:
        s.close()
    summary = {"wall_min": round((time.monotonic() - t0) / 60, 2), "episodes_written": n, "errors": stopped,
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
             float(arg("--s-per-episode", "20")), float(arg("--calls-per-episode", "10")), "--judge" in a,
             "--keep-prompts" in a)
