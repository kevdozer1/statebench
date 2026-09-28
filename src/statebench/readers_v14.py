"""Turn 14 readers: ``readers_v13`` generalized to a task's skill list, plus the deterministic ``rules`` selectors.

Transport is ``readers_v13``'s, with ``transport_v13b`` installed (retries on 429/529/5xx; Sonnet calls spaced
3.2 s apart across threads). Every paid call reserves and reconciles through ``spend_guard_v13`` (the same cap).

``rules`` is a deterministic selector restricted to the declared observations of each condition; it computes any
derived quantity itself from what the condition provides:
* pick and place, T+M+P (``RulesPP``): the rules_v2 decision table on the view's fields;
* pick and place, T+H (``RulesPPTH``): grip, contact, motion with the gripper, at-receptacle, released,
  aligned-for-grasp, estimate staleness and close attempts since the last lift are recomputed from the view's
  numbers (current geometry, ``tracking_history``, the last 3 actions' events and ``calibration``) with the same
  formulas the state builder uses, then the same table;
* the button task's rules readers are declared in ``button_task_v14``.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request

from . import transport_v13b  # noqa: F401  (retrying, rate-spaced transport)
from .readers_v13 import CHARS_PER_TOKEN_FLOOR, LOCAL_MODELS, OLLAMA_URL, PAID, PRICES

JEV_PRICE_PER_INPUT_TOKEN = 0.042e-6  # USD; output free (Turn 14 brief)


def _result(task, skill=None, valid=False, tin=None, tout=None, lat=0.0, cost=None, error=None, raw=None):
    ok = bool(valid and skill in task.skills)
    return {"skill": skill if ok else "inspect", "valid": ok, "raw_skill": skill, "tokens_in": tin, "tokens_out": tout,
            "latency_s": round(lat, 4), "cost_usd": cost, "error": error, "raw": raw}


def _parse(task, content: str):
    try:
        s = json.loads(content)["skill"]
        return s, s in task.skills
    except Exception:  # noqa: BLE001
        return None, False


class OllamaReader:
    def __init__(self, name: str, task, num_ctx: int = 8192, seed: int = 0):
        self.name, self.model, self.task, self.num_ctx, self.seed = name, LOCAL_MODELS[name], task, num_ctx, seed

    def decide(self, prompt: str, view: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "format": self.task.output_schema(), "think": False, "stream": False, "keep_alive": "60m",
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx, "num_predict": 32}}
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                d = json.load(resp)
        except Exception as ex:  # noqa: BLE001
            return _result(self.task, lat=time.perf_counter() - t0, error=f"local {type(ex).__name__}: {ex}"[:300])
        content = (d.get("message") or {}).get("content", "")
        skill, ok = _parse(self.task, content)
        return _result(self.task, skill, ok, d.get("prompt_eval_count"), d.get("eval_count"), time.perf_counter() - t0,
                       0.0, None if ok else f"unparsed: {content[:200]}", content[:200])


class OpenRouterReader:
    def __init__(self, name: str, task, guard, bucket: str = "main", max_tokens: int = 64):
        from .backends.base import load_secret

        self.name, self.model, self.task, self.guard, self.bucket, self.max_tokens = name, PAID[name], task, guard, bucket, max_tokens
        self.key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")

    def estimate(self, prompt: str) -> float:
        from .spend_guard_v13 import estimate_usd

        return estimate_usd(int(len(prompt) / CHARS_PER_TOKEN_FLOOR) + 50, self.max_tokens, *PRICES[self.model])

    def decide(self, prompt: str, view: dict) -> dict:
        from .backends.http import post_json

        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens, "usage": {"include": True},
                "reasoning": {"enabled": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "skill_choice", "strict": True,
                                                                           "schema": self.task.output_schema()}},
                "messages": [{"role": "user", "content": prompt}]}
        rid = self.guard.reserve(self.estimate(prompt), self.name, bucket=self.bucket)
        res = post_json("https://openrouter.ai/api/v1/chat/completions", body, self.key, timeout=120.0)
        if not res.ok:
            self.guard.reconcile(rid, None)
            return _result(self.task, lat=res.latency_seconds, error=f"http {res.status}: {res.error}"[:300])
        raw = res.payload or {}
        usage = raw.get("usage") or {}
        cost = usage.get("cost")
        self.guard.reconcile(rid, float(cost) if cost is not None else None)
        try:
            content = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        skill, ok = _parse(self.task, content)
        return _result(self.task, skill, ok, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                       res.latency_seconds, cost, None if ok else f"unparsed: {content[:200]}", content[:200])


class JevReader:
    def __init__(self, task):
        from . import jev_client_v6  # noqa: F401  (declared User-Agent)
        from .backends import jev

        self.name, self.task, self.b = "jev", task, jev.JevBackend()

    def decide(self, prompt: str, view: dict) -> dict:
        from .backends import jev

        body = {"model": self.b.model, "state": prompt,
                "questions": {"skill": {"type": "choice", "instructions": prompt.split("\n", 1)[0],
                                        "criteria": {s: self.task.skill_lines[s] for s in self.task.skills}}}}
        res = jev.post_json(self.b.url, body, self.b.key, timeout=90.0)
        if not res.ok:
            return _result(self.task, lat=res.latency_seconds, error=f"http {res.status}: {res.error}"[:300])
        raw = res.payload or {}
        usage = raw.get("usage") or {}
        try:
            skill = raw["answers"]["skill"]["choice"]
        except (KeyError, TypeError):
            skill = None
        tin = usage.get("input_tokens")
        return _result(self.task, skill, skill in self.task.skills, tin, usage.get("output_tokens"), res.latency_seconds,
                       None if tin is None else round(tin * JEV_PRICE_PER_INPUT_TOKEN, 8),
                       None if skill in self.task.skills else f"unparsed: {str(raw)[:200]}", {"model": raw.get("model")})


# ------------------------------------------------------------------ deterministic selectors (pick and place)
def _table(p: dict) -> tuple[str, str]:
    from .backends.rules_v2 import choose

    return choose(p)


class RulesPP:
    name = "rules"

    def __init__(self, task):
        self.task = task

    def decide(self, prompt: str, view: dict) -> dict:
        t0 = time.perf_counter()
        g = view.get("grip")
        p = {"estimate_stale": view.get("estimate_stale"), "gripper_closed_empty": g == "empty",
             "gripper_closed_on_object": g == "on_object", "released_at_receptacle": view.get("released_at_receptacle"),
             "object_at_receptacle": view.get("object_at_receptacle"),
             "object_moving_with_gripper": view.get("object_moving_with_gripper"),
             "aligned_for_grasp": view.get("aligned_for_grasp")}
        skill, rule = _table(p)
        return _result(self.task, skill, True, 0, 0, time.perf_counter() - t0, 0.0, raw={"rule": rule, "p": p})


def derive_pp_from_th(view: dict, last_actions_events: list[list[str]]) -> dict:
    """Recompute the rules_v2 inputs from the T+H numbers only."""
    c = view["calibration"]
    op, cmd = float(view["gripper_opening_m"]), view["gripper_command"]
    th = (c["gripper_opening_closed_on_nothing_m"] + c["object_size_m"] - c["gripper_opening_held_offset_m"]) / 2
    closed = cmd == "closed"
    on_object = bool(closed and th < op <= c["release_opening_m"])
    empty = bool(closed and op <= th)
    rg, rr = view["target_rel_gripper_m"], view["target_rel_receptacle_m"]
    # motion with the gripper: the tracked samples at t_rel 0 and -motion_window_s
    hist = {round(s["t_rel_s"] - view["tracking_history"][0]["t_rel_s"], 3): s for s in view["tracking_history"]}
    s0, s1 = hist.get(0.0), hist.get(-c["motion_window_s"])
    moving = "unknown"
    if s0 is not None and s1 is not None:
        dtcp = math.dist([s0["gripper_tcp_m"][k] for k in "xyz"], [s1["gripper_tcp_m"][k] for k in "xyz"])
        if dtcp >= c["motion_min_gripper_travel_m"]:
            drel = math.dist([s0["target_rel_gripper_m"][k] for k in ("dx", "dy", "dz")],
                             [s1["target_rel_gripper_m"][k] for k in ("dx", "dy", "dz")])
            moving = bool(drel < c["motion_match_m"])
    at_rec = bool(abs(rr["dx"]) < c["receptacle_max_xy_m"] and abs(rr["dy"]) < c["receptacle_max_xy_m"]
                  and 0.0 <= rr["dz"] < c["receptacle_max_dz_m"])
    last = last_actions_events[-1] if last_actions_events else []
    released = bool(last and last[-1] == "release_done" and at_rec)
    aligned = bool(abs(rg["dx"]) < c["align_max_xy_m"] and abs(rg["dy"]) < c["align_max_xy_m"]
                   and abs(rg["dz"] + c["grasp_height_offset_m"]) < c["align_max_z_error_m"]
                   and abs(view["target_yaw_rel_gripper_rad"]) < c["align_max_yaw_rad"])
    contact_now = bool(c["fingers_on_something_opening_m"] < op <= c["release_opening_m"]
                       and math.hypot(rg["dx"], rg["dy"]) < c["contact_max_xy_m"] and abs(rg["dz"]) < c["contact_max_z_m"])
    stale = bool((contact_now or on_object) and view["estimate_age_s"] > c["estimate_max_age_s"])
    return {"estimate_stale": stale, "gripper_closed_empty": empty, "gripper_closed_on_object": on_object,
            "released_at_receptacle": released, "object_at_receptacle": at_rec, "object_moving_with_gripper": moving,
            "aligned_for_grasp": aligned}


class RulesPPTH:
    name = "rules"

    def __init__(self, task):
        self.task = task

    def decide(self, prompt: str, view: dict) -> dict:
        t0 = time.perf_counter()
        # the last actions' events, as the prompt lists them (the condition provides them in the prompt text)
        acts = []
        block = prompt.split("ACTIONS (oldest first)", 1)[1].split("\n\nSTATE:", 1)[0]
        for line in block.strip().split("\n")[1:]:
            if ":" in line and not line.startswith("(none"):
                ev = line.split(":", 1)[1].strip()
                acts.append([] if ev == "no event reported" else [e.strip() for e in ev.split(",")])
        p = derive_pp_from_th(view, acts)
        skill, rule = _table(p)
        return _result(self.task, skill, True, 0, 0, time.perf_counter() - t0, 0.0, raw={"rule": rule, "p": p})


def make_reader(name: str, task, condition: str | None = None, guard=None, bucket: str = "main"):
    if name in LOCAL_MODELS:
        return OllamaReader(name, task)
    if name in PAID:
        return OpenRouterReader(name, task, guard, bucket=bucket)
    if name == "jev":
        return JevReader(task)
    if name == "rules":
        if task.name == "pick_and_place":
            return RulesPPTH(task) if condition == "T+H" else RulesPP(task)
        from .button_task_v14 import make_rules

        return make_rules(task, condition)
    raise ValueError(name)
