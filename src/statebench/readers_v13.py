"""Turn 13 readers: every planner reads the frozen ``prompt_v13`` text and returns one skill.

``decide(prompt, view)`` returns a dict: skill, valid (the reader's own answer parsed as one of the skills), tokens
in and out, latency (wall seconds), cost (USD, OpenRouter only), error. An invalid or failed answer becomes
``inspect`` (the runner convention since Turn 2) and is counted.

* ``OllamaReader``: a local open-weights model through Ollama's ``/api/chat``; the output is constrained by the
  JSON schema whose ``skill`` enum is the skill list; thinking off; temperature 0, fixed seed.
* ``OpenRouterReader``: chat completions with a strict JSON schema; reasoning off; temperature 0. Every call
  reserves its upper-bound cost with ``spend_guard_v13`` before it is sent and is reconciled with the call's own
  ``usage.cost`` after.
* ``JevReader``: TypeSafe System One ``choice`` with the declared User-Agent (``jev_client_v6``). The ``state`` is
  the full prompt text (byte-identical to the other readers' message); ``instructions`` is the prompt's first line;
  ``criteria`` are the same skill lines.
* ``RulesReader``: ``rules_v2``'s decision table on the view's fields (needs the T+M+P fields).
"""

from __future__ import annotations

import json
import time
import urllib.request

from . import prompt_v13 as PV

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LOCAL_MODELS = {"local-small": "qwen3.5:4b", "local-mid": "qwen3.5:9b"}
PAID = {"sonnet": "anthropic/claude-sonnet-5", "astra": "openai/gpt-6-astra"}
#: listed prices per token, read from the OpenRouter catalogue on 2026-09-23 (runs/turn13/openrouter_models_20260923.json)
PRICES = {"anthropic/claude-sonnet-5": (2e-6, 1e-5), "openai/gpt-6-astra": (1e-5, 5e-5)}
CHARS_PER_TOKEN_FLOOR = 2.5  # upper-bound token estimate for reservations: len(text) / 2.5


def _result(skill=None, valid=False, tin=None, tout=None, lat=0.0, cost=None, error=None, raw=None):
    return {"skill": skill if (valid and skill in PV.SKILLS) else "inspect", "valid": bool(valid and skill in PV.SKILLS),
            "raw_skill": skill, "tokens_in": tin, "tokens_out": tout, "latency_s": round(lat, 4),
            "cost_usd": cost, "error": error, "raw": raw}


def _parse(content: str):
    try:
        s = json.loads(content)["skill"]
        return s, s in PV.SKILLS
    except Exception:  # noqa: BLE001
        return None, False


class OllamaReader:
    def __init__(self, name: str, num_ctx: int = 4096, seed: int = 0):
        self.name, self.model = name, LOCAL_MODELS[name]
        self.num_ctx, self.seed = num_ctx, seed

    def info(self) -> dict:
        return {"reader": self.name, "model": self.model, "runtime": "ollama", "think": False, "temperature": 0}

    def decide(self, prompt: str, view: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "format": PV.output_schema(),
                "think": False, "stream": False, "keep_alive": "60m",
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx, "num_predict": 32}}
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                d = json.load(resp)
        except Exception as ex:  # noqa: BLE001
            return _result(lat=time.perf_counter() - t0, error=f"{type(ex).__name__}: {ex}"[:300])
        content = (d.get("message") or {}).get("content", "")
        skill, ok = _parse(content)
        return _result(skill, ok, d.get("prompt_eval_count"), d.get("eval_count"), time.perf_counter() - t0,
                       error=None if ok else f"unparsed: {content[:200]}", raw=content[:200])


class OpenRouterReader:
    def __init__(self, name: str, guard, bucket: str = "main", max_tokens: int = 64, reasoning_off: bool = True):
        from .backends.base import load_secret

        self.name, self.model = name, PAID[name]
        self.guard, self.bucket, self.max_tokens, self.reasoning_off = guard, bucket, max_tokens, reasoning_off
        self.key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")

    def info(self) -> dict:
        return {"reader": self.name, "model": self.model, "runtime": "openrouter", "temperature": 0,
                "reasoning_off": self.reasoning_off, "max_tokens": self.max_tokens,
                "price_per_token": PRICES[self.model]}

    def estimate(self, prompt: str) -> float:
        from .spend_guard_v13 import estimate_usd

        return estimate_usd(int(len(prompt) / CHARS_PER_TOKEN_FLOOR) + 50, self.max_tokens, *PRICES[self.model])

    def decide(self, prompt: str, view: dict) -> dict:
        from .backends.http import post_json

        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens, "usage": {"include": True},
                "response_format": {"type": "json_schema", "json_schema": {"name": "skill_choice", "strict": True,
                                                                           "schema": PV.output_schema()}},
                "messages": [{"role": "user", "content": prompt}]}
        if self.reasoning_off:
            body["reasoning"] = {"enabled": False}
        rid = self.guard.reserve(self.estimate(prompt), f"{self.name}", bucket=self.bucket)
        res = post_json("https://openrouter.ai/api/v1/chat/completions", body, self.key, timeout=120.0)
        if not res.ok:
            self.guard.reconcile(rid, None)
            return _result(lat=res.latency_seconds, error=f"http {res.status}: {res.error}"[:300])
        raw = res.payload or {}
        usage = raw.get("usage") or {}
        cost = usage.get("cost")
        self.guard.reconcile(rid, float(cost) if cost is not None else None)
        try:
            content = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        skill, ok = _parse(content)
        return _result(skill, ok, usage.get("prompt_tokens"), usage.get("completion_tokens"), res.latency_seconds,
                       cost, None if ok else f"unparsed: {content[:200]}", content[:200])


class JevReader:
    def __init__(self):
        from . import jev_client_v6  # noqa: F401  (declared User-Agent)
        from .backends import jev

        self.name = "jev"
        self.b = jev.JevBackend()

    def info(self) -> dict:
        return {"reader": "jev", "model": self.b.model, "runtime": "typesafe systemone", "endpoint": self.b.url}

    def decide(self, prompt: str, view: dict) -> dict:
        from .backends import jev

        first_line = prompt.split("\n", 1)[0]
        body = {"model": self.b.model, "state": prompt,
                "questions": {"skill": {"type": "choice", "instructions": first_line,
                                        "criteria": {s: PV.SKILL_LINES[s] for s in PV.SKILLS}}}}
        res = jev.post_json(self.b.url, body, self.b.key, timeout=90.0)
        if not res.ok:
            return _result(lat=res.latency_seconds, error=f"http {res.status}: {res.error}"[:300])
        raw = res.payload or {}
        usage = raw.get("usage") or {}
        try:
            skill = raw["answers"]["skill"]["choice"]
        except (KeyError, TypeError):
            skill = None
        ok = skill in PV.SKILLS
        return _result(skill, ok, usage.get("input_tokens"), usage.get("output_tokens"), res.latency_seconds,
                       None, None if ok else f"unparsed: {str(raw)[:200]}", {"model": raw.get("model")})


class RulesReader:
    name = "rules_v2"

    def info(self) -> dict:
        return {"reader": "rules_v2", "model": "r4c-lookup-v2", "runtime": "local table"}

    def decide(self, prompt: str, view: dict) -> dict:
        from .backends.rules_v2 import choose

        g = view.get("grip")
        p = {"estimate_stale": view.get("estimate_stale"), "gripper_closed_empty": g == "empty",
             "gripper_closed_on_object": g == "on_object", "released_at_receptacle": view.get("released_at_receptacle"),
             "object_at_receptacle": view.get("object_at_receptacle"),
             "object_moving_with_gripper": view.get("object_moving_with_gripper"),
             "aligned_for_grasp": view.get("aligned_for_grasp")}
        t0 = time.perf_counter()
        skill, rule = choose(p)
        return _result(skill, True, 0, 0, time.perf_counter() - t0, raw={"rule": rule})


def make_reader(name: str, guard=None, bucket: str = "main"):
    if name in LOCAL_MODELS:
        return OllamaReader(name)
    if name in PAID:
        return OpenRouterReader(name, guard, bucket=bucket)
    if name == "jev":
        return JevReader()
    if name == "rules_v2":
        return RulesReader()
    raise ValueError(name)
