"""Turn 19 spend cap: the Turn 13 guard (``spend_guard_v13``, unchanged) with a Turn 19 start file.

The cap is the OpenRouter lifetime usage at turn start plus ``TURN_BUDGET_USD`` (10 dollars, the brief's cap). Before any
paid run the API key's own limit is checked as well as the account balance (``key_status``). Drop order: the color demo
samples beyond 1 per reference, then Phase 4's preview renders; Phase 4's rated drawings are never dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import spend_guard_v13 as G
from .config import RUNS

TURN_BUDGET_USD = 10.0
DIR = RUNS / "turn19"


def freeze_start() -> dict:
    p = DIR / "spend_start.json"
    if p.exists():
        return json.loads(p.read_text())
    DIR.mkdir(parents=True, exist_ok=True)
    old = G.TURN_BUDGET_USD
    G.TURN_BUDGET_USD = TURN_BUDGET_USD
    try:
        return G.freeze_start(path=p)
    finally:
        G.TURN_BUDGET_USD = old


def guard() -> G.SpendGuardV13:
    return G.SpendGuardV13(cap_usd=freeze_start()["cap_usd"], calls_path=DIR / "spend_calls.jsonl")


def key_status() -> dict:
    """The API key's own limit and the account balance (no key material is printed)."""
    import urllib.request

    from .backends.base import load_secret

    key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
    out = {}
    for url in ("https://openrouter.ai/api/v1/key", "https://openrouter.ai/api/v1/credits"):
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + key}), timeout=30))["data"]
        out.update({k: d[k] for k in ("limit", "limit_remaining", "total_credits", "total_usage") if k in d})
    del key
    out["account_balance"] = round(float(out["total_credits"]) - float(out["total_usage"]), 4)
    return out


if __name__ == "__main__":
    print(json.dumps(freeze_start(), indent=1))
