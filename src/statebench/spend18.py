"""Turn 18 spend cap: the Turn 13 guard (``spend_guard_v13``, unchanged) with a Turn 18 start file.

The cap is the OpenRouter lifetime usage at turn start plus ``TURN_BUDGET_USD`` (12 dollars, the brief's cap). The start
figure is read once and stored in ``runs/turn18/spend_start.json``; every Turn 18 guard reads the cap from that file and
logs its calls to ``runs/turn18/spend_calls.jsonl``. Drop order if a projection passes the cap: Statue of Liberty samples
1 and 2, then Study B's second execution sample per source, then extra dev passes. Study A's rated drawings are never
dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import spend_guard_v13 as G
from .config import RUNS

TURN_BUDGET_USD = 12.0
DIR = RUNS / "turn18"


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


if __name__ == "__main__":
    print(json.dumps(freeze_start(), indent=1))
