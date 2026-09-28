"""Turn 17 spend cap: the Turn 13 guard (``spend_guard_v13``, unchanged) with a Turn 17 start file.

The Turn 17 cap is the OpenRouter lifetime usage at turn start ($41.51) plus ``TURN_BUDGET_USD`` (12 dollars, the
brief's turn cap; raised to 22 dollars mid-turn, see ``cap_usd``). The start figure is read once and stored in ``runs/turn17/spend_start.json``; every Turn 17 guard
reads the cap from that file and logs its calls to ``runs/turn17/spend_calls.jsonl``. If a projection passes the cap,
cells are dropped in the brief's order: the Phase 5 frames arm, the Phase 4 cheap model, Phase 2 drawing samples 4
and 5.
"""
from __future__ import annotations

from pathlib import Path

from . import spend_guard_v13 as G
from .config import RUNS

TURN_BUDGET_USD = 12.0
DIR = RUNS / "turn17"


def freeze_start() -> dict:
    import json

    p = DIR / "spend_start.json"
    if p.exists():
        return json.loads(p.read_text())
    old = G.TURN_BUDGET_USD
    G.TURN_BUDGET_USD = TURN_BUDGET_USD
    try:
        return G.freeze_start(path=p)
    finally:
        G.TURN_BUDGET_USD = old


RAISED_BUDGET_USD = 22.0  # Kevin, mid-turn: about 10 dollars more than the brief's 12 is available


def cap_usd() -> float:
    """The frozen start figure plus the raised budget (``spend_raise.json`` records the raise)."""
    import json
    import time

    start = freeze_start()
    r = DIR / "spend_raise.json"
    if not r.exists():
        r.write_text(json.dumps({"reason": "Kevin said mid-turn that about 10 dollars more than the brief's 12 dollar cap is "
                                           "available", "budget_usd": RAISED_BUDGET_USD,
                                 "cap_usd": round(start["lifetime_at_turn_start_usd"] + RAISED_BUDGET_USD, 8),
                                 "raised_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}, indent=1))
    return json.loads(r.read_text())["cap_usd"]


def guard() -> G.SpendGuardV13:
    return G.SpendGuardV13(cap_usd=cap_usd(), calls_path=DIR / "spend_calls.jsonl")


if __name__ == "__main__":
    import json

    print(json.dumps(freeze_start(), indent=1))
