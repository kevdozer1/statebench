"""Turn 18 raised spend cap (Kevin's interjection): the turn cap is $16. ``spend18.py`` is pinned by ``rules_turn18``
and stays unchanged; this file reads the same frozen start figure (``runs/turn18/spend_start.json``) and adds $16, and
logs to the same calls file. Every call made after the interjection (A5, the bigger Statue, the A5 runs, Study B) uses
this guard. Drop order if a projection passes the cap: Study B's second execution sample per source first; A5 and the
48 rated drawings are never dropped.
"""
from __future__ import annotations

import json
import time

from . import spend18 as S
from .spend_guard_v13 import SpendGuardV13

TURN_BUDGET_USD = 16.0


def cap_usd() -> float:
    start = S.freeze_start()
    r = S.DIR / "spend_raise.json"
    if not r.exists():
        r.write_text(json.dumps({"reason": "Kevin's Turn 18 interjection: raise the turn cap to $16", "budget_usd": TURN_BUDGET_USD,
                                 "cap_usd": round(start["lifetime_at_turn_start_usd"] + TURN_BUDGET_USD, 8),
                                 "raised_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}, indent=1))
    return json.loads(r.read_text())["cap_usd"]


def guard() -> SpendGuardV13:
    return SpendGuardV13(cap_usd=cap_usd(), calls_path=S.DIR / "spend_calls.jsonl")


if __name__ == "__main__":
    print(cap_usd(), round(guard().headroom(), 3))
