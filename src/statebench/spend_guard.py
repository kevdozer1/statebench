"""Spend guard v2: a lagging usage endpoint can no longer let a run pass its cap.

Turn 4 passed its $11.00 lifetime cap by $0.0161 because ``turn2.SpendGuard``
seeded itself from OpenRouter's ``/credits`` figure, which had not yet counted
the calls of a run that finished minutes earlier.

This guard keeps a ledger on D: (``$STATEBENCH_DATA/runs/spend_ledger.json``):
the last known lifetime figure and every cost measured locally since. Its idea
of lifetime usage is the **larger** of

* the endpoint figure (when it answers), and
* the ledger's last known figure plus every locally measured cost since,

and it refuses any spend that would bring that figure within ``margin`` (default
$0.25) of the cap. When the endpoint catches up with or passes the ledger, the
ledger rebases on it and clears the local costs it has now absorbed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

DEFAULT_MARGIN_USD = 0.25


def default_ledger_path() -> Path:
    from .config import data_root

    return data_root() / "runs" / "spend_ledger.json"


def default_endpoint() -> float | None:
    return openrouter_usage_usd()


class LedgerGuard:
    def __init__(
        self,
        cap_usd: float,
        ledger_path: Path | None = None,
        endpoint: Callable[[], float | None] | None = None,
        margin_usd: float = DEFAULT_MARGIN_USD,
    ):
        self.cap = float(cap_usd)
        self.margin = float(margin_usd)
        self.path = Path(ledger_path) if ledger_path is not None else default_ledger_path()
        self.endpoint = endpoint or default_endpoint
        self.ledger = self._load()
        self.last_endpoint: float | None = None

    # ----------------------------------------------------------------- ledger
    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"last_known_lifetime_usd": 0.0, "last_known_at": None, "local_costs_since": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.ledger, indent=2), encoding="utf-8")

    def local_estimate(self) -> float:
        return float(self.ledger["last_known_lifetime_usd"]) + sum(
            float(c["usd"]) for c in self.ledger["local_costs_since"])

    # --------------------------------------------------------------- figures
    def lifetime(self) -> float:
        """max(endpoint, last known + local costs since). Rebases when the endpoint catches up."""
        reported = self.endpoint()
        self.last_endpoint = reported
        local = self.local_estimate()
        if reported is not None and reported >= local - 1e-9:
            self.ledger = {"last_known_lifetime_usd": float(reported),
                           "last_known_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                           "local_costs_since": []}
            self._save()
            return float(reported)
        return local if reported is None else max(float(reported), local)

    def record(self, usd: float, label: str = "") -> None:
        """Record a cost measured locally (a call's own usage.cost)."""
        if not usd:
            return
        self.ledger["local_costs_since"].append(
            {"usd": float(usd), "label": label, "at": time.strftime("%Y-%m-%d %H:%M:%S %z")})
        self._save()

    def allows(self, projected_usd: float) -> bool:
        return self.lifetime() + float(projected_usd) + self.margin <= self.cap

    def remaining(self) -> float:
        return max(0.0, self.cap - self.margin - self.lifetime())


def openrouter_usage_usd() -> float | None:
    """OpenRouter's own record of what this key has spent (None when the call fails)."""
    import urllib.request

    key = None
    try:
        from .backends.base import load_secret

        key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
        request = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        return float(data["data"]["total_usage"])
    except Exception:  # noqa: BLE001
        return None
    finally:
        del key
