"""Spend guard v13: reserve before every paid call, reconcile after, stop before the cap.

Turn 13's cap is the OpenRouter lifetime usage when the turn starts plus 28 dollars. The start figure is read once
(``freeze_start``) and stored in ``runs/turn13/spend_start.json``; every later guard reads the cap from that file.

Each paid call goes through three steps:

1. ``reserve(estimate_usd, label, bucket)``: the estimate is an upper bound (all prompt tokens at the listed input
   price plus ``max_tokens`` at the listed output price). It is refused, before anything is sent, when
   ``lifetime + open reservations + estimate`` would pass the cap, or when the bucket's own sub-cap would be passed
   (the Astra probe has a hard 3 dollar bucket).
2. The call is sent.
3. ``reconcile(rid, actual_usd)``: the reservation is released and the call's own ``usage.cost`` is recorded in
   the pinned Turn 5 ledger (``spend_guard.LedgerGuard``), so the lifetime figure includes it at once, even when
   the usage endpoint lags. When the actual cost is unknown (a failed or truncated response), the full estimate is
   recorded instead.

``lifetime`` is the larger of the usage endpoint and the ledger's last known figure plus local costs (the Turn 5
rule), cached for ``refresh_s`` seconds so it does not add an HTTP round trip to every call. Every reservation and
reconciliation is appended to ``runs/turn13/spend_calls.jsonl``. Keys are never logged.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from pathlib import Path
from typing import Callable

from .spend_guard import LedgerGuard

TURN_BUDGET_USD = 28.0
BUCKET_CAPS_USD = {"astra": 3.0}


class SpendCapReached(RuntimeError):
    pass


def turn_dir() -> Path:
    from .config import data_root

    return data_root() / "runs" / "turn13"


def freeze_start(endpoint: Callable[[], float | None] | None = None, path: Path | None = None,
                 ledger_path: Path | None = None) -> dict:
    """Read the lifetime figure once, at turn start, and fix the cap from it. Refuses to overwrite."""
    path = Path(path) if path is not None else turn_dir() / "spend_start.json"
    if path.exists():
        return json.loads(path.read_text())
    g = LedgerGuard(cap_usd=1e9, ledger_path=ledger_path, endpoint=endpoint)
    start = g.lifetime()
    rec = {"lifetime_at_turn_start_usd": round(start, 8), "endpoint_answered": g.last_endpoint is not None,
           "turn_budget_usd": TURN_BUDGET_USD, "cap_usd": round(start + TURN_BUDGET_USD, 8),
           "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def estimate_usd(prompt_tokens: int, max_output_tokens: int, price_in_per_tok: float, price_out_per_tok: float,
                 per_request_usd: float = 0.0) -> float:
    """Upper bound on one call's cost from listed per-token prices."""
    return prompt_tokens * price_in_per_tok + max_output_tokens * price_out_per_tok + per_request_usd


class SpendGuardV13:
    def __init__(self, cap_usd: float | None = None, ledger_path: Path | None = None,
                 endpoint: Callable[[], float | None] | None = None, calls_path: Path | None = None,
                 bucket_caps: dict[str, float] | None = None, refresh_s: float = 60.0):
        if cap_usd is None:
            cap_usd = json.loads((turn_dir() / "spend_start.json").read_text())["cap_usd"]
        self.cap = float(cap_usd)
        self.ledger = LedgerGuard(cap_usd=self.cap, ledger_path=ledger_path, endpoint=endpoint, margin_usd=0.0)
        self.calls_path = Path(calls_path) if calls_path is not None else turn_dir() / "spend_calls.jsonl"
        self.bucket_caps = dict(BUCKET_CAPS_USD if bucket_caps is None else bucket_caps)
        self.refresh_s = refresh_s
        self.open: dict[int, dict] = {}
        self.spent_by_bucket: dict[str, float] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._cached: tuple[float, float] | None = None  # (monotonic time, lifetime)
        self._load_buckets()

    # ------------------------------------------------------------ bookkeeping
    def _load_buckets(self) -> None:
        if self.calls_path.exists():
            for line in self.calls_path.read_text().splitlines():
                r = json.loads(line)
                if r["event"] == "reconcile":
                    self.spent_by_bucket[r["bucket"]] = self.spent_by_bucket.get(r["bucket"], 0.0) + r["recorded_usd"]

    def _log(self, rec: dict) -> None:
        self.calls_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calls_path, "a") as fh:
            fh.write(json.dumps({**rec, "at": time.strftime("%Y-%m-%d %H:%M:%S %z")}) + "\n")

    def lifetime(self, fresh: bool = False) -> float:
        now = time.monotonic()
        if fresh or self._cached is None or now - self._cached[0] > self.refresh_s:
            self._cached = (now, self.ledger.lifetime())
            return self._cached[1]
        # between refreshes, the ledger's local figure still moves with every reconciled call
        return max(self._cached[1], self.ledger.local_estimate())

    def reserved(self) -> float:
        return sum(r["usd"] for r in self.open.values())

    def bucket_total(self, bucket: str) -> float:
        return self.spent_by_bucket.get(bucket, 0.0) + sum(r["usd"] for r in self.open.values() if r["bucket"] == bucket)

    def headroom(self) -> float:
        with self._lock:
            return self.cap - self.lifetime() - self.reserved()

    # ------------------------------------------------------------ the protocol
    def reserve(self, estimate: float, label: str, bucket: str = "main") -> int:
        estimate = float(estimate)
        with self._lock:
            life = self.lifetime()
            total = life + self.reserved() + estimate
            if total > self.cap:
                self._log({"event": "refused", "label": label, "bucket": bucket, "estimate_usd": estimate,
                           "lifetime_usd": life, "reserved_usd": self.reserved(), "cap_usd": self.cap})
                raise SpendCapReached(f"reserving {estimate:.6f} would bring the reserved lifetime total to "
                                      f"{total:.6f}, past the cap {self.cap:.6f}")
            bcap = self.bucket_caps.get(bucket)
            if bcap is not None and self.bucket_total(bucket) + estimate > bcap:
                self._log({"event": "refused_bucket", "label": label, "bucket": bucket, "estimate_usd": estimate,
                           "bucket_total_usd": self.bucket_total(bucket), "bucket_cap_usd": bcap})
                raise SpendCapReached(f"bucket {bucket}: reserving {estimate:.6f} would pass its {bcap:.2f} cap")
            rid = next(self._ids)
            self.open[rid] = {"usd": estimate, "label": label, "bucket": bucket}
            self._log({"event": "reserve", "rid": rid, "label": label, "bucket": bucket, "estimate_usd": estimate,
                       "lifetime_usd": life, "reserved_after_usd": self.reserved()})
            return rid

    def reconcile(self, rid: int, actual_usd: float | None) -> float:
        with self._lock:
            r = self.open.pop(rid)
            recorded = r["usd"] if actual_usd is None else float(actual_usd)
            self.ledger.record(recorded, f"turn13 {r['bucket']} {r['label']}")
            self.spent_by_bucket[r["bucket"]] = self.spent_by_bucket.get(r["bucket"], 0.0) + recorded
            self._log({"event": "reconcile", "rid": rid, "label": r["label"], "bucket": r["bucket"],
                       "estimate_usd": r["usd"], "actual_usd": actual_usd, "recorded_usd": recorded,
                       "over_estimate": actual_usd is not None and actual_usd > r["usd"]})
            return recorded


def fake_price_test(tmp: Path) -> dict:
    """The guard on a fake price table and a fake endpoint. Touches no real ledger and sends nothing."""
    tmp.mkdir(parents=True, exist_ok=True)
    for f in ("ledger.json", "calls.jsonl", "start.json"):
        (tmp / f).unlink(missing_ok=True)
    reported = {"v": 10.0}
    endpoint = lambda: reported["v"]  # noqa: E731
    start = freeze_start(endpoint=endpoint, path=tmp / "start.json", ledger_path=tmp / "ledger.json")
    assert start["cap_usd"] == 38.0, start
    assert freeze_start(endpoint=lambda: 99.0, path=tmp / "start.json", ledger_path=tmp / "ledger.json")["cap_usd"] == 38.0  # frozen, not re-read
    prices = {"fake/model": (2e-6, 8e-6)}  # per token: 2 and 8 dollars per million
    est = estimate_usd(1_000_000, 250_000, *prices["fake/model"])  # 2 + 2 = 4 dollars
    assert abs(est - 4.0) < 1e-12
    g = SpendGuardV13(cap_usd=start["cap_usd"], ledger_path=tmp / "ledger.json", endpoint=endpoint,
                      calls_path=tmp / "calls.jsonl", bucket_caps={"astra": 3.0}, refresh_s=0.0)
    out = {"cap": g.cap}
    # 1. seven 4-dollar calls, each actually costing 3: 10 + 7*3 = 31; the 8th reserve (31 + 4 = 35) passes; 9th
    #    (34 + 4 = 38) passes exactly at the cap; 10th (37 + 4 = 41) is refused.
    done = 0
    try:
        while True:
            rid = g.reserve(est, f"call{done}")
            g.reconcile(rid, 3.0)
            done += 1
    except SpendCapReached:
        pass
    out["calls_before_stop"] = done
    assert done == 9, done
    assert abs(g.lifetime() - 37.0) < 1e-9
    # 2. open reservations count: with 1 dollar left, two concurrent 0.6 reservations cannot both be open
    rid = g.reserve(0.6, "concurrent a")
    try:
        g.reserve(0.6, "concurrent b")
        raise AssertionError("second reservation should be refused")
    except SpendCapReached:
        out["open_reservation_counted"] = True
    g.reconcile(rid, None)  # unknown actual cost: the estimate is recorded
    assert abs(g.lifetime() - 37.6) < 1e-9
    # 3. a lagging endpoint cannot lower the figure; a leading one raises it
    reported["v"] = 12.0
    assert abs(g.lifetime(fresh=True) - 37.6) < 1e-9
    reported["v"] = 37.9
    assert abs(g.lifetime(fresh=True) - 37.9) < 1e-9
    # 4. bucket sub-cap (astra 3 dollars) on a fresh guard with plenty of main headroom
    reported["v"] = 0.0
    (tmp / "ledger.json").unlink()
    g2 = SpendGuardV13(cap_usd=100.0, ledger_path=tmp / "ledger.json", endpoint=endpoint,
                       calls_path=tmp / "calls_b.jsonl", bucket_caps={"astra": 3.0}, refresh_s=0.0)
    n = 0
    try:
        while True:
            g2.reconcile(g2.reserve(0.7, f"astra{n}", bucket="astra"), 0.7)
            n += 1
    except SpendCapReached:
        pass
    assert n == 4, n  # 4 * 0.7 = 2.8; a 5th would make 3.5
    out["astra_calls_before_bucket_stop"] = n
    out["refusals_logged"] = sum(1 for x in (tmp / "calls.jsonl").read_text().splitlines()
                                 if json.loads(x)["event"].startswith("refused"))
    out["passed"] = True
    return out


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "test":
        import tempfile

        print(json.dumps(fake_price_test(Path(tempfile.mkdtemp(prefix="sg13_"))), indent=1))
    elif sys.argv[1] == "freeze":
        print(json.dumps(freeze_start(), indent=1))
