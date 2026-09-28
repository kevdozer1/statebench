"""The spend guard must hold when the usage endpoint lags. No network, no spend.

Run: python -m unittest tests.test_spend_guard -v
"""

import tempfile
import unittest
from pathlib import Path

from statebench.spend_guard import LedgerGuard


class LaggingEndpoint:
    """Reports true usage minus the most recent `lag` calls' worth of cost."""

    def __init__(self, start: float):
        self.true = start
        self.pending: list[float] = []

    def spend(self, usd: float) -> None:
        self.true += usd
        self.pending.append(usd)

    def __call__(self) -> float:
        # The endpoint has not yet counted the last 5 calls.
        return self.true - sum(self.pending[-5:])


class TestLedgerGuard(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "ledger.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_turn4_scenario_is_refused(self):
        """Turn 4: endpoint said 10.9955 while true usage was 11.0161."""
        guard = LedgerGuard(11.00, self.path, endpoint=lambda: 10.9955)
        guard.ledger = {"last_known_lifetime_usd": 10.6581, "last_known_at": None,
                        "local_costs_since": []}
        for _ in range(87):  # the judge calls that followed, at their logged cost
            guard.record(0.0041)
        self.assertGreater(guard.lifetime(), 10.9955)
        self.assertAlmostEqual(guard.lifetime(), 10.6581 + 87 * 0.0041, places=6)
        self.assertFalse(guard.allows(0.0041))

    def test_lagging_endpoint_never_passes_cap_minus_margin(self):
        end = LaggingEndpoint(7.78)
        guard = LedgerGuard(11.00, self.path, endpoint=end, margin_usd=0.25)
        guard.ledger = {"last_known_lifetime_usd": 7.78, "last_known_at": None,
                        "local_costs_since": []}
        cost = 0.0073
        calls = 0
        while guard.allows(cost):
            end.spend(cost)
            guard.record(cost)
            calls += 1
        self.assertLessEqual(end.true, 11.00 - 0.25 + 1e-9)
        self.assertGreater(end.true, 11.00 - 0.25 - 2 * cost)
        self.assertGreater(calls, 0)

    def test_endpoint_ahead_rebases_ledger(self):
        guard = LedgerGuard(11.00, self.path, endpoint=lambda: 9.0)
        guard.ledger = {"last_known_lifetime_usd": 8.0, "last_known_at": None,
                        "local_costs_since": [{"usd": 0.5, "label": "x", "at": None}]}
        self.assertEqual(guard.lifetime(), 9.0)
        self.assertEqual(guard.ledger["local_costs_since"], [])

    def test_endpoint_down_uses_ledger(self):
        guard = LedgerGuard(11.00, self.path, endpoint=lambda: None)
        guard.ledger = {"last_known_lifetime_usd": 10.70, "last_known_at": None,
                        "local_costs_since": [{"usd": 0.02, "label": "x", "at": None}]}
        self.assertAlmostEqual(guard.lifetime(), 10.72)
        self.assertFalse(guard.allows(0.04))  # 10.72 + 0.04 + 0.25 > 11.00


if __name__ == "__main__":
    unittest.main()
