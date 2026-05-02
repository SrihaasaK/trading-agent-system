import unittest
from datetime import datetime
from types import SimpleNamespace

from agents.trade_executor import _extract_exit_details


class TradeExecutorTests(unittest.TestCase):
    def test_extract_exit_details_uses_filled_exit_leg(self):
        filled_leg = SimpleNamespace(
            status="filled",
            filled_avg_price=110.0,
            filled_at=datetime(2026, 4, 12, 10, 30),
            updated_at=datetime(2026, 4, 12, 10, 30),
            limit_price=110.0,
            stop_price=None,
        )
        order = SimpleNamespace(legs=[filled_leg])
        row = {
            "direction": "LONG",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "shares": 10,
            "timestamp": "2026-04-12T09:30:00",
        }

        result = _extract_exit_details(order, row)
        self.assertEqual(result["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(result["outcome"], "WIN")
        self.assertAlmostEqual(result["pnl_r"], 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
