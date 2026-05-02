import unittest
from unittest.mock import patch

from agents.risk_analyst import assess_risk, calculate_position_size
from agents.state import initial_state


class RiskAnalystTests(unittest.TestCase):
    def test_short_trade_uses_short_stop_and_target(self):
        state = initial_state("TSLA")
        state["technical_setup"] = {"direction": "SHORT"}
        state["price_data"] = {"current_price": 100.0, "atr": 2.0}
        state["macro_context"] = {"size_multiplier": 1.0, "regime": "TRENDING_BEAR"}
        portfolio = {
            "portfolio_value": 100000.0,
            "cash": 100000.0,
            "buying_power": 100000.0,
            "open_positions": [],
            "open_orders": [],
            "position_count": 0,
            "open_symbols": set(),
            "open_order_symbols": set(),
            "estimated_open_risk": 0.0,
            "heat_pct": 0.0,
            "available_slots": 5,
        }

        with patch("agents.risk_analyst.get_recent_signals", return_value=[]):
            result = assess_risk(state, portfolio)

        self.assertGreater(result["stop_loss"], state["price_data"]["current_price"])
        self.assertLess(result["take_profit"], state["price_data"]["current_price"])
        self.assertTrue(result["approved"])

    def test_position_size_is_capped_by_notional_and_capital(self):
        sizing = calculate_position_size(
            portfolio_value=100000.0,
            cash=5000.0,
            buying_power=5000.0,
            entry_price=250.0,
            stop_price=245.0,
            direction="LONG",
            size_multiplier=1.0,
        )
        self.assertLessEqual(sizing["shares"], 20)
        self.assertIn("capital", sizing["constraints"])


if __name__ == "__main__":
    unittest.main()
