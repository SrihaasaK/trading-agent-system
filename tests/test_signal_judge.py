import unittest

from agents.signal_judge import (
    calculate_weighted_score,
    score_macro,
    score_news,
    score_risk,
    score_technical,
    signal_judge_node,
)
from agents.state import initial_state
from config.settings import MIN_CONFIDENCE_SCORE


class SignalJudgeTests(unittest.TestCase):
    def _base_short_state(self):
        state = initial_state("NVDA")
        state["technical_setup"] = {
            "valid": True,
            "direction": "SHORT",
            "confidence": 0.95,
            "adx": {"adx": 35.0},
            "rsi": {"rsi": 62.0},
            "rvol": 2.1,
            "smc_count": 3,
            "momentum_count": 2,
            "htf_levels": {"in_premium": True, "in_discount": False, "in_ote": True},
        }
        state["news_sentiment"] = "BEARISH"
        state["news_summary"] = "Weak demand and negative catalyst flow"
        state["macro_regime"] = "TRENDING_BEAR"
        state["macro_context"] = {"vix": 18.0, "breadth_score": 0.20}
        state["risk_grade"] = "GREEN"
        state["risk_assessment"] = {"reward_risk_ratio": 2.8, "warnings": []}
        state["position_size"] = 100
        state["session_valid"] = True
        return state

    def test_strong_short_setup_can_clear_confidence_gate(self):
        state = self._base_short_state()
        verdicts = [
            score_technical(state),
            score_news(state),
            score_macro(state),
            score_risk(state),
        ]
        weighted = calculate_weighted_score(verdicts)
        self.assertGreaterEqual(weighted, MIN_CONFIDENCE_SCORE)

    def test_signal_judge_keeps_short_direction(self):
        state = self._base_short_state()
        state = signal_judge_node(state)
        self.assertTrue(state["confidence_gate_passed"])
        self.assertEqual(state["final_direction"], "SHORT")


if __name__ == "__main__":
    unittest.main()
