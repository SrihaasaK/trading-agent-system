import unittest
from datetime import datetime

import pandas as pd

from agents.market_scanner import (
    allow_single_signal_setup,
    calculate_rvol,
    direction_matches_signal,
    normalize_direction,
    required_rvol,
    resolve_direction,
)


class MarketScannerDirectionTests(unittest.TestCase):
    def test_normalize_direction_maps_bullish_and_bearish_labels(self):
        self.assertEqual(normalize_direction("BULLISH"), "LONG")
        self.assertEqual(normalize_direction("BEARISH"), "SHORT")
        self.assertEqual(normalize_direction("LONG"), "LONG")
        self.assertEqual(normalize_direction("SHORT"), "SHORT")

    def test_direction_match_accepts_mixed_label_families(self):
        self.assertTrue(direction_matches_signal("LONG", "BULLISH"))
        self.assertTrue(direction_matches_signal("SHORT", "BEARISH"))
        self.assertFalse(direction_matches_signal("LONG", "BEARISH"))

    def test_resolve_direction_accepts_guarded_single_vote_for_bos(self):
        direction, resolution, counts = resolve_direction({"LONG": ["bos"], "SHORT": []})
        self.assertEqual(direction, "LONG")
        self.assertEqual(resolution, "single_vote_override:bos")
        self.assertEqual(counts, {"LONG": 1, "SHORT": 0})

    def test_resolve_direction_rejects_tied_vote(self):
        direction, resolution, counts = resolve_direction({"LONG": ["fvg"], "SHORT": ["vwap_deviation"]})
        self.assertEqual(direction, "NONE")
        self.assertEqual(resolution, "no_consensus")
        self.assertEqual(counts, {"LONG": 1, "SHORT": 1})


class MarketScannerRvolTests(unittest.TestCase):
    def test_calculate_rvol_blends_recent_and_same_time_history(self):
        full_index = pd.to_datetime(
            [
                datetime(2026, 4, 14, 9, 45),
                datetime(2026, 4, 15, 9, 30),
                datetime(2026, 4, 15, 9, 35),
                datetime(2026, 4, 15, 9, 40),
                datetime(2026, 4, 15, 9, 45),
            ]
        )
        full_volume = pd.Series([100, 80, 90, 100, 220], index=full_index)
        session_volume = full_volume.iloc[1:]

        self.assertAlmostEqual(calculate_rvol(session_volume, full_volume), 2.316, places=3)

    def test_required_rvol_relaxes_opening_and_midday_thresholds(self):
        self.assertEqual(required_rvol("MORNING", 4), 0.95)
        self.assertEqual(required_rvol("MIDDAY", 20), 0.95)
        self.assertEqual(required_rvol("AFTERNOON", 20), 1.0)


class MarketScannerGuardedSingleSignalTests(unittest.TestCase):
    def test_single_signal_override_requires_clean_confirmation(self):
        self.assertTrue(
            allow_single_signal_setup(
                "single_vote_override:fvg",
                smc_count=1,
                momentum_count=2,
                ema_check={"with_trend": True},
                adx_check={"strong": True},
                rvol=1.1,
                rvol_floor=0.95,
            )
        )

    def test_single_signal_override_stays_blocked_without_confirmation(self):
        self.assertFalse(
            allow_single_signal_setup(
                "single_vote_override:fvg",
                smc_count=1,
                momentum_count=1,
                ema_check={"with_trend": True},
                adx_check={"strong": False},
                rvol=1.0,
                rvol_floor=0.95,
            )
        )


if __name__ == "__main__":
    unittest.main()
