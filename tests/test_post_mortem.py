import unittest

from agents.post_mortem import cluster_failure_patterns, propose_updates


class PostMortemTests(unittest.TestCase):
    def test_failure_clusters_include_session_and_score_bucket(self):
        trades = [
            {
                "outcome": "LOSS",
                "macro_regime": "RANGING",
                "session_label": "MIDDAY",
                "signal_score": 0.70,
                "ticker": "AAPL",
                "pnl_r": -1.0,
            },
            {
                "outcome": "LOSS",
                "macro_regime": "RANGING",
                "session_label": "MIDDAY",
                "signal_score": 0.71,
                "ticker": "MSFT",
                "pnl_r": -0.9,
            },
            {
                "outcome": "LOSS",
                "macro_regime": "RANGING",
                "session_label": "MIDDAY",
                "signal_score": 0.69,
                "ticker": "NVDA",
                "pnl_r": -1.1,
            },
        ]
        clusters = cluster_failure_patterns(trades)
        self.assertEqual(len(clusters), 1)
        self.assertIn("MIDDAY", clusters[0]["pattern"])
        self.assertIn("0.68-0.72", clusters[0]["pattern"])

    def test_proposals_require_large_sample_size(self):
        closed = [{"outcome": "WIN"}] * 30
        sessions = [{"bucket": "MIDDAY", "count": 12, "win_rate": 0.30, "avg_r": -0.2}]
        score_buckets = [
            {"bucket": "0.68-0.72", "count": 12, "win_rate": 0.35, "avg_r": -0.1},
            {"bucket": "0.78+", "count": 12, "win_rate": 0.60, "avg_r": 0.5},
        ]
        proposals = propose_updates(closed, sessions, score_buckets)
        self.assertGreaterEqual(len(proposals), 1)


if __name__ == "__main__":
    unittest.main()
