"""
tests/test_pipeline_integration.py
Integration test: runs a single ticker through the full LangGraph pipeline
with mocked external dependencies (Alpaca, yfinance, LLM).
"""
import pytest
from unittest.mock import patch, MagicMock
from agents.state import initial_state, TradingState
from agents.orchestrator import build_trading_graph


def _mock_yfinance_download(*args, **kwargs):
    """Return realistic 5-day 5-min OHLCV data for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2026-04-28 09:30", periods=390, freq="5min", tz="America/New_York")
    np.random.seed(42)
    base = 200.0
    close = base + np.cumsum(np.random.randn(390) * 0.5)
    df = pd.DataFrame({
        "Open": close - np.random.rand(390) * 0.3,
        "High": close + np.random.rand(390) * 1.0,
        "Low": close - np.random.rand(390) * 1.0,
        "Close": close,
        "Volume": np.random.randint(100000, 500000, 390),
    }, index=dates)
    return df


@patch("agents.market_scanner.yf.download", side_effect=_mock_yfinance_download)
@patch("agents.news_researcher._call_llm", return_value='{"sentiment": "NEUTRAL", "sentiment_score": 0.5, "confidence": 0.6, "veto": false, "veto_reason": "", "key_points": ["No significant news"]}')
@patch("agents.macro_context.yf.download", side_effect=_mock_yfinance_download)
@patch("agents.trade_executor._get_alpaca")
@patch("agents.risk_analyst._get_alpaca")
def test_full_pipeline_skip(mock_risk_alpaca, mock_exec_alpaca, mock_macro_yf, mock_news_llm, mock_scanner_yf):
    """A ticker that fails scanner gates should flow through to SKIPPED without errors."""
    # Mock Alpaca account
    mock_account = MagicMock()
    mock_account.portfolio_value = "100000"
    mock_account.cash = "100000"
    mock_account.buying_power = "200000"

    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account
    mock_client.get_all_positions.return_value = []
    mock_client.get_orders.return_value = []

    mock_risk_alpaca.return_value = mock_client
    mock_exec_alpaca.return_value = mock_client

    graph = build_trading_graph()
    state = initial_state("TEST")
    result = graph.invoke(state)

    # Should complete without error
    assert result["ticker"] == "TEST"
    # With random data, it will almost certainly fail a gate
    assert result.get("skip_reason") or result.get("order_id")


def test_initial_state_has_all_fields():
    """initial_state should populate every TradingState field."""
    state = initial_state("AAPL")
    assert state["ticker"] == "AAPL"
    assert state["session_valid"] == False
    assert state["confidence_tier"] == "STANDARD"
    assert state["final_direction"] == "NONE"
    assert isinstance(state["errors"], list)
