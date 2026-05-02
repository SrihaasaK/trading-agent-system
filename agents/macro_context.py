"""
agents/macro_context.py
Session-level macro context agent. Runs once per session (not per ticker).
Monitors VIX, broad market trend, and upcoming macro events.
Outputs a regime label and optional veto for extreme conditions.
"""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf
from datetime import datetime
from loguru import logger

from config.settings import (
    BASE_DIR, MACRO_TICKERS,
    VIX_NORMAL_MAX, VIX_CAUTION_MAX, VIX_HALT_THRESHOLD,
)
from agents.state import TradingState

# Cached so it only runs once per session
_session_cache: dict = {}
_cache_timestamp: datetime = None
CACHE_MINUTES = 30


def get_vix() -> float:
    """Pull current VIX level from yfinance."""
    try:
        vix = yf.download("^VIX", period="2d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        return float(vix["Close"].dropna().iloc[-1])
    except Exception as e:
        logger.warning(f"[macro_context] VIX fetch error: {e}")
        return 20.0  # default to neutral


def get_market_breadth() -> dict:
    """Check broad market trend relative to each symbol's 20-day MA."""
    try:
        tickers = yf.download(MACRO_TICKERS, period="30d", interval="1d", progress=False, auto_adjust=True)
        breadth = {}
        for sym in MACRO_TICKERS:
            try:
                if isinstance(tickers.columns, pd.MultiIndex):
                    close = tickers["Close"][sym].dropna()
                else:
                    close = tickers["Close"].dropna()
                ma20  = close.rolling(20).mean().iloc[-1]
                price = close.iloc[-1]
                daily_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
                breadth[sym] = {
                    "price":         round(float(price), 2),
                    "ma20":          round(float(ma20), 2),
                    "above_ma20":    bool(price > ma20),
                    "daily_chg_pct": round(float(daily_change), 2),
                }
            except Exception:
                breadth[sym] = {"price": 0, "ma20": 0, "above_ma20": True, "daily_chg_pct": 0}
        return breadth
    except Exception as e:
        logger.warning(f"[macro_context] breadth error: {e}")
        return {}


def get_economic_events() -> list:
    """
    Fetch upcoming high-impact economic events from a local cache file.
    If data/economic_events.json exists, it should contain:
    [{"name": "...", "timestamp": "ISO8601", "impact": "HIGH"}]
    """
    try:
        calendar_path = Path(BASE_DIR) / "data" / "economic_events.json"
        if not calendar_path.exists():
            return []

        payload = json.loads(calendar_path.read_text())
        now = datetime.now()
        upcoming = []
        for item in payload:
            timestamp = item.get("timestamp", "")
            if not timestamp:
                continue
            event_time = datetime.fromisoformat(timestamp)
            delta_hours = (event_time - now).total_seconds() / 3600
            if 0 <= delta_hours <= 24 and item.get("impact", "HIGH").upper() == "HIGH":
                upcoming.append(f"{item.get('name', 'macro event')} @ {event_time.strftime('%Y-%m-%d %H:%M')}")
        return upcoming
    except Exception as e:
        logger.warning(f"[macro_context] events error: {e}")
        return []


def classify_regime(vix: float, breadth: dict) -> str:
    """
    Classify current market regime based on VIX and broad market breadth.
    Used by signal judge to adjust confidence thresholds.
    """
    spy = breadth.get("SPY", {})
    qqq = breadth.get("QQQ", {})

    spy_bullish = spy.get("above_ma20", True)
    qqq_bullish = qqq.get("above_ma20", True)
    spy_change  = spy.get("daily_chg_pct", 0)

    if vix > VIX_CAUTION_MAX:
        return "HIGH_VOL"

    if spy_bullish and qqq_bullish and spy_change > -0.5:
        return "TRENDING_BULL"

    if not spy_bullish and not qqq_bullish and spy_change < 0.5:
        return "TRENDING_BEAR"

    return "RANGING"


def get_session_macro() -> dict:
    """
    Full macro context for the current session.
    Cached for CACHE_MINUTES to avoid redundant API calls.
    """
    global _session_cache, _cache_timestamp

    now = datetime.now()
    if _cache_timestamp and (now - _cache_timestamp).total_seconds() < CACHE_MINUTES * 60:
        return _session_cache

    vix     = get_vix()
    breadth = get_market_breadth()
    events  = get_economic_events()
    regime  = classify_regime(vix, breadth)

    # Determine macro veto conditions
    veto        = False
    veto_reason = ""

    if vix >= VIX_HALT_THRESHOLD:
        veto        = True
        veto_reason = f"VIX at {vix:.1f} — extreme volatility, no new positions"
    elif events:
        veto        = True
        veto_reason = f"High-impact macro event within 2h: {events[0]}"

    # Position size multiplier based on regime
    size_multiplier = 1.0
    if vix > VIX_NORMAL_MAX:
        size_multiplier = 0.5
    if regime == "HIGH_VOL":
        size_multiplier = 0.25

    result = {
        "vix":              round(vix, 2),
        "regime":           regime,
        "breadth":          breadth,
        "breadth_score":    round(
            sum(1 for row in breadth.values() if row.get("above_ma20")) / max(len(breadth), 1),
            3,
        ),
        "upcoming_events":  events,
        "veto":             veto,
        "veto_reason":      veto_reason,
        "size_multiplier":  size_multiplier,
        "timestamp":        now.isoformat(),
    }

    _session_cache    = result
    _cache_timestamp  = now
    return result


# ── LangGraph Node ────────────────────────────────────────────────────────────

def macro_context_node(state: TradingState) -> TradingState:
    """LangGraph node: injects macro context into the trading state."""
    logger.info(f"[macro_context] fetching session macro")

    if state.get("news_veto"):
        return state  # Already vetoed — skip

    macro = get_session_macro()

    state["macro_context"] = macro
    state["macro_regime"]  = macro["regime"]
    state["macro_veto"]    = macro["veto"]

    if macro["veto"]:
        state["skip_reason"] = macro["veto_reason"]

    logger.info(f"[macro_context] regime={macro['regime']} vix={macro['vix']} veto={macro['veto']}")
    return state
