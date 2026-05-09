"""
agents/risk_analyst.py
Deterministic risk engine for position sizing, duplicate exposure checks,
and portfolio heat control.
"""

from __future__ import annotations

from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from loguru import logger

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CONFIDENCE_TIERS,
    DAILY_LOSS_LIMIT_PCT,
    IS_PAPER,
    MAX_OPEN_POSITIONS,
    MAX_PORTFOLIO_HEAT_PCT,
    MAX_POSITION_NOTIONAL_PCT,
    PORTFOLIO_CACHE_SECONDS,
    RISK_PER_TRADE_PCT,
    TRADE_COOLDOWN_MINUTES,
)
from agents.journal import estimate_open_risk_from_journal, get_daily_realized_pnl, get_daily_trade_count, get_open_trades, get_recent_signals
from agents.state import TradingState

_alpaca_client = None

def _get_alpaca():
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
    return _alpaca_client

_portfolio_cache: dict = {"timestamp": None, "data": None}

MAX_DAILY_TRADES_PER_TICKER = 3
MAX_DAILY_TRADES_TOTAL = 12

CORRELATED_GROUPS = {
    "mega_cap_tech": {"NVDA", "META", "AMZN", "AMD", "NFLX", "TSLA", "NET"},
    "semiconductors": {"NVDA", "AMD", "MU"},
    "crypto_fintech": {"COIN", "SOFI"},
    "energy": {"CVX"},
    "index_beta": {"SPY", "QQQ", "IWM", "VXX"},
}


def _resolve_tier(scanner_confidence: float) -> tuple[str, dict]:
    """Map scanner confidence to a provisional confidence tier.

    Uses conservative thresholds (scanner conf skews ~0.05-0.10 higher than
    the final weighted score) so we don't over-commit on tier before signal_judge
    runs its authoritative re-check.
    """
    if scanner_confidence >= 0.85:
        return "SWING", CONFIDENCE_TIERS["SWING"]
    if scanner_confidence >= 0.68:
        return "STANDARD", CONFIDENCE_TIERS["STANDARD"]
    return "SCALP", CONFIDENCE_TIERS["SCALP"]


def _side_value(side) -> str:
    return side.value if hasattr(side, "value") else str(side)


def get_open_orders() -> list:
    """Fetch currently open orders from Alpaca."""
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100, nested=True)
        return _get_alpaca().get_orders(filter=request)
    except Exception as e:
        logger.warning(f"[risk_analyst] open orders fetch error: {e}")
        return []


def get_portfolio_state(force_refresh: bool = False) -> dict:
    """Pull current portfolio value, positions, and open orders from Alpaca with a short cache."""
    now = datetime.now()
    cached_at = _portfolio_cache.get("timestamp")
    if (
        not force_refresh
        and cached_at
        and (now - cached_at).total_seconds() < PORTFOLIO_CACHE_SECONDS
        and _portfolio_cache.get("data") is not None
    ):
        return _portfolio_cache["data"]

    try:
        account = _get_alpaca().get_account()
        positions = _get_alpaca().get_all_positions()
        open_orders = get_open_orders()

        portfolio_value = float(account.portfolio_value)
        cash = float(account.cash)
        buying_power = float(getattr(account, "buying_power", cash))
        open_position_rows = [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "side": _side_value(p.side),
            }
            for p in positions
        ]
        open_order_rows = [
            {
                "symbol": o.symbol,
                "qty": float(o.qty or 0),
                "side": _side_value(o.side),
                "status": _side_value(o.status),
                "order_id": str(o.id),
            }
            for o in open_orders
        ]

        estimated_open_risk = estimate_open_risk_from_journal()
        heat_pct = estimated_open_risk / portfolio_value if portfolio_value > 0 else 0.0

        data = {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "buying_power": buying_power,
            "open_positions": open_position_rows,
            "open_orders": open_order_rows,
            "position_count": len(open_position_rows),
            "open_symbols": [row["symbol"] for row in open_position_rows],
            "open_order_symbols": [row["symbol"] for row in open_order_rows],
            "estimated_open_risk": round(estimated_open_risk, 2),
            "heat_pct": round(heat_pct, 4),
            "available_slots": MAX_OPEN_POSITIONS - len(open_position_rows),
        }
        _portfolio_cache["timestamp"] = now
        _portfolio_cache["data"] = data
        return data
    except Exception as e:
        logger.error(f"[risk_analyst] portfolio fetch error: {e}")
        fallback = {
            "portfolio_value": 100000.0,
            "cash": 100000.0,
            "buying_power": 100000.0,
            "open_positions": [],
            "open_orders": [],
            "position_count": 0,
            "open_symbols": [],
            "open_order_symbols": [],
            "estimated_open_risk": 0.0,
            "heat_pct": 0.0,
            "available_slots": 0,
        }
        _portfolio_cache["timestamp"] = now
        _portfolio_cache["data"] = fallback
        return fallback


def calculate_position_size(
    portfolio_value: float,
    cash: float,
    buying_power: float,
    entry_price: float,
    stop_price: float,
    direction: str,
    size_multiplier: float = 1.0,
) -> dict:
    """
    Position sizing based on fixed-fractional risk with additional capital/notional caps.
    """
    if entry_price <= 0 or stop_price <= 0:
        return {"shares": 0, "risk_dollars": 0, "actual_risk_dollars": 0, "position_value": 0, "constraints": []}

    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return {"shares": 0, "risk_dollars": 0, "actual_risk_dollars": 0, "position_value": 0, "constraints": []}

    risk_budget = portfolio_value * RISK_PER_TRADE_PCT * size_multiplier
    max_notional = portfolio_value * MAX_POSITION_NOTIONAL_PCT
    capital_limit = buying_power if direction == "SHORT" else min(cash, buying_power)

    shares_by_risk = int(risk_budget / stop_distance)
    shares_by_capital = int(capital_limit / entry_price) if entry_price > 0 else 0
    shares_by_notional = int(max_notional / entry_price) if entry_price > 0 else 0

    constraints = []
    candidates = {
        "risk_budget": shares_by_risk,
        "capital": shares_by_capital,
        "max_notional": shares_by_notional,
    }
    shares = max(0, min(candidates.values()))
    for name, candidate in candidates.items():
        if shares == candidate:
            constraints.append(name)

    actual_risk_dollars = shares * stop_distance
    position_value = shares * entry_price

    return {
        "shares": shares,
        "risk_dollars": round(risk_budget, 2),
        "actual_risk_dollars": round(actual_risk_dollars, 2),
        "position_value": round(position_value, 2),
        "stop_distance": round(stop_distance, 4),
        "constraints": constraints,
    }


def _correlation_pressure(ticker: str, open_symbols: list[str]) -> int:
    highest_overlap = 0
    for symbols in CORRELATED_GROUPS.values():
        if ticker in symbols:
            overlap = len(symbols.intersection(set(open_symbols)))
            highest_overlap = max(highest_overlap, overlap)
    return highest_overlap


def assess_risk(state: TradingState, portfolio: dict) -> dict:
    """Build deterministic risk constraints and return an approval decision."""
    ticker = state["ticker"]
    setup = state.get("technical_setup", {})
    price = float(state["price_data"].get("current_price", 0) or 0)
    atr = float(state["price_data"].get("atr", 0) or 0)
    direction = setup.get("direction", "LONG")
    macro = state.get("macro_context", {})
    size_multiplier = float(macro.get("size_multiplier", 1.0) or 1.0)

    reasons = []
    warnings = []
    approved = True
    grade = "GREEN"

    if price <= 0 or atr <= 0:
        return {
            "grade": "RED",
            "approved": False,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "shares": 0,
            "risk_dollars": 0.0,
            "actual_risk_dollars": 0.0,
            "position_value": 0.0,
            "reward_risk_ratio": 0.0,
            "predicted_heat_pct": portfolio.get("heat_pct", 0.0),
            "reasons": ["Missing price or ATR for risk sizing"],
            "warnings": [],
            "constraints": [],
            "direction": direction,
        }

    tier_name, tier = _resolve_tier(float(setup.get("confidence", 0.65) or 0.65))
    atr_mult = tier["atr_stop_mult"]
    rr_target = tier["reward_risk"]

    if direction == "LONG":
        stop = price - (atr * atr_mult)
        target = price + (atr * atr_mult * rr_target)
    else:
        stop = price + (atr * atr_mult)
        target = price - (atr * atr_mult * rr_target)

    sizing = calculate_position_size(
        portfolio["portfolio_value"],
        portfolio["cash"],
        portfolio["buying_power"],
        price,
        stop,
        direction,
        size_multiplier * tier["size_factor"],
    )
    rr_ratio = round(abs(target - price) / abs(stop - price), 2) if abs(stop - price) > 0 else 0.0
    predicted_heat_pct = (
        (portfolio.get("estimated_open_risk", 0.0) + sizing["actual_risk_dollars"]) / portfolio["portfolio_value"]
        if portfolio["portfolio_value"] > 0
        else 0.0
    )

    if portfolio["available_slots"] <= 0:
        reasons.append(f"Max positions ({MAX_OPEN_POSITIONS}) already open")
    if portfolio["heat_pct"] >= MAX_PORTFOLIO_HEAT_PCT:
        reasons.append(f"Portfolio heat {portfolio['heat_pct'] * 100:.1f}% already at max")
    if predicted_heat_pct > MAX_PORTFOLIO_HEAT_PCT:
        reasons.append(
            f"Trade would push portfolio heat to {predicted_heat_pct * 100:.1f}%"
        )
    daily_pnl = get_daily_realized_pnl()
    daily_loss_limit = portfolio["portfolio_value"] * DAILY_LOSS_LIMIT_PCT
    if daily_pnl < 0 and abs(daily_pnl) >= daily_loss_limit:
        reasons.append(
            f"Daily loss circuit breaker: ${daily_pnl:+,.2f} exceeds "
            f"-${daily_loss_limit:,.2f} ({DAILY_LOSS_LIMIT_PCT*100:.0f}% limit)"
        )

    if ticker in portfolio["open_symbols"]:
        reasons.append(f"Already holding {ticker}")
    if ticker in portfolio["open_order_symbols"]:
        reasons.append(f"Open order already exists for {ticker}")

    recent_signals = [
        row for row in get_recent_signals(ticker, TRADE_COOLDOWN_MINUTES)
        if row.get("direction") != "SKIP"
    ]
    if recent_signals:
        reasons.append(f"{ticker} already triggered within the last {TRADE_COOLDOWN_MINUTES} minutes")

    daily_ticker_count = get_daily_trade_count(ticker)
    if daily_ticker_count >= MAX_DAILY_TRADES_PER_TICKER:
        reasons.append(f"Daily max trades for {ticker} reached ({daily_ticker_count}/{MAX_DAILY_TRADES_PER_TICKER})")

    daily_total_count = get_daily_trade_count()
    if daily_total_count >= MAX_DAILY_TRADES_TOTAL:
        reasons.append(f"Daily trade limit reached ({daily_total_count}/{MAX_DAILY_TRADES_TOTAL})")

    if sizing["shares"] <= 0:
        reasons.append("Position size rounded to zero under current caps")
    if rr_ratio < rr_target:
        reasons.append(f"Reward:risk {rr_ratio:.2f} below {tier_name} minimum {rr_target:.2f}")

    correlation_overlap = _correlation_pressure(ticker, portfolio["open_symbols"])
    if correlation_overlap >= 4:
        reasons.append(f"Correlation risk too high: {correlation_overlap} similar names already open")
    elif correlation_overlap == 3:
        warnings.append("Elevated correlation risk with existing positions")
        grade = "YELLOW"

    # Directional balance check
    open_trades = get_open_trades(include_pending_approval=False)
    if len(open_trades) >= 3:
        long_count = sum(1 for t in open_trades if t.get("direction") == "LONG")
        short_count = sum(1 for t in open_trades if t.get("direction") == "SHORT")
        total_open = long_count + short_count
        if total_open > 0:
            dominant_pct = max(long_count, short_count) / total_open
            if dominant_pct > 0.8 and direction == ("LONG" if long_count > short_count else "SHORT"):
                warnings.append(f"Directional concentration: {max(long_count, short_count)}/{total_open} positions are {direction}")
                grade = "YELLOW"

    if "capital" in sizing["constraints"] or "max_notional" in sizing["constraints"]:
        warnings.append("Position size capped by capital or single-name notional limits")
        grade = "YELLOW"
    if predicted_heat_pct >= MAX_PORTFOLIO_HEAT_PCT * 0.8 and not reasons:
        warnings.append("Portfolio heat will be near the maximum after this trade")
        grade = "YELLOW"
    if macro.get("regime") == "HIGH_VOL" and not reasons:
        warnings.append("High-volatility regime suggests smaller size or faster profit-taking")
        grade = "YELLOW"

    if reasons:
        approved = False
        grade = "RED"

    return {
        "grade": grade,
        "approved": approved,
        "stop_loss": round(stop, 4),
        "take_profit": round(target, 4),
        "shares": sizing["shares"],
        "risk_dollars": sizing["risk_dollars"],
        "actual_risk_dollars": sizing["actual_risk_dollars"],
        "position_value": sizing["position_value"],
        "reward_risk_ratio": rr_ratio,
        "predicted_heat_pct": round(predicted_heat_pct, 4),
        "reasons": reasons,
        "warnings": warnings,
        "constraints": sizing["constraints"],
        "direction": direction,
        "confidence_tier": tier_name,
    }


def risk_analyst_node(state: TradingState) -> TradingState:
    """LangGraph node: calculates risk parameters and updates state."""
    ticker = state["ticker"]
    logger.info(f"[risk_analyst] assessing {ticker}")

    if state.get("news_veto") or state.get("macro_veto") or not state.get("session_valid"):
        return state

    portfolio = get_portfolio_state()
    assessment = assess_risk(state, portfolio)

    state["risk_assessment"] = {**assessment, "portfolio": portfolio}
    state["risk_grade"] = assessment.get("grade", "GREEN")
    state["position_size"] = int(assessment.get("shares", 0) or 0)
    state["stop_level"] = float(assessment.get("stop_loss", 0.0) or 0.0)
    state["target_level"] = float(assessment.get("take_profit", 0.0) or 0.0)
    state["confidence_tier"] = assessment.get("confidence_tier", "STANDARD")

    if not assessment.get("approved", True):
        state["skip_reason"] = " | ".join(assessment.get("reasons", ["risk rejected"]))

    logger.info(
        f"[risk_analyst] {ticker} → grade={state['risk_grade']} tier={state['confidence_tier']} "
        f"size={state['position_size']} stop={state['stop_level']:.2f} "
        f"target={state['target_level']:.2f} heat={assessment.get('predicted_heat_pct', 0) * 100:.1f}%"
    )
    return state
