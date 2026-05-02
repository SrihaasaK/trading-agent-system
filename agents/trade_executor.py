"""
agents/trade_executor.py
Places bracket orders on Alpaca, syncs trade lifecycle updates back into SQLite,
and logs both executed and skipped signals.
"""

from __future__ import annotations

from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrderByIdRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from loguru import logger

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    IS_PAPER,
    NTFY_TOPIC,
    REQUIRE_TRADE_APPROVAL,
    TRAILING_STOP_BREAKEVEN_R,
    TRAILING_STOP_TRAIL_R,
)
from agents.journal import (
    get_open_trades,
    log_skip,
    log_trade,
    mark_trade_closed,
    update_order_state,
)
from agents.state import TradingState

_alpaca = None

def _get_alpaca():
    global _alpaca
    if _alpaca is None:
        _alpaca = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
    return _alpaca


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def send_ntfy(message: str, title: str = "Trading Agent", priority: str = "default", tags: str = "chart_increasing") -> None:
    """Send a push notification via ntfy.sh."""
    if not NTFY_TOPIC:
        return
    try:
        import requests
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", errors="replace").decode("ascii"),  # HTTP headers must be latin-1 safe
                "Priority": priority,
                "Tags": tags,
            },
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"[trade_executor] ntfy error: {e}")


def format_trade_alert(state: TradingState) -> str:
    """Format a human-readable trade alert."""
    direction = state.get("final_direction", "?")
    ticker = state.get("ticker", "?")
    price = state.get("price_data", {}).get("current_price", 0)
    stop = state.get("stop_level", 0)
    target = state.get("target_level", 0)
    shares = state.get("position_size", 0)
    score = state.get("signal_score", 0)
    regime = state.get("macro_regime", "?")
    risk_assess = state.get("risk_assessment", {})
    rr = risk_assess.get("reward_risk_ratio", 0)
    risk_dollars = risk_assess.get("actual_risk_dollars", 0)

    verdict = "\n".join(
        [f"  {v['agent']}: {v['score']:.2f} ({v['signal']})" for v in state.get("agent_verdicts", [])]
    )

    stop_pct = abs(price - stop) / price * 100 if price > 0 else 0
    target_pct = abs(target - price) / price * 100 if price > 0 else 0

    tier = state.get("confidence_tier", "STANDARD")

    return (
        f"{'LONG' if direction == 'LONG' else 'SHORT'} SIGNAL — {ticker} [{tier}]\n"
        f"Score: {score:.3f} | R:R = {rr:.1f}:1 | Tier: {tier}\n"
        f"\n"
        f"Entry: ${price:.2f}\n"
        f"Stop: ${stop:.2f} ({stop_pct:.1f}% away)\n"
        f"Target: ${target:.2f} ({target_pct:.1f}% away)\n"
        f"Size: {shares} shares (${shares * price:,.0f})\n"
        f"Risk: ${risk_dollars:,.0f}\n"
        f"\n"
        f"Regime: {regime}\n"
        f"News: {state.get('news_sentiment', '?')}\n"
        f"\n"
        f"Agent scores:\n{verdict}\n"
        f"\n"
        f"Reasoning: {state.get('decision_reasoning', '')[:120]}\n"
        f"Key risk: {state.get('decision_key_risk', '')[:80]}\n"
        f"\n"
        f"{'PAPER' if IS_PAPER else 'LIVE'} MODE"
    )


def _get_live_symbols() -> tuple[set[str], set[str]]:
    try:
        positions = {p.symbol for p in _get_alpaca().get_all_positions()}
    except Exception:
        positions = set()

    try:
        open_orders = _get_alpaca().get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100, nested=True))
        orders = {o.symbol for o in open_orders}
    except Exception:
        orders = set()

    return positions, orders


def has_live_exposure(ticker: str) -> bool:
    positions, orders = _get_live_symbols()
    return ticker in positions or ticker in orders


def place_bracket_order(state: TradingState) -> dict:
    """
    Place a bracket order (entry + stop loss + take profit) on Alpaca.
    """
    ticker = state["ticker"]
    direction = state["final_direction"]
    shares = state["position_size"]
    stop = round(state["stop_level"], 2)
    target = round(state["target_level"], 2)

    side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

    try:
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=target),
            stop_loss=StopLossRequest(stop_price=stop),
        )
        order = _get_alpaca().submit_order(order_request)
        return {
            "order_id": str(order.id),
            "status": _enum_value(order.status),
            "success": True,
        }
    except Exception as e:
        logger.error(f"[trade_executor] order error for {ticker}: {e}")
        return {"order_id": "", "status": "ERROR", "success": False, "error": str(e)}


def _extract_exit_details(order, row: dict) -> dict | None:
    """Infer realized close information from a parent bracket order and its filled exit leg."""
    legs = list(order.legs or [])
    filled_legs = [leg for leg in legs if _enum_value(leg.status).upper() == "FILLED" and leg.filled_avg_price]
    if not filled_legs:
        return None

    exit_leg = sorted(
        filled_legs,
        key=lambda leg: getattr(leg, "filled_at", None) or getattr(leg, "updated_at", None) or datetime.now(),
    )[-1]
    close_price = float(exit_leg.filled_avg_price or 0)
    if close_price <= 0:
        return None

    direction = row.get("direction", "LONG")
    entry_price = float(row.get("entry_price", 0) or 0)
    stop_loss = float(row.get("stop_loss", 0) or 0)
    shares = int(row.get("shares", 0) or 0)
    closed_at_obj = getattr(exit_leg, "filled_at", None) or getattr(exit_leg, "updated_at", None) or datetime.now()
    closed_at = closed_at_obj.isoformat() if hasattr(closed_at_obj, "isoformat") else datetime.now().isoformat()

    if direction == "LONG":
        pnl_dollars = (close_price - entry_price) * shares
    else:
        pnl_dollars = (entry_price - close_price) * shares

    risk_per_share = abs(entry_price - stop_loss)
    total_risk = risk_per_share * shares
    pnl_r = pnl_dollars / total_risk if total_risk > 0 else 0.0
    outcome = "WIN" if pnl_dollars > 0 else "LOSS" if pnl_dollars < 0 else "BREAKEVEN"

    if getattr(exit_leg, "limit_price", None):
        exit_reason = "TAKE_PROFIT"
    elif getattr(exit_leg, "stop_price", None):
        exit_reason = "STOP_LOSS"
    else:
        exit_reason = "MANUAL_EXIT"

    opened_at = row.get("timestamp", "")
    hold_minutes = 0.0
    if opened_at:
        try:
            dt_close = datetime.fromisoformat(closed_at)
            dt_open = datetime.fromisoformat(opened_at)
            # Normalize: strip tzinfo from both to avoid naive vs aware mismatch
            if dt_close.tzinfo is not None:
                dt_close = dt_close.replace(tzinfo=None)
            if dt_open.tzinfo is not None:
                dt_open = dt_open.replace(tzinfo=None)
            hold_minutes = round((dt_close - dt_open).total_seconds() / 60, 2)
        except (ValueError, TypeError):
            hold_minutes = 0.0

    return {
        "close_price": round(close_price, 4),
        "pnl_dollars": round(pnl_dollars, 2),
        "pnl_r": round(pnl_r, 3),
        "outcome": outcome,
        "exit_reason": exit_reason,
        "closed_at": closed_at,
        "hold_minutes": hold_minutes,
        "order_status": _enum_value(exit_leg.status).upper(),
    }


def sync_trade_journal() -> dict:
    """
    Reconcile open journal rows against Alpaca so realized exits flow into reports.
    """
    rows = get_open_trades(include_pending_approval=False)
    if not rows:
        return {"checked": 0, "closed": 0, "updated": 0}

    try:
        live_positions = {p.symbol for p in _get_alpaca().get_all_positions()}
    except Exception as e:
        logger.warning(f"[trade_executor] sync positions error: {e}")
        live_positions = set()

    updated = 0
    closed = 0
    for row in rows:
        order_id = row.get("order_id", "")
        ticker = row.get("ticker", "")
        if not order_id:
            continue

        try:
            import signal as _sig

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Alpaca order fetch timed out for {ticker}")

            old_handler = _sig.signal(_sig.SIGALRM, _timeout_handler)
            _sig.alarm(10)
            try:
                order = _get_alpaca().get_order_by_id(order_id, filter=GetOrderByIdRequest(nested=True))
            finally:
                _sig.alarm(0)
                _sig.signal(_sig.SIGALRM, old_handler)
        except Exception as e:
            logger.warning(f"[trade_executor] sync order fetch failed for {ticker}: {e}")
            continue

        status = _enum_value(order.status).upper()
        update_order_state(order_id, status)
        updated += 1

        # Check if THIS specific order has been filled and exited,
        # not just whether the ticker has any live position.
        # A ticker can have multiple journal entries from repeated trades.
        if status in ("FILLED", "PARTIALLY_FILLED") and ticker not in live_positions:
            exit_details = _extract_exit_details(order, row)
            if not exit_details:
                # Both bracket legs canceled (e.g., EOD force-close or manual liquidation).
                # Position is gone but no exit leg filled — mark closed with entry price.
                entry = float(row.get("entry_price", 0) or 0)
                exit_details = {
                    "close_price": entry,
                    "pnl_dollars": 0.0,
                    "pnl_r": 0.0,
                    "outcome": "BREAKEVEN",
                    "exit_reason": "MANUAL_EXIT",
                    "closed_at": datetime.now().isoformat(),
                    "hold_minutes": 0.0,
                    "order_status": status,
                }
                logger.info(f"[trade_executor] {ticker} closed externally (no exit leg) — marking as MANUAL_EXIT")
            if exit_details:
                mark_trade_closed(order_id=order_id, **exit_details)
                closed += 1
                pnl = exit_details.get("pnl_dollars", 0)
                pnl_r = exit_details.get("pnl_r", 0)
                outcome = exit_details.get("outcome", "?")
                exit_reason = exit_details.get("exit_reason", "?")
                hold_min = exit_details.get("hold_minutes", 0)
                emoji = "money_with_wings" if pnl >= 0 else "chart_with_downwards_trend"
                send_ntfy(
                    f"{ticker} CLOSED — {outcome}\n"
                    f"Exit: ${exit_details.get('close_price', 0):.2f} ({exit_reason})\n"
                    f"P&L: ${pnl:+,.2f} ({pnl_r:+.2f}R)\n"
                    f"Hold time: {hold_min:.0f} min\n"
                    f"Entry was: ${row.get('entry_price', 0):.2f}",
                    title=f"Trade Closed: {ticker} {outcome}",
                    priority="high" if abs(pnl) > 50 else "default",
                    tags=emoji,
                )
            continue

        if status in {"CANCELED", "REJECTED", "EXPIRED"}:
            update_order_state(order_id, status, close_open_trade=True)

    return {"checked": len(rows), "closed": closed, "updated": updated}


def check_trailing_stops() -> dict:
    """Check open positions and adjust stops based on R-multiple profit levels."""
    open_trades = get_open_trades(include_pending_approval=False)
    if not open_trades:
        return {"checked": 0, "adjusted": 0}

    try:
        positions = {p.symbol: float(p.current_price) for p in _get_alpaca().get_all_positions()}
    except Exception as e:
        logger.warning(f"[trade_executor] trailing stops: failed to fetch positions: {e}")
        return {"checked": 0, "adjusted": 0}

    adjusted = 0
    for trade in open_trades:
        ticker = trade.get("ticker", "")
        if ticker not in positions:
            continue

        current_price = positions[ticker]
        entry_price = float(trade.get("entry_price", 0) or 0)
        stop_loss = float(trade.get("stop_loss", 0) or 0)
        direction = trade.get("direction", "LONG")
        order_id = trade.get("order_id", "")

        if entry_price <= 0 or stop_loss <= 0 or not order_id:
            continue

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            continue

        if direction == "LONG":
            current_r = (current_price - entry_price) / risk_per_share
        else:
            current_r = (entry_price - current_price) / risk_per_share

        new_stop = None
        reason = ""

        if current_r >= TRAILING_STOP_TRAIL_R:
            # At 2R+: move stop to lock in 1R profit
            if direction == "LONG":
                candidate = entry_price + risk_per_share
                if candidate > stop_loss:
                    new_stop = candidate
                    reason = f"trail to 1R profit (at {current_r:.1f}R)"
            else:
                candidate = entry_price - risk_per_share
                if candidate < stop_loss:
                    new_stop = candidate
                    reason = f"trail to 1R profit (at {current_r:.1f}R)"
        elif current_r >= TRAILING_STOP_BREAKEVEN_R:
            # At 1R+: move stop to breakeven
            if direction == "LONG" and entry_price > stop_loss:
                new_stop = entry_price
                reason = f"breakeven (at {current_r:.1f}R)"
            elif direction == "SHORT" and entry_price < stop_loss:
                new_stop = entry_price
                reason = f"breakeven (at {current_r:.1f}R)"

        if new_stop is None:
            continue

        new_stop = round(new_stop, 2)

        # Try to replace the stop order via Alpaca
        try:
            order = _get_alpaca().get_order_by_id(order_id, filter=GetOrderByIdRequest(nested=True))
            stop_leg = None
            for leg in (order.legs or []):
                if getattr(leg, "stop_price", None) and _enum_value(leg.status).upper() not in ("FILLED", "CANCELED", "EXPIRED"):
                    stop_leg = leg
                    break

            if stop_leg is None:
                continue

            try:
                from alpaca.trading.requests import ReplaceOrderRequest
                _get_alpaca().replace_order_by_id(
                    str(stop_leg.id),
                    ReplaceOrderRequest(stop_price=new_stop),
                )
            except Exception:
                # Fallback: cancel and resubmit
                _get_alpaca().cancel_order_by_id(str(stop_leg.id))
                from alpaca.trading.requests import StopOrderRequest
                side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY
                _get_alpaca().submit_order(
                    MarketOrderRequest(
                        symbol=ticker,
                        qty=int(trade.get("shares", 0) or 0),
                        side=side,
                        time_in_force=TimeInForce.DAY,
                        type="stop",
                        stop_price=new_stop,
                    )
                )

            adjusted += 1
            logger.info(f"[trade_executor] trailing stop: {ticker} stop moved to ${new_stop:.2f} ({reason})")
            send_ntfy(
                f"{ticker} stop adjusted: ${stop_loss:.2f} -> ${new_stop:.2f}\n"
                f"Reason: {reason}\n"
                f"Current: ${current_price:.2f} | Entry: ${entry_price:.2f}",
                title=f"Stop Adjusted: {ticker}",
                priority="default",
                tags="shield",
            )
        except Exception as e:
            logger.warning(f"[trade_executor] trailing stop: failed to adjust {ticker}: {e}")

    return {"checked": len(open_trades), "adjusted": adjusted}


def eod_force_close() -> dict:
    """Close all day-traded positions at EOD to prevent overnight gap risk."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    open_trades = get_open_trades(include_pending_approval=False)
    today_trades = [t for t in open_trades if (t.get("timestamp", "") or "").startswith(today)]

    if not today_trades:
        logger.info("[trade_executor] EOD close: no same-day positions to close")
        return {"closed": 0, "errors": 0}

    try:
        live_positions = {p.symbol for p in _get_alpaca().get_all_positions()}
    except Exception as e:
        logger.error(f"[trade_executor] EOD close: failed to fetch positions: {e}")
        return {"closed": 0, "errors": 1}

    closed = 0
    errors = 0
    symbols_closed = []
    for trade in today_trades:
        symbol = trade.get("ticker", "")
        if symbol not in live_positions:
            continue
        try:
            _get_alpaca().close_position(symbol)
            symbols_closed.append(symbol)
            closed += 1
            logger.info(f"[trade_executor] EOD close: closed {symbol}")
        except Exception as e:
            logger.error(f"[trade_executor] EOD close: failed to close {symbol}: {e}")
            errors += 1

    if closed > 0:
        sync_trade_journal()
        send_ntfy(
            f"EOD force-close: {closed} position(s) closed\n"
            f"Symbols: {', '.join(symbols_closed)}",
            title="EOD Force-Close",
            priority="high",
            tags="clock3",
        )

    return {"closed": closed, "errors": errors}


def trade_executor_node(state: TradingState) -> TradingState:
    """LangGraph node: executes approved trades or logs skips."""
    ticker = state["ticker"]

    if not state.get("confidence_gate_passed"):
        state["approval_status"] = "SKIPPED"
        log_skip(state)
        logger.info(f"[trade_executor] {ticker} → SKIPPED: {state.get('skip_reason', 'unknown')}")
        return state

    if has_live_exposure(ticker):
        state["confidence_gate_passed"] = False
        state["skip_reason"] = f"Live exposure already exists for {ticker}"
        state["approval_status"] = "SKIPPED"
        log_skip(state)
        logger.warning(f"[trade_executor] {ticker} → duplicate exposure blocked")
        return state

    alert = format_trade_alert(state)
    logger.info(f"[trade_executor]\n{alert}")
    tier = state.get("confidence_tier", "STANDARD")
    send_ntfy(alert, title=f"Signal: {ticker} {state.get('final_direction', '')} [{tier}]", tags="eyes")

    if REQUIRE_TRADE_APPROVAL:
        logger.info("[trade_executor] APPROVAL MODE — signal logged, awaiting confirmation")
        state["order_id"] = "PENDING_APPROVAL"
        state["order_status"] = "PENDING_APPROVAL"
        state["approval_status"] = "PENDING_APPROVAL"
        log_trade(state, "PENDING_APPROVAL", "PENDING_APPROVAL", "PENDING_APPROVAL")
        return state

    result = place_bracket_order(state)

    if not result["success"]:
        state["confidence_gate_passed"] = False
        state["skip_reason"] = f"order failed: {result.get('error', 'unknown error')}"
        state["approval_status"] = "FAILED"
        send_ntfy(
            f"ORDER FAILED: {ticker} — {result.get('error', 'unknown error')}",
            title="Order Failed",
            priority="urgent",
            tags="warning",
        )
        log_skip(state)
        logger.error(f"[trade_executor] {ticker} → ORDER FAILED")
        return state

    state["order_id"] = result["order_id"]
    state["order_status"] = result["status"]
    state["approval_status"] = "AUTO_EXECUTED"
    log_trade(state, result["order_id"], result["status"], "AUTO_EXECUTED")

    send_ntfy(
        f"Order placed: {ticker} {state['final_direction']} x{state['position_size']} — ID: {result['order_id']}",
        title=f"ORDER: {ticker} {state['final_direction']}",
        priority="high",
        tags="white_check_mark",
    )
    logger.info(f"[trade_executor] {ticker} → ORDER PLACED id={result['order_id']}")

    return state
