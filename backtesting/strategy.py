"""
backtesting/strategy.py
Backtesting harness that mirrors the live technical scanner more closely.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_scanner import (
    RVOL_MIN,
    calculate_adx,
    calculate_all_emas,
    calculate_atr,
    calculate_macd,
    calculate_rsi,
    calculate_rvol,
    calculate_vwap,
    check_adx,
    check_ema_stack,
    check_macd,
    check_rsi,
    check_vwap_position,
    detect_bos_choch,
    detect_fvg,
    detect_order_block,
    detect_vwap_deviation,
    direction_matches_signal,
    normalize_direction,
)
from config.settings import CONFIDENCE_TIERS, MIN_CONFIDENCE_SCORE

# Use STANDARD tier parameters as the backtesting baseline
ATR_STOP_MULTIPLIER = CONFIDENCE_TIERS["STANDARD"]["atr_stop_mult"]
MIN_REWARD_RISK_RATIO = CONFIDENCE_TIERS["STANDARD"]["reward_risk"]


def session_label(timestamp: pd.Timestamp) -> str:
    hour = timestamp.hour
    if 9 <= hour < 11:
        return "MORNING"
    if 11 <= hour < 14:
        return "MIDDAY"
    if 14 <= hour < 16:
        return "AFTERNOON"
    return "CLOSED"


def session_multiplier(label: str) -> float:
    return {"MORNING": 1.0, "AFTERNOON": 0.85, "MIDDAY": 0.65, "CLOSED": 0.0}.get(label, 0.5)


def live_style_signal(df: pd.DataFrame) -> dict:
    """Replicate the live scanner logic on a historical intraday session slice."""
    if len(df) < 60:
        return {"valid": False, "reason": "insufficient bars"}

    session_df = df[df.index.normalize() == df.index[-1].normalize()].copy()
    if len(session_df) < 20:
        return {"valid": False, "reason": "insufficient bars in session"}

    close = session_df["Close"]
    high = session_df["High"]
    low = session_df["Low"]
    volume = session_df["Volume"]
    price = float(close.iloc[-1])

    emas = calculate_all_emas(close)
    adx, pdi, mdi = calculate_adx(high, low, close)
    rsi = calculate_rsi(close)
    _, _, macd_h = calculate_macd(close)
    vwap = calculate_vwap(session_df)
    atr = calculate_atr(session_df)
    rvol = calculate_rvol(volume)
    fvg = detect_fvg(session_df)
    bos = detect_bos_choch(session_df)
    ob = detect_order_block(session_df)
    vwap_dev = detect_vwap_deviation(session_df, vwap)

    direction_votes = {"LONG": 0, "SHORT": 0}
    for label in [
        fvg.get("type"),
        bos.get("direction"),
        ob.get("type"),
        vwap_dev.get("direction"),
    ]:
        mapped = normalize_direction(label or "")
        if mapped in direction_votes:
            direction_votes[mapped] += 1

    if direction_votes["LONG"] >= 2:
        direction = "LONG"
    elif direction_votes["SHORT"] >= 2:
        direction = "SHORT"
    else:
        return {"valid": False, "reason": "no directional consensus"}

    ema_check = check_ema_stack(emas, direction, price)
    adx_check = check_adx(float(adx.iloc[-1]), float(pdi.iloc[-1]), float(mdi.iloc[-1]), direction)
    rsi_check = check_rsi(float(rsi.iloc[-1]), direction)
    macd_check = check_macd(macd_h, direction)
    vwap_position = check_vwap_position(price, float(vwap.iloc[-1]), direction)
    session = session_label(session_df.index[-1])

    hard_gates = {
        "session": session != "CLOSED",
        "ema_aligned": ema_check["passes"],
        "vwap_position": vwap_position,
        "adx_strength": adx_check["passes"],
        "rvol": rvol >= RVOL_MIN,
    }
    if not all(hard_gates.values()):
        return {"valid": False, "reason": "hard gates failed"}

    smc_signals = {
        "fvg": fvg["detected"] and direction_matches_signal(direction, fvg.get("type", "")) if fvg["detected"] else False,
        "bos": bos["detected"] and direction_matches_signal(direction, bos.get("direction", "")),
        "order_block": ob["detected"] and direction_matches_signal(direction, ob.get("type", "")) and ob.get("in_zone", False) if ob["detected"] else False,
    }
    if sum(smc_signals.values()) < 2:
        return {"valid": False, "reason": "smc confluence failed"}

    momentum_signals = {"rsi": rsi_check["passes"], "macd": macd_check["passes"]}
    if sum(momentum_signals.values()) < 1:
        return {"valid": False, "reason": "momentum failed"}

    score = 0.60
    if ema_check["with_trend"]:
        score += 0.04
    if adx_check["strong"]:
        score += 0.04
    if rvol >= 2.0:
        score += 0.03
    if sum(smc_signals.values()) == 3:
        score += 0.06
    else:
        score += 0.03
    if sum(momentum_signals.values()) == 2:
        score += 0.03
    if rsi_check["bonus"]:
        score += 0.02
    if macd_check["expanding"] and macd_check["aligned"]:
        score += 0.02

    score = round(min(score * session_multiplier(session), 0.95), 4)
    return {"valid": score >= MIN_CONFIDENCE_SCORE, "direction": direction, "atr": atr, "score": score}


class LiveScannerReplicaStrategy(Strategy):
    risk_reward = MIN_REWARD_RISK_RATIO

    def next(self):
        if len(self.data) < 80 or self.position:
            return

        df = pd.DataFrame(
            {
                "Open": np.array(self.data.Open),
                "High": np.array(self.data.High),
                "Low": np.array(self.data.Low),
                "Close": np.array(self.data.Close),
                "Volume": np.array(self.data.Volume),
            },
            index=pd.to_datetime(self.data.index),
        )
        signal = live_style_signal(df.tail(120))
        if not signal.get("valid"):
            return

        price = float(self.data.Close[-1])
        atr = float(signal.get("atr", 0) or 0)
        if atr <= 0:
            return

        if signal["direction"] == "LONG":
            stop = price - atr * ATR_STOP_MULTIPLIER
            target = price + atr * ATR_STOP_MULTIPLIER * self.risk_reward
            self.buy(sl=stop, tp=target)
        elif signal["direction"] == "SHORT":
            stop = price + atr * ATR_STOP_MULTIPLIER
            target = price - atr * ATR_STOP_MULTIPLIER * self.risk_reward
            self.sell(sl=stop, tp=target)


def run_backtest(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
    cash: float = 100000,
    commission: float = 0.001,
):
    print(f"\nBacktesting {ticker} ({period}, {interval} bars)...")

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()

    if len(df) < 100:
        print(f"Insufficient data ({len(df)} bars) — try a longer period")
        return None

    bt = Backtest(
        df,
        LiveScannerReplicaStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS — {ticker}")
    print("=" * 60)
    for metric in [
        "Return [%]",
        "Max. Drawdown [%]",
        "Win Rate [%]",
        "Profit Factor",
        "Sharpe Ratio",
        "# Trades",
        "Avg. Trade Duration",
    ]:
        if metric in stats.index:
            print(f"  {metric:<30} {stats[metric]}")
    print("=" * 60)

    output_path = f"backtesting/{ticker}_backtest.html"
    bt.plot(filename=output_path, open_browser=False)
    print(f"Chart saved to {output_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the live-style technical strategy")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    parser.add_argument("--period", default="60d", help="Data period (5d,1mo,3mo,6mo,1y)")
    parser.add_argument("--interval", default="5m", help="Bar interval (5m,15m,30m,1h)")
    parser.add_argument("--cash", default=100000, type=float, help="Starting cash")
    args = parser.parse_args()

    run_backtest(args.ticker, args.period, args.interval, args.cash)
