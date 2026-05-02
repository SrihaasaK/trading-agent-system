"""
backtesting/strategy.py
Backtesting harness that mirrors the live technical scanner.
Uses the same indicator functions and gate logic as the live system.
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
    ADX_STRONG,
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
    collect_direction_votes,
    detect_bos_choch,
    detect_fvg,
    detect_order_block,
    detect_vwap_deviation,
    direction_matches_signal,
    normalize_direction,
    resolve_direction,
    allow_single_signal_setup,
    RVOL_HARD_FLOOR,
)
from config.settings import CONFIDENCE_TIERS, MIN_CONFIDENCE_SCORE


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
    """Backtest uses permissive session weights to measure all-session performance.
    Live system may block sessions (e.g., AFTERNOON=0.0) based on observed data."""
    return {"MORNING": 1.0, "AFTERNOON": 0.85, "MIDDAY": 0.80, "CLOSED": 0.0}.get(label, 0.5)


def _resolve_tier(score: float) -> tuple[str, dict]:
    """Map confidence score to tier, matching live signal_judge logic."""
    if score >= CONFIDENCE_TIERS["SWING"]["min_score"]:
        return "SWING", CONFIDENCE_TIERS["SWING"]
    if score >= CONFIDENCE_TIERS["STANDARD"]["min_score"]:
        return "STANDARD", CONFIDENCE_TIERS["STANDARD"]
    return "SCALP", CONFIDENCE_TIERS["SCALP"]


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
    price = float(close.iloc[-1])

    emas = calculate_all_emas(close)
    adx, pdi, mdi = calculate_adx(high, low, close)
    rsi = calculate_rsi(close)
    _, _, macd_h = calculate_macd(close)
    vwap = calculate_vwap(session_df)
    atr = calculate_atr(session_df)
    rvol = calculate_rvol(session_df["Volume"], df["Volume"])
    fvg = detect_fvg(session_df)
    bos = detect_bos_choch(session_df)
    ob = detect_order_block(session_df)
    vwap_dev = detect_vwap_deviation(session_df, vwap)

    # Use the same direction resolution as live (collect_direction_votes + resolve_direction)
    vote_sources = collect_direction_votes(fvg, bos, ob, vwap_dev)
    direction, resolution_method, vote_counts = resolve_direction(vote_sources)
    if direction == "NONE":
        return {"valid": False, "reason": f"no directional consensus: L={vote_counts.get('LONG',0)} S={vote_counts.get('SHORT',0)}"}

    adx_last = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
    pdi_last = float(pdi.iloc[-1]) if not np.isnan(pdi.iloc[-1]) else 0
    mdi_last = float(mdi.iloc[-1]) if not np.isnan(mdi.iloc[-1]) else 0
    rsi_last = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50

    ema_check = check_ema_stack(emas, direction, price)
    adx_check = check_adx(adx_last, pdi_last, mdi_last, direction)
    rsi_check = check_rsi(rsi_last, direction)
    macd_check = check_macd(macd_h, direction)
    vwap_position = check_vwap_position(price, float(vwap.iloc[-1]), direction)
    session = session_label(session_df.index[-1])

    # Hard gates — 4 of 5 must pass (matching live)
    hard_gates = {
        "session": session != "CLOSED",
        "ema_aligned": ema_check["passes"],
        "vwap_position": vwap_position,
        "adx_strength": adx_check["passes"],
        "rvol": rvol >= RVOL_HARD_FLOOR,
    }
    if sum(hard_gates.values()) < 4:
        failed = [k for k, v in hard_gates.items() if not v]
        return {"valid": False, "reason": f"hard gate(s) failed: {failed}"}

    # SMC structure — 2/3 or strong override with 1/3 (matching live)
    smc_signals = {
        "fvg": fvg["detected"] and direction_matches_signal(direction, fvg.get("type", "")) if fvg["detected"] else False,
        "bos": bos["detected"] and direction_matches_signal(direction, bos.get("direction", "")),
        "order_block": ob["detected"] and direction_matches_signal(direction, ob.get("type", "")) and ob.get("in_zone", False) if ob["detected"] else False,
    }
    smc_count = sum(smc_signals.values())
    smc_strong_override = smc_count >= 1 and adx_last >= ADX_STRONG and rvol >= 1.0
    smc_passes = smc_count >= 2 or smc_strong_override

    momentum_signals = {"rsi": rsi_check["passes"], "macd": macd_check["passes"]}
    momentum_count = sum(momentum_signals.values())
    if momentum_count < 1:
        return {"valid": False, "reason": "momentum failed"}

    guarded_single_signal = allow_single_signal_setup(
        resolution_method, smc_count, momentum_count, ema_check, adx_check, rvol,
    )
    if not smc_passes and not guarded_single_signal:
        return {"valid": False, "reason": f"SMC structure: only {smc_count}/3 signals (need 2)"}

    # Confidence scoring (matching live)
    score = 0.56 if guarded_single_signal else 0.60
    if ema_check.get("with_trend"):
        score += 0.04
    if adx_check["strong"]:
        score += 0.04
    if rvol >= 1.5:
        score += 0.04
    elif rvol >= 1.0:
        score += 0.02
    elif rvol >= 0.7:
        score += 0.01
    if smc_count == 3:
        score += 0.06
    elif smc_count == 2:
        score += 0.03
    if momentum_count == 2:
        score += 0.03
    if rsi_check["bonus"]:
        score += 0.02
    if macd_check["expanding"] and macd_check["aligned"]:
        score += 0.02
    if guarded_single_signal:
        score -= 0.01

    score *= session_multiplier(session)
    score = round(min(score, 0.95), 4)

    if score < MIN_CONFIDENCE_SCORE:
        return {"valid": False, "reason": f"Score {score:.3f} below threshold {MIN_CONFIDENCE_SCORE}"}

    tier_name, tier = _resolve_tier(score)
    return {
        "valid": True,
        "direction": direction,
        "atr": atr,
        "score": score,
        "tier": tier_name,
        "atr_stop_mult": tier["atr_stop_mult"],
        "reward_risk": tier["reward_risk"],
    }


class LiveScannerReplicaStrategy(Strategy):

    def init(self):
        pass

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

        atr_mult = signal["atr_stop_mult"]
        rr = signal["reward_risk"]

        if signal["direction"] == "LONG":
            stop = price - atr * atr_mult
            target = price + atr * atr_mult * rr
            self.buy(sl=stop, tp=target)
        elif signal["direction"] == "SHORT":
            stop = price + atr * atr_mult
            target = price - atr * atr_mult * rr
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
        print(f"Insufficient data ({len(df)} bars) -- try a longer period")
        return None

    bt = Backtest(
        df,
        LiveScannerReplicaStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True,
    )
    stats = bt.run()

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS -- {ticker}")
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
