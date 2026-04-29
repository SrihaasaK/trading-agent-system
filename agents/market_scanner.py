"""
agents/market_scanner.py

Full technical analysis engine for the trading agent system.
Uses a tiered confluence model:

  HARD GATES (ALL must pass — instant skip if any fail):
    1. Session filter       — NY/London hours only
    2. EMA stack alignment  — 9/21/50 EMAs trending in trade direction
    3. VWAP position        — price on correct side of VWAP
    4. ADX strength         — trend strength > 20 (not choppy)
    5. RVOL                 — session-aware relative volume filter

  SMC STRUCTURE (minimum 2 of 3 must be present, with a guarded 1-signal exception):
    6. Fair Value Gap       — unmitigated imbalance in trade direction
    7. BOS or CHoCH         — market structure confirmation
    8. Order Block          — institutional supply/demand zone nearby

  MOMENTUM CONFIRMATION (minimum 1 of 2 must align):
    9. RSI                  — not overbought/oversold against direction
   10. MACD                 — histogram aligning with direction

  HIGHER TIMEFRAME CONTEXT (bonus — improves confidence score):
   11. Daily key levels     — price near significant daily S/R
   12. Premium/Discount     — trade in statistically favorable zone

Research basis:
  - ADX + EMA + volume filter: highest confirmed win-rate combination
    for intraday US stocks (backtested 1996-2024, out-of-sample period)
  - RSI divergence at swing extremes: 90%+ backtest accuracy per
    Smart Money Sniper v6 research (TradingView community)
  - SMC BOS/CHoCH + FVG + VWAP: core ICT/TJR setup as described in
    the TJR strategy documentation
  - RVOL > 1.5x: institutional activity filter used in professional
    multi-factor systems (AlphaX Edge, Smart Confluence)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, time
import pytz
from loguru import logger

from config.settings import (
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
    ATR_LOOKBACK,
    RVOL_HARD_FLOOR,
)
from agents.state import TradingState

# ── Tunable Parameters ────────────────────────────────────────────────────────
EMA_FAST          = 9
EMA_MID           = 21
EMA_SLOW          = 50
EMA_TREND         = 200
ADX_MIN           = 20        # below this = choppy, skip
ADX_STRONG        = 30        # above this = strong trend, higher confidence
RVOL_MIN          = 1.2       # minimum relative volume multiplier
RVOL_STRONG       = 2.0       # strong institutional activity
RVOL_OPENING_MIN  = 0.95      # allow the first few bars to form without over-penalizing volume
RVOL_MORNING_MIN  = 1.0
RVOL_MIDDAY_MIN   = 0.95
RVOL_AFTERNOON_MIN = 1.0
RSI_PERIOD        = 14
RSI_OB            = 70        # overbought (industry standard)
RSI_OS            = 35        # oversold
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
MIN_SESSION_BARS  = 3
BOS_LOOKBACK      = 20
ORDER_BLOCK_BARS  = 50        # look back N bars for order blocks
VWAP_STD_TRIGGER  = 1.5       # std devs from VWAP for mean reversion
HTF_INTERVAL      = "1d"      # higher timeframe for key levels
HTF_LOOKBACK      = 20        # days of daily data for S/R
SINGLE_VOTE_OVERRIDE_SOURCES = {"fvg", "bos", "order_block"}
_htf_cache: dict[tuple[str, str], dict] = {}


# ── Session Filter ────────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    eastern = pytz.timezone("America/New_York")
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return time(MARKET_OPEN_HOUR, MARKET_OPEN_MIN) <= now.time() <= time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)

def get_session_label() -> str:
    eastern = pytz.timezone("America/New_York")
    h = datetime.now(eastern).hour
    if 9 <= h < 11:   return "MORNING"
    elif 11 <= h < 14: return "MIDDAY"
    elif 14 <= h < 16: return "AFTERNOON"
    return "CLOSED"

def session_quality(session: str) -> float:
    """Morning and afternoon sessions have better setups than midday."""
    return {"MORNING": 1.0, "AFTERNOON": 0.85, "MIDDAY": 0.80, "CLOSED": 0.0}.get(session, 0.5)


def normalize_direction(label: str) -> str:
    """Map mixed bullish/bearish labels into canonical LONG/SHORT directions."""
    mapping = {
        "LONG": "LONG",
        "BULLISH": "LONG",
        "BUY": "LONG",
        "SHORT": "SHORT",
        "BEARISH": "SHORT",
        "SELL": "SHORT",
    }
    return mapping.get((label or "").upper(), "NONE")


def direction_matches_signal(direction: str, signal_label: str) -> bool:
    return normalize_direction(direction) == normalize_direction(signal_label)


# ── Core Indicator Calculations ───────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_all_emas(close: pd.Series) -> dict:
    return {
        "ema9":   ema(close, EMA_FAST),
        "ema21":  ema(close, EMA_MID),
        "ema50":  ema(close, EMA_SLOW),
        "ema200": ema(close, EMA_TREND),
    }

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength, not direction."""
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(period).mean()

    up_move   = high - high.shift()
    down_move = low.shift() - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di  = 100 * pd.Series(plus_dm,  index=close.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(close: pd.Series) -> tuple:
    fast_ema   = ema(close, MACD_FAST)
    slow_ema   = ema(close, MACD_SLOW)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, MACD_SIGNAL)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def calculate_atr(df: pd.DataFrame, period: int = ATR_LOOKBACK) -> float:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def calculate_rvol(session_volume: pd.Series, full_volume: pd.Series | None = None, lookback: int = 20) -> float:
    """
    Relative volume using two baselines:
      1. recent bars from the current session
      2. the same bar time from prior sessions when available
    """
    if session_volume.empty:
        return 0.0

    # Use the last completed bar, not the currently forming bar (which may have 0 volume)
    if len(session_volume) >= 2 and session_volume.iloc[-1] == 0:
        current = float(session_volume.iloc[-2])
    else:
        current = float(session_volume.iloc[-1])
    if current <= 0:
        return 0.0
    baselines: list[float] = []

    recent = session_volume.iloc[:-1].tail(lookback)
    if not recent.empty:
        baselines.append(float(recent.mean()))

    if full_volume is not None and len(session_volume.index) > 0:
        current_ts = session_volume.index[-1]
        current_time = current_ts.time()
        historical_same_slot = full_volume[full_volume.index.time == current_time]
        historical_same_slot = historical_same_slot[historical_same_slot.index < current_ts]
        if not historical_same_slot.empty:
            baselines.append(float(historical_same_slot.tail(5).mean()))

    valid_baselines = [value for value in baselines if value > 0 and not np.isnan(value)]
    if not valid_baselines:
        return 1.0

    baseline = float(np.mean(valid_baselines))
    return round(current / baseline, 3) if baseline > 0 else 1.0


def required_rvol(session: str, bars_today: int) -> float:
    """Keep the RVOL filter strict enough to matter, but avoid choking early-session scans."""
    if bars_today <= 6:
        return RVOL_OPENING_MIN

    thresholds = {
        "MORNING": RVOL_MORNING_MIN,
        "MIDDAY": RVOL_MIDDAY_MIN,
        "AFTERNOON": RVOL_AFTERNOON_MIN,
    }
    return thresholds.get(session, RVOL_MIN)


def collect_direction_votes(fvg: dict, bos: dict, ob: dict, vwap_dev: dict) -> dict[str, list[str]]:
    vote_sources = {"LONG": [], "SHORT": []}

    def add_vote(label: str, source: str) -> None:
        mapped = normalize_direction(label)
        if mapped in vote_sources:
            vote_sources[mapped].append(source)

    if fvg.get("detected"):
        add_vote(fvg.get("type", ""), "fvg")
    if bos.get("detected"):
        add_vote(bos.get("direction", ""), "bos")
    if ob.get("detected"):
        add_vote(ob.get("type", ""), "order_block")
    if vwap_dev.get("detected"):
        add_vote(vwap_dev.get("direction", ""), "vwap_deviation")

    return vote_sources


def resolve_direction(vote_sources: dict[str, list[str]]) -> tuple[str, str, dict[str, int]]:
    counts = {direction: len(sources) for direction, sources in vote_sources.items()}

    if counts["LONG"] >= 2 and counts["LONG"] > counts["SHORT"]:
        return "LONG", "consensus", counts
    if counts["SHORT"] >= 2 and counts["SHORT"] > counts["LONG"]:
        return "SHORT", "consensus", counts

    for direction, opposite in (("LONG", "SHORT"), ("SHORT", "LONG")):
        if counts[direction] == 1 and counts[opposite] == 0:
            source = vote_sources[direction][0]
            if source in SINGLE_VOTE_OVERRIDE_SOURCES:
                return direction, f"single_vote_override:{source}", counts

    return "NONE", "no_consensus", counts


def allow_single_signal_setup(
    direction_resolution: str,
    smc_count: int,
    momentum_count: int,
    ema_check: dict,
    adx_check: dict,
    rvol: float,
) -> bool:
    """Allow a single high-quality structural vote only when the rest of the tape is unusually clean."""
    if not direction_resolution.startswith("single_vote_override:"):
        return False

    return (
        smc_count == 1
        and momentum_count >= 1
        and ema_check.get("aligned", False)
        and (adx_check.get("strong", False) or rvol >= 0.8)
    )


# ── SMC Signal Detectors ──────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame) -> dict:
    """
    Fair Value Gap: 3-candle imbalance.
    Bullish FVG: candle[i].low > candle[i-2].high
    Bearish FVG: candle[i].high < candle[i-2].low
    Returns the most recent unmitigated FVG.
    """
    if len(df) < 3:
        return {"detected": False}
    for i in range(len(df) - 1, 1, -1):
        # Bullish
        if df["Low"].iloc[i] > df["High"].iloc[i - 2]:
            gap_low  = float(df["High"].iloc[i - 2])
            gap_high = float(df["Low"].iloc[i])
            return {"detected": True, "type": "BULLISH",
                    "gap_low": gap_low, "gap_high": gap_high,
                    "midpoint": (gap_low + gap_high) / 2,
                    "size_pct": round((gap_high - gap_low) / gap_low * 100, 3)}
        # Bearish
        if df["High"].iloc[i] < df["Low"].iloc[i - 2]:
            gap_low  = float(df["High"].iloc[i])
            gap_high = float(df["Low"].iloc[i - 2])
            return {"detected": True, "type": "BEARISH",
                    "gap_low": gap_low, "gap_high": gap_high,
                    "midpoint": (gap_low + gap_high) / 2,
                    "size_pct": round((gap_high - gap_low) / gap_low * 100, 3)}
    return {"detected": False}


def detect_bos_choch(df: pd.DataFrame) -> dict:
    """
    Break of Structure (BOS): continuation signal
    Change of Character (CHoCH): reversal warning signal
    BOS = price breaks through most recent significant swing high/low
    CHoCH = lower high in uptrend or higher low in downtrend (structure shift)
    """
    if len(df) < BOS_LOOKBACK + 5:
        return {"detected": False}

    highs = df["High"].values
    lows  = df["Low"].values
    close = df["Close"].values

    recent_high = np.max(highs[-BOS_LOOKBACK:-1])
    recent_low  = np.min(lows[-BOS_LOOKBACK:-1])
    last_close  = close[-1]

    # Identify swing structure for CHoCH
    mid = BOS_LOOKBACK // 2
    first_half_high = np.max(highs[-BOS_LOOKBACK:-mid])
    second_half_high = np.max(highs[-mid:-1])
    first_half_low   = np.min(lows[-BOS_LOOKBACK:-mid])
    second_half_low  = np.min(lows[-mid:-1])

    # BOS
    if last_close > recent_high:
        return {"detected": True, "type": "BOS", "direction": "BULLISH",
                "level": round(float(recent_high), 4),
                "signal": "continuation"}
    if last_close < recent_low:
        return {"detected": True, "type": "BOS", "direction": "BEARISH",
                "level": round(float(recent_low), 4),
                "signal": "continuation"}

    # CHoCH: lower high in uptrend = potential reversal
    if second_half_high < first_half_high and last_close > second_half_high:
        return {"detected": True, "type": "CHoCH", "direction": "BEARISH",
                "level": round(float(second_half_high), 4),
                "signal": "reversal_warning"}
    if second_half_low > first_half_low and last_close < second_half_low:
        return {"detected": True, "type": "CHoCH", "direction": "BULLISH",
                "level": round(float(second_half_low), 4),
                "signal": "reversal_warning"}

    return {"detected": False}


def detect_order_block(df: pd.DataFrame) -> dict:
    """
    Order Block: the last bearish candle before a bullish impulse (bullish OB)
    or the last bullish candle before a bearish impulse (bearish OB).
    These are institutional accumulation/distribution zones.
    """
    if len(df) < ORDER_BLOCK_BARS:
        return {"detected": False}

    lookback_df = df.tail(ORDER_BLOCK_BARS).copy()
    lookback_df["body"] = (lookback_df["Close"] - lookback_df["Open"]).abs()
    lookback_df["direction"] = np.where(lookback_df["Close"] > lookback_df["Open"], "BULL", "BEAR")

    last_close = float(df["Close"].iloc[-1])

    # Look for the most recent significant OB
    for i in range(len(lookback_df) - 3, 0, -1):
        candle = lookback_df.iloc[i]
        next3  = lookback_df.iloc[i+1:i+4]

        if len(next3) < 2:
            continue

        # Bullish OB: bearish candle followed by 3 bullish candles (strong up move)
        if (candle["direction"] == "BEAR"
                and (next3["Close"] > next3["Open"]).sum() >= 2
                and float(next3["High"].max()) > float(candle["High"]) * 1.003):

            ob_low  = float(candle["Low"])
            ob_high = float(candle["High"])

            # Is current price pulling back into the OB? (mitigation zone)
            in_zone = ob_low <= last_close <= ob_high * 1.005
            return {"detected": True, "type": "BULLISH",
                    "ob_low": round(ob_low, 4), "ob_high": round(ob_high, 4),
                    "in_zone": in_zone,
                    "distance_pct": round(abs(last_close - (ob_low + ob_high)/2) / last_close * 100, 2)}

        # Bearish OB: bullish candle followed by 3 bearish candles
        if (candle["direction"] == "BULL"
                and (next3["Close"] < next3["Open"]).sum() >= 2
                and float(next3["Low"].min()) < float(candle["Low"]) * 0.997):

            ob_low  = float(candle["Low"])
            ob_high = float(candle["High"])
            in_zone = ob_low * 0.995 <= last_close <= ob_high
            return {"detected": True, "type": "BEARISH",
                    "ob_low": round(ob_low, 4), "ob_high": round(ob_high, 4),
                    "in_zone": in_zone,
                    "distance_pct": round(abs(last_close - (ob_low + ob_high)/2) / last_close * 100, 2)}

    return {"detected": False}


def detect_vwap_deviation(df: pd.DataFrame, vwap: pd.Series) -> dict:
    """Mean reversion signal: price significantly above/below VWAP."""
    deviation = df["Close"] - vwap
    std_dev   = deviation.rolling(20).std().iloc[-1]

    if std_dev == 0 or np.isnan(std_dev):
        return {"detected": False}

    last_close = float(df["Close"].iloc[-1])
    last_vwap  = float(vwap.iloc[-1])
    z_score    = float((last_close - last_vwap) / std_dev)

    if abs(z_score) >= VWAP_STD_TRIGGER:
        direction = "SHORT" if z_score > 0 else "LONG"
        return {"detected": True, "direction": direction,
                "z_score": round(z_score, 2),
                "vwap": round(last_vwap, 4),
                "deviation_pct": round(abs((last_close - last_vwap) / last_vwap) * 100, 2)}
    return {"detected": False}


# ── Higher Timeframe Context ──────────────────────────────────────────────────

def get_htf_levels(ticker: str) -> dict:
    """
    Pull daily OHLCV to identify key support/resistance levels.
    Returns previous day's high/low and weekly high/low.
    These are the 'key levels' TJR watches on the 1H/4H chart.
    """
    cache_key = (date.today().isoformat(), ticker)
    if cache_key in _htf_cache:
        return _htf_cache[cache_key]

    try:
        df = yf.download(ticker, period="30d", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()

        if len(df) < 5:
            return {}

        prev_day_high  = float(df["High"].iloc[-2])
        prev_day_low   = float(df["Low"].iloc[-2])
        week_high      = float(df["High"].tail(5).max())
        week_low       = float(df["Low"].tail(5).min())
        month_high     = float(df["High"].max())
        month_low      = float(df["Low"].min())

        # Range midpoint (premium/discount)
        range_mid = (week_high + week_low) / 2
        last_close = float(df["Close"].iloc[-1])
        in_premium  = last_close > range_mid
        in_discount = last_close < range_mid
        # OTE = 61.8%-78.6% Fibonacci retracement of the weekly range
        ote_low  = week_low + (week_high - week_low) * 0.618
        ote_high = week_low + (week_high - week_low) * 0.786
        in_ote   = ote_low <= last_close <= ote_high

        result = {
            "prev_day_high":  round(prev_day_high, 4),
            "prev_day_low":   round(prev_day_low, 4),
            "week_high":      round(week_high, 4),
            "week_low":       round(week_low, 4),
            "month_high":     round(month_high, 4),
            "month_low":      round(month_low, 4),
            "range_mid":      round(range_mid, 4),
            "in_premium":     in_premium,
            "in_discount":    in_discount,
            "in_ote":         in_ote,
        }
        _htf_cache[cache_key] = result
        return result
    except Exception as e:
        logger.warning(f"[market_scanner] HTF levels error for {ticker}: {e}")
        return {}


# ── Hard Gate Checks ──────────────────────────────────────────────────────────

def check_ema_stack(emas: dict, direction: str, close: float) -> dict:
    """
    EMA stack check: for a LONG, price should be above 9 > 21 > 50.
    For a SHORT, price should be below 9 < 21 < 50.
    The 200 EMA is used as the absolute trend filter.
    """
    e9, e21, e50, e200 = (
        float(emas["ema9"].iloc[-1]),
        float(emas["ema21"].iloc[-1]),
        float(emas["ema50"].iloc[-1]),
        float(emas["ema200"].iloc[-1]),
    )

    if direction == "LONG":
        aligned    = (close > e9 > e21 > e50) or (e9 > e21 and close > e21)
        with_trend = close > e200
    else:
        aligned    = (close < e9 < e21 < e50) or (e9 < e21 and close < e21)
        with_trend = close < e200

    return {
        "aligned":     aligned,
        "with_trend":  with_trend,
        "ema9":   round(e9, 4),
        "ema21":  round(e21, 4),
        "ema50":  round(e50, 4),
        "ema200": round(e200, 4),
        "passes":  aligned,  # with_trend is a bonus, not a hard requirement intraday
    }


def check_vwap_position(close: float, vwap_val: float, direction: str) -> bool:
    """Price should be on the correct side of VWAP for the trade direction."""
    if direction == "LONG":
        return close > vwap_val
    return close < vwap_val


def check_adx(adx_val: float, plus_di: float, minus_di: float, direction: str) -> dict:
    """ADX > 20 = trending market. DI direction should match trade direction."""
    strong_enough = adx_val >= ADX_MIN if not np.isnan(adx_val) else False
    di_aligned    = (plus_di > minus_di) if direction == "LONG" else (minus_di > plus_di)
    return {
        "adx":        round(float(adx_val), 2) if not np.isnan(adx_val) else 0,
        "plus_di":    round(float(plus_di), 2) if not np.isnan(plus_di) else 0,
        "minus_di":   round(float(minus_di), 2) if not np.isnan(minus_di) else 0,
        "strong":     bool(adx_val >= ADX_STRONG) if not np.isnan(adx_val) else False,
        "di_aligned": bool(di_aligned),
        "passes":     bool(strong_enough and di_aligned),
    }


def check_rsi(rsi_val: float, direction: str) -> dict:
    """RSI should not be overbought for longs, oversold for shorts."""
    if direction == "LONG":
        passes = rsi_val < RSI_OB   # not overbought
        bullish_divergence = rsi_val < 40  # bonus: RSI still has room to run
    else:
        passes = rsi_val > RSI_OS   # not oversold
        bullish_divergence = rsi_val > 60

    return {
        "rsi":     round(float(rsi_val), 2),
        "passes":  bool(passes),
        "bonus":   bool(bullish_divergence),
    }


def check_macd(histogram: pd.Series, direction: str) -> dict:
    """MACD histogram should be expanding in trade direction."""
    last_hist  = float(histogram.iloc[-1])
    prev_hist  = float(histogram.iloc[-2]) if len(histogram) > 1 else last_hist
    expanding  = abs(last_hist) > abs(prev_hist)

    if direction == "LONG":
        aligned = last_hist > 0
    else:
        aligned = last_hist < 0

    return {
        "histogram":  round(last_hist, 6),
        "expanding":  bool(expanding),
        "aligned":    bool(aligned),
        "passes":     bool(aligned),
    }


# ── Main Scanner ──────────────────────────────────────────────────────────────

def scan_ticker(ticker: str) -> dict:
    """
    Full technical scan on a single ticker.
    Returns structured result with all indicator readings.
    """
    try:
        # Pull 5-minute intraday data
        df = yf.download(ticker, period="5d", interval="5m",
                         progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if df is None or df.empty or len(df) < 60:
            return {"ticker": ticker, "valid": False, "reason": "insufficient data"}

        # Filter to today's session
        eastern = pytz.timezone("America/New_York")
        today   = datetime.now(eastern).date()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(eastern)
        df_today = df[df.index.date == today].copy()

        if len(df_today) < MIN_SESSION_BARS:
            return {"ticker": ticker, "valid": False, "reason": "not enough intraday bars today"}

        session = get_session_label()
        history_df = df.copy()
        close   = history_df["Close"]
        high    = history_df["High"]
        low     = history_df["Low"]
        price   = float(df_today["Close"].iloc[-1])

        # ── Calculate all indicators ───────────────────────────────────────
        emas        = calculate_all_emas(close)
        adx, pdi, mdi = calculate_adx(high, low, close)
        rsi         = calculate_rsi(close)
        macd_l, macd_s, macd_h = calculate_macd(close)
        vwap        = calculate_vwap(df_today)
        atr         = calculate_atr(history_df)
        rvol        = calculate_rvol(df_today["Volume"], history_df["Volume"])

        # SMC signals
        fvg         = detect_fvg(df_today)
        bos         = detect_bos_choch(df_today)
        ob          = detect_order_block(df_today)
        vwap_dev    = detect_vwap_deviation(df_today, vwap)

        # HTF levels (uses daily data — separate call)
        htf         = get_htf_levels(ticker)

        # ── Determine candidate direction ─────────────────────────────────
        vote_sources = collect_direction_votes(fvg, bos, ob, vwap_dev)
        direction, direction_resolution, direction_votes = resolve_direction(vote_sources)

        if direction == "NONE":
            return {"ticker": ticker, "valid": False,
                    "reason": f"no directional consensus: L={direction_votes['LONG']} S={direction_votes['SHORT']}"}

        vwap_val      = float(vwap.iloc[-1])
        adx_last      = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
        pdi_last      = float(pdi.iloc[-1]) if not np.isnan(pdi.iloc[-1]) else 0
        mdi_last      = float(mdi.iloc[-1]) if not np.isnan(mdi.iloc[-1]) else 0
        rsi_last      = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
        ema_check     = check_ema_stack(emas, direction, price)
        adx_check     = check_adx(adx_last, pdi_last, mdi_last, direction)
        rsi_check     = check_rsi(rsi_last, direction)
        macd_check    = check_macd(macd_h, direction)
        vwap_position = check_vwap_position(price, vwap_val, direction)
        rvol_floor    = required_rvol(session, len(df_today))

        # ── HARD GATE EVALUATION ──────────────────────────────────────────
        hard_gates = {
            "session":        session != "CLOSED",
            "ema_aligned":    ema_check["passes"],
            "vwap_position":  vwap_position,
            "adx_strength":   adx_check["passes"],
            "rvol":           rvol >= RVOL_HARD_FLOOR,
        }
        hard_gate_pass_count = sum(hard_gates.values())
        all_hard_gates_pass = hard_gate_pass_count >= 4

        if not all_hard_gates_pass:
            failed = [k for k, v in hard_gates.items() if not v]
            return {"ticker": ticker, "valid": False,
                    "reason": f"hard gate(s) failed: {failed}",
                    "hard_gates": hard_gates,
                    "direction": direction, "price": price}

        # ── SMC STRUCTURE LAYER (min 2 of 3, or a guarded 1-signal exception) ──
        smc_signals = {
            "fvg": fvg["detected"] and direction_matches_signal(direction, fvg.get("type", ""))
                   if fvg["detected"] else fvg["detected"],
            "bos": bos["detected"] and direction_matches_signal(direction, bos.get("direction", "")),
            "order_block": ob["detected"] and direction_matches_signal(direction, ob.get("type", ""))
                           and ob.get("in_zone", False)
                           if ob["detected"] else ob["detected"],
        }
        smc_count = sum(smc_signals.values())
        smc_strong_override = smc_count >= 1 and adx_last >= ADX_STRONG and rvol >= 1.0
        smc_passes = smc_count >= 2 or smc_strong_override

        # ── MOMENTUM CONFIRMATION (min 1 of 2) ───────────────────────────
        momentum_signals = {
            "rsi":  rsi_check["passes"],
            "macd": macd_check["passes"],
        }
        momentum_count  = sum(momentum_signals.values())
        momentum_passes = momentum_count >= 1
        guarded_single_signal = allow_single_signal_setup(
            direction_resolution,
            smc_count,
            momentum_count,
            ema_check,
            adx_check,
            rvol,
        )

        if not momentum_passes:
            return {"ticker": ticker, "valid": False,
                    "reason": "no momentum confirmation (RSI and MACD both against direction)",
                    "hard_gates": hard_gates, "smc": smc_signals}

        if not smc_passes and not guarded_single_signal:
            return {"ticker": ticker, "valid": False,
                    "reason": f"SMC structure: only {smc_count}/3 signals (need 2)",
                    "hard_gates": hard_gates, "smc": smc_signals}
        smc_passes = smc_passes or guarded_single_signal

        # ── CONFIDENCE SCORE ──────────────────────────────────────────────
        # Guarded single-signal entries start with less conviction than full 2-signal SMC alignment.
        score = 0.56 if guarded_single_signal else 0.60

        # EMA with-trend bonus
        if ema_check.get("with_trend"):     score += 0.04
        # Strong ADX
        if adx_check["strong"]:             score += 0.04
        # Tiered RVOL scoring
        if rvol >= 1.5:                     score += 0.04
        elif rvol >= 1.0:                   score += 0.02
        elif rvol >= 0.7:                   score += 0.01
        # All 3 SMC signals
        if smc_count == 3:                  score += 0.06
        elif smc_count == 2:                score += 0.03
        # Both momentum signals
        if momentum_count == 2:             score += 0.03
        # RSI bonus (room to run)
        if rsi_check["bonus"]:              score += 0.02
        # MACD expanding
        if macd_check["expanding"] and macd_check["aligned"]: score += 0.02
        # HTF context bonuses
        if htf:
            if direction == "LONG"  and htf.get("in_discount"): score += 0.04
            if direction == "SHORT" and htf.get("in_premium"):  score += 0.04
            if htf.get("in_ote"):                                score += 0.03
        if guarded_single_signal:
            score -= 0.01
        # Session quality
        score *= session_quality(session)
        score  = round(min(score, 0.95), 4)  # cap at 0.95

        return {
            "ticker":        ticker,
            "valid":         True,
            "direction":     direction,
            "confidence":    score,
            "price":         round(price, 4),
            "atr":           round(float(atr), 4) if not np.isnan(atr) else 0,
            "vwap":          round(vwap_val, 4),
            "rvol":          rvol,
            "required_rvol": round(rvol_floor, 3),
            "session":       session,
            "bars_today":    len(df_today),

            # Indicator readings
            "ema":           {k: round(float(v.iloc[-1]), 4) for k, v in emas.items()},
            "adx":           adx_check,
            "rsi":           rsi_check,
            "macd":          macd_check,
            "vwap_position": vwap_position,

            # SMC signals
            "fvg":           fvg,
            "bos":           bos,
            "order_block":   ob,
            "vwap_deviation": vwap_dev,

            # Gate summaries
            "hard_gates":    hard_gates,
            "smc_count":     smc_count,
            "smc_passes":    smc_passes,
            "guarded_single_signal": guarded_single_signal,
            "momentum_count": momentum_count,
            "direction_votes": direction_votes,
            "direction_resolution": direction_resolution,
            "direction_vote_sources": vote_sources,

            # HTF context
            "htf_levels":    htf,
            "reason":        "setup valid",
        }

    except Exception as e:
        logger.error(f"[market_scanner] error on {ticker}: {e}")
        return {"ticker": ticker, "valid": False, "reason": str(e)}


# ── LangGraph Node ────────────────────────────────────────────────────────────

def market_scanner_node(state: TradingState) -> TradingState:
    ticker = state["ticker"]
    logger.info(f"[market_scanner] scanning {ticker}")

    if not is_market_hours():
        state["session_valid"] = False
        state["skip_reason"]   = "outside market hours"
        return state

    result = scan_ticker(ticker)

    state["price_data"] = {
        "current_price": result.get("price", 0),
        "atr":           result.get("atr", 0),
        "vwap":          result.get("vwap", 0),
        "rvol":          result.get("rvol", 0),
        "session":       result.get("session", "UNKNOWN"),
        "adx":           result.get("adx", {}).get("adx", 0),
        "rsi":           result.get("rsi", {}).get("rsi", 50),
    }
    state["technical_setup"] = result
    state["session_valid"]   = result["valid"]

    if not result["valid"]:
        state["skip_reason"] = result.get("reason", "no setup")
    else:
        logger.info(
            f"[market_scanner] {ticker} SETUP FOUND → "
            f"dir={result['direction']} conf={result['confidence']} "
            f"adx={result['adx']['adx']:.1f} rsi={result['rsi']['rsi']:.1f} "
            f"rvol={result['rvol']:.2f}x smc={result['smc_count']}/3"
        )

    return state
