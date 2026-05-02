"""
config/settings.py
Central configuration for the trading agent system.
All tunable parameters live here — never scatter magic numbers through agent code.
"""

import os
import pathlib

from dotenv import load_dotenv

load_dotenv()


# ── API Keys ──────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ALPACA_API_KEY      = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY   = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL     = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
FINNHUB_API_KEY     = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY   = os.getenv("ALPHA_VANTAGE_API_KEY", "")
REDDIT_CLIENT_ID    = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET       = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT   = os.getenv("REDDIT_USER_AGENT", "trading-agent-system/1.0")
NTFY_TOPIC          = os.getenv("NTFY_TOPIC", "")


# ── Environment ───────────────────────────────────────────────────────────────

ENVIRONMENT         = os.getenv("ENVIRONMENT", "paper")   # paper | live
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
IS_PAPER            = ENVIRONMENT == "paper"
REQUIRE_TRADE_APPROVAL = os.getenv("REQUIRE_TRADE_APPROVAL", "true").lower() == "true"


# ── Watchlist ─────────────────────────────────────────────────────────────────
# Keep this under 40 tickers — quality over quantity

WATCHLIST: list[str] = [
    # Mega-cap tech (backtest-validated)
    "NVDA", "GOOGL", "META", "AMZN", "AMD", "NFLX",
    # High-volume growth (strong backtest performance)
    "TSLA", "COIN", "BA", "NOW", "MU", "SOFI", "NET",
    # Sector diversification (healthcare, energy)
    "UNH", "CVX",
]

# Tickers used only for macro context — not traded directly
MACRO_TICKERS: list[str] = ["SPY", "QQQ", "IWM", "VXX", "TLT", "GLD"]


# ── Market Hours (Eastern Time) ───────────────────────────────────────────────

MARKET_OPEN_HOUR    = 9
MARKET_OPEN_MIN     = 30
MARKET_CLOSE_HOUR   = 16
MARKET_CLOSE_MIN    = 0
PRE_MARKET_BUFFER   = 30   # minutes before open to start preparation
POST_MARKET_BUFFER  = 10   # minutes after close for cleanup


# ── LLM Model Routing ─────────────────────────────────────────────────────────
# Haiku for fast/cheap tasks, Sonnet for analysis, use Opus sparingly

LLM_FAST    = "claude-haiku-4-5-20251001"      # news triage, data formatting
LLM_ANALYZE = "claude-sonnet-4-6"             # signal analysis, risk calc
LLM_DECIDE  = "claude-sonnet-4-6"             # final trade decisions
LLM_REPORT  = "claude-sonnet-4-6"             # post-mortem, reports

# Ollama local model (used when OLLAMA_ENABLED=True)
OLLAMA_MODEL    = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_ENABLED  = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"

# Groq (free-tier cloud LLM — alternative to Ollama for news synthesis)
# Get a free key at console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ── Risk Management ───────────────────────────────────────────────────────────

RISK_PER_TRADE_PCT      = 0.01      # 1% of portfolio per trade
MAX_PORTFOLIO_HEAT_PCT  = 0.05      # max 5% total at risk across all positions
MAX_OPEN_POSITIONS      = 5         # never hold more than 5 positions
MAX_POSITION_NOTIONAL_PCT = 0.20    # max 20% of portfolio in a single name
ATR_LOOKBACK            = 14        # periods for ATR calculation
TRADE_COOLDOWN_MINUTES  = 60        # avoid repeated entries on the same ticker
PORTFOLIO_CACHE_SECONDS = 15        # reuse Alpaca account state within one scan cycle


# ── Signal Confidence Gate ────────────────────────────────────────────────────
# Tune these after backtesting — don't change them based on a bad week

MIN_CONFIDENCE_SCORE    = 0.65      # below this: skip trade entirely (SCALP tier floor)
HIGH_CONFIDENCE_SCORE   = 0.82      # above this: SWING tier
RVOL_HARD_FLOOR         = 0.5       # below this = truly dead tape, hard block

# Signal weight breakdown (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "technical":    0.35,
    "news":         0.25,
    "macro":        0.20,
    "risk":         0.20,
}

# Confidence-tiered TP/SL parameters
# risk_analyst uses scanner confidence for a provisional tier; signal_judge
# overwrites with the final weighted score tier if they differ.
CONFIDENCE_TIERS = {
    "SWING":    {"min_score": 0.82, "atr_stop_mult": 3.5, "reward_risk": 3.0, "size_factor": 1.25},
    "STANDARD": {"min_score": 0.68, "atr_stop_mult": 2.75, "reward_risk": 2.0, "size_factor": 1.15},
    "SCALP":    {"min_score": 0.65, "atr_stop_mult": 2.0, "reward_risk": 1.5, "size_factor": 0.75},
}


# ── EOD Force-Close ──────────────────────────────────────────────────────────
EOD_CLOSE_MINUTE    = 50        # close day-traded positions at 15:50 ET
EOD_CLOSE_ENABLED   = os.getenv("EOD_CLOSE_ENABLED", "true").lower() == "true"

# ── Dead-Man Switch ──────────────────────────────────────────────────────────
HEARTBEAT_FILE      = None      # set after BASE_DIR is defined below
HEARTBEAT_STALE_MIN = 30        # minutes before watchdog fires alert

# ── Drawdown Circuit Breaker ─────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT = 0.02     # halt new trades if daily realized losses exceed 2%

# ── Trailing Stops ───────────────────────────────────────────────────────────
TRAILING_STOP_ENABLED       = os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true"
TRAILING_STOP_INTERVAL_MIN  = 5
TRAILING_STOP_BREAKEVEN_R   = 1.0   # move stop to breakeven at 1R profit
TRAILING_STOP_TRAIL_R       = 2.0   # move stop to 1R profit level at 2R profit

# ── VIX Regime Filters ────────────────────────────────────────────────────────

VIX_NORMAL_MAX          = 20        # below: normal regime, full sizing
VIX_CAUTION_MAX         = 30        # 20-30: reduce position size 50%
VIX_HALT_THRESHOLD      = 35        # above: no new positions at all


# ── Post-Mortem / Learning ────────────────────────────────────────────────────

POSTMORTEM_MIN_TRADES   = 5          # minimum trades before allowing param updates
POSTMORTEM_RUN_AT       = "16:30"   # time to run daily post-mortem (ET)
MONTHLY_REVIEW_DAY      = 1         # day of month for full review
STRATEGY_VERSION        = "3.0"


# ── Data Paths ────────────────────────────────────────────────────────────────

BASE_DIR        = pathlib.Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
HISTORICAL_DIR  = DATA_DIR / "historical"
LOGS_DIR        = BASE_DIR / "logs"
DB_PATH         = DATA_DIR / "journal.db"
CHROMA_PATH     = str(DATA_DIR / "chroma_db")

# Ensure directories exist
for d in [DATA_DIR, HISTORICAL_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Resolve HEARTBEAT_FILE now that BASE_DIR is available
HEARTBEAT_FILE = DATA_DIR / ".heartbeat"
