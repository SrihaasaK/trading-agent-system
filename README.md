# Trading Agent System

A multi-agent algorithmic trading system for US equities built with LangGraph, Claude AI,
and Alpaca. Written by an 18-year-old incoming CS freshman as a learning project. Currently
running in paper-trading mode on a Mac Mini. No live capital has ever been deployed.

---

## Architecture

```
main.py (APScheduler — runs every 15min during market hours)
    │
    └── orchestrator.py (LangGraph state machine)
            │
            ├── market_scanner     → VWAP deviation, FVG, BOS detection
            ├── news_researcher    → Finnhub + SEC EDGAR + Reddit sentiment
            ├── macro_context      → VIX regime + market breadth (cached)
            ├── risk_analyst       → Position sizing + portfolio heat (Alpaca)
            ├── signal_judge       → Weighted scoring matrix + confidence gate
            ├── trade_executor     → Alpaca bracket orders + SQLite journal
            └── post_mortem        → Daily analysis + agent attribution
```

**Flow**: Scanner detects setup → News checks for vetoes → Macro checks regime →
Risk sizes the position → Judge scores all inputs → Executor places order (or skips)

---

## Current Status

Paper trading on Alpaca. No live trading has been attempted.

**What works end-to-end:**

- **Scanner** — VWAP, FVG, BOS/CHoCH detection, EMA stack, ADX, RVOL gates
- **News researcher** — Finnhub + SEC EDGAR + Reddit, Claude Haiku sentiment synthesis
- **Macro context** — VIX regime classification, broad market breadth, economic calendar
- **Risk analyst** — Position sizing, portfolio heat, daily loss limits, correlation checks
- **Signal judge** — Weighted multi-factor scoring, confidence tier gating
- **Trade executor** — Alpaca bracket orders with take-profit and stop-loss
- **Journal sync** — SQLite trade log with lifecycle tracking (open → filled → closed)
- **Post-mortem** — Daily agent attribution analysis, win-rate and R-multiple tracking
- **Dashboard** — FastAPI + vanilla JS live dashboard for monitoring signals and trades

---

## Limitations

- **No slippage or commission modeling.** Backtests assume fills at signal price. Real
  fills differ, especially on smaller-cap names or at open.
- **Not HFT or even fast intraday.** Claude API round-trips add 1–3 seconds per ticker.
  The system runs on 5-minute bars and scans every 15 minutes — it is deliberately slow.
- **Single-exchange US equities only.** Alpaca paper trading; no crypto, futures, or
  international markets.
- **Backtester diverges from live logic.** The `backtesting/strategy.py` file is a
  separate harness that does not share code with the live agents. Backtested numbers
  should be treated as rough directional signal, not reliable performance estimates.
- **Free-tier API rate limits.** Finnhub and the Alpaca free tier have request caps.
  Scanning large watchlists (40+ tickers) can exhaust limits within a session.

---

## What I'd Do Differently

- **Proper event sourcing for trade state.** The current SQLite journal is append-only
  but trade lifecycle updates are patched in place. A proper event log (each state
  transition as an immutable row) would make replay and debugging much easier.
- **Backtester as a thin wrapper around live agent code.** Right now the backtester
  reimplements signal logic separately, which means it drifts. The right approach is
  to run the actual LangGraph pipeline against historical bars with mocked I/O.
- **More integration tests.** Unit tests exist for individual agents but there are few
  end-to-end tests that exercise the full pipeline with mocked dependencies. Bugs that
  cross agent boundaries are hard to catch.
- **TypedDict return values throughout.** Several agent functions return plain `dict`.
  Using `TypedDict` (already defined in `state.py`) everywhere would let the type
  checker catch field name typos at write-time instead of runtime.
- **Separate config validation from config loading.** `config/settings.py` reads env
  vars and sets defaults at import time. There's no validation pass, so a missing key
  surfaces as an obscure `None`-related error deep in an agent rather than a clear
  startup failure.

---

## Quick Start

### 1. Clone and set up
```bash
git clone https://github.com/destroyer123456-dev/trading-agent-system.git
cd trading-agent-system
bash setup.sh
```

### 2. Add API keys to `.env`
```bash
cp .env.example .env
nano .env   # fill in your keys
```

Free API keys needed:
| Service | URL | What it's for |
|---------|-----|---------------|
| Alpaca | https://alpaca.markets | Paper trading + market data |
| Finnhub | https://finnhub.io | Real-time quotes + news |
| Anthropic | https://console.anthropic.com | Claude AI agents |

### 3. Run a backtest first
```bash
source venv/bin/activate
python backtesting/strategy.py --ticker AAPL --period 1y
```
Check win rate and profit factor. Treat the output as directional only — see Limitations.

### 4. Start the bot (paper trading)
```bash
python main.py
```

### 5. Open the dashboard
```bash
uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
# Opens at http://localhost:8000
```

---

## Configuration

All tunable parameters are in `config/settings.py`. Key ones:

```python
# Watchlist — keep under 40 tickers
WATCHLIST = ["AAPL", "MSFT", "NVDA", ...]

# Risk per trade (1% of portfolio = conservative default)
RISK_PER_TRADE_PCT = 0.01

# Minimum confidence to take a trade (0.68 = conservative)
MIN_CONFIDENCE_SCORE = 0.68

# Signal weights (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "technical": 0.35,
    "news":      0.25,
    "macro":     0.20,
    "risk":      0.20,
}

# VIX thresholds
VIX_NORMAL_MAX      = 20   # full size
VIX_CAUTION_MAX     = 30   # 50% size
VIX_HALT_THRESHOLD  = 35   # no new positions
```

**Important**: `REQUIRE_TRADE_APPROVAL=true` in `.env` means the bot will log
signals but will not auto-submit orders. Keep that enabled while validating,
and switch it off only when you want the paper account to execute automatically.

---

## File Structure

```
trading-agent-system/
├── agents/
│   ├── state.py              ← Shared LangGraph state definition
│   ├── orchestrator.py       ← LangGraph graph wiring
│   ├── market_scanner.py     ← Technical signals (VWAP, FVG, BOS)
│   ├── news_researcher.py    ← News + SEC + Reddit + Claude Haiku
│   ├── macro_context.py      ← VIX + regime + economic calendar
│   ├── risk_analyst.py       ← Position sizing + portfolio heat
│   ├── signal_judge.py       ← Weighted scoring + confidence gate
│   ├── trade_executor.py     ← Alpaca orders + SQLite logging
│   └── post_mortem.py        ← Daily analysis + agent attribution
├── skills/
│   ├── financial-analyst/    ← DCF + ratio analysis skill
│   ├── financial-data-collector/ ← Validated data fetching skill
│   ├── financial-services/   ← Earnings + trade thesis skill
│   └── post-mortem-analyst/  ← Post-mortem analysis skill
├── backtesting/
│   └── strategy.py           ← backtesting.py harness
├── dashboard/
│   └── app.py                ← FastAPI monitoring dashboard
├── config/
│   └── settings.py           ← All configurable parameters
├── data/                     ← Created at runtime (gitignored)
│   ├── journal.db            ← SQLite trade log
│   ├── chroma_db/            ← Agent memory
│   └── historical/           ← Parquet files
├── logs/                     ← Created at runtime (gitignored)
├── main.py                   ← Entry point
├── setup.sh                  ← One-time setup
├── com.tradingbot.plist      ← Mac Mini launchd config
└── requirements.txt
```

---

## Mac Mini Auto-Start (launchd)

The `setup.sh` script handles this, but if you want to do it manually:

```bash
# Edit the plist — replace YOUR_USERNAME with your Mac username
nano com.tradingbot.plist

# Copy to LaunchAgents
cp com.tradingbot.plist ~/Library/LaunchAgents/

# Load it (starts now and on every login)
launchctl load ~/Library/LaunchAgents/com.tradingbot.plist

# Check status
launchctl list | grep tradingbot

# View logs
tail -f logs/bot_$(date +%Y-%m-%d).log

# Stop the bot
launchctl unload ~/Library/LaunchAgents/com.tradingbot.plist
```

---

## LLM Cost Optimization

The system uses a tiered model strategy to keep API costs low:

| Task | Model | Est. cost/trade |
|------|-------|----------------|
| News triage, formatting | Claude Haiku | ~$0.001 |
| Risk analysis, signal scoring | Claude Sonnet | ~$0.004 |
| Post-mortem reports | Claude Sonnet | ~$0.010/day |
| Local preprocessing | Ollama (free) | $0.000 |

**At 5 trades/day → ~$5-8/month total API cost.**

To use Ollama for preprocessing (reduces cost further):
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2:8b
# Set in .env:
OLLAMA_ENABLED=true
```

---

## Remote Dashboard Access

Use Tailscale (free) to access the dashboard from anywhere:
```bash
# Install Tailscale on Mac Mini and your other devices
# https://tailscale.com/download

# Then access dashboard from any device on your Tailscale network:
http://mac-mini:8000
```

---

## Safety Checklist Before Going Live

- [ ] Paper traded for at least 30 days
- [ ] Backtest profit factor > 1.5 on out-of-sample data
- [ ] Post-mortem shows consistent agent accuracy
- [ ] `REQUIRE_APPROVAL = True` tested and working
- [ ] Telegram notifications tested
- [ ] VIX halt threshold understood and tested
- [ ] Change `ALPACA_BASE_URL` in `.env` to live endpoint
- [ ] Set `ENVIRONMENT=live` in `.env`

---

## Disclaimer

This system is for educational and research purposes. Past performance in
backtests does not guarantee future results. Always paper trade extensively
before using real money. Never risk more than you can afford to lose.
