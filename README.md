# Trading Agent System

A multi-agent algorithmic trading system for US stocks built with LangGraph,
Claude AI, and Alpaca. Runs fully locally on your Mac Mini at $0 infrastructure cost.

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
Check win rate and profit factor. Aim for profit factor > 1.5 before going live.

### 4. Start the bot (paper trading)
```bash
python main.py
```

### 5. Open the dashboard
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
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
│   └── app.py                ← Streamlit monitoring UI
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
http://mac-mini:8501
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
