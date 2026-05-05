"""
dashboard/app.py
FastAPI dashboard server.
Run: cd /path/to/trading-agent-system && source venv/bin/activate
     uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.journal import get_dashboard_stats, get_trades_for_dashboard, _get_executed_trades
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER

app = FastAPI(title="Trading Agent Dashboard", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/portfolio")
def portfolio() -> JSONResponse:
    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
        account = client.get_account()
        positions = client.get_all_positions()
        return JSONResponse({
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "buying_power":    float(getattr(account, "buying_power", account.cash)),
            "pnl_today":       float(account.equity) - float(account.last_equity),
            "mode":            "PAPER" if IS_PAPER else "LIVE",
            "positions": [
                {
                    "symbol":  p.symbol,
                    "qty":     float(p.qty),
                    "value":   float(p.market_value),
                    "pnl":     float(p.unrealized_pl),
                    "pnl_pct": round(float(p.unrealized_plpc) * 100, 2),
                    "side":    p.side.value if hasattr(p.side, "value") else str(p.side),
                }
                for p in positions
            ],
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "portfolio_value": 0, "cash": 0, "buying_power": 0,
            "pnl_today": 0, "mode": "PAPER" if IS_PAPER else "LIVE",
            "positions": [],
        })


@app.get("/api/stats")
def stats(days: int = 30) -> JSONResponse:
    return JSONResponse(get_dashboard_stats(days))


@app.get("/api/trades")
def trades(days: int = 30) -> JSONResponse:
    rows = get_trades_for_dashboard(days)
    clean = [{k: ("" if v is None else v) for k, v in row.items()} for row in rows]
    return JSONResponse(clean)


@app.get("/api/positions")
def positions_history(days: int = 7) -> JSONResponse:
    """Return all non-SKIP trades (open and closed) for the positions view."""
    rows = _get_executed_trades(days)
    trades = []
    for r in rows:
        trades.append({
            "id": r.get("id"),
            "ticker": r.get("ticker", ""),
            "direction": r.get("direction", ""),
            "entry_price": r.get("entry_price", 0),
            "stop_loss": r.get("stop_loss", 0),
            "take_profit": r.get("take_profit", 0),
            "shares": r.get("shares", 0),
            "signal_score": round(r.get("signal_score", 0), 3),
            "confidence_tier": r.get("confidence_tier", ""),
            "risk_grade": r.get("risk_grade", ""),
            "order_status": r.get("order_status", ""),
            "closed": bool(r.get("closed", 0)),
            "pnl_dollars": round(r.get("pnl_dollars", 0) or 0, 2),
            "pnl_r": round(r.get("pnl_r", 0) or 0, 3),
            "outcome": r.get("outcome", ""),
            "exit_reason": r.get("exit_reason", ""),
            "hold_minutes": round(r.get("hold_minutes", 0) or 0, 1),
            "timestamp": r.get("timestamp", ""),
            "closed_at": r.get("closed_at", ""),
        })
    return JSONResponse(trades)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=False)
