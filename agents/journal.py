"""
agents/journal.py
Shared SQLite helpers for trade journaling, schema migrations, and reporting.
Keeps execution, post-mortem, and dashboard logic on the same trade record model.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List

from loguru import logger

from config.settings import DB_PATH, STRATEGY_VERSION


CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT,
    direction         TEXT,
    entry_price       REAL,
    stop_loss         REAL,
    take_profit       REAL,
    shares            INTEGER,
    signal_score      REAL,
    technical_score   REAL DEFAULT 0,
    news_score        REAL DEFAULT 0,
    macro_score       REAL DEFAULT 0,
    risk_score        REAL DEFAULT 0,
    risk_dollars      REAL DEFAULT 0,
    risk_grade        TEXT,
    macro_regime      TEXT,
    vix               REAL DEFAULT 0,
    news_sentiment    TEXT,
    session_label     TEXT DEFAULT '',
    adx               REAL DEFAULT 0,
    rvol              REAL DEFAULT 0,
    atr               REAL DEFAULT 0,
    strategy_version  TEXT DEFAULT '',
    agent_verdicts    TEXT,
    signal_reasoning  TEXT DEFAULT '',
    key_risk          TEXT DEFAULT '',
    order_id          TEXT,
    order_status      TEXT,
    approval_status   TEXT DEFAULT '',
    timestamp         TEXT,
    close_price       REAL DEFAULT 0,
    closed_at         TEXT DEFAULT '',
    pnl_dollars       REAL DEFAULT 0,
    pnl_r             REAL DEFAULT 0,
    outcome           TEXT DEFAULT '',
    exit_reason       TEXT DEFAULT '',
    hold_minutes      REAL DEFAULT 0,
    closed            INTEGER DEFAULT 0,
    postmortem_run    INTEGER DEFAULT 0,
    skip_reason       TEXT DEFAULT '',
    confidence_tier   TEXT DEFAULT 'STANDARD'
)
"""


MIGRATION_COLUMNS = {
    "technical_score": "REAL DEFAULT 0",
    "news_score": "REAL DEFAULT 0",
    "macro_score": "REAL DEFAULT 0",
    "risk_score": "REAL DEFAULT 0",
    "risk_dollars": "REAL DEFAULT 0",
    "vix": "REAL DEFAULT 0",
    "session_label": "TEXT DEFAULT ''",
    "adx": "REAL DEFAULT 0",
    "rvol": "REAL DEFAULT 0",
    "atr": "REAL DEFAULT 0",
    "strategy_version": "TEXT DEFAULT ''",
    "signal_reasoning": "TEXT DEFAULT ''",
    "key_risk": "TEXT DEFAULT ''",
    "approval_status": "TEXT DEFAULT ''",
    "closed_at": "TEXT DEFAULT ''",
    "exit_reason": "TEXT DEFAULT ''",
    "hold_minutes": "REAL DEFAULT 0",
    "confidence_tier": "TEXT DEFAULT 'STANDARD'",
}


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.execute("PRAGMA table_info(trades)")
    return {row[1] for row in cursor.fetchall()}


def init_db() -> None:
    """Create the trades table and backfill any newer columns."""
    conn = get_connection()
    try:
        conn.execute(CREATE_TRADES_SQL)
        existing = _existing_columns(conn)
        for column, sql_type in MIGRATION_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {column} {sql_type}")
        conn.commit()
    finally:
        conn.close()


def rowdicts(cursor: sqlite3.Cursor) -> List[Dict[str, Any]]:
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def score_bucket(score: float) -> str:
    if score < 0.60:
        return "<0.60"
    if score < 0.68:
        return "0.60-0.68"
    if score < 0.72:
        return "0.68-0.72"
    if score < 0.78:
        return "0.72-0.78"
    return "0.78+"


def _verdict_score_map(state: Dict[str, Any]) -> Dict[str, float]:
    verdicts = state.get("agent_verdicts", [])
    return {v.get("agent", ""): float(v.get("score", 0)) for v in verdicts}


def _safe_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def log_trade(state: Dict[str, Any], order_id: str, order_status: str, approval_status: str) -> None:
    """Insert a new trade signal/execution record."""
    init_db()
    verdict_scores = _verdict_score_map(state)
    price_data = state.get("price_data", {})
    risk_assessment = state.get("risk_assessment", {})
    macro_context = state.get("macro_context", {})

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO trades (
                ticker, direction, entry_price, stop_loss, take_profit, shares,
                signal_score, technical_score, news_score, macro_score, risk_score,
                risk_dollars, risk_grade, macro_regime, vix, news_sentiment,
                session_label, adx, rvol, atr, strategy_version,
                agent_verdicts, signal_reasoning, key_risk,
                order_id, order_status, approval_status, timestamp, confidence_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.get("ticker", ""),
                state.get("final_direction", "NONE"),
                float(price_data.get("current_price", 0) or 0),
                float(state.get("stop_level", 0) or 0),
                float(state.get("target_level", 0) or 0),
                int(state.get("position_size", 0) or 0),
                float(state.get("signal_score", 0) or 0),
                float(verdict_scores.get("market_scanner", 0)),
                float(verdict_scores.get("news_researcher", 0)),
                float(verdict_scores.get("macro_context", 0)),
                float(verdict_scores.get("risk_analyst", 0)),
                float(risk_assessment.get("risk_dollars", 0) or 0),
                state.get("risk_grade", ""),
                state.get("macro_regime", ""),
                float(macro_context.get("vix", 0) or 0),
                state.get("news_sentiment", ""),
                price_data.get("session", ""),
                float(price_data.get("adx", 0) or 0),
                float(price_data.get("rvol", 0) or 0),
                float(price_data.get("atr", 0) or 0),
                STRATEGY_VERSION,
                json.dumps(state.get("agent_verdicts", [])),
                state.get("decision_reasoning", ""),
                state.get("decision_key_risk", ""),
                order_id,
                order_status,
                approval_status,
                datetime.now().isoformat(),
                state.get("confidence_tier", "STANDARD"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_skip(state: Dict[str, Any]) -> None:
    """Insert a skipped signal record for later analysis."""
    init_db()
    verdict_scores = _verdict_score_map(state)
    price_data = state.get("price_data", {})
    macro_context = state.get("macro_context", {})

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO trades (
                ticker, direction, entry_price, stop_loss, take_profit, shares,
                signal_score, technical_score, news_score, macro_score, risk_score,
                risk_dollars, risk_grade, macro_regime, vix, news_sentiment,
                session_label, adx, rvol, atr, strategy_version,
                agent_verdicts, signal_reasoning, key_risk,
                order_id, order_status, approval_status, timestamp, skip_reason, closed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.get("ticker", ""),
                "SKIP",
                float(price_data.get("current_price", 0) or 0),
                0.0,
                0.0,
                0,
                float(state.get("signal_score", 0) or 0),
                float(verdict_scores.get("market_scanner", 0)),
                float(verdict_scores.get("news_researcher", 0)),
                float(verdict_scores.get("macro_context", 0)),
                float(verdict_scores.get("risk_analyst", 0)),
                float(state.get("risk_assessment", {}).get("risk_dollars", 0) or 0),
                state.get("risk_grade", ""),
                state.get("macro_regime", ""),
                float(macro_context.get("vix", 0) or 0),
                state.get("news_sentiment", ""),
                price_data.get("session", ""),
                float(price_data.get("adx", 0) or 0),
                float(price_data.get("rvol", 0) or 0),
                float(price_data.get("atr", 0) or 0),
                STRATEGY_VERSION,
                json.dumps(state.get("agent_verdicts", [])),
                state.get("decision_reasoning", ""),
                state.get("decision_key_risk", ""),
                "",
                "SKIPPED",
                "SKIPPED",
                datetime.now().isoformat(),
                state.get("skip_reason", ""),
                1,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_open_trades(include_pending_approval: bool = False) -> List[Dict[str, Any]]:
    """Return journal rows that still represent potentially open market exposure."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT *
            FROM trades
            WHERE direction != 'SKIP'
              AND closed = 0
            ORDER BY timestamp ASC
            """
        )
        rows = rowdicts(cursor)
    finally:
        conn.close()

    if include_pending_approval:
        return rows

    return [
        row for row in rows
        if row.get("order_status") not in ("PENDING_APPROVAL", "CANCELED", "ERROR", "REJECTED")
    ]


def get_recent_signals(ticker: str, within_minutes: int) -> List[Dict[str, Any]]:
    """Return recent signal rows for cooldown checks."""
    init_db()
    cutoff = datetime.now() - timedelta(minutes=within_minutes)
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT *
            FROM trades
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT 25
            """,
            (ticker,),
        )
        rows = rowdicts(cursor)
    finally:
        conn.close()

    return [
        row for row in rows
        if (ts := _safe_timestamp(row.get("timestamp", ""))) and ts >= cutoff
    ]


def update_order_state(order_id: str, order_status: str, close_open_trade: bool = False) -> None:
    """Update the latest order state when Alpaca status changes."""
    if not order_id:
        return

    init_db()
    conn = get_connection()
    try:
        if close_open_trade:
            conn.execute(
                """
                UPDATE trades
                SET order_status = ?, closed = 1, closed_at = COALESCE(NULLIF(closed_at, ''), ?)
                WHERE order_id = ?
                """,
                (order_status, datetime.now().isoformat(), order_id),
            )
        else:
            conn.execute(
                """
                UPDATE trades
                SET order_status = ?
                WHERE order_id = ?
                """,
                (order_status, order_id),
            )
        conn.commit()
    finally:
        conn.close()


def mark_trade_closed(
    order_id: str,
    close_price: float,
    pnl_dollars: float,
    pnl_r: float,
    outcome: str,
    exit_reason: str,
    closed_at: str,
    hold_minutes: float,
    order_status: str = "CLOSED",
) -> None:
    """Mark an open trade as closed with realized performance fields."""
    if not order_id:
        return

    init_db()
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE trades
            SET close_price = ?,
                pnl_dollars = ?,
                pnl_r = ?,
                outcome = ?,
                exit_reason = ?,
                closed_at = ?,
                hold_minutes = ?,
                order_status = ?,
                closed = 1
            WHERE order_id = ?
            """,
            (
                close_price,
                pnl_dollars,
                pnl_r,
                outcome,
                exit_reason,
                closed_at,
                hold_minutes,
                order_status,
                order_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_daily_realized_pnl() -> float:
    """Return the sum of realized P&L from today's closed trades."""
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT COALESCE(SUM(pnl_dollars), 0)
            FROM trades
            WHERE closed = 1
              AND direction != 'SKIP'
              AND closed_at LIKE ?
            """,
            (f"{today}%",),
        )
        return float(cursor.fetchone()[0])
    finally:
        conn.close()


def get_daily_trade_count(ticker: str = None) -> int:
    """Count today's non-SKIP trades, optionally filtered by ticker."""
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        if ticker:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE direction != 'SKIP' AND ticker = ? AND timestamp LIKE ?",
                (ticker, f"{today}%"),
            )
        else:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE direction != 'SKIP' AND timestamp LIKE ?",
                (f"{today}%",),
            )
        return int(cursor.fetchone()[0])
    finally:
        conn.close()


def estimate_open_risk_from_journal() -> float:
    """Approximate open portfolio heat as stop-distance risk from the journal."""
    total = 0.0
    for row in get_open_trades(include_pending_approval=False):
        entry = float(row.get("entry_price", 0) or 0)
        stop = float(row.get("stop_loss", 0) or 0)
        shares = int(row.get("shares", 0) or 0)
        if entry > 0 and stop > 0 and shares > 0:
            total += abs(entry - stop) * shares
    return round(total, 2)


def latest_postmortem_path() -> str:
    """Return the newest saved post-mortem JSON path if available."""
    from config.settings import LOGS_DIR

    files = sorted(LOGS_DIR.glob("postmortem_*.json"))
    return str(files[-1]) if files else ""


def get_trades_for_dashboard(days: int = 30) -> List[Dict[str, Any]]:
    """Return all trade rows within the lookback window for the dashboard."""
    init_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM trades
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 500
            """,
            (cutoff,),
        )
        return rowdicts(cursor)
    finally:
        conn.close()


def _get_closed_trades(days: int = 30) -> List[Dict[str, Any]]:
    """Return all closed non-SKIP trades within the lookback window (no row limit)."""
    init_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM trades
            WHERE direction != 'SKIP' AND closed = 1 AND timestamp >= ?
            ORDER BY closed_at ASC
            """,
            (cutoff,),
        )
        return rowdicts(cursor)
    finally:
        conn.close()


def _get_executed_trades(days: int = 30) -> List[Dict[str, Any]]:
    """Return all non-SKIP trades within the lookback window (no row limit)."""
    init_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM trades
            WHERE direction != 'SKIP' AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
        )
        return rowdicts(cursor)
    finally:
        conn.close()


def get_dashboard_stats(days: int = 30) -> Dict[str, Any]:
    """Compute aggregate performance stats for the dashboard header and charts."""
    closed = _get_closed_trades(days)
    all_executed = _get_executed_trades(days)
    wins   = [r for r in closed if r.get("outcome") == "WIN"]
    losses = [r for r in closed if r.get("outcome") == "LOSS"]

    total_closed  = len(closed)
    win_rate      = len(wins) / total_closed if total_closed else 0.0
    avg_r         = sum(r.get("pnl_r", 0) for r in closed) / total_closed if total_closed else 0.0
    total_pnl     = sum(r.get("pnl_dollars", 0) for r in closed)
    gross_wins    = sum(r.get("pnl_dollars", 0) for r in wins)
    gross_losses  = abs(sum(r.get("pnl_dollars", 0) for r in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

    cumulative = 0.0
    equity_curve = []
    for r in closed:
        if r.get("closed_at"):
            cumulative += r.get("pnl_dollars", 0)
            equity_curve.append({
                "date": r["closed_at"][:10],
                "cumulative_pnl": round(cumulative, 2),
            })

    return {
        "total_signals":     len(get_trades_for_dashboard(days)),
        "executed_trades":   len(all_executed),
        "pending_approvals": len([r for r in all_executed if r.get("approval_status") == "PENDING_APPROVAL"]),
        "closed_trades":     total_closed,
        "win_rate":          round(win_rate, 4),
        "avg_r":             round(avg_r, 3),
        "total_pnl":         round(total_pnl, 2),
        "profit_factor":     round(profit_factor, 2),
        "equity_curve":      equity_curve,
    }


def safe_json_loads(payload: str) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        logger.debug("Failed to decode JSON payload from journal row")
        return []
