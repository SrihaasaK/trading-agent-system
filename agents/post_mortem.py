"""
agents/post_mortem.py
Deterministic daily post-mortem analysis with richer breakdowns for regime,
session, score buckets, and agent attribution.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from statistics import mean

from loguru import logger

from config.settings import DB_PATH, LOGS_DIR, POSTMORTEM_MIN_TRADES
from agents.journal import init_db, rowdicts, safe_json_loads, score_bucket


def get_todays_trades() -> list:
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.execute(
            """
            SELECT *
            FROM trades
            WHERE date(timestamp) = date('now', 'localtime')
            ORDER BY timestamp ASC
            """
        )
        return rowdicts(cursor)
    finally:
        conn.close()


def get_all_closed_trades(limit: int = 500) -> list:
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.execute(
            """
            SELECT *
            FROM trades
            WHERE closed = 1
              AND direction != 'SKIP'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return rowdicts(cursor)
    finally:
        conn.close()


def _directional_correctness(signal: str, trade_direction: str, won: bool) -> bool:
    bias_map = {
        "BULLISH": "LONG",
        "BEARISH": "SHORT",
        "GREEN": trade_direction,
        "YELLOW": trade_direction,
    }
    bias = bias_map.get(signal, "NONE")
    if bias == "NONE":
        return not won
    if bias == trade_direction:
        return won
    return not won


def attribute_agent_performance(trades: list) -> dict:
    if not trades:
        return {}

    agent_stats = {
        "market_scanner": {"correct": 0, "wrong": 0, "total": 0},
        "news_researcher": {"correct": 0, "wrong": 0, "total": 0},
        "macro_context": {"correct": 0, "wrong": 0, "total": 0},
        "risk_analyst": {"correct": 0, "wrong": 0, "total": 0},
    }

    for trade in trades:
        verdicts = safe_json_loads(trade.get("agent_verdicts", "[]"))
        if not verdicts:
            continue

        direction = trade.get("direction", "LONG")
        won = trade.get("outcome") == "WIN"
        pnl_r = float(trade.get("pnl_r", 0) or 0)

        for verdict in verdicts:
            agent = verdict.get("agent", "")
            if agent not in agent_stats:
                continue

            agent_stats[agent]["total"] += 1
            if agent == "risk_analyst":
                correct = won or pnl_r >= -1.05
            else:
                correct = _directional_correctness(verdict.get("signal", "NEUTRAL"), direction, won)

            if correct:
                agent_stats[agent]["correct"] += 1
            else:
                agent_stats[agent]["wrong"] += 1

    for agent, stats in agent_stats.items():
        total = stats["total"]
        stats["accuracy"] = round(stats["correct"] / total, 3) if total else 0.0

    return agent_stats


def _performance_rows(trades: list, key: str) -> list[dict]:
    grouped: dict[str, list] = {}
    for trade in trades:
        bucket = trade.get(key, "") or "UNKNOWN"
        grouped.setdefault(bucket, []).append(trade)

    rows = []
    for bucket, bucket_trades in grouped.items():
        wins = [t for t in bucket_trades if t.get("outcome") == "WIN"]
        rows.append(
            {
                "bucket": bucket,
                "count": len(bucket_trades),
                "win_rate": round(len(wins) / len(bucket_trades), 3) if bucket_trades else 0.0,
                "avg_r": round(mean(float(t.get("pnl_r", 0) or 0) for t in bucket_trades), 3),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def regime_performance_matrix(trades: list) -> list[dict]:
    matrix: dict[tuple[str, str], list] = {}
    for trade in trades:
        key = (trade.get("macro_regime", "UNKNOWN"), trade.get("direction", "UNKNOWN"))
        matrix.setdefault(key, []).append(trade)

    rows = []
    for (regime, direction), bucket_trades in matrix.items():
        wins = [t for t in bucket_trades if t.get("outcome") == "WIN"]
        rows.append(
            {
                "regime": regime,
                "direction": direction,
                "count": len(bucket_trades),
                "win_rate": round(len(wins) / len(bucket_trades), 3) if bucket_trades else 0.0,
                "avg_r": round(mean(float(t.get("pnl_r", 0) or 0) for t in bucket_trades), 3),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def cluster_failure_patterns(trades: list) -> list:
    clusters: dict[str, dict] = {}
    for trade in trades:
        if trade.get("outcome") != "LOSS":
            continue

        bucket = score_bucket(float(trade.get("signal_score", 0) or 0))
        key = f"{trade.get('macro_regime', 'UNKNOWN')}|{trade.get('session_label', 'UNKNOWN')}|{bucket}"
        cluster = clusters.setdefault(
            key,
            {"count": 0, "tickers": set(), "avg_pnl_r": [], "pattern": key},
        )
        cluster["count"] += 1
        cluster["tickers"].add(trade.get("ticker", ""))
        cluster["avg_pnl_r"].append(float(trade.get("pnl_r", 0) or 0))

    results = []
    for cluster in clusters.values():
        if cluster["count"] < 3:
            continue
        results.append(
            {
                "pattern": cluster["pattern"],
                "count": cluster["count"],
                "avg_loss_r": round(mean(cluster["avg_pnl_r"]), 3),
                "tickers": sorted(cluster["tickers"])[:5],
            }
        )
    return sorted(results, key=lambda row: row["count"], reverse=True)


def propose_updates(closed: list, sessions: list[dict], score_buckets: list[dict]) -> list[dict]:
    proposals = []
    if len(closed) < POSTMORTEM_MIN_TRADES:
        return proposals

    overall_win_rate = sum(1 for trade in closed if trade.get("outcome") == "WIN") / max(len(closed), 1)
    midday = next((row for row in sessions if row["bucket"] == "MIDDAY"), None)
    if midday and midday["count"] >= 8 and midday["win_rate"] + 0.10 < overall_win_rate:
        proposals.append(
            {
                "parameter": "session filter",
                "current_value": "MIDDAY allowed with score penalty",
                "proposed_value": "Require stronger score or skip MIDDAY",
                "evidence": f"MIDDAY win rate {midday['win_rate']:.1%} vs overall {overall_win_rate:.1%}",
                "requires_human_review": False,
                "sample_size_met": True,
            }
        )

    low_bucket = next((row for row in score_buckets if row["bucket"] == "0.68-0.72"), None)
    high_bucket = next((row for row in score_buckets if row["bucket"] == "0.78+"), None)
    if low_bucket and high_bucket and low_bucket["count"] >= 8 and high_bucket["count"] >= 8:
        if low_bucket["win_rate"] + 0.12 < high_bucket["win_rate"]:
            proposals.append(
                {
                    "parameter": "MIN_CONFIDENCE_SCORE",
                    "current_value": "0.68",
                    "proposed_value": "0.70-0.72",
                    "evidence": (
                        f"Low score bucket win rate {low_bucket['win_rate']:.1%} "
                        f"vs high score bucket {high_bucket['win_rate']:.1%}"
                    ),
                    "requires_human_review": False,
                    "sample_size_met": True,
                }
            )

    return proposals


def build_session_summary(trades: list) -> dict:
    closed = [trade for trade in trades if trade.get("closed") and trade.get("direction") != "SKIP"]
    wins = [trade for trade in closed if trade.get("outcome") == "WIN"]
    losses = [trade for trade in closed if trade.get("outcome") == "LOSS"]
    total_pnl = round(sum(float(trade.get("pnl_dollars", 0) or 0) for trade in closed), 2)
    avg_r = round(mean(float(trade.get("pnl_r", 0) or 0) for trade in closed), 3) if closed else 0.0
    win_rate = round(len(wins) / len(closed), 3) if closed else 0.0

    best_trade = max(closed, key=lambda trade: float(trade.get("pnl_r", 0) or 0), default=None)
    worst_trade = min(closed, key=lambda trade: float(trade.get("pnl_r", 0) or 0), default=None)
    gross_wins = sum(max(float(trade.get("pnl_dollars", 0) or 0), 0) for trade in closed)
    gross_losses = abs(sum(min(float(trade.get("pnl_dollars", 0) or 0), 0) for trade in closed))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0.0

    return {
        "signals_logged": len(trades),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "best_trade": {
            "ticker": best_trade.get("ticker"),
            "pnl_r": round(float(best_trade.get("pnl_r", 0) or 0), 3),
        }
        if best_trade
        else {},
        "worst_trade": {
            "ticker": worst_trade.get("ticker"),
            "pnl_r": round(float(worst_trade.get("pnl_r", 0) or 0), 3),
        }
        if worst_trade
        else {},
    }


def generate_postmortem_report(trades: list, historical_closed: list) -> dict:
    closed = [trade for trade in trades if trade.get("closed") and trade.get("direction") != "SKIP"]
    attribution = attribute_agent_performance(historical_closed)
    regime_rows = regime_performance_matrix(closed)
    session_rows = _performance_rows(closed, "session_label")

    for trade in closed:
        trade["score_bucket"] = score_bucket(float(trade.get("signal_score", 0) or 0))
    score_bucket_rows = _performance_rows(closed, "score_bucket")
    direction_rows = _performance_rows(closed, "direction")
    exit_reason_rows = _performance_rows(closed, "exit_reason")
    clusters = cluster_failure_patterns(historical_closed)
    summary = build_session_summary(trades)
    proposals = propose_updates(historical_closed, session_rows, score_bucket_rows)

    strengths = []
    weaknesses = []
    if regime_rows:
        best_regime = max(regime_rows, key=lambda row: row["avg_r"])
        worst_regime = min(regime_rows, key=lambda row: row["avg_r"])
        strengths.append(
            f"Best regime/direction so far: {best_regime['regime']} {best_regime['direction']} "
            f"({best_regime['avg_r']:.2f}R over {best_regime['count']} trades)"
        )
        weaknesses.append(
            f"Weakest regime/direction so far: {worst_regime['regime']} {worst_regime['direction']} "
            f"({worst_regime['avg_r']:.2f}R over {worst_regime['count']} trades)"
        )
    if session_rows:
        best_session = max(session_rows, key=lambda row: row["avg_r"])
        worst_session = min(session_rows, key=lambda row: row["avg_r"])
        strengths.append(f"Best session: {best_session['bucket']} ({best_session['avg_r']:.2f}R)")
        weaknesses.append(f"Worst session: {worst_session['bucket']} ({worst_session['avg_r']:.2f}R)")

    return {
        "summary": summary,
        "agent_scorecard": attribution,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "regime_performance": regime_rows,
        "session_performance": session_rows,
        "score_bucket_performance": score_bucket_rows,
        "direction_performance": direction_rows,
        "exit_reason_breakdown": exit_reason_rows,
        "failure_clusters": clusters,
        "proposed_updates": proposals,
        "sample_size_for_updates": len(historical_closed),
    }


def render_markdown_report(report: dict) -> str:
    summary = report.get("summary", {})
    lines = [
        f"DAILY POST-MORTEM — {date.today().isoformat()}",
        "",
        "SESSION SUMMARY",
        f"- Signals logged: {summary.get('signals_logged', 0)}",
        f"- Closed trades: {summary.get('closed_trades', 0)}",
        f"- Win/Loss: {summary.get('wins', 0)}/{summary.get('losses', 0)} ({summary.get('win_rate', 0):.1%})",
        f"- Avg R: {summary.get('avg_r', 0):.2f}R",
        f"- Total P&L: ${summary.get('total_pnl', 0):,.2f}",
        f"- Profit factor: {summary.get('profit_factor', 0):.2f}",
        "",
        "AGENT SCORECARD",
    ]
    for agent, stats in report.get("agent_scorecard", {}).items():
        lines.append(f"- {agent}: {stats.get('accuracy', 0):.1%} accuracy ({stats.get('total', 0)} trades)")

    lines.extend(["", "WHAT WORKED"])
    for item in report.get("strengths", []) or ["- No statistically meaningful strengths yet."]:
        lines.append(f"- {item}" if not str(item).startswith("- ") else item)

    lines.extend(["", "WHAT DIDN'T"])
    for item in report.get("weaknesses", []) or ["- No statistically meaningful weaknesses yet."]:
        lines.append(f"- {item}" if not str(item).startswith("- ") else item)

    lines.extend(["", "PROPOSALS"])
    if report.get("proposed_updates"):
        for proposal in report["proposed_updates"]:
            lines.append(
                f"- {proposal['parameter']}: {proposal['current_value']} -> {proposal['proposed_value']} "
                f"({proposal['evidence']})"
            )
    else:
        lines.append("- No parameter changes proposed.")

    return "\n".join(lines) + "\n"


def save_postmortem(report: dict, trades: list) -> dict:
    today = date.today().isoformat()
    json_path = LOGS_DIR / f"postmortem_{today}.json"
    md_path = LOGS_DIR / f"postmortem_{today}.md"

    output = {
        "date": today,
        "generated_at": datetime.now().isoformat(),
        "report": report,
        "trade_count": len(trades),
    }
    json_path.write_text(json.dumps(output, indent=2))
    md_path.write_text(render_markdown_report(report))

    logger.info(f"[post_mortem] report saved to {json_path} and {md_path}")
    return {"json": str(json_path), "markdown": str(md_path)}


def post_mortem_node(state: dict | None = None) -> dict:
    logger.info("[post_mortem] starting daily analysis")

    todays_trades = get_todays_trades()
    historical_closed = get_all_closed_trades()

    if not todays_trades:
        logger.info("[post_mortem] no trades today")
        return {"status": "no trades"}

    report = generate_postmortem_report(todays_trades, historical_closed)
    paths = save_postmortem(report, todays_trades)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            UPDATE trades
            SET postmortem_run = 1
            WHERE date(timestamp) = date('now', 'localtime')
            """
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("[post_mortem] complete")
    return {
        "status": "complete",
        "report": report,
        "filepath": paths["json"],
        "markdown_path": paths["markdown"],
        "trade_count": len(todays_trades),
    }
