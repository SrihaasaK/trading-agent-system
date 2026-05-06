"""
main.py
Entry point for the trading agent system.
Runs the full scan cycle on a schedule during market hours.
Managed by launchd on Mac Mini for automatic start/restart.
"""

import time
import signal
import sys
from datetime import datetime
import pytz
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler

from config.settings import (
    WATCHLIST, LOGS_DIR, LOG_LEVEL,
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
    POSTMORTEM_RUN_AT,
    ENVIRONMENT,
    REQUIRE_TRADE_APPROVAL,
    DB_PATH,
    IS_PAPER,
    EOD_CLOSE_ENABLED,
    EOD_CLOSE_MINUTE,
    HEARTBEAT_FILE,
    TRAILING_STOP_ENABLED,
    TRAILING_STOP_INTERVAL_MIN,
)
from agents.orchestrator import build_trading_graph, run_ticker
from agents.post_mortem import post_mortem_node
from agents.trade_executor import sync_trade_journal, send_ntfy, eod_force_close


# ── Logging Setup ─────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)
logger.add(
    str(LOGS_DIR / "bot_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="zip",
)


# ── Market Hours Check ────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    eastern = pytz.timezone("America/New_York")
    now     = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    open_h, open_m   = MARKET_OPEN_HOUR, MARKET_OPEN_MIN
    close_h, close_m = MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN
    from datetime import time
    return time(open_h, open_m) <= now.time() <= time(close_h, close_m)


# ── Notification Helpers ──────────────────────────────────────────────────────

def _bucket_skip_reason(reason: str) -> str:
    """Collapse verbose skip reasons into short categories."""
    r = reason.lower()
    if "hard gate" in r:
        return "hard gates"
    if "intraday bars" in r:
        return "not enough bars"
    if "directional consensus" in r:
        return "no consensus"
    if "smc" in r:
        return "SMC structure"
    if "momentum" in r:
        return "no momentum"
    if "threshold" in r or "score" in r:
        return "below threshold"
    if "risk" in r or "heat" in r or "position" in r:
        return "risk rejected"
    if "veto" in r:
        return "news/macro veto"
    if "cooldown" in r:
        return "cooldown"
    if "exposure" in r or "duplicate" in r:
        return "duplicate exposure"
    return reason[:30]


def _send_scan_summary(
    time_str: str,
    executed: int,
    pending: int,
    skipped: int,
    errors: int,
    near_misses: list[dict],
    skip_reasons: dict[str, int],
) -> None:
    """Push a concise scan cycle summary via ntfy."""
    total = executed + pending + skipped + errors

    lines = [f"Scanned {total} tickers at {time_str}"]

    if executed:
        lines.append(f"EXECUTED: {executed}")
    if pending:
        lines.append(f"PENDING APPROVAL: {pending}")
    if errors:
        lines.append(f"ERRORS: {errors}")

    if skip_reasons:
        breakdown = ", ".join(f"{v}x {k}" for k, v in sorted(skip_reasons.items(), key=lambda x: -x[1]))
        lines.append(f"Skipped: {breakdown}")

    if near_misses:
        lines.append("")
        lines.append("Closest signals:")
        for nm in sorted(near_misses, key=lambda x: -x["score"])[:3]:
            lines.append(
                f"  {nm['ticker']} {nm['direction']} "
                f"score={nm['score']:.3f} — {nm['reason']}"
            )

    if executed > 0:
        priority = "high"
        tags = "white_check_mark"
        title = f"Scan: {executed} trade(s) placed"
    elif near_misses:
        priority = "default"
        tags = "mag"
        title = f"Scan: {total} tickers, {skipped} skipped"
    else:
        priority = "low"
        tags = "eyes"
        title = f"Scan: {total} tickers — no setups"

    send_ntfy("\n".join(lines), title=title, priority=priority, tags=tags)


def _write_heartbeat() -> None:
    """Write current timestamp to heartbeat file for watchdog monitoring."""
    try:
        HEARTBEAT_FILE.write_text(datetime.now(pytz.timezone("America/New_York")).isoformat())
    except Exception as e:
        logger.warning(f"Failed to write heartbeat file: {e}")


def heartbeat_notify() -> None:
    """Send a periodic status notification with portfolio summary."""
    _write_heartbeat()
    try:
        from agents.risk_analyst import get_portfolio_state
        pf = get_portfolio_state(force_refresh=True)
        val = pf.get("portfolio_value", 0)
        positions = pf.get("open_positions", [])
        total_pl = sum(p.get("unrealized_pl", 0) for p in positions)
        pos_count = len(positions)

        lines = [
            f"Portfolio: ${val:,.0f}",
            f"Positions: {pos_count}/{5}",
            f"Unrealized P&L: ${total_pl:+,.2f}",
            f"Heat: {pf.get('heat_pct', 0)*100:.1f}%",
        ]
        if positions:
            lines.append("")
            for p in sorted(positions, key=lambda x: abs(x.get("unrealized_pl", 0)), reverse=True)[:5]:
                pl = p.get("unrealized_pl", 0)
                lines.append(f"  {p['symbol']}: ${p['market_value']:,.0f} ({'+' if pl >= 0 else ''}{pl:,.2f})")

        send_ntfy(
            "\n".join(lines),
            title="Heartbeat — System Active",
            priority="low",
            tags="heartbeat",
        )
    except Exception as e:
        logger.warning(f"Heartbeat notification failed: {e}")


# ── Scan Cycle ────────────────────────────────────────────────────────────────

# Build graph once and reuse — avoids cold-start on every scan
_graph = None

def scan_cycle():
    """
    Main scan cycle — runs every 15 minutes during market hours.
    Scans every ticker in the watchlist and runs the full agent pipeline.
    """
    global _graph

    if not is_market_hours():
        logger.debug("Outside market hours — skipping scan")
        return

    eastern = pytz.timezone("America/New_York")
    now     = datetime.now(eastern).strftime("%H:%M ET")
    logger.info(f"{'='*50}")
    logger.info(f"SCAN CYCLE starting at {now} — {len(WATCHLIST)} tickers")
    logger.info(f"{'='*50}")

    if _graph is None:
        logger.info("Building trading graph...")
        _graph = build_trading_graph()

    sync_summary = sync_trade_journal()
    logger.info(
        f"Journal sync — checked={sync_summary['checked']} "
        f"updated={sync_summary['updated']} closed={sync_summary['closed']}"
    )

    executed = 0
    pending = 0
    skipped = 0
    errors = 0
    near_misses: list[dict] = []
    skip_reasons: dict[str, int] = {}

    for ticker in WATCHLIST:
        try:
            result = run_ticker(ticker, _graph)
            if result.get("order_status") == "PENDING_APPROVAL":
                pending += 1
            elif result.get("order_id") and result["order_id"] not in ("", "PENDING_APPROVAL"):
                executed += 1
            else:
                skipped += 1
                reason = result.get("skip_reason", "unknown")
                bucket = _bucket_skip_reason(reason)
                skip_reasons[bucket] = skip_reasons.get(bucket, 0) + 1
                if result.get("signal_score", 0) > 0 or result.get("session_valid"):
                    near_misses.append({
                        "ticker": ticker,
                        "score": result.get("signal_score", 0),
                        "direction": result.get("final_direction", "NONE"),
                        "reason": reason[:80],
                    })
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            errors += 1
        time.sleep(1.5)

    logger.info(
        f"Scan complete — executed={executed} pending={pending} "
        f"skipped={skipped} errors={errors}"
    )

    _send_scan_summary(now, executed, pending, skipped, errors, near_misses, skip_reasons)
    _write_heartbeat()


def run_post_mortem():
    """Run the daily post-mortem after market close."""
    logger.info("Running daily post-mortem analysis...")
    try:
        sync_trade_journal()
        result = post_mortem_node()
        logger.info(f"Post-mortem complete: {result.get('status')}")
    except Exception as e:
        logger.error(f"Post-mortem error: {e}")


# ── Graceful Shutdown ─────────────────────────────────────────────────────────

def shutdown(sig, frame):
    logger.info("Shutdown signal received — stopping scheduler")
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── Main Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Trading Agent System starting up...")
    logger.info(f"Environment: {ENVIRONMENT} | Journal: {DB_PATH}")
    if REQUIRE_TRADE_APPROVAL:
        logger.warning("Trade approval mode is ENABLED — signals can be logged, but orders will not auto-execute.")
    logger.info(f"Watching {len(WATCHLIST)} tickers: {', '.join(WATCHLIST[:5])}...")
    logger.info(f"Market hours: {MARKET_OPEN_HOUR}:{MARKET_OPEN_MIN:02d}–{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MIN:02d} ET")

    scheduler = BlockingScheduler(timezone="America/New_York")

    # Scan every 15 minutes during market hours
    scheduler.add_job(
        scan_cycle,
        "cron",
        day_of_week="mon-fri",
        hour    = f"{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR}",
        minute  = "*/15",
        id      = "scan_cycle",
        name    = "Main scan cycle",
    )

    # Daily post-mortem after close
    pm_hour, pm_min = POSTMORTEM_RUN_AT.split(":")
    scheduler.add_job(
        run_post_mortem,
        "cron",
        day_of_week = "mon-fri",
        hour        = int(pm_hour),
        minute      = int(pm_min),
        id          = "post_mortem",
        name        = "Daily post-mortem",
    )

    # Heartbeat every hour during market hours
    scheduler.add_job(
        heartbeat_notify,
        "cron",
        day_of_week="mon-fri",
        hour=f"{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR}",
        minute="0",
        id="heartbeat",
        name="Hourly heartbeat",
    )

    # EOD force-close at 15:50 ET
    if EOD_CLOSE_ENABLED:
        scheduler.add_job(
            eod_force_close,
            "cron",
            day_of_week="mon-fri",
            hour=15,
            minute=EOD_CLOSE_MINUTE,
            id="eod_close",
            name="EOD force-close",
        )

    # Trailing stop check every N minutes during market hours
    if TRAILING_STOP_ENABLED:
        from agents.trade_executor import check_trailing_stops
        scheduler.add_job(
            check_trailing_stops,
            "cron",
            day_of_week="mon-fri",
            hour=f"{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR}",
            minute=f"*/{TRAILING_STOP_INTERVAL_MIN}",
            id="trailing_stops",
            name="Trailing stop check",
        )

    # Startup notification
    send_ntfy(
        f"Pipeline started at {datetime.now(pytz.timezone('America/New_York')).strftime('%I:%M %p ET')}\n"
        f"Watching {len(WATCHLIST)} tickers: {', '.join(WATCHLIST[:6])}...\n"
        f"Scans every 15 min during market hours\n"
        f"{'PAPER' if IS_PAPER else 'LIVE'} mode",
        title="System Started",
        priority="default",
        tags="rocket",
    )

    # Reconcile journal on every startup — catches externally-closed trades
    logger.info("Running startup journal reconciliation...")
    try:
        sync_summary = sync_trade_journal()
        logger.info(
            f"Startup sync — checked={sync_summary['checked']} "
            f"closed={sync_summary['closed']} updated={sync_summary['updated']}"
        )
    except Exception as e:
        logger.warning(f"Startup sync failed: {e}")

    # Run an immediate scan if we're currently in market hours
    if is_market_hours():
        logger.info("Market is open — running immediate scan")
        scan_cycle()

    logger.info("Scheduler started — waiting for next scan window")
    try:
        scheduler.start()
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        sys.exit(1)
