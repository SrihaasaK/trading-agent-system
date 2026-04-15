"""
agents/orchestrator.py
LangGraph state machine — the central nervous system of the trading bot.
Defines the graph: which agents run, in what order, and what conditions
trigger skips, vetoes, or execution.
"""

from langgraph.graph import StateGraph, END
from loguru import logger

from agents.state import TradingState, initial_state
from agents.market_scanner import market_scanner_node
from agents.news_researcher import news_researcher_node
from agents.macro_context import macro_context_node
from agents.risk_analyst import risk_analyst_node
from agents.signal_judge import signal_judge_node
from agents.trade_executor import trade_executor_node, send_ntfy


# ── Conditional Edge Functions ────────────────────────────────────────────────

def should_continue_after_scanner(state: TradingState) -> str:
    """After market scan: continue only if a valid setup was detected."""
    if not state.get("session_valid"):
        logger.debug(f"[orchestrator] {state['ticker']} → skip after scanner: {state.get('skip_reason')}")
        return "skip"
    # Setup found — send a signal-detected notification
    setup = state.get("technical_setup", {})
    price_data = state.get("price_data", {})
    ticker = state["ticker"]
    direction = setup.get("direction", "?")
    conf = setup.get("confidence", 0)
    price = price_data.get("current_price", 0)
    adx = setup.get("adx", {}).get("adx", 0)
    rsi = setup.get("rsi", {}).get("rsi", 0)
    rvol = setup.get("rvol", 0)
    smc = setup.get("smc_count", 0)

    send_ntfy(
        f"{ticker} @ ${price:.2f}\n"
        f"Direction: {direction} | Conf: {conf:.0%}\n"
        f"ADX={adx:.0f} RSI={rsi:.0f} RVOL={rvol:.1f}x SMC={smc}/3\n"
        f"Advancing to news/macro/risk checks...",
        title=f"Setup Found: {ticker} {direction}",
        priority="default",
        tags="mag",
    )
    return "continue"


def should_continue_after_news(state: TradingState) -> str:
    """After news research: stop if a hard veto (earnings, legal, etc.)."""
    if state.get("news_veto"):
        ticker = state["ticker"]
        reason = state.get("news_veto_reason", "unknown")
        logger.debug(f"[orchestrator] {ticker} → veto after news: {reason}")
        send_ntfy(
            f"{ticker} setup VETOED by news research\n"
            f"Reason: {reason}\n"
            f"Sentiment: {state.get('news_sentiment', '?')}",
            title=f"News Veto: {ticker}",
            priority="default",
            tags="no_entry",
        )
        return "veto"
    return "continue"


def should_continue_after_macro(state: TradingState) -> str:
    """After macro context: stop if extreme volatility or macro event."""
    if state.get("macro_veto"):
        ticker = state["ticker"]
        macro = state.get("macro_context", {})
        logger.debug(f"[orchestrator] {ticker} → veto after macro")
        send_ntfy(
            f"{ticker} setup VETOED by macro context\n"
            f"Regime: {state.get('macro_regime', '?')}\n"
            f"VIX: {macro.get('vix', '?')}",
            title=f"Macro Veto: {ticker}",
            priority="default",
            tags="no_entry",
        )
        return "veto"
    return "continue"


def should_execute(state: TradingState) -> str:
    """After signal judge: execute if confidence gate passed, else log and skip."""
    if state.get("confidence_gate_passed"):
        return "execute"
    ticker = state["ticker"]
    score = state.get("signal_score", 0)
    reason = state.get("skip_reason", "")
    if score > 0:
        send_ntfy(
            f"{ticker} passed all checks but failed confidence gate\n"
            f"Score: {score:.3f} | Direction: {state.get('final_direction', '?')}\n"
            f"Reason: {reason}\n"
            f"Risk grade: {state.get('risk_grade', '?')}",
            title=f"Below Threshold: {ticker}",
            priority="low",
            tags="chart_with_downwards_trend",
        )
    return "skip"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_trading_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph trading workflow.

    Flow:
    scanner → [skip|→] news → [veto|→] macro → [veto|→] risk → judge → [skip|execute]
    """
    workflow = StateGraph(TradingState)

    # Register all nodes
    workflow.add_node("market_scanner",  market_scanner_node)
    workflow.add_node("news_researcher", news_researcher_node)
    workflow.add_node("macro_context",   macro_context_node)
    workflow.add_node("risk_analyst",    risk_analyst_node)
    workflow.add_node("signal_judge",    signal_judge_node)
    workflow.add_node("trade_executor",  trade_executor_node)

    # Entry point
    workflow.set_entry_point("market_scanner")

    # Conditional edges
    workflow.add_conditional_edges(
        "market_scanner",
        should_continue_after_scanner,
        {"continue": "news_researcher", "skip": "trade_executor"},
    )
    workflow.add_conditional_edges(
        "news_researcher",
        should_continue_after_news,
        {"continue": "macro_context", "veto": "trade_executor"},
    )
    workflow.add_conditional_edges(
        "macro_context",
        should_continue_after_macro,
        {"continue": "risk_analyst", "veto": "trade_executor"},
    )

    # Risk always feeds into signal judge
    workflow.add_edge("risk_analyst", "signal_judge")

    # Signal judge decides execute vs skip
    workflow.add_conditional_edges(
        "signal_judge",
        should_execute,
        {"execute": "trade_executor", "skip": "trade_executor"},
    )

    # Trade executor is always the terminal node
    workflow.add_edge("trade_executor", END)

    return workflow.compile()


# ── Run One Ticker ────────────────────────────────────────────────────────────

def run_ticker(ticker: str, graph=None) -> TradingState:
    """
    Run the full trading pipeline for a single ticker.
    Returns the final state (useful for logging and testing).
    """
    if graph is None:
        graph = build_trading_graph()

    state  = initial_state(ticker)
    result = graph.invoke(state)

    status = "EXECUTED" if result.get("order_id") and result["order_id"] not in ("", "PENDING_APPROVAL") else (
        "PENDING" if result.get("order_id") == "PENDING_APPROVAL" else "SKIPPED"
    )

    logger.info(
        f"[orchestrator] {ticker} → {status} "
        f"score={result.get('signal_score', 0):.3f} "
        f"direction={result.get('final_direction', 'NONE')}"
    )
    return result
