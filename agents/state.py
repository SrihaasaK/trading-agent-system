"""
agents/state.py
Shared LangGraph state definition — the single object passed between all agents.
Every agent reads from and writes to this object.
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class TradeSignal(TypedDict):
    ticker: str
    direction: str          # LONG | SHORT | NONE
    confidence: float       # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int      # number of shares
    reasoning: str


class AgentVerdict(TypedDict):
    agent: str
    score: float            # 0.0 - 1.0
    signal: str             # BULLISH | BEARISH | NEUTRAL | VETO
    summary: str
    raw_data: Dict[str, Any]


class TradingState(TypedDict):
    # Input
    ticker: str
    scan_timestamp: str

    # Market scanner output
    price_data: Dict[str, Any]
    technical_setup: Dict[str, Any]
    session_valid: bool

    # News researcher output
    news_summary: str
    news_sentiment: str     # BULLISH | BEARISH | NEUTRAL | UNCLEAR
    news_veto: bool         # True = hard stop, don't trade
    news_veto_reason: str

    # Macro context output
    macro_context: Dict[str, Any]
    macro_regime: str       # TRENDING_BULL | TRENDING_BEAR | RANGING | HIGH_VOL
    macro_veto: bool

    # Risk analyst output
    risk_assessment: Dict[str, Any]
    risk_grade: str         # GREEN | YELLOW | RED
    position_size: int
    stop_level: float
    target_level: float
    confidence_tier: str    # SWING | STANDARD | SCALP

    # Signal judge output
    agent_verdicts: List[AgentVerdict]
    signal_score: float
    final_direction: str    # LONG | SHORT | NONE
    confidence_gate_passed: bool
    decision_reasoning: str
    decision_key_risk: str

    # Trade executor output
    order_id: str
    order_status: str
    approval_status: str
    execution_price: float

    # Post-mortem (filled after trade closes)
    closed: bool
    close_price: float
    pnl_dollars: float
    pnl_r: float            # result in R-multiples
    outcome: str            # WIN | LOSS | BREAKEVEN

    # System
    errors: List[str]
    skip_reason: str


def initial_state(ticker: str) -> TradingState:
    """Return a clean state object for a new scan cycle."""
    return TradingState(
        ticker=ticker,
        scan_timestamp=datetime.now().isoformat(),
        price_data={},
        technical_setup={},
        session_valid=False,
        news_summary="",
        news_sentiment="NEUTRAL",
        news_veto=False,
        news_veto_reason="",
        macro_context={},
        macro_regime="RANGING",
        macro_veto=False,
        risk_assessment={},
        risk_grade="GREEN",
        position_size=0,
        stop_level=0.0,
        target_level=0.0,
        confidence_tier="STANDARD",
        agent_verdicts=[],
        signal_score=0.0,
        final_direction="NONE",
        confidence_gate_passed=False,
        decision_reasoning="",
        decision_key_risk="",
        order_id="",
        order_status="",
        approval_status="",
        execution_price=0.0,
        closed=False,
        close_price=0.0,
        pnl_dollars=0.0,
        pnl_r=0.0,
        outcome="",
        errors=[],
        skip_reason="",
    )
