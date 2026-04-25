"""
agents/signal_judge.py
Deterministic weighted signal scoring and trade gate evaluation.
"""

from __future__ import annotations

from loguru import logger

from config.settings import CONFIDENCE_TIERS, HIGH_CONFIDENCE_SCORE, MIN_CONFIDENCE_SCORE, SIGNAL_WEIGHTS
from agents.market_scanner import normalize_direction
from agents.risk_analyst import calculate_position_size
from agents.state import AgentVerdict, TradingState


def _final_tier(score: float) -> tuple[str, dict]:
    """Map final weighted score to authoritative confidence tier."""
    if score >= HIGH_CONFIDENCE_SCORE:
        return "SWING", CONFIDENCE_TIERS["SWING"]
    if score >= CONFIDENCE_TIERS["STANDARD"]["min_score"]:
        return "STANDARD", CONFIDENCE_TIERS["STANDARD"]
    return "SCALP", CONFIDENCE_TIERS["SCALP"]


def _recompute_levels(state: TradingState, tier: dict) -> tuple[float, float, int]:
    """Recompute stop/target/size for the authoritative tier.

    Falls back to existing state values if price or ATR are unavailable.
    """
    price_data = state.get("price_data", {})
    price = float(price_data.get("current_price", 0) or 0)
    atr = float(price_data.get("atr", 0) or 0)

    if price <= 0 or atr <= 0:
        return state["stop_level"], state["target_level"], state["position_size"]

    direction = state.get("technical_setup", {}).get("direction", "LONG")
    atr_mult = tier["atr_stop_mult"]
    rr = tier["reward_risk"]

    if direction == "LONG":
        stop = price - (atr * atr_mult)
        target = price + (atr * atr_mult * rr)
    else:
        stop = price + (atr * atr_mult)
        target = price - (atr * atr_mult * rr)

    portfolio = state.get("risk_assessment", {}).get("portfolio", {})
    macro = state.get("macro_context", {})
    size_multiplier = float(macro.get("size_multiplier", 1.0) or 1.0) * tier["size_factor"]

    if portfolio.get("portfolio_value", 0) > 0:
        sizing = calculate_position_size(
            portfolio["portfolio_value"],
            portfolio.get("cash", 0),
            portfolio.get("buying_power", 0),
            price,
            stop,
            direction,
            size_multiplier,
        )
        shares = sizing["shares"]
    else:
        shares = state["position_size"]

    return round(stop, 4), round(target, 4), shares


def score_technical(state: TradingState) -> AgentVerdict:
    setup = state.get("technical_setup", {})

    if not setup.get("valid"):
        return AgentVerdict(
            agent="market_scanner",
            score=0.0,
            signal="NEUTRAL",
            summary="No valid setup detected",
            raw_data=setup,
        )

    scanner_confidence = float(setup.get("confidence", 0.5))
    direction = normalize_direction(setup.get("direction", "NONE"))
    signal = "BULLISH" if direction == "LONG" else "BEARISH" if direction == "SHORT" else "NEUTRAL"

    adx = setup.get("adx", {}).get("adx", 0)
    rsi = setup.get("rsi", {}).get("rsi", 50)
    rvol = setup.get("rvol", 1.0)
    smc = setup.get("smc_count", 0)
    mom = setup.get("momentum_count", 0)
    htf = setup.get("htf_levels", {})

    summary = (
        f"{direction} | ADX={adx:.1f} RSI={rsi:.1f} RVOL={rvol:.2f}x "
        f"SMC={smc}/3 MOM={mom}/2"
        + (" | discount zone" if htf.get("in_discount") and direction == "LONG" else "")
        + (" | premium zone" if htf.get("in_premium") and direction == "SHORT" else "")
        + (" | OTE zone" if htf.get("in_ote") else "")
    )

    return AgentVerdict(
        agent="market_scanner",
        score=round(scanner_confidence, 3),
        signal=signal,
        summary=summary,
        raw_data={
            "direction": direction,
            "adx": adx,
            "rsi": rsi,
            "rvol": rvol,
            "smc_count": smc,
            "momentum_count": mom,
            "htf": htf,
        },
    )


def score_news(state: TradingState) -> AgentVerdict:
    direction = normalize_direction(state.get("technical_setup", {}).get("direction", "NONE"))
    sentiment = state.get("news_sentiment", "NEUTRAL")

    if state.get("news_veto"):
        return AgentVerdict(
            agent="news_researcher",
            score=0.0,
            signal="VETO",
            summary=state.get("news_veto_reason", "news veto"),
            raw_data={"veto": True},
        )

    aligned_scores = {
        "LONG": {"BULLISH": 0.85, "NEUTRAL": 0.55, "UNCLEAR": 0.45, "BEARISH": 0.15},
        "SHORT": {"BEARISH": 0.85, "NEUTRAL": 0.55, "UNCLEAR": 0.45, "BULLISH": 0.15},
    }
    score = aligned_scores.get(direction, {}).get(sentiment, 0.5)

    return AgentVerdict(
        agent="news_researcher",
        score=round(score, 3),
        signal=sentiment,
        summary=state.get("news_summary", "No news data"),
        raw_data={"sentiment": sentiment},
    )


def score_macro(state: TradingState) -> AgentVerdict:
    direction = normalize_direction(state.get("technical_setup", {}).get("direction", "NONE"))
    regime = state.get("macro_regime", "RANGING")
    macro = state.get("macro_context", {})
    vix = float(macro.get("vix", 20) or 20)
    breadth_score = float(macro.get("breadth_score", 0.5) or 0.5)

    if state.get("macro_veto"):
        return AgentVerdict(
            agent="macro_context",
            score=0.0,
            signal="VETO",
            summary=f"Macro veto — VIX={vix:.1f}",
            raw_data=macro,
        )

    regime_scores = {
        "LONG": {"TRENDING_BULL": 0.82, "RANGING": 0.55, "TRENDING_BEAR": 0.25, "HIGH_VOL": 0.15},
        "SHORT": {"TRENDING_BEAR": 0.82, "RANGING": 0.55, "TRENDING_BULL": 0.25, "HIGH_VOL": 0.15},
    }
    score = regime_scores.get(direction, {}).get(regime, 0.5)

    if direction == "LONG" and breadth_score >= 0.66:
        score = min(score + 0.03, 1.0)
    if direction == "SHORT" and breadth_score <= 0.34:
        score = min(score + 0.03, 1.0)

    signal = "BULLISH" if regime in ("TRENDING_BULL", "RANGING") else "BEARISH"
    return AgentVerdict(
        agent="macro_context",
        score=round(score, 3),
        signal=signal,
        summary=f"Regime={regime} VIX={vix:.1f} breadth={breadth_score:.2f}",
        raw_data=macro,
    )


def score_risk(state: TradingState) -> AgentVerdict:
    grade = state.get("risk_grade", "GREEN")
    assess = state.get("risk_assessment", {})
    rr = float(assess.get("reward_risk_ratio", 0) or 0)

    grade_map = {"GREEN": 0.88, "YELLOW": 0.62, "RED": 0.0}
    score = grade_map.get(grade, 0.5)
    if rr >= 3.0:
        score = min(score + 0.04, 1.0)
    elif rr >= 2.5:
        score = min(score + 0.02, 1.0)

    signal = "GREEN" if grade == "GREEN" else ("YELLOW" if grade == "YELLOW" else "VETO")
    return AgentVerdict(
        agent="risk_analyst",
        score=round(score, 3),
        signal=signal,
        summary=f"Grade={grade} R:R={rr:.2f} size={state.get('position_size', 0)}sh",
        raw_data=assess,
    )


def calculate_weighted_score(verdicts: list[AgentVerdict]) -> float:
    weight_map = {
        "market_scanner": SIGNAL_WEIGHTS["technical"],
        "news_researcher": SIGNAL_WEIGHTS["news"],
        "macro_context": SIGNAL_WEIGHTS["macro"],
        "risk_analyst": SIGNAL_WEIGHTS["risk"],
    }
    return round(sum(v["score"] * weight_map.get(v["agent"], 0.25) for v in verdicts), 4)


def build_reasoning(state: TradingState, verdicts: list[AgentVerdict], weighted_score: float) -> tuple[str, str]:
    direction = normalize_direction(state.get("technical_setup", {}).get("direction", "NONE"))
    strong_support = [v["summary"] for v in verdicts if v["score"] >= 0.75]
    risk_warnings = state.get("risk_assessment", {}).get("warnings", [])

    if strong_support:
        thesis = f"{direction} setup with strong confluence: " + "; ".join(strong_support[:2])
    else:
        thesis = f"{direction} setup cleared the weighted gate with score {weighted_score:.3f}"

    key_risk = risk_warnings[0] if risk_warnings else state.get("news_veto_reason") or "Loss of technical momentum"

    if weighted_score >= HIGH_CONFIDENCE_SCORE:
        thesis += ". Confidence is high enough to allow full intended size."
    else:
        thesis += ". Quality is acceptable but not top-tier, so follow normal risk caps closely."

    return thesis, key_risk


def signal_judge_node(state: TradingState) -> TradingState:
    ticker = state["ticker"]
    logger.info(f"[signal_judge] evaluating {ticker}")

    if state.get("news_veto") or state.get("macro_veto"):
        state.update(
            signal_score=0.0,
            final_direction="NONE",
            confidence_gate_passed=False,
            decision_reasoning=state.get("skip_reason", "Macro/news veto"),
            decision_key_risk=state.get("skip_reason", "Macro/news veto"),
        )
        logger.info(f"[signal_judge] {ticker} → VETO")
        return state

    if state.get("risk_grade") == "RED":
        reason = state.get("skip_reason", "Risk engine rejected the setup")
        state.update(
            signal_score=0.0,
            final_direction="NONE",
            confidence_gate_passed=False,
            decision_reasoning=reason,
            decision_key_risk=reason,
        )
        logger.info(f"[signal_judge] {ticker} → RED risk")
        return state

    verdicts = [
        score_technical(state),
        score_news(state),
        score_macro(state),
        score_risk(state),
    ]
    weighted_score = calculate_weighted_score(verdicts)
    direction = normalize_direction(state.get("technical_setup", {}).get("direction", "NONE"))
    reasoning, key_risk = build_reasoning(state, verdicts, weighted_score)

    state["signal_score"] = weighted_score
    state["agent_verdicts"] = verdicts
    state["decision_reasoning"] = reasoning
    state["decision_key_risk"] = key_risk

    if weighted_score < MIN_CONFIDENCE_SCORE:
        state["final_direction"] = "NONE"
        state["confidence_gate_passed"] = False
        state["skip_reason"] = f"Score {weighted_score:.3f} below threshold {MIN_CONFIDENCE_SCORE}"
        logger.info(f"[signal_judge] {ticker} → BELOW THRESHOLD {weighted_score:.3f}")
        return state

    state["final_direction"] = direction
    state["confidence_gate_passed"] = direction in ("LONG", "SHORT")

    # Final authoritative tier assignment based on weighted score
    final_tier_name, final_tier = _final_tier(weighted_score)
    if final_tier_name != state.get("confidence_tier", "STANDARD"):
        new_stop, new_target, new_shares = _recompute_levels(state, final_tier)
        state["stop_level"] = new_stop
        state["target_level"] = new_target
        state["position_size"] = new_shares
        risk_assess = dict(state.get("risk_assessment", {}))
        risk_assess["stop_loss"] = new_stop
        risk_assess["take_profit"] = new_target
        risk_assess["shares"] = new_shares
        price = float(state.get("price_data", {}).get("current_price", 0) or 0)
        if price > 0 and abs(new_stop - price) > 0:
            risk_assess["reward_risk_ratio"] = round(abs(new_target - price) / abs(new_stop - price), 2)
        state["risk_assessment"] = risk_assess
        logger.info(f"[signal_judge] {ticker} tier upgraded {state.get('confidence_tier')} → {final_tier_name}")
    state["confidence_tier"] = final_tier_name

    logger.info(
        f"[signal_judge] {ticker} → dir={state['final_direction']} "
        f"score={state['signal_score']:.3f} tier={state['confidence_tier']} "
        f"gate={state['confidence_gate_passed']} — {reasoning[:70]}"
    )
    return state
