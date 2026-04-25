"""
agents/news_researcher.py
Researches news, SEC filings, and Reddit sentiment for a flagged ticker.
Uses Claude Haiku for speed and cost efficiency.
Outputs: sentiment score, brief, and veto flag if earnings/FDA/legal events pending.
"""

import finnhub
import praw
import requests
from datetime import datetime, timedelta
from loguru import logger
import json
import anthropic

from config.settings import (
    FINNHUB_API_KEY, REDDIT_CLIENT_ID, REDDIT_SECRET,
    OLLAMA_ENABLED, OLLAMA_MODEL, OLLAMA_BASE_URL,
    GROQ_API_KEY, GROQ_MODEL,
    REDDIT_USER_AGENT, ANTHROPIC_API_KEY, LLM_FAST,
)
from agents.state import TradingState

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
_news_cache: dict[str, tuple[datetime, dict]] = {}
CACHE_MINUTES = 15


# ── Data Fetchers ─────────────────────────────────────────────────────────────

def fetch_finnhub_news(ticker: str, hours_back: int = 12) -> list:
    """Pull company news from Finnhub for the last N hours."""
    try:
        end   = datetime.now()
        start = end - timedelta(hours=hours_back)
        news  = finnhub_client.company_news(
            ticker,
            _from=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
        )
        # Return headline + summary only (not full body — saves tokens)
        filtered = []
        for item in news or []:
            timestamp = item.get("datetime")
            if timestamp:
                published_at = datetime.fromtimestamp(timestamp)
                if published_at < start:
                    continue
            filtered.append(
                {
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", "")[:300],
                }
            )
            if len(filtered) >= 10:
                break
        return filtered
    except Exception as e:
        logger.warning(f"[news_researcher] finnhub news error: {e}")
        return []


def fetch_reddit_sentiment(ticker: str, limit: int = 20) -> list:
    """Scrape recent Reddit posts mentioning the ticker from r/stocks + r/wallstreetbets."""
    if not REDDIT_CLIENT_ID:
        return []
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        posts = []
        for sub in ["stocks", "wallstreetbets", "investing"]:
            for post in reddit.subreddit(sub).search(ticker, limit=limit // 3, time_filter="day"):
                posts.append(f"{post.title} — score:{post.score}")
        return posts[:15]
    except Exception as e:
        logger.warning(f"[news_researcher] reddit error: {e}")
        return []


def fetch_sec_recent(ticker: str) -> list:
    """Check SEC EDGAR for recent 8-K filings (material events) in last 7 days."""
    try:
        url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
        # Use the EDGAR full-text search API instead
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={(datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')}&enddt={datetime.now().strftime('%Y-%m-%d')}&forms=8-K"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "trading-agent-system contact@example.com"})
        if resp.status_code == 200:
            data  = resp.json()
            hits  = data.get("hits", {}).get("hits", [])
            return [h.get("_source", {}).get("display_names", [""])[0] + ": " +
                    h.get("_source", {}).get("file_date", "") for h in hits[:5]]
    except Exception as e:
        logger.warning(f"[news_researcher] SEC error: {e}")
    return []


def check_earnings_calendar(ticker: str) -> dict:
    """Check if earnings are due within 48 hours — automatic veto if so."""
    try:
        earnings = finnhub_client.earnings_calendar(
            symbol=ticker,
            _from=datetime.now().strftime("%Y-%m-%d"),
            to=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        upcoming = (earnings or {}).get("earningsCalendar", [])
        if upcoming:
            return {"pending": True, "date": upcoming[0].get("date", ""), "eps_estimate": upcoming[0].get("epsEstimate")}
    except Exception as e:
        logger.warning(f"[news_researcher] earnings calendar error: {e}")
    return {"pending": False}


# ── LLM Synthesis ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a concise financial news analyst for an algorithmic trading system.
Your job is to analyze news and sentiment data for a stock and return a structured assessment.
Be direct and data-driven. Never speculate beyond what the data shows.
Always respond with valid JSON only — no markdown, no preamble."""

def _call_llm(prompt: str) -> str:
    """Route to Ollama, Groq, or Claude based on environment config."""
    if OLLAMA_ENABLED:
        from openai import OpenAI
        client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        logger.debug("[news_researcher] using Ollama backend")
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        return resp.choices[0].message.content
    elif GROQ_API_KEY:
        from openai import OpenAI
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        logger.debug("[news_researcher] using Groq backend")
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        return resp.choices[0].message.content
    else:
        logger.debug("[news_researcher] using Claude backend")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=LLM_FAST,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


def synthesize_news(ticker: str, news: list, reddit: list, sec: list) -> dict:
    """Ask Claude Haiku to score the news and flag any veto conditions."""
    cached = _news_cache.get(ticker)
    now = datetime.now()
    if cached and (now - cached[0]).total_seconds() < CACHE_MINUTES * 60:
        return cached[1]

    if not news and not reddit and not sec:
        result = {
            "sentiment":       "NEUTRAL",
            "sentiment_score": 0.5,
            "key_points":      ["No material fresh news detected."],
            "veto":            False,
            "veto_reason":     "",
            "confidence":      0.4,
        }
        _news_cache[ticker] = (now, result)
        return result

    news_text   = "\n".join([f"- {n['headline']}: {n['summary']}" for n in news]) or "No recent news."
    reddit_text = "\n".join(reddit[:10]) or "No Reddit data."
    sec_text    = "\n".join(sec) or "No recent SEC filings."

    prompt = f"""Analyze this data for {ticker} and return JSON with exactly these fields:

NEWS (last 12h):
{news_text}

REDDIT SENTIMENT:
{reddit_text}

SEC FILINGS (last 7 days):
{sec_text}

Return JSON:
{{
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL" | "UNCLEAR",
  "sentiment_score": 0.0-1.0 (0=very bearish, 0.5=neutral, 1=very bullish),
  "key_points": ["point1", "point2", "point3"],
  "veto": true | false,
  "veto_reason": "reason if veto=true, else empty string",
  "confidence": 0.0-1.0 (how confident are you in this assessment)
}}

Set veto=true if: earnings within 48h, FDA decision pending, active legal proceedings, CEO/CFO departure announced, fraud allegations, trading halt risk."""

    try:
        raw    = _call_llm(prompt)
        result = json.loads(raw)
        _news_cache[ticker] = (now, result)
        return result
    except Exception as e:
        logger.error(f"[news_researcher] LLM synthesis error: {e}")
        result = {
            "sentiment":       "UNCLEAR",
            "sentiment_score": 0.5,
            "key_points":      [],
            "veto":            False,
            "veto_reason":     "",
            "confidence":      0.0,
        }
        _news_cache[ticker] = (now, result)
        return result


# ── LangGraph Node ────────────────────────────────────────────────────────────

def news_researcher_node(state: TradingState) -> TradingState:
    """LangGraph node: researches news for the ticker and updates state."""
    ticker = state["ticker"]
    logger.info(f"[news_researcher] researching {ticker}")

    # Skip if technical setup already failed
    if not state.get("session_valid"):
        return state

    news    = fetch_finnhub_news(ticker)
    reddit  = fetch_reddit_sentiment(ticker)
    sec     = fetch_sec_recent(ticker)
    earnings = check_earnings_calendar(ticker)

    # Hard veto on upcoming earnings — no need to call LLM
    if earnings["pending"]:
        state["news_veto"]       = True
        state["news_veto_reason"] = f"Earnings due {earnings['date']} — avoiding binary event"
        state["news_sentiment"]  = "UNCLEAR"
        state["skip_reason"]     = state["news_veto_reason"]
        logger.warning(f"[news_researcher] VETO: {state['news_veto_reason']}")
        return state

    result = synthesize_news(ticker, news, reddit, sec)

    state["news_summary"]   = " | ".join(result.get("key_points", []))
    state["news_sentiment"] = result.get("sentiment", "NEUTRAL")
    state["news_veto"]      = result.get("veto", False)
    state["news_veto_reason"] = result.get("veto_reason", "")

    if state["news_veto"]:
        state["skip_reason"] = state["news_veto_reason"]

    logger.info(f"[news_researcher] {ticker} → sentiment={state['news_sentiment']} veto={state['news_veto']}")
    return state
