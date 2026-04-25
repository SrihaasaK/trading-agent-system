"""
scripts/watchdog.py
Dead-man switch: checks the heartbeat file and sends an urgent ntfy alert
if the trading bot hasn't updated it within HEARTBEAT_STALE_MIN minutes.
Designed to run as a separate launchd job every 10 minutes.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import HEARTBEAT_FILE, HEARTBEAT_STALE_MIN, NTFY_TOPIC


def _is_market_hours() -> bool:
    """Quick check — only alert during market hours."""
    import pytz
    eastern = pytz.timezone("America/New_York")
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    from datetime import time
    return time(9, 30) <= now.time() <= time(16, 0)


def check_heartbeat() -> None:
    if not _is_market_hours():
        return

    if not HEARTBEAT_FILE.exists():
        _send_alert("Heartbeat file missing -- bot may not have started")
        return

    try:
        content = HEARTBEAT_FILE.read_text().strip()
        last_beat = datetime.fromisoformat(content)
    except (ValueError, OSError) as e:
        _send_alert(f"Cannot read heartbeat file: {e}")
        return

    # Make comparison timezone-naive
    now = datetime.now()
    if last_beat.tzinfo is not None:
        last_beat = last_beat.replace(tzinfo=None)

    age_minutes = (now - last_beat).total_seconds() / 60
    if age_minutes > HEARTBEAT_STALE_MIN:
        _send_alert(
            f"Bot heartbeat stale: last update {age_minutes:.0f} min ago\n"
            f"Last beat: {content}\n"
            f"Threshold: {HEARTBEAT_STALE_MIN} min"
        )


def _send_alert(message: str) -> None:
    if not NTFY_TOPIC:
        print(f"WATCHDOG ALERT (no ntfy topic): {message}")
        return
    try:
        import requests
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": "WATCHDOG: Trading Bot Unresponsive",
                "Priority": "urgent",
                "Tags": "rotating_light",
            },
            timeout=5,
        )
    except Exception as e:
        print(f"WATCHDOG: ntfy send failed: {e}")


if __name__ == "__main__":
    check_heartbeat()
