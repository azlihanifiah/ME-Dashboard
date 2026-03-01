"""Keep a Streamlit Community Cloud app warm by periodically pinging it.

Why this exists:
- Streamlit Community Cloud apps go to sleep after inactivity.
- You cannot *fully* prevent sleeping from inside the Streamlit app itself.
- The practical workaround is an external ping (cron/GitHub Actions/UptimeRobot).

Usage:
  python scripts/keep_streamlit_awake.py --url https://your-app.streamlit.app --interval-seconds 600

Or via env var:
  set STREAMLIT_APP_URL=https://your-app.streamlit.app
  python scripts/keep_streamlit_awake.py

Notes:
- Pinging too frequently may be unnecessary; 10-15 minutes is usually enough.
- This keeps the app warm; it does not guarantee 0 cold starts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def _ping(url: str, timeout_seconds: int = 30) -> tuple[int | None, str]:
    req = Request(
        url,
        headers={
            "User-Agent": "keep-streamlit-awake/1.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        method="GET",
    )

    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None)
            return status, "OK"
    except HTTPError as e:
        return getattr(e, "code", None), f"HTTPError: {e}"
    except URLError as e:
        return None, f"URLError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("STREAMLIT_APP_URL", "").strip(),
        help="Streamlit app URL, e.g. https://your-app.streamlit.app",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.environ.get("KEEPALIVE_INTERVAL_SECONDS", "900")),
        help="Ping interval in seconds (default: 900).",
    )
    args = parser.parse_args(argv)

    url = (args.url or "").strip().rstrip("/")
    if not url:
        print("Missing --url (or STREAMLIT_APP_URL).", file=sys.stderr)
        return 2

    interval = max(60, int(args.interval_seconds))
    ping_url = f"{url}/?keepalive=1"

    print(f"Pinging {ping_url} every {interval}s. Ctrl+C to stop.")
    while True:
        status, msg = _ping(ping_url)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_str = str(status) if status is not None else "(no status)"
        print(f"[{ts}] {status_str} {msg}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
