"""Launch the ForeSightAI Quantitative Finance Dashboard.

Usage:
    python run_ui.py
    python run_ui.py --port 8000 --no-browser
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ForeSightAI UI Dashboard Runner"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the web browser automatically",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reloading for development",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("  FORESIGHT.AI — Quantitative Forecasting & Risk Decision Support")
    print(f"  Dashboard Live URL: {url}")
    print("  Pages: Market Overview | Forecasting | Performance | Risk")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 72)

    if not args.no_browser:
        def _open() -> None:
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
