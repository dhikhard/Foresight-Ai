"""Automated screenshot capture for ForeSightAI documentation."""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Locate browser
EDGE_PATH = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

browser_bin = None
if EDGE_PATH.exists():
    browser_bin = str(EDGE_PATH)
elif CHROME_PATH.exists():
    browser_bin = str(CHROME_PATH)
else:
    print("No Edge or Chrome binary found!")
    sys.exit(1)

PORT = 8099
BASE_URL = f"http://127.0.0.1:{PORT}"

print(f"Starting uvicorn server on {BASE_URL}...")
server_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--port",
        str(PORT),
        "--host",
        "127.0.0.1",
    ],
    cwd=str(PROJECT_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

# Wait for server to start
server_ready = False
for _ in range(30):
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1) as resp:
            if resp.status == 200:
                server_ready = True
                break
    except Exception:
        time.sleep(0.5)

if not server_ready:
    print("Server failed to start in time!")
    server_process.terminate()
    sys.exit(1)

print("Server is ready! Capturing screenshots...")

TABS = [
    ("market_overview.png", f"{BASE_URL}/?tab=market-overview"),
    ("forecasting_engine.png", f"{BASE_URL}/?tab=forecasting-engine"),
    ("model_performance.png", f"{BASE_URL}/?tab=model-performance"),
    ("risk_analysis.png", f"{BASE_URL}/?tab=risk-analysis"),
]

try:
    for filename, url in TABS:
        out_path = SCREENSHOTS_DIR / filename
        cmd = [
            browser_bin,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--virtual-time-budget=4000",
            "--window-size=1600,1050",
            f"--screenshot={out_path}",
            url,
        ]
        print(f"Capturing {filename} from {url}...")
        subprocess.run(cmd, timeout=30, check=True)
        if out_path.exists():
            print(f"  Saved {filename} ({out_path.stat().st_size // 1024} KB)")
        else:
            print(f"  Warning: {filename} was not created!")
finally:
    print("Terminating server...")
    server_process.terminate()
    server_process.wait()

print("All screenshots successfully captured!")
