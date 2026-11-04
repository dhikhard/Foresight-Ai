"""Configuration constants for the external data layer.

Separated from the frozen ``backend/pipeline/config.py`` to avoid
modifying any existing pipeline file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at backend/pipeline/external/external_config.py, so the
# repository root is parents[3] (external -> pipeline -> backend -> repo root).
# parents[2] resolved to backend/ and silently produced nonexistent paths.
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW_EXTERNAL = REPO_ROOT / "data" / "raw_external"
DATA_PROCESSED_EXTERNAL = REPO_ROOT / "data" / "processed_external"
SP500_CLEAN_CSV = REPO_ROOT / "data" / "processed" / "sp500_clean.csv"

# ---------------------------------------------------------------------------
# Environment-variable names (credentials — never hardcoded)
# ---------------------------------------------------------------------------

ENV_KALSHI_API_KEY_ID = "KALSHI_API_KEY_ID"
ENV_KALSHI_PRIVATE_KEY_PATH = "KALSHI_PRIVATE_KEY_PATH"
ENV_NYTIMES_API_KEY = "NYTIMES_API_KEY"
ENV_GUARDIAN_API_KEY = "GUARDIAN_API_KEY"

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc"
NYT_ARCHIVE_BASE = "https://api.nytimes.com/svc/archive/v1"
GUARDIAN_BASE = "https://content.guardianapis.com"

# ---------------------------------------------------------------------------
# Rate limits (documented; enforced via caller-side throttling)
# ---------------------------------------------------------------------------

RATE_LIMITS: dict[str, int | float] = {
    "kalshi_basic_read_tokens_per_sec": 20,   # 200 budget / 10 cost
    "polymarket_gamma_general_per_10s": 4000,
    "polymarket_clob_prices_history_per_10s": 1000,
    "gdelt_doc_api_calls_per_sec": 1,        # Be gentle
    "nyt_archive_per_day": 2000,
    "guardian_per_day": 500,
}

# ---------------------------------------------------------------------------
# Deduplication key prefixes (used by normalize module)
# ---------------------------------------------------------------------------

NEWS_DEDUP_KEYS = ("url", "published_at_utc")
KALSHI_DEDUP_KEYS = ("ticker", "candle_end_unix_s", "interval_minutes")
POLYMARKET_DEDUP_KEYS = ("token_id", "t_unix_s")
