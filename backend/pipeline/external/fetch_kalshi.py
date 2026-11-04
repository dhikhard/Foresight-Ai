"""Kalshi historical data fetcher.

Requires:
- ``cryptography`` package (pip install cryptography) for RSA-PSS-SHA256 signing.
- ``KALSHI_API_KEY_ID`` and ``KALSHI_PRIVATE_KEY_PATH`` environment variables.

If ``cryptography`` is not installed, functions raise ``ImportError`` with
a clear message rather than silently degrading.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import (
    SnapshotWriter,
    TransportFn,
    default_transport,
    http_get_json,
)
from .external_config import (
    ENV_KALSHI_API_KEY_ID,
    ENV_KALSHI_PRIVATE_KEY_PATH,
    KALSHI_BASE_URL,
    RATE_LIMITS,
)

SOURCE = "kalshi"

# ---------------------------------------------------------------------------
# RSA-PSS-SHA256 signing (requires cryptography)
# ---------------------------------------------------------------------------


def _require_crypto():
    """Import and return cryptography primitives or raise ImportError."""
    try:
        from cryptography.hazmat.primitives import hashes as _h  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import padding as _p  # noqa: F401
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
        )
        return load_pem_private_key
    except ImportError as exc:
        raise ImportError(
            "The 'cryptography' package is required for Kalshi RSA-PSS-SHA256 "
            "request signing.  Install it with:  pip install cryptography"
        ) from exc


def load_private_key(path: Path):
    """Load a PEM private key from disk."""
    load_pem_private_key = _require_crypto()
    pem = path.read_bytes()
    return load_pem_private_key(pem, password=None)


def sign_request(
    private_key: Any,
    timestamp_ms: int,
    method: str,
    path: str,
) -> str:
    """Sign ``timestamp_ms + method + path`` with RSA-PSS-SHA256, return base64."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    message = f"{timestamp_ms}{method}{path}".encode()
    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


# ---------------------------------------------------------------------------
# Authenticated request helper
# ---------------------------------------------------------------------------

_BASE_URL_CACHE: dict[str, str] = {}


def _get_auth_headers() -> tuple[str, str, Path]:
    """Return (key_id, private_key_path, ...) from environment."""
    key_id = os.environ.get(ENV_KALSHI_API_KEY_ID, "")
    kp = os.environ.get(ENV_KALSHI_PRIVATE_KEY_PATH, "")
    if not key_id or not kp:
        raise EnvironmentError(
            f"Kalshi signing requires environment variables "
            f"{ENV_KALSHI_API_KEY_ID} and {ENV_KALSHI_PRIVATE_KEY_PATH}."
        )
    return key_id, Path(kp)


def kalshi_authed_get(
    subpath: str,
    *,
    params: dict[str, Any] | None = None,
    base_url: str | None = None,
    transport: TransportFn | None = None,
) -> tuple[int, dict]:
    """Make an authenticated GET request to the Kalshi API.

    Returns (status, parsed_json).
    """
    base = base_url or KALSHI_BASE_URL
    key_id, key_path = _get_auth_headers()
    private_key = load_private_key(key_path)

    # Build URL
    qs = ""
    if params:
        qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{base}{subpath}{qs}"

    # Sign path (without query params, per docs)
    timestamp_ms = int(time.time() * 1000)
    sig = sign_request(private_key, timestamp_ms, "GET", subpath)

    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "User-Agent": "ForeSightAI-Research/1.0",
    }

    # Manual GET with custom headers
    req = urllib.request.Request(full_url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=30)
    return resp.status, json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Public read helpers (may or may not work without auth — probe at runtime)
# ---------------------------------------------------------------------------


def fetch_markets(
    *,
    status: str = "settled",
    limit: int = 10,
    cursor: str | None = None,
    transport: TransportFn | None = None,
) -> tuple[int, dict]:
    """GET /markets with optional status filter."""
    url = f"{KALSHI_BASE_URL}/markets?status={status}&limit={limit}"
    if cursor:
        url += f"&cursor={cursor}"
    return http_get_json(url, transport=transport)


def fetch_market(ticker: str, *, transport: TransportFn | None = None) -> tuple[int, dict]:
    """GET /markets/{ticker}."""
    url = f"{KALSHI_BASE_URL}/markets/{ticker}"
    return http_get_json(url, transport=transport)


def fetch_historical_candles(
    ticker: str,
    start_ts: int,
    end_ts: int,
    *,
    period_interval: int = 1440,
    include_latest_before_start: bool = False,
    transport: TransportFn | None = None,
) -> tuple[int, dict]:
    """GET /historical/markets/{ticker}/candlesticks.

    Returns candles whose ``end_period_ts`` satisfies
    ``start_ts <= end_period_ts <= end_ts``.
    ``include_latest_before_start`` controls the synthetic forward-projected
    candle — callers should keep it ``False`` for point-in-time correctness.
    """
    params = {
        "period_interval": period_interval,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "include_latest_before_start": str(include_latest_before_start).lower(),
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{KALSHI_BASE_URL}/historical/markets/{ticker}/candlesticks?{qs}"
    return http_get_json(url, transport=transport)


# ---------------------------------------------------------------------------
# Snapshot writers
# ---------------------------------------------------------------------------


def snapshot_markets(
    writer: SnapshotWriter,
    raw: dict,
    *,
    slug: str = "settled-markets",
    request_params: dict | None = None,
    http_status: int = 200,
) -> Path:
    """Write a markets-list snapshot and return its directory."""
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "markets.json", raw)
    markets = raw.get("markets", [])
    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=f"{KALSHI_BASE_URL}/markets",
        request_params=request_params or {},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"markets": len(markets)},
    )
    return snap


def snapshot_candles(
    writer: SnapshotWriter,
    raw: dict,
    *,
    ticker: str,
    request_params: dict | None = None,
    http_status: int = 200,
) -> Path:
    """Write a candlesticks snapshot and return its directory."""
    slug = f"candles-{ticker.replace(' ', '-')[:60]}"
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "candles.json", raw)
    candles = raw.get("candlesticks", [])
    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=f"{KALSHI_BASE_URL}/historical/markets/{ticker}/candlesticks",
        request_params=request_params or {},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"candlesticks": len(candles)},
    )
    return snap


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe(
    *,
    transport: TransportFn | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Small probe: attempt to list 2 markets (keyless first, then authed if available).

    Returns a dict with probe results for reporting.  This function does NOT
    write snapshots — it only checks reachability and auth behaviour.
    """
    base = base_url or KALSHI_BASE_URL
    result: dict[str, Any] = {
        "base_url": base,
        "auth_required_for_reads": None,
        "markets_listed": None,
        "error": None,
    }

    # Attempt 1: keyless GET /markets?limit=2
    try:
        url = f"{base}/markets?limit=2&status=settled"
        status, body = http_get_json(url, transport=transport)
        result["markets_listed"] = body.get("markets", [])[:2]
        result["auth_required_for_reads"] = False
        return result
    except Exception as exc:
        keyless_err = str(exc)

    # Attempt 2: try authenticated request if env vars are set
    key_id = os.environ.get(ENV_KALSHI_API_KEY_ID, "")
    kp = os.environ.get(ENV_KALSHI_PRIVATE_KEY_PATH, "")
    if key_id and kp:
        try:
            status, body = kalshi_authed_get(
                "/markets",
                params={"limit": "2", "status": "settled"},
                base_url=base,
                transport=transport,
            )
            result["markets_listed"] = body.get("markets", [])[:2]
            result["auth_required_for_reads"] = True
            return result
        except Exception as exc:
            result["error"] = f"Authenticated probe failed: {exc}"
            result["auth_required_for_reads"] = True
            return result

    result["error"] = (
        f"Keyless probe failed ({keyless_err}); "
        f"no {ENV_KALSHI_API_KEY_ID}/{ENV_KALSHI_PRIVATE_KEY_PATH} set for authenticated probe."
    )
    result["auth_required_for_reads"] = None
    return result
