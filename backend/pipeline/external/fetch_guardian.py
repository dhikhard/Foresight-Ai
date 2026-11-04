"""Guardian Open Platform fetcher.

Historical depth: articles back to **1999**.
Requires ``GUARDIAN_API_KEY`` environment variable.
Free tier: 500 calls/day.
"""
from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Any

from . import (
    SnapshotWriter,
    TransportFn,
    http_get_json,
)
from .external_config import ENV_GUARDIAN_API_KEY, GUARDIAN_BASE

SOURCE = "guardian"


def _search_url(
    query: str,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
    page: int = 1,
    page_size: int = 50,
    api_key: str,
) -> str:
    params: dict[str, str] = {
        "q": query,
        "api-key": api_key,
        "page": str(page),
        "page-size": str(min(page_size, 50)),
        "show-fields": "bodyText",
    }
    if from_date:
        params["from-date"] = from_date
    if to_date:
        params["to-date"] = to_date
    return f"{GUARDIAN_BASE}/search?{urllib.parse.urlencode(params)}"


def fetch_search(
    query: str,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
    page: int = 1,
    page_size: int = 50,
    transport: TransportFn | None = None,
) -> tuple[int, dict]:
    """Search Guardian articles. Requires API key."""
    api_key = os.environ.get(ENV_GUARDIAN_API_KEY, "")
    if not api_key:
        raise EnvironmentError(
            f"Guardian search requires environment variable {ENV_GUARDIAN_API_KEY}."
        )
    url = _search_url(
        query,
        from_date=from_date,
        to_date=to_date,
        page=page,
        page_size=page_size,
        api_key=api_key,
    )
    return http_get_json(url, transport=transport)


# ---------------------------------------------------------------------------
# Snapshot writers
# ---------------------------------------------------------------------------


def snapshot_search(
    writer: SnapshotWriter,
    raw: dict,
    *,
    slug: str = "search-results",
    request_params: dict | None = None,
    http_status: int = 200,
) -> Path:
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "search.json", raw)
    results = raw.get("response", {}).get("results", [])
    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=f"{GUARDIAN_BASE}/search",
        request_params=request_params or {},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"results": len(results)},
    )
    return snap


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe(*, transport: TransportFn | None = None) -> dict[str, Any]:
    """Probe: attempt a search if API key is available."""
    result: dict[str, Any] = {
        "api_key_present": False,
        "reachable": None,
        "results_returned": None,
        "sample_fields": None,
        "error": None,
    }
    api_key = os.environ.get(ENV_GUARDIAN_API_KEY, "")
    if not api_key:
        result["error"] = f"{ENV_GUARDIAN_API_KEY} not set; skipping probe."
        return result
    result["api_key_present"] = True
    try:
        status, body = fetch_search(
            "S&P 500", page_size=2, transport=transport
        )
        result["reachable"] = True
        results = body.get("response", {}).get("results", [])
        result["results_returned"] = len(results)
        if results:
            result["sample_fields"] = sorted(results[0].keys())
    except Exception as exc:
        result["error"] = str(exc)
    return result
