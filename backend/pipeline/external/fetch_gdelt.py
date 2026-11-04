"""GDELT DOC 2.0 article fetcher (public, keyless).

Historical depth: articles back to **Jan 1, 2017**.
15-minute timestamp resolution.  Returns article metadata + URLs (no body text).
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from . import (
    SnapshotWriter,
    TransportFn,
    http_get_json,
)
from .external_config import GDELT_DOC_BASE

SOURCE = "gdelt"


def fetch_doc_api(
    query: str,
    *,
    mode: str = "ArtList",
    format_: str = "json",
    startdatetime: str | None = None,
    enddatetime: str | None = None,
    maxrecords: int = 250,
    transport: TransportFn | None = None,
) -> tuple[int, Any]:
    """GET from the GDELT DOC 2.0 API.

    ``startdatetime``/``enddatetime`` use YYYYMMDDHHMMSS format (UTC).
    ``maxrecords`` is capped at 250 by GDELT per request.
    """
    params: dict[str, str] = {
        "query": query,
        "mode": mode,
        "format": format_,
        "maxrecords": str(min(maxrecords, 250)),
    }
    if startdatetime:
        params["startdatetime"] = startdatetime
    if enddatetime:
        params["enddatetime"] = enddatetime

    qs = urllib.parse.urlencode(params)
    url = f"{GDELT_DOC_BASE}?{qs}"
    status, body = http_get_json(url, transport=transport)
    return status, body


# ---------------------------------------------------------------------------
# Snapshot writers
# ---------------------------------------------------------------------------


def snapshot_articles(
    writer: SnapshotWriter,
    raw: Any,
    *,
    slug: str = "doc-articles",
    request_params: dict | None = None,
    http_status: int = 200,
) -> Path:
    """Write a GDELT ArtList snapshot."""
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "articles.json", raw)

    articles = []
    if isinstance(raw, dict):
        articles = raw.get("articles", [])
    elif isinstance(raw, list):
        articles = raw

    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=GDELT_DOC_BASE,
        request_params=request_params or {},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"articles": len(articles)},
    )
    return snap


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe(*, transport: TransportFn | None = None) -> dict[str, Any]:
    """Small probe: fetch a few recent articles for S&P 500 related terms.

    Returns probe results without writing snapshots.
    """
    result: dict[str, Any] = {
        "api_reachable": None,
        "articles_returned": None,
        "sample_fields": None,
        "error": None,
    }
    try:
        status, body = fetch_doc_api(
            "sp500 OR S&P500 OR 'S&P 500'",
            mode="ArtList",
            maxrecords=3,
            transport=transport,
        )
        result["api_reachable"] = True
        articles = body.get("articles", []) if isinstance(body, dict) else []
        result["articles_returned"] = len(articles)
        if articles:
            result["sample_fields"] = sorted(articles[0].keys())
    except Exception as exc:
        result["error"] = str(exc)
    return result
