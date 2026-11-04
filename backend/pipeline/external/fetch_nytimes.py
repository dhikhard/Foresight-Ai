"""New York Times Archive API fetcher.

Historical depth: articles back to **1851**.
Requires ``NYTIMES_API_KEY`` environment variable.
2,000 requests/day; up to 1 request per month of data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import (
    SnapshotWriter,
    TransportFn,
    http_get_json,
)
from .external_config import ENV_NYTIMES_API_KEY, NYT_ARCHIVE_BASE

SOURCE = "nytimes"


def _api_url(year: int, month: int, api_key: str) -> str:
    return f"{NYT_ARCHIVE_BASE}/{year}/{month}.json?api-key={api_key}"


def fetch_archive_month(
    year: int, month: int, *, transport: TransportFn | None = None
) -> tuple[int, dict]:
    """Fetch all NYT articles for a given year/month.

    Raises ``EnvironmentError`` if the API key is not set.
    """
    api_key = os.environ.get(ENV_NYTIMES_API_KEY, "")
    if not api_key:
        raise EnvironmentError(
            f"NYT Archive requires environment variable {ENV_NYTIMES_API_KEY}."
        )
    url = _api_url(year, month, api_key)
    return http_get_json(url, transport=transport)


# ---------------------------------------------------------------------------
# Snapshot writers
# ---------------------------------------------------------------------------


def snapshot_archive_month(
    writer: SnapshotWriter,
    raw: dict,
    *,
    year: int,
    month: int,
    http_status: int = 200,
) -> Path:
    slug = f"archive-{year}-{month:02d}"
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "archive.json", raw)
    docs = raw.get("response", {}).get("docs", [])
    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=_api_url(year, month, "(key-redacted)"),
        request_params={"year": year, "month": month},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"docs": len(docs)},
    )
    return snap


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe(*, transport: TransportFn | None = None) -> dict[str, Any]:
    """Probe: attempt a single recent month if the API key is available."""
    result: dict[str, Any] = {
        "api_key_present": False,
        "reachable": None,
        "sample_doc_count": None,
        "sample_fields": None,
        "error": None,
    }
    api_key = os.environ.get(ENV_NYTIMES_API_KEY, "")
    if not api_key:
        result["error"] = f"{ENV_NYTIMES_API_KEY} not set; skipping probe."
        return result
    result["api_key_present"] = True
    try:
        status, body = fetch_archive_month(2025, 6, transport=transport)
        result["reachable"] = True
        docs = body.get("response", {}).get("docs", [])
        result["sample_doc_count"] = len(docs)
        if docs:
            result["sample_fields"] = sorted(docs[0].keys())
    except Exception as exc:
        result["error"] = str(exc)
    return result
