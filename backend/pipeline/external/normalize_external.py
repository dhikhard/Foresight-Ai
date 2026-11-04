"""Deterministic normalization of raw external snapshots into canonical CSVs.

Every function is a pure transformation over raw dicts/lists (no I/O except
the final CSV writers).  All outputs are byte-reproducible given identical inputs
(except ``generated_at_utc`` in provenance reports).
"""
from __future__ import annotations

import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import sha256_bytes


# ─────────────────────────────────────────────────────────────────────
# Kalshi parsers
# ─────────────────────────────────────────────────────────────────────


def parse_kalshi_markets(raw: dict | list) -> list[dict]:
    """Parse Kalshi market metadata (lifecycle side-table rows).

    Fields: ticker, event_ticker, market_type, title, status, open_time,
    close_time, settlement_ts, settlement_value_dollars, result,
    can_close_early, created_time, updated_time.
    """
    markets = raw.get("markets", []) if isinstance(raw, dict) else raw
    rows = []
    for m in markets:
        rows.append({
            "ticker": m.get("ticker", ""),
            "event_ticker": m.get("event_ticker", ""),
            "market_type": m.get("market_type", ""),
            "status": m.get("status", ""),
            "open_time": m.get("open_time", ""),
            "close_time": m.get("close_time", ""),
            "settlement_ts": m.get("settlement_ts", ""),
            "settlement_value_dollars": m.get("settlement_value_dollars", ""),
            "result": m.get("result", ""),
            "can_close_early": m.get("can_close_early", ""),
            "created_time": m.get("created_time", ""),
            "updated_time": m.get("updated_time", ""),
            "expiration_value": m.get("expiration_value", ""),
            "occurrence_datetime": m.get("occurrence_datetime", ""),
            "latest_expiration_time": m.get("latest_expiration_time", ""),
            "notional_value_dollars": m.get("notional_value_dollars", ""),
        })
    return rows


def parse_kalshi_candles(raw: dict) -> list[dict]:
    """Parse Kalshi candlestick data into point-in-time observation rows.

    Excludes synthetic forward-projected candles (where the API
    ``include_latest_before_start`` flag injects a projected candle).
    The raw snapshot should never contain synthetic candles if fetched
    with ``include_latest_before_start=false`` (the default), but this
    parser double-checks.

    Returns dicts with: ticker, candle_end_unix_s, interval_minutes,
    prob_close, prob_open, prob_high, prob_low, vwap,
    yes_bid_close, yes_ask_close, volume_fp, open_interest_fp.
    """
    candles = raw.get("candlesticks", [])
    ticker = raw.get("ticker", "")
    rows = []
    for c in candles:
        if c.get("synthetic"):
            continue
        rows.append({
            "ticker": ticker,
            "candle_end_unix_s": c.get("end_period_ts", 0),
            "interval_minutes": c.get("period_interval", c.get("period", "")),
            "prob_close": _safe_float(c.get("price", c.get("close", ""))),
            "prob_open": _safe_float(c.get("price_open", c.get("open", ""))),
            "prob_high": _safe_float(c.get("price_high", c.get("high", ""))),
            "prob_low": _safe_float(c.get("price_low", c.get("low", ""))),
            "vwap": _safe_float(c.get("price_mean", "")),
            "yes_bid_close": _safe_float(c.get("yes_bid", c.get("yes_bid_close", ""))),
            "yes_ask_close": _safe_float(c.get("yes_ask", c.get("yes_ask_close", ""))),
            "volume_fp": c.get("volume_fp", "0"),
            "open_interest_fp": c.get("open_interest_fp", "0"),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Polymarket parsers
# ─────────────────────────────────────────────────────────────────────


def parse_polymarket_markets(raw: dict | list) -> list[dict]:
    """Parse Gamma market discovery response into metadata rows.

    Fields: market_id, slug, question, condition_id,
    yes_token_id, no_token_id, start_date, end_date, closed.
    """
    if isinstance(raw, list):
        markets = raw
    elif isinstance(raw, dict):
        markets = raw.get("markets", raw.get("items", []))
    else:
        markets = []

    rows = []
    for m in markets:
        yes_tid, no_tid = _extract_token_ids(m)
        rows.append({
            "market_id": m.get("id", ""),
            "slug": m.get("slug", ""),
            "question": m.get("question", ""),
            "condition_id": m.get("conditionId", m.get("condition_id", "")),
            "yes_token_id": yes_tid,
            "no_token_id": no_tid,
            "start_date": m.get("startDate", m.get("startDateIso", "")),
            "end_date": m.get("endDate", m.get("endDateIso", "")),
            "closed": m.get("closed", ""),
        })
    return rows


def parse_polymarket_prices(raw: dict) -> list[dict]:
    """Parse CLOB /prices-history response into observation rows.

    Returns dicts with: token_id, t_unix_s, probability.
    """
    history = raw.get("history", [])
    return [
        {
            "token_id": raw.get("token_id", raw.get("market", "")),
            "t_unix_s": int(p.get("t", 0)),
            "probability": float(p.get("p", 0.0)),
        }
        for p in history
        if "t" in p and "p" in p
    ]


# ─────────────────────────────────────────────────────────────────────
# News parsers
# ─────────────────────────────────────────────────────────────────────


def parse_gdelt_articles(raw: dict | Any) -> list[dict]:
    """Parse GDELT DOC 2.0 ArtList response.

    Fields: article_id, source, published_at_utc, title, url,
    domain, language, tone (if available).
    """
    if isinstance(raw, dict):
        articles = raw.get("articles", [])
    elif isinstance(raw, list):
        articles = raw
    else:
        articles = []

    rows = []
    for i, a in enumerate(articles):
        # GDELT article structure: seendate (YYYYMMDDHHMMSS), url, title, domain, language, socialimage
        seendate = a.get("seendate", "")
        published_utc = _gdelt_seendate_to_iso(seendate)

        rows.append({
            "article_id": a.get("url", f"gdelt-{i}"),
            "source": "gdelt",
            "published_at_utc": published_utc,
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "domain": a.get("domain", ""),
            "language": a.get("language", ""),
            "snippet": "",
        })
    return rows


def parse_nytimes_archive(raw: dict) -> list[dict]:
    """Parse NYT Archive API monthly response.

    Fields: article_id, source, published_at_utc, title, url,
    section, snippet, keywords, byline.
    """
    docs = raw.get("response", {}).get("docs", [])
    rows = []
    for doc in docs:
        pub_date = doc.get("pub_date", "")
        rows.append({
            "article_id": doc.get("_id", ""),
            "source": "nytimes",
            "published_at_utc": _parse_iso_timestamp(pub_date),
            "title": doc.get("headline", {}).get("main", ""),
            "url": doc.get("web_url", ""),
            "section": doc.get("section_name", ""),
            "snippet": doc.get("snippet", ""),
            "keywords": json.dumps(doc.get("keywords", []), ensure_ascii=False)
                        if isinstance(doc.get("keywords"), list)
                        else str(doc.get("keywords", "")),
            "byline": doc.get("byline", {}).get("original", ""),
        })
    return rows


def parse_guardian_search(raw: dict) -> list[dict]:
    """Parse Guardian Open Platform search response.

    Fields: article_id, source, published_at_utc, title, url,
    section, snippet/bodyText, fields.
    """
    results = raw.get("response", {}).get("results", [])
    rows = []
    for r in results:
        web_pub = r.get("webPublicationDate", "")
        fields = r.get("fields", {})
        body = fields.get("bodyText", "")
        rows.append({
            "article_id": r.get("id", ""),
            "source": "guardian",
            "published_at_utc": _parse_iso_timestamp(web_pub),
            "title": r.get("webTitle", ""),
            "url": r.get("webUrl", ""),
            "section": r.get("sectionName", ""),
            "snippet": body[:500] if body else "",  # Truncate for sidecar
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Resolution quarantine split
# ─────────────────────────────────────────────────────────────────────


def split_kalshi_resolution(market_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate lifecycle metadata into point-in-time fields and
    resolution/settlement fields.

    Returns (lifecycle_rows, resolution_rows).
    """
    RESOLUTION_FIELDS = {
        "settlement_ts", "settlement_value_dollars", "result",
        "expiration_value",
    }
    lifecycles = []
    resolutions = []
    for row in market_rows:
        lc = {k: v for k, v in row.items() if k not in RESOLUTION_FIELDS}
        res = {k: row.get(k, "") for k in ("ticker", *RESOLUTION_FIELDS)}
        lifecycles.append(lc)
        resolutions.append(res)
    return lifecycles, resolutions


def split_polymarket_resolution(market_rows: list[dict]) -> list[dict]:
    """Polymarket Gamma markets don't expose settlement inline in a way
    that differs from point-in-time; but we strip any future-resolved
    fields into a separate file if present.

    For now, returns the metadata rows unchanged; resolution status is
    tracked via UMA endpoints (out of scope for Phase ④ snapshot layer).
    """
    return market_rows


# ─────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────


def dedup_rows(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Deterministic deduplication: keep first row per unique key tuple.

    Keys are extracted as a tuple; empty/missing values are treated as
    empty strings (never as None) for deterministic ordering.
    """
    seen: set[tuple[str, ...]] = set()
    out = []
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────


def validate_probability_rows(rows: list[dict], field: str = "probability") -> None:
    """Raise ValueError if any value in the given field is outside [0, 1].

    Skips rows where the field is missing, empty, or non-numeric (caller
    decides whether missing values are acceptable).
    """
    bad = []
    for i, r in enumerate(rows):
        v = r.get(field)
        if v is None or v == "":
            continue
        f = float(v)
        if f < 0.0 or f > 1.0:
            bad.append((i, f))
    if bad:
        raise ValueError(
            f"Probability field '{field}' has {len(bad)} out-of-range values: "
            f"first few: {bad[:5]}"
        )


# ─────────────────────────────────────────────────────────────────────
# CSV / report writers
# ─────────────────────────────────────────────────────────────────────


def write_csv(rows: list[dict], path: Path) -> None:
    """Write rows to a deterministic CSV (sorted columns, stable row order)."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    fieldnames = sorted(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def compute_csv_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_normalization_report(
    path: Path,
    *,
    source: str,
    input_snapshots: list[str],
    input_snapshot_hashes: list[str],
    output_files: dict[str, str],  # filename -> sha256
    row_counts: dict[str, int],
    dedup_counts: dict[str, int],
) -> Path:
    """Write a normalization provenance report (tracked, reproducible)."""
    report = {
        "source": source,
        "input_snapshots": input_snapshots,
        "input_snapshot_hashes": input_snapshot_hashes,
        "output_files": output_files,
        "row_counts": row_counts,
        "dedup_counts": dedup_counts,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode()
    path.write_bytes(raw)
    return path


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _safe_float(v: Any) -> float:
    """Convert to float safely, returning 0.0 for empty/unparseable."""
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _gdelt_seendate_to_iso(seendate: str) -> str:
    """Convert GDELT YYYYMMDDHHMMSS UTC string to ISO-8601."""
    if not seendate or len(seendate) < 14:
        return seendate
    try:
        dt = datetime(
            int(seendate[:4]), int(seendate[4:6]), int(seendate[6:8]),
            int(seendate[8:10]), int(seendate[10:12]), int(seendate[12:14]),
            tzinfo=timezone.utc,
        )
        return dt.isoformat()
    except (ValueError, IndexError):
        return seendate


def _parse_iso_timestamp(ts: str) -> str:
    """Parse ISO-8601 timestamp; return ISO string or original."""
    if not ts:
        return ts
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        return ts


def _extract_token_ids(m: dict) -> tuple[str, str]:
    """Extract YES/NO token IDs from a Gamma market object."""
    raw = m.get("clobTokenIds", m.get("clob_token_ids", ""))
    if isinstance(raw, str):
        try:
            ids = json.loads(raw)
            if isinstance(ids, list) and len(ids) >= 2:
                return str(ids[0]), str(ids[1])
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(raw, list) and len(raw) >= 2:
        return str(raw[0]), str(raw[1])
    # Fallback: check outcomes dict
    outcomes = m.get("outcomes", {})
    if isinstance(outcomes, dict):
        yes = outcomes.get("yes", {})
        no = outcomes.get("no", {})
        return str(yes.get("tokenId", yes.get("token_id", ""))), str(no.get("tokenId", no.get("token_id", "")))
    return "", ""
