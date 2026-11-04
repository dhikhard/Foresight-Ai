"""Polymarket historical data fetcher.

All research-relevant endpoints are public (no auth).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import (
    SnapshotWriter,
    TransportFn,
    http_get_json,
    sha256_bytes,
)
from .external_config import (
    DATA_PROCESSED_EXTERNAL,
    DATA_RAW_EXTERNAL,
    POLYMARKET_CLOB_BASE,
    POLYMARKET_GAMMA_BASE,
)

SOURCE = "polymarket"

# ---------------------------------------------------------------------------
# Gamma API (market discovery)
# ---------------------------------------------------------------------------


def gamma_list_markets(
    *,
    closed: bool = False,
    limit: int = 20,
    cursor: str | None = None,
    transport: TransportFn | None = None,
) -> tuple[int, dict]:
    """GET /markets on the Gamma API with keyset pagination."""
    url = f"{POLYMARKET_GAMMA_BASE}/markets/keyset?closed={'true' if closed else 'false'}&limit={limit}"
    if cursor:
        url += f"&after_cursor={cursor}"
    return http_get_json(url, transport=transport)


def gamma_get_market(
    market_id: str, *, transport: TransportFn | None = None
) -> tuple[int, dict]:
    """GET /markets/{id}."""
    url = f"{POLYMARKET_GAMMA_BASE}/markets/{market_id}"
    return http_get_json(url, transport=transport)


# ---------------------------------------------------------------------------
# CLOB API (price history + CLOB market info)
# ---------------------------------------------------------------------------


def clob_prices_history(
    token_id: str,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
    interval: str = "max",
    fidelity: int = 1,
    transport: TransportFn | None = None,
) -> tuple[int, dict]:
    """GET /prices-history on the CLOB API (public, keyless).

    ``interval`` must be one of: max, all, 1m, 1w, 1d, 6h, 1h.
    ``fidelity`` is in minutes (default 1).
    """
    url = (
        f"{POLYMARKET_CLOB_BASE}/prices-history"
        f"?market={token_id}&interval={interval}&fidelity={fidelity}"
    )
    if start_ts is not None:
        url += f"&startTs={start_ts}"
    if end_ts is not None:
        url += f"&endTs={end_ts}"
    return http_get_json(url, transport=transport)


# ---------------------------------------------------------------------------
# Snapshot writers
# ---------------------------------------------------------------------------


def snapshot_gamma_discovery(
    writer: SnapshotWriter,
    raw: dict,
    *,
    slug: str = "gamma-discovery",
    request_params: dict | None = None,
    http_status: int = 200,
) -> Path:
    """Write a Gamma discovery snapshot (list of markets)."""
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "gamma_markets.json", raw)
    markets = raw if isinstance(raw, list) else raw.get("markets", [])
    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=f"{POLYMARKET_GAMMA_BASE}/markets",
        request_params=request_params or {},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"markets": len(markets)},
    )
    return snap


def snapshot_prices_history(
    writer: SnapshotWriter,
    raw: dict,
    *,
    token_id: str,
    request_params: dict | None = None,
    http_status: int = 200,
) -> Path:
    """Write a CLOB price-history snapshot."""
    slug = f"prices-{token_id[:40]}"
    snap = writer.new_snapshot(SOURCE, slug)
    raw_file = writer.write_json(snap, "prices_history.json", raw)
    history = raw.get("history", [])
    writer.write_manifest(
        snap,
        source=SOURCE,
        endpoint=f"{POLYMARKET_CLOB_BASE}/prices-history",
        request_params=request_params or {"token_id": token_id},
        http_status=http_status,
        raw_files=[raw_file],
        row_counts={"price_points": len(history)},
    )
    return snap


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe(*, transport: TransportFn | None = None) -> dict[str, Any]:
    """Small probe: fetch 1 closed market + its price history.

    Returns probe results without writing snapshots.
    """
    result: dict[str, Any] = {
        "gamma_discoverable": None,
        "prices_history_points": None,
        "sample_market": None,
        "error": None,
    }

    try:
        status, body = gamma_list_markets(closed=True, limit=2, transport=transport)
        if isinstance(body, dict):
            markets = body.get("markets", body.get("items", []))
        else:
            markets = body if isinstance(body, list) else []

        if not markets:
            result["error"] = "No closed markets returned by Gamma"
            return result

        mkt = markets[0]
        result["sample_market"] = {
            "id": mkt.get("id"),
            "slug": mkt.get("slug"),
            "question": mkt.get("question"),
        }
        result["gamma_discoverable"] = True

        # Extract token ID from clobTokenIds
        clob_ids_raw = mkt.get("clobTokenIds")
        if isinstance(clob_ids_raw, str):
            import json
            token_ids = json.loads(clob_ids_raw)
        elif isinstance(clob_ids_raw, list):
            token_ids = clob_ids_raw
        else:
            result["error"] = "No clobTokenIds on sample market"
            return result

        if token_ids:
            status2, prices = clob_prices_history(token_ids[0], transport=transport)
            hist = prices.get("history", [])
            result["prices_history_points"] = len(hist)

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Full ingestion
# ---------------------------------------------------------------------------

_POLYMARKET_DAILY_FIDELITY = 1440
_POLYMARKET_MAX_PAGE_SIZE = 100
_MAX_GAMMA_PAGES = 50


def _discover_token_ids(
    market_id: str,
    *,
    transport: TransportFn | None = None,
) -> tuple[str, str, dict] | None:
    """Fetch a single Gamma market and extract (yes_token_id, no_token_id, raw_market).

    Returns None if the market cannot be fetched or has no token IDs.
    """
    try:
        status, raw = gamma_get_market(market_id, transport=transport)
        if status != 200:
            return None
        clob_raw = raw.get("clobTokenIds", "")
        if isinstance(clob_raw, str):
            token_ids = json.loads(clob_raw)
        elif isinstance(clob_raw, list):
            token_ids = clob_raw
        else:
            return None
        if len(token_ids) < 2:
            return None
        return str(token_ids[0]), str(token_ids[1]), raw
    except Exception:
        return None


def _fetch_daily_history(
    token_id: str,
    *,
    transport: TransportFn | None = None,
) -> tuple[int, dict] | None:
    """Fetch full daily price history from CLOB (interval=max, fidelity=1440).

    Returns (status, raw_response) or None on failure.
    """
    try:
        return clob_prices_history(
            token_id,
            interval="max",
            fidelity=_POLYMARKET_DAILY_FIDELITY,
            transport=transport,
        )
    except Exception:
        return None


class PolymarketIngestResult:
    """Outcome of a Polymarket ingestion run."""

    def __init__(self) -> None:
        self.markets_discovered: int = 0
        self.markets_fetched: int = 0
        self.price_snapshots: int = 0
        self.total_observations: int = 0
        self.deduped_observations: int = 0
        self.written_rows: int = 0
        self.errors: list[str] = []
        self.output_csv: Path | None = None
        self.output_report: Path | None = None
        self.snapshot_dir: Path | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "markets_discovered": self.markets_discovered,
            "markets_fetched": self.markets_fetched,
            "price_snapshots": self.price_snapshots,
            "total_observations": self.total_observations,
            "deduped_observations": self.deduped_observations,
            "written_rows": self.written_rows,
            "errors": self.errors,
            "output_csv": str(self.output_csv) if self.output_csv else None,
            "output_report": str(self.output_report) if self.output_report else None,
        }


def run_polymarket_ingest(
    *,
    market_ids: list[str] | None = None,
    closed: bool = False,
    limit: int = 20,
    transport: TransportFn | None = None,
) -> PolymarketIngestResult:
    """End-to-end Polymarket ingestion: discover → fetch → snapshot → normalize → write.

    Steps:
      1. Discover markets via Gamma (or use provided market_ids)
      2. For each market, fetch the YES token daily price history from CLOB
      3. Write immutable snapshots via SnapshotWriter
      4. Normalize with parse_polymarket_prices()
      5. Deduplicate across markets
      6. Validate probability range
      7. Write deterministic CSV + provenance report

    No current-state Gamma fields (outcomePrices, lastTradePrice, updatedAt)
    are used in the output. Only CLOB price observations with explicit timestamps.
    """
    from .normalize_external import (
        dedup_rows,
        parse_polymarket_prices,
        validate_probability_rows,
        write_csv,
        write_normalization_report,
    )

    result = PolymarketIngestResult()
    writer = SnapshotWriter(DATA_RAW_EXTERNAL)
    all_observations: list[dict] = []
    snapshot_paths: list[Path] = []

    # ── Step 1: Discover markets ────────────────────────────────────────
    if market_ids:
        ids_to_fetch = market_ids
    else:
        try:
            status, body = gamma_list_markets(
                closed=closed, limit=limit, transport=transport,
            )
            if isinstance(body, list):
                markets = body
            elif isinstance(body, dict):
                markets = body.get("markets", body.get("items", []))
            else:
                markets = []
            ids_to_fetch = [m.get("id", "") for m in markets if m.get("id")]
        except Exception as exc:
            result.errors.append(f"Gamma discovery failed: {exc}")
            return result

    result.markets_discovered = len(ids_to_fetch)

    # ── Steps 2-4: Fetch + snapshot + normalize per market ──────────────
    for mid in ids_to_fetch:
        # Discover token IDs
        tokens = _discover_token_ids(mid, transport=transport)
        if tokens is None:
            result.errors.append(f"Could not get tokens for market {mid}")
            continue
        yes_tid, no_tid, raw_market = tokens

        # Fetch daily price history for YES token
        fetched = _fetch_daily_history(yes_tid, transport=transport)
        if fetched is None:
            result.errors.append(f"Price history failed for market {mid} token {yes_tid[:20]}")
            continue

        hist_status, raw_prices = fetched
        result.markets_fetched += 1

        # Write immutable snapshot
        try:
            snap = snapshot_prices_history(
                writer,
                raw_prices,
                token_id=yes_tid,
                request_params={
                    "interval": "max",
                    "fidelity": _POLYMARKET_DAILY_FIDELITY,
                    "market_id": mid,
                },
                http_status=hist_status,
            )
            snapshot_paths.append(snap)
            result.price_snapshots += 1
        except FileExistsError:
            # Snapshot already exists (append-only invariant) — skip write
            pass
        except Exception as exc:
            result.errors.append(f"Snapshot write failed for market {mid}: {exc}")
            continue

        # Normalize
        observations = parse_polymarket_prices(raw_prices)
        # Stamp token_id (CLOB response lacks top-level token_id)
        for obs in observations:
            obs["token_id"] = yes_tid
            obs["market_id"] = mid
        result.total_observations += len(observations)
        all_observations.extend(observations)

    # ── Step 5: Deduplicate ─────────────────────────────────────────────
    from .external_config import POLYMARKET_DEDUP_KEYS
    deduped = dedup_rows(all_observations, POLYMARKET_DEDUP_KEYS)
    result.deduped_observations = len(deduped)

    # ── Step 6: Validate ────────────────────────────────────────────────
    try:
        validate_probability_rows(deduped, field="probability")
    except ValueError as exc:
        result.errors.append(str(exc))

    # ── Step 7: Write output ────────────────────────────────────────────
    DATA_PROCESSED_EXTERNAL.mkdir(parents=True, exist_ok=True)
    output_csv = DATA_PROCESSED_EXTERNAL / "polymarket_observations.csv"
    write_csv(deduped, output_csv)
    result.written_rows = len(deduped)
    result.output_csv = output_csv

    output_report = DATA_PROCESSED_EXTERNAL / "polymarket_ingest_report.json"
    write_normalization_report(
        output_report,
        source=SOURCE,
        input_snapshots=[str(s) for s in snapshot_paths],
        input_snapshot_hashes=[
            sha256_bytes((s / "prices_history.json").read_bytes())
            for s in snapshot_paths
            if (s / "prices_history.json").exists()
        ],
        output_files={
            "polymarket_observations.csv": sha256_bytes(output_csv.read_bytes()),
        },
        row_counts={"observations": len(deduped)},
        dedup_counts={
            "before_dedup": result.total_observations,
            "after_dedup": result.deduped_observations,
        },
    )
    result.output_report = output_report

    if snapshot_paths:
        result.snapshot_dir = snapshot_paths[0].parent

    return result
