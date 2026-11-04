"""CLI for running external data ingestion from all sources.

Usage:
    python -m backend.pipeline.external.run_external_ingest --source kalshi --probe
    python -m backend.pipeline.external.run_external_ingest --source all --probe
    python -m backend.pipeline.external.run_external_ingest --source polymarket --ingest
    python -m backend.pipeline.external.run_external_ingest --source polymarket --ingest --market-ids 559652 559651
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import SnapshotWriter, load_dotenv_if_available
from .external_config import DATA_RAW_EXTERNAL, DATA_PROCESSED_EXTERNAL

load_dotenv_if_available()


def _probe(source: str) -> dict:
    if source == "kalshi":
        from .fetch_kalshi import probe
    elif source == "polymarket":
        from .fetch_polymarket import probe
    elif source == "gdelt":
        from .fetch_gdelt import probe
    elif source == "nytimes":
        from .fetch_nytimes import probe
    elif source == "guardian":
        from .fetch_guardian import probe
    else:
        return {"error": f"Unknown source: {source}"}
    return probe()


def run_probe(sources: list[str]) -> dict[str, dict]:
    results = {}
    for src in sources:
        print(f"  Probing {src}...", end=" ", flush=True)
        r = _probe(src)
        results[src] = r
        err = r.get("error")
        if err:
            print(f"ERROR: {err}")
        else:
            print("OK")
    return results


def _ingest_polymarket(args: argparse.Namespace) -> dict:
    from .fetch_polymarket import run_polymarket_ingest

    market_ids = None
    if args.market_ids:
        market_ids = [mid.strip() for mid in args.market_ids.split(",")]

    print(f"  Running Polymarket ingestion...")
    result = run_polymarket_ingest(
        market_ids=market_ids,
        closed=args.closed,
        limit=args.limit,
    )

    print(f"\n--- Polymarket Ingestion Summary ---")
    print(f"  Markets discovered: {result.markets_discovered}")
    print(f"  Markets fetched:    {result.markets_fetched}")
    print(f"  Price snapshots:    {result.price_snapshots}")
    print(f"  Total observations: {result.total_observations}")
    print(f"  Deduped observations: {result.deduped_observations}")
    print(f"  Written rows:       {result.written_rows}")

    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    - {err}")

    if result.output_csv:
        print(f"  Output CSV: {result.output_csv}")
    if result.output_report:
        print(f"  Output report: {result.output_report}")

    return result.summary()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="External data ingestion for ForeSightAI"
    )
    p.add_argument(
        "--source",
        choices=["kalshi", "polymarket", "gdelt", "nytimes", "guardian", "all"],
        default="all",
        help="Which source to probe/ingest (default: all)",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--probe",
        action="store_true",
        help="Run small probes (no snapshots, no writes).",
    )
    mode.add_argument(
        "--ingest",
        action="store_true",
        help="Run full ingestion (snapshots + normalization + CSV).",
    )

    # Polymarket-specific options
    p.add_argument(
        "--market-ids",
        default=None,
        help="Comma-separated Gamma market IDs to fetch (Polymarket only).",
    )
    p.add_argument(
        "--closed",
        action="store_true",
        default=False,
        help="Include closed markets (Polymarket only).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max markets to discover if no --market-ids (default: 20).",
    )

    args = p.parse_args(argv)

    sources = (
        ["kalshi", "polymarket", "gdelt", "nytimes", "guardian"]
        if args.source == "all"
        else [args.source]
    )

    if args.probe:
        print(f"Running probes for: {', '.join(sources)}")
        results = run_probe(sources)
        print("\n--- Probe Summary ---")
        for src, r in results.items():
            print(f"  {src}: {json.dumps(r, indent=2, default=str)}")
        return

    if args.ingest:
        if args.source == "polymarket":
            print("Running Polymarket ingestion...")
            _ingest_polymarket(args)
        else:
            print(f"Ingestion for '{args.source}' not yet implemented.")
            print("Currently supported: --source polymarket --ingest")
        return


if __name__ == "__main__":
    main()
