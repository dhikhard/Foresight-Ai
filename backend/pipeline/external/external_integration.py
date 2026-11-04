"""Controlled external-data integration contract.

Reads normalized Polymarket observations and aligns them to the S&P 500
trading calendar using the existing as-of temporal join. Produces one row
per (trading_date, market_id) with the latest eligible observation.

This module is READ-ONLY with respect to V1/V2 data. It reads from:
  - data/processed_external/polymarket_observations.csv  (normalized output)
  - data/processed/sp500_clean.csv                      (trading calendar)

It does not write to any directory. Callers decide output destination.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .external_config import DATA_PROCESSED_EXTERNAL, SP500_CLEAN_CSV
from .temporal_alignment import asof_join, load_trading_calendar, trading_day_close_utc


def load_polymarket_observations(csv_path: Path | None = None) -> list[dict]:
    """Load normalized Polymarket observations from the ingestion CSV.

    Returns list of dicts with: market_id, token_id, t_unix_s (int),
    probability (float).
    """
    path = csv_path or (DATA_PROCESSED_EXTERNAL / "polymarket_observations.csv")
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "market_id": row.get("market_id", ""),
                "token_id": row.get("token_id", ""),
                "t_unix_s": int(row["t_unix_s"]) if row.get("t_unix_s") else 0,
                "probability": float(row["probability"]) if row.get("probability") else 0.0,
            })
    return rows


def align_polymarket_to_trading_days(
    observations: list[dict],
    calendar: list[date],
) -> list[dict]:
    """Align Polymarket observations to S&P 500 trading days via as-of join.

    Uses the existing asof_join infrastructure. For each market (token_id),
    performs an as-of join selecting the latest observation ≤ trading day
    close. Returns one row per (trading_date, market) with aligned fields.

    Output fields per row:
      - trading_date, trading_close_utc
      - aligned_market_id, aligned_token_id
      - aligned_probability, aligned_t_unix_s
      - aligned_staleness_hours, aligned_is_stale

    If observations is empty, returns calendar-only rows (no aligned fields).
    """
    if not observations:
        return [
            {
                "trading_date": d.isoformat(),
                "trading_close_utc": trading_day_close_utc(d).isoformat(),
            }
            for d in calendar
        ]

    obs_by_market: dict[str, list[dict]] = {}
    for obs in observations:
        key = obs.get("market_id", "") or obs.get("token_id", "")
        if key not in obs_by_market:
            obs_by_market[key] = []
        obs_by_market[key].append(obs)

    market_results: dict[str, list[dict]] = {}
    for market_key, market_obs in obs_by_market.items():
        aligned = asof_join(market_obs, calendar, "t_unix_s")
        market_results[market_key] = aligned

    if len(market_results) == 1:
        market_key = next(iter(market_results))
        rows = market_results[market_key]
        out = []
        for row in rows:
            aligned_row = {
                "trading_date": row["trading_date"],
                "trading_close_utc": row["trading_close_utc"],
            }
            if "probability" in row:
                aligned_row["aligned_market_id"] = market_key
                aligned_row["aligned_token_id"] = row.get("token_id", "")
                aligned_row["aligned_probability"] = row["probability"]
                aligned_row["aligned_t_unix_s"] = row.get("t_unix_s", 0)
                aligned_row["aligned_staleness_hours"] = row.get("staleness_hours", -1.0)
                aligned_row["aligned_is_stale"] = row.get("is_stale", True)
            out.append(aligned_row)
        return out

    merged: dict[str, dict] = {}
    for market_key, rows in market_results.items():
        for row in rows:
            td = row["trading_date"]
            if td not in merged:
                merged[td] = {
                    "trading_date": td,
                    "trading_close_utc": row["trading_close_utc"],
                }
            if "probability" in row:
                prefix = f"m{len([k for k in merged[td] if k.startswith('aligned_market_id_')])}"
                merged[td][f"aligned_market_id_{prefix}"] = market_key
                merged[td][f"aligned_token_id_{prefix}"] = row.get("token_id", "")
                merged[td][f"aligned_probability_{prefix}"] = row["probability"]
                merged[td][f"aligned_t_unix_s_{prefix}"] = row.get("t_unix_s", 0)
                merged[td][f"aligned_staleness_hours_{prefix}"] = row.get("staleness_hours", -1.0)
                merged[td][f"aligned_is_stale_{prefix}"] = row.get("is_stale", True)

    return [merged[d.isoformat()] for d in calendar if d.isoformat() in merged]
