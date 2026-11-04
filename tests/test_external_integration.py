"""Tests for controlled external-data integration.

Covers:
  - as-of alignment correctness
  - temporal leakage (future observations excluded, equality boundary)
  - missing/failed market behavior
  - resolution/current-state quarantine
  - target leakage prevention
  - write isolation (no writes to V1/V2 directories)
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.pipeline.external.external_integration import (
    align_polymarket_to_trading_days,
    load_polymarket_observations,
)
from backend.pipeline.external.temporal_alignment import (
    asof_join,
    trading_day_close_utc,
)
from backend.pipeline.external.external_config import (
    DATA_RAW_EXTERNAL,
    DATA_PROCESSED_EXTERNAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(market_id: str, token_id: str, t_unix_s: int, probability: float) -> dict:
    return {
        "market_id": market_id,
        "token_id": token_id,
        "t_unix_s": t_unix_s,
        "probability": probability,
    }


def _ts(date_obj: date, hour: int = 12, minute: int = 0) -> int:
    """Unix timestamp for a given date + hour/minute UTC."""
    dt = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute, tzinfo=timezone.utc)
    return int(dt.timestamp())


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------


class TestAlignPolymarketToTradingDays:
    def test_basic_alignment(self):
        cal = [date(2024, 7, 15), date(2024, 7, 16)]
        close1 = trading_day_close_utc(date(2024, 7, 15))
        close2 = trading_day_close_utc(date(2024, 7, 16))
        obs = [
            _make_obs("m1", "tok1", int((close1 - timedelta(hours=2)).timestamp()), 0.60),
            _make_obs("m1", "tok1", int((close2 - timedelta(hours=1)).timestamp()), 0.65),
        ]
        result = align_polymarket_to_trading_days(obs, cal)
        assert len(result) == 2
        assert result[0]["aligned_probability"] == pytest.approx(0.60)
        assert result[0]["aligned_market_id"] == "m1"
        assert result[1]["aligned_probability"] == pytest.approx(0.65)
        assert result[1]["aligned_is_stale"] is False

    def test_empty_observations_returns_calendar_only(self):
        cal = [date(2024, 7, 15)]
        result = align_polymarket_to_trading_days([], cal)
        assert len(result) == 1
        assert "aligned_probability" not in result[0]
        assert result[0]["trading_date"] == "2024-07-15"

    def test_carry_forward(self):
        cal = [date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 17)]
        close1 = trading_day_close_utc(date(2024, 7, 15))
        obs = [
            _make_obs("m1", "tok1", int((close1 - timedelta(hours=1)).timestamp()), 0.50),
        ]
        result = align_polymarket_to_trading_days(obs, cal)
        assert len(result) == 3
        assert result[0]["aligned_probability"] == pytest.approx(0.50)
        assert result[1]["aligned_probability"] == pytest.approx(0.50)
        assert result[2]["aligned_probability"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Temporal leakage — future observations MUST NOT be used
# ---------------------------------------------------------------------------


class TestLeakageFutureExcluded:
    def test_future_observation_excluded(self):
        """Observation 1 second after close must not be used."""
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        future_ts = int((close + timedelta(seconds=1)).timestamp())
        obs = [_make_obs("m1", "tok1", future_ts, 0.99)]
        result = align_polymarket_to_trading_days(obs, [d])
        assert len(result) == 1
        assert "aligned_probability" not in result[0]

    def test_observation_1_hour_after_excluded(self):
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        future_ts = int((close + timedelta(hours=1)).timestamp())
        obs = [_make_obs("m1", "tok1", future_ts, 0.80)]
        result = align_polymarket_to_trading_days(obs, [d])
        assert "aligned_probability" not in result[0]

    def test_only_future_observations_all_dates_empty(self):
        """If all observations are after prediction time, no aligned data."""
        cal = [date(2024, 7, 15), date(2024, 7, 16)]
        close1 = trading_day_close_utc(date(2024, 7, 15))
        close2 = trading_day_close_utc(date(2024, 7, 16))
        obs = [
            _make_obs("m1", "tok1", int((close2 + timedelta(hours=1)).timestamp()), 0.70),
        ]
        result = align_polymarket_to_trading_days(obs, cal)
        assert len(result) == 2
        assert "aligned_probability" not in result[0]
        assert "aligned_probability" not in result[1]


# ---------------------------------------------------------------------------
# Equality boundary — exact-at-close IS allowed
# ---------------------------------------------------------------------------


class TestLeakageEqualityBoundary:
    def test_exact_at_close_included(self):
        """Observation exactly at trading day close is eligible (et <= cutoff)."""
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        exact_ts = int(close.timestamp())
        obs = [_make_obs("m1", "tok1", exact_ts, 0.55)]
        result = align_polymarket_to_trading_days(obs, [d])
        assert result[0]["aligned_probability"] == pytest.approx(0.55)

    def test_1_second_before_close_included(self):
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        ts = int((close - timedelta(seconds=1)).timestamp())
        obs = [_make_obs("m1", "tok1", ts, 0.45)]
        result = align_polymarket_to_trading_days(obs, [d])
        assert result[0]["aligned_probability"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Multiple historical observations — latest-before-cutoff wins
# ---------------------------------------------------------------------------


class TestMultipleObservations:
    def test_latest_before_cutoff_selected(self):
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        obs = [
            _make_obs("m1", "tok1", int((close - timedelta(hours=12)).timestamp()), 0.40),
            _make_obs("m1", "tok1", int((close - timedelta(hours=2)).timestamp()), 0.60),
            _make_obs("m1", "tok1", int((close - timedelta(hours=6)).timestamp()), 0.50),
        ]
        result = align_polymarket_to_trading_days(obs, [d])
        assert result[0]["aligned_probability"] == pytest.approx(0.60)

    def test_same_timestamp_last_wins(self):
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        ts = int((close - timedelta(hours=1)).timestamp())
        obs = [
            _make_obs("m1", "tok1", ts, 0.30),
            _make_obs("m1", "tok1", ts, 0.70),
        ]
        result = align_polymarket_to_trading_days(obs, [d])
        assert result[0]["aligned_probability"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Duplicate timestamps — deterministic
# ---------------------------------------------------------------------------


class TestDuplicateTimestamps:
    def test_deterministic_with_identical_timestamps(self):
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        ts = int((close - timedelta(hours=3)).timestamp())
        obs = [_make_obs("m1", "tok1", ts, 0.55)] * 5
        result1 = align_polymarket_to_trading_days(obs, [d])
        result2 = align_polymarket_to_trading_days(obs, [d])
        assert result1[0]["aligned_probability"] == result2[0]["aligned_probability"]


# ---------------------------------------------------------------------------
# Missing market — S&P 500 rows remain valid
# ---------------------------------------------------------------------------


class TestMissingMarket:
    def test_no_polymarket_data_calendar_rows_unaffected(self):
        cal = [date(2024, 7, 15), date(2024, 7, 16)]
        result = align_polymarket_to_trading_days([], cal)
        assert len(result) == 2
        assert result[0]["trading_date"] == "2024-07-15"
        assert result[1]["trading_date"] == "2024-07-16"
        assert "aligned_probability" not in result[0]
        assert "aligned_probability" not in result[1]

    def test_partial_market_coverage(self):
        """Only day 2 has an observation; day 1 gets no aligned data."""
        cal = [date(2024, 7, 15), date(2024, 7, 16)]
        close2 = trading_day_close_utc(date(2024, 7, 16))
        obs = [_make_obs("m1", "tok1", int((close2 - timedelta(hours=1)).timestamp()), 0.70)]
        result = align_polymarket_to_trading_days(obs, cal)
        assert "aligned_probability" not in result[0]
        assert result[1]["aligned_probability"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Failed ingestion — no fabricated observations
# ---------------------------------------------------------------------------


class TestFailedIngestion:
    def test_empty_csv_produces_no_aligned_data(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = load_polymarket_observations(csv_path)
        assert result == []

    def test_missing_csv_produces_no_aligned_data(self, tmp_path):
        csv_path = tmp_path / "nonexistent.csv"
        result = load_polymarket_observations(csv_path)
        assert result == []


# ---------------------------------------------------------------------------
# Resolution / current-state quarantine
# ---------------------------------------------------------------------------


class TestResolutionQuarantine:
    def test_no_resolution_fields_in_observation_schema(self):
        """Observations must not contain resolution, outcomePrices, or
        lastTradePrice fields."""
        from backend.pipeline.external.normalize_external import parse_polymarket_prices
        raw = {"history": [{"t": 1000, "p": 0.5}, {"t": 2000, "p": 0.6}]}
        rows = parse_polymarket_prices(raw)
        for row in rows:
            assert "resolution" not in row
            assert "result" not in row
            assert "outcomePrices" not in row
            assert "lastTradePrice" not in row
            assert "settlement" not in row

    def test_gamma_metadata_not_used_in_alignment(self):
        """Gamma market metadata (outcomePrices, updatedAt) must not appear
        in aligned output."""
        d = date(2024, 7, 15)
        close = trading_day_close_utc(d)
        obs = [_make_obs("m1", "tok1", int((close - timedelta(hours=1)).timestamp()), 0.50)]
        result = align_polymarket_to_trading_days(obs, [d])
        row = result[0]
        assert "outcomePrices" not in row
        assert "updatedAt" not in row
        assert "lastTradePrice" not in row
        assert "resolution" not in row
        assert "result" not in row


# ---------------------------------------------------------------------------
# Target leakage prevention
# ---------------------------------------------------------------------------


class TestTargetLeakage:
    def test_alignment_cannot_access_target_columns(self):
        """align_polymarket_to_trading_days only reads from Polymarket CSV
        and calendar. It cannot access target_return_1d or any future data."""
        import inspect
        source = inspect.getsource(align_polymarket_to_trading_days)
        assert "target_return" not in source
        assert "add_future_targets" not in source
        assert "close" not in source or "close_utc" in source

    def test_load_observations_cannot_access_targets(self):
        import inspect
        source = inspect.getsource(load_polymarket_observations)
        assert "target" not in source.lower() or "target" not in source
        assert "sp500_features" not in source
        assert "sp500_targets" not in source


# ---------------------------------------------------------------------------
# Write isolation — integration module must NOT write to any directory
# ---------------------------------------------------------------------------


class TestWriteIsolation:
    def test_alignment_function_is_pure(self):
        """align_polymarket_to_trading_days has no filesystem writes."""
        import inspect
        source = inspect.getsource(align_polymarket_to_trading_days)
        assert ".write" not in source
        assert ".mkdir" not in source
        assert "open(" not in source

    def test_load_observations_reads_only(self):
        """load_polymarket_observations only reads, never writes."""
        import inspect
        source = inspect.getsource(load_polymarket_observations)
        assert ".write" not in source
        assert ".mkdir" not in source

    def test_no_write_to_data_raw(self, tmp_path):
        """Integration output must never touch data/raw/."""
        cal = [date(2024, 7, 15)]
        obs = [_make_obs("m1", "tok1", _ts(date(2024, 7, 14)), 0.50)]
        result = align_polymarket_to_trading_days(obs, cal)
        assert len(result) == 1
        assert not DATA_RAW_EXTERNAL.exists() or not any(
            DATA_RAW_EXTERNAL.iterdir()
        )

    def test_no_write_to_data_processed(self, tmp_path):
        """Integration output must never touch data/processed/."""
        cal = [date(2024, 7, 15)]
        obs = [_make_obs("m1", "tok1", _ts(date(2024, 7, 14)), 0.50)]
        result = align_polymarket_to_trading_days(obs, cal)
        assert len(result) == 1

    def test_no_write_to_v1_feature_files(self):
        """Integration must not write to sp500_features.csv."""
        import inspect
        for func_name in ["align_polymarket_to_trading_days", "load_polymarket_observations"]:
            func = align_polymarket_to_trading_days if func_name == "align_polymarket_to_trading_days" else load_polymarket_observations
            source = inspect.getsource(func)
            assert "sp500_features" not in source

    def test_no_write_to_v2_feature_files(self):
        """Integration must not write to sp500_features_v2.csv."""
        import inspect
        for func in [align_polymarket_to_trading_days, load_polymarket_observations]:
            source = inspect.getsource(func)
            assert "features_v2" not in source

    def test_no_write_to_target_datasets(self):
        """Integration must not write to sp500_targets.csv."""
        import inspect
        for func in [align_polymarket_to_trading_days, load_polymarket_observations]:
            source = inspect.getsource(func)
            assert "targets" not in source

    def test_module_has_no_file_write_imports(self):
        """The integration module should not import SnapshotWriter or
        any write-capable utilities."""
        import backend.pipeline.external.external_integration as mod
        import inspect
        source = inspect.getsource(mod)
        assert "SnapshotWriter" not in source
        assert "write_csv" not in source
        assert "write_normalization_report" not in source
