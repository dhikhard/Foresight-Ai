"""Tests for temporal alignment: DST, boundaries, as-of join, news aggregation, leakage."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from backend.pipeline.external.temporal_alignment import (
    trading_day_close_utc,
    prev_trading_day_close,
    is_eligible,
    filter_point_in_time,
    asof_join,
    aggregate_news_by_day,
    load_trading_calendar,
    _parse_to_utc,
)
from backend.pipeline.external.external_config import SP500_CLEAN_CSV
from pathlib import Path


# ---------------------------------------------------------------------------
# DST / close-time computation
# ---------------------------------------------------------------------------


class TestTradingDayCloseUTC:
    def test_winter_standard_time(self):
        # 2024-01-15 is a Monday; 16:00 ET = 21:00 UTC (EST = UTC-5)
        d = date(2024, 1, 15)
        utc = trading_day_close_utc(d)
        assert utc.hour == 21
        assert utc.minute == 0
        assert utc.tzinfo is not None

    def test_summer_daylight_time(self):
        # 2024-07-15 is a Monday; 16:00 ET = 20:00 UTC (EDT = UTC-4)
        d = date(2024, 7, 15)
        utc = trading_day_close_utc(d)
        assert utc.hour == 20
        assert utc.minute == 0

    def test_spring_forward_transition(self):
        # 2024-03-10 DST spring forward (2:00 AM -> 3:00 AM)
        # Trading day close on March 8 (Friday before) is EST
        d_mar8 = date(2024, 3, 8)
        utc_mar8 = trading_day_close_utc(d_mar8)
        assert utc_mar8.hour == 21  # EST

        # Trading day close on March 11 (Monday after) is EDT
        d_mar11 = date(2024, 3, 11)
        utc_mar11 = trading_day_close_utc(d_mar11)
        assert utc_mar11.hour == 20  # EDT

    def test_fall_back_transition(self):
        # 2024-11-03 DST fall back
        d_nov1 = date(2024, 11, 1)  # Friday — EDT
        utc_nov1 = trading_day_close_utc(d_nov1)
        assert utc_nov1.hour == 20  # EDT

        d_nov4 = date(2024, 11, 4)  # Monday — EST
        utc_nov4 = trading_day_close_utc(d_nov4)
        assert utc_nov4.hour == 21  # EST

    def test_utc_conversion_consistency(self):
        d = date(2024, 6, 15)
        utc = trading_day_close_utc(d)
        # Summer: 16:00 EDT = 20:00 UTC
        assert utc.hour == 20
        assert utc.minute == 0
        # Winter: 16:00 EST = 21:00 UTC
        d2 = date(2024, 1, 15)
        utc2 = trading_day_close_utc(d2)
        assert utc2.hour == 21


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


class TestParseToUTC:
    def test_iso_z(self):
        dt = _parse_to_utc("2024-06-15T14:30:00Z")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.year == 2024

    def test_iso_offset(self):
        dt = _parse_to_utc("2024-06-15T14:30:00+05:00")
        assert dt is not None

    def test_unix_timestamp_int(self):
        dt = _parse_to_utc(1700000000)
        assert dt is not None
        assert dt.year >= 2023

    def test_unix_timestamp_float(self):
        dt = _parse_to_utc(1700000000.5)
        assert dt is not None

    def test_gdelt_seendate(self):
        dt = _parse_to_utc("20240115120000")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_empty_string(self):
        assert _parse_to_utc("") is None

    def test_none(self):
        assert _parse_to_utc(None) is None


# ---------------------------------------------------------------------------
# Point-in-time eligibility
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_exact_at_close_included(self):
        d = date(2024, 7, 15)
        cutoff = trading_day_close_utc(d)
        # Exactly at cutoff
        assert is_eligible(cutoff.isoformat(), cutoff) is True

    def test_one_second_after_excluded(self):
        d = date(2024, 7, 15)
        cutoff = trading_day_close_utc(d)
        after = cutoff + timedelta(seconds=1)
        assert is_eligible(after.isoformat(), cutoff) is False

    def test_one_second_before_included(self):
        d = date(2024, 7, 15)
        cutoff = trading_day_close_utc(d)
        before = cutoff - timedelta(seconds=1)
        assert is_eligible(before.isoformat(), cutoff) is True

    def test_filter_point_in_time(self):
        d = date(2024, 7, 15)
        cutoff = trading_day_close_utc(d)
        rows = [
            {"published_at": (cutoff - timedelta(hours=1)).isoformat()},
            {"published_at": (cutoff + timedelta(hours=1)).isoformat()},
            {"published_at": cutoff.isoformat()},
        ]
        result = filter_point_in_time(rows, cutoff, "published_at")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# As-of join
# ---------------------------------------------------------------------------


class TestAsofJoin:
    def test_basic_asof(self):
        cal = [date(2024, 7, 15), date(2024, 7, 16)]
        obs = [
            {"t": trading_day_close_utc(date(2024, 7, 14)).isoformat(), "val": "before"},
            {"t": trading_day_close_utc(date(2024, 7, 15)).isoformat(), "val": "day1"},
            {"t": trading_day_close_utc(date(2024, 7, 15,)).isoformat(), "val": "day1b"},
        ]
        result = asof_join(obs, cal, "t")
        assert len(result) == 2
        assert result[0]["val"] == "day1b"  # last match for day1
        assert result[1]["val"] == "day1b"  # same obs carries forward to day2

    def test_empty_observations(self):
        cal = [date(2024, 7, 15)]
        result = asof_join([], cal, "t")
        assert len(result) == 1
        assert "trading_date" in result[0]

    def test_post_close_excluded(self):
        cal = [date(2024, 7, 15)]
        close = trading_day_close_utc(date(2024, 7, 15))
        obs = [{"t": (close + timedelta(hours=1)).isoformat(), "val": "future"}]
        result = asof_join(obs, cal, "t")
        assert "val" not in result[0]  # no match

    def test_staleness_flag(self):
        cal = [date(2024, 7, 15)]
        close = trading_day_close_utc(date(2024, 7, 15))
        obs = [{"t": (close - timedelta(hours=48)).isoformat(), "val": "old"}]
        result = asof_join(obs, cal, "t")
        assert result[0]["is_stale"] is True
        assert result[0]["staleness_hours"] >= 48.0


# ---------------------------------------------------------------------------
# News aggregation
# ---------------------------------------------------------------------------


class TestNewsAggregation:
    def test_basic_aggregation(self):
        cal = [date(2024, 7, 15), date(2024, 7, 16)]
        c1 = trading_day_close_utc(date(2024, 7, 15))
        c2 = trading_day_close_utc(date(2024, 7, 16))

        articles = [
            {"article_id": "a1", "published_at": (c1 - timedelta(hours=2)).isoformat()},
            {"article_id": "a2", "published_at": (c1 + timedelta(hours=1)).isoformat()},
            {"article_id": "a3", "published_at": (c2 - timedelta(hours=1)).isoformat()},
            {"article_id": "a4", "published_at": (c2 + timedelta(hours=1)).isoformat()},
        ]
        result = aggregate_news_by_day(articles, cal, "published_at")
        assert len(result) == 2
        # Day 1: a1 only (a2 is at day1_close + 1h, post-close to day1)
        assert result[0]["article_count"] == 1
        # Day 2: a2 + a3 (both in (day1_close, day2_close])
        assert result[1]["article_count"] == 2

    def test_empty_articles(self):
        cal = [date(2024, 7, 15)]
        result = aggregate_news_by_day([], cal, "published_at")
        assert result[0]["article_count"] == 0


# ---------------------------------------------------------------------------
# Leakage guard: feature time can never exceed cutoff
# ---------------------------------------------------------------------------


class TestLeakageGuards:
    def test_cannot_use_future_data(self):
        """Any post-close data must be excluded by the alignment contract."""
        d = date(2024, 7, 15)
        cutoff = trading_day_close_utc(d)
        future = cutoff + timedelta(seconds=1)
        assert is_eligible(future.isoformat(), cutoff) is False

    def test_all_rows_eligible_before_cutoff(self):
        """All observations before cutoff are eligible (inclusive)."""
        d = date(2024, 7, 15)
        cutoff = trading_day_close_utc(d)
        for delta_h in [24, 12, 6, 1, 0]:
            ts = (cutoff - timedelta(hours=delta_h)).isoformat()
            assert is_eligible(ts, cutoff) is True


# ---------------------------------------------------------------------------
# Real calendar loading (integration)
# ---------------------------------------------------------------------------


class TestCalendarIntegration:
    @pytest.mark.skipif(
        not SP500_CLEAN_CSV.exists(),
        reason="sp500_clean.csv not present",
    )
    def test_load_calendar(self):
        cal = load_trading_calendar(SP500_CLEAN_CSV)
        assert len(cal) > 4000
        assert cal[0] < cal[-1]
        # Every date should produce a valid UTC close
        utc_closes = [trading_day_close_utc(d) for d in cal[:5]]
        assert all(utc.tzinfo is not None for utc in utc_closes)
        # Closes should be monotonically increasing
        assert all(utc_closes[i] < utc_closes[i + 1] for i in range(len(utc_closes) - 1))
