"""Tests for external data normalization: parsers, dedup, validation, CSV output."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.pipeline.external.normalize_external import (
    parse_kalshi_markets,
    parse_kalshi_candles,
    parse_polymarket_markets,
    parse_polymarket_prices,
    parse_gdelt_articles,
    parse_nytimes_archive,
    parse_guardian_search,
    split_kalshi_resolution,
    dedup_rows,
    validate_probability_rows,
    write_csv,
    compute_csv_sha256,
)


# ---------------------------------------------------------------------------
# Kalshi parsers
# ---------------------------------------------------------------------------


class TestKalshiParsers:
    MARKET_RAW = {
        "markets": [
            {
                "ticker": "TST-26-DEC-31-5000",
                "event_ticker": "TST-26-DEC-31",
                "market_type": "binary",
                "status": "finalized",
                "open_time": "2026-01-01T00:00:00Z",
                "close_time": "2026-12-31T20:00:00Z",
                "settlement_ts": "2026-12-31T21:00:00Z",
                "settlement_value_dollars": "1.000000",
                "result": "yes",
                "can_close_early": False,
                "created_time": "2025-10-01T00:00:00Z",
                "updated_time": "2026-12-31T20:01:00Z",
                "expiration_value": "5200",
                "occurrence_datetime": "2026-12-31T20:00:00Z",
                "latest_expiration_time": "2026-12-31T20:05:00Z",
                "notional_value_dollars": "100.00",
            }
        ]
    }

    CANDLE_RAW = {
        "ticker": "TST",
        "candlesticks": [
            {"end_period_ts": 1700000000, "price_close": 0.55, "volume_fp": "100"},
            {"end_period_ts": 1700001440, "price_close": 0.60, "volume_fp": "200", "synthetic": True},
        ],
    }

    def test_parse_markets(self):
        rows = parse_kalshi_markets(self.MARKET_RAW)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "TST-26-DEC-31-5000"
        assert rows[0]["status"] == "finalized"

    def test_parse_candles_excludes_synthetic(self):
        rows = parse_kalshi_candles(self.CANDLE_RAW)
        assert len(rows) == 1
        assert rows[0]["candle_end_unix_s"] == 1700000000

    def test_parse_candles_empty(self):
        rows = parse_kalshi_candles({"candlesticks": []})
        assert rows == []

    def test_split_resolution(self):
        lc_rows, res_rows = split_kalshi_resolution(parse_kalshi_markets(self.MARKET_RAW))
        assert len(lc_rows) == 1
        assert "ticker" in lc_rows[0]
        assert "settlement_value_dollars" not in lc_rows[0]
        assert res_rows[0]["result"] == "yes"
        assert res_rows[0]["settlement_ts"] != ""


# ---------------------------------------------------------------------------
# Polymarket parsers
# ---------------------------------------------------------------------------


class TestPolymarketParsers:
    MARKET_RAW = [
        {
            "id": "559",
            "slug": "test-q",
            "question": "Test?",
            "conditionId": "0xabc",
            "clobTokenIds": '["tok_y", "tok_n"]',
            "startDate": "2024-01-01T00:00:00Z",
            "closed": True,
        }
    ]

    PRICES_RAW = {"history": [{"t": 100, "p": 0.42}, {"t": 200, "p": 0.55}]}

    def test_parse_markets(self):
        rows = parse_polymarket_markets(self.MARKET_RAW)
        assert len(rows) == 1
        assert rows[0]["yes_token_id"] == "tok_y"
        assert rows[0]["no_token_id"] == "tok_n"
        assert rows[0]["condition_id"] == "0xabc"

    def test_parse_prices(self):
        rows = parse_polymarket_prices(self.PRICES_RAW)
        assert len(rows) == 2
        assert rows[0]["probability"] == 0.42

    def test_parse_markets_outcomes_dict(self):
        raw = [{"id": "1", "question": "Q", "outcomes": {"yes": {"token_id": "a"}, "no": {"token_id": "b"}}}]
        rows = parse_polymarket_markets(raw)
        assert rows[0]["yes_token_id"] == "a"


# ---------------------------------------------------------------------------
# News parsers
# ---------------------------------------------------------------------------


class TestGDDELTParser:
    def test_parse_articles(self):
        raw = {
            "articles": [
                {"title": "S&P 500 up", "url": "http://a.com", "seendate": "20240115120000", "domain": "a.com"},
            ]
        }
        rows = parse_gdelt_articles(raw)
        assert len(rows) == 1
        assert rows[0]["source"] == "gdelt"
        assert "2024-01-15" in rows[0]["published_at_utc"]
        assert rows[0]["domain"] == "a.com"

    def test_parse_empty(self):
        assert parse_gdelt_articles({}) == []


class TestNYTArchiveParser:
    def test_parse_archive(self):
        raw = {
            "response": {
                "docs": [
                    {
                        "_id": "nyt-1",
                        "pub_date": "2024-06-01T14:30:00Z",
                        "headline": {"main": "Markets Rally"},
                        "web_url": "http://nyt.com/article1",
                        "section_name": "Business",
                        "snippet": "Markets rose today...",
                        "keywords": [{"name": "subject", "value": "stocks"}],
                        "byline": {"original": "By John Doe"},
                    }
                ]
            }
        }
        rows = parse_nytimes_archive(raw)
        assert len(rows) == 1
        assert rows[0]["source"] == "nytimes"
        assert rows[0]["section"] == "Business"
        assert "stocks" in rows[0]["keywords"]


class TestGuardianParser:
    def test_parse_search(self):
        raw = {
            "response": {
                "results": [
                    {
                        "id": "business/2024/jun/01/test",
                        "webTitle": "Markets up",
                        "webUrl": "https://theguardian.com/test",
                        "webPublicationDate": "2024-06-01T10:00:00Z",
                        "sectionName": "Business",
                        "fields": {"bodyText": "The S&P 500 rose..."},
                    }
                ]
            }
        }
        rows = parse_guardian_search(raw)
        assert len(rows) == 1
        assert rows[0]["source"] == "guardian"
        assert rows[0]["section"] == "Business"
        assert "S&P 500" in rows[0]["snippet"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    def test_basic_dedup(self):
        rows = [
            {"url": "http://a.com", "published_at_utc": "2024-01-01T00:00:00Z", "title": "A"},
            {"url": "http://a.com", "published_at_utc": "2024-01-01T00:00:00Z", "title": "A dup"},
            {"url": "http://b.com", "published_at_utc": "2024-01-01T00:00:00Z", "title": "B"},
        ]
        result = dedup_rows(rows, ("url", "published_at_utc"))
        assert len(result) == 2
        assert result[0]["title"] == "A"
        assert result[1]["title"] == "B"

    def test_dedup_empty_key_treated_as_string(self):
        rows = [{"url": "", "x": 1}, {"url": "", "x": 2}]
        result = dedup_rows(rows, ("url",))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_probabilities(self):
        rows = [{"probability": 0.0}, {"probability": 0.5}, {"probability": 1.0}]
        validate_probability_rows(rows)

    def test_out_of_range_raises(self):
        rows = [{"probability": 1.5}]
        with pytest.raises(ValueError, match="out-of-range"):
            validate_probability_rows(rows)

    def test_missing_field_skipped(self):
        rows = [{"other_field": "val"}]
        validate_probability_rows(rows)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestCSVOutput:
    def test_deterministic_csv(self, tmp_path):
        rows = [
            {"col_b": 2, "col_a": 1},
            {"col_b": 4, "col_a": 3},
        ]
        p1 = tmp_path / "out1.csv"
        p2 = tmp_path / "out2.csv"
        write_csv(rows, p1)
        write_csv(rows, p2)
        assert p1.read_bytes() == p2.read_bytes()
        assert compute_csv_sha256(p1) == compute_csv_sha256(p2)

    def test_sorted_columns(self, tmp_path):
        rows = [{"z": 1, "a": 2, "m": 3}]
        p = tmp_path / "sorted.csv"
        write_csv(rows, p)
        lines = p.read_text().strip().split("\n")
        assert lines[0] == "a,m,z"

    def test_empty_rows(self, tmp_path):
        p = tmp_path / "empty.csv"
        write_csv([], p)
        assert p.read_text() == ""
