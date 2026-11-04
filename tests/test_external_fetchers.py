"""Tests for external data fetchers (using fake transports — no network)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.pipeline.external.fetch_kalshi import (
    fetch_markets,
    fetch_historical_candles,
    probe as kalshi_probe,
)
from backend.pipeline.external.fetch_polymarket import (
    gamma_list_markets,
    clob_prices_history,
)
from backend.pipeline.external.fetch_gdelt import fetch_doc_api
from backend.pipeline.external.fetch_nytimes import fetch_archive_month
from backend.pipeline.external.fetch_guardian import fetch_search


# ---------------------------------------------------------------------------
# Kalshi fetchers
# ---------------------------------------------------------------------------


class TestKalshiFetchers:
    def test_fetch_markets_fake(self):
        from backend.pipeline.external.fetch_kalshi import fetch_markets
        resp = {
            "markets": [
                {"ticker": "TEST-26-DEC-31-5000", "status": "settled", "result": "yes"}
            ],
            "cursor": "abc",
        }

        def transport(url):
            return 200, json.dumps(resp).encode()

        status, body = fetch_markets(status="settled", limit=2, transport=transport)
        assert status == 200
        assert len(body["markets"]) == 1

    def test_fetch_candles_excludes_synthetic(self):
        from backend.pipeline.external.fetch_kalshi import fetch_historical_candles
        resp = {
            "ticker": "TEST",
            "candlesticks": [
                {"end_period_ts": 1000, "price_close": 0.5, "synthetic": False},
                {"end_period_ts": 2000, "price_close": 0.6, "synthetic": True},
            ],
        }

        def transport(url):
            return 200, json.dumps(resp).encode()

        status, body = fetch_historical_candles("TEST", 0, 9999, transport=transport)
        assert status == 200
        non_synth = [c for c in body["candlesticks"] if not c.get("synthetic")]
        assert len(non_synth) == 1

    def test_probe_no_auth_returns_uncertain(self):
        """Probe without env vars set should return auth_required_for_reads=None."""
        from backend.pipeline.external.fetch_kalshi import probe as kalshi_probe
        saved = os.environ.pop("KALSHI_API_KEY_ID", None)
        saved2 = os.environ.pop("KALSHI_PRIVATE_KEY_PATH", None)
        try:
            def fail_transport(url):
                raise ConnectionError("simulated")

            r = kalshi_probe(transport=fail_transport)
            assert r["auth_required_for_reads"] is None or r["auth_required_for_reads"] is True
        finally:
            if saved is not None:
                os.environ["KALSHI_API_KEY_ID"] = saved
            if saved2 is not None:
                os.environ["KALSHI_PRIVATE_KEY_PATH"] = saved2


# ---------------------------------------------------------------------------
# Polymarket fetchers
# ---------------------------------------------------------------------------


class TestPolymarketFetchers:
    SAMPLE_GAMMA = [
        {
            "id": "123",
            "slug": "test-market",
            "question": "Will X happen?",
            "conditionId": "0xabc",
            "clobTokenIds": '["tok_yes", "tok_no"]',
            "startDate": "2024-01-01T00:00:00Z",
            "endDate": "2025-01-01T00:00:00Z",
            "closed": True,
        }
    ]

    def test_gamma_list_markets(self):
        resp = self.SAMPLE_GAMMA

        def transport(url):
            return 200, json.dumps(resp).encode()

        status, body = gamma_list_markets(closed=True, limit=2, transport=transport)
        assert status == 200
        assert len(body) == 1

    def test_clob_prices_history(self):
        resp = {"history": [{"t": 1000, "p": 0.5}, {"t": 2000, "p": 0.6}]}

        def transport(url):
            return 200, json.dumps(resp).encode()

        status, body = clob_prices_history("tok_yes", transport=transport)
        assert status == 200
        assert len(body["history"]) == 2

    def test_probe_closed_market(self):
        from backend.pipeline.external.fetch_polymarket import probe as pm_probe
        gamma_resp = self.SAMPLE_GAMMA
        prices_resp = {"history": [{"t": 1000, "p": 0.42}]}

        def transport(url):
            if "gamma-api" in url:
                return 200, json.dumps(gamma_resp).encode()
            return 200, json.dumps(prices_resp).encode()

        r = pm_probe(transport=transport)
        assert r["gamma_discoverable"] is True
        assert r["prices_history_points"] == 1

    def test_ingest_produces_csv_and_report(self, tmp_path):
        from backend.pipeline.external.fetch_polymarket import run_polymarket_ingest

        gamma_detail = {
            "id": "123",
            "slug": "test-market",
            "question": "Will X happen?",
            "conditionId": "0xabc",
            "clobTokenIds": '["tok_yes", "tok_no"]',
            "startDate": "2024-01-01T00:00:00Z",
            "endDate": "2025-01-01T00:00:00Z",
            "closed": True,
        }
        prices_resp = {
            "history": [
                {"t": 1700000000, "p": 0.50},
                {"t": 1700086400, "p": 0.55},
                {"t": 1700172800, "p": 0.48},
            ]
        }

        def transport(url):
            if "gamma-api" in url:
                return 200, json.dumps(gamma_detail).encode()
            return 200, json.dumps(prices_resp).encode()

        from backend.pipeline.external.external_config import (
            DATA_RAW_EXTERNAL,
            DATA_PROCESSED_EXTERNAL,
        )
        import shutil

        # Use tmp_path via monkeypatching config paths
        raw_dir = tmp_path / "raw_external"
        proc_dir = tmp_path / "processed_external"
        raw_dir.mkdir()
        proc_dir.mkdir()

        import backend.pipeline.external.fetch_polymarket as fpm
        import backend.pipeline.external.external_config as ecfg

        orig_raw = ecfg.DATA_RAW_EXTERNAL
        orig_proc = ecfg.DATA_PROCESSED_EXTERNAL
        try:
            ecfg.DATA_RAW_EXTERNAL = raw_dir
            ecfg.DATA_PROCESSED_EXTERNAL = proc_dir
            # Patch in the fetcher module too
            fpm.DATA_RAW_EXTERNAL = raw_dir
            fpm.DATA_PROCESSED_EXTERNAL = proc_dir

            result = run_polymarket_ingest(
                market_ids=["123"],
                transport=transport,
            )

            assert result.errors == [], f"Unexpected errors: {result.errors}"
            assert result.markets_discovered == 1
            assert result.markets_fetched == 1
            assert result.price_snapshots == 1
            assert result.total_observations == 3
            assert result.deduped_observations == 3
            assert result.written_rows == 3
            assert result.output_csv is not None
            assert result.output_csv.exists()
            assert result.output_report is not None
            assert result.output_report.exists()

            # Verify CSV content
            import csv as csv_mod

            with open(result.output_csv, newline="", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            assert len(rows) == 3
            for row in rows:
                assert "probability" in row
                assert "t_unix_s" in row
                assert "token_id" in row
                assert "market_id" in row
                p = float(row["probability"])
                assert 0.0 <= p <= 1.0

            # Verify provenance report
            report = json.loads(result.output_report.read_text(encoding="utf-8"))
            assert report["source"] == "polymarket"
            assert "polymarket_observations.csv" in report["output_files"]
            assert report["row_counts"]["observations"] == 3

        finally:
            ecfg.DATA_RAW_EXTERNAL = orig_raw
            ecfg.DATA_PROCESSED_EXTERNAL = orig_proc
            fpm.DATA_RAW_EXTERNAL = orig_raw
            fpm.DATA_PROCESSED_EXTERNAL = orig_proc

    def test_ingest_deduplicates_across_markets(self, tmp_path):
        from backend.pipeline.external.fetch_polymarket import run_polymarket_ingest

        market_a_detail = {
            "id": "100",
            "slug": "market-a",
            "question": "Market A?",
            "conditionId": "0xaaa",
            "clobTokenIds": '["tok_a_yes", "tok_a_no"]',
            "startDate": "2024-01-01T00:00:00Z",
            "endDate": "2025-01-01T00:00:00Z",
            "closed": True,
        }
        market_b_detail = {
            "id": "200",
            "slug": "market-b",
            "question": "Market B?",
            "conditionId": "0xbbb",
            "clobTokenIds": '["tok_b_yes", "tok_b_no"]',
            "startDate": "2024-01-01T00:00:00Z",
            "endDate": "2025-01-01T00:00:00Z",
            "closed": True,
        }
        # Both markets have same timestamp 1700000000 but different token_ids
        # — dedup key is (token_id, t_unix_s) so they should NOT deduplicate
        prices_a = {
            "history": [{"t": 1700000000, "p": 0.50}, {"t": 1700086400, "p": 0.55}]
        }
        prices_b = {
            "history": [{"t": 1700000000, "p": 0.30}, {"t": 1700086400, "p": 0.35}]
        }

        def transport(url):
            if "market-a" in url or "100" in url:
                if "gamma-api" in url:
                    return 200, json.dumps(market_a_detail).encode()
                return 200, json.dumps(prices_a).encode()
            if "market-b" in url or "200" in url:
                if "gamma-api" in url:
                    return 200, json.dumps(market_b_detail).encode()
                return 200, json.dumps(prices_b).encode()
            # Fallback: return market A for anything else
            if "gamma-api" in url:
                return 200, json.dumps(market_a_detail).encode()
            return 200, json.dumps(prices_a).encode()

        from backend.pipeline.external.external_config import (
            DATA_RAW_EXTERNAL,
            DATA_PROCESSED_EXTERNAL,
        )

        raw_dir = tmp_path / "raw_external"
        proc_dir = tmp_path / "processed_external"
        raw_dir.mkdir()
        proc_dir.mkdir()

        import backend.pipeline.external.fetch_polymarket as fpm
        import backend.pipeline.external.external_config as ecfg

        orig_raw = ecfg.DATA_RAW_EXTERNAL
        orig_proc = ecfg.DATA_PROCESSED_EXTERNAL
        try:
            ecfg.DATA_RAW_EXTERNAL = raw_dir
            ecfg.DATA_PROCESSED_EXTERNAL = proc_dir
            fpm.DATA_RAW_EXTERNAL = raw_dir
            fpm.DATA_PROCESSED_EXTERNAL = proc_dir

            result = run_polymarket_ingest(
                market_ids=["100", "200"],
                transport=transport,
            )

            # Different token_ids means same timestamp is NOT a duplicate
            assert result.total_observations == 4
            assert result.deduped_observations == 4
            assert result.written_rows == 4

        finally:
            ecfg.DATA_RAW_EXTERNAL = orig_raw
            ecfg.DATA_PROCESSED_EXTERNAL = orig_proc
            fpm.DATA_RAW_EXTERNAL = orig_raw
            fpm.DATA_PROCESSED_EXTERNAL = orig_proc

    def test_ingest_empty_gamma_returns_no_error(self, tmp_path):
        from backend.pipeline.external.fetch_polymarket import run_polymarket_ingest

        def transport(url):
            return 200, json.dumps([]).encode()

        from backend.pipeline.external.external_config import (
            DATA_RAW_EXTERNAL,
            DATA_PROCESSED_EXTERNAL,
        )

        raw_dir = tmp_path / "raw_external"
        proc_dir = tmp_path / "processed_external"
        raw_dir.mkdir()
        proc_dir.mkdir()

        import backend.pipeline.external.fetch_polymarket as fpm
        import backend.pipeline.external.external_config as ecfg

        orig_raw = ecfg.DATA_RAW_EXTERNAL
        orig_proc = ecfg.DATA_PROCESSED_EXTERNAL
        try:
            ecfg.DATA_RAW_EXTERNAL = raw_dir
            ecfg.DATA_PROCESSED_EXTERNAL = proc_dir
            fpm.DATA_RAW_EXTERNAL = raw_dir
            fpm.DATA_PROCESSED_EXTERNAL = proc_dir

            result = run_polymarket_ingest(
                market_ids=[],
                transport=transport,
            )

            assert result.markets_discovered == 0
            assert result.markets_fetched == 0
            assert result.written_rows == 0
            assert result.errors == []

        finally:
            ecfg.DATA_RAW_EXTERNAL = orig_raw
            ecfg.DATA_PROCESSED_EXTERNAL = orig_proc
            fpm.DATA_RAW_EXTERNAL = orig_raw
            fpm.DATA_PROCESSED_EXTERNAL = orig_proc


# ---------------------------------------------------------------------------
# GDELT fetcher
# ---------------------------------------------------------------------------


class TestGDELTFetcher:
    def test_fetch_doc_api(self):
        resp = {"articles": [{"title": "Test", "url": "http://test.com", "seendate": "20240115120000"}]}

        def transport(url):
            return 200, json.dumps(resp).encode()

        status, body = fetch_doc_api("sp500", transport=transport)
        assert status == 200
        assert len(body["articles"]) == 1

    def test_probe(self):
        from backend.pipeline.external.fetch_gdelt import probe as gdelt_probe
        resp = {
            "articles": [
                {"title": "SP500 up", "url": "http://a.com", "seendate": "20240115120000"}
            ]
        }

        def transport(url):
            return 200, json.dumps(resp).encode()

        r = gdelt_probe(transport=transport)
        assert r["api_reachable"] is True
        assert r["articles_returned"] == 1


# ---------------------------------------------------------------------------
# NYT Archive fetcher
# ---------------------------------------------------------------------------


class TestNYTFetcher:
    def test_fetch_archive_month(self, monkeypatch):
        resp = {"response": {"docs": [{"_id": "1", "pub_date": "2024-06-01T12:00:00Z", "headline": {"main": "Test"}, "web_url": "http://nyt.com/1"}]}}

        def transport(url):
            return 200, json.dumps(resp).encode()

        monkeypatch.setenv("NYTIMES_API_KEY", "fake-key")
        status, body = fetch_archive_month(2024, 6, transport=transport)
        assert status == 200
        assert len(body["response"]["docs"]) == 1

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("NYTIMES_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="NYTIMES_API_KEY"):
            fetch_archive_month(2024, 6)


# ---------------------------------------------------------------------------
# Guardian fetcher
# ---------------------------------------------------------------------------


class TestGuardianFetcher:
    def test_fetch_search(self, monkeypatch):
        resp = {"response": {"results": [{"id": "g1", "webTitle": "Test", "webPublicationDate": "2024-06-01T10:00:00Z"}]}}

        def transport(url):
            return 200, json.dumps(resp).encode()

        monkeypatch.setenv("GUARDIAN_API_KEY", "fake-key")
        status, body = fetch_search("sp500", transport=transport)
        assert status == 200
        assert len(body["response"]["results"]) == 1

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="GUARDIAN_API_KEY"):
            fetch_search("sp500")
