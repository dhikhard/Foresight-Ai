"""Feature Set V2 generator CLI tests.

Reproducibility, schema determinism, and failure-mode coverage for
backend.pipeline.run_features_v2. Expected values are derived independently
(inline arithmetic on literal inputs, hashlib computed in the test); the
generator is never used to produce expectations.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from backend.pipeline.features_v2 import FEATURE_COLUMNS_V2
from backend.pipeline.run_features_v2 import generate, load_clean_rows

FIELDNAMES = ["date", "open", "high", "low", "close", "volume",
              "symbol", "asset_name", "asset_type", "region"]

# 30 varied positive closes: enough for ma_dist_20/rsi/pctb maturity,
# deliberately fewer than 60 so long-window columns serialize as empty.
CLOSES = [100.0 + (i % 7) * 1.5 - 2 + (i * 37 % 11) * 0.8 for i in range(30)]


def make_clean_rows(closes):
    rows = []
    year, month, day = 2024, 1, 1
    for i, c in enumerate(closes):
        rows.append({
            "date": f"{year}-{month:02d}-{day:02d}",
            "open": f"{c * 0.995:.4f}", "high": f"{c * 1.01:.4f}",
            "low": f"{c * 0.99:.4f}", "close": repr(float(c)),
            "volume": str(1000 + i), "symbol": "^GSPC",
            "asset_name": "S&P500", "asset_type": "Stock Index",
            "region": "Global",
        })
        day += 1
        if day > 28:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return rows


def write_clean(tmp_path: Path, closes=CLOSES) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "sp500_clean.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(make_clean_rows(closes))
    return path


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def expected_dates(n):
    year, month, day = 2024, 1, 1
    dates = []
    for _ in range(n):
        dates.append(f"{year}-{month:02d}-{day:02d}")
        day += 1
        if day > 28:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return dates


# ---------- reproducibility ----------
# Real overwrite scenario: same input, same flags, SAME output directory.

def test_repeated_generation_is_byte_identical(tmp_path):
    clean_path = write_clean(tmp_path / "in")
    out = tmp_path / "out"
    first = generate(clean_path, out)
    bytes_after_first_run = Path(first["output"]["path"]).read_bytes()
    second = generate(clean_path, out)
    assert Path(second["output"]["path"]).read_bytes() == bytes_after_first_run


def test_report_content_stable_across_runs_except_timestamp(tmp_path):
    clean_path = write_clean(tmp_path / "in")
    out = tmp_path / "out"

    def load_report():
        report = generate(clean_path, out)
        with open(report["output"]["report_path"], encoding="utf-8") as fh:
            return json.load(fh)

    r1, r2 = load_report(), load_report()
    # Non-vacuity: the timestamp field must actually exist before we exclude it.
    assert "generated_at_utc" in r1 and "generated_at_utc" in r2
    r1.pop("generated_at_utc")
    r2.pop("generated_at_utc")
    # Every substantive field must match: provenance hash, schema, rows,
    # column order, declared/observed warm-up, none counts, notes.
    assert r1 == r2


# ---------- schema / alignment ----------

def test_schema_is_date_plus_16_columns_in_contract_order(tmp_path):
    report = generate(write_clean(tmp_path / "in"), tmp_path / "out")
    assert report["output"]["columns"] == ["date", *FEATURE_COLUMNS_V2]
    with open(report["output"]["path"], newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == ["date", *FEATURE_COLUMNS_V2]


def test_row_count_and_date_alignment_preserved(tmp_path):
    report = generate(write_clean(tmp_path / "in"), tmp_path / "out")
    rows = read_rows(report["output"]["path"])
    assert len(rows) == len(CLOSES) == report["output"]["rows"]
    assert [r["date"] for r in rows] == expected_dates(len(CLOSES))


# ---------- independent value verification ----------
# Spot checks computed by hand from the literal CLOSES list; generator
# outputs are never used to derive expectations.

def test_spot_values_match_manual_arithmetic_after_roundtrip(tmp_path):
    report = generate(write_clean(tmp_path / "in"), tmp_path / "out")
    rows = read_rows(report["output"]["path"])
    t = 25
    # mom_5[t] = CLOSES[t]/CLOSES[t-5] - 1
    expected_mom = CLOSES[25] / CLOSES[20] - 1
    assert float(rows[t]["mom_5"]) == pytest.approx(expected_mom, rel=1e-9)
    # ma_dist_20[t] uses the trailing window CLOSES[6..25].
    window = CLOSES[6:26]
    expected_dist = CLOSES[25] / (math.fsum(window) / 20) - 1
    assert float(rows[t]["ma_dist_20"]) == pytest.approx(expected_dist, rel=1e-9)


def test_warmup_rows_serialize_as_empty_fields(tmp_path):
    report = generate(write_clean(tmp_path / "in"), tmp_path / "out")
    rows = read_rows(report["output"]["path"])
    # Hardcoded warm-up expectations (independent of module constants).
    declared = {"mom_5": 5, "ma_dist_20": 19, "rsi_14": 14,
                "bollinger_pctb_20": 19}
    for column, warmup in declared.items():
        empties = sum(1 for r in rows if r[column] == "")
        assert empties == warmup, column
        assert all(r[column] != "" for r in rows[warmup:]), column


def test_long_window_columns_empty_when_history_insufficient(tmp_path):
    report = generate(write_clean(tmp_path / "in"), tmp_path / "out")
    rows = read_rows(report["output"]["path"])
    assert all(r["vol_60_annualized"] == "" for r in rows)
    assert all(r["vol_ratio_20_60"] == "" for r in rows)
    assert any(r["rsi_14"] != "" for r in rows)   # anti-vacuity: short windows live


# ---------- provenance report ----------

def test_report_records_input_sha256_and_warmup_audit(tmp_path):
    clean_path = write_clean(tmp_path / "in")
    report = generate(clean_path, tmp_path / "out")
    expected_sha = hashlib.sha256(clean_path.read_bytes()).hexdigest()
    assert report["input"]["sha256"] == expected_sha
    observed = report["feature_warmup"]["observed_first_valid"]
    assert observed["rsi_14"] == 14
    assert observed["vol_60_annualized"] is None      # never matured in 30 rows
    assert report["none_counts"]["vol_60_annualized"] == len(CLOSES)
    mismatches = [n for n in report["notes"] if "first valid" in n]
    assert any("vol_60_annualized" in n for n in mismatches)


# ---------- loud failure modes ----------

def test_missing_close_value_fails_loudly(tmp_path):
    rows = make_clean_rows(CLOSES)
    rows[10]["close"] = ""
    path = tmp_path / "broken.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=r"row 10.*empty close"):
        load_clean_rows(path)


def test_non_numeric_close_fails_loudly(tmp_path):
    rows = make_clean_rows(CLOSES)
    rows[5]["close"] = "N/A"
    path = tmp_path / "broken.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="non-numeric close"):
        generate(path, tmp_path / "out")


def test_header_only_series_fails_loudly(tmp_path):
    path = write_clean(tmp_path, closes=[])
    with pytest.raises(ValueError, match="no data rows"):
        generate(path, tmp_path / "out")


def test_missing_input_column_fails_loudly(tmp_path):
    path = tmp_path / "mangled.csv"
    path.write_text(
        "date,open,high,low\n2024-01-02,10,12,9\n2024-01-03,11,13,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_clean_rows(path)


def test_single_row_series_generates_all_empty_features(tmp_path):
    path = write_clean(tmp_path / "one", CLOSES[:1])
    report = generate(path, tmp_path / "out")
    rows = read_rows(report["output"]["path"])
    assert len(rows) == 1
    assert all(rows[0][c] == "" for c in FEATURE_COLUMNS_V2)


def test_load_clean_rows_parses_numerics_and_keeps_metadata(tmp_path):
    rows = load_clean_rows(write_clean(tmp_path))
    assert rows[0]["close"] == pytest.approx(CLOSES[0])
    assert isinstance(rows[3]["volume"], float)
    assert rows[3]["symbol"] == "^GSPC"
    assert rows[3]["date"] == "2024-01-04"
