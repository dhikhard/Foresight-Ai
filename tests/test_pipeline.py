import csv
import json
import math
import statistics
import zipfile
from pathlib import Path

import pytest

from backend.pipeline.clean import clean_ohlcv, extract_symbol
from backend.pipeline.config import PipelineConfig
from backend.pipeline.features import add_features
from backend.pipeline.ingest import load_raw_rows
from backend.pipeline.run_preprocess import run
from backend.pipeline.splits import chronological_split
from backend.pipeline.targets import add_future_targets


def make_row(date, close, *, open_=None, high=None, low=None, volume="1000",
             symbol="^GSPC", name="S&P500", atype="Stock Index", region="Global"):
    o = open_ if open_ is not None else close
    h = high if high is not None else max(o, close)
    l = low if low is not None else min(o, close)
    return {
        "date": date, "open": str(o), "high": str(h), "low": str(l),
        "close": str(close), "volume": str(volume), "symbol": symbol,
        "asset_name": name, "asset_type": atype, "region": region,
    }


def make_series(closes, start_day=1):
    rows = []
    year, month, day = 2024, 1, start_day
    for c in closes:
        rows.append(make_row(f"{year}-{month:02d}-{day:02d}", c))
        day += 1
        if day > 28:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return rows


def build_features(rows, ma_windows=(20,), volatility_window=20):
    """Mirror the production flow: clean (str->float) then engineer features."""
    cleaned, _ = clean_ohlcv(rows, PipelineConfig())
    return add_features(cleaned, ma_windows=ma_windows,
                        volatility_window=volatility_window,
                        annualization_factor=math.sqrt(252))


# ---------- extraction ----------

def test_extract_symbol_filters_only_requested():
    rows = [
        make_row("2024-01-01", 5000),
        make_row("2024-01-01", 16000, symbol="^IXIC", name="NASDAQ"),
        make_row("2024-01-02", 5010),
    ]
    config = PipelineConfig()
    selected = extract_symbol(rows, config)
    assert len(selected) == 2
    assert all(r["symbol"] == "^GSPC" for r in selected)


# ---------- cleaning ----------

def test_clean_sorts_chronologically_and_keeps_metadata():
    rows = [make_row("2024-01-03", 5002), make_row("2024-01-02", 5001)]
    cleaned, report = clean_ohlcv(rows, PipelineConfig())
    assert [r["date"] for r in cleaned] == ["2024-01-02", "2024-01-03"]
    assert cleaned[0]["asset_name"] == "S&P500"
    assert report.rows_removed == 0


def test_clean_drops_duplicate_dates_keeping_first():
    rows = [make_row("2024-01-02", 5001), make_row("2024-01-02", 9999)]
    cleaned, report = clean_ohlcv(rows, PipelineConfig())
    assert len(cleaned) == 1
    assert cleaned[0]["close"] == 5001.0
    assert report.duplicates_found == 1


def test_clean_removes_unparseable_prices_with_log():
    bad = make_row("2024-01-02", 5001)
    bad["close"] = "N/A"
    good = make_row("2024-01-03", 5002)
    cleaned, report = clean_ohlcv([bad, good], PipelineConfig())
    assert len(cleaned) == 1
    assert report.rows_removed == 1
    assert "missing or non-numeric OHLC price" in report.removal_log[0]["reason"]


def test_clean_logs_but_keeps_ohlc_violation():
    bad = make_row("2024-01-02", 100, high=101, low=105)  # low > high violation
    cleaned, report = clean_ohlcv([bad], PipelineConfig())
    assert len(cleaned) == 1
    assert report.ohlc_violations_logged == 1
    assert report.rows_removed == 0


def test_negative_prices_are_logged_not_deleted():
    row = make_row("2020-04-20", -37.63, high=17.85, low=-40.32, volume="247947")
    cleaned, report = clean_ohlcv([row], PipelineConfig())
    assert len(cleaned) == 1
    assert report.negative_price_rows_logged == 1


# ---------- features ----------

def test_return_and_log_return_exact():
    out = build_features(make_series([100.0, 110.0, 99.0]))
    assert out[0]["return"] is None
    assert out[1]["return"] == pytest.approx(0.10)
    assert out[2]["return"] == pytest.approx(99.0 / 110.0 - 1)
    assert out[1]["log_return"] == pytest.approx(math.log(110.0 / 100.0))


def test_moving_average_matches_manual_computation():
    closes = [float(i + 1) for i in range(10)]  # 1..10
    out = build_features(make_series(closes), ma_windows=(3,))
    assert out[0]["ma_3"] is None
    assert out[1]["ma_3"] is None
    assert out[2]["ma_3"] == pytest.approx(2.0)
    assert out[9]["ma_3"] == pytest.approx(9.0)


def test_rolling_volatility_matches_manual_calculation():
    closes = [100.0, 102.0, 101.0, 104.0]
    out = build_features(make_series(closes), volatility_window=3)
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(3)]
    expected_last = (
        math.sqrt(((rets[-3] - (sum(rets) / 3)) ** 2
                   + (rets[-2] - (sum(rets) / 3)) ** 2
                   + (rets[-1] - (sum(rets) / 3)) ** 2) / 2)
        * math.sqrt(252)
    )
    assert out[1]["vol_20_annualized"] is None  # needs 3 returns, only 1 so far
    assert out[3]["vol_20_annualized"] == pytest.approx(expected_last)


def test_drawdown_tracks_previous_peak():
    out = build_features(make_series([100.0, 120.0, 90.0, 130.0]))
    assert out[0]["rolling_peak"] == 100.0
    assert out[2]["rolling_peak"] == 120.0
    assert out[2]["drawdown"] == pytest.approx(90.0 / 120.0 - 1.0)
    assert out[3]["drawdown"] == pytest.approx(0.0)


def test_range_features_exact():
    row = make_row("2024-01-01", 101, open_=98.0, high=103.0, low=97.0)
    out = build_features([row])
    assert out[0]["daily_range"] == pytest.approx((103.0 - 97.0) / 101.0)
    assert out[0]["open_close_return"] == pytest.approx((101.0 - 98.0) / 98.0)


def test_warmup_rows_are_none_not_zero_or_nan_string():
    rows = make_series([100.0 + i for i in range(5)])
    out = build_features(rows, ma_windows=(200,), volatility_window=200)
    assert all(r["ma_200"] is None for r in out)
    assert all(r["vol_20_annualized"] is None for r in out)


# ---------- look-ahead bias ----------

LOOKAHEAD_COLUMNS = ("return", "log_return", "ma_20", "ma_50", "ma_200",
                     "vol_20_annualized", "rolling_peak", "drawdown",
                     "daily_range", "open_close_return")


@pytest.mark.parametrize("index", [30, 120, 210, 255])
def test_no_lookahead_future_values_do_not_change_past_features(index):
    closes = [100.0 + (i % 7) * 1.5 - 2 + (i * 37 % 11) * 0.8 for i in range(260)]
    full = build_features(make_series(closes), ma_windows=(20, 50, 200),
                          volatility_window=20)
    truncated = build_features(make_series(closes[: index + 1]),
                               ma_windows=(20, 50, 200), volatility_window=20)
    for col in LOOKAHEAD_COLUMNS:
        assert full[index][col] == truncated[index][col], f"leakage in {col} at {index}"
    # Guard against a fully vacuous pass: these two columns are guaranteed
    # populated at every chosen index, so at least some comparisons above
    # compare real numbers rather than None == None.
    assert truncated[index]["ma_20"] is not None
    assert truncated[index]["vol_20_annualized"] is not None


# ---------- splitting ----------

def test_chronological_split_boundaries_respected():
    rows = [{"date": d} for d in ("2018-12-31", "2019-01-01", "2022-12-31",
                                  "2023-01-01", "2024-05-01")]
    parts = chronological_split(rows, "2019-01-01", "2023-01-01")
    assert [r["date"] for r in parts["train"]] == ["2018-12-31"]
    assert [r["date"] for r in parts["validation"]] == ["2019-01-01", "2022-12-31"]
    assert [r["date"] for r in parts["test"]] == ["2023-01-01", "2024-05-01"]
    assert sum(len(v) for v in parts.values()) == len(rows)


# ---------- end-to-end reproducibility ----------

RAW_CSV = "date,open,high,low,close,volume,symbol,asset_name,asset_type,region\n" + "".join(
    f"2024-01-{d:02d},10,12,9,{11 + d * 0.5},1000,^GSPC,S&P500,Stock Index,Global\n".replace(
        f"{11 + d * 0.5}", f"{11 + d * 0.5}"
    )
    for d in range(1, 6)
) + "2024-01-01,5,6,4,5.5,2000,^IXIC,NASDAQ,Stock Index,Global\n"


def test_pipeline_run_is_reproducible_byte_for_byte(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "raw.csv").write_text(RAW_CSV, encoding="utf-8")

    outs = []
    for i in range(2):
        processed = tmp_path / f"out{i}"
        config = PipelineConfig(raw_dir=raw_dir, processed_dir=processed)
        payload = run(config)
        outs.append(processed)

        features_file = Path(payload["outputs"]["features"]["path"])
        with open(features_file, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 5
        assert rows[0]["symbol"] == "^GSPC"

    names = ["sp500_clean.csv", "sp500_features.csv", "sp500_features_train.csv",
             "sp500_features_validation.csv", "sp500_features_test.csv"]
    for name in names:
        a = (outs[0] / name).read_bytes()
        b = (outs[1] / name).read_bytes()
        assert a == b, f"{name} differs between runs"


# ---------- targets: independent verification ----------
# Expected values below are derived inside the tests using independent
# arithmetic (subtraction form of returns, statistics.stdev for volatility).
# The implementation's ratio form and _sample_std are never used for
# expectations.

def target_rows(closes):
    rows = [{"date": f"2024-01-{i + 1:02d}", "close": c} for i, c in enumerate(closes)]
    return add_future_targets(rows)


LINEAR_CLOSES = [1000.0 + 5.0 * i for i in range(30)]


def test_target_return_1d_independently_verified():
    out = target_rows(LINEAR_CLOSES)
    for t in range(len(LINEAR_CLOSES) - 1):
        expected = (LINEAR_CLOSES[t + 1] - LINEAR_CLOSES[t]) / LINEAR_CLOSES[t]
        assert out[t]["target_return_1d"] == pytest.approx(expected), f"t={t}"


def test_target_return_5d_independently_verified():
    out = target_rows(LINEAR_CLOSES)
    for t in range(len(LINEAR_CLOSES) - 5):
        expected = (LINEAR_CLOSES[t + 5] - LINEAR_CLOSES[t]) / LINEAR_CLOSES[t]
        assert out[t]["target_return_5d"] == pytest.approx(expected), f"t={t}"
        # Off-by-one discrimination: the 4-step and 6-step horizons must differ.
        wrong_forward = (LINEAR_CLOSES[t + 4] - LINEAR_CLOSES[t]) / LINEAR_CLOSES[t]
        assert out[t]["target_return_5d"] != pytest.approx(wrong_forward)


def test_target_return_10d_independently_verified():
    out = target_rows(LINEAR_CLOSES)
    for t in range(len(LINEAR_CLOSES) - 10):
        expected = (LINEAR_CLOSES[t + 10] - LINEAR_CLOSES[t]) / LINEAR_CLOSES[t]
        assert out[t]["target_return_10d"] == pytest.approx(expected), f"t={t}"
        wrong_forward = (LINEAR_CLOSES[t + 9] - LINEAR_CLOSES[t]) / LINEAR_CLOSES[t]
        assert out[t]["target_return_10d"] != pytest.approx(wrong_forward)


def test_target_future_vol_20_matches_statistics_module():
    closes = [100.0 * (1 + ((i * 37) % 13) / 50 - ((i * 11) % 7) / 60) for i in range(40)]
    out = target_rows(closes)
    t = 5
    forward_returns = [closes[k + 1] / closes[k] - 1.0 for k in range(t, t + 20)]
    expected = statistics.stdev(forward_returns) * math.sqrt(252)
    assert out[t]["target_future_vol_20"] == pytest.approx(expected, rel=1e-12)
    # A different window or annualization must NOT match: guards against
    # silent horizon/annualization regressions.
    wrong_window = statistics.stdev(
        [closes[k + 1] / closes[k] - 1.0 for k in range(t, t + 19)]) * math.sqrt(252)
    assert out[t]["target_future_vol_20"] != pytest.approx(wrong_window, rel=1e-6)


def test_target_tail_rows_lack_forward_targets():
    n = len(LINEAR_CLOSES)
    out = target_rows(LINEAR_CLOSES)
    # Exact boundaries: last row missing 1d; rows n-5..n-1 missing 5d;
    # rows n-10..n-1 missing 10d; rows n-20..n-1 missing future vol.
    assert out[n - 2]["target_return_1d"] is not None
    assert out[n - 1]["target_return_1d"] is None

    assert out[n - 6]["target_return_5d"] is not None
    for t in range(n - 5, n):
        assert out[t]["target_return_5d"] is None, f"5d present at tail row {t}"

    assert out[n - 11]["target_return_10d"] is not None
    for t in range(n - 10, n):
        assert out[t]["target_return_10d"] is None, f"10d present at tail row {t}"

    assert out[n - 21]["target_future_vol_20"] is not None
    for t in range(n - 20, n):
        assert out[t]["target_future_vol_20"] is None, f"vol present at tail row {t}"


def test_targets_preserve_date_alignment():
    out = target_rows(LINEAR_CLOSES)
    assert len(out) == len(LINEAR_CLOSES)
    assert [r["date"] for r in out] == [f"2024-01-{i + 1:02d}" for i in range(len(out))]
    # Spot-check identity: target_return_1d at row t equals feature-space
    # close ratio to the NEXT date's row, proving row-wise alignment.
    assert out[3]["target_return_1d"] == pytest.approx(
        LINEAR_CLOSES[4] / LINEAR_CLOSES[3] - 1.0)


def test_target_off_by_one_manual_micro_series():
    # Hand-computed expectations on a tiny series where every value is obvious.
    out = target_rows([100.0, 110.0, 121.0])
    assert out[0]["target_return_1d"] == pytest.approx(0.10)
    assert out[1]["target_return_1d"] == pytest.approx(0.10)
    assert out[2]["target_return_1d"] is None
    assert all(out[i]["target_return_5d"] is None for i in range(3))
    assert all(out[i]["target_return_10d"] is None for i in range(3))
    assert all(out[i]["target_future_vol_20"] is None for i in range(3))
    # If implementation mistakenly used t+1 for the 1d label at row 1 it would
    # produce a value instead of None; if it used t+0 it would yield 0.0.
    assert out[0]["target_return_1d"] != 0.0


# ---------- ingest hardening ----------

CSV_TEXT = ("date,open,high,low,close,volume,symbol,asset_name,asset_type,region\n"
            "2024-01-02,10,12,9,11,1000,^GSPC,S&P500,Stock Index,Global\n"
            "2024-01-03,11,13,10,12,1100,^GSPC,S&P500,Stock Index,Global\n")


def write_raw(tmp_path, name, text=CSV_TEXT):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_single_csv_loads(tmp_path):
    write_raw(tmp_path, "dataset.csv")
    rows, source = load_raw_rows(tmp_path)
    assert len(rows) == 2
    assert source.name == "dataset.csv"


def test_ingest_zero_candidates_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .zip or .csv raw dataset"):
        load_raw_rows(tmp_path)


def test_ingest_multiple_candidates_fail_ambiguity(tmp_path):
    write_raw(tmp_path, "a.csv")
    write_raw(tmp_path, "b.csv")
    with pytest.raises(ValueError, match="Ambiguous raw data directory.*--raw-path"):
        load_raw_rows(tmp_path)

    # Mixed csv+zip ambiguity must also fail.
    other = tmp_path / "b.csv"
    other.unlink()
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner.csv", CSV_TEXT)
    with pytest.raises(ValueError, match="Ambiguous raw data directory"):
        load_raw_rows(tmp_path)


def test_ingest_zip_archive_loads(tmp_path):
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner.csv", CSV_TEXT)
    rows, source = load_raw_rows(tmp_path)
    assert len(rows) == 2
    assert source.name == "archive.zip"
    assert rows[0]["close"] == "11"


def test_ingest_explicit_raw_path_bypasses_directory_ambiguity(tmp_path):
    write_raw(tmp_path, "a.csv")
    chosen = write_raw(tmp_path, "chosen.csv")
    rows, source = load_raw_rows(tmp_path, raw_path=chosen)
    assert len(rows) == 2
    assert source == chosen


def test_ingest_explicit_raw_path_validated(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_rows(tmp_path, raw_path=missing)
    bad = write_raw(tmp_path, "notes.txt", "not a dataset")
    with pytest.raises(ValueError, match="Unsupported raw file type"):
        load_raw_rows(tmp_path, raw_path=bad)


# ---------- edge cases ----------

def test_malformed_dates_removed_and_logged():
    good1 = make_row("2024-01-02", 100.0)
    bad = make_row("2024-13-45", 101.0)   # impossible month/day
    good2 = make_row("2024-02-01", 102.0)
    cleaned, report = clean_ohlcv([good1, bad, good2], PipelineConfig())
    assert [r["date"] for r in cleaned] == ["2024-01-02", "2024-02-01"]
    assert report.invalid_records == 1
    assert any("malformed" in entry["reason"] for entry in report.removal_log)


def test_report_json_basic_structure(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "raw.csv").write_text(RAW_CSV, encoding="utf-8")
    payload = run(PipelineConfig(raw_dir=raw_dir, processed_dir=tmp_path / "out"))
    assert len(payload["raw_source"]["sha256"]) == 64
    assert payload["config"]["symbol"] == "^GSPC"
    assert payload["cleaning"]["rows_retained"] == 5
    assert payload["outputs"]["features"]["rows"] == 5
    assert set(payload["outputs"]["splits"]) == {"train", "validation", "test"}
    assert sum(v["rows"] for v in payload["outputs"]["splits"].values()) == 5
    assert isinstance(payload["weekday_gaps"], dict)
    report_file = tmp_path / "out" / "preprocessing_report.json"
    assert json.loads(report_file.read_text(encoding="utf-8"))["config"] == payload["config"]
