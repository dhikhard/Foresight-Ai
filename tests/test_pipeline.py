import csv
import math
from pathlib import Path

import pytest

from backend.pipeline.clean import clean_ohlcv, extract_symbol
from backend.pipeline.config import PipelineConfig
from backend.pipeline.features import add_features
from backend.pipeline.run_preprocess import run
from backend.pipeline.splits import chronological_split


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
    for i, c in enumerate(closes):
        day = start_day + i
        rows.append(make_row(f"2024-01-{day:02d}", c))
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

@pytest.mark.parametrize("index", [5, 15, 30])
def test_no_lookahead_future_values_do_not_change_past_features(index):
    closes = [100.0 + (i % 7) * 1.5 - 2 for i in range(40)]
    full = build_features(make_series(closes), ma_windows=(5,), volatility_window=5)
    truncated = build_features(make_series(closes[: index + 1]), ma_windows=(5,),
                               volatility_window=5)
    feature_cols = ("return", "log_return", "ma_5", "vol_20_annualized",
                    "rolling_peak", "drawdown", "daily_range", "open_close_return")
    for col in feature_cols:
        assert full[index][col] == truncated[index][col], f"leakage in {col} at {index}"


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
