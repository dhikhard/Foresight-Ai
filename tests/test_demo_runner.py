"""Tests for the one-command Phase 1 demo runner (run_demo.py).

The runner shells out to the real CLI entry points, so these tests exercise
the actual repository entry points end to end on a small synthetic raw
dataset. The canonical files under data/processed are never touched and are
asserted byte-identical across a run.
"""

import csv
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "run_demo.py"
CANONICAL_V1_REPORT = PROJECT_ROOT / "data" / "processed" / "modeling_report.json"
CANONICAL_V2_REPORT = PROJECT_ROOT / "data" / "processed" / "modeling_report_v2.json"

RAW_COLUMNS = (
    "date", "open", "high", "low", "close", "volume",
    "symbol", "asset_name", "asset_type", "region",
)


def _write_synthetic_raw(path: Path, n_rows: int = 1600,
                         start: date = date(2018, 1, 1)) -> None:
    """Write a deterministic synthetic raw dataset spanning the split dates.

    Dates start before the default validation boundary (2019-01-01) and run
    past the default test boundary (2023-01-01) so all three splits are
    populated. Prices follow a deterministic, strictly positive walk.
    """
    rows = []
    day = start
    close = 100.0
    i = 0
    while len(rows) < n_rows:
        if day.weekday() < 5:  # trading days only
            drift = ((i * 37) % 11 - 5) / 1000.0
            close = close * (1.0 + drift)
            open_ = close * 0.999
            high = close * 1.002
            low = close * 0.998
            rows.append((
                day.isoformat(), open_, high, low, close,
                1_000_000 + i, "^GSPC", "S&P 500", "index", "US",
            ))
            i += 1
        day += timedelta(days=1)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(RAW_COLUMNS)
        writer.writerows(rows)


def test_runner_refuses_nonempty_out_dir_without_force(tmp_path):
    """Existing user files must never be silently overwritten."""
    raw = tmp_path / "raw.csv"
    _write_synthetic_raw(raw, n_rows=5)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep_me.txt"
    sentinel.write_text("do not clobber", encoding="utf-8")

    before_v1 = CANONICAL_V1_REPORT.read_bytes()
    before_v2 = CANONICAL_V2_REPORT.read_bytes()

    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--raw-path", str(raw), "--out-dir", str(out)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )

    assert proc.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "do not clobber"
    assert CANONICAL_V1_REPORT.read_bytes() == before_v1
    assert CANONICAL_V2_REPORT.read_bytes() == before_v2


def test_runner_rejects_missing_raw_path(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--raw-path", str(tmp_path / "nope.csv"),
         "--out-dir", str(out)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_runner_end_to_end_uses_existing_pipeline(tmp_path):
    """Full V1 + V2 workflow runs, writes artifacts, and leaves canon alone."""
    raw = tmp_path / "raw.csv"
    _write_synthetic_raw(raw)
    out = tmp_path / "demo_out"

    before_v1 = CANONICAL_V1_REPORT.read_bytes()
    before_v2 = CANONICAL_V2_REPORT.read_bytes()

    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--raw-path", str(raw), "--out-dir", str(out)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    for name in (
        "sp500_clean.csv",
        "sp500_features.csv",
        "sp500_targets.csv",
        "sp500_features_v2.csv",
        "modeling_report.json",
        "modeling_report_v2.json",
        "demo_results.md",
    ):
        assert (out / name).exists(), f"missing {name}"

    # The demo must not touch the canonical reports.
    assert CANONICAL_V1_REPORT.read_bytes() == before_v1
    assert CANONICAL_V2_REPORT.read_bytes() == before_v2

    results = (out / "demo_results.md").read_text(encoding="utf-8")
    assert "V1 feature set" in results
    assert "V2 feature set" in results
    assert "validation MAE" in results
