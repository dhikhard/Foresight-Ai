"""Generate Feature Set V2 from the canonical cleaned market series.

Reads {slug}_clean.csv produced by backend.pipeline.run_preprocess (an
immutable pipeline output) and writes:

    {processed_dir}/{slug}_features_v2.csv          date + 16 V2 columns
    {processed_dir}/{slug}_features_v2_report.json  provenance + warmup audit

Determinism: generation is a pure function of the input file and flags.
No randomness, no hidden state, no future information. Repeated runs on
identical input produce byte-identical feature CSVs; the only field that
varies between runs is the report's generated_at_utc timestamp.

Serialization intentionally reuses run_preprocess._write_csv/_format so
every file under data/processed shares one numeric byte format.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline.config import DEFAULT_PROCESSED_DIR
from backend.pipeline.features_v2 import (
    FEATURE_COLUMNS_V2,
    FEATURE_WARMUP_V2,
    add_features_v2,
)
from backend.pipeline.run_preprocess import _sha256, _write_csv, symbol_slug

NUMERIC_FIELDS = ("open", "high", "low", "close", "volume")
REQUIRED_INPUT_COLUMNS = ("date", *NUMERIC_FIELDS)

DEFAULT_CLEAN_PATH = DEFAULT_PROCESSED_DIR / "sp500_clean.csv"


def load_clean_rows(clean_path: Path) -> list[dict]:
    """Parse the cleaned series CSV into cleaned-shape rows (numeric floats).

    Parsing is strict: this file is machine-generated upstream, so any
    missing column or unparseable numeric price is an error rather than a
    silent skip. Only volume may legitimately be empty in the source.
    """
    with open(clean_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Clean series is missing required columns: {missing}")
        raw_rows = list(reader)

    rows: list[dict] = []
    for index, raw in enumerate(raw_rows):
        row = dict(raw)
        if not (row.get("date") or "").strip():
            raise ValueError(f"row {index}: empty date in clean series")
        for field in NUMERIC_FIELDS:
            value = (raw.get(field) or "").strip()
            if value == "":
                if field == "volume":
                    continue  # legitimate: cleaning may emit empty volume
                raise ValueError(
                    f"row {index} ({row['date']}): empty {field} in clean series"
                )
            try:
                row[field] = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"row {index} ({row['date']}): non-numeric {field}={value!r}"
                ) from exc
        rows.append(row)
    return rows


def _observed_column_stats(enriched: list[dict]) -> tuple[dict, dict]:
    """First valid index and None count per V2 column."""
    first_valid: dict[str, int | None] = {}
    none_counts: dict[str, int] = {}
    for column in FEATURE_COLUMNS_V2:
        first_valid[column] = None
        none_count = 0
        for t, row in enumerate(enriched):
            if row[column] is None:
                none_count += 1
            elif first_valid[column] is None:
                first_valid[column] = t
        none_counts[column] = none_count
    return first_valid, none_counts


def generate(clean_path: Path, processed_dir: Path, symbol: str = "^GSPC") -> dict:
    rows = load_clean_rows(clean_path)
    if not rows:
        raise ValueError("clean series contains no data rows")

    enriched = add_features_v2(rows)
    slug = symbol_slug(symbol)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_path = processed_dir / f"{slug}_features_v2.csv"
    _write_csv(features_path, ("date",) + FEATURE_COLUMNS_V2, enriched)

    observed_first_valid, none_counts = _observed_column_stats(enriched)
    mismatches = [
        f"{column}: declared first valid {declared}, "
        f"observed {observed_first_valid[column]}"
        for column, declared in FEATURE_WARMUP_V2.items()
        if observed_first_valid[column] != declared
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(clean_path),
            "sha256": _sha256(clean_path),
            "size_bytes": clean_path.stat().st_size,
            "note": "canonical cleaned series produced by backend.pipeline.run_preprocess",
        },
        "output": {
            "path": str(features_path),
            "rows": len(enriched),
            "columns": ["date", *FEATURE_COLUMNS_V2],
        },
        "feature_warmup": {
            "declared": dict(FEATURE_WARMUP_V2),
            "observed_first_valid": observed_first_valid,
        },
        "none_counts": none_counts,
        "notes": [
            "deterministic generation: identical input yields identical feature bytes",
            *mismatches,
        ],
    }
    report_path = processed_dir / f"{slug}_features_v2_report.json"
    report["output"]["report_path"] = str(report_path)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ForeSight Feature Set V2 generator")
    parser.add_argument("--clean-path", default=str(DEFAULT_CLEAN_PATH))
    parser.add_argument("--symbol", default="^GSPC")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    args = parser.parse_args()

    report = generate(
        Path(args.clean_path), Path(args.processed_dir), symbol=args.symbol
    )
    print(json.dumps({
        "symbol": args.symbol,
        "rows": report["output"]["rows"],
        "outputs": {
            "features_v2": report["output"]["path"],
            "report": report["output"]["report_path"],
        },
        "warmup_mismatches": [n for n in report["notes"] if "first valid" in n],
    }, indent=2))


if __name__ == "__main__":
    main()
