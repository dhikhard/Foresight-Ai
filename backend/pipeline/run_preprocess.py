import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline.clean import CleaningReport, clean_ohlcv, extract_symbol
from backend.pipeline.config import PipelineConfig
from backend.pipeline.features import FEATURE_COLUMNS, add_features
from backend.pipeline.splits import chronological_split
from backend.pipeline.targets import add_future_targets

BASE_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "asset_name",
    "asset_type",
    "region",
)

SYMBOL_SLUGS = {"^GSPC": "sp500", "^IXIC": "nasdaq", "^GDAXI": "dax", "^FTSE": "ftse100"}


def symbol_slug(symbol: str) -> str:
    if symbol in SYMBOL_SLUGS:
        return SYMBOL_SLUGS[symbol]
    return (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace(".", "_")
        .replace("/", "_")
        .lower()
    )


def _format(value) -> str:
    if value is None or value == "":
        return ""
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    _format(row[c]) if c not in ("date", *BASE_COLUMNS[6:]) else row[c]
                    for c in columns
                ]
            )


def _weekday_gap_histogram(rows: list[dict]) -> dict[str, int]:
    from datetime import date, timedelta

    gaps: dict[int, int] = {}
    previous = None
    for row in rows:
        current = date.fromisoformat(row["date"])
        if previous is not None:
            missing = 0
            cursor = previous
            while cursor < current - timedelta(days=1):
                cursor += timedelta(days=1)
                if cursor.weekday() < 5:
                    missing += 1
            if missing:
                gaps[missing] = gaps.get(missing, 0) + 1
        previous = current
    return {f"missing_{k}_weekdays": v for k, v in sorted(gaps.items())}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: PipelineConfig) -> dict:
    from backend.pipeline.ingest import load_raw_rows

    rows, raw_source = load_raw_rows(config.raw_dir, config.raw_path)
    selected = extract_symbol(rows, config)
    cleaned, report = clean_ohlcv(selected, config)
    featured = add_features(
        cleaned,
        ma_windows=config.ma_windows,
        volatility_window=config.volatility_window,
        annualization_factor=config.annualization_factor,
    )
    if config.with_targets:
        targeted = add_future_targets(
            [dict(r) for r in featured],
            horizons=config.target_horizons,
            future_volatility_window=config.future_volatility_window,
            annualization_factor=config.annualization_factor,
        )
        target_columns = ("date",) + tuple(
            c for c in targeted[0] if c.startswith("target_")
        )
    else:
        target_columns = ()

    slug = symbol_slug(config.symbol)
    out_dir = config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = out_dir / f"{slug}_clean.csv"
    features_path = out_dir / f"{slug}_features.csv"
    targets_path = out_dir / f"{slug}_targets.csv"

    _write_csv(clean_path, BASE_COLUMNS, cleaned)
    _write_csv(features_path, BASE_COLUMNS + FEATURE_COLUMNS, featured)
    if config.with_targets and targeted:
        _write_csv(targets_path, target_columns, targeted)

    splits = chronological_split(featured, config.validation_start, config.test_start)
    split_paths = {}
    for name, part in splits.items():
        path = out_dir / f"{slug}_features_{name}.csv"
        _write_csv(path, BASE_COLUMNS + FEATURE_COLUMNS, part)
        split_paths[name] = {"path": str(path), "rows": len(part)}

    gaps = _weekday_gap_histogram(cleaned)
    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_source": {
            "path": str(raw_source),
            "sha256": _sha256(raw_source),
            "size_bytes": raw_source.stat().st_size,
            "note": "immutable; read-only input",
        },
        "config": {
            "symbol": config.symbol,
            "ma_windows": list(config.ma_windows),
            "volatility_window": config.volatility_window,
            "annualization": "sample std of daily log returns x sqrt(252)",
            "validation_start": config.validation_start,
            "test_start": config.test_start,
            "with_targets": config.with_targets,
        },
        "cleaning": report.as_dict(),
        "weekday_gaps": {
            **gaps,
            "interpretation": "US market holidays/closures; not fabricated or filled",
        },
        "outputs": {
            "clean": {"path": str(clean_path), "rows": len(cleaned)},
            "features": {"path": str(features_path), "rows": len(featured)},
            "targets": {"path": str(targets_path), "rows": len(targeted)} if target_columns else None,
            "splits": split_paths,
        },
        "notes": config.log,
    }
    report_path = out_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2)

    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ForeSight market-data preprocessing")
    parser.add_argument("--symbol", default=PipelineConfig.symbol)
    parser.add_argument("--raw-path", default=None, help="explicit raw .csv/.zip file")
    parser.add_argument("--validation-start", default=PipelineConfig.validation_start)
    parser.add_argument("--test-start", default=PipelineConfig.test_start)
    parser.add_argument("--processed-dir", default=None, help="output directory override")
    parser.add_argument("--with-targets", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        symbol=args.symbol,
        raw_path=Path(args.raw_path) if args.raw_path else None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else PipelineConfig.processed_dir,
        validation_start=args.validation_start,
        test_start=args.test_start,
        with_targets=args.with_targets,
    )
    payload = run(config)
    print(json.dumps({
        "symbol": payload["config"]["symbol"],
        "rows_retained": payload["cleaning"]["rows_retained"],
        "rows_removed": payload["cleaning"]["rows_removed"],
        "outputs": {
            k: v["path"] if isinstance(v, dict) and "path" in v else v
            for k, v in payload["outputs"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
