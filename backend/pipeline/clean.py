import math
from dataclasses import dataclass, field

from backend.pipeline.config import PipelineConfig

NUMERIC_FIELDS = ("open", "high", "low", "close")


@dataclass
class CleaningReport:
    original_rows: int = 0
    symbol_rows: int = 0
    rows_removed: int = 0
    rows_retained: int = 0
    duplicates_found: int = 0
    missing_values: int = 0
    invalid_records: int = 0
    ohlc_violations_logged: int = 0
    negative_price_rows_logged: int = 0
    zero_volume_rows: int = 0
    date_range_before: tuple[str, str] | None = None
    date_range_after: tuple[str, str] | None = None
    removal_log: list = field(default_factory=list)
    warning_log: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original_rows_total": self.original_rows,
            "symbol_rows": self.symbol_rows,
            "rows_removed": self.rows_removed,
            "rows_retained": self.rows_retained,
            "duplicates_found": self.duplicates_found,
            "missing_values_in_symbol": self.missing_values,
            "invalid_records_removed": self.invalid_records,
            "ohlc_violations_logged_not_removed": self.ohlc_violations_logged,
            "negative_price_rows_kept": self.negative_price_rows_logged,
            "zero_volume_rows_kept": self.zero_volume_rows,
            "date_range_before_cleaning": list(self.date_range_before or ()),
            "date_range_after_cleaning": list(self.date_range_after or ()),
            "removals": self.removal_log[:200],
            "warnings": self.warning_log[:200],
        }


def _parse_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_date(value: str) -> bool:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    y, m, d = value[:4], value[5:7], value[8:10]
    return y.isdigit() and m.isdigit() and d.isdigit() and 1 <= int(m) <= 12 and 1 <= int(d) <= 31


def extract_symbol(rows: list[dict], config: PipelineConfig) -> list[dict]:
    selected = [r for r in rows if r.get("symbol") == config.symbol]
    config.note(
        f"extracted {len(selected)} of {len(rows)} rows for symbol {config.symbol}"
    )
    return selected


def clean_ohlcv(rows: list[dict], config: PipelineConfig) -> tuple[list[dict], CleaningReport]:
    """Conservative financial cleaning.

    Never fabricates prices. Rows are removed only when a critical field is
    missing/unparseable or the date is malformed. OHLC relationship violations
    and negative prices are logged and kept (futures settlement artifacts and
    events such as the April 2020 negative WTI print are legitimate history).
    """
    report = CleaningReport(original_rows=len(rows))
    seen_dates: set[str] = set()
    cleaned: list[dict] = []
    dates_before: list[str] = []

    for index, row in enumerate(rows):
        date_value = (row.get("date") or "").strip()
        dates_before.append(date_value)
        location = f"row {index} ({date_value or 'missing-date'})"

        if not _valid_date(date_value):
            report.invalid_records += 1
            report.rows_removed += 1
            report.removal_log.append({"location": location, "reason": "malformed or missing date"})
            continue

        parsed: dict[str, float | None] = {
            name: _parse_float((row.get(name) or "").strip())
            for name in NUMERIC_FIELDS
        }
        volume_raw = (row.get("volume") or "").strip()
        volume = _parse_float(volume_raw) if volume_raw != "" else None

        critical_missing = any(parsed[name] is None for name in NUMERIC_FIELDS)
        if critical_missing:
            report.missing_values += sum(1 for v in parsed.values() if v is None)
            report.rows_removed += 1
            report.removal_log.append(
                {"location": location, "reason": "missing or non-numeric OHLC price"}
            )
            continue

        if date_value in seen_dates:
            report.duplicates_found += 1
            report.rows_removed += 1
            report.removal_log.append(
                {"location": location, "reason": "duplicate date for symbol; kept first occurrence"}
            )
            continue
        seen_dates.add(date_value)

        o, h, l, c = (parsed[name] for name in NUMERIC_FIELDS)
        negatives = [v for v in (o, h, l, c, volume) if v is not None and v < 0]
        if negatives:
            report.negative_price_rows_logged += 1
            report.warning_log.append(
                {"location": location, "note": "negative price/volume retained (legitimate market event)"}
            )

        if volume == 0:
            report.zero_volume_rows += 1

        if not (l <= min(o, c) + 1e-9 and h >= max(o, c) - 1e-9 and l <= h + 1e-9):
            report.ohlc_violations_logged += 1
            report.warning_log.append(
                {"location": location, "note": "OHLC relationship violation retained (settlement-price artifact)"}
            )

        cleaned.append(
            {
                "date": date_value,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume if volume is not None else "",
                "symbol": row.get("symbol", ""),
                "asset_name": row.get("asset_name", ""),
                "asset_type": row.get("asset_type", ""),
                "region": row.get("region", ""),
            }
        )

    if dates_before and all(_valid_date(d) for d in dates_before if d):
        valid_before = [d for d in dates_before if d]
        if valid_before:
            report.date_range_before = (min(valid_before), max(valid_before))

    cleaned.sort(key=lambda r: r["date"])
    if cleaned:
        report.date_range_after = (cleaned[0]["date"], cleaned[-1]["date"])

    report.symbol_rows = len(rows)
    report.rows_retained = len(cleaned)
    config.note(f"cleaning retained {len(cleaned)} of {len(rows)} rows for {config.symbol}")
    return cleaned, report


def validate_ohlc(row: dict) -> bool:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    return l <= min(o, c) + 1e-9 and h >= max(o, c) - 1e-9 and l <= h + 1e-9


def close_to_close_change(previous: float, current: float) -> float:
    return current / previous - 1.0


def log_return(previous: float, current: float) -> float:
    if previous <= 0 or current <= 0:
        return math.nan
    return math.log(current / previous)
