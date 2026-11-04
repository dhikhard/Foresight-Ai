"""Temporal alignment contract enforcement.

Frozen point-in-time rules:
- T(t) = S&P 500 trading-day close at 16:00 America/New_York -> UTC
- event_time <= T(t) (exact-at-close included)
- Post-close excluded
- Weekend/holiday events roll into next trading day
- News aggregated over (T(t-1), T(t)]
- Prediction-market: latest observation <= T(t)
- Staleness flag if > 24h old
- All storage UTC
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ET_TZ = ZoneInfo("America/New_York")

CLOSE_HOUR = 16
CLOSE_MINUTE = 0


def trading_day_close_utc(d: date) -> datetime:
    """16:00 America/New_York on trading day *d*, converted to UTC.

    Uses the IANA timezone database (via ``zoneinfo`` + ``tzdata``) for
    correct DST handling.
    """
    close_local = datetime(d.year, d.month, d.day, CLOSE_HOUR, CLOSE_MINUTE, tzinfo=_ET_TZ)
    return close_local.astimezone(timezone.utc)


def load_trading_calendar(clean_csv: Path) -> list[date]:
    dates: set[date] = set()
    with open(clean_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("date") or row.get("Date") or ""
            if raw:
                dates.add(_parse_date(raw.strip()))
    return sorted(dates)


def prev_trading_day_close(dates: list[date], d: date) -> datetime | None:
    idx = _find_index(dates, d)
    if idx is None or idx <= 0:
        return None
    return trading_day_close_utc(dates[idx - 1])


def _find_index(dates: list[date], d: date) -> int | None:
    for i, dd in enumerate(dates):
        if dd == d:
            return i
    return None


def is_eligible(event_time_str: str, cutoff_utc: datetime) -> bool:
    et = _parse_to_utc(event_time_str)
    if et is None:
        return False
    return et <= cutoff_utc


def filter_point_in_time(
    rows: list[dict],
    cutoff_utc: datetime,
    time_field: str,
) -> list[dict]:
    return [r for r in rows if is_eligible(r.get(time_field, ""), cutoff_utc)]


def asof_join(
    observations: list[dict],
    calendar: list[date],
    obs_time_field: str,
) -> list[dict]:
    if not observations:
        return [
            {"trading_date": d.isoformat(), "trading_close_utc": trading_day_close_utc(d).isoformat()}
            for d in calendar
        ]

    def _sort_key(r: dict) -> datetime:
        et = _parse_to_utc(r.get(obs_time_field, ""))
        return et or datetime(2000, 1, 1, tzinfo=timezone.utc)

    obs_sorted = sorted(observations, key=_sort_key)
    results = []
    obs_idx = 0
    n = len(obs_sorted)

    for d in calendar:
        cutoff = trading_day_close_utc(d)
        while obs_idx < n:
            et = _parse_to_utc(obs_sorted[obs_idx].get(obs_time_field, ""))
            if et is None:
                obs_idx += 1
                continue
            if et <= cutoff:
                obs_idx += 1
            else:
                break
        best = obs_sorted[obs_idx - 1] if obs_idx > 0 else None

        row = {
            "trading_date": d.isoformat(),
            "trading_close_utc": cutoff.isoformat(),
        }
        if best is not None:
            row.update(best)
            row["staleness_hours"] = _staleness_hours(best.get(obs_time_field, ""), cutoff)
            row["is_stale"] = row["staleness_hours"] > 24.0
        results.append(row)

    return results


def aggregate_news_by_day(
    articles: list[dict],
    calendar: list[date],
    time_field: str,
) -> list[dict]:
    if not articles:
        return [
            {
                "trading_date": d.isoformat(),
                "trading_close_utc": trading_day_close_utc(d).isoformat(),
                "article_count": 0,
                "article_ids": "[]",
            }
            for d in calendar
        ]

    def _sort_key(a: dict) -> datetime:
        et = _parse_to_utc(a.get(time_field, ""))
        return et or datetime(2000, 1, 1, tzinfo=timezone.utc)

    art_sorted = sorted(articles, key=_sort_key)
    results = []
    art_idx = 0
    n = len(art_sorted)

    for i, d in enumerate(calendar):
        close = trading_day_close_utc(d)
        prev_close = trading_day_close_utc(calendar[i - 1]) if i > 0 else None

        ids: list[str] = []
        while art_idx < n:
            et = _parse_to_utc(art_sorted[art_idx].get(time_field, ""))
            if et is None:
                art_idx += 1
                continue
            if et > close:
                break
            if prev_close is None or et > prev_close:
                ids.append(
                    art_sorted[art_idx].get("article_id", art_sorted[art_idx].get("url", ""))
                )
            art_idx += 1

        results.append({
            "trading_date": d.isoformat(),
            "trading_close_utc": close.isoformat(),
            "article_count": len(ids),
            "article_ids": str(ids),
        })

    return results


def _staleness_hours(event_time_str: str, cutoff_utc: datetime) -> float:
    et = _parse_to_utc(event_time_str)
    if et is None:
        return -1.0
    diff = cutoff_utc - et
    return diff.total_seconds() / 3600.0


def _parse_date(raw: str) -> date:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    return datetime.strptime(raw[:10], "%Y-%m-%d").date()


def _parse_to_utc(ts) -> datetime | None:
    if ts is None or ts == "":
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    s = str(ts).strip()
    # GDELT seendate YYYYMMDDHHMMSS
    if len(s) >= 14 and s.isdigit():
        try:
            return datetime(
                int(s[:4]), int(s[4:6]), int(s[6:8]),
                int(s[8:10]), int(s[10:12]), int(s[12:14]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass
    # ISO-8601
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        pass
    return None
