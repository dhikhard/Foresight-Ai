"""Feature Set V2: stationary, scale-free market features.

Generated from *cleaned* OHLCV rows (same representation consumed by V1's
add_features). V1 outputs are never read or modified; this module is a pure
function of the cleaned series.

Leakage policy: every feature at time t uses only rows <= t. All rolling
windows terminate at t (inclusive). No centered windows, no full-dataset
normalization, no target_* access, no future statistics of any kind.

Non-positive price policy (mirrors V1): any computation whose required
inputs include a non-positive price yields None instead of crashing or
fabricating a value.

Warm-up table (first row index with a defined value):

    ret_lag_1          2     ret_lag_2           3     ret_lag_5      6
    ret_lag_10        11     ret_lag_20         21
    mom_5              5     mom_10             10     mom_20       20
    ma_dist_20        19     ma_dist_50         49     ma_dist_200 199
    vol_60_annualized 60     vol_ratio_20_60    60     vol_change_5d 25
    rsi_14            14     bollinger_pctb_20  19

RSI-14 convention (Wilder-style, deterministic):
- daily change d[i] = close[i] - close[i-1]
- seed: average gain/loss = arithmetic mean of max(d,0)/max(-d,0) over the
  FIRST 14 changes (rows 1..14); first RSI value is emitted at row 14
- update: avg = (prev_avg * 13 + current) / 14  (Wilder smoothing)
- RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss
- zero cases (deterministic): avg_loss == 0 and avg_gain > 0 -> 100;
  avg_gain == 0 and avg_loss > 0 -> 0; both zero -> 50
- a non-positive close permanently invalidates the series: RSI becomes None
  from the first affected row onward and the smoothing state stops updating
"""

import math

from backend.pipeline.clean import close_to_close_change, log_return

ANNUALIZATION_FACTOR = math.sqrt(252.0)

FEATURE_COLUMNS_V2 = (
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_5",
    "ret_lag_10",
    "ret_lag_20",
    "mom_5",
    "mom_10",
    "mom_20",
    "ma_dist_20",
    "ma_dist_50",
    "ma_dist_200",
    "vol_60_annualized",
    "vol_ratio_20_60",
    "vol_change_5d",
    "rsi_14",
    "bollinger_pctb_20",
)

LAG_HORIZONS = (1, 2, 5, 10, 20)
MOMENTUM_HORIZONS = (5, 10, 20)
MA_DIST_WINDOWS = (20, 50, 200)
RSI_PERIOD = 14
BOLLINGER_WINDOW = 20

FEATURE_WARMUP_V2 = {
    **{f"ret_lag_{k}": k + 1 for k in LAG_HORIZONS},
    **{f"mom_{k}": k for k in MOMENTUM_HORIZONS},
    **{f"ma_dist_{k}": k - 1 for k in MA_DIST_WINDOWS},
    "vol_60_annualized": 60,
    "vol_ratio_20_60": 60,
    "vol_change_5d": 25,
    "rsi_14": RSI_PERIOD,
    "bollinger_pctb_20": BOLLINGER_WINDOW - 1,
}


def _sample_std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1), same convention as V1 features."""
    n = len(values)
    if n < 2:
        return math.nan
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _rolling_volatility(log_returns: list[float | None], window: int,
                        n_rows: int) -> list[float | None]:
    """Annualized sample std of the last `window` log returns ending at t."""
    out: list[float | None] = [None] * n_rows
    for t in range(window, n_rows):
        segment = log_returns[t - window + 1 : t + 1]
        if any(v is None for v in segment):
            continue
        out[t] = _sample_std(segment) * ANNUALIZATION_FACTOR
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _rsi_series(closes: list[float], n_rows: int) -> list[float | None]:
    rsis: list[float | None] = [None] * n_rows
    period = RSI_PERIOD
    avg_gain: float | None = None
    avg_loss: float | None = None
    seed_gains: list[float] = []
    seed_losses: list[float] = []
    for i in range(1, n_rows):
        previous, current = closes[i - 1], closes[i]
        if previous <= 0 or current <= 0:
            break  # documented dead-series policy: None from here onward
        change = current - previous
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if avg_gain is None or avg_loss is None:
            seed_gains.append(gain)
            seed_losses.append(loss)
            if len(seed_gains) == period:
                avg_gain = math.fsum(seed_gains) / period
                avg_loss = math.fsum(seed_losses) / period
                rsis[i] = _rsi_value(avg_gain, avg_loss)
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            rsis[i] = _rsi_value(avg_gain, avg_loss)
    return rsis


def add_features_v2(rows: list[dict]) -> list[dict]:
    """Attach the 16 V2 columns to cleaned OHLCV rows.

    Input rows are not mutated; enriched copies are returned with all
    original fields preserved. Warm-up rows hold None.
    """
    n = len(rows)
    if n == 0:
        return []

    closes = [row["close"] for row in rows]

    returns: list[float | None] = [None] * n
    log_returns: list[float | None] = [None] * n
    for i in range(1, n):
        previous, current = closes[i - 1], closes[i]
        if previous > 0 and current > 0:
            returns[i] = close_to_close_change(previous, current)
            log_returns[i] = log_return(previous, current)

    vol20 = _rolling_volatility(log_returns, 20, n)
    vol60 = _rolling_volatility(log_returns, 60, n)
    rsis = _rsi_series(closes, n)

    ma_values: dict[int, list[float | None]] = {}
    for k in MA_DIST_WINDOWS:
        column: list[float | None] = [None] * n
        for t in range(k - 1, n):
            window = closes[t - k + 1 : t + 1]
            if not all(c > 0 for c in window):
                continue  # documented policy: non-positive input -> None
            mean = math.fsum(window) / k
            if mean > 0 and closes[t] > 0:
                column[t] = closes[t] / mean - 1.0
        ma_values[k] = column

    enriched: list[dict] = []
    for t, row in enumerate(rows):
        out = dict(row)
        for k in LAG_HORIZONS:
            out[f"ret_lag_{k}"] = returns[t - k] if t >= k else None
        for k in MOMENTUM_HORIZONS:
            out[f"mom_{k}"] = (
                closes[t] / closes[t - k] - 1.0
                if t >= k and closes[t] > 0 and closes[t - k] > 0
                else None
            )
        for k in MA_DIST_WINDOWS:
            out[f"ma_dist_{k}"] = ma_values[k][t]

        out["vol_60_annualized"] = vol60[t]
        if vol20[t] is not None and vol60[t] is not None and vol60[t] != 0:
            out["vol_ratio_20_60"] = vol20[t] / vol60[t]
        else:
            out["vol_ratio_20_60"] = None
        if (
            t >= 5
            and vol20[t] is not None
            and vol20[t - 5] is not None
            and vol20[t - 5] != 0
        ):
            out["vol_change_5d"] = vol20[t] / vol20[t - 5] - 1.0
        else:
            out["vol_change_5d"] = None

        out["rsi_14"] = rsis[t]

        if t >= BOLLINGER_WINDOW - 1:
            window = closes[t - BOLLINGER_WINDOW + 1 : t + 1]
            if all(c > 0 for c in window):
                middle = math.fsum(window) / BOLLINGER_WINDOW
                sigma = _sample_std(window)
                lower = middle - 2.0 * sigma
                upper = middle + 2.0 * sigma
                out["bollinger_pctb_20"] = (
                    (closes[t] - lower) / (upper - lower)
                    if upper != lower
                    else None
                )
            else:
                out["bollinger_pctb_20"] = None
        else:
            out["bollinger_pctb_20"] = None

        enriched.append(out)
    return enriched
