import math

from backend.pipeline.clean import close_to_close_change, log_return

FEATURE_COLUMNS = (
    "return",
    "log_return",
    "ma_20",
    "ma_50",
    "ma_200",
    "vol_20_annualized",
    "rolling_peak",
    "drawdown",
    "daily_range",
    "open_close_return",
)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return math.nan
    m = _mean(values)
    variance = math.fsum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _windows(values: list[float], size: int):
    """Yield the trailing `size`-length window ending at each index (inclusive),
    or None while insufficient history exists. Strictly backward-looking."""
    buffer: list[float] = []
    for value in values:
        buffer.append(value)
        if len(buffer) > size:
            buffer.pop(0)
        yield list(buffer) if len(buffer) == size else None


def add_features(
    rows: list[dict],
    ma_windows: tuple[int, ...],
    volatility_window: int,
    annualization_factor: float,
) -> list[dict]:
    """Engineer features using only information available at time t.

    All windows end at the current row (inclusive); no future values are read.
    Warmup rows get None and are serialized as empty CSV fields.
    """
    closes = [row["close"] for row in rows]
    n = len(rows)

    returns: list[float | None] = [None] * n
    log_returns: list[float | None] = [None] * n
    for i in range(1, n):
        previous, current = closes[i - 1], closes[i]
        if previous > 0 and current > 0:
            returns[i] = close_to_close_change(previous, current)
            log_returns[i] = log_return(previous, current)

    ma_columns: dict[int, list[float | None]] = {}
    for window in ma_windows:
        ma_columns[window] = [
            _mean(w) if w is not None else None for w in _windows(closes, window)
        ]

    vol_values: list[float | None] = [None] * n
    valid_log_returns = [(i, lr) for i, lr in enumerate(log_returns) if lr is not None]
    lr_series = [lr for _, lr in valid_log_returns]
    vol_windows = _windows(lr_series, volatility_window)
    for (row_index, _), w in zip(valid_log_returns, vol_windows):
        if w is not None:
            vol_values[row_index] = _sample_std(w) * annualization_factor

    peaks: list[float] = []
    running_max = -math.inf
    for close in closes:
        running_max = max(running_max, close)
        peaks.append(running_max)

    enriched: list[dict] = []
    for i, row in enumerate(rows):
        out = dict(row)
        out["return"] = returns[i]
        out["log_return"] = log_returns[i]
        for window in ma_windows:
            out[f"ma_{window}"] = ma_columns[window][i]
        out["vol_20_annualized"] = vol_values[i]
        peak = peaks[i]
        out["rolling_peak"] = peak
        out["drawdown"] = (row["close"] / peak - 1.0) if peak > 0 else None
        out["daily_range"] = (
            (row["high"] - row["low"]) / row["close"]
            if row["close"] not in (0, None)
            else None
        )
        out["open_close_return"] = (
            (row["close"] - row["open"]) / row["open"]
            if row["open"] not in (0, None)
            else None
        )
        enriched.append(out)
    return enriched
