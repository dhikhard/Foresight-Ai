import math

from backend.pipeline.features import _sample_std


def add_future_targets(
    rows: list[dict],
    horizons: tuple[int, ...] = (1, 5, 10),
    future_volatility_window: int = 20,
    annualization_factor: float = 252 ** 0.5,
) -> list[dict]:
    """Forward-looking target columns. These use FUTURE closes by definition
    and must never be used as model input features.

    Rows without enough forward history get None (empty CSV field).
    """
    closes = [row["close"] for row in rows]
    n = len(rows)
    out = [dict(row) for row in rows]

    for horizon in sorted(set(horizons)):
        column = f"target_return_{horizon}d"
        for i in range(n):
            j = i + horizon
            out[i][column] = (
                closes[j] / closes[i] - 1.0 if j < n and closes[i] > 0 else None
            )

    vol_column = f"target_future_vol_{future_volatility_window}"
    for i in range(n):
        window_closes = closes[i : i + future_volatility_window + 1]
        if len(window_closes) < future_volatility_window + 1:
            out[i][vol_column] = None
            continue
        returns = [
            window_closes[k + 1] / window_closes[k] - 1.0
            for k in range(future_volatility_window)
            if window_closes[k] > 0
        ]
        std = _sample_std(returns)
        out[i][vol_column] = std * annualization_factor if not math.isnan(std) else None
    return out
