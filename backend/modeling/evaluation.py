"""Evaluation metrics for next-day return forecasts.

All metrics operate on plain lists so the same interface scores baselines
and learned models identically.
"""

import math


def metrics(y_true: list[float], y_pred: list[float]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot evaluate on an empty set")

    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)

    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    directional = sum(
        1 for t, p in zip(y_true, y_pred) if math.copysign(1, t) == math.copysign(1, p)
    ) / n

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional,
        "mean_actual_return": mean_true,
        "mean_predicted_return": sum(y_pred) / n,
        "prediction_std": (
            math.sqrt(sum((p - sum(y_pred) / n) ** 2 for p in y_pred) / n)
        ),
    }
