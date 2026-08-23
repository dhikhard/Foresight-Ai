"""Model-ready dataset contract for ForeSight.

Bridges the preprocessing pipeline outputs (sp500_features.csv,
sp500_targets.csv) to leakage-safe arrays for modeling.

Contract:
- X uses exactly FEATURE_COLUMNS (or another registered feature set), all
  backward-looking.
- y is exactly TARGET_COLUMN (forward-looking; never part of X).
- Rows where any feature or the target is unavailable are DROPPED
  (warm-up rows and the final row). No filling of any kind.
"""

from backend.pipeline.config import PipelineConfig
from backend.pipeline.features_v2 import FEATURE_COLUMNS_V2

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

TARGET_COLUMN = "target_return_1d"

# Registered feature sets. "v1" is the original 10-column contract;
# "v2" swaps raw price-level features for the stationary Feature Set V2.
FEATURE_SETS = {
    "v1": {
        "columns": FEATURE_COLUMNS,
        "features_csv": "sp500_features.csv",
        "report_json": "modeling_report.json",
    },
    "v2": {
        "columns": FEATURE_COLUMNS_V2,
        "features_csv": "sp500_features_v2.csv",
        "report_json": "modeling_report_v2.json",
    },
}


def resolve_feature_set(name: str) -> dict:
    """Map a feature-set name to its column contract and canonical files."""
    try:
        return FEATURE_SETS[name]
    except KeyError:
        raise ValueError(
            f"unknown feature set {name!r}; expected one of {sorted(FEATURE_SETS)}"
        ) from None


class DatasetError(ValueError):
    pass


def _to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def build_model_dataset(feature_rows: list[dict], target_rows: list[dict],
                        feature_columns: tuple = FEATURE_COLUMNS) -> dict:
    """Merge feature and target rows by date into model-ready arrays.

    Both inputs must be chronologically sorted and share an identical date
    index. Returns {"dates", "columns", "X", "y"} with warm-up/tail rows
    removed.
    """
    feature_dates = [r["date"] for r in feature_rows]
    target_dates = [r["date"] for r in target_rows]
    if feature_dates != sorted(feature_dates):
        raise DatasetError("feature rows are not chronologically sorted")
    if target_dates != sorted(target_dates):
        raise DatasetError("target rows are not chronologically sorted")
    if feature_dates != target_dates:
        raise DatasetError(
            "feature and target date indices do not align "
            f"({len(feature_dates)} vs {len(target_dates)} rows)"
        )

    dates, X, y = [], [], []
    dropped_missing_feature = 0
    dropped_missing_target = 0
    for f_row, t_row in zip(feature_rows, target_rows):
        x_values = [_to_float(f_row.get(col)) for col in feature_columns]
        target = _to_float(t_row.get(TARGET_COLUMN))
        if any(v is None for v in x_values):
            dropped_missing_feature += 1
            continue
        if target is None:
            dropped_missing_target += 1
            continue
        dates.append(f_row["date"])
        X.append(x_values)
        y.append(target)

    return {
        "dates": dates,
        "columns": list(feature_columns),
        "target": TARGET_COLUMN,
        "X": X,
        "y": y,
        "dropped_missing_feature": dropped_missing_feature,
        "dropped_missing_target": dropped_missing_target,
    }


def split_chronological(dataset: dict, config: PipelineConfig) -> dict[str, dict]:
    """Split model-ready arrays by date order using pipeline boundary config.

    train:      date <  validation_start
    validation: validation_start <= date < test_start
    test:       date >= test_start
    """
    validation_start = config.validation_start
    test_start = config.test_start
    parts: dict[str, dict] = {}
    for name, predicate in (
        ("train", lambda d: d < validation_start),
        ("validation", lambda d: validation_start <= d < test_start),
        ("test", lambda d: d >= test_start),
    ):
        indices = [i for i, d in enumerate(dataset["dates"]) if predicate(d)]
        parts[name] = {
            "dates": [dataset["dates"][i] for i in indices],
            "columns": dataset["columns"],
            "target": dataset["target"],
            "X": [dataset["X"][i] for i in indices],
            "y": [dataset["y"][i] for i in indices],
        }
    return parts


def load_processed_frames(features_path, targets_path) -> tuple[list[dict], list[dict]]:
    """Read the two processed CSVs produced by the preprocessing pipeline."""
    import csv

    with open(features_path, newline="", encoding="utf-8") as fh:
        features = list(csv.DictReader(fh))
    with open(targets_path, newline="", encoding="utf-8") as fh:
        targets = list(csv.DictReader(fh))
    return features, targets
