"""Train and evaluate the first ForeSight forecasting baselines.

Usage (from repository root):
    python -m backend.modeling.run_modeling                # Feature Set V1
    python -m backend.modeling.run_modeling --feature-set v2

Loads data/processed/sp500_features[_v2].csv + sp500_targets.csv, builds the
model-ready dataset, fits preprocessing on TRAIN only, and evaluates
baselines plus ridge regression. Ridge hyperparameters are selected on
VALIDATION only; TEST is scored once as a final holdout.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.modeling.baselines import (
    PersistenceBaseline,
    TrainMeanBaseline,
    ZeroBaseline,
)
from backend.modeling.dataset import (
    TARGET_COLUMN,
    build_model_dataset,
    load_processed_frames,
    resolve_feature_set,
    split_chronological,
)
from backend.modeling.evaluation import metrics
from backend.modeling.preprocessing import TrainOnlyStandardizer
from backend.modeling.regression import RidgeRegressor
from backend.pipeline.config import PipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _evaluate_split(name, y_true, y_pred) -> dict:
    return {"split": name, **metrics(y_true, y_pred)}


def run(features_path: Path, targets_path: Path, out_path: Path,
        alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
        config: PipelineConfig | None = None,
        feature_set: str = "v1") -> dict:
    config = config or PipelineConfig()
    spec = resolve_feature_set(feature_set)
    columns = spec["columns"]
    return_index = columns.index("return") if "return" in columns else None

    feature_rows, target_rows = load_processed_frames(features_path, targets_path)
    dataset = build_model_dataset(feature_rows, target_rows,
                                  feature_columns=columns)
    splits = split_chronological(dataset, config)

    train = splits["train"]
    standardizer = TrainOnlyStandardizer().fit(train["X"])
    transformed = {name: standardizer.transform(part["X"])
                   for name, part in splits.items()}

    results: dict[str, dict] = {}

    zero = ZeroBaseline().fit(train["X"], train["y"])
    mean_model = TrainMeanBaseline().fit(train["X"], train["y"])
    baselines = {"zero_return": zero, "train_mean": mean_model}
    if return_index is not None:
        # Persistence reads the raw 'return' column, so it must receive the
        # un-transformed feature matrix (zero/mean baselines ignore X
        # entirely). Feature sets without a same-day return column (V2)
        # cannot express persistence; ret_lag_1 is yesterday's return and is
        # NOT a persistence input.
        persistence = PersistenceBaseline(return_index).fit(train["X"], train["y"])
        baselines["persistence"] = persistence

    predictions = {
        model_name: {
            split: model.predict(
                splits[split]["X"] if model_name == "persistence"
                else transformed[split]
            )
            for split in splits
        }
        for model_name, model in baselines.items()
    }
    results.update({
        model_name: {
            "model": model_name,
            "selection": "fixed rule",
            "splits": [
                _evaluate_split(split, splits[split]["y"], preds[split])
                for split in ("train", "validation", "test")
            ],
        }
        for model_name, preds in predictions.items()
    })

    # Ridge: choose alpha on VALIDATION only. Correctness note (2026-08-23):
    # ridge MUST be fit on the standardized TRAIN matrix — the previous
    # implementation fit on raw train["X"] while predicting on transformed
    # matrices, a scale mismatch that invalidated earlier ridge results.
    val_scores = []
    for alpha in alpha_grid:
        ridge = RidgeRegressor(alpha=alpha).fit(transformed["train"],
                                                train["y"])
        val_metrics = metrics(splits["validation"]["y"],
                              ridge.predict(transformed["validation"]))
        val_scores.append({"alpha": alpha, "val_mae": val_metrics["mae"]})
    best_alpha = min(val_scores, key=lambda s: s["val_mae"])["alpha"]
    ridge = RidgeRegressor(alpha=best_alpha).fit(transformed["train"],
                                                 train["y"])
    results["ridge"] = {
        "model": "ridge",
        "selection": {"method": "validation MAE", "grid": list(alpha_grid),
                      "scores": val_scores, "chosen_alpha": best_alpha},
        "weights": ridge.to_dict(),
        "splits": [
            _evaluate_split(split, splits[split]["y"],
                            ridge.predict(transformed[split]))
            for split in ("train", "validation", "test")
        ],
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": feature_set,
        "target": TARGET_COLUMN,
        "feature_columns": list(columns),
        "protocol": {
            "horizon": "1 trading day",
            "prediction_semantics":
                "at close(t), features use information through close(t); "
                "target is close(t+1)/close(t)-1",
            "split_boundaries": {
                "train": f"< {config.validation_start}",
                "validation": f"{config.validation_start} <= d < {config.test_start}",
                "test": f">= {config.test_start}",
            },
            "row_policy":
                "rows missing any feature or the target are dropped before "
                "splitting; warm-up values are never imputed",
            "scaling": "standardize; statistics fit on TRAIN only, then frozen",
            "alpha_selection": {
                "criterion": "validation MAE",
                "grid": list(alpha_grid),
            },
            "test_discipline":
                "TEST scored once as holdout after all decisions are frozen; "
                "never used for model, feature-set, or hyperparameter choices",
            "feature_set_winner_rule":
                "lower validation ridge MAE between v1 and v2 wins; test "
                "metrics do not change the winner",
            "methodology_version": 2,
            "methodology_notes":
                "2026-08-23: corrected ridge fit/predict scale mismatch "
                "(ridge was previously fit on raw TRAIN X while predicting "
                "on standardized matrices). Reports generated before this "
                "correction, including the original Week-3 modeling report, "
                "are superseded.",
        },
        "rows": {name: len(part["y"]) for name, part in splits.items()},
        "dataset_dropped_missing_feature": dataset["dropped_missing_feature"],
        "dataset_dropped_missing_target": dataset["dropped_missing_target"],
        "date_range": [dataset["dates"][0], dataset["dates"][-1]],
        "preprocessing": {
            "method": "standardize, fit on TRAIN only",
            "means": standardizer.means,
            "stds": standardizer.stds,
        },
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ForeSight baseline modeling")
    parser.add_argument("--feature-set", choices=("v1", "v2"), default="v1",
                        help="feature contract to model (default: v1)")
    parser.add_argument("--features", default=None,
                        help="override features CSV path")
    parser.add_argument("--targets", default=None,
                        help="override targets CSV path")
    parser.add_argument("--out", default=None,
                        help="override report JSON path")
    args = parser.parse_args()

    spec = resolve_feature_set(args.feature_set)
    processed = PROJECT_ROOT / "data" / "processed"
    features_path = Path(args.features) if args.features else (
        processed / spec["features_csv"])
    targets_path = Path(args.targets) if args.targets else (
        processed / "sp500_targets.csv")
    out_path = Path(args.out) if args.out else processed / spec["report_json"]

    payload = run(features_path, targets_path, out_path,
                  feature_set=args.feature_set)

    print(f"feature_set={args.feature_set} target={payload['target']} "
          f"rows={payload['rows']} range={payload['date_range']}")
    header = f"{'model':<14}{'split':<12}{'MAE':>10}{'RMSE':>10}{'R2':>9}{'DirAcc':>9}"
    print(header)
    for model_name, result in payload["results"].items():
        for row in result["splits"]:
            print(f"{model_name:<14}{row['split']:<12}{row['mae']:>10.6f}"
                  f"{row['rmse']:>10.6f}{row['r2']:>9.4f}"
                  f"{row['directional_accuracy']:>9.4f}")
    chosen = payload["results"]["ridge"]["selection"]["chosen_alpha"]
    print(f"\nridge alpha selected on validation only: {chosen}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
