"""Tests for the ForeSight modeling layer.

All fixtures are small synthetic datasets; the real processed CSVs are never
loaded here.
"""

import math

import pytest

from backend.modeling.baselines import (
    PersistenceBaseline,
    TrainMeanBaseline,
    ZeroBaseline,
)
from backend.modeling.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DatasetError,
    build_model_dataset,
    split_chronological,
)
from backend.modeling.evaluation import metrics
from backend.modeling.preprocessing import TrainOnlyStandardizer
from backend.modeling.regression import RidgeRegressor
from backend.pipeline.config import PipelineConfig


def date_for(i: int) -> str:
    return f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"


def make_synthetic(n_rows=25, ma200_warmup=20, include_targets=True):
    """Feature values are distinct per column (col j at row i = i + j*100),
    which makes column-order mistakes detectable."""
    feature_rows, target_rows = [], []
    for i in range(n_rows):
        row = {"date": date_for(i)}
        for j, col in enumerate(FEATURE_COLUMNS):
            if col == "return":
                # Realistic fraction scale, strictly positive (never reads as
                # a zero-fill sentinel), unique per row, far from the >=100
                # offsets of the other columns.
                row[col] = 0.001 + 0.0001 * i
            else:
                row[col] = None if (col == "ma_200" and i < ma200_warmup) else float(i + j * 100)
        feature_rows.append(row)
        target = {"date": date_for(i)}
        if include_targets:
            # Final row gets no target: mirrors the real pipeline's tail,
            # where no next-day close exists yet.
            value = None if i == n_rows - 1 else 0.001 * (i + 1)
            target[TARGET_COLUMN] = value
            target["target_return_5d"] = None
            target["target_return_10d"] = None
            target["target_future_vol_20"] = None
        target_rows.append(target)
    return feature_rows, target_rows


def build_synthetic(**kwargs):
    features, targets = make_synthetic(**kwargs)
    return build_model_dataset(features, targets)


# ---------- 1/2. contract ----------

def test_feature_columns_are_exact_contract():
    assert FEATURE_COLUMNS == (
        "return", "log_return", "ma_20", "ma_50", "ma_200",
        "vol_20_annualized", "rolling_peak", "drawdown",
        "daily_range", "open_close_return",
    )
    assert TARGET_COLUMN == "target_return_1d"


def test_target_cannot_appear_in_X():
    dataset = build_synthetic(n_rows=23, ma200_warmup=0)
    assert TARGET_COLUMN not in dataset["columns"]
    assert len(dataset["X"][0]) == len(FEATURE_COLUMNS)
    # Forward targets supplied alongside must never leak into X columns.
    assert not any(str(c).startswith("target_") for c in dataset["columns"])
    # Column-order integrity: row 5 must equal the fixture formula per column
    # ('return' is small-scale; every other column carries a j*100 offset).
    assert dataset["X"][5] == [0.0015] + [
        float(5 + j * 100) for j in range(1, len(FEATURE_COLUMNS))
    ]


# ---------- 3. missing-value policy ----------

def test_missing_value_policy_drops_warmup_and_tail():
    n, warmup = 25, 20
    dataset = build_synthetic(n_rows=n, ma200_warmup=warmup)
    # Rows 0..19 lack ma_200 -> dropped. Final row lacks next-day target ->
    # dropped. Usable: indices 20..23 (4 rows).
    assert len(dataset["y"]) == n - warmup - 1
    assert dataset["dropped_missing_feature"] == warmup
    assert dataset["dropped_missing_target"] == 1
    assert dataset["dates"][0] == date_for(warmup)
    assert dataset["dates"][-1] == date_for(n - 2)
    # Dropped, not imputed: synthetic values are all nonzero, so any
    # zero-fill imputation would show up as 0.0 entries here.
    assert all(v != 0.0 for row in dataset["X"] for v in row)


def test_misaligned_dates_rejected():
    features, targets = make_synthetic(n_rows=5, ma200_warmup=0)
    targets = targets[1:]
    with pytest.raises(DatasetError, match="do not align"):
        build_model_dataset(features, targets)


# ---------- 4/5. train-only preprocessing ----------

def test_standardizer_fit_on_train_only():
    full = build_synthetic(n_rows=60, ma200_warmup=0)
    parts = split_chronological(
        full, PipelineConfig(validation_start="2024-02-01", test_start="2024-03-01"))
    train = parts["train"]
    standardizer = TrainOnlyStandardizer().fit(train["X"])

    col0 = [row[0] for row in train["X"]]
    expected_mean = sum(col0) / len(col0)
    expected_std = math.sqrt(sum((v - expected_mean) ** 2 for v in col0) / len(col0))
    assert standardizer.means[0] == pytest.approx(expected_mean)
    assert standardizer.stds[0] == pytest.approx(expected_std)

    # Full-dataset statistics MUST differ from train-only statistics.
    full_col0 = [row[0] for row in full["X"]]
    full_mean = sum(full_col0) / len(full_col0)
    assert standardizer.means[0] != pytest.approx(full_mean)


def test_transform_does_not_alter_fitted_statistics():
    dataset = build_synthetic(n_rows=60, ma200_warmup=0)
    parts = split_chronological(
        dataset, PipelineConfig(validation_start="2024-02-01",
                                test_start="2024-03-01"))
    standardizer = TrainOnlyStandardizer().fit(parts["train"]["X"])
    before = (list(standardizer.means), list(standardizer.stds))

    for name in ("train", "validation", "test"):
        transformed = standardizer.transform(parts[name]["X"])
        assert len(transformed) == len(parts[name]["X"])

    assert (standardizer.means, standardizer.stds) == before

    # Transformed TRAIN has ~zero mean per column.
    for j in range(len(FEATURE_COLUMNS)):
        col = [row[j] for row in standardizer.transform(parts["train"]["X"])]
        assert abs(sum(col) / len(col)) < 1e-9


def test_standardizer_zero_variance_and_serialization():
    X = [[1.0, 5.0], [1.0, 7.0], [1.0, 6.0]]
    s = TrainOnlyStandardizer().fit(X)
    assert s.stds[0] == 1.0  # zero-variance guard
    clone = TrainOnlyStandardizer.from_dict(s.to_dict())
    assert clone.transform(X) == s.transform(X)


# ---------- 6/7. baselines ----------

def test_zero_baseline_predicts_exact_zeros():
    model = ZeroBaseline().fit([[1.0]], [0.05])
    assert model.predict([[9.0] * 10, [-3.0] * 10]) == [0.0, 0.0]


def test_mean_baseline_frozen_on_train_only():
    train_y = [0.01, 0.03]
    other_split_y = [0.5, -0.7]
    model = TrainMeanBaseline().fit([], train_y)
    predictions = model.predict([[], []])
    assert predictions[0] == pytest.approx(0.02)
    assert all(p == pytest.approx(sum(train_y) / 2) for p in predictions)
    assert 0.5 not in predictions and -0.7 not in predictions


def test_persistence_baseline_reads_return_column():
    X = [[0.012, 99], [-0.004, 98]]
    model = PersistenceBaseline(return_column_index=0).fit(X, [])
    assert model.predict(X) == [0.012, -0.004]


# ---------- 8. chronological safety ----------

def test_splits_are_chronological_non_overlapping_deterministic():
    dataset = build_synthetic(n_rows=70, ma200_warmup=0)
    config = PipelineConfig(validation_start="2024-02-01", test_start="2024-03-01")
    parts = split_chronological(dataset, config)

    seen = []
    for name in ("train", "validation", "test"):
        dates = parts[name]["dates"]
        assert dates == sorted(dates), f"{name} not chronological"
        seen.extend(dates)
    assert len(seen) == len(set(seen)) == len(dataset["dates"])

    again = split_chronological(build_synthetic(n_rows=70, ma200_warmup=0), config)
    assert [p["dates"] for p in again.values()] == [p["dates"] for p in parts.values()]


# ---------- 9. evaluation metrics ----------

def test_evaluation_metrics_mathematically_correct():
    y_true = [1.0, -2.0, 3.0]
    y_pred = [2.0, 0.0, 1.0]
    m = metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(5.0 / 3.0)
    assert m["rmse"] == pytest.approx(math.sqrt(3.0))
    assert m["r2"] == pytest.approx(33.0 / 114.0)   # hand-derived
    assert m["directional_accuracy"] == pytest.approx(2.0 / 3.0)
    assert m["mean_actual_return"] == pytest.approx(2.0 / 3.0)
    assert m["mean_predicted_return"] == pytest.approx(1.0)
    assert m["prediction_std"] == pytest.approx(math.sqrt(2.0 / 3.0))


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics([1.0], [1.0, 2.0])


# ---------- ridge correctness ----------

def test_ridge_recovers_exact_linear_relationship():
    X = [[float(i), float(i % 5)] for i in range(30)]
    y = [2.0 * row[0] - 1.0 * row[1] + 3.0 for row in X]
    model = RidgeRegressor(alpha=1e-6).fit(X, y)
    assert model.weights[0] == pytest.approx(2.0, abs=1e-6)
    assert model.weights[1] == pytest.approx(-1.0, abs=1e-6)
    assert model.intercept == pytest.approx(3.0, abs=1e-5)
    assert model.predict(X)[7] == pytest.approx(y[7], rel=1e-6)


# ---------- end-to-end smoke test ----------

def test_run_modeling_end_to_end_on_synthetic_csvs(tmp_path):
    """Full run() on tiny synthetic CSVs.

    Guards the wiring: persistence must be scored on RAW features (a
    standardized matrix would inflate its errors ~100x), and every model
    must produce metrics on all three splits.
    """
    import csv
    import json

    from backend.modeling.run_modeling import run

    n_rows, warmup = 70, 20
    features, targets = make_synthetic(n_rows=n_rows, ma200_warmup=warmup)

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    features_path = tmp_path / "features.csv"
    targets_path = tmp_path / "targets.csv"
    out_path = tmp_path / "report.json"
    write_csv(features_path, features)
    write_csv(targets_path, targets)

    payload = run(
        features_path, targets_path, out_path,
        config=PipelineConfig(validation_start="2024-02-01",
                              test_start="2024-03-01"),
    )

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(saved["results"]) == {"zero_return", "train_mean",
                                     "persistence", "ridge", "elastic_net"}
    assert sum(payload["rows"].values()) > 0
    for result in payload["results"].values():
        splits_scored = [s["split"] for s in result["splits"]]
        assert splits_scored == ["train", "validation", "test"]
        for row in result["splits"]:
            # Errors must live on raw-return scale (~1e-3), not z-score
            # scale (~1). Catches accidental use of transformed inputs.
            assert row["rmse"] < 1.0
