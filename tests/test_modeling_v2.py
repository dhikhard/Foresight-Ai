"""Tests for Feature Set V2 modeling wiring and the selection protocol.

All fixtures are small synthetic datasets; the real processed CSVs are never
loaded here. Expected values are derived from the fixture construction
formulas, never from the implementation under test.
"""

import csv
import json
import math

import pytest

from backend.modeling.dataset import (
    FEATURE_COLUMNS,
    FEATURE_SETS,
    TARGET_COLUMN,
    build_model_dataset,
    resolve_feature_set,
    split_chronological,
)
from backend.modeling.run_modeling import DEFAULT_ALPHA_GRID, run
from backend.pipeline.config import PipelineConfig
from backend.pipeline.features_v2 import FEATURE_COLUMNS_V2


def date_for(i: int) -> str:
    return f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"


def make_column_synthetic(columns, n_rows, warmup, warmup_col):
    """Feature values are distinct per column (col j at row i = i + j*100),
    which makes column-order mistakes detectable."""
    feature_rows, target_rows = [], []
    for i in range(n_rows):
        row = {"date": date_for(i)}
        for j, col in enumerate(columns):
            row[col] = None if (col == warmup_col and i < warmup) \
                else float(i + j * 100)
        feature_rows.append(row)
        target = {"date": date_for(i)}
        # Final row gets no target: mirrors the real pipeline's tail.
        target[TARGET_COLUMN] = None if i == n_rows - 1 else 0.001 * (i + 1)
        target["target_return_5d"] = None
        target["target_return_10d"] = None
        target["target_future_vol_20"] = None
        target_rows.append(target)
    return feature_rows, target_rows


def make_v2_synthetic(n_rows=70, warmup=25):
    """ma_dist_200 carries the warm-up gap (largest lookback in the set)."""
    return make_column_synthetic(FEATURE_COLUMNS_V2, n_rows, warmup,
                                 warmup_col="ma_dist_200")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------- feature-set resolution ----------

def test_resolve_feature_set_contracts():
    v1 = resolve_feature_set("v1")
    v2 = resolve_feature_set("v2")
    assert v1["columns"] == FEATURE_COLUMNS and len(v1["columns"]) == 10
    assert v2["columns"] == tuple(FEATURE_COLUMNS_V2)
    assert len(v2["columns"]) == 16
    assert v1["features_csv"] == "sp500_features.csv"
    assert v1["report_json"] == "modeling_report.json"
    assert v2["features_csv"] == "sp500_features_v2.csv"
    assert v2["report_json"] == "modeling_report_v2.json"
    assert set(FEATURE_SETS) == {"v1", "v2"}


def test_unknown_feature_set_fails_loudly():
    with pytest.raises(ValueError, match="unknown feature set 'v3'"):
        resolve_feature_set("v3")


# ---------- dataset building with explicit columns ----------

def test_build_model_dataset_accepts_v2_columns_and_preserves_order():
    features, targets = make_v2_synthetic(n_rows=30, warmup=0)
    dataset = build_model_dataset(features, targets,
                                  feature_columns=FEATURE_COLUMNS_V2)
    assert dataset["columns"] == list(FEATURE_COLUMNS_V2)
    assert len(dataset["X"][0]) == 16
    # Exact ordering: row 5 must equal the fixture formula per column.
    assert dataset["X"][5] == [float(5 + j * 100) for j in range(16)]


def test_default_call_still_uses_v1_contract():
    features, targets = make_v2_synthetic(n_rows=5, warmup=0)
    dataset = build_model_dataset(features, targets)
    assert dataset["columns"] == list(FEATURE_COLUMNS)


def test_target_never_in_either_feature_contract():
    for name in ("v1", "v2"):
        columns = resolve_feature_set(name)["columns"]
        assert TARGET_COLUMN not in columns
        assert not any(str(c).startswith("target_") for c in columns)


def test_warmup_drop_accounting_on_v2_dataset():
    n, warmup = 40, 25
    features, targets = make_v2_synthetic(n_rows=n, warmup=warmup)
    dataset = build_model_dataset(features, targets,
                                  feature_columns=FEATURE_COLUMNS_V2)
    assert dataset["dropped_missing_feature"] == warmup
    assert dataset["dropped_missing_target"] == 1
    assert len(dataset["y"]) == n - warmup - 1
    assert dataset["dates"][0] == date_for(warmup)
    assert all(v != 0.0 for row in dataset["X"] for v in row)  # dropped, not filled


# ---------- chronological safety ----------

def test_v2_splits_are_chronological_deterministic_non_overlapping():
    features, targets = make_v2_synthetic(n_rows=70, warmup=25)
    dataset = build_model_dataset(features, targets,
                                  feature_columns=FEATURE_COLUMNS_V2)
    config = PipelineConfig(validation_start="2024-02-01",
                            test_start="2024-03-01")
    parts = split_chronological(dataset, config)

    seen = []
    for name in ("train", "validation", "test"):
        dates = parts[name]["dates"]
        assert dates == sorted(dates), f"{name} not chronological"
        seen.extend(dates)
    assert len(seen) == len(set(seen)) == len(dataset["dates"])

    again_features, again_targets = make_v2_synthetic(n_rows=70, warmup=25)
    again = split_chronological(
        build_model_dataset(again_features, again_targets,
                            feature_columns=FEATURE_COLUMNS_V2), config)
    assert [p["dates"] for p in again.values()] == [p["dates"] for p in parts.values()]


# ---------- end-to-end wiring ----------

def _run_synthetic(tmp_path, feature_set):
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_rows = 70
    if feature_set == "v1":
        features, targets = make_column_synthetic(
            FEATURE_COLUMNS, n_rows=n_rows, warmup=20, warmup_col="ma_200")
    else:
        features, targets = make_v2_synthetic(n_rows=n_rows, warmup=25)
    features_path = tmp_path / "features.csv"
    targets_path = tmp_path / "targets.csv"
    out_path = tmp_path / "report.json"
    write_csv(features_path, features)
    write_csv(targets_path, targets)
    payload = run(
        features_path, targets_path, out_path,
        config=PipelineConfig(validation_start="2024-02-01",
                              test_start="2024-03-01"),
        feature_set=feature_set,
    )
    return payload, out_path


def test_run_v2_end_to_end_excludes_persistence(tmp_path):
    payload, out_path = _run_synthetic(tmp_path, "v2")

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["feature_set"] == "v2"
    assert saved["feature_columns"] == list(FEATURE_COLUMNS_V2)
    # V2 has no same-day return column; ret_lag_1 is NOT a persistence input.
    assert set(saved["results"]) == {"zero_return", "train_mean", "ridge"}
    assert sum(payload["rows"].values()) > 0
    for result in payload["results"].values():
        assert [s["split"] for s in result["splits"]] == \
            ["train", "validation", "test"]
        for row in result["splits"]:
            # Errors must live on raw-return scale (~1e-3), not z-score scale.
            assert row["rmse"] < 1.0


def test_run_v1_end_to_end_includes_persistence(tmp_path):
    payload, _ = _run_synthetic(tmp_path, "v1")
    assert set(payload["results"]) == {"zero_return", "train_mean",
                                       "persistence", "ridge"}


# ---------- protocol discipline ----------

def test_ridge_fit_receives_train_standardized_matrix(tmp_path, monkeypatch):
    """Regression guard for the raw-fit/scaled-predict defect.

    Spies on every RidgeRegressor.fit call during run() and requires the
    captured design matrix to equal the independently reconstructed
    TRAIN-standardized matrix (statistics frozen on TRAIN only). Under the
    old bug the fit received raw X (column means ~j*100), so the comparison
    fails loudly instead of merely producing odd metrics.
    """
    from backend.modeling import run_modeling as rm

    captured = []

    class SpyRidge(rm.RidgeRegressor):
        def fit(self, X, y):
            captured.append([row[:] for row in X])
            return super().fit(X, y)

    monkeypatch.setattr(rm, "RidgeRegressor", SpyRidge)
    payload, _ = _run_synthetic(tmp_path, "v2")
    assert payload["results"]["ridge"]["selection"]["scores"]  # sanity
    # One fit per grid alpha plus one refit at the chosen alpha.
    assert len(captured) == len(DEFAULT_ALPHA_GRID) + 1

    # Independent reconstruction from the fixture CSVs alone: usable rows,
    # chronological train boundary, population statistics over TRAIN rows.
    with open(tmp_path / "features.csv", newline="", encoding="utf-8") as fh:
        feats = list(csv.DictReader(fh))
    with open(tmp_path / "targets.csv", newline="", encoding="utf-8") as fh:
        tgts = list(csv.DictReader(fh))

    dates, raw_X = [], []
    for f, t in zip(feats, tgts):
        vals = [float(f[c]) if f[c] != "" else None for c in FEATURE_COLUMNS_V2]
        tgt = float(t[TARGET_COLUMN]) if t[TARGET_COLUMN] != "" else None
        if any(v is None for v in vals) or tgt is None:
            continue
        dates.append(f["date"])
        raw_X.append(vals)

    n_cols = len(FEATURE_COLUMNS_V2)
    train_idx = [i for i, d in enumerate(dates) if d < "2024-02-01"]
    mu = [sum(raw_X[i][j] for i in train_idx) / len(train_idx)
          for j in range(n_cols)]
    sd = [math.sqrt(sum((raw_X[i][j] - mu[j]) ** 2 for i in train_idx)
                    / len(train_idx)) for j in range(n_cols)]
    expected = [[(raw_X[i][j] - mu[j]) / sd[j] for j in range(n_cols)]
                for i in train_idx]

    for fitted in captured:
        assert len(fitted) == len(expected)
        for row_obs, row_exp in zip(fitted, expected):
            assert row_obs == pytest.approx(row_exp, rel=1e-9, abs=1e-12)

    # Prediction-side consistency: validation must be transformed with the
    # SAME frozen statistics (spot-check via reported preprocessing stats).
    payload_means = _run_synthetic(tmp_path / "b", "v2")[0] \
        ["preprocessing"]["means"]
    assert payload_means == pytest.approx(mu, rel=1e-9, abs=1e-12)


def test_scaling_statistics_are_train_only(tmp_path):
    """Reported means/stds must equal statistics recomputed independently
    from the TRAIN slice of the fixture (formula-derived, not read back from
    the standardizer)."""
    payload, _ = _run_synthetic(tmp_path, "v2")
    means = payload["preprocessing"]["means"]
    stds = payload["preprocessing"]["stds"]

    # Rebuild exactly what run() saw. Fixture months are 28 days:
    # indices 0-27 -> Jan, 28-55 -> Feb, 56+ -> Mar. Usable rows are
    # 25..68 (warm-up 25, tail row 69 lacks a target); train split
    # (< 2024-02-01) is therefore indices 25..27.
    train_indices = range(25, 28)
    for j in range(len(FEATURE_COLUMNS_V2)):
        # Fixture formula: column j at row i holds i + j*100.
        col_values = [i + j * 100 for i in train_indices]
        expected_mean = sum(col_values) / len(col_values)
        expected_std = math.sqrt(
            sum((v - expected_mean) ** 2 for v in col_values) / len(col_values))
        assert means[j] == pytest.approx(expected_mean)
        assert stds[j] == pytest.approx(expected_std)

    # Frozen statistics must differ from full-sample statistics (row 31..68
    # belong to validation/test and must NOT have influenced them).
    full_values = [25 + i + j * 100 for i in range(25, 69)]
    full_mean = sum(full_values) / len(full_values)
    assert means[0] != pytest.approx(full_mean)


def test_ridge_selection_provenance_is_validation_only(tmp_path):
    payload, _ = _run_synthetic(tmp_path, "v2")
    selection = payload["results"]["ridge"]["selection"]
    assert selection["method"] == "validation MAE"
    assert tuple(selection["grid"]) == DEFAULT_ALPHA_GRID
    assert len(selection["scores"]) == len(DEFAULT_ALPHA_GRID)
    assert all("val_mae" in s for s in selection["scores"])
    assert selection["chosen_alpha"] in DEFAULT_ALPHA_GRID
    # TEST metrics must not appear anywhere inside the selection record.
    assert "test_mae" not in json.dumps(selection)


def test_protocol_block_records_experiment_provenance(tmp_path):
    payload, _ = _run_synthetic(tmp_path, "v2")
    protocol = payload["protocol"]
    assert protocol["split_boundaries"] == {
        "train": "< 2024-02-01",
        "validation": "2024-02-01 <= d < 2024-03-01",
        "test": ">= 2024-03-01",
    }
    assert "close(t+1)/close(t)-1" in protocol["prediction_semantics"]
    assert protocol["alpha_selection"]["grid"] == list(DEFAULT_ALPHA_GRID)
    assert protocol["alpha_selection"]["criterion"] == "validation MAE"
    assert "never used" in protocol["test_discipline"] or \
        "not" in protocol["test_discipline"]


# ---------- reproducibility ----------

def test_repeated_runs_are_deterministic_modulo_timestamp(tmp_path):
    payload_a, out_a = _run_synthetic(tmp_path / "a", "v2")
    payload_b, out_b = _run_synthetic(tmp_path / "b", "v2")
    a = json.loads(out_a.read_text(encoding="utf-8"))
    b = json.loads(out_b.read_text(encoding="utf-8"))
    assert "generated_at_utc" in a and "generated_at_utc" in b
    a.pop("generated_at_utc")
    b.pop("generated_at_utc")
    assert a == b
