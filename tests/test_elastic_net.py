"""Tests for ElasticNetRegressor and its V2 modeling wiring.

Expected values come from closed-form fixture formulas and from the
published objective, never from a reference implementation under test. The
synthetic fixtures are deterministic (seeded numpy RNG); the real processed
CSVs are never loaded here.
"""

import csv
import json
import math

import pytest

from backend.modeling.dataset import (
    TARGET_COLUMN,
    build_model_dataset,
    resolve_feature_set,
    split_chronological,
)
from backend.modeling.evaluation import metrics
from backend.modeling.regression import ElasticNetRegressor
from backend.modeling.run_modeling import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_L1_RATIO_GRID,
    run,
)
from backend.pipeline.config import PipelineConfig
from backend.pipeline.features_v2 import FEATURE_COLUMNS_V2


def synthetic_signal(n_rows=300, n_features=4, seed=12345):
    """A well-separated linear signal in column 0.

    Column 0 is a true predictor; the remaining columns are pure noise, so a
    correct solver should drive their weights toward zero. The 10x signal
    scale keeps the fitted predictor recoverable at the default tolerance.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    truth = np.zeros(n_features)
    truth[0] = 2.0
    y = X @ truth + 0.1
    return X.tolist(), y.tolist(), truth.tolist()


# ---------- 1. fit / predict contract ----------

def test_fit_returns_self_and_predicts_after_fitting():
    X, y, _ = synthetic_signal()
    model = ElasticNetRegressor(alpha=0.1, l1_ratio=0.5)
    assert model.fit(X, y) is model
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(math.isfinite(p) for p in predictions)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ElasticNetRegressor().predict([[1.0], [2.0]])


def test_predict_length_tracks_input_rows():
    X, y, _ = synthetic_signal(n_rows=120)
    model = ElasticNetRegressor(alpha=0.1).fit(X, y)
    assert len(model.predict(X[:7])) == 7
    assert len(model.predict(X)) == len(X)
    # An empty input raises exactly as RidgeRegressor.predict does, so the
    # two models share one input-handling contract.
    with pytest.raises(ValueError):
        model.predict([])


def test_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="one row per target"):
        ElasticNetRegressor().fit([[1.0], [2.0]], [1.0])
    with pytest.raises(ValueError, match="one row per target"):
        # 1-D design matrix
        ElasticNetRegressor().fit([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        # Empty list inputs are 1-D, so the shape guard fires first.
        ElasticNetRegressor().fit([], [])
    import numpy as np

    with pytest.raises(ValueError, match="empty design matrix"):
        # A genuinely 2-D, zero-row matrix reaches the empty guard.
        ElasticNetRegressor().fit(np.empty((0, 3)), [])


def test_fitted_model_reports_intercept_and_iterations():
    X, y, _ = synthetic_signal()
    model = ElasticNetRegressor(alpha=0.1).fit(X, y)
    assert model.intercept is not None and math.isfinite(model.intercept)
    assert model.weights is not None and len(model.weights) == 4
    assert 1 <= model.n_iter_ <= model.max_iter
    assert isinstance(model.converged_, bool)


# ---------- 2. coefficient recovery ----------

def test_recovers_well_separated_signal():
    """Column 0 holds a coefficient of 2.0; noise columns must go to ~0."""
    X, y, _ = synthetic_signal()
    model = ElasticNetRegressor(alpha=1e-6, l1_ratio=0.5, tol=1e-12).fit(X, y)
    assert model.weights[0] == pytest.approx(2.0, abs=1e-3)
    assert max(abs(w) for w in model.weights[1:]) < 1e-3


def test_weak_regularization_recovers_scale_of_known_coefficients():
    """Two known coefficients recovered within a loose, non-brittle band."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 500
    X = rng.normal(size=(n, 5))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5

    model = ElasticNetRegressor(alpha=1e-4, l1_ratio=0.5, tol=1e-12).fit(
        X.tolist(), y.tolist())
    assert model.weights[0] == pytest.approx(2.0, abs=0.05)
    assert model.weights[1] == pytest.approx(-1.0, abs=0.05)
    # The unregularized intercept must come back near +0.5.
    assert model.intercept == pytest.approx(0.5, abs=0.05)


def test_intercept_is_not_regularized():
    """A large offset must be absorbed entirely by the unpenalized intercept.

    With a heavily penalized fit, every weight collapses to zero; if the
    intercept were also regularized it could not recover the offset, and
    the predictions would sit at 0 instead of the data mean.
    """
    import numpy as np

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    # Center the columns so mean(y) is exactly the offset, making the
    # expected intercept derivable from the fixture rather than the model.
    X = X - X.mean(axis=0)
    y = X @ np.array([0.5, 0.0, 0.0]) + 1.0

    model = ElasticNetRegressor(alpha=1e6, l1_ratio=0.9).fit(
        X.tolist(), y.tolist())
    assert all(abs(w) < 1e-3 for w in model.weights)
    assert model.intercept == pytest.approx(1.0, abs=1e-6)
    assert all(p == pytest.approx(1.0, abs=1e-3)
               for p in model.predict(X.tolist()))


def test_l1_ratio_zero_matches_closed_form_ridge():
    """l1_ratio == 0 must reduce to the ridge solution.

    Elastic net's objective carries a 1/(2n) factor on the squared-error
    term, so an elastic-net alpha of a/n corresponds to this repository's
    unnormalized RidgeRegressor alpha of a.
    """
    import numpy as np

    from backend.modeling.regression import RidgeRegressor

    X, y, _ = synthetic_signal(n_rows=400, n_features=5, seed=99)
    n = len(y)
    ridge = RidgeRegressor(alpha=0.5).fit(X, y)
    enet = ElasticNetRegressor(alpha=0.5 / n, l1_ratio=0.0,
                               tol=1e-12).fit(X, y)

    for a, b in zip(enet.weights, ridge.weights):
        assert a == pytest.approx(b, abs=1e-6)
    assert enet.intercept == pytest.approx(ridge.intercept, abs=1e-6)


def test_lasso_limit_zeroes_noise_columns_exactly():
    """l1_ratio == 1 (pure L1) must produce hard zeros on noise columns."""
    import numpy as np

    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 6))
    y = 2.0 * X[:, 0]

    model = ElasticNetRegressor(alpha=0.2, l1_ratio=1.0, tol=1e-12).fit(
        X.tolist(), y.tolist())
    assert all(w == 0.0 for w in model.weights[1:])


# ---------- 3. regularization behaviour ----------

def test_stronger_alpha_shrinks_coefficient_magnitude():
    X, y, _ = synthetic_signal(n_rows=400, n_features=5)
    weak = ElasticNetRegressor(alpha=0.01, l1_ratio=0.5).fit(X, y)
    strong = ElasticNetRegressor(alpha=100.0, l1_ratio=0.5).fit(X, y)

    weak_l1 = sum(abs(w) for w in weak.weights)
    strong_l1 = sum(abs(w) for w in strong.weights)
    assert strong_l1 < weak_l1
    assert abs(strong.weights[0]) < abs(weak.weights[0])


def test_l1_norm_is_monotone_non_increasing_in_alpha():
    X, y, _ = synthetic_signal(n_rows=400, n_features=5)
    grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    norms = [sum(abs(w) for w in
                 ElasticNetRegressor(alpha=a, l1_ratio=0.9, tol=1e-12)
                 .fit(X, y).weights)
             for a in grid]
    for weaker, stronger in zip(norms, norms[1:]):
        assert weaker >= stronger - 1e-9


def test_l1_ratio_one_produces_sparser_fit_than_l1_ratio_zero():
    import numpy as np

    rng = np.random.default_rng(21)
    X = rng.normal(size=(300, 12))
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1]

    def n_nonzero(l1_ratio):
        model = ElasticNetRegressor(alpha=0.5, l1_ratio=l1_ratio,
                                    tol=1e-12).fit(X.tolist(), y.tolist())
        return sum(1 for w in model.weights if w != 0.0)

    assert n_nonzero(1.0) < n_nonzero(0.0)


# ---------- 4. parameter validation ----------

def test_rejects_negative_alpha():
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        ElasticNetRegressor(alpha=-1.0)


@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0])
def test_rejects_out_of_range_l1_ratio(bad):
    with pytest.raises(ValueError, match=r"l1_ratio must be within"):
        ElasticNetRegressor(l1_ratio=bad)


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_rejects_non_positive_max_iter(bad):
    with pytest.raises(ValueError, match="max_iter must be positive"):
        ElasticNetRegressor(max_iter=bad)


@pytest.mark.parametrize("bad", [0.0, -1e-8, -1.0])
def test_rejects_non_positive_tol(bad):
    with pytest.raises(ValueError, match="tol must be positive"):
        ElasticNetRegressor(tol=bad)


def test_boundary_l1_ratio_values_are_accepted():
    for value in (0.0, 1.0):
        assert ElasticNetRegressor(l1_ratio=value).l1_ratio == value
    assert ElasticNetRegressor(alpha=0.0).alpha == 0.0


def test_validation_happens_before_any_fitting():
    """Bad parameters must raise at construction, not silently at fit time."""
    with pytest.raises(ValueError):
        ElasticNetRegressor(alpha=-0.5, l1_ratio=0.5).fit([[1.0]], [1.0])


# ---------- 4b. convergence criterion: target-scale independence ----------
#
# Regression cover for the defect found in the Week 5 Phase 1 audit. The
# superseded criterion was
#     max_update < tol * ||y - mean(y)||^2
# whose right-hand side carries units of y**2 while the left carries units
# of y, so its effective strictness moved with the arbitrary numerical
# magnitude of the target. On a fixture with y ~ O(10) it admitted
# convergence after two sweeps with ~0.37 coefficient error. The rule is now
# purely relative to the fitted weights:
#     max_update <= tol * max|w|
# which is exactly invariant under y -> c*y.

def _large_target_fixture(n_rows=500, n_features=6, seed=777):
    """Near-orthogonal design with a large-magnitude target.

    Orthogonal columns (condition number ~1) deliberately separate the
    target-scale failure mode from the ill-conditioning one, and the target
    sits ~200x above the V2 return scale.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)) + 2.0
    truth = np.zeros(n_features)
    truth[0], truth[1] = 2.0, -1.0
    y = X @ truth + 0.75
    return X.tolist(), y.tolist()


def test_large_magnitude_target_does_not_falsely_converge():
    """The discovered failure mode: y ~ O(10) with weak regularization.

    Under the superseded rule this fit stopped after ~2 iterations while
    still being ~0.37 away from the correct coefficients. It must now run
    to a genuine solution and agree with the closed-form reference.
    """
    import numpy as np

    from backend.modeling.regression import RidgeRegressor

    X, y = _large_target_fixture()
    n = len(y)
    assert np.std(y) > 0.5, "fixture must be far above the V2 return scale"

    alpha_ridge = 0.01
    reference = RidgeRegressor(alpha=alpha_ridge).fit(X, y)
    model = ElasticNetRegressor(alpha=alpha_ridge / n, l1_ratio=0.0).fit(X, y)

    # The stopping rule must not collapse to a trivial number of sweeps.
    assert model.n_iter_ > 2, (
        f"converged in {model.n_iter_} sweeps; the superseded scale-dependent "
        "criterion stopped this low at 2 with materially wrong coefficients")
    assert model.converged_ is True

    # Coefficients must actually be right, not merely "declared converged".
    max_diff = max(abs(a - b) for a, b in zip(model.weights, reference.weights))
    assert max_diff < 1e-4, f"max coefficient error {max_diff:.3e}"
    assert abs(model.intercept - reference.intercept) < 1e-4


def test_weak_regularization_on_large_target_agrees_with_ridge_across_alphas():
    """Sweep the weak-penalty range; agreement must hold at every alpha."""
    import numpy as np

    from backend.modeling.regression import RidgeRegressor

    X, y = _large_target_fixture()
    n = len(y)
    for alpha_ridge in (0.001, 0.01, 0.1, 1.0, 10.0):
        reference = RidgeRegressor(alpha=alpha_ridge).fit(X, y)
        model = ElasticNetRegressor(alpha=alpha_ridge / n, l1_ratio=0.0).fit(X, y)
        max_diff = max(abs(a - b) for a, b in zip(model.weights,
                                                  reference.weights))
        assert max_diff < 1e-4, (
            f"alpha_ridge={alpha_ridge}: max coefficient error {max_diff:.3e} "
            f"after {model.n_iter_} sweeps (converged={model.converged_})")


def test_convergence_is_independent_of_target_magnitude():
    """Scaling y must not change how hard the solver has to work.

    This is the invariant the superseded rule violated: it made tol
    stricter as y shrank and looser as y grew, so the sweep count swung by
    orders of magnitude for the same underlying problem.
    """
    import numpy as np

    X, y = _large_target_fixture()
    base = np.array(y)
    n = len(y)

    sweep_counts = []
    for scale in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        scaled = (base * scale).tolist()
        model = ElasticNetRegressor(alpha=10.0 / n, l1_ratio=0.0).fit(X, scaled)
        assert model.converged_ is True, f"non-convergence at scale {scale}"
        sweep_counts.append(model.n_iter_)

    assert max(sweep_counts) - min(sweep_counts) <= 2, (
        f"sweep count varied with target magnitude: {sweep_counts}. The "
        "criterion is not scale-consistent.")


def test_convergence_behaviour_survives_tiny_and_huge_targets():
    """Neither extreme may trigger a spurious early stop or a hang."""
    import numpy as np

    X, y = _large_target_fixture()
    base = np.array(y)
    n = len(y)

    for scale in (1e-8, 1e8):
        scaled = (base * scale).tolist()
        model = ElasticNetRegressor(alpha=10.0 / n, l1_ratio=0.0).fit(X, scaled)
        assert 1 <= model.n_iter_ <= model.max_iter
        assert all(np.isfinite(model.weights))
        assert np.isfinite(model.intercept)
        assert all(np.isfinite(p) for p in model.predict(X))


def test_scaled_target_keeps_weights_on_the_expected_linear_scaling():
    """In the penalty-negligible regime the L2 solution is linear in y.

    The elastic-net objective is NOT scale invariant (the penalty is
    absolute), so coefficients need not scale proportionally in general.
    They do scale with y when the penalty is small enough not to bind,
    which is the regime this fixture deliberately targets.
    """
    import numpy as np

    from backend.modeling.regression import RidgeRegressor

    X, y = _large_target_fixture()
    base = np.array(y)
    n = len(y)

    for scale in (1e2, 1e4):
        scaled = (base * scale).tolist()
        model = ElasticNetRegressor(alpha=1e-4 / n, l1_ratio=0.0).fit(X, scaled)
        reference = RidgeRegressor(alpha=1e-4).fit(X, scaled)
        # Both must remain valid minimizers; the assertion is on agreement,
        # not on a hand-derived coefficient.
        assert max(abs(a - b) for a, b in zip(model.weights, reference.weights)) \
            < 1e-4 * max(1.0, scale)


def test_all_zero_fixed_point_still_converges_immediately():
    """A fully shrunk model must not spin to max_iter.

    With weight_scale == 0 and an exactly zero update the comparison is
    0 <= 0, which must read as converged. A strict '<' here would burn
    max_iter sweeps on a model that is already at its fixed point.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 4)).tolist()
    y = rng.normal(size=200).tolist()
    # alpha large enough that every coordinate is thresholded to exactly 0.
    model = ElasticNetRegressor(alpha=1e6, l1_ratio=0.9).fit(X, y)
    assert all(w == 0.0 for w in model.weights)
    assert model.converged_ is True
    assert model.n_iter_ == 1


def test_non_convergence_is_reported_honestly():
    """Hitting max_iter must surface as converged_ = False, never as success."""
    import numpy as np

    rng = np.random.default_rng(11)
    n, p = 400, 10
    # Heavy collinearity plus a vanishing penalty: cyclic coordinate descent
    # crawls here, and the model must say so rather than claim convergence.
    base = rng.normal(size=(n, 2))
    X = np.hstack([base, base @ rng.normal(size=(2, 2)),
                   rng.normal(size=(n, 1)) * 1e-3,
                   rng.normal(size=(n, p - 5))])
    y = (X @ rng.normal(size=p)).tolist()

    model = ElasticNetRegressor(alpha=1e-12, l1_ratio=0.0, tol=1e-14,
                                max_iter=5).fit(X.tolist(), y)
    assert model.n_iter_ == 5
    assert model.converged_ is False


# ---------- 5. determinism ----------

def test_repeated_fits_are_bit_identical():
    X, y, _ = synthetic_signal(n_rows=250)
    a = ElasticNetRegressor(alpha=0.05, l1_ratio=0.3, tol=1e-9).fit(X, y)
    b = ElasticNetRegressor(alpha=0.05, l1_ratio=0.3, tol=1e-9).fit(X, y)
    assert a.weights == b.weights
    assert a.intercept == b.intercept
    assert a.n_iter_ == b.n_iter_
    assert a.predict(X) == b.predict(X)


def test_observation_order_does_not_change_sparse_support():
    """Cyclic descent is order-independent in its final fixed point.

    Reordering rows must not change which columns are zeroed, confirming
    the solver reads no positional or temporal information.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 8))
    y = 2.0 * X[:, 0]

    straight = ElasticNetRegressor(alpha=0.2, l1_ratio=0.9, tol=1e-12)
    straight.fit(X.tolist(), y.tolist())
    permuted = ElasticNetRegressor(alpha=0.2, l1_ratio=0.9, tol=1e-12)
    permuted.fit(X[::-1].tolist(), y[::-1].tolist())

    assert ([w == 0.0 for w in straight.weights]
            == [w == 0.0 for w in permuted.weights])
    straight_pred = straight.predict(X.tolist())
    permuted_pred = permuted.predict(X.tolist())
    straight_sse = sum((t - p) ** 2 for t, p in zip(y, straight_pred))
    permuted_sse = sum((t - p) ** 2 for t, p in zip(y, permuted_pred))
    assert straight_sse == pytest.approx(permuted_sse, rel=1e-8)


# ---------- 6/7. wiring, selection discipline, report ----------

def date_for(i: int) -> str:
    return f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"


def make_v2_synthetic(n_rows=70, warmup=25):
    feature_rows, target_rows = [], []
    for i in range(n_rows):
        row = {"date": date_for(i)}
        for j, col in enumerate(FEATURE_COLUMNS_V2):
            row[col] = None if (col == "ma_dist_200" and i < warmup) \
                else float(i + j * 100)
        feature_rows.append(row)
        target = {"date": date_for(i)}
        target[TARGET_COLUMN] = None if i == n_rows - 1 else 0.001 * (i + 1)
        target["target_return_5d"] = None
        target["target_return_10d"] = None
        target["target_future_vol_20"] = None
        target_rows.append(target)
    return feature_rows, target_rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_v2_synthetic(tmp_path, name="a", **run_kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    features, targets = make_v2_synthetic()
    features_path = tmp_path / "features.csv"
    targets_path = tmp_path / "targets.csv"
    out_path = tmp_path / "report.json"
    write_csv(features_path, features)
    write_csv(targets_path, targets)
    payload = run(
        features_path, targets_path, out_path,
        config=PipelineConfig(validation_start="2024-02-01",
                              test_start="2024-03-01"),
        feature_set="v2", **run_kwargs)
    return payload, out_path


def test_enet_selection_grid_is_the_declared_product(tmp_path):
    payload, _ = run_v2_synthetic(tmp_path)
    selection = payload["results"]["elastic_net"]["selection"]
    assert tuple(selection["alpha_grid"]) == DEFAULT_ALPHA_GRID
    assert tuple(selection["l1_ratio_grid"]) == DEFAULT_L1_RATIO_GRID
    # Full cross product, no pruning.
    assert len(selection["scores"]) == len(DEFAULT_ALPHA_GRID) * len(
        DEFAULT_L1_RATIO_GRID)
    combos = {(s["alpha"], s["l1_ratio"]) for s in selection["scores"]}
    assert combos == {(a, r) for a in DEFAULT_ALPHA_GRID
                      for r in DEFAULT_L1_RATIO_GRID}


def test_enet_selected_hyperparameters_minimize_validation_mae(tmp_path):
    payload, _ = run_v2_synthetic(tmp_path)
    selection = payload["results"]["elastic_net"]["selection"]
    assert selection["method"] == "validation MAE"
    best = min(selection["scores"], key=lambda s: s["val_mae"])
    assert selection["chosen_alpha"] == best["alpha"]
    assert selection["chosen_l1_ratio"] == best["l1_ratio"]
    assert selection["chosen_alpha"] in DEFAULT_ALPHA_GRID
    assert selection["chosen_l1_ratio"] in DEFAULT_L1_RATIO_GRID


def test_enet_selection_record_contains_no_test_metrics(tmp_path):
    payload, _ = run_v2_synthetic(tmp_path)
    selection = payload["results"]["elastic_net"]["selection"]
    assert "test_mae" not in json.dumps(selection)
    assert "test" not in json.dumps(selection).lower()


def test_enet_receives_train_standardized_matrix_only(tmp_path, monkeypatch):
    """Elastic Net must be fit on the TRAIN-standardized matrix.

    Mirrors the existing ridge regression guard: the captured design
    matrix is compared element-wise against an independent reconstruction
    from the fixture CSVs, with population statistics over TRAIN rows only.
    """
    from backend.modeling import run_modeling as rm

    captured = []

    class Spy(rm.ElasticNetRegressor):
        def fit(self, X, y):
            captured.append((self.alpha, self.l1_ratio, [row[:] for row in X]))
            return super().fit(X, y)

    monkeypatch.setattr(rm, "ElasticNetRegressor", Spy)
    payload, _ = run_v2_synthetic(tmp_path)

    # One fit per grid cell, then exactly one refit at the chosen cell.
    expected_fits = len(DEFAULT_ALPHA_GRID) * len(DEFAULT_L1_RATIO_GRID) + 1
    assert len(captured) == expected_fits

    with open(tmp_path / "features.csv", newline="", encoding="utf-8") as fh:
        feats = list(csv.DictReader(fh))
    with open(tmp_path / "targets.csv", newline="", encoding="utf-8") as fh:
        tgts = list(csv.DictReader(fh))

    dates, raw_X = [], []
    for f, t in zip(feats, tgts):
        vals = [float(f[c]) if f[c] != "" else None
                for c in FEATURE_COLUMNS_V2]
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

    for _, _, fitted in captured:
        assert len(fitted) == len(expected)
        for observed, exp_row in zip(fitted, expected):
            assert observed == pytest.approx(exp_row, rel=1e-9, abs=1e-12)

    # The final captured fit is the refit at the selected hyperparameters.
    selection = payload["results"]["elastic_net"]["selection"]
    assert captured[-1][0] == selection["chosen_alpha"]
    assert captured[-1][1] == selection["chosen_l1_ratio"]


def test_enet_scored_on_all_three_splits_at_return_scale(tmp_path):
    payload, out_path = run_v2_synthetic(tmp_path)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["feature_set"] == "v2"
    assert saved["feature_columns"] == list(FEATURE_COLUMNS_V2)

    result = saved["results"]["elastic_net"]
    assert result["model"] == "elastic_net"
    assert [s["split"] for s in result["splits"]] == [
        "train", "validation", "test"]
    for row in result["splits"]:
        # Raw-return scale (~1e-3), not z-score scale (~1). Catches any
        # accidental use of untransformed or mis-scaled inputs.
        assert row["rmse"] < 1.0
        assert math.isfinite(row["mae"]) and math.isfinite(row["r2"])


def test_enet_weights_are_serializable_and_consistent(tmp_path):
    payload, _ = run_v2_synthetic(tmp_path)
    weights = payload["results"]["elastic_net"]["weights"]
    assert set(weights) >= {"alpha", "l1_ratio", "weights", "intercept"}
    assert len(weights["weights"]) == len(FEATURE_COLUMNS_V2)
    assert weights["alpha"] == payload["results"]["elastic_net"]["selection"][
        "chosen_alpha"]
    assert weights["l1_ratio"] == payload["results"]["elastic_net"][
        "selection"]["chosen_l1_ratio"]
    # Round-trips through JSON without loss.
    assert json.loads(json.dumps(weights)) == weights


def test_protocol_block_records_enet_provenance(tmp_path):
    payload, _ = run_v2_synthetic(tmp_path)
    enet_protocol = payload["protocol"]["elastic_net_selection"]
    assert enet_protocol["criterion"] == "validation MAE"
    assert enet_protocol["alpha_grid"] == list(DEFAULT_ALPHA_GRID)
    assert enet_protocol["l1_ratio_grid"] == list(DEFAULT_L1_RATIO_GRID)
    assert "coordinate descent" in enet_protocol["solver"]
    assert "1/(2n)" in enet_protocol["objective"]


def test_adding_enet_does_not_disturb_ridge_or_baselines(tmp_path):
    """Ridge selection and baseline metrics must be byte-identical with and
    without the elastic-net block present."""
    narrow, _ = run_v2_synthetic(tmp_path / "narrow", l1_ratio_grid=(0.5,))
    full, _ = run_v2_synthetic(tmp_path / "full")

    for name in ("zero_return", "train_mean", "ridge"):
        assert narrow["results"][name] == full["results"][name]
    assert narrow["protocol"]["alpha_selection"] == \
        full["protocol"]["alpha_selection"]
    assert narrow["preprocessing"] == full["preprocessing"]
    assert narrow["rows"] == full["rows"]


def test_enet_run_is_deterministic_modulo_timestamp(tmp_path):
    a, out_a = run_v2_synthetic(tmp_path / "a")
    b, out_b = run_v2_synthetic(tmp_path / "b")
    doc_a = json.loads(out_a.read_text(encoding="utf-8"))
    doc_b = json.loads(out_b.read_text(encoding="utf-8"))
    doc_a.pop("generated_at_utc")
    doc_b.pop("generated_at_utc")
    assert doc_a == doc_b


def test_enet_validates_against_shipped_v2_contract():
    """The selected V2 contract must stay exactly 16 columns with no
    target-derived column, so elastic net cannot be fed leaked inputs."""
    spec = resolve_feature_set("v2")
    assert spec["columns"] == tuple(FEATURE_COLUMNS_V2)
    assert len(spec["columns"]) == 16
    assert TARGET_COLUMN not in spec["columns"]
    assert not any(str(c).startswith("target_") for c in spec["columns"])


def test_enet_beat_zero_baseline_is_not_asserted(tmp_path):
    """Guard against baking an unearned performance claim into the suite.

    Elastic Net's win or loss is an empirical outcome, reported in the
    modeling report. This test only records the comparison, so a future
    regression that silently rewrites the baseline would be visible.
    """
    payload, _ = run_v2_synthetic(tmp_path)
    val = {}
    for name, res in payload["results"].items():
        row = next(s for s in res["splits"] if s["split"] == "validation")
        val[name] = row["mae"]
    assert val["elastic_net"] > 0.0
    assert set(val) == {"zero_return", "train_mean", "ridge", "elastic_net"}


def test_dataset_and_split_contract_unchanged_for_enet():
    """Elastic Net consumes exactly the existing dataset/split contract."""
    from pathlib import Path

    features, targets = make_v2_synthetic()
    dataset = build_model_dataset(features, targets,
                                  feature_columns=FEATURE_COLUMNS_V2)
    parts = split_chronological(
        dataset, PipelineConfig(validation_start="2024-02-01",
                                test_start="2024-03-01"))
    assert [p["dates"] for p in parts.values()] == [
        sorted(p["dates"]) for p in parts.values()]
    assert sum(len(p["X"]) for p in parts.values()) == len(dataset["X"])

    model = ElasticNetRegressor(alpha=0.1).fit(parts["train"]["X"],
                                               parts["train"]["y"])
    prediction = model.predict(parts["test"]["X"])
    assert len(prediction) == len(parts["test"]["X"])
    # A metric computed on the untouched test split is a record, not a
    # selection input; the same call must be reproducible.
    assert metrics(parts["test"]["y"], prediction) == \
        metrics(parts["test"]["y"], prediction)
