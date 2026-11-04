"""Linear regressors (no external ML framework).

RidgeRegressor solves min ||y - Xw - b||^2 + alpha * ||w||^2 via normal
equations on internally centered data. ElasticNetRegressor solves the
combined L1/L2 objective via NumPy coordinate descent. Both are fully
deterministic and share one fit/predict interface; see each class docstring
for the objective and the degeneracy guards.
"""

class RidgeRegressor:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.weights: list[float] | None = None
        self.intercept: float | None = None

    def fit(self, X: list[list[float]], y: list[float]) -> "RidgeRegressor":
        import numpy as np

        A = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if A.ndim != 2 or A.shape[0] != target.shape[0]:
            raise ValueError("X must be 2-D with one row per target")
        if A.shape[0] <= A.shape[1]:
            # Keep the linear system well-posed even for tiny synthetic sets.
            self.alpha = max(self.alpha, 1e-6)

        x_mean = A.mean(axis=0)
        y_mean = target.mean()
        Ac = A - x_mean
        yc = target - y_mean

        n_features = A.shape[1]
        gram = Ac.T @ Ac + self.alpha * np.eye(n_features)
        self.weights = np.linalg.solve(gram, Ac.T @ yc).tolist()
        self.intercept = float(y_mean - x_mean @ np.asarray(self.weights))
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        import numpy as np

        if self.weights is None or self.intercept is None:
            raise RuntimeError("model is not fitted")
        return (np.asarray(X, dtype=float) @ np.asarray(self.weights)
                + self.intercept).tolist()

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "weights": list(self.weights),
                "intercept": self.intercept}


# Floor for the coordinate-descent denominator. The denominator is
# ||x_j||^2/n + alpha*(1-l1_ratio); it is positive for any non-degenerate
# column, but an all-constant column with no L2 penalty would give exactly
# 0.0. Flooring keeps the update finite and deterministic instead of
# dividing by zero, and never silently rewrites the caller's alpha.
_MIN_DENOM = 1e-12


def _soft_threshold(value: float, threshold: float) -> float:
    """Sign-preserving soft-threshold (shrink toward zero).

    Branch form rather than sign(v)*max(|v|-t, 0): avoids the
    sign(0)*something and 0*inf product forms that produce NaN, and is
    exact when threshold == 0.0 (plain ridge limit).
    """
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


class ElasticNetRegressor:
    """Elastic Net linear regression via NumPy cyclic coordinate descent.

    Minimizes, with n = number of rows and b unregularized:

        (1 / (2n)) * ||y - Xw - b||^2
        + alpha * l1_ratio * ||w||_1
        + alpha * (1 - l1_ratio) / 2 * ||w||^2

    The objective is identical to sklearn's ElasticNet, but the solver is
    self-contained NumPy -- no sklearn, no scipy, matching RidgeRegressor's
    dependency footprint.

    Conventions match RidgeRegressor exactly:
    - the design matrix is centered internally and the intercept is
      recovered as mean(y) - mean(X) @ w, so the intercept is never
      regularized;
    - the caller supplies already-scaled data (TrainOnlyStandardizer owns
      all preprocessing); this model never rescales anything;
    - observations are visited in a fixed order and never shuffled, so
      repeated fits are bit-identical.

    Degenerate limits: l1_ratio == 0 recovers the ridge solution, and
    alpha == 0 recovers unregularized least squares.

    Convergence: a sweep is the last one when
    max_j |w_j_new - w_j_old| <= tol * max_j |w_j|. This is relative to the
    fitted weights, so tol keeps one fixed meaning for any target
    magnitude (see the comment in fit()). Reaching max_iter is reported
    honestly through converged_ = False rather than being masked as success.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-4) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be within [0.0, 1.0]")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.weights: list[float] | None = None
        self.intercept: float | None = None
        self.n_iter_: int | None = None
        self.converged_: bool | None = None

    def fit(self, X: list[list[float]],
            y: list[float]) -> "ElasticNetRegressor":
        import numpy as np

        A = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if A.ndim != 2 or A.shape[0] != target.shape[0]:
            raise ValueError("X must be 2-D with one row per target")
        if A.shape[0] == 0:
            raise ValueError("cannot fit on an empty design matrix")

        n, n_features = A.shape
        x_mean = A.mean(axis=0)
        y_mean = target.mean()
        Ac = A - x_mean
        yc = target - y_mean

        threshold = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        w = np.zeros(n_features, dtype=float)
        # residual with the current w; updated in place, never recomputed
        # from scratch, so each sweep costs O(n) per coordinate.
        residual = yc - Ac @ w

        n_iter = 0
        converged = False
        for n_iter in range(1, self.max_iter + 1):
            max_update = 0.0
            for j in range(n_features):
                column = Ac[:, j]
                # residual with w_j temporarily set to zero
                zeroed = residual + w[j] * column
                partial = float(column @ zeroed) / n
                denom = max(float(column @ column) / n + l2_penalty,
                            _MIN_DENOM)
                updated = _soft_threshold(partial, threshold) / denom
                max_update = max(max_update, abs(updated - w[j]))
                w[j] = updated
                residual = zeroed - updated * column
            # Convergence is judged on the RELATIVE size of the coefficient
            # update, never on the magnitude of the target.
            #
            # Under y -> c*y (c > 0) every weight and every update scales by
            # |c|, so max_update and max|w| both carry a factor |c| and the
            # comparison below is exactly invariant. The superseded rule,
            # max_update < tol * ||y - mean(y)||^2, scaled its right-hand
            # side by c**2 while the left-hand side scaled by c, so the
            # effective strictness moved with the arbitrary numerical scale
            # of the target -- for y ~ O(10) it admitted convergence after
            # two sweeps with ~0.37 coefficient error. The solver itself was
            # never the problem; only the stopping rule was.
            #
            # Deliberately NOT max(1, max|w|): at the V2 return scale the
            # largest weight is ~5e-4, so the constant 1 would dominate and
            # hand back a threshold of tol*1 = 1e-4 -- twice as loose as the
            # old rule. A fixed constant re-introduces an arbitrary scale.
            #
            # Comparison is <= so the all-zero fixed point converges: a
            # fully shrunk model has max|w| == 0 and an exactly zero update,
            # and 0 <= 0 must be True. Using < would spin to max_iter.
            if max_update <= self.tol * float(np.abs(w).max()):
                converged = True
                break

        self.weights = w.tolist()
        self.intercept = float(y_mean - x_mean @ w)
        self.n_iter_ = n_iter
        self.converged_ = converged
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        import numpy as np

        if self.weights is None or self.intercept is None:
            raise RuntimeError("model is not fitted")
        return (np.asarray(X, dtype=float) @ np.asarray(self.weights)
                + self.intercept).tolist()

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "weights": list(self.weights),
            "intercept": self.intercept,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
        }

