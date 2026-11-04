"""Baseline predictors for next-day S&P 500 return forecasting.

All models share one interface: fit(X, y) then predict(X) -> list[float].
Baselines exist to contextualize any learned model's metrics.
"""

class ZeroBaseline:
    """Predicts 0.0 for every observation.

    The null model of return forecasting: financial returns are nearly
    unpredictable at daily horizon, so any useful model must beat this.
    """

    def fit(self, X: list[list[float]], y: list[float]) -> "ZeroBaseline":
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        return [0.0] * len(X)


class TrainMeanBaseline:
    """Predicts the mean training target for every observation.

    Statistics are frozen at fit time from TRAIN only; validation and test
    are scored against the frozen value.
    """

    def __init__(self) -> None:
        self.mean: float | None = None

    def fit(self, X: list[list[float]], y: list[float]) -> "TrainMeanBaseline":
        if not y:
            raise ValueError("cannot fit on empty targets")
        self.mean = sum(y) / len(y)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        if self.mean is None:
            raise RuntimeError("model is not fitted")
        return [self.mean] * len(X)


class PersistenceBaseline:
    """Predicts tomorrow's return equals today's observed return.

    Reads the 'return' column (today's close-to-close return) from the
    feature matrix. Useful benchmark: it captures raw return autocorrelation
    with zero learning; a learned model must outperform it to justify
    existing.
    """

    def __init__(self, return_column_index: int = 0) -> None:
        self.index = return_column_index

    def fit(self, X: list[list[float]], y: list[float]) -> "PersistenceBaseline":
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        return [row[self.index] for row in X]
