"""Train-only preprocessing for the ForeSight modeling layer.

The standardizer is fitted exclusively on the training split; validation and
test are transformed with those frozen statistics. Fitted statistics are plain
JSON-serializable values so they can be persisted alongside a trained model.
"""

import math


class TrainOnlyStandardizer:
    def __init__(self, means: list[float] | None = None,
                 stds: list[float] | None = None) -> None:
        self.means = means
        self.stds = stds

    @property
    def is_fitted(self) -> bool:
        return self.means is not None

    def fit(self, X: list[list[float]]) -> "TrainOnlyStandardizer":
        if not X:
            raise ValueError("cannot fit standardizer on an empty matrix")
        n_cols = len(X[0])
        self.means, self.stds = [], []
        for j in range(n_cols):
            column = [row[j] for row in X]
            mean = sum(column) / len(column)
            variance = sum((v - mean) ** 2 for v in column) / len(column)
            std = math.sqrt(variance)
            self.stds.append(std if std > 0 else 1.0)
            self.means.append(mean)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        if not self.is_fitted:
            raise RuntimeError("standardizer is not fitted")
        return [
            [(v - m) / s for v, m, s in zip(row, self.means, self.stds)]
            for row in X
        ]

    def to_dict(self) -> dict:
        return {"means": list(self.means), "stds": list(self.stds)}

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainOnlyStandardizer":
        return cls(means=list(payload["means"]), stds=list(payload["stds"]))
