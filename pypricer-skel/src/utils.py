from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TIME_ATOL = 1e-10


@dataclass(frozen=True, slots=True)
class MCResult:
    mean: float
    sample_stddev: float
    standard_error: float
    ci95_half_width: float

    @property
    def stddev(self) -> float:
        return self.standard_error

    @property
    def value(self) -> float:
        return self.mean


@dataclass(frozen=True, slots=True)
class DeltaResult:
    mean: np.ndarray
    sample_stddev: np.ndarray
    standard_error: np.ndarray
    ci95_half_width: np.ndarray

    @property
    def stddev(self) -> np.ndarray:
        return self.standard_error

    @property
    def delta(self) -> np.ndarray:
        return self.mean


class OnlineMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total: np.ndarray | None = None
        self.total_squares: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.shape[0] == 0:
            return
        batch_count = values.shape[0]
        batch_sum = np.sum(values, axis=0, dtype=float)
        batch_sum_squares = np.sum(values * values, axis=0, dtype=float)

        if self.total is None:
            self.total = np.array(batch_sum, dtype=float, copy=True)
            self.total_squares = np.array(batch_sum_squares, dtype=float, copy=True)
        else:
            self.total += batch_sum
            self.total_squares += batch_sum_squares
        self.count += int(batch_count)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.count == 0 or self.total is None or self.total_squares is None:
            raise ValueError("cannot finalize empty Monte Carlo statistics")
        mean = self.total / self.count
        if self.count > 1:
            variance = (self.total_squares - self.total * self.total / self.count) / (
                self.count - 1
            )
        else:
            variance = np.zeros_like(mean, dtype=float)
        sample_stddev = np.sqrt(np.maximum(variance, 0.0))
        standard_error = sample_stddev / np.sqrt(self.count)
        ci95_half_width = 1.96 * standard_error
        return mean, sample_stddev, standard_error, ci95_half_width

    def as_mc_result(self) -> MCResult:
        mean, sample_stddev, standard_error, ci95_half_width = self.finalize()
        return MCResult(
            mean=float(np.asarray(mean)),
            sample_stddev=float(np.asarray(sample_stddev)),
            standard_error=float(np.asarray(standard_error)),
            ci95_half_width=float(np.asarray(ci95_half_width)),
        )

    def as_delta_result(self) -> DeltaResult:
        mean, sample_stddev, standard_error, ci95_half_width = self.finalize()
        return DeltaResult(
            mean=np.asarray(mean, dtype=float),
            sample_stddev=np.asarray(sample_stddev, dtype=float),
            standard_error=np.asarray(standard_error, dtype=float),
            ci95_half_width=np.asarray(ci95_half_width, dtype=float),
        )


def getVector(params: dict, key: str, n: int):
    v = params[key]
    if isinstance(v, (int, float)):
        return np.full(n, float(v))
    elif isinstance(v, (list, tuple)):
        if len(v) == 1:
            return np.full(n, v[0])
        elif len(v) != n:
            raise ValueError("Size mismatch when reading vector.")
        else:
            return np.array(v, dtype=float)
    else:
        raise ValueError("Element can not be converted to array.")
