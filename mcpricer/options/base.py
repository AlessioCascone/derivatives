from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class BaseOption(ABC):
    """Abstract vectorized payoff.

    Payoffs accept arrays with shape ``(..., N + 1, D)`` and return arrays
    with shape ``(...)``.
    """

    maturity: float
    fixing_dates_number: int
    dimension: int
    strike: float
    coefficients: np.ndarray
    _fixing_times: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.maturity = float(self.maturity)
        self.fixing_dates_number = int(self.fixing_dates_number)
        self.dimension = int(self.dimension)
        self.strike = float(self.strike)
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative")
        if self.fixing_dates_number < 1:
            raise ValueError("fixing_dates_number must be positive")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.coefficients.shape != (self.dimension,):
            raise ValueError(f"coefficients must have shape ({self.dimension},)")
        if np.any(~np.isfinite(self.coefficients)):
            raise ValueError("coefficients must be finite")
        self._fixing_times = np.linspace(
            0.0, self.maturity, self.fixing_dates_number + 1
        )

    @property
    def fixing_times(self) -> np.ndarray:
        return self._fixing_times

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Return payoff for paths with shape ``(..., N + 1, D)``."""

    def validate_paths(self, paths: np.ndarray) -> np.ndarray:
        paths = np.asarray(paths, dtype=float)
        expected_tail = (self.fixing_dates_number + 1, self.dimension)
        if paths.ndim < 2 or paths.shape[-2:] != expected_tail:
            raise ValueError(f"paths must have trailing shape {expected_tail}")
        if np.any(~np.isfinite(paths)):
            raise ValueError("paths must contain finite values")
        return paths

    def basket_values(self, paths: np.ndarray) -> np.ndarray:
        paths = self.validate_paths(paths)
        return np.tensordot(paths, self.coefficients, axes=([-1], [0]))

    def validate_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        values = np.asarray(basket_values, dtype=float)
        expected_last = self.fixing_dates_number + 1
        if values.ndim < 1 or values.shape[-1] != expected_last:
            raise ValueError(
                f"basket_values must have last dimension {expected_last}"
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("basket_values must contain finite values")
        return values

    @abstractmethod
    def payoff_from_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        """Return payoff from precomputed basket values with shape ``(..., N + 1)``."""

    def payoff_from_valid_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        """Return payoff for already validated/generated basket values."""

        return self.payoff_from_basket_values(basket_values)

    def centered_payoff_diff_from_basket_shifts(
        self,
        base_basket_values: np.ndarray,
        shifts: np.ndarray,
    ) -> np.ndarray:
        """Return payoff(base + shift) - payoff(base - shift) for delta samples."""

        plus = base_basket_values[None, ...] + shifts
        minus = base_basket_values[None, ...] - shifts
        return self.payoff_from_valid_basket_values(
            plus
        ) - self.payoff_from_valid_basket_values(minus)
