from __future__ import annotations

import numpy as np

from utils import getVector


class Option:
    option_type: str = ""

    @staticmethod
    def new(params: dict):
        option_type = params["option type"]
        for cls in Option.__subclasses__():
            if cls.option_type == option_type:
                return cls(params)
        raise ValueError(f"Unknown option: {option_type}")

    def __init__(self, params: dict):
        self.size = params["option size"]
        self.nDates = params["fixing dates number"]
        self.T = params["maturity"]
        self.strike = params.get("strike", 0.0)
        self.weights = getVector(params, "payoff coefficients", self.size)

    @property
    def dimension(self) -> int:
        return self.size

    @property
    def maturity(self) -> float:
        return self.T

    @property
    def fixing_dates_number(self) -> int:
        return self.nDates

    @property
    def coefficients(self) -> np.ndarray:
        return self.weights

    @property
    def fixing_times(self) -> np.ndarray:
        return np.linspace(0.0, self.maturity, self.fixing_dates_number + 1)

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

    def payoff(self, path):
        return float(self.payoffs(np.asarray(path, dtype=float)[None, :, :])[0])

    def payoffs(self, paths):
        raise NotImplementedError


class BasketOption(Option):
    option_type = "basket"

    def payoffs(self, paths):
        paths = self.validate_paths(paths)
        terminal_basket = np.tensordot(
            paths[..., -1, :], self.coefficients, axes=([-1], [0])
        )
        return np.maximum(terminal_basket - self.strike, 0.0)


class AsianOption(Option):
    option_type = "asian"

    def payoffs(self, paths):
        basket_values = self.basket_values(paths)
        average_basket = np.mean(basket_values, axis=-1)
        return np.maximum(average_basket - self.strike, 0.0)


class PerformanceOption(Option):
    option_type = "performance"

    def payoffs(self, paths):
        basket_values = self.basket_values(paths)
        denominators = basket_values[..., :-1]
        if np.any(np.isclose(denominators, 0.0, atol=1e-14, rtol=1e-14)):
            raise ValueError(
                "performance payoff encountered a zero basket denominator"
            )
        period_returns = basket_values[..., 1:] / denominators - 1.0
        return np.maximum(1.0 + np.sum(period_returns, axis=-1), 0.0)
