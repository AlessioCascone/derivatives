from __future__ import annotations

import numpy as np

from mcpricer.options.base import BaseOption


class PerformanceOption(BaseOption):
    """Cumulative positive basket performance payoff."""

    option_type = "performance"

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return self.payoff_from_basket_values(self.basket_values(paths))

    def payoff_from_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        return self.payoff_from_valid_basket_values(
            self.validate_basket_values(basket_values)
        )

    def payoff_from_valid_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        denominators = basket_values[..., :-1]
        if np.any(np.abs(denominators) <= 1e-14):
            raise ValueError("performance payoff encountered a zero basket denominator")
        period_returns = basket_values[..., 1:] / denominators - 1.0
        return 1.0 + np.sum(np.maximum(period_returns, 0.0), axis=-1)

    def centered_payoff_diff_from_basket_shifts(
        self,
        base_basket_values: np.ndarray,
        shifts: np.ndarray,
    ) -> np.ndarray:
        base_previous = base_basket_values[:, :-1][None, :, :]
        base_next = base_basket_values[:, 1:][None, :, :]
        shift_previous = shifts[..., :-1]
        shift_next = shifts[..., 1:]

        plus_denominators = base_previous + shift_previous
        minus_denominators = base_previous - shift_previous
        if np.any(np.abs(plus_denominators) <= 1e-14) or np.any(
            np.abs(minus_denominators) <= 1e-14
        ):
            raise ValueError("performance payoff encountered a zero basket denominator")

        plus_returns = (base_next + shift_next) / plus_denominators - 1.0
        minus_returns = (base_next - shift_next) / minus_denominators - 1.0
        return np.sum(
            np.maximum(plus_returns, 0.0) - np.maximum(minus_returns, 0.0),
            axis=-1,
        )
