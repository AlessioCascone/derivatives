from __future__ import annotations

import numpy as np

from mcpricer.options.base import BaseOption


class AsianOption(BaseOption):
    """Discrete arithmetic Asian basket option."""

    option_type = "asian"

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return self.payoff_from_basket_values(self.basket_values(paths))

    def payoff_from_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        return self.payoff_from_valid_basket_values(
            self.validate_basket_values(basket_values)
        )

    def payoff_from_valid_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        average_basket = np.mean(basket_values, axis=-1)
        return np.maximum(average_basket - self.strike, 0.0)

    def centered_payoff_diff_from_basket_shifts(
        self,
        base_basket_values: np.ndarray,
        shifts: np.ndarray,
    ) -> np.ndarray:
        return self.centered_payoff_diff_from_shift_averages(
            base_basket_values,
            np.mean(shifts, axis=-1),
        )

    def centered_payoff_diff_from_shift_averages(
        self,
        base_basket_values: np.ndarray,
        shift_averages: np.ndarray,
    ) -> np.ndarray:
        base_average = np.mean(base_basket_values, axis=-1)
        return np.maximum(
            base_average[None, :] + shift_averages - self.strike,
            0.0,
        ) - np.maximum(
            base_average[None, :] - shift_averages - self.strike,
            0.0,
        )
