from __future__ import annotations

import numpy as np

from mcpricer.options.base import BaseOption


class BasketOption(BaseOption):
    """European basket call payoff on terminal fixing values."""

    option_type = "basket"

    def terminal_basket(self, paths: np.ndarray) -> np.ndarray:
        paths = self.validate_paths(paths)
        return np.tensordot(paths[..., -1, :], self.coefficients, axes=([-1], [0]))

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        terminal_basket = self.terminal_basket(paths)
        return self.payoff_from_terminal_basket_values(terminal_basket)

    def payoff_from_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        return self.payoff_from_valid_basket_values(
            self.validate_basket_values(basket_values)
        )

    def payoff_from_valid_basket_values(self, basket_values: np.ndarray) -> np.ndarray:
        terminal_basket = basket_values[..., -1]
        return self.payoff_from_terminal_basket_values(terminal_basket)

    def payoff_from_terminal_basket_values(
        self, terminal_basket: np.ndarray
    ) -> np.ndarray:
        return np.maximum(terminal_basket - self.strike, 0.0)

    def centered_payoff_diff_from_basket_shifts(
        self,
        base_basket_values: np.ndarray,
        shifts: np.ndarray,
    ) -> np.ndarray:
        terminal_basket = base_basket_values[..., -1]
        terminal_shift = shifts[..., -1]
        return self.payoff_from_terminal_basket_values(
            terminal_basket[None, :] + terminal_shift
        ) - self.payoff_from_terminal_basket_values(
            terminal_basket[None, :] - terminal_shift
        )


class CallOption(BasketOption):
    """Standard one-asset European call payoff."""

    option_type = "call"

    def __post_init__(self) -> None:
        if int(self.dimension) != 1:
            raise ValueError("call option requires exactly one asset")
        self.coefficients = np.ones(1, dtype=float)
        super().__post_init__()


class PutOption(BasketOption):
    """Standard one-asset European put payoff."""

    option_type = "put"

    def __post_init__(self) -> None:
        if int(self.dimension) != 1:
            raise ValueError("put option requires exactly one asset")
        self.coefficients = np.ones(1, dtype=float)
        super().__post_init__()

    def payoff_from_terminal_basket_values(
        self, terminal_basket: np.ndarray
    ) -> np.ndarray:
        return np.maximum(self.strike - terminal_basket, 0.0)
