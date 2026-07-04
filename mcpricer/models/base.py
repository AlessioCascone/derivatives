from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract stochastic model interface.

    Implementations work with path arrays of shape ``(n_paths, n_times, D)``,
    where ``D`` is the model dimension.
    """

    dimension: int
    interest_rate: float
    spot: np.ndarray

    @abstractmethod
    def simulate_paths(
        self,
        times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate asset values at ``times``.

        Parameters
        ----------
        times:
            Increasing non-negative times. If ``0`` is present, the returned
            path includes the initial spot at that position.
        n_paths:
            Number of Monte Carlo paths.
        rng:
            NumPy random generator.
        normals:
            Optional standard normal increments with shape
            ``(n_paths, n_steps, D)``.
        """

    @abstractmethod
    def simulate_conditional(
        self,
        current_time: float,
        current_spot: np.ndarray,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate future values conditionally on ``S_current_time``.

        Returns an array of shape ``(n_paths, len(future_times), D)``.
        """

    @abstractmethod
    def future_multipliers(
        self,
        current_time: float,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
        correlated_normals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate componentwise multipliers ``S_future / S_current``.

        Returns an array of shape ``(n_paths, len(future_times), D)``.
        """

    @abstractmethod
    def validate(self) -> None:
        """Raise ``ValueError`` if the model parameters are invalid."""
