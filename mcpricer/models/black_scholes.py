from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mcpricer.models.base import BaseModel


_TIME_ATOL = 1e-12


@dataclass(slots=True)
class BlackScholesModel(BaseModel):
    """Multidimensional Black-Scholes model with constant coefficients."""

    spot: np.ndarray
    volatility: np.ndarray
    interest_rate: float
    correlation: float | np.ndarray
    correlation_matrix: np.ndarray = field(init=False)
    cholesky: np.ndarray = field(init=False)
    dimension: int = field(init=False)
    log_drift: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.spot = np.asarray(self.spot, dtype=float)
        self.volatility = np.asarray(self.volatility, dtype=float)
        if self.spot.ndim != 1:
            raise ValueError("spot must have shape (D,)")
        if self.volatility.ndim != 1:
            raise ValueError("volatility must have shape (D,)")
        if self.spot.shape != self.volatility.shape:
            raise ValueError("spot and volatility must have the same shape")
        self.dimension = int(self.spot.size)
        self.interest_rate = float(self.interest_rate)
        self.validate()
        self.correlation_matrix = self._build_correlation_matrix(self.correlation)
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        self.log_drift = self.interest_rate - 0.5 * self.volatility**2

    def validate(self) -> None:
        if self.dimension <= 0:
            raise ValueError("model dimension must be positive")
        if np.any(~np.isfinite(self.spot)) or np.any(self.spot <= 0.0):
            raise ValueError("spot values must be positive finite numbers")
        if np.any(~np.isfinite(self.volatility)) or np.any(self.volatility < 0.0):
            raise ValueError("volatility values must be non-negative finite numbers")
        if not np.isfinite(self.interest_rate):
            raise ValueError("interest_rate must be finite")

    def simulate_paths(
        self,
        times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        times = self._validate_times(times, minimum=0.0)
        if n_paths < 0:
            raise ValueError("n_paths must be non-negative")
        if times.size == 0:
            return np.empty((n_paths, 0, self.dimension), dtype=float)

        output = np.empty((n_paths, times.size, self.dimension), dtype=float)
        first_future = 0
        initial_time = 0.0
        initial_spot = self.spot

        if np.isclose(times[0], 0.0, atol=_TIME_ATOL, rtol=0.0):
            output[:, 0, :] = initial_spot
            first_future = 1

        future = self._simulate_from_initial(
            initial_spot=initial_spot,
            initial_time=initial_time,
            future_times=times[first_future:],
            n_paths=n_paths,
            rng=rng,
            normals=normals,
        )
        if future.size:
            output[:, first_future:, :] = future
        elif normals is not None and normals.shape[1] != 0:
            raise ValueError("normals has too many time steps")
        return output

    def simulate_conditional(
        self,
        current_time: float,
        current_spot: np.ndarray,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        current_spot = self._validate_spot(current_spot, "current_spot")
        multipliers = self.future_multipliers(
            current_time=current_time,
            future_times=future_times,
            n_paths=n_paths,
            rng=rng,
            normals=normals,
        )
        return current_spot[None, None, :] * multipliers

    def future_multipliers(
        self,
        current_time: float,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
        correlated_normals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate componentwise ratios from the current state to future times."""

        current_time = float(current_time)
        if current_time < -_TIME_ATOL:
            raise ValueError("current_time must be non-negative")
        future_times = self._validate_times(future_times, minimum=current_time)
        if n_paths < 0:
            raise ValueError("n_paths must be non-negative")
        if future_times.size == 0:
            if normals is not None and normals.shape[1] != 0:
                raise ValueError("normals has too many time steps")
            if correlated_normals is not None and correlated_normals.shape[1] != 0:
                raise ValueError("correlated_normals has too many time steps")
            return np.empty((n_paths, 0, self.dimension), dtype=float)

        # Multipliers let pricing and bumped-delta paths share the same Brownian
        # future while changing only the current spot.
        relative_times = future_times - current_time
        return self._simulate_from_initial(
            initial_spot=np.ones(self.dimension, dtype=float),
            initial_time=0.0,
            future_times=relative_times,
            n_paths=n_paths,
            rng=rng,
            normals=normals,
            correlated_normals=correlated_normals,
        )

    def future_multipliers_batch(
        self,
        current_times: np.ndarray,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
        correlated_normals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate shared-normal future multipliers for several current times.

        Returns an array with shape ``(n_current_times, n_paths, n_future_times, D)``.
        The same normal increment table is reused for each current time, matching
        the common-random-number hedge-date reuse in the sequential portfolio path.
        """

        current_times = np.asarray(current_times, dtype=float)
        if current_times.ndim != 1:
            raise ValueError("current_times must have shape (n_current_times,)")
        if np.any(~np.isfinite(current_times)) or np.any(current_times < -_TIME_ATOL):
            raise ValueError("current_times must be finite and non-negative")
        future_times = self._validate_times(
            future_times,
            minimum=float(np.max(current_times)) if current_times.size else 0.0,
        )
        if n_paths < 0:
            raise ValueError("n_paths must be non-negative")
        n_current = current_times.size
        if future_times.size == 0:
            if normals is not None and normals.shape[1] != 0:
                raise ValueError("normals has too many time steps")
            if correlated_normals is not None and correlated_normals.shape[1] != 0:
                raise ValueError("correlated_normals has too many time steps")
            return np.empty((n_current, n_paths, 0, self.dimension), dtype=float)

        dts = np.empty((n_current, future_times.size), dtype=float)
        dts[:, 0] = future_times[0] - current_times
        if future_times.size > 1:
            dts[:, 1:] = np.diff(future_times)
        if np.any(dts < -_TIME_ATOL):
            raise ValueError("future_times must be later than every current time")
        dts = np.maximum(dts, 0.0)

        correlated = self._get_correlated_normals(
            normals=normals,
            correlated_normals=correlated_normals,
            n_paths=n_paths,
            n_steps=future_times.size,
            rng=rng,
        )
        log_relative = np.broadcast_to(
            correlated[None, :, :, :],
            (n_current, n_paths, future_times.size, self.dimension),
        ).copy()
        log_relative *= self.volatility[None, None, None, :] * np.sqrt(dts)[
            :, None, :, None
        ]
        log_relative += self.log_drift[None, None, None, :] * dts[:, None, :, None]
        np.cumsum(log_relative, axis=2, out=log_relative)
        np.exp(log_relative, out=log_relative)
        return log_relative

    def _simulate_from_initial(
        self,
        initial_spot: np.ndarray,
        initial_time: float,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None,
        correlated_normals: np.ndarray | None = None,
    ) -> np.ndarray:
        future_times = np.asarray(future_times, dtype=float)
        if future_times.size == 0:
            return np.empty((n_paths, 0, self.dimension), dtype=float)

        dts = np.diff(np.concatenate(([float(initial_time)], future_times)))
        if np.any(dts < -_TIME_ATOL):
            raise ValueError("future_times must be increasing")
        dts = np.maximum(dts, 0.0)
        correlated = self._get_correlated_normals(
            normals=normals,
            correlated_normals=correlated_normals,
            n_paths=n_paths,
            n_steps=dts.size,
            rng=rng,
        )

        # Simulate in log space to match the exact Black-Scholes transition.
        log_relative = np.array(correlated, dtype=float, copy=True)
        log_relative *= self.volatility[None, None, :] * np.sqrt(dts)[None, :, None]
        log_relative += self.log_drift[None, None, :] * dts[None, :, None]
        np.cumsum(log_relative, axis=1, out=log_relative)
        np.exp(log_relative, out=log_relative)
        if np.all(initial_spot == 1.0):
            return log_relative
        log_relative *= initial_spot[None, None, :]
        return log_relative

    def _get_normals(
        self,
        normals: np.ndarray | None,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        expected_shape = (n_paths, n_steps, self.dimension)
        if normals is None:
            return rng.standard_normal(expected_shape)
        normals = np.asarray(normals, dtype=float)
        if normals.shape != expected_shape:
            raise ValueError(
                f"normals must have shape {expected_shape}, got {normals.shape}"
            )
        return normals

    def correlated_normals(
        self,
        normals: np.ndarray,
    ) -> np.ndarray:
        normals = np.asarray(normals, dtype=float)
        if normals.ndim != 3 or normals.shape[2] != self.dimension:
            raise ValueError(
                f"normals must have trailing dimension {self.dimension}, "
                f"got {normals.shape}"
            )
        return normals @ self.cholesky.T

    def _get_correlated_normals(
        self,
        normals: np.ndarray | None,
        correlated_normals: np.ndarray | None,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        expected_shape = (n_paths, n_steps, self.dimension)
        if correlated_normals is not None:
            if normals is not None:
                raise ValueError("normals and correlated_normals cannot both be provided")
            correlated = np.asarray(correlated_normals, dtype=float)
            if correlated.shape != expected_shape:
                raise ValueError(
                    f"correlated_normals must have shape {expected_shape}, "
                    f"got {correlated.shape}"
                )
            return correlated

        normals = self._get_normals(normals, n_paths, n_steps, rng)
        return self.correlated_normals(normals)

    def _build_correlation_matrix(self, correlation: float | np.ndarray) -> np.ndarray:
        corr = np.asarray(correlation, dtype=float)
        if corr.ndim == 0:
            rho = float(corr)
            if self.dimension == 1:
                return np.ones((1, 1), dtype=float)
            # A scalar input is interpreted as equicorrelation; the lower bound
            # is the positive-definiteness condition for that matrix.
            lower = -1.0 / (self.dimension - 1)
            if not lower < rho < 1.0:
                raise ValueError(
                    f"scalar correlation must satisfy {lower} < rho < 1"
                )
            matrix = np.full((self.dimension, self.dimension), rho, dtype=float)
            np.fill_diagonal(matrix, 1.0)
            return matrix

        if corr.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"correlation matrix must have shape "
                f"({self.dimension}, {self.dimension})"
            )
        if not np.allclose(corr, corr.T, atol=1e-12, rtol=1e-12):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("correlation matrix diagonal must contain ones")
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation matrix must be positive definite") from exc
        return corr

    def _validate_times(self, times: np.ndarray, minimum: float) -> np.ndarray:
        arr = np.asarray(times, dtype=float)
        if arr.ndim != 1:
            raise ValueError("times must have shape (n_times,)")
        if np.any(~np.isfinite(arr)):
            raise ValueError("times must be finite")
        if arr.size and arr[0] < minimum - _TIME_ATOL:
            raise ValueError("times are earlier than the allowed minimum")
        if np.any(np.diff(arr) < -_TIME_ATOL):
            raise ValueError("times must be increasing")
        return arr

    def _validate_spot(self, spot: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(spot, dtype=float)
        if arr.shape != (self.dimension,):
            raise ValueError(f"{name} must have shape ({self.dimension},)")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must contain positive finite values")
        return arr
