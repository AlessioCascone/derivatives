from __future__ import annotations

import numpy as np

from utils import getVector, _TIME_ATOL


class BSModel:
    model_type = "bs"

    def __init__(self, params: dict):
        self.size = params["option size"]
        self.r = params["interest rate"]
        self.T = params["maturity"]
        self.correlation = params["correlation"]
        self.nDates = params["fixing dates number"]
        self.spot = getVector(params, "spot", self.size)
        self.sigma = getVector(params, "volatility", self.size)
        self.correlationMatrix = self._build_correlation_matrix(self.correlation)
        self.cholesky = np.linalg.cholesky(self.correlationMatrix)

    @property
    def dimension(self) -> int:
        return self.size

    @property
    def interest_rate(self) -> float:
        return self.r

    @property
    def fixing_times(self) -> np.ndarray:
        return np.linspace(0.0, self.T, self.nDates + 1)

    def __str__(self):
        s = [
            f"maturity: {self.T}",
            f"fixing dates number: {self.nDates}",
            f"size: {self.size}",
            f"correlation: {self.correlation}",
            f"spot: {self.spot}",
            f"interest rate: {self.r}",
            f"volatility: {self.sigma}",
            "",
        ]
        return "\n".join(s)

    # --- New vectorized public API -------------------------------------------

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

        if np.isclose(times[0], 0.0, atol=_TIME_ATOL, rtol=0.0):
            output[:, 0, :] = self.spot
            first_future = 1

        future = self._simulate_from_initial(
            initial_spot=self.spot,
            initial_time=0.0,
            future_times=times[first_future:],
            n_paths=n_paths,
            rng=rng,
            normals=normals,
        )
        if future.size:
            output[:, first_future:, :] = future
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
    ) -> np.ndarray:
        current_time = float(current_time)
        if current_time < -_TIME_ATOL:
            raise ValueError("current_time must be non-negative")
        future_times = self._validate_times(future_times, minimum=current_time)
        if n_paths < 0:
            raise ValueError("n_paths must be non-negative")
        if future_times.size == 0:
            return np.empty((n_paths, 0, self.dimension), dtype=float)

        relative_times = future_times - current_time
        return self._simulate_from_initial(
            initial_spot=np.ones(self.dimension, dtype=float),
            initial_time=0.0,
            future_times=relative_times,
            n_paths=n_paths,
            rng=rng,
            normals=normals,
        )

    def validate(self) -> None:
        if self.dimension <= 0:
            raise ValueError("model dimension must be positive")
        if np.any(~np.isfinite(self.spot)) or np.any(self.spot <= 0.0):
            raise ValueError("spot values must be positive finite numbers")
        if np.any(~np.isfinite(self.sigma)) or np.any(self.sigma < 0.0):
            raise ValueError("volatility values must be non-negative finite numbers")
        if not np.isfinite(self.r):
            raise ValueError("interest_rate must be finite")

    # --- Backward-compatible legacy API --------------------------------------

    def asset0(self, rng: np.random.Generator):
        return self.simulate_paths(self.fixing_times, 1, rng)[0]

    def asset(self, rng: np.random.Generator, t=0, past=None):
        if t == 0 and past is None:
            return self.asset0(rng)
        return self.assets(rng, 1, t, past)[0]

    def assets(self, rng: np.random.Generator, nSamples: int, t=0, past=None):
        """Compute a batch of model paths on the option fixing grid."""
        t = float(t)
        paths = np.empty((nSamples, self.nDates + 1, self.size), dtype=float)

        if t == 0 and past is None:
            paths[:, 0, :] = self.spot
            first_future = 1
            current_spot_arr = self.spot
        else:
            if past is None:
                raise ValueError("past must be provided when t > 0")
            past_array = self._as_path_array(past)
            last_fixing = self.last_fixing_index(t)
            needed_rows = last_fixing + 1
            if not self.is_fixing_time(t):
                needed_rows += 1
            if past_array.shape[0] != needed_rows:
                raise ValueError(
                    f"past should have {needed_rows} rows at time {t}, "
                    f"got {past_array.shape[0]}"
                )
            paths[:, : last_fixing + 1, :] = past_array[: last_fixing + 1]
            current_spot_arr = np.asarray(past_array[-1], dtype=float)
            first_future = last_fixing + 1

        future_times = self.fixing_times[first_future:]
        if future_times.size:
            n_paths = nSamples
            if t == 0 and past is None:
                future_values = self._simulate_from_initial(
                    initial_spot=current_spot_arr,
                    initial_time=float(t),
                    future_times=future_times,
                    n_paths=n_paths,
                    rng=rng,
                )
            else:
                future_values = self.simulate_conditional(
                    current_time=float(t),
                    current_spot=current_spot_arr,
                    future_times=future_times,
                    n_paths=n_paths,
                    rng=rng,
                )
            paths[:, first_future:, :] = future_values

        return paths

    def shiftAsset(self, path, t, d, fd):
        shifted = np.array(path, dtype=float, copy=True)
        start = 0 if t == 0 else self.last_fixing_index(t) + 1
        if shifted.ndim == 2:
            shifted[start:, d] *= 1.0 + fd
        elif shifted.ndim == 3:
            shifted[:, start:, d] *= 1.0 + fd
        else:
            raise ValueError("path must be a 2D path or a 3D batch of paths")
        return shifted

    def current_spot(self, t=0, past=None):
        if t == 0 and past is None:
            return self.spot
        if past is None:
            raise ValueError("past must be provided when t > 0")
        return self._as_path_array(past)[-1]

    def last_fixing_index(self, t: float):
        return min(self.nDates, int(np.floor((float(t) + 1e-12) / (self.T / self.nDates))))

    def is_fixing_time(self, t: float):
        idx = self.last_fixing_index(t)
        dt = self.T / self.nDates if self.nDates > 0 else 1.0
        return abs(float(t) - idx * dt) <= 1e-10

    # --- Internal helpers ----------------------------------------------------

    def _simulate_from_initial(
        self,
        initial_spot: np.ndarray,
        initial_time: float,
        future_times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator,
        normals: np.ndarray | None = None,
    ) -> np.ndarray:
        future_times = np.asarray(future_times, dtype=float)
        if future_times.size == 0:
            return np.empty((n_paths, 0, self.dimension), dtype=float)

        dts = np.diff(np.concatenate(([float(initial_time)], future_times)))
        if np.any(dts < -_TIME_ATOL):
            raise ValueError("future_times must be increasing")
        dts = np.maximum(dts, 0.0)
        normals = self._get_normals(normals, n_paths, dts.size, rng)
        correlated = normals @ self.cholesky.T

        drift = (self.r - 0.5 * self.sigma**2)[None, None, :]
        diffusion_scale = self.sigma[None, None, :]
        increments = (
            drift * dts[None, :, None]
            + diffusion_scale * np.sqrt(dts)[None, :, None] * correlated
        )
        log_relative = np.cumsum(increments, axis=1)
        return initial_spot[None, None, :] * np.exp(log_relative)

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

    def _build_correlation_matrix(self, correlation: float | np.ndarray) -> np.ndarray:
        corr = np.asarray(correlation, dtype=float)
        if corr.ndim == 0:
            rho = float(corr)
            if self.dimension == 1:
                return np.ones((1, 1), dtype=float)
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

    def _as_path_array(self, past):
        past = np.asarray(past, dtype=float)
        if past.ndim == 1:
            if self.size != 1:
                raise ValueError("one-dimensional past is only valid for a one-asset model")
            past = past.reshape(-1, 1)
        if past.ndim != 2 or past.shape[1] != self.size:
            raise ValueError("past must have shape (nRows, modelSize)")
        return past
