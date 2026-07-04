from __future__ import annotations

from typing import Iterator

import numpy as np

from bsmodel import BSModel
from option import Option
from utils import OnlineMoments, _TIME_ATOL


class MonteCarlo:
    def __init__(self, params: dict, mod: BSModel, opt: Option):
        self.mod = mod
        self.opt = opt
        self.rng = np.random.Generator(np.random.MT19937(seed=params.get("seed", None)))
        self.nSamples = params["sample number"]
        self.fdStep = params["fd step"]
        self.chunk_size = int(params.get("chunk size", min(25_000, self.nSamples)))

    def __str__(self):
        s = [
            f"Number of samples: {self.nSamples}",
            f"fd step: {self.fdStep}",
        ]
        return "\n".join(s)

    def price0(self):
        return self.price()

    def delta0(self):
        return self.delta(t=0.0, current_spot=self.mod.spot)

    def price(
        self,
        t: float = 0.0,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
    ) -> OnlineMoments:
        stats = OnlineMoments()
        t = float(t)
        current_spot = self._current_spot(t, market_path, market_times)
        for n_chunk in self._chunk_lengths():
            paths = self._build_price_paths(
                t=t,
                current_spot=current_spot,
                n_paths=n_chunk,
                market_path=market_path,
                market_times=market_times,
            )
            discounted = np.exp(-self.mod.r * (self.opt.T - t))
            stats.update(discounted * self.opt.payoffs(paths))
        return stats.as_mc_result()

    def delta(
        self,
        t: float,
        current_spot: np.ndarray,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
    ) -> OnlineMoments:
        t = float(t)
        current_spot = np.asarray(current_spot, dtype=float)
        if current_spot.shape != (self.mod.dimension,):
            raise ValueError(
                f"current_spot must have shape ({self.mod.dimension},)"
            )
        if np.any(current_spot <= 0.0) or np.any(~np.isfinite(current_spot)):
            raise ValueError("current_spot must be positive finite numbers")
        if self.fdStep <= 0.0:
            raise ValueError("fd_step must be positive")

        stats = OnlineMoments()
        fixing_times = self.opt.fixing_times
        future_mask = fixing_times > t + _TIME_ATOL
        future_times = fixing_times[future_mask]
        d = self.mod.dimension
        diagonal = np.arange(d)
        current_plus = np.broadcast_to(current_spot, (d, d)).astype(float, copy=True)
        current_minus = current_plus.copy()
        current_plus[diagonal, diagonal] *= 1.0 + self.fdStep
        current_minus[diagonal, diagonal] *= 1.0 - self.fdStep

        for n_chunk in self._chunk_lengths():
            multipliers = self.mod.future_multipliers(
                current_time=t,
                future_times=future_times,
                n_paths=n_chunk,
                rng=self.rng,
            )
            paths_plus = self._build_shifted_paths(
                t=t,
                shifted_current=current_plus,
                multipliers=multipliers,
                market_path=market_path,
                market_times=market_times,
            )
            paths_minus = self._build_shifted_paths(
                t=t,
                shifted_current=current_minus,
                multipliers=multipliers,
                market_path=market_path,
                market_times=market_times,
            )
            payoff_diff = self.opt.payoffs(paths_plus) - self.opt.payoffs(paths_minus)
            discounted = np.exp(-self.mod.r * (self.opt.T - t))
            delta_samples = discounted * payoff_diff / (
                2.0 * self.fdStep * current_spot[:, None]
            )
            stats.update(delta_samples.T)
        return stats.as_delta_result()

    def market_path_on_fixing_grid(
        self, market_path: np.ndarray, market_times: np.ndarray
    ) -> np.ndarray:
        market_path, market_times = self._prepare_market(market_path, market_times)
        rows = [
            self._lookup_market_value(fixing_time, market_path, market_times)
            for fixing_time in self.opt.fixing_times
        ]
        return np.asarray(rows, dtype=float)

    # --- Internal helpers ----------------------------------------------------

    def _chunk_lengths(self) -> Iterator[int]:
        remaining = self.nSamples
        while remaining > 0:
            n_chunk = min(self.chunk_size, remaining)
            remaining -= n_chunk
            yield n_chunk

    def _build_price_paths(
        self,
        t: float,
        current_spot: np.ndarray,
        n_paths: int,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
    ) -> np.ndarray:
        fixing_times = self.opt.fixing_times
        future_mask = fixing_times > t + _TIME_ATOL
        future_times = fixing_times[future_mask]

        future_values = self.mod.simulate_conditional(
            current_time=t,
            current_spot=current_spot,
            future_times=future_times,
            n_paths=n_paths,
            rng=self.rng,
        )

        paths = np.empty(
            (n_paths, fixing_times.size, self.mod.dimension), dtype=float
        )
        self._fill_observed_and_current(
            paths=paths,
            t=t,
            current_values=current_spot,
            market_path=market_path,
            market_times=market_times,
        )
        if future_times.size:
            paths[:, future_mask, :] = future_values
        return paths

    def _build_shifted_paths(
        self,
        t: float,
        shifted_current: np.ndarray,
        multipliers: np.ndarray,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
    ) -> np.ndarray:
        d = self.mod.dimension
        n_paths = multipliers.shape[0]
        fixing_times = self.opt.fixing_times
        future_mask = fixing_times > t + _TIME_ATOL
        paths = np.empty((d, n_paths, fixing_times.size, d), dtype=float)
        self._fill_observed_and_current(
            paths=paths,
            t=t,
            current_values=shifted_current,
            market_path=market_path,
            market_times=market_times,
        )
        if np.any(future_mask):
            paths[:, :, future_mask, :] = (
                shifted_current[:, None, None, :] * multipliers[None, :, :, :]
            )
        return paths

    def _fill_observed_and_current(
        self,
        paths: np.ndarray,
        t: float,
        current_values: np.ndarray,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
    ) -> None:
        fixing_times = self.opt.fixing_times
        past_mask = fixing_times < t - _TIME_ATOL
        current_mask = np.isclose(fixing_times, t, atol=_TIME_ATOL, rtol=0.0)

        if np.any(past_mask):
            if market_path is None:
                raise ValueError(
                    "market_path is required for pricing after past fixings"
                )
            market_path, market_times = self._prepare_market(
                market_path, market_times, t=t
            )
            for fixing_index in np.flatnonzero(past_mask):
                value = self._lookup_market_value(
                    fixing_times[fixing_index], market_path, market_times
                )
                paths[..., fixing_index, :] = value

        if np.any(current_mask):
            current_index = int(np.flatnonzero(current_mask)[0])
            current_values = np.asarray(current_values, dtype=float)
            if current_values.ndim == 2 and paths.ndim == 4:
                paths[..., current_index, :] = current_values[:, None, :]
            else:
                paths[..., current_index, :] = current_values

    def _current_spot(
        self,
        t: float,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
    ) -> np.ndarray:
        if np.isclose(t, 0.0, atol=_TIME_ATOL, rtol=0.0):
            return np.asarray(self.mod.spot, dtype=float)
        if market_path is None:
            raise ValueError("market_path is required when t > 0")
        market_path, market_times = self._prepare_market(
            market_path, market_times, t=t
        )
        return self._lookup_market_value(t, market_path, market_times)

    def _prepare_market(
        self,
        market_path: np.ndarray,
        market_times: np.ndarray | None,
        t: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        path = np.asarray(market_path, dtype=float)
        if path.ndim == 1:
            if self.mod.dimension != 1:
                raise ValueError(
                    "one-dimensional market_path is only valid for D=1"
                )
            path = path.reshape(-1, 1)
        if path.ndim != 2 or path.shape[1] != self.mod.dimension:
            raise ValueError(
                f"market_path must have shape (n_dates, {self.mod.dimension})"
            )
        if np.any(~np.isfinite(path)) or np.any(path <= 0.0):
            raise ValueError("market_path must contain positive finite values")
        if market_times is None:
            times = self._infer_market_times(path, t)
        else:
            times = np.asarray(market_times, dtype=float)
        if times.ndim != 1 or times.shape[0] != path.shape[0]:
            raise ValueError(
                "market_times must have shape (market_path rows,)"
            )
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) < -_TIME_ATOL):
            raise ValueError("market_times must be finite and increasing")
        return path, times

    def _infer_market_times(
        self, market_path: np.ndarray, t: float | None
    ) -> np.ndarray:
        fixing_times = self.opt.fixing_times
        if market_path.shape[0] == fixing_times.size:
            return fixing_times
        if t is not None:
            past_times = fixing_times[fixing_times < t - _TIME_ATOL]
            inferred = np.concatenate((past_times, np.array([float(t)])))
            if inferred.shape[0] == market_path.shape[0]:
                return inferred
        raise ValueError("market_times is required for this market_path")

    def _lookup_market_value(
        self,
        time: float,
        market_path: np.ndarray,
        market_times: np.ndarray,
    ) -> np.ndarray:
        matches = np.flatnonzero(
            np.isclose(market_times, time, atol=_TIME_ATOL, rtol=0.0)
        )
        if matches.size == 0:
            raise ValueError(f"market_path does not contain time {time}")
        return np.asarray(market_path[int(matches[-1])], dtype=float)
