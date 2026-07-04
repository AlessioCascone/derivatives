from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from mcpricer.engine.monte_carlo import MonteCarloPricer
from mcpricer.engine.stats import DeltaResult, MCResult
from mcpricer.models.base import BaseModel
from mcpricer.options.base import BaseOption


_TIME_ATOL = 1e-10


@dataclass(slots=True)
class _SharedNormalCache:
    future_count: int | None = None
    correlated_chunks: tuple[np.ndarray, ...] | None = None


@dataclass(slots=True)
class PortfolioRecord:
    date: int
    value: float
    price: float
    price_stddev: float
    deltas: np.ndarray
    deltas_stddev: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "date": int(self.date),
            "value": float(self.value),
            "price": float(self.price),
            "priceStdDev": float(self.price_stddev),
            "deltas": np.asarray(self.deltas, dtype=float).tolist(),
            "deltasStdDev": np.asarray(self.deltas_stddev, dtype=float).tolist(),
        }


@dataclass(slots=True)
class Portfolio:
    """Discrete self-financing hedging portfolio on market dates."""

    model: BaseModel
    option: BaseOption
    pricer: MonteCarloPricer
    market_times: np.ndarray
    market_path: np.ndarray
    records: list[PortfolioRecord] = field(default_factory=list)
    final_pnl: float | None = None
    final_payoff: float | None = None

    def __post_init__(self) -> None:
        self.market_times = np.asarray(self.market_times, dtype=float)
        self.market_path = np.asarray(self.market_path, dtype=float)
        if self.market_path.ndim == 1:
            if self.model.dimension != 1:
                raise ValueError("one-dimensional market_path is only valid for D=1")
            self.market_path = self.market_path.reshape(-1, 1)
        if (
            self.market_path.ndim != 2
            or self.market_path.shape[1] != self.model.dimension
        ):
            raise ValueError(
                f"market_path must have shape (H + 1, {self.model.dimension})"
            )
        if (
            self.market_times.ndim != 1
            or self.market_times.shape[0] != self.market_path.shape[0]
        ):
            raise ValueError("market_times must have shape (H + 1,)")
        if not np.isclose(self.market_times[0], 0.0):
            raise ValueError("market_times must start at 0")
        if not np.isclose(self.market_times[-1], self.option.maturity):
            raise ValueError("market_times must end at option maturity")
        if np.any(np.diff(self.market_times) <= 0.0):
            raise ValueError("market_times must be strictly increasing")
        if np.any(~np.isfinite(self.market_path)) or np.any(self.market_path <= 0.0):
            raise ValueError("market_path must contain positive finite values")

    def run(self) -> "Portfolio":
        hedge_count = self.market_times.size - 1
        # Asian and performance payoffs are grouped by future fixing grid to
        # avoid rerunning identical multiplier simulations for nearby hedge dates.
        price_deltas = (
            self._batched_price_deltas(hedge_count)
            if self._uses_batched_fixing_intervals()
            else None
        )
        return self._run_hedging_path(price_deltas)

    def _run_hedging_path(
        self,
        price_deltas: list[tuple[MCResult, DeltaResult]] | None,
    ) -> "Portfolio":
        self.records.clear()
        self.final_pnl = None
        self.final_payoff = None

        hedge_count = self.market_times.size - 1
        discounted_wealth: float | None = None
        previous_delta: np.ndarray | None = None
        normal_cache = _SharedNormalCache()

        for i, t in enumerate(self.market_times):
            current_spot = self.market_path[i]
            if i == 0:
                price_result, delta_result = self._price_delta_for_hedge_date(
                    date_index=i,
                    t=0.0,
                    current_spot=current_spot,
                    precomputed=price_deltas,
                    normal_cache=normal_cache,
                )
                discounted_wealth = price_result.mean
                wealth = price_result.mean
            else:
                if discounted_wealth is None or previous_delta is None:
                    raise RuntimeError("portfolio state was not initialized")
                # Updating discounted wealth implements the discrete
                # self-financing condition without separately tracking cash.
                discounted_previous = self._discounted_spot(
                    i - 1,
                    self.market_times[i - 1],
                )
                discounted_current = self._discounted_spot(i, t)
                discounted_wealth += float(
                    previous_delta @ (discounted_current - discounted_previous)
                )
                wealth = discounted_wealth * np.exp(self.model.interest_rate * t)
                if i < hedge_count:
                    price_result, delta_result = self._price_delta_for_hedge_date(
                        date_index=i,
                        t=float(t),
                        current_spot=current_spot,
                        precomputed=price_deltas,
                        normal_cache=normal_cache,
                    )
                else:
                    price_result = self._terminal_price(date_index=i, t=float(t))

            if i < hedge_count:
                deltas = delta_result.mean
                deltas_stddev = delta_result.standard_error
                previous_delta = deltas
            else:
                deltas = np.zeros(self.model.dimension, dtype=float)
                deltas_stddev = np.zeros(self.model.dimension, dtype=float)

            self.records.append(
                PortfolioRecord(
                    date=i,
                    value=float(wealth),
                    price=price_result.mean,
                    price_stddev=price_result.standard_error,
                    deltas=deltas,
                    deltas_stddev=deltas_stddev,
                )
            )

        self._set_terminal_metadata()
        return self

    def _price_delta_for_hedge_date(
        self,
        date_index: int,
        t: float,
        current_spot: np.ndarray,
        precomputed: list[tuple[MCResult, DeltaResult]] | None,
        normal_cache: _SharedNormalCache,
    ) -> tuple[MCResult, DeltaResult]:
        if precomputed is not None:
            return precomputed[date_index]

        market_path = None if date_index == 0 else self.market_path[: date_index + 1]
        market_times = None if date_index == 0 else self.market_times[: date_index + 1]
        return self._price_and_delta(
            t=t,
            current_spot=current_spot,
            market_path=market_path,
            market_times=market_times,
            correlated_normal_chunks=self._correlated_normal_chunks_for_time(
                t,
                normal_cache,
            ),
        )

    def _correlated_normal_chunks_for_time(
        self,
        t: float,
        cache: _SharedNormalCache,
    ) -> tuple[np.ndarray, ...] | None:
        if not isinstance(self.pricer, MonteCarloPricer):
            return None

        future_count = int(
            np.count_nonzero(self.option.fixing_times > float(t) + _TIME_ATOL)
        )
        # Reuse common random numbers while the number of future fixing dates is
        # unchanged; this makes consecutive hedge-date estimates easier to compare.
        if future_count != cache.future_count:
            cache.future_count = future_count
            cache.correlated_chunks = self.pricer.correlated_normal_chunks(future_count)
        return cache.correlated_chunks

    def _discounted_spot(self, date_index: int, t: float) -> np.ndarray:
        return self.market_path[date_index] * np.exp(-self.model.interest_rate * t)

    def _terminal_price(self, date_index: int, t: float) -> MCResult:
        return self.pricer.price(
            t=t,
            market_path=self.market_path[: date_index + 1],
            market_times=self.market_times[: date_index + 1],
        )

    def _set_terminal_metadata(self) -> None:
        fixing_path = self.pricer.market_path_on_fixing_grid(
            self.market_path,
            self.market_times,
        )
        self.final_payoff = float(self.option.payoff(fixing_path))
        self.final_pnl = float(self.records[-1].value - self.final_payoff)

    def _uses_batched_fixing_intervals(self) -> bool:
        return (
            isinstance(self.pricer, MonteCarloPricer)
            and getattr(self.option, "option_type", None) in {"asian", "performance"}
            and hasattr(self.pricer, "price_and_delta_batch")
        )

    def _batched_price_deltas(
        self, hedge_count: int
    ) -> list[tuple[MCResult, DeltaResult]]:
        results: list[tuple[MCResult, DeltaResult] | None] = [None] * hedge_count
        start = 0
        future_counts = np.count_nonzero(
            self.option.fixing_times[None, :]
            > self.market_times[:hedge_count, None] + _TIME_ATOL,
            axis=1,
        )
        while start < hedge_count:
            future_count = int(future_counts[start])
            end = start + 1
            while end < hedge_count and int(future_counts[end]) == future_count:
                end += 1

            # The group has one future fixing grid, so one shared normal table is
            # enough for every hedge date in the group.
            group = np.arange(start, end)
            correlated_normal_chunks = self.pricer.correlated_normal_chunks(future_count)
            group_results = self.pricer.price_and_delta_batch(
                times=self.market_times[group],
                current_spots=self.market_path[group],
                market_path=self.market_path,
                market_times=self.market_times,
                correlated_normal_chunks=correlated_normal_chunks,
            )
            for offset, result in enumerate(group_results):
                results[start + offset] = result
            start = end

        missing = [i for i, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError("batched price/delta calculation missed hedge dates")
        return [
            result
            for result in results
            if result is not None
        ]

    def _price_and_delta(
        self,
        t: float,
        current_spot: np.ndarray,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
        correlated_normal_chunks: tuple[np.ndarray, ...] | None = None,
    ) -> tuple[MCResult, DeltaResult]:
        combined = getattr(self.pricer, "price_and_delta", None)
        if combined is not None:
            kwargs = {
                "t": t,
                "current_spot": current_spot,
                "market_path": market_path,
                "market_times": market_times,
            }
            if correlated_normal_chunks is not None:
                kwargs["correlated_normal_chunks"] = correlated_normal_chunks
            return combined(**kwargs)
        return (
            self.pricer.price(t=t, market_path=market_path, market_times=market_times),
            self.pricer.delta(
                t=t,
                current_spot=current_spot,
                market_path=market_path,
                market_times=market_times,
            ),
        )

    def to_json_records(self, include_terminal: bool = False) -> list[dict[str, object]]:
        if not self.records:
            self.run()
        records = self.records if include_terminal else self.records[:-1]
        return [record.to_dict() for record in records]

    def to_json(self) -> str:
        return json.dumps(self.to_json_records(), indent=4)

    def metadata(self) -> dict[str, float]:
        if not self.records:
            self.run()
        return {
            "finalPnL": float(self.final_pnl),
            "finalPayoff": float(self.final_payoff),
            "initialPrice": float(self.records[0].price),
            "initialPriceStdDev": float(self.records[0].price_stddev),
        }
