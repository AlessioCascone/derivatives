from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from mcpricer.engine.stats import DeltaResult, MCResult, OnlineMoments
from mcpricer.models.base import BaseModel
from mcpricer.options.base import BaseOption


_TIME_ATOL = 1e-10


@dataclass(frozen=True, slots=True)
class _BasketBuildContext:
    fixing_times: np.ndarray
    future_mask: np.ndarray
    future_times: np.ndarray
    known_mask: np.ndarray
    known_basket_values: np.ndarray
    current_index: int | None


@dataclass(slots=True)
class MonteCarloPricer:
    """Chunked vectorized Monte Carlo pricer.

    Full payoff paths have shape ``(M, N + 1, D)`` for prices and
    ``(D, M, N + 1, D)`` for finite-difference delta paths.
    """

    model: BaseModel
    option: BaseOption
    n_samples: int
    fd_step: float
    seed: int | None = None
    chunk_size: int = 25_000
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_samples = int(self.n_samples)
        self.fd_step = float(self.fd_step)
        self.chunk_size = int(self.chunk_size)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.fd_step <= 0.0:
            raise ValueError("fd_step must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.model.dimension != self.option.dimension:
            raise ValueError("model and option dimensions do not match")
        self._rng = np.random.default_rng(self.seed)

    def price(
        self,
        t: float = 0.0,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
    ) -> MCResult:
        """Price conditionally at time ``t``.

        Observed market values are inserted on fixing dates before ``t``.
        Future fixing values are simulated conditionally from the current spot.
        """

        stats = OnlineMoments()
        t = float(t)
        current_spot = self._current_spot(t, market_path, market_times)
        context = self._basket_context(
            t=t,
            current_spot=current_spot,
            market_path=market_path,
            market_times=market_times,
        )
        discounted = np.exp(-self.model.interest_rate * (self.option.maturity - t))
        for n_chunk in self._chunk_lengths():
            multipliers = self.model.future_multipliers(
                current_time=t,
                future_times=context.future_times,
                n_paths=n_chunk,
                rng=self._rng,
            )
            payoff_samples = self._payoff_samples_from_multipliers(
                context=context,
                current_spot=current_spot,
                multipliers=multipliers,
            )
            stats.update(discounted * payoff_samples)
        return stats.as_mc_result()

    def delta(
        self,
        t: float,
        current_spot: np.ndarray,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
    ) -> DeltaResult:
        """Compute centered finite-difference deltas with common random numbers."""

        t = float(t)
        current_spot = self._validate_current_spot(current_spot)

        stats = OnlineMoments()
        context = self._basket_context(
            t=t,
            current_spot=current_spot,
            market_path=market_path,
            market_times=market_times,
        )
        discounted = np.exp(-self.model.interest_rate * (self.option.maturity - t))

        for n_chunk in self._chunk_lengths():
            multipliers = self.model.future_multipliers(
                current_time=t,
                future_times=context.future_times,
                n_paths=n_chunk,
                rng=self._rng,
            )
            payoff_diff = self._centered_payoff_diff_from_multipliers(
                context=context,
                current_spot=current_spot,
                multipliers=multipliers,
            )
            delta_samples = discounted * payoff_diff / (
                2.0 * self.fd_step * current_spot[:, None]
            )
            stats.update(delta_samples.T)
        return stats.as_delta_result()

    def price_and_delta(
        self,
        t: float = 0.0,
        current_spot: np.ndarray | None = None,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
        normal_chunks: tuple[np.ndarray, ...] | None = None,
        correlated_normal_chunks: tuple[np.ndarray, ...] | None = None,
    ) -> tuple[MCResult, DeltaResult]:
        """Estimate price and centered finite-difference deltas in one simulation pass."""

        if normal_chunks is not None and correlated_normal_chunks is not None:
            raise ValueError(
                "normal_chunks and correlated_normal_chunks cannot both be provided"
            )
        t = float(t)
        if current_spot is None:
            current_spot = self._current_spot(t, market_path, market_times)
        current_spot = self._validate_current_spot(current_spot)

        price_stats = OnlineMoments()
        delta_stats = OnlineMoments()
        context = self._basket_context(
            t=t,
            current_spot=current_spot,
            market_path=market_path,
            market_times=market_times,
        )
        discounted = np.exp(-self.model.interest_rate * (self.option.maturity - t))

        # One simulation pass gives the price and the plus/minus bump payoffs
        # the same future multipliers, reducing finite-difference noise.
        chunk_index = 0
        for chunk_index, n_chunk in enumerate(self._chunk_lengths()):
            normals = None
            correlated_normals = None
            if normal_chunks is not None:
                if chunk_index >= len(normal_chunks):
                    raise ValueError("normal_chunks has too few chunks")
                normals = normal_chunks[chunk_index]
            if correlated_normal_chunks is not None:
                if chunk_index >= len(correlated_normal_chunks):
                    raise ValueError("correlated_normal_chunks has too few chunks")
                correlated_normals = correlated_normal_chunks[chunk_index]
            multipliers = self.model.future_multipliers(
                current_time=t,
                future_times=context.future_times,
                n_paths=n_chunk,
                rng=self._rng,
                normals=normals,
                correlated_normals=correlated_normals,
            )
            fast_payoffs = self._asian_price_and_delta_payoffs_from_multipliers(
                context=context,
                current_spot=current_spot,
                multipliers=multipliers,
            )
            if fast_payoffs is None:
                payoff_samples = self._payoff_samples_from_multipliers(
                    context=context,
                    current_spot=current_spot,
                    multipliers=multipliers,
                )
                payoff_diff = self._centered_payoff_diff_from_multipliers(
                    context=context,
                    current_spot=current_spot,
                    multipliers=multipliers,
                )
            else:
                payoff_samples, payoff_diff = fast_payoffs
            price_stats.update(discounted * payoff_samples)
            delta_samples = discounted * payoff_diff / (
                2.0 * self.fd_step * current_spot[:, None]
            )
            delta_stats.update(delta_samples.T)

        if normal_chunks is not None and len(normal_chunks) != chunk_index + 1:
            raise ValueError("normal_chunks has too many chunks")
        if (
            correlated_normal_chunks is not None
            and len(correlated_normal_chunks) != chunk_index + 1
        ):
            raise ValueError("correlated_normal_chunks has too many chunks")
        return price_stats.as_mc_result(), delta_stats.as_delta_result()

    def price_and_delta_batch(
        self,
        times: np.ndarray,
        current_spots: np.ndarray,
        market_path: np.ndarray,
        market_times: np.ndarray,
        correlated_normal_chunks: tuple[np.ndarray, ...] | None = None,
    ) -> list[tuple[MCResult, DeltaResult]]:
        """Estimate price and finite-difference deltas for one fixing interval.

        All ``times`` must share the same future fixing grid. The numerical
        estimator is the same centered finite difference used by
        ``price_and_delta``; this method only adds a leading hedge-date axis.
        """

        option_type = getattr(self.option, "option_type", None)
        batch_simulator = getattr(self.model, "future_multipliers_batch", None)
        # Batching is only useful when several hedge dates have the same future
        # fixing grid and the payoff depends on the whole fixing path.
        if option_type not in {"asian", "performance"} or batch_simulator is None:
            return [
                self.price_and_delta(
                    t=float(t),
                    current_spot=current_spots[i],
                    market_path=market_path,
                    market_times=market_times,
                )
                for i, t in enumerate(times)
            ]

        times = np.asarray(times, dtype=float)
        current_spots = np.asarray(current_spots, dtype=float)
        if times.ndim != 1:
            raise ValueError("times must have shape (n_times,)")
        if current_spots.shape != (times.size, self.model.dimension):
            raise ValueError(
                f"current_spots must have shape ({times.size}, {self.model.dimension})"
            )
        if times.size == 0:
            return []

        contexts = [
            self._basket_context(
                t=float(t),
                current_spot=current_spots[i],
                market_path=market_path,
                market_times=market_times,
            )
            for i, t in enumerate(times)
        ]
        future_times = contexts[0].future_times
        for context in contexts[1:]:
            if context.future_times.shape != future_times.shape or not np.allclose(
                context.future_times,
                future_times,
                atol=_TIME_ATOL,
                rtol=0.0,
            ):
                raise ValueError("batched times must share one future fixing grid")

        price_stats = OnlineMoments()
        delta_stats = OnlineMoments()
        discounts = np.exp(-self.model.interest_rate * (self.option.maturity - times))

        chunk_index = 0
        for chunk_index, n_chunk in enumerate(self._chunk_lengths()):
            correlated_normals = None
            if correlated_normal_chunks is not None:
                if chunk_index >= len(correlated_normal_chunks):
                    raise ValueError("correlated_normal_chunks has too few chunks")
                correlated_normals = correlated_normal_chunks[chunk_index]
            multipliers = batch_simulator(
                current_times=times,
                future_times=future_times,
                n_paths=n_chunk,
                rng=self._rng,
                correlated_normals=correlated_normals,
            )
            if option_type == "asian":
                payoff_samples, payoff_diff = (
                    self._asian_batch_price_and_delta_payoffs(
                        contexts=contexts,
                        current_spots=current_spots,
                        multipliers=multipliers,
                    )
                )
            else:
                basket_values = self._batch_basket_values_from_multipliers(
                    contexts=contexts,
                    current_spots=current_spots,
                    multipliers=multipliers,
                )
                payoff_samples = self.option.payoff_from_valid_basket_values(
                    basket_values
                )
                payoff_diff = self._performance_batch_centered_payoff_diff(
                    basket_values=basket_values,
                    contexts=contexts,
                    current_spots=current_spots,
                    multipliers=multipliers,
                )

            price_stats.update((discounts[:, None] * payoff_samples).T)
            delta_samples = (
                discounts[:, None, None]
                * payoff_diff
                / (2.0 * self.fd_step * current_spots[:, :, None])
            )
            delta_stats.update(np.moveaxis(delta_samples, 2, 0))

        if (
            correlated_normal_chunks is not None
            and len(correlated_normal_chunks) != chunk_index + 1
        ):
            raise ValueError("correlated_normal_chunks has too many chunks")

        price_mean, price_sample, price_se, price_ci = price_stats.finalize()
        delta_mean, delta_sample, delta_se, delta_ci = delta_stats.finalize()
        results: list[tuple[MCResult, DeltaResult]] = []
        for i in range(times.size):
            results.append(
                (
                    MCResult(
                        mean=float(price_mean[i]),
                        sample_stddev=float(price_sample[i]),
                        standard_error=float(price_se[i]),
                        ci95_half_width=float(price_ci[i]),
                    ),
                    DeltaResult(
                        mean=np.asarray(delta_mean[i], dtype=float),
                        sample_stddev=np.asarray(delta_sample[i], dtype=float),
                        standard_error=np.asarray(delta_se[i], dtype=float),
                        ci95_half_width=np.asarray(delta_ci[i], dtype=float),
                    ),
                )
            )
        return results

    def normal_chunks(self, n_steps: int) -> tuple[np.ndarray, ...]:
        """Generate reusable standard-normal chunks for common-random-number runs."""

        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        return tuple(
            self._rng.standard_normal((n_chunk, n_steps, self.model.dimension))
            if n_steps
            else np.empty((n_chunk, 0, self.model.dimension), dtype=float)
            for n_chunk in self._chunk_lengths()
        )

    def correlated_normal_chunks(self, n_steps: int) -> tuple[np.ndarray, ...]:
        """Generate reusable correlated-normal chunks for common-random-number runs."""

        return tuple(
            self.model.correlated_normals(normals)
            if normals.shape[1]
            else normals
            for normals in self.normal_chunks(n_steps)
        )

    def market_path_on_fixing_grid(
        self, market_path: np.ndarray, market_times: np.ndarray
    ) -> np.ndarray:
        """Select observed market values on the option fixing grid."""

        market_path, market_times = self._prepare_market(market_path, market_times)
        return self._lookup_market_values(
            self.option.fixing_times,
            market_path,
            market_times,
        )

    def _build_basket_values_from_multipliers(
        self,
        context: _BasketBuildContext,
        multipliers: np.ndarray,
        current_spot: np.ndarray,
    ) -> np.ndarray:
        n_paths = multipliers.shape[0]
        basket_values = np.empty((n_paths, context.fixing_times.size), dtype=float)
        if np.any(context.known_mask):
            basket_values[:, context.known_mask] = context.known_basket_values[
                context.known_mask
            ]
        if np.any(context.future_mask):
            weighted_spot = current_spot * self.option.coefficients
            basket_values[:, context.future_mask] = multipliers @ weighted_spot
        return basket_values

    def _payoff_samples_from_multipliers(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        asian_samples = self._asian_payoff_samples_from_multipliers(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        if asian_samples is not None:
            return asian_samples

        basket_values = self._build_basket_values_from_multipliers(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        return self.option.payoff_from_valid_basket_values(basket_values)

    def _asian_payoff_samples_from_multipliers(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray | None:
        if getattr(self.option, "option_type", None) != "asian":
            return None
        average_basket = self._asian_average_basket_from_multipliers(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        return np.maximum(average_basket - self.option.strike, 0.0)

    def _asian_average_basket_from_multipliers(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        n_paths = multipliers.shape[0]
        total = np.full(n_paths, self._known_basket_sum(context), dtype=float)
        if np.any(context.future_mask):
            future_multiplier_sums = np.sum(multipliers, axis=1)
            total += future_multiplier_sums @ (
                current_spot * self.option.coefficients
            )
        return total / context.fixing_times.size

    def _asian_price_and_delta_payoffs_from_multipliers(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if getattr(self.option, "option_type", None) != "asian":
            return None

        # The Asian payoff only depends on the arithmetic average, so the code
        # shifts the average directly instead of materializing bumped paths.
        n_paths = multipliers.shape[0]
        known_sum = self._known_basket_sum(context)
        future_multiplier_sums = np.sum(multipliers, axis=1)

        total = np.full(n_paths, known_sum, dtype=float)
        if np.any(context.future_mask):
            total += future_multiplier_sums @ (
                current_spot * self.option.coefficients
            )
        base_average = total / context.fixing_times.size
        payoff_samples = np.maximum(base_average - self.option.strike, 0.0)

        component_shift = self.fd_step * self.option.coefficients * current_spot
        shift_totals = np.zeros((self.model.dimension, n_paths), dtype=float)
        if context.current_index is not None:
            shift_totals += component_shift[:, None]
        if np.any(context.future_mask):
            shift_totals += component_shift[:, None] * future_multiplier_sums.T

        shift_averages = shift_totals / context.fixing_times.size
        payoff_diff = np.maximum(
            base_average[None, :] + shift_averages - self.option.strike,
            0.0,
        ) - np.maximum(
            base_average[None, :] - shift_averages - self.option.strike,
            0.0,
        )
        return payoff_samples, payoff_diff

    def _known_basket_sum(self, context: _BasketBuildContext) -> float:
        if not np.any(context.known_mask):
            return 0.0
        return float(np.sum(context.known_basket_values[context.known_mask]))

    def _known_basket_sums(
        self, contexts: list[_BasketBuildContext]
    ) -> np.ndarray:
        return np.asarray([self._known_basket_sum(context) for context in contexts])

    def _asian_batch_price_and_delta_payoffs(
        self,
        contexts: list[_BasketBuildContext],
        current_spots: np.ndarray,
        multipliers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_dates, n_paths = multipliers.shape[:2]
        future_multiplier_sums = np.sum(multipliers, axis=2)
        weighted_spots = current_spots * self.option.coefficients[None, :]

        total = np.full(
            (n_dates, n_paths),
            self._known_basket_sums(contexts)[:, None],
            dtype=float,
        )
        if multipliers.shape[2]:
            total += np.einsum("bmd,bd->bm", future_multiplier_sums, weighted_spots)

        fixing_count = contexts[0].fixing_times.size
        base_average = total / fixing_count
        payoff_samples = np.maximum(base_average - self.option.strike, 0.0)

        component_shift = self.fd_step * weighted_spots
        shift_totals = np.zeros((n_dates, self.model.dimension, n_paths), dtype=float)
        current_rows = np.array(
            [context.current_index is not None for context in contexts],
            dtype=bool,
        )
        if np.any(current_rows):
            shift_totals[current_rows] += component_shift[current_rows, :, None]
        if multipliers.shape[2]:
            shift_totals += (
                component_shift[:, :, None]
                * np.moveaxis(future_multiplier_sums, 2, 1)
            )

        shift_averages = shift_totals / fixing_count
        payoff_diff = np.maximum(
            base_average[:, None, :] + shift_averages - self.option.strike,
            0.0,
        ) - np.maximum(
            base_average[:, None, :] - shift_averages - self.option.strike,
            0.0,
        )
        return payoff_samples, payoff_diff

    def _batch_basket_values_from_multipliers(
        self,
        contexts: list[_BasketBuildContext],
        current_spots: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        n_dates, n_paths = multipliers.shape[:2]
        fixing_count = contexts[0].fixing_times.size
        basket_values = np.empty((n_dates, n_paths, fixing_count), dtype=float)
        for i, context in enumerate(contexts):
            if np.any(context.known_mask):
                basket_values[i][:, context.known_mask] = context.known_basket_values[
                    context.known_mask
                ][None, :]
        future_mask = contexts[0].future_mask
        if np.any(future_mask):
            weighted_spots = current_spots * self.option.coefficients[None, :]
            basket_values[:, :, future_mask] = np.einsum(
                "bmfd,bd->bmf",
                multipliers,
                weighted_spots,
            )
        return basket_values

    def _performance_batch_centered_payoff_diff(
        self,
        basket_values: np.ndarray,
        contexts: list[_BasketBuildContext],
        current_spots: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        n_dates, n_paths = multipliers.shape[:2]
        payoff_diff = np.empty((n_dates, self.model.dimension, n_paths), dtype=float)
        component_shift = (
            self.fd_step * current_spots * self.option.coefficients[None, :]
        )
        current_indices = np.asarray(
            [
                -1 if context.current_index is None else context.current_index
                for context in contexts
            ],
            dtype=int,
        )
        future_positions = np.full(contexts[0].fixing_times.size, -1, dtype=int)
        future_positions[contexts[0].future_mask] = np.arange(multipliers.shape[2])

        # A performance period can move through its denominator, numerator, or
        # both, so the shifted payoff is accumulated period by period.
        for asset_index in range(self.model.dimension):
            asset_diff = payoff_diff[:, asset_index, :]
            asset_diff.fill(0.0)
            for period_index in range(contexts[0].fixing_times.size - 1):
                previous_shift = self._performance_batch_shift_at_fixing(
                    fixing_index=period_index,
                    asset_index=asset_index,
                    component_shift=component_shift,
                    current_indices=current_indices,
                    multipliers=multipliers,
                    future_positions=future_positions,
                )
                next_shift = self._performance_batch_shift_at_fixing(
                    fixing_index=period_index + 1,
                    asset_index=asset_index,
                    component_shift=component_shift,
                    current_indices=current_indices,
                    multipliers=multipliers,
                    future_positions=future_positions,
                )
                if previous_shift is None and next_shift is None:
                    continue
                self._accumulate_performance_batch_period_diff(
                    asset_diff=asset_diff,
                    base_previous=basket_values[:, :, period_index],
                    base_next=basket_values[:, :, period_index + 1],
                    previous_shift=previous_shift,
                    next_shift=next_shift,
                )
        return payoff_diff

    def _performance_batch_shift_at_fixing(
        self,
        fixing_index: int,
        asset_index: int,
        component_shift: np.ndarray,
        current_indices: np.ndarray,
        multipliers: np.ndarray,
        future_positions: np.ndarray,
    ) -> np.ndarray | None:
        shift: np.ndarray | None = None
        current_rows = current_indices == fixing_index
        if np.any(current_rows):
            shift = np.zeros(multipliers.shape[:2], dtype=float)
            shift[current_rows] = component_shift[current_rows, asset_index, None]

        future_position = int(future_positions[fixing_index])
        if future_position >= 0:
            future_shift = (
                component_shift[:, asset_index, None]
                * multipliers[:, :, future_position, asset_index]
            )
            if shift is None:
                return future_shift
            shift += future_shift
        return shift

    def _accumulate_performance_batch_period_diff(
        self,
        asset_diff: np.ndarray,
        base_previous: np.ndarray,
        base_next: np.ndarray,
        previous_shift: np.ndarray | None,
        next_shift: np.ndarray | None,
    ) -> None:
        plus_denominator = np.array(base_previous, copy=True)
        minus_denominator = np.array(base_previous, copy=True)
        plus_return = base_next - base_previous
        minus_return = np.array(plus_return, copy=True)

        if previous_shift is not None:
            plus_denominator += previous_shift
            minus_denominator -= previous_shift
            plus_return -= previous_shift
            minus_return += previous_shift
        if next_shift is not None:
            plus_return += next_shift
            minus_return -= next_shift

        if np.any(np.abs(plus_denominator) <= 1e-14) or np.any(
            np.abs(minus_denominator) <= 1e-14
        ):
            raise ValueError("performance payoff encountered a zero basket denominator")

        plus_return /= plus_denominator
        minus_return /= minus_denominator
        np.maximum(plus_return, 0.0, out=plus_return)
        np.maximum(minus_return, 0.0, out=minus_return)
        asset_diff += plus_return - minus_return

    def _build_basket_shifts(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        d = self.model.dimension
        n_paths = multipliers.shape[0]
        shifts = np.zeros((d, n_paths, context.fixing_times.size), dtype=float)
        component_shift = self.fd_step * self.option.coefficients * current_spot
        if context.current_index is not None:
            shifts[:, :, context.current_index] = component_shift[:, None]
        if np.any(context.future_mask):
            shifts[:, :, context.future_mask] = (
                component_shift[:, None, None] * np.moveaxis(multipliers, 2, 0)
            )
        return shifts

    def _centered_payoff_diff_from_multipliers(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        asian_diff = self._asian_centered_payoff_diff_from_multipliers(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        if asian_diff is not None:
            return asian_diff

        basket_values = self._build_basket_values_from_multipliers(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        performance_diff = self._performance_centered_payoff_diff_from_multipliers(
            basket_values=basket_values,
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        if performance_diff is not None:
            return performance_diff

        return self._centered_payoff_diff_from_basket_bumps(
            basket_values,
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )

    def _asian_centered_payoff_diff_from_multipliers(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray | None:
        if getattr(self.option, "option_type", None) != "asian":
            return None

        base_average = self._asian_average_basket_from_multipliers(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        component_shift = self.fd_step * self.option.coefficients * current_spot
        shift_totals = np.zeros((self.model.dimension, multipliers.shape[0]), dtype=float)

        if context.current_index is not None:
            shift_totals += component_shift[:, None]
        if np.any(context.future_mask):
            shift_totals += component_shift[:, None] * np.sum(multipliers, axis=1).T

        shift_averages = shift_totals / context.fixing_times.size
        return np.maximum(
            base_average[None, :] + shift_averages - self.option.strike,
            0.0,
        ) - np.maximum(
            base_average[None, :] - shift_averages - self.option.strike,
            0.0,
        )

    def _performance_centered_payoff_diff_from_multipliers(
        self,
        basket_values: np.ndarray,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray | None:
        if getattr(self.option, "option_type", None) != "performance":
            return None

        n_paths = multipliers.shape[0]
        payoff_diff = np.empty((self.model.dimension, n_paths), dtype=float)
        component_shift = self.fd_step * self.option.coefficients * current_spot
        future_positions = np.full(context.fixing_times.size, -1, dtype=int)
        future_positions[context.future_mask] = np.arange(multipliers.shape[1])

        for asset_index, shift_size in enumerate(component_shift):
            asset_diff = payoff_diff[asset_index]
            asset_diff.fill(0.0)
            for period_index in range(context.fixing_times.size - 1):
                previous_shift = self._performance_shift_at_fixing(
                    fixing_index=period_index,
                    asset_index=asset_index,
                    shift_size=float(shift_size),
                    context=context,
                    multipliers=multipliers,
                    future_positions=future_positions,
                )
                next_shift = self._performance_shift_at_fixing(
                    fixing_index=period_index + 1,
                    asset_index=asset_index,
                    shift_size=float(shift_size),
                    context=context,
                    multipliers=multipliers,
                    future_positions=future_positions,
                )
                if previous_shift is None and next_shift is None:
                    continue
                self._accumulate_performance_period_diff(
                    asset_diff=asset_diff,
                    base_previous=basket_values[:, period_index],
                    base_next=basket_values[:, period_index + 1],
                    previous_shift=previous_shift,
                    next_shift=next_shift,
                )
        return payoff_diff

    def _performance_shift_at_fixing(
        self,
        fixing_index: int,
        asset_index: int,
        shift_size: float,
        context: _BasketBuildContext,
        multipliers: np.ndarray,
        future_positions: np.ndarray,
    ) -> float | np.ndarray | None:
        if context.current_index == fixing_index:
            return shift_size

        future_position = int(future_positions[fixing_index])
        if future_position >= 0:
            return shift_size * multipliers[:, future_position, asset_index]
        return None

    def _accumulate_performance_period_diff(
        self,
        asset_diff: np.ndarray,
        base_previous: np.ndarray,
        base_next: np.ndarray,
        previous_shift: float | np.ndarray | None,
        next_shift: float | np.ndarray | None,
    ) -> None:
        plus_denominator = np.array(base_previous, copy=True)
        minus_denominator = np.array(base_previous, copy=True)
        plus_return = base_next - base_previous
        minus_return = np.array(plus_return, copy=True)

        if previous_shift is not None:
            plus_denominator += previous_shift
            minus_denominator -= previous_shift
            plus_return -= previous_shift
            minus_return += previous_shift
        if next_shift is not None:
            plus_return += next_shift
            minus_return -= next_shift

        if np.any(np.abs(plus_denominator) <= 1e-14) or np.any(
            np.abs(minus_denominator) <= 1e-14
        ):
            raise ValueError("performance payoff encountered a zero basket denominator")

        plus_return /= plus_denominator
        minus_return /= minus_denominator
        np.maximum(plus_return, 0.0, out=plus_return)
        np.maximum(minus_return, 0.0, out=minus_return)
        asset_diff += plus_return - minus_return

    def _centered_payoff_diff_from_basket_bumps(
        self,
        basket_values: np.ndarray,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        average_shift_payoff = getattr(
            self.option,
            "centered_payoff_diff_from_shift_averages",
            None,
        )
        if average_shift_payoff is not None:
            shift_averages = self._build_basket_shift_averages(
                context=context,
                current_spot=current_spot,
                multipliers=multipliers,
            )
            return average_shift_payoff(basket_values, shift_averages)

        shifts = self._build_basket_shifts(
            context=context,
            current_spot=current_spot,
            multipliers=multipliers,
        )
        return self.option.centered_payoff_diff_from_basket_shifts(
            basket_values,
            shifts,
        )

    def _build_basket_shift_averages(
        self,
        context: _BasketBuildContext,
        current_spot: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        n_paths = multipliers.shape[0]
        component_shift = self.fd_step * self.option.coefficients * current_spot
        shift_averages = np.zeros((self.model.dimension, n_paths), dtype=float)
        scale = 1.0 / context.fixing_times.size

        if context.current_index is not None:
            shift_averages += component_shift[:, None] * scale
        if np.any(context.future_mask):
            future_multiplier_sums = np.sum(multipliers, axis=1).T
            shift_averages += component_shift[:, None] * future_multiplier_sums * scale
        return shift_averages

    def _basket_context(
        self,
        t: float,
        current_spot: np.ndarray,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
    ) -> _BasketBuildContext:
        fixing_times = self.option.fixing_times
        # At a hedge date, past and current fixings are known. The conditional
        # simulation only covers the remaining future fixing dates.
        past_mask = fixing_times < t - _TIME_ATOL
        current_mask = np.isclose(fixing_times, t, atol=_TIME_ATOL, rtol=0.0)
        future_mask = fixing_times > t + _TIME_ATOL
        known_mask = past_mask | current_mask
        known_basket_values = np.empty(fixing_times.size, dtype=float)

        if np.any(past_mask):
            if market_path is None:
                raise ValueError("market_path is required for pricing after past fixings")
            prepared_path, prepared_times = self._prepare_market(
                market_path,
                market_times,
                t=t,
            )
            past_values = self._lookup_market_values(
                fixing_times[past_mask],
                prepared_path,
                prepared_times,
            )
            known_basket_values[past_mask] = past_values @ self.option.coefficients

        current_index: int | None = None
        if np.any(current_mask):
            current_index = int(np.flatnonzero(current_mask)[0])
            known_basket_values[current_index] = float(
                current_spot @ self.option.coefficients
            )

        return _BasketBuildContext(
            fixing_times=fixing_times,
            future_mask=future_mask,
            future_times=fixing_times[future_mask],
            known_mask=known_mask,
            known_basket_values=known_basket_values,
            current_index=current_index,
        )

    def _validate_current_spot(self, current_spot: np.ndarray) -> np.ndarray:
        current_spot = np.asarray(current_spot, dtype=float)
        if current_spot.shape != (self.model.dimension,):
            raise ValueError(f"current_spot must have shape ({self.model.dimension},)")
        if np.any(current_spot <= 0.0) or np.any(~np.isfinite(current_spot)):
            raise ValueError("current_spot values must be positive finite numbers")
        if self.fd_step <= 0.0:
            raise ValueError("fd_step must be positive")
        return current_spot

    def _current_spot(
        self,
        t: float,
        market_path: np.ndarray | None,
        market_times: np.ndarray | None,
    ) -> np.ndarray:
        if np.isclose(t, 0.0, atol=_TIME_ATOL, rtol=0.0):
            return np.asarray(self.model.spot, dtype=float)
        if market_path is None:
            raise ValueError("market_path is required when t > 0")
        market_path, market_times = self._prepare_market(market_path, market_times, t=t)
        return self._lookup_market_value(t, market_path, market_times)

    def _prepare_market(
        self,
        market_path: np.ndarray,
        market_times: np.ndarray | None,
        t: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        path = np.asarray(market_path, dtype=float)
        if path.ndim == 1:
            if self.model.dimension != 1:
                raise ValueError("one-dimensional market_path is only valid for D=1")
            path = path.reshape(-1, 1)
        if path.ndim != 2 or path.shape[1] != self.model.dimension:
            raise ValueError(
                f"market_path must have shape (n_dates, {self.model.dimension})"
            )
        if np.any(~np.isfinite(path)) or np.any(path <= 0.0):
            raise ValueError("market_path must contain positive finite values")
        if market_times is None:
            times = self._infer_market_times(path, t)
        else:
            times = np.asarray(market_times, dtype=float)
        if times.ndim != 1 or times.shape[0] != path.shape[0]:
            raise ValueError("market_times must have shape (market_path rows,)")
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) < -_TIME_ATOL):
            raise ValueError("market_times must be finite and increasing")
        return path, times

    def _infer_market_times(self, market_path: np.ndarray, t: float | None) -> np.ndarray:
        fixing_times = self.option.fixing_times
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
        return self._lookup_market_values(
            np.array([float(time)], dtype=float),
            market_path,
            market_times,
        )[0]

    def _lookup_market_values(
        self,
        times: np.ndarray,
        market_path: np.ndarray,
        market_times: np.ndarray,
    ) -> np.ndarray:
        requested = np.asarray(times, dtype=float)
        if requested.ndim != 1:
            raise ValueError("times must have shape (n_times,)")
        if requested.size == 0:
            return np.empty((0, self.model.dimension), dtype=float)

        matches = np.isclose(
            market_times[:, None],
            requested[None, :],
            atol=_TIME_ATOL,
            rtol=0.0,
        )
        has_match = np.any(matches, axis=0)
        if not np.all(has_match):
            missing = float(requested[int(np.flatnonzero(~has_match)[0])])
            raise ValueError(f"market_path does not contain time {missing}")

        reversed_indices = np.argmax(matches[::-1], axis=0)
        indices = market_times.size - 1 - reversed_indices
        return np.asarray(market_path[indices], dtype=float)

    def _chunk_lengths(self) -> Iterator[int]:
        remaining = self.n_samples
        while remaining > 0:
            n_chunk = min(self.chunk_size, remaining)
            remaining -= n_chunk
            yield n_chunk
