from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from bsmodel import BSModel
from montecarlo import MonteCarlo
from option import Option


@dataclass(slots=True)
class Position:
    date: int
    value: float
    price: float
    price_stddev: float
    deltas: np.ndarray
    deltas_stddev: np.ndarray

    def toDict(self) -> dict:
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
    mod: BSModel
    opt: Option
    mc: MonteCarlo
    market_times: np.ndarray
    market_path: np.ndarray
    positions: list[Position] = field(default_factory=list)
    final_pnl: float | None = None
    final_payoff: float | None = None

    def __post_init__(self) -> None:
        self.market_times = np.asarray(self.market_times, dtype=float)
        self.market_path = np.asarray(self.market_path, dtype=float)
        if self.market_path.ndim == 1:
            if self.mod.dimension != 1:
                raise ValueError(
                    "one-dimensional market_path is only valid for D=1"
                )
            self.market_path = self.market_path.reshape(-1, 1)
        if (
            self.market_path.ndim != 2
            or self.market_path.shape[1] != self.mod.dimension
        ):
            raise ValueError(
                f"market_path must have shape (H + 1, {self.mod.dimension})"
            )
        if (
            self.market_times.ndim != 1
            or self.market_times.shape[0] != self.market_path.shape[0]
        ):
            raise ValueError("market_times must have shape (H + 1,)")
        if not np.isclose(self.market_times[0], 0.0):
            raise ValueError("market_times must start at 0")
        if not np.isclose(self.market_times[-1], self.opt.T):
            raise ValueError("market_times must end at option maturity")
        if np.any(np.diff(self.market_times) <= 0.0):
            raise ValueError("market_times must be strictly increasing")
        if np.any(~np.isfinite(self.market_path)) or np.any(
            self.market_path <= 0.0
        ):
            raise ValueError("market_path must contain positive finite values")

    def run(self) -> Portfolio:
        self.positions.clear()
        self.final_pnl = None
        self.final_payoff = None
        h = self.market_times.size - 1
        discounted_wealth: float | None = None
        previous_delta: np.ndarray | None = None

        for i, t in enumerate(self.market_times):
            current_spot = self.market_path[i]
            if i == 0:
                price_result = self.mc.price(t=0.0)
                discounted_wealth = price_result.mean
                wealth = price_result.mean
            else:
                if discounted_wealth is None or previous_delta is None:
                    raise RuntimeError("portfolio state was not initialized")
                prev_t = self.market_times[i - 1]
                discounted_previous = self.market_path[i - 1] * np.exp(
                    -self.mod.r * prev_t
                )
                discounted_current = current_spot * np.exp(-self.mod.r * t)
                discounted_wealth += float(
                    previous_delta @ (discounted_current - discounted_previous)
                )
                wealth = discounted_wealth * np.exp(self.mod.r * t)
                price_result = self.mc.price(
                    t=float(t),
                    market_path=self.market_path[: i + 1],
                    market_times=self.market_times[: i + 1],
                )

            if i < h:
                delta_result = self.mc.delta(
                    t=float(t),
                    current_spot=current_spot,
                    market_path=self.market_path[: i + 1],
                    market_times=self.market_times[: i + 1],
                )
                deltas = delta_result.mean
                deltas_stddev = delta_result.standard_error
                previous_delta = deltas
            else:
                deltas = np.zeros(self.mod.dimension, dtype=float)
                deltas_stddev = np.zeros(self.mod.dimension, dtype=float)

            self.positions.append(
                Position(
                    date=i,
                    value=float(wealth),
                    price=price_result.mean,
                    price_stddev=price_result.standard_error,
                    deltas=deltas,
                    deltas_stddev=deltas_stddev,
                )
            )

        fixing_path = self.mc.market_path_on_fixing_grid(
            self.market_path, self.market_times
        )
        self.final_payoff = float(self.opt.payoff(fixing_path))
        self.final_pnl = float(self.positions[-1].value - self.final_payoff)
        return self

    @property
    def model(self):
        return self.mod

    @property
    def option(self):
        return self.opt

    @property
    def pricer(self):
        return self.mc

    def to_json_records(self) -> list[dict]:
        if not self.positions:
            self.run()
        return [p.toDict() for p in self.positions]

    def toJson(self) -> str:
        return json.dumps(self.to_json_records(), indent=4)

    def metadata(self) -> dict[str, float]:
        if not self.positions:
            self.run()
        return {
            "finalPnL": float(self.final_pnl),
            "finalPayoff": float(self.final_payoff),
            "initialPrice": float(self.positions[0].price),
            "initialPriceStdDev": float(self.positions[0].price_stddev),
        }
