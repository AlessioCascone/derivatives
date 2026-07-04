from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mcpricer.engine.monte_carlo import MonteCarloPricer
from mcpricer.io.params import load_params
from mcpricer.models.black_scholes import BlackScholesModel
from mcpricer.options.base import BaseOption
from mcpricer.options.factory import create_option


@dataclass(frozen=True, slots=True)
class PricingSetup:
    model: BlackScholesModel
    option: BaseOption
    pricer: MonteCarloPricer
    hedging_dates_number: int
    params: dict[str, Any]


def load_pricing_setup(path: str | Path) -> PricingSetup:
    return build_pricing_setup(load_params(path))


def build_pricing_setup(params: dict[str, Any]) -> PricingSetup:
    model_type = str(params.get("model type", "bs")).lower()
    if model_type != "bs":
        raise ValueError(f"unsupported model type: {model_type}")

    dimension = _read_positive_int(params, "option size")
    maturity = _read_float(params, "maturity")
    fixing_dates_number = _read_positive_int(params, "fixing dates number")
    n_samples = _read_positive_int(params, "sample number")
    hedging_dates_number = _read_positive_int(params, "hedging dates number")
    fd_step = _read_float(params, "fd step")
    if fd_step <= 0.0:
        raise ValueError("fd step must be positive")

    spot = expand_vector(params["spot"], dimension, "spot")
    volatility = expand_vector(params["volatility"], dimension, "volatility")
    correlation = _parse_correlation(params["correlation"])
    strike = float(params.get("strike", 0.0))
    option_type = str(params["option type"])
    # Plain calls and puts are one-asset instruments, so the JSON coefficient
    # field is intentionally ignored for those two aliases.
    if option_type.lower() in {"call", "put"}:
        coefficients = np.ones(1, dtype=float)
    else:
        coefficients = expand_vector(
            params["payoff coefficients"], dimension, "payoff coefficients"
        )
    # Path-dependent payoffs create larger temporary arrays; smaller chunks keep
    # memory use predictable without changing the Monte Carlo estimator.
    default_chunk_size = (
        5_000 if option_type.lower() in {"asian", "performance"} else 25_000
    )
    chunk_size = int(params.get("chunk size", min(default_chunk_size, n_samples)))
    seed = params.get("seed")
    if seed is not None:
        seed = int(seed)

    model = BlackScholesModel(
        spot=spot,
        volatility=volatility,
        interest_rate=_read_float(params, "interest rate"),
        correlation=correlation,
    )
    option = create_option(
        option_type=option_type,
        maturity=maturity,
        fixing_dates_number=fixing_dates_number,
        dimension=dimension,
        strike=strike,
        coefficients=coefficients,
    )
    pricer = MonteCarloPricer(
        model=model,
        option=option,
        n_samples=n_samples,
        fd_step=fd_step,
        seed=seed,
        chunk_size=chunk_size,
    )
    return PricingSetup(
        model=model,
        option=option,
        pricer=pricer,
        hedging_dates_number=hedging_dates_number,
        params=dict(params),
    )


def expand_vector(value: Any, dimension: int, key: str) -> np.ndarray:
    if np.isscalar(value):
        return np.full(dimension, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(dimension, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{key} must be a scalar or one-dimensional vector")
    if arr.size == 1:
        return np.full(dimension, float(arr[0]), dtype=float)
    if arr.size != dimension:
        raise ValueError(f"{key} must have length {dimension}")
    return arr.astype(float, copy=False)


def _parse_correlation(value: Any) -> float | np.ndarray:
    if np.isscalar(value):
        return float(value)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim != 2:
        raise ValueError("correlation must be a scalar or square matrix")
    return arr


def _read_positive_int(params: dict[str, Any], key: str) -> int:
    value = int(params[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _read_float(params: dict[str, Any], key: str) -> float:
    value = float(params[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value
