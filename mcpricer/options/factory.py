from __future__ import annotations

import numpy as np

from mcpricer.options.asian import AsianOption
from mcpricer.options.base import BaseOption
from mcpricer.options.basket import BasketOption, CallOption, PutOption
from mcpricer.options.performance import PerformanceOption


_OPTIONS: dict[str, type[BaseOption]] = {
    BasketOption.option_type: BasketOption,
    CallOption.option_type: CallOption,
    PutOption.option_type: PutOption,
    AsianOption.option_type: AsianOption,
    PerformanceOption.option_type: PerformanceOption,
}


def create_option(
    option_type: str,
    maturity: float,
    fixing_dates_number: int,
    dimension: int,
    strike: float,
    coefficients: np.ndarray,
) -> BaseOption:
    key = option_type.lower()
    try:
        cls = _OPTIONS[key]
    except KeyError as exc:
        raise ValueError(f"unknown option type: {option_type}") from exc
    return cls(
        maturity=maturity,
        fixing_dates_number=fixing_dates_number,
        dimension=dimension,
        strike=strike,
        coefficients=coefficients,
    )
