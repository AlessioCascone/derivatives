from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcpricer.config import build_pricing_setup
from mcpricer.cli import price_summary
from mcpricer.engine.portfolio import Portfolio
from mcpricer.engine.stats import DeltaResult, MCResult
from mcpricer.io.market import load_market


DATA_DIR = Path(__file__).resolve().parents[1] / "pypricer-skel" / "data"
PRICE_CASES = [
    "asian",
    "basket_2d",
    "basket_5d",
    "basket_5d_1",
    "call",
    "perf",
]


class ReplayPricer:
    """Pricer test double backed by reference portfolio records."""

    def __init__(self, model, option, records: list[dict[str, object]]) -> None:
        self.model = model
        self.option = option
        self.records = records

    def price(
        self,
        t: float = 0.0,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
    ) -> MCResult:
        index = self._index(t)
        if index < len(self.records):
            record = self.records[index]
            stddev = float(record["priceStdDev"])
            return MCResult(
                mean=float(record["price"]),
                sample_stddev=stddev,
                standard_error=stddev,
                ci95_half_width=1.96 * stddev,
            )

        if market_path is None or market_times is None:
            raise ValueError("terminal replay price requires market data")
        payoff = float(self.option.payoff(self.market_path_on_fixing_grid(market_path, market_times)))
        return MCResult(
            mean=payoff,
            sample_stddev=0.0,
            standard_error=0.0,
            ci95_half_width=0.0,
        )

    def delta(
        self,
        t: float,
        current_spot: np.ndarray,
        market_path: np.ndarray | None = None,
        market_times: np.ndarray | None = None,
    ) -> DeltaResult:
        del current_spot, market_path, market_times
        record = self.records[self._index(t)]
        stddev = np.asarray(record["deltasStdDev"], dtype=float)
        return DeltaResult(
            mean=np.asarray(record["deltas"], dtype=float),
            sample_stddev=stddev,
            standard_error=stddev,
            ci95_half_width=1.96 * stddev,
        )

    def market_path_on_fixing_grid(
        self,
        market_path: np.ndarray,
        market_times: np.ndarray,
    ) -> np.ndarray:
        market_path = np.asarray(market_path, dtype=float)
        market_times = np.asarray(market_times, dtype=float)
        rows = []
        for fixing_time in self.option.fixing_times:
            matches = np.flatnonzero(
                np.isclose(market_times, fixing_time, atol=1e-10, rtol=0.0)
            )
            if matches.size == 0:
                raise ValueError(f"market_path does not contain time {fixing_time}")
            rows.append(market_path[int(matches[-1])])
        return np.asarray(rows, dtype=float)

    def _index(self, t: float) -> int:
        if np.isclose(t, self.option.maturity):
            return len(self.records)
        return int(round(float(t) / self.option.maturity * len(self.records)))


def load_json(name: str) -> dict[str, object]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def mc_tolerance(actual_se: float, expected_se: float) -> float:
    combined_se = float(np.hypot(actual_se, expected_se))
    return max(6.0 * combined_se, 1e-12)


def build_reference_portfolio(case_name: str) -> tuple[Portfolio, list[dict[str, object]]]:
    params = load_json(f"{case_name}.json")
    setup = build_pricing_setup(params)
    expected_records = json.loads(
        (DATA_DIR / f"{case_name}_expected_portfolio.json").read_text(encoding="utf-8")
    )
    market_path = load_market(
        DATA_DIR / f"{case_name}_market.txt",
        dimension=setup.model.dimension,
        expected_rows=setup.hedging_dates_number + 1,
    )
    market_times = np.linspace(
        0.0,
        setup.option.maturity,
        setup.hedging_dates_number + 1,
    )
    pricer = ReplayPricer(setup.model, setup.option, expected_records)
    portfolio = Portfolio(
        model=setup.model,
        option=setup.option,
        pricer=pricer,
        market_times=market_times,
        market_path=market_path,
    ).run()
    return portfolio, expected_records


def hedge_pnl_tolerance(
    setup_records: list[dict[str, object]],
    market_path: np.ndarray,
    market_times: np.ndarray,
    interest_rate: float,
    maturity: float,
) -> float:
    discounted_market = market_path * np.exp(-interest_rate * market_times[:, None])
    discounted_moves = (
        discounted_market[1 : len(setup_records) + 1]
        - discounted_market[: len(setup_records)]
    )
    delta_stddev = np.asarray(
        [record["deltasStdDev"] for record in setup_records],
        dtype=float,
    )
    initial_price_stddev = float(setup_records[0]["priceStdDev"])
    discounted_wealth_stddev = np.sqrt(
        initial_price_stddev * initial_price_stddev
        + np.sum((delta_stddev * discounted_moves) ** 2)
    )
    return max(12.0 * float(np.exp(interest_rate * maturity) * discounted_wealth_stddev), 1e-12)


@pytest.mark.parametrize("case_name", PRICE_CASES)
def test_price_summary_matches_reference_data(case_name: str) -> None:
    actual = price_summary(DATA_DIR / f"{case_name}.json")
    expected = load_json(f"{case_name}_expected_price.json")

    assert actual["price"] == pytest.approx(
        expected["price"],
        abs=mc_tolerance(
            float(actual["priceStdDev"]),
            float(expected["priceStdDev"]),
        ),
    )
    assert actual["priceStdDev"] == pytest.approx(
        expected["priceStdDev"],
        rel=0.25,
        abs=1e-12,
    )

    actual_delta = np.asarray(actual["delta"], dtype=float)
    expected_delta = np.asarray(expected["delta"], dtype=float)
    actual_delta_se = np.asarray(actual["deltaStdDev"], dtype=float)
    expected_delta_se = np.asarray(expected["deltaStdDev"], dtype=float)

    assert actual_delta.shape == expected_delta.shape
    assert actual_delta_se == pytest.approx(expected_delta_se, rel=0.25, abs=1e-12)
    np.testing.assert_allclose(
        actual_delta,
        expected_delta,
        atol=mc_tolerance(float(np.max(actual_delta_se)), float(np.max(expected_delta_se))),
        rtol=0.0,
    )


@pytest.mark.parametrize("case_name", PRICE_CASES)
def test_market_files_match_reference_dimensions(case_name: str) -> None:
    params = load_json(f"{case_name}.json")
    setup = build_pricing_setup(params)

    market_path = load_market(
        DATA_DIR / f"{case_name}_market.txt",
        dimension=setup.model.dimension,
        expected_rows=setup.hedging_dates_number + 1,
    )

    assert market_path.shape == (
        setup.hedging_dates_number + 1,
        setup.model.dimension,
    )


@pytest.mark.parametrize("case_name", PRICE_CASES)
def test_portfolio_records_match_reference_data(case_name: str) -> None:
    portfolio, expected_records = build_reference_portfolio(case_name)
    actual_records = portfolio.to_json_records(include_terminal=False)

    assert len(actual_records) == len(expected_records)
    for actual, expected in zip(actual_records, expected_records):
        assert actual["date"] == expected["date"]
        assert actual["price"] == pytest.approx(expected["price"], abs=1e-12)
        assert actual["priceStdDev"] == pytest.approx(expected["priceStdDev"], abs=1e-12)
        assert actual["value"] == pytest.approx(expected["value"], abs=1e-9)
        assert actual["deltas"] == pytest.approx(expected["deltas"], abs=1e-12)
        assert actual["deltasStdDev"] == pytest.approx(expected["deltasStdDev"], abs=1e-12)


@pytest.mark.parametrize("case_name", PRICE_CASES)
def test_hedge_metadata_matches_reference_data(case_name: str) -> None:
    params = load_json(f"{case_name}.json")
    setup = build_pricing_setup(params)
    portfolio, expected_records = build_reference_portfolio(case_name)
    expected_hedge = load_json(f"{case_name}_expected_hedge.json")
    market_path = load_market(
        DATA_DIR / f"{case_name}_market.txt",
        dimension=setup.model.dimension,
        expected_rows=setup.hedging_dates_number + 1,
    )
    market_times = np.linspace(
        0.0,
        setup.option.maturity,
        setup.hedging_dates_number + 1,
    )
    metadata = portfolio.metadata()

    assert metadata["initialPrice"] == pytest.approx(
        expected_hedge["initialPrice"],
        abs=mc_tolerance(
            float(expected_records[0]["priceStdDev"]),
            float(expected_hedge["initialPriceStdDev"]),
        ),
    )
    assert metadata["initialPriceStdDev"] == pytest.approx(
        expected_hedge["initialPriceStdDev"],
        rel=0.25,
        abs=1e-12,
    )
    assert metadata["finalPnL"] == pytest.approx(
        expected_hedge["finalPnL"],
        abs=hedge_pnl_tolerance(
            expected_records,
            market_path,
            market_times,
            setup.model.interest_rate,
            setup.option.maturity,
        ),
    )


def test_main_package_does_not_import_skeleton_code() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "mcpricer"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_dir.rglob("*.py"))

    assert "pypricer-skel" not in source
    assert "pypricer_skel" not in source
