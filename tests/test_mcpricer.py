from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mcpricer.config import build_pricing_setup, expand_vector
from mcpricer.engine.monte_carlo import MonteCarloPricer
from mcpricer.engine.portfolio import Portfolio
from mcpricer.models.black_scholes import BlackScholesModel
from mcpricer.options.asian import AsianOption
from mcpricer.options.basket import BasketOption, CallOption, PutOption
from mcpricer.options.factory import create_option
from mcpricer.options.performance import PerformanceOption


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(s0: float, strike: float, r: float, sigma: float, maturity: float) -> float:
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma * sigma) * maturity) / (
        sigma * math.sqrt(maturity)
    )
    d2 = d1 - sigma * math.sqrt(maturity)
    return s0 * normal_cdf(d1) - strike * math.exp(-r * maturity) * normal_cdf(d2)


def bs_call_delta(s0: float, strike: float, r: float, sigma: float, maturity: float) -> float:
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma * sigma) * maturity) / (
        sigma * math.sqrt(maturity)
    )
    return normal_cdf(d1)


def test_black_scholes_dimensions_cholesky_and_shape() -> None:
    model = BlackScholesModel(
        spot=np.array([100.0, 95.0, 90.0]),
        volatility=np.array([0.2, 0.25, 0.3]),
        interest_rate=0.03,
        correlation=0.25,
    )
    assert model.dimension == 3
    assert model.correlation_matrix.shape == (3, 3)
    assert np.allclose(model.correlation_matrix, model.cholesky @ model.cholesky.T)

    rng = np.random.default_rng(123)
    paths = model.simulate_paths(np.array([0.0, 0.5, 1.0]), 7, rng)
    assert paths.shape == (7, 3, 3)
    assert np.allclose(paths[:, 0, :], model.spot)

    with pytest.raises(ValueError):
        BlackScholesModel(
            spot=np.ones(3),
            volatility=np.ones(3) * 0.2,
            interest_rate=0.03,
            correlation=-0.5,
        )


def test_empirical_log_return_correlation() -> None:
    rho = 0.6
    model = BlackScholesModel(
        spot=np.array([100.0, 100.0]),
        volatility=np.array([0.2, 0.35]),
        interest_rate=0.01,
        correlation=rho,
    )
    rng = np.random.default_rng(321)
    paths = model.simulate_paths(np.array([0.0, 1.0]), 80_000, rng)
    log_returns = np.log(paths[:, 1, :] / paths[:, 0, :])
    empirical_corr = np.corrcoef(log_returns.T)[0, 1]
    assert empirical_corr == pytest.approx(rho, abs=0.015)


def test_payoffs_with_manual_arrays_and_leading_dimensions() -> None:
    paths = np.array(
        [
            [[100.0, 100.0], [110.0, 90.0], [120.0, 80.0]],
            [[100.0, 100.0], [90.0, 100.0], [80.0, 110.0]],
        ]
    )
    weights = np.array([0.5, 0.5])
    basket = BasketOption(1.0, 2, 2, 95.0, weights)
    asian = AsianOption(1.0, 2, 2, 95.0, weights)
    performance = PerformanceOption(1.0, 2, 2, 0.0, weights)

    assert np.allclose(basket.payoff(paths), np.array([5.0, 0.0]))
    assert np.allclose(asian.payoff(paths), np.array([5.0, 1.6666666666666714]))
    manual_perf = 1.0 + np.sum(
        np.maximum(
            (paths[:, 1:, :] @ weights) / (paths[:, :-1, :] @ weights) - 1.0,
            0.0,
        ),
        axis=1,
    )
    assert np.allclose(performance.payoff(paths), manual_perf)

    stacked = np.stack([paths, paths + 1.0])
    assert basket.payoff(stacked).shape == (2, 2)

    for option in (basket, asian, performance):
        basket_values = option.basket_values(paths)
        assert np.allclose(option.payoff_from_basket_values(basket_values), option.payoff(paths))


def test_factory_accepts_call_and_put_payoffs() -> None:
    paths = np.array(
        [
            [[100.0], [120.0]],
            [[100.0], [80.0]],
        ]
    )
    weights = np.array([0.5])

    call = create_option("call", 1.0, 1, 1, 100.0, weights)
    put = create_option("put", 1.0, 1, 1, 100.0, weights)

    assert isinstance(call, CallOption)
    assert isinstance(put, PutOption)
    assert np.allclose(call.coefficients, np.array([1.0]))
    assert np.allclose(put.coefficients, np.array([1.0]))
    assert np.allclose(call.payoff(paths), np.array([20.0, 0.0]))
    assert np.allclose(put.payoff(paths), np.array([0.0, 20.0]))
    assert np.allclose(
        call.payoff_from_basket_values(call.basket_values(paths)),
        call.payoff(paths),
    )
    assert np.allclose(
        put.payoff_from_basket_values(put.basket_values(paths)),
        put.payoff(paths),
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_standard_call_and_put_reject_multi_asset_dimension(option_type: str) -> None:
    with pytest.raises(ValueError, match="requires exactly one asset"):
        create_option(
            option_type,
            1.0,
            1,
            2,
            100.0,
            np.array([0.5, 0.5]),
        )


def test_d1_call_price_against_analytic_black_scholes() -> None:
    s0 = 100.0
    strike = 100.0
    r = 0.04879
    sigma = 0.2
    maturity = 1.0
    model = BlackScholesModel(np.array([s0]), np.array([sigma]), r, 0.0)
    option = BasketOption(maturity, 1, 1, strike, np.array([1.0]))
    pricer = MonteCarloPricer(model, option, 160_000, fd_step=0.01, seed=7, chunk_size=40_000)

    result = pricer.price()
    expected = bs_call_price(s0, strike, r, sigma, maturity)
    assert result.mean == pytest.approx(expected, abs=4.0 * result.standard_error + 0.03)


def test_d1_call_delta_against_analytic_black_scholes() -> None:
    s0 = 100.0
    strike = 100.0
    r = 0.04879
    sigma = 0.2
    maturity = 1.0
    model = BlackScholesModel(np.array([s0]), np.array([sigma]), r, 0.0)
    option = BasketOption(maturity, 1, 1, strike, np.array([1.0]))
    pricer = MonteCarloPricer(model, option, 180_000, fd_step=0.005, seed=11, chunk_size=45_000)

    result = pricer.delta(0.0, model.spot)
    expected = bs_call_delta(s0, strike, r, sigma, maturity)
    assert result.mean[0] == pytest.approx(expected, abs=0.025)


def test_conditional_simulation_can_include_current_time() -> None:
    model = BlackScholesModel(np.array([100.0, 90.0]), np.array([0.2, 0.3]), 0.01, 0.2)
    rng = np.random.default_rng(44)
    current_spot = np.array([101.0, 87.0])
    paths = model.simulate_conditional(
        current_time=0.5,
        current_spot=current_spot,
        future_times=np.array([0.5, 0.75, 1.0]),
        n_paths=12,
        rng=rng,
    )
    assert paths.shape == (12, 3, 2)
    assert np.allclose(paths[:, 0, :], current_spot)


def test_pre_correlated_normals_match_raw_normals() -> None:
    model = BlackScholesModel(np.array([100.0, 90.0]), np.array([0.2, 0.3]), 0.01, 0.2)
    normals = np.random.default_rng(55).standard_normal((7, 3, 2))
    future_times = np.array([0.25, 0.5, 1.0])

    from_raw = model.future_multipliers(
        current_time=0.0,
        future_times=future_times,
        n_paths=7,
        rng=np.random.default_rng(1),
        normals=normals,
    )
    from_correlated = model.future_multipliers(
        current_time=0.0,
        future_times=future_times,
        n_paths=7,
        rng=np.random.default_rng(2),
        correlated_normals=model.correlated_normals(normals),
    )

    assert np.allclose(from_correlated, from_raw)


@pytest.mark.parametrize(
    "option",
    [
        AsianOption(1.0, 2, 2, 100.0, np.array([0.5, 0.5])),
        PerformanceOption(1.0, 2, 2, 0.0, np.array([0.5, 0.5])),
    ],
)
def test_batched_interval_price_and_delta_matches_sequential(option) -> None:
    model = BlackScholesModel(
        spot=np.array([100.0, 95.0]),
        volatility=np.array([0.2, 0.25]),
        interest_rate=0.03,
        correlation=0.2,
    )
    pricer = MonteCarloPricer(model, option, n_samples=20, fd_step=0.01, seed=12, chunk_size=10)
    market_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    market_path = np.array(
        [
            [100.0, 95.0],
            [101.0, 96.0],
            [102.0, 97.0],
            [103.0, 98.0],
            [104.0, 99.0],
        ]
    )
    times = market_times[:2]
    current_spots = market_path[:2]
    correlated_chunks = pricer.correlated_normal_chunks(n_steps=2)

    batched = pricer.price_and_delta_batch(
        times=times,
        current_spots=current_spots,
        market_path=market_path,
        market_times=market_times,
        correlated_normal_chunks=correlated_chunks,
    )
    sequential = [
        pricer.price_and_delta(
            t=float(t),
            current_spot=current_spots[i],
            market_path=market_path,
            market_times=market_times,
            correlated_normal_chunks=correlated_chunks,
        )
        for i, t in enumerate(times)
    ]

    for (batch_price, batch_delta), (seq_price, seq_delta) in zip(batched, sequential):
        assert batch_price.mean == pytest.approx(seq_price.mean, abs=1e-12)
        assert batch_price.standard_error == pytest.approx(
            seq_price.standard_error,
            abs=1e-12,
        )
        np.testing.assert_allclose(batch_delta.mean, seq_delta.mean, atol=1e-12)
        np.testing.assert_allclose(
            batch_delta.standard_error,
            seq_delta.standard_error,
            atol=1e-12,
        )


def test_delta_uses_one_future_multiplier_call_per_chunk() -> None:
    class CountingModel(BlackScholesModel):
        def __init__(self) -> None:
            super().__init__(np.array([100.0]), np.array([0.2]), 0.01, 0.0)
            self.calls = 0

        def future_multipliers(self, *args, **kwargs):
            self.calls += 1
            return super().future_multipliers(*args, **kwargs)

    model = CountingModel()
    option = BasketOption(1.0, 1, 1, 100.0, np.array([1.0]))
    pricer = MonteCarloPricer(model, option, n_samples=8, fd_step=0.01, seed=1, chunk_size=3)

    result = pricer.delta(0.0, model.spot)
    assert result.mean.shape == (1,)
    assert model.calls == 3


def test_price_and_delta_share_future_multiplier_calls() -> None:
    class CountingModel(BlackScholesModel):
        def __init__(self) -> None:
            super().__init__(np.array([100.0]), np.array([0.2]), 0.01, 0.0)
            self.calls = 0

        def future_multipliers(self, *args, **kwargs):
            self.calls += 1
            return super().future_multipliers(*args, **kwargs)

    model = CountingModel()
    option = BasketOption(1.0, 1, 1, 100.0, np.array([1.0]))
    pricer = MonteCarloPricer(model, option, n_samples=8, fd_step=0.01, seed=1, chunk_size=3)

    price, delta = pricer.price_and_delta(0.0, model.spot)
    assert np.isfinite(price.mean)
    assert delta.mean.shape == (1,)
    assert model.calls == 3


def test_portfolio_output_schema() -> None:
    model = BlackScholesModel(np.array([100.0]), np.array([0.2]), 0.01, 0.0)
    option = BasketOption(1.0, 1, 1, 100.0, np.array([1.0]))
    pricer = MonteCarloPricer(model, option, n_samples=300, fd_step=0.01, seed=4, chunk_size=100)
    market_times = np.array([0.0, 0.5, 1.0])
    market_path = np.array([[100.0], [101.0], [102.0]])

    portfolio = Portfolio(model, option, pricer, market_times, market_path).run()
    records = portfolio.to_json_records()
    assert len(records) == 2
    assert len(portfolio.to_json_records(include_terminal=True)) == 3
    for record in records:
        assert {"date", "value", "price", "priceStdDev", "deltas", "deltasStdDev"} <= set(record)


def test_config_scalar_to_vector_expansion() -> None:
    assert np.allclose(expand_vector([0.2], 3, "volatility"), np.array([0.2, 0.2, 0.2]))
    params = {
        "model type": "bs",
        "option size": 2,
        "strike": 100.0,
        "spot": [100.0],
        "maturity": 1.0,
        "volatility": 0.2,
        "interest rate": 0.03,
        "correlation": 0.1,
        "trend": [0.04],
        "option type": "basket",
        "payoff coefficients": [0.5],
        "fixing dates number": 1,
        "sample number": 10,
        "hedging dates number": 2,
        "fd step": 0.01,
    }
    setup = build_pricing_setup(params)
    assert np.allclose(setup.model.spot, np.array([100.0, 100.0]))
    assert np.allclose(setup.option.coefficients, np.array([0.5, 0.5]))


def test_config_call_defaults_to_unit_coefficient() -> None:
    params = {
        "model type": "bs",
        "option size": 1,
        "strike": 100.0,
        "spot": [100.0],
        "maturity": 1.0,
        "volatility": [0.2],
        "interest rate": 0.03,
        "correlation": 0.0,
        "trend": [0.04],
        "option type": "call",
        "fixing dates number": 1,
        "sample number": 10,
        "hedging dates number": 2,
        "fd step": 0.01,
    }

    setup = build_pricing_setup(params)

    assert isinstance(setup.option, CallOption)
    assert np.allclose(setup.option.coefficients, np.array([1.0]))


def test_config_call_rejects_multi_asset_dimension() -> None:
    params = {
        "model type": "bs",
        "option size": 2,
        "strike": 100.0,
        "spot": [100.0, 100.0],
        "maturity": 1.0,
        "volatility": [0.2, 0.2],
        "interest rate": 0.03,
        "correlation": 0.0,
        "trend": [0.04, 0.04],
        "option type": "call",
        "fixing dates number": 1,
        "sample number": 10,
        "hedging dates number": 2,
        "fd step": 0.01,
    }

    with pytest.raises(ValueError, match="requires exactly one asset"):
        build_pricing_setup(params)


def test_cli_smoke_with_temporary_files(tmp_path: Path) -> None:
    params = {
        "model type": "bs",
        "option size": 1,
        "strike": 100.0,
        "spot": [100.0],
        "maturity": 1.0,
        "volatility": [0.2],
        "interest rate": 0.03,
        "correlation": 0.0,
        "trend": [0.04],
        "option type": "basket",
        "payoff coefficients": [1.0],
        "fixing dates number": 1,
        "sample number": 200,
        "hedging dates number": 2,
        "fd step": 0.01,
        "seed": 123,
        "chunk size": 100,
    }
    params_path = tmp_path / "params.json"
    market_path = tmp_path / "market.csv"
    out_path = tmp_path / "out.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    market_path.write_text("0,100\n1,101\n2,103\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "mcpricer.py", str(market_path), str(params_path), str(out_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["date"] == 0


def test_cli_all_outputs_smoke_with_temporary_files(tmp_path: Path) -> None:
    params = {
        "model type": "bs",
        "option size": 1,
        "strike": 100.0,
        "spot": [100.0],
        "maturity": 1.0,
        "volatility": [0.2],
        "interest rate": 0.03,
        "correlation": 0.0,
        "trend": [0.04],
        "option type": "basket",
        "payoff coefficients": [1.0],
        "fixing dates number": 1,
        "sample number": 40,
        "hedging dates number": 2,
        "fd step": 0.01,
        "seed": 123,
        "chunk size": 20,
    }
    params_path = tmp_path / "params.json"
    market_path = tmp_path / "market.csv"
    output_prefix = tmp_path / "live_case"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    market_path.write_text("0,100\n1,101\n2,103\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "mcpricer.py",
            "--all",
            str(market_path),
            str(params_path),
            str(output_prefix),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    price_payload = json.loads(
        (tmp_path / "live_case_price_output.json").read_text(encoding="utf-8")
    )
    portfolio_payload = json.loads(
        (tmp_path / "live_case_portfolio_output.json").read_text(encoding="utf-8")
    )
    hedge_payload = json.loads(
        (tmp_path / "live_case_hedge_output.json").read_text(encoding="utf-8")
    )
    assert {"price", "delta", "time"} <= set(price_payload)
    assert len(portfolio_payload) == 2
    assert {"finalPnL", "finalPayoff", "initialPrice", "initialPriceStdDev", "time"} <= set(
        hedge_payload
    )
