from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from mcpricer.config import build_pricing_setup, load_pricing_setup
from mcpricer.engine.portfolio import Portfolio
from mcpricer.io.market import load_market
from mcpricer.io.output import write_json
from mcpricer.io.params import load_params


def price_summary(params_path: str | Path) -> dict[str, object]:
    """Return the price-output JSON payload for one parameter file."""

    setup = load_pricing_setup(params_path)
    start = time.perf_counter()
    price, delta = setup.pricer.price_and_delta(0.0, setup.model.spot)
    return {
        "time": time.perf_counter() - start,
        "delta": delta.mean.tolist(),
        "deltaStdDev": delta.standard_error.tolist(),
        "deltaSampleStdDev": delta.sample_stddev.tolist(),
        "deltaCi95HalfWidth": delta.ci95_half_width.tolist(),
        "price": price.mean,
        "priceStdDev": price.standard_error,
        "priceSampleStdDev": price.sample_stddev,
        "priceCi95HalfWidth": price.ci95_half_width,
    }


def hedge_records(
    market_path: str | Path,
    params_path: str | Path,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Run the discrete hedge on an observed market path."""

    params = load_params(params_path)
    setup = build_pricing_setup(params)
    market = load_market(
        market_path,
        dimension=setup.model.dimension,
        expected_rows=setup.hedging_dates_number + 1,
    )
    market_times = np.linspace(
        0.0, setup.option.maturity, setup.hedging_dates_number + 1
    )
    portfolio = Portfolio(
        model=setup.model,
        option=setup.option,
        pricer=setup.pricer,
        market_times=market_times,
        market_path=market,
    ).run()
    return portfolio.to_json_records(include_terminal=False), portfolio.metadata()


def write_all_outputs(
    market_path: str | Path,
    params_path: str | Path,
    output_prefix: str | Path,
) -> dict[str, str]:
    """Generate the three benchmark output files from one common setup."""

    prefix = Path(output_prefix)
    price_path = prefix.with_name(f"{prefix.name}_price_output.json")
    portfolio_path = prefix.with_name(f"{prefix.name}_portfolio_output.json")
    hedge_path = prefix.with_name(f"{prefix.name}_hedge_output.json")

    write_json(price_path, price_summary(params_path))

    start = time.perf_counter()
    records, metadata = hedge_records(market_path, params_path)
    metadata["time"] = time.perf_counter() - start
    write_json(portfolio_path, records)
    write_json(hedge_path, metadata)

    return {
        "price": str(price_path),
        "portfolio": str(portfolio_path),
        "hedge": str(hedge_path),
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) == 4 and argv[0] == "--all":
            _, market_file, params_file, output_prefix = argv
            write_all_outputs(market_file, params_file, output_prefix)
            return 0
        if len(argv) == 1:
            print(json.dumps(price_summary(argv[0]), indent=4))
            return 0
        if len(argv) == 2:
            summary = price_summary(argv[0])
            write_json(argv[1], summary)
            return 0
        if len(argv) == 3:
            market_file, params_file, out_file = argv
            records, _ = hedge_records(market_file, params_file)
            write_json(out_file, records)
            return 0
        if len(argv) == 4 and argv[0] == "--hedge":
            _, market_file, params_file, out_file = argv
            start = time.perf_counter()
            _, metadata = hedge_records(market_file, params_file)
            metadata["time"] = time.perf_counter() - start
            write_json(out_file, metadata)
            return 0
    except Exception as exc:
        print(f"mcpricer: {exc}", file=sys.stderr)
        return 2

    print(
        "Usage:\n"
        "  python ./mcpricer.py params.json [outfile.json]\n"
        "  python ./mcpricer.py market.txt params.json outfile.json\n"
        "  python ./mcpricer.py --hedge market.txt params.json outfile.json\n"
        "  python ./mcpricer.py --all market.txt params.json output_prefix",
        file=sys.stderr,
    )
    return 1
