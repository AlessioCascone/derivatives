from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from bsmodel import BSModel
from montecarlo import MonteCarlo
from option import Option
from portfolio import Portfolio


def load_params(paramFile):
    with open(paramFile, encoding="utf-8") as fp:
        return json.load(fp)


def new_pricer(params):
    mod = BSModel(params)
    opt = Option.new(params)
    mc = MonteCarlo(params, mod, opt)
    return mod, opt, mc


def price_summary(params):
    start = time.perf_counter()
    _, _, mc = new_pricer(params)
    price_result = mc.price()
    delta_result = mc.delta(t=0.0, current_spot=mc.mod.spot)
    return {
        "time": time.perf_counter() - start,
        "delta": delta_result.mean.tolist(),
        "deltaStdDev": delta_result.standard_error.tolist(),
        "deltaSampleStdDev": delta_result.sample_stddev.tolist(),
        "deltaCi95HalfWidth": delta_result.ci95_half_width.tolist(),
        "price": price_result.mean,
        "priceStdDev": price_result.standard_error,
        "priceSampleStdDev": price_result.sample_stddev,
        "priceCi95HalfWidth": price_result.ci95_half_width,
    }


def load_market(marketFile, modelSize, expected_rows=None):
    import pandas as pd

    table = pd.read_csv(
        marketFile,
        comment="#",
        header=None,
        sep=r"[\s,;]+",
        engine="python",
    )
    table = table.dropna(axis=1, how="all")
    if table.empty:
        raise ValueError("market file contains no data rows")

    if table.shape[1] == modelSize:
        numeric = table.apply(pd.to_numeric, errors="raise")
        prices = numeric.to_numpy(dtype=float)
    elif table.shape[1] == modelSize + 1:
        numeric = table.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
        prices = numeric.to_numpy(dtype=float)
    else:
        converted = table.apply(pd.to_numeric, errors="coerce")
        numeric_columns = [
            col for col in converted.columns if converted[col].notna().all()
        ]
        if len(numeric_columns) < modelSize:
            raise ValueError(
                f"market file must contain at least {modelSize} numeric price columns"
            )
        prices = converted[numeric_columns[-modelSize:]].to_numpy(dtype=float)

    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)
    if expected_rows is not None and prices.shape[0] != expected_rows:
        raise ValueError(
            f"market data must have shape ({expected_rows}, {modelSize}), "
            f"got {prices.shape}"
        )
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("market prices must be positive finite numbers")
    return prices


def hedge_portfolio(params, market):
    mod, opt, mc = new_pricer(params)
    H = params["hedging dates number"]
    if market.shape[0] < H + 1:
        raise ValueError(f"market path should contain at least {H + 1} dates")

    market_times = np.linspace(0.0, mod.T, H + 1)
    portfolio = Portfolio(
        mod=mod,
        opt=opt,
        mc=mc,
        market_times=market_times,
        market_path=market[: H + 1],
    ).run()
    return portfolio, portfolio.metadata()


def write_json(data, outfile):
    with open(outfile, "w", encoding="utf-8") as fp:
        if isinstance(data, str):
            fp.write(data)
        else:
            json.dump(data, fp, indent=4)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1:
        params = load_params(argv[0])
        print(json.dumps(price_summary(params), indent=4))
        return 0
    if len(argv) == 2:
        params = load_params(argv[0])
        write_json(price_summary(params), argv[1])
        return 0
    if len(argv) == 3:
        marketFile, paramFile, outFile = argv
        params = load_params(paramFile)
        mod = BSModel(params)
        market = load_market(marketFile, mod.size)
        portfolio, _ = hedge_portfolio(params, market)
        write_json(portfolio.toJson(), outFile)
        return 0
    if len(argv) == 4 and argv[0] == "--hedge":
        _, marketFile, paramFile, outFile = argv
        params = load_params(paramFile)
        mod = BSModel(params)
        market = load_market(marketFile, mod.size)
        start = time.perf_counter()
        _, hedge = hedge_portfolio(params, market)
        hedge["time"] = time.perf_counter() - start
        write_json(hedge, outFile)
        return 0

    print(
        "Usage:\n"
        "  python ./mcpricer.py params.json [outfile.json]\n"
        "  python ./mcpricer.py market.txt params.json outfile.json\n"
        "  python ./mcpricer.py --hedge market.txt params.json outfile.json",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
