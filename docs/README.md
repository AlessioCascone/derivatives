# Scalable Monte Carlo Derivatives Pricer

This project prices, differentiates, and hedges derivatives in a
multidimensional Black-Scholes model. It is written as a small Python package
(`mcpricer`) with command wrappers at the repository root and under
`pypricer-skel`, so it can be used both as library code and as a terminal
program for the benchmark cases.

The active implementation is the `mcpricer` package. The `pypricer-skel/src`
folder is kept as a legacy/reference skeleton, while `mcpricer.py` and
`pypricer-skel/mcpricer.py` are thin wrappers that call the active package.

## What The Project Does

For an input parameter file and, when hedging is requested, a market path file,
the project can produce three kinds of result:

- Price summary: Monte Carlo price and time-0 deltas with standard errors.
- Portfolio path: the value, option price, and deltas of a discrete
  self-financing hedging portfolio at each hedge date.
- Hedge summary: final profit and loss, final payoff, and initial price
  metadata for the same hedging run.

The bundled benchmark cases are:

| Case | Option | Dimension | Files |
| --- | --- | ---: | --- |
| `call` | one-asset European call | 1 | `call.json`, `call_market.txt` |
| `asian` | arithmetic Asian basket option | 2 | `asian.json`, `asian_market.txt` |
| `basket_2d` | two-asset terminal basket call | 2 | `basket_2d.json`, `basket_2d_market.txt` |
| `basket_5d` | five-asset terminal basket call | 5 | `basket_5d.json`, `basket_5d_market.txt` |
| `basket_5d_1` | second five-asset basket benchmark | 5 | `basket_5d_1.json`, `basket_5d_1_market.txt` |
| `perf` | cumulative positive basket performance payoff | 5 | `perf.json`, `perf_market.txt` |

## Project Layout

```text
mcpricer.py                 Root command wrapper around mcpricer.cli.
pytest.ini                  Pytest configuration for local package imports.

mcpricer/
  cli.py                    Terminal command behavior.
  config.py                 Builds model, option, and pricer objects from JSON.
  models/
    base.py                 Abstract stochastic model interface.
    black_scholes.py        Multidimensional Black-Scholes simulator.
  options/
    base.py                 Abstract vectorized payoff interface.
    basket.py               Basket, standard call, and standard put payoffs.
    asian.py                Arithmetic Asian basket payoff.
    performance.py          Performance payoff.
    factory.py              Maps JSON option type names to option classes.
  engine/
    monte_carlo.py          Price and delta estimators.
    portfolio.py            Discrete self-financing hedging portfolio.
    stats.py                Online Monte Carlo mean, standard error, and CI.
  io/
    params.py               JSON parameter loading.
    market.py               Market path parsing.
    output.py               JSON output writing.

pypricer-skel/
  mcpricer.py               Compatibility wrapper around mcpricer.cli.
  data/                     Benchmark parameter, market, and expected files.
  src/                      Legacy skeleton/reference implementation.

scripts/
  generate_live_outputs.py  Regenerates live outputs for benchmark cases.

outputs/                    Generated price, portfolio, and hedge JSON files.
tests/                      Unit, CLI smoke, and reference-data tests.
docs/
  code_architecture.png     Rendered architecture diagram.
  render_code_architecture.py Regenerates code_architecture.png.
```

The architecture diagram is available at `docs/code_architecture.png`.

## Model And Numerical Method

The implemented model is a risk-neutral multidimensional Black-Scholes model
with constant coefficients:

```text
dS_i(t) / S_i(t) = r dt + sigma_i dW_i(t)
```

The Brownian motions are correlated. A scalar correlation in the JSON file is
expanded into an equicorrelation matrix; a full square correlation matrix is
also accepted. Simulated paths have shape:

```text
(number_of_paths, number_of_times, dimension)
```

For fixing dates `0 = t0 < ... < tN = T`, the Monte Carlo price at time `t` is:

```text
exp(-r * (T - t)) * E[payoff | F_t]
```

Deltas are centered finite differences. The implementation reuses the same
future random multipliers for the plus and minus bumps, which reduces variance.
When both price and deltas are needed, the pricer estimates them in one Monte
Carlo pass so the base price and bumped payoffs share the same simulated future
multipliers.

Implementation choices worth pointing out in a presentation:

- Paths are processed in chunks to keep memory stable for large samples.
- Deltas use common random numbers for the centered bumps to reduce estimator
  noise without changing the formula.
- After a hedge date, observed fixings are read from the market path and only
  future fixings are simulated.
- Asian and performance hedge dates are batched when they share the same future
  fixing grid, so one table of simulated future shocks can serve several
  conditional price/delta estimates.
- The portfolio update is done in discounted units, which is the compact form
  of the discrete self-financing condition.

The discounted hedging wealth is updated on hedge dates `tau_i` by:

```text
V_tilde[i+1] = V_tilde[i] + delta[i] . (S_tilde[i+1] - S_tilde[i])
```

where `S_tilde` and `V_tilde` are discounted by the risk-free rate.

### Hedge-Date Batching

The code uses two different forms of grouping:

- `chunk size` controls memory use inside one Monte Carlo estimate. A large
  sample is split into path chunks, accumulated by `OnlineMoments`, and the
  estimator is unchanged.
- Hedge-date batching groups several conditional repricings during a portfolio
  run. This is an execution shortcut for path-dependent payoffs, not a change
  to the pricing formula.

The batching path is in `mcpricer/engine/portfolio.py`:

1. `Portfolio.run()` checks `_uses_batched_fixing_intervals()`.
2. Batching is enabled only for `asian` and `performance` options, because
   their payoff depends on the full fixing path.
3. `_batched_price_deltas()` scans the hedge dates before maturity and groups
   consecutive dates with the same number of future fixing dates.
4. For each group, it creates one reusable set of correlated normal chunks with
   `MonteCarloPricer.correlated_normal_chunks(future_count)`.
5. It calls `MonteCarloPricer.price_and_delta_batch(...)`, which prices every
   hedge date in that group using the same future shock table.

The model-side helper is
`BlackScholesModel.future_multipliers_batch(...)`. It returns multipliers with
shape `(number_of_hedge_dates, number_of_paths, number_of_future_fixings,
dimension)`. Each hedge date has its own current time and current market spot,
but the simulated normal increments are shared across the group. That keeps the
Monte Carlo estimates easier to compare across nearby hedge dates and avoids
regenerating the same future-shock table repeatedly.

Basket, call, and put payoffs do not use this hedge-date batching path. They
are cheaper terminal-payoff cases, so `Portfolio` calls the regular
`price_and_delta()` path at each hedge date while still reusing common random
numbers when the future grid size is unchanged.

## Supported Payoffs

The payoff input array always has trailing shape `(fixing_dates_number + 1,
dimension)`.

Let:

```text
B_i = sum_j coefficient_j * S_j(t_i)
```

| JSON `option type` | Payoff |
| --- | --- |
| `basket` | `max(B_N - strike, 0)` |
| `call` | one-asset payoff `max(S_N - strike, 0)`; requires `option size = 1` |
| `put` | one-asset payoff `max(strike - S_N, 0)`; requires `option size = 1` |
| `asian` | `max(mean(B_0, ..., B_N) - strike, 0)` |
| `performance` | `1 + sum_i max(B_i / B_{i-1} - 1, 0)` |

## Input Files

### Parameter JSON

Benchmark parameter files live in `pypricer-skel/data/*.json`.

Required fields:

| Field | Meaning |
| --- | --- |
| `model type` | Currently only `bs` is supported. |
| `option size` | Number of assets, also called dimension `D`. |
| `spot` | Initial spot; scalar, one-element list, or length-`D` vector. |
| `volatility` | Asset volatility; scalar, one-element list, or length-`D` vector. |
| `interest rate` | Constant risk-free rate. |
| `correlation` | Scalar equicorrelation or a `D x D` correlation matrix. |
| `maturity` | Option maturity `T`. |
| `option type` | `basket`, `call`, `put`, `asian`, or `performance`. |
| `payoff coefficients` | Basket weights; scalar, one-element list, or length-`D` vector. Not required for `call` or `put`, which use coefficient `1.0`. |
| `fixing dates number` | Number of time intervals between `0` and maturity. |
| `sample number` | Number of Monte Carlo paths. |
| `hedging dates number` | Number of hedge intervals. |
| `fd step` | Relative finite-difference bump used for deltas. |

Optional fields:

| Field | Meaning |
| --- | --- |
| `strike` | Strike. Defaults to `0.0` if omitted. |
| `seed` | Random seed for reproducible Monte Carlo runs. |
| `chunk size` | Number of paths processed per vectorized chunk. Defaults to `min(25000, sample number)` for basket/call/put and `min(5000, sample number)` for Asian/performance payoffs. |
| `trend` | Present in benchmark files for compatibility; the risk-neutral pricer does not use it. |

### Market Path Files

Benchmark market files live in `pypricer-skel/data/*_market.txt`.

Market files may be whitespace, comma, or semicolon separated. Lines beginning
with `#` are ignored. The first column may be a date/index column; otherwise the
file should contain exactly `D` price columns. For hedging, the file must contain
`hedging dates number + 1` rows of positive finite prices.

## Output Files

Price summary output is a JSON object:

```json
{
    "time": 0.0,
    "delta": [],
    "deltaStdDev": [],
    "deltaSampleStdDev": [],
    "deltaCi95HalfWidth": [],
    "price": 0.0,
    "priceStdDev": 0.0,
    "priceSampleStdDev": 0.0,
    "priceCi95HalfWidth": 0.0
}
```

Portfolio output is a JSON array with one record per hedge date before
maturity:

```json
{
    "date": 0,
    "value": 0.0,
    "price": 0.0,
    "priceStdDev": 0.0,
    "deltas": [],
    "deltasStdDev": []
}
```

Hedge output is a JSON object:

```json
{
    "finalPnL": 0.0,
    "finalPayoff": 0.0,
    "initialPrice": 0.0,
    "initialPriceStdDev": 0.0,
    "time": 0.0
}
```

## Comparing Outputs

The `time` fields are runtime diagnostics, not correctness targets. They depend
on hardware, Python/NumPy versions, and how much work the command performs. In
particular, portfolio and hedge commands are expensive because they rerun Monte
Carlo price and delta estimation at every hedge date.

The `priceStdDev` and `deltasStdDev` fields are standard errors of the Monte
Carlo estimators. When comparing a live output with an expected output, both
numbers are noisy independent estimates. Compare prices with the combined
standard error:

```text
combinedStdErr = sqrt(actualPriceStdDev^2 + expectedPriceStdDev^2)
z = abs(actualPrice - expectedPrice) / combinedStdErr
```

For a 95% informal check, `z` should usually be below about `1.96`. The tests
use a wider tolerance to avoid random Monte Carlo failures.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas pytest
```

If your environment already has these packages, you can run the commands below
directly.

## Command Reference

All commands in this section are intended to be run from the repository root:

```powershell
cd C:\Users\aless\my_project\derivatives
```

The primary command wrapper in this checkout is:

```powershell
python .\mcpricer.py
```

The compatibility wrapper below accepts the same arguments and calls the same
implementation:

```powershell
python .\pypricer-skel\mcpricer.py
```

### Command Forms

| Form | What it does |
| --- | --- |
| `python .\mcpricer.py PARAMS_JSON` | Prints the price and time-0 deltas to the terminal. |
| `python .\mcpricer.py PARAMS_JSON OUTFILE_JSON` | Writes the same price summary to a JSON file. |
| `python .\mcpricer.py MARKET_TXT PARAMS_JSON OUTFILE_JSON` | Writes the discrete hedging portfolio path. |
| `python .\mcpricer.py --hedge MARKET_TXT PARAMS_JSON OUTFILE_JSON` | Writes only the final hedge summary. |
| `python .\mcpricer.py --all MARKET_TXT PARAMS_JSON OUTPUT_PREFIX` | Writes price, portfolio, and hedge files together. |

Argument meanings:

- `PARAMS_JSON`: option, model, Monte Carlo, and hedging parameters.
- `MARKET_TXT`: observed market path used for portfolio and hedge runs.
- `OUTFILE_JSON`: exact JSON file to create.
- `OUTPUT_PREFIX`: base path used by `--all`; suffixes are added automatically.

For example, this command:

```powershell
python .\mcpricer.py --all pypricer-skel\data\call_market.txt pypricer-skel\data\call.json outputs\call
```

creates:

```text
outputs\call_price_output.json
outputs\call_portfolio_output.json
outputs\call_hedge_output.json
```

### Price Summary Commands

Print price summaries to the terminal:

```powershell
python .\mcpricer.py pypricer-skel\data\call.json
python .\mcpricer.py pypricer-skel\data\asian.json
python .\mcpricer.py pypricer-skel\data\basket_2d.json
python .\mcpricer.py pypricer-skel\data\basket_5d.json
python .\mcpricer.py pypricer-skel\data\basket_5d_1.json
python .\mcpricer.py pypricer-skel\data\perf.json
```

Write price summaries to `outputs/`:

```powershell
python .\mcpricer.py pypricer-skel\data\call.json outputs\call_price_output.json
python .\mcpricer.py pypricer-skel\data\asian.json outputs\asian_price_output.json
python .\mcpricer.py pypricer-skel\data\basket_2d.json outputs\basket_2d_price_output.json
python .\mcpricer.py pypricer-skel\data\basket_5d.json outputs\basket_5d_price_output.json
python .\mcpricer.py pypricer-skel\data\basket_5d_1.json outputs\basket_5d_1_price_output.json
python .\mcpricer.py pypricer-skel\data\perf.json outputs\perf_price_output.json
```

### Portfolio Commands

Write hedging portfolio paths to `outputs/`:

```powershell
python .\mcpricer.py pypricer-skel\data\call_market.txt pypricer-skel\data\call.json outputs\call_portfolio_output.json
python .\mcpricer.py pypricer-skel\data\asian_market.txt pypricer-skel\data\asian.json outputs\asian_portfolio_output.json
python .\mcpricer.py pypricer-skel\data\basket_2d_market.txt pypricer-skel\data\basket_2d.json outputs\basket_2d_portfolio_output.json
python .\mcpricer.py pypricer-skel\data\basket_5d_market.txt pypricer-skel\data\basket_5d.json outputs\basket_5d_portfolio_output.json
python .\mcpricer.py pypricer-skel\data\basket_5d_1_market.txt pypricer-skel\data\basket_5d_1.json outputs\basket_5d_1_portfolio_output.json
python .\mcpricer.py pypricer-skel\data\perf_market.txt pypricer-skel\data\perf.json outputs\perf_portfolio_output.json
```

### Hedge Metadata Commands

Write hedge summaries to `outputs/`:

```powershell
python .\mcpricer.py --hedge pypricer-skel\data\call_market.txt pypricer-skel\data\call.json outputs\call_hedge_output.json
python .\mcpricer.py --hedge pypricer-skel\data\asian_market.txt pypricer-skel\data\asian.json outputs\asian_hedge_output.json
python .\mcpricer.py --hedge pypricer-skel\data\basket_2d_market.txt pypricer-skel\data\basket_2d.json outputs\basket_2d_hedge_output.json
python .\mcpricer.py --hedge pypricer-skel\data\basket_5d_market.txt pypricer-skel\data\basket_5d.json outputs\basket_5d_hedge_output.json
python .\mcpricer.py --hedge pypricer-skel\data\basket_5d_1_market.txt pypricer-skel\data\basket_5d_1.json outputs\basket_5d_1_hedge_output.json
python .\mcpricer.py --hedge pypricer-skel\data\perf_market.txt pypricer-skel\data\perf.json outputs\perf_hedge_output.json
```

### All Outputs For One Case

Write price, portfolio, and hedge outputs for each benchmark:

```powershell
python .\mcpricer.py --all pypricer-skel\data\call_market.txt pypricer-skel\data\call.json outputs\call
python .\mcpricer.py --all pypricer-skel\data\asian_market.txt pypricer-skel\data\asian.json outputs\asian
python .\mcpricer.py --all pypricer-skel\data\basket_2d_market.txt pypricer-skel\data\basket_2d.json outputs\basket_2d
python .\mcpricer.py --all pypricer-skel\data\basket_5d_market.txt pypricer-skel\data\basket_5d.json outputs\basket_5d
python .\mcpricer.py --all pypricer-skel\data\basket_5d_1_market.txt pypricer-skel\data\basket_5d_1.json outputs\basket_5d_1
python .\mcpricer.py --all pypricer-skel\data\perf_market.txt pypricer-skel\data\perf.json outputs\perf
```

### Regenerate Benchmark Outputs

Regenerate all benchmark output files in `outputs/`:

```powershell
python scripts\generate_live_outputs.py
```

Regenerate selected cases only:

```powershell
python scripts\generate_live_outputs.py call asian
python scripts\generate_live_outputs.py basket_2d basket_5d basket_5d_1 perf
```

Use custom input and output directories:

```powershell
python scripts\generate_live_outputs.py --data-dir pypricer-skel\data --output-dir outputs call
```

Full portfolio and hedge generation can be slow because Monte Carlo price and
delta estimation are repeated at every hedge date.

### Tests

Run the full test suite:

```powershell
pytest
```

Run a specific test file:

```powershell
pytest tests\test_mcpricer.py
pytest tests\test_reference_data.py
```

## Python API

You can also import the package directly:

```python
from mcpricer import load_pricing_setup

setup = load_pricing_setup("pypricer-skel/data/call.json")
price = setup.pricer.price()
delta = setup.pricer.delta(0.0, setup.model.spot)

print(price.mean, price.standard_error)
print(delta.mean, delta.standard_error)
```

For a portfolio run, use `mcpricer.cli.hedge_records` or construct
`mcpricer.engine.portfolio.Portfolio` with a model, option, pricer, market
times, and market path.

## Extending The Project

To add a new payoff:

1. Create a new `BaseOption` subclass in `mcpricer/options/`.
2. Implement `payoff(self, paths)` using vectorized NumPy operations.
3. Add the class to `_OPTIONS` in `mcpricer/options/factory.py`.
4. Add or update tests and a JSON parameter file using the new `option type`.

To add a new stochastic model:

1. Create a `BaseModel` subclass in `mcpricer/models/`.
2. Implement `simulate_paths`, `simulate_conditional`, `future_multipliers`,
   and `validate`.
3. Update `mcpricer/config.py` so JSON parameters can build the new model.
4. Add tests for path shapes, parameter validation, and pricing behavior.

## Notes

- The active CLI is exposed through `mcpricer.py`; the compatibility wrapper in
  `pypricer-skel/mcpricer.py` calls the same code.
- Monte Carlo estimates can change between runs unless a `seed` is provided in
  the parameter JSON.
- The tests compare live Monte Carlo outputs to benchmark reference data using
  tolerances based on the reported standard errors.
  