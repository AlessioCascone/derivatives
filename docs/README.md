# Monte Carlo Derivatives Pricer

`mcpricer` is a small Python package and command-line tool for Monte Carlo
pricing, finite-difference deltas, and discrete hedging experiments in a
multi-asset Black-Scholes setting.

The repository is organized around the reusable implementation. External
reference inputs, local data bundles, and PDFs are intentionally kept out of
version control; pass those files by path when you run the tool locally.

## Capabilities

- Price an option at time 0 and report Monte Carlo standard errors.
- Estimate deltas with centered finite differences and common random numbers.
- Reprice conditionally along an observed market path.
- Simulate a discrete self-financing delta-hedging portfolio.
- Write price, portfolio, and hedge summaries as JSON.
- Support one-asset calls and puts, terminal basket options, arithmetic Asian
  basket options, and cumulative positive performance payoffs.

## Project Layout

```text
mcpricer.py                 Command wrapper around mcpricer.cli.
pytest.ini                  Pytest configuration for local package imports.

mcpricer/
  cli.py                    Command-line behavior and output orchestration.
  config.py                 Builds model, option, and pricer objects from JSON.
  models/
    base.py                 Abstract stochastic model interface.
    black_scholes.py        Multi-asset Black-Scholes simulator.
  options/
    base.py                 Abstract vectorized payoff interface.
    basket.py               Basket, call, and put payoffs.
    asian.py                Arithmetic Asian basket payoff.
    performance.py          Positive performance payoff.
    factory.py              Maps JSON option type names to option classes.
  engine/
    monte_carlo.py          Price and delta estimators.
    portfolio.py            Discrete self-financing hedging portfolio.
    stats.py                Online Monte Carlo statistics.
  io/
    params.py               JSON parameter loading.
    market.py               Market path parsing.
    output.py               JSON output writing.

scripts/
  generate_live_outputs.py  Regenerates output JSON files from local inputs.

outputs/                    Generated JSON outputs from local runs.
tests/                      Unit tests and optional private reference tests.
docs/
  README.md                 This document.
```

## Model

The current model is a risk-neutral multi-asset Black-Scholes model with
constant coefficients:

```text
dS_i(t) / S_i(t) = r dt + sigma_i dW_i(t)
```

The Brownian motions may be correlated. The JSON configuration accepts either a
scalar equicorrelation value or a full square correlation matrix. Simulated
paths have shape:

```text
(number_of_paths, number_of_times, dimension)
```

The Monte Carlo engine processes paths in chunks, so large runs do not require
holding every simulated path in memory at once. Price and delta estimation can
share the same simulated future multipliers, which reduces estimator noise and
keeps related estimates easier to compare.

## Input Files

The command-line tool takes two kinds of local input:

- A parameter JSON file describing the model, payoff, Monte Carlo settings, and
  hedge grid.
- A market path text file when portfolio or hedge output is requested.

Reference or private input folders should stay local. The `.gitignore` already
excludes common local data folder names for this purpose.

### Parameter JSON

Required fields:

| Field | Meaning |
| --- | --- |
| `model type` | Currently only `bs` is supported. |
| `option size` | Number of assets. |
| `spot` | Initial spot; scalar, one-element list, or length-`D` vector. |
| `volatility` | Volatility; scalar, one-element list, or length-`D` vector. |
| `interest rate` | Constant risk-free rate. |
| `correlation` | Scalar equicorrelation or a `D x D` correlation matrix. |
| `maturity` | Option maturity. |
| `option type` | `basket`, `call`, `put`, `asian`, or `performance`. |
| `payoff coefficients` | Basket weights for basket-style payoffs. |
| `fixing dates number` | Number of intervals between time 0 and maturity. |
| `sample number` | Number of Monte Carlo paths. |
| `hedging dates number` | Number of hedge intervals. |
| `fd step` | Relative finite-difference bump used for deltas. |

Optional fields:

| Field | Meaning |
| --- | --- |
| `strike` | Strike. Defaults to `0.0`. |
| `seed` | Random seed for reproducible Monte Carlo runs. |
| `chunk size` | Number of paths processed per vectorized chunk. |

For one-asset `call` and `put` options, the implementation uses a unit payoff
coefficient internally.

### Market Path Text

Market files may be whitespace, comma, or semicolon separated. Lines beginning
with `#` are ignored. The first column may be a date or index column; otherwise
the file should contain exactly `D` price columns.

For hedging, the market file must contain `hedging dates number + 1` rows of
positive finite prices.

## Output Files

Price output is a JSON object with the price, deltas, standard errors, sample
standard deviations, confidence-interval half widths, and runtime.

Portfolio output is a JSON array with one record per hedge date before maturity:

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

Hedge output is a compact JSON object:

```json
{
    "finalPnL": 0.0,
    "finalPayoff": 0.0,
    "initialPrice": 0.0,
    "initialPriceStdDev": 0.0,
    "time": 0.0
}
```

Runtime fields are diagnostics, not correctness targets.

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

Run commands from the repository root:

```powershell
cd C:\Users\aless\my_project\derivatives
```

The command wrapper is:

```powershell
python .\mcpricer.py
```

Command forms:

| Form | What it does |
| --- | --- |
| `python .\mcpricer.py PARAMS_JSON` | Prints the price and time-0 deltas. |
| `python .\mcpricer.py PARAMS_JSON OUTFILE_JSON` | Writes the same price summary to JSON. |
| `python .\mcpricer.py MARKET_TXT PARAMS_JSON OUTFILE_JSON` | Writes the hedging portfolio path. |
| `python .\mcpricer.py --hedge MARKET_TXT PARAMS_JSON OUTFILE_JSON` | Writes only the final hedge summary. |
| `python .\mcpricer.py --all MARKET_TXT PARAMS_JSON OUTPUT_PREFIX` | Writes price, portfolio, and hedge outputs together. |

Example commands using local input paths:

```powershell
python .\mcpricer.py local-data\case.json
python .\mcpricer.py local-data\case.json outputs\case_price_output.json
python .\mcpricer.py local-data\case_market.txt local-data\case.json outputs\case_portfolio_output.json
python .\mcpricer.py --hedge local-data\case_market.txt local-data\case.json outputs\case_hedge_output.json
python .\mcpricer.py --all local-data\case_market.txt local-data\case.json outputs\case
```

The `--all` form creates files with these suffixes:

```text
*_price_output.json
*_portfolio_output.json
*_hedge_output.json
```

## Regenerating Outputs

Use the helper script when a local data directory contains parameter JSON files
with matching `*_market.txt` files:

```powershell
python scripts\generate_live_outputs.py --data-dir local-data --output-dir outputs
```

Regenerate selected cases only:

```powershell
python scripts\generate_live_outputs.py --data-dir local-data case_a case_b
```

Full portfolio and hedge generation can be slow because Monte Carlo price and
delta estimation are repeated at every hedge date.