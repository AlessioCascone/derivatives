from __future__ import annotations

from pathlib import Path

import numpy as np


def load_market(path: str | Path, dimension: int, expected_rows: int) -> np.ndarray:
    """Read a market file and return prices with shape ``(expected_rows, D)``.

    The first column may be a date/index column. Files may be whitespace,
    comma, or semicolon separated, and lines starting with ``#`` are ignored.
    """

    rows = _read_rows(path)
    if not rows:
        raise ValueError("market file contains no data rows")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("market file rows must have the same number of columns")
    column_count = widths.pop()

    if column_count == dimension:
        prices = _to_float_array(rows)
    elif column_count == dimension + 1:
        prices = _to_float_array([row[1:] for row in rows])
    else:
        numeric_columns = _numeric_columns(rows)
        if len(numeric_columns) < dimension:
            raise ValueError(
                f"market file must contain at least {dimension} numeric price columns"
            )
        selected = numeric_columns[-dimension:]
        prices = _to_float_array([[row[col] for col in selected] for row in rows])

    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)
    if prices.shape != (expected_rows, dimension):
        raise ValueError(
            f"market data must have shape ({expected_rows}, {dimension}), "
            f"got {prices.shape}"
        )
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("market prices must be positive finite numbers")
    return prices


def _read_rows(path: str | Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        normalized = line.replace(",", " ").replace(";", " ")
        tokens = normalized.split()
        if tokens:
            rows.append(tokens)
    return rows


def _to_float_array(rows: list[list[str]]) -> np.ndarray:
    try:
        return np.asarray([[float(value) for value in row] for row in rows], dtype=float)
    except ValueError as exc:
        raise ValueError("market file contains non-numeric price values") from exc


def _numeric_columns(rows: list[list[str]]) -> list[int]:
    numeric: list[int] = []
    for col in range(len(rows[0])):
        try:
            for row in rows:
                float(row[col])
        except ValueError:
            continue
        numeric.append(col)
    return numeric
