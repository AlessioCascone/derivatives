from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcpricer.cli import write_all_outputs


BENCHMARK_CASES = (
    "asian",
    "basket_2d",
    "basket_5d",
    "basket_5d_1",
    "call",
    "perf",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate live Monte Carlo price, portfolio, and hedge outputs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("pypricer-skel") / "data",
        help="Directory containing benchmark params and market files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where output JSON files will be written.",
    )
    parser.add_argument(
        "cases",
        nargs="*",
        default=list(BENCHMARK_CASES),
        help="Case names to generate. Defaults to all benchmark cases.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_name in args.cases:
        paths = write_all_outputs(
            market_path=args.data_dir / f"{case_name}_market.txt",
            params_path=args.data_dir / f"{case_name}.json",
            output_prefix=args.output_dir / case_name,
        )
        print(
            f"{case_name}: wrote {paths['price']}, "
            f"{paths['portfolio']}, {paths['hedge']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
