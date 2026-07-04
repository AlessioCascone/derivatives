from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcpricer.cli import write_all_outputs


def discover_cases(data_dir: Path) -> list[str]:
    return sorted(
        params_path.stem
        for params_path in data_dir.glob("*.json")
        if (data_dir / f"{params_path.stem}_market.txt").exists()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate live Monte Carlo price, portfolio, and hedge outputs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("local-data"),
        help="Directory containing params JSON files and matching market files.",
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
        help=(
            "Case names to generate. Defaults to every JSON file in --data-dir "
            "with a matching *_market.txt file."
        ),
    )
    args = parser.parse_args(argv)

    cases = args.cases or discover_cases(args.data_dir)
    if not cases:
        parser.error(
            f"no cases found in {args.data_dir}; pass case names or choose another --data-dir"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_name in cases:
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
