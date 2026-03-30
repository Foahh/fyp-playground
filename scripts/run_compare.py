"""Parse STM32 model zoo README metrics and/or compare them to benchmark_results.csv.

Official README NPU metrics assume **overdrive mode** under **default STM32Cube.AI
configuration** (per each object_detection family README *Performances → Metrics*).

From repo root:
  python scripts/run_compare.py parse
  python scripts/run_compare.py compare
  python scripts/run_compare.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure `benchmark` resolves to scripts/benchmark when run as scripts/run_compare.py
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from benchmark.compare import (  # noqa: E402
    DEFAULT_OVERDRIVE_CSV,
    DEFAULT_PARSED_CSV,
    compare_readme_to_bench,
    filter_rows_by_abs_delta_pct,
    main as compare_main,
    print_comparison_report,
)

# If argv has no subcommand and starts with `-`, we default to `all` or `compare`.
# These flags are only valid on `compare` / compare_main — default to `compare` when seen.
_COMPARE_ONLY_FLAGS = frozenset(
    ("--mode", "--benchmark", "--nominal", "--min-abs-delta-pct")
)


def _cmd_parse(argv: list[str]) -> int:
    from benchmark.parse_modelzoo_readme import write_metric_parsed_csv

    p = argparse.ArgumentParser(prog="run_compare.py parse", description="Write benchmark_parsed.csv from README tables.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output CSV (default: {DEFAULT_PARSED_CSV})",
    )
    args = p.parse_args(argv)
    out, n = write_metric_parsed_csv(path=args.out)
    print(f"Wrote {out} ({n} rows)")
    return 0


def _cmd_compare(argv: list[str]) -> int:
    return compare_main(argv)


def _cmd_all(argv: list[str]) -> int:
    from benchmark.parse_modelzoo_readme import write_metric_parsed_csv

    p = argparse.ArgumentParser(
        prog="run_compare.py all",
        description="Parse README metrics to CSV, then compare to benchmark_results.csv.",
    )
    p.add_argument(
        "--parsed",
        type=Path,
        default=DEFAULT_PARSED_CSV,
        help=f"README-parsed CSV written and read (default: {DEFAULT_PARSED_CSV})",
    )
    p.add_argument(
        "--overdrive",
        type=Path,
        default=DEFAULT_OVERDRIVE_CSV,
        help=f"Overdrive benchmark_results.csv (default: {DEFAULT_OVERDRIVE_CSV})",
    )
    p.add_argument(
        "--min-abs-delta-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Hide metric rows below PCT |Δ%%|; non-numeric Δ%% rows always shown (compare subcommand).",
    )
    args = p.parse_args(argv)

    out, n = write_metric_parsed_csv(path=args.parsed)
    print(f"Wrote {out} ({n} rows)\n")

    if not args.parsed.is_file():
        print(f"error: parsed CSV not found after write: {args.parsed}", file=sys.stderr)
        return 2
    if not args.overdrive.is_file():
        print(f"error: overdrive CSV not found: {args.overdrive}", file=sys.stderr)
        return 2

    result = compare_readme_to_bench(
        args.parsed,
        args.overdrive,
        mode="readme-overdrive",
        headline="README vs overdrive (delta = measured − readme)",
    )
    before_filter: int | None = None
    if args.min_abs_delta_pct is not None:
        if args.min_abs_delta_pct < 0:
            print("error: --min-abs-delta-pct must be >= 0", file=sys.stderr)
            return 2
        before_filter = len(result.delta_rows)
        result.delta_rows = filter_rows_by_abs_delta_pct(
            result.delta_rows, args.min_abs_delta_pct
        )

    print_comparison_report(
        result,
        delta_rows_before_filter=before_filter,
        min_abs_delta_pct=args.min_abs_delta_pct,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        argv = ["all"]
    elif argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        if any(token in _COMPARE_ONLY_FLAGS for token in argv):
            argv = ["compare"] + argv
        else:
            argv = ["all"] + argv

    if argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Commands: parse | compare | all (default: all)")
        print(f"  parse   — write README metrics CSV (default out: {DEFAULT_PARSED_CSV})")
        print(
            "  compare — tabulate deltas; --mode readme-overdrive|readme-nominal|nominal-overdrive "
            "(see --help on compare)"
        )
        print("  all     — parse then compare")
        return 0

    cmd = argv[0]
    rest = argv[1:]
    if cmd == "parse":
        return _cmd_parse(rest)
    if cmd == "compare":
        return _cmd_compare(rest)
    if cmd == "all":
        return _cmd_all(rest)

    print(f"error: unknown command {cmd!r} (use parse, compare, or all)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
