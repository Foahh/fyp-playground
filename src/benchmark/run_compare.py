"""Compare benchmark results (no parsing - use 'python project.py parse-modelzoo-readme' for that).

From repo root:
  python src/benchmark/run_compare.py [compare flags]
  python src/benchmark/run_compare.py --delta METRIC:NUMBER ...
  python src/benchmark/run_compare.py --exclude METRIC_OR_MODEL_SUBSTRING ...
  python project.py compare [flags]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.benchmark.compare import compare_main
from src.benchmark.utils.logutil import configure_logging, typer_install_exception_hook


def main() -> int:
    configure_logging()
    typer_install_exception_hook()
    return compare_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
