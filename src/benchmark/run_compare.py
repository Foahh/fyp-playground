"""Compare benchmark results (no parsing — use ``./project.py parse-modelzoo`` for that).

From repo root:
  ./project.py compare [compare flags]
  ./project.py compare --delta METRIC:NUMBER ...
  ./project.py compare --exclude METRIC_OR_MODEL_SUBSTRING ...
  ./project.py compare --table-format ai ...
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
