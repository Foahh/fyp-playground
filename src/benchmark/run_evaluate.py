"""Host-side model evaluation (AP metrics).

From repo root:
  python src/benchmark/run_evaluate.py [flags]
  python project.py evaluate [flags]
"""

from __future__ import annotations

import sys

from src.benchmark.evaluate import evaluate_main
from src.benchmark.utils.logutil import configure_logging, typer_install_exception_hook


def main() -> int:
    configure_logging()
    typer_install_exception_hook()
    return evaluate_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
