"""Model selection / filtering for STM32N6570-DK deployment.

From repo root:
  ./project.py select-model [flags]
"""

from __future__ import annotations

import sys

from src.benchmark.model_selection import selection_main
from src.benchmark.utils.logutil import configure_logging, typer_install_exception_hook


def main() -> int:
    configure_logging()
    typer_install_exception_hook()
    return selection_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
