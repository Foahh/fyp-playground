"""CLI wrapper for model generation workflow.

Appended rows in ``results/generate_result.csv`` use :mod:`csv` with
``QUOTE_ALL``, so commas inside fields (e.g. paths) never split columns.
"""

from __future__ import annotations

import sys

from src.benchmark.generate_model import generate_main
from src.benchmark.utils.logutil import configure_logging, typer_install_exception_hook


def main() -> int:
    configure_logging()
    typer_install_exception_hook()
    return generate_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

