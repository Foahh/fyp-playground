"""Host-side model evaluation (AP metrics).

From repo root:
  python src/benchmark/run_evaluate.py [flags]
  python project.py evaluate [flags]

Use ``--force`` / ``-f`` to re-evaluate models already present in the output CSV.

Structured logs are appended to ``evaluation.log`` in the same directory as the output CSV
(default: ``results/evaluation.log``).
"""

from __future__ import annotations

import sys

from src.benchmark.evaluate import evaluate_main


def main() -> int:
    return evaluate_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
