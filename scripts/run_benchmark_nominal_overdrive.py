#!/usr/bin/env python3
"""Run benchmark in nominal mode, wait, then overdrive mode."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_benchmark.py"


def _run_mode(mode: str, passthrough: list[str]) -> None:
    cmd = [sys.executable, str(BENCHMARK_SCRIPT), "--mode", mode, *passthrough]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark in nominal then overdrive mode."
    )
    parser.add_argument(
        "--pause-sec",
        type=float,
        default=5.0,
        help="Pause duration between nominal and overdrive runs (default: 5.0)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to each benchmark run (example: -- --filter xxx)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    passthrough = args.args[1:] if args.args and args.args[0] == "--" else args.args

    _run_mode("nominal", passthrough)
    print(f"Pausing for {args.pause_sec:.1f} seconds...", flush=True)
    time.sleep(args.pause_sec)
    _run_mode("overdrive", passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
