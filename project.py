#!/usr/bin/env python3
"""Unified command entrypoint for common FYP playground workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LOCAL_COMMANDS = {
    "coco": "scripts/load_coco.py",
    "benchmark": "scripts/run_benchmark.py",
    "conda-yolo": "scripts/conda_setup_yolo.py",
    "conda-benchmark": "scripts/conda_setup_benchmark.py",
    "train": "scripts/run_train_tinyissimo_coco_person.py",
    "export": "scripts/run_export.py",
    "quant": "scripts/run_quantize.py",
}


def _normalize_passthrough(args: list[str]) -> list[str]:
    # Allow both styles:
    #   python project.py export -- --img_size 192
    #   python project.py export --img_size 192
    if args and args[0] == "--":
        return args[1:]
    return args


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _run_local(script: str, passthrough: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script), *_normalize_passthrough(passthrough)]
    return _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FYP playground command runner")
    parser.add_argument(
        "command",
        choices=list(LOCAL_COMMANDS.keys()),
        help="Workflow command to execute",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the target script")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return _run_local(LOCAL_COMMANDS[args.command], args.args)


if __name__ == "__main__":
    raise SystemExit(main())
