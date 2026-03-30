#!/usr/bin/env python3
"""Unified command entrypoint for common FYP playground workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LOCAL_COMMANDS: dict[str, str] = {
    "download-coco": "src/dataset/run_download_coco_dataset.py",
    "download-finetune": "src/dataset/run_download_finetune_dataset.py",
    "benchmark": "src/benchmark/run_benchmark.py",
    "compare-runs": "src/benchmark/run_compare.py",
    "verify-model-dtypes": "src/benchmark/run_check_model_dtypes.py",
    "parse-modelzoo-readme": "src/benchmark/run_parse_modelzoo_readme.py",
    "setup-conda-ml": "src/conda/run_conda_setup_ml.py",
    "setup-conda-bhmk": "src/conda/run_conda_setup_bhmk.py",
    "train": "src/ml/run_train_tinyissimo_coco_person.py",
    "quantize": "src/ml/run_quantize.py",
    "prepare-finetune-dataset": "src/ml/run_prepare_finetune_dataset.py",
    "finetune": "src/ml/run_finetune.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="FYP playground command runner")
    parser.add_argument(
        "command",
        choices=list(LOCAL_COMMANDS.keys()),
        help="Workflow command to execute",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the target script")
    args = parser.parse_args()

    # Normalize passthrough args (allow both "-- --flag" and "--flag" styles)
    passthrough = args.args[1:] if args.args and args.args[0] == "--" else args.args

    script_path = LOCAL_COMMANDS[args.command]
    module_path = script_path.replace("/", ".").replace(".py", "")
    cmd = [sys.executable, "-m", module_path, *passthrough]
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
