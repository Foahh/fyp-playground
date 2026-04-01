#!/usr/bin/env python3
"""Unified command entrypoint for common FYP playground workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.conda.conda_setup_common import (
    conda_cli_available,
    conda_run_argv,
    target_conda_env_for_command,
)

ROOT = Path(__file__).resolve().parent

LOCAL_COMMANDS: dict[str, str] = {
    "download-coco": "src/dataset/run_download_coco_dataset.py",
    "download-finetune": "src/dataset/run_download_finetune_dataset.py",
    "benchmark": "src/benchmark/run_benchmark.py",
    "generate-model": "src/benchmark/run_generate_model.py",
    "evaluate": "src/benchmark/run_evaluate.py",
    "compare": "src/benchmark/run_compare.py",
    "select-model": "src/benchmark/run_model_selection.py",
    "verify-model-config": "src/benchmark/run_check_model_dtypes.py",
    "verify-idle-power": "src/benchmark/run_verify_idle_power.py",
    "estimate-battery": "src/benchmark/run_estimate_battery.py",
    "parse-modelzoo": "src/benchmark/run_parse_modelzoo_readme.py",
    "setup-env-qtlz": "src/conda/run_conda_setup_qtlz.py",
    "setup-env-ml": "src/conda/run_conda_setup_ml.py",
    "setup-env-bhmk": "src/conda/run_conda_setup_bhmk.py",
    "train": "src/ml/run_train_tinyissimo_coco_person.py",
    "quantize": "src/ml/run_quantize.py",
    "prepare-finetune-dataset": "src/ml/run_prepare_finetune_dataset.py",
    "finetune": "src/ml/run_finetune_st.py",
    "finetune-st": "src/ml/run_finetune_st.py",
    "finetune-tinyissimoyolo": "src/ml/run_finetune_tinyissimoyolo.py",
    "finetune-yolo26": "src/ml/run_finetune_yolo26.py",
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
    inner = ["python", "-m", module_path, *passthrough]
    conda_env = target_conda_env_for_command(args.command)
    if conda_env is not None and conda_cli_available():
        cmd = conda_run_argv(conda_env, inner)
    else:
        if conda_env is not None and not conda_cli_available():
            print(
                "conda/mamba not on PATH; running with current interpreter "
                f"(expected env {conda_env!r})",
                file=sys.stderr,
            )
        cmd = [sys.executable, *inner[1:]]
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
