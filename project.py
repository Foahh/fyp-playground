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
    ml_conda_env_name,
    st_conda_env_name,
)

ROOT = Path(__file__).resolve().parent

LOCAL_COMMANDS: dict[str, str] = {
    # env
    "setup-env-ml": "src/conda/run_conda_setup_ml.py",
    "setup-env-st": "src/conda/run_conda_setup_st.py",
    # dataset
    "download-dataset": "src/dataset/download_dataset.py",
    # benchmark
    "evaluate": "src/benchmark/run_evaluate.py",
    "generate-model": "src/benchmark/run_generate_model.py",
    "benchmark": "src/benchmark/run_benchmark.py",
    "parse-modelzoo": "src/benchmark/run_parse_modelzoo_readme.py",
    # post-benchmark
    "compare": "src/benchmark/run_compare.py",
    "select-model": "src/benchmark/run_model_selection.py",
    "verify-model-config": "src/benchmark/run_check_model_dtypes.py",
    "verify-idle-power": "src/benchmark/run_verify_idle_power.py",
    "estimate-battery": "src/benchmark/run_estimate_battery.py",
    # train
    "train-person": "src/ml/run_train_tinyissimo_coco_person.py",
    # finetune
    # quantize
    "quantize-tiny": "src/ml/run_quantize_tinyissimo_coco_person.py",
    # dev
    "format": "src/dev/run_format.py",
}

_BASE_ENV_COMMANDS = frozenset(
    {
        "setup-env-ml",
        "setup-env-st",
    }
)

_ML_COMMANDS = frozenset(
    {
        "download-dataset",
        "train-person",
        "quantize-tiny",
    }
)

_FORMAT_COMMANDS = frozenset({"format"})

_ST_COMMANDS = frozenset(
    {
        "benchmark",
        "generate-model",
        "evaluate",
        "compare",
        "select-model",
        "verify-model-config",
        "verify-idle-power",
        "estimate-battery",
        "parse-modelzoo",
    }
)


def target_conda_env_for_command(command: str) -> str | None:
    """Return the conda env name to run ``command`` in, or ``None`` for base-env setup."""
    if command in _BASE_ENV_COMMANDS:
        return None
    if command in _FORMAT_COMMANDS:
        return ml_conda_env_name()
    if command in _ML_COMMANDS:
        return ml_conda_env_name()
    if command in _ST_COMMANDS:
        return st_conda_env_name()
    raise ValueError(f"Unknown command for conda routing: {command!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FYP playground command runner")
    parser.add_argument(
        "command",
        choices=list(LOCAL_COMMANDS.keys()),
        help="Workflow command to execute",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the target script",
    )
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
