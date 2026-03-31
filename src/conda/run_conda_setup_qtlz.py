#!/usr/bin/env python3
"""Create the ``fyp-qtlz`` conda env for Ultralytics export / INT8 TFLite quantization.

Installs repo-root ``requirements-qtlz.txt`` and applies the Ultralytics onnx2tf
per-channel quantization patch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PATCH_SCRIPT = Path(__file__).resolve().parent / "run_patch_ultralytics_per_channel_quant.py"

from src.conda.conda_setup_common import (
    conda_activate_hint,
    conda_run,
    conda_env_spec_args,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

QTLZ_ENV_NAME = os.environ.get("FYP_QTLZ_ENV", "fyp-qtlz")
PYTHON_VERSION = os.environ.get("FYP_QTLZ_PYTHON", "3.12")

def export_conda_env_yaml(env_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        ["conda", "env", "export", *conda_env_spec_args(env_name)],
        check=True,
        capture_output=True,
        text=True,
    )
    out_path.write_text(p.stdout)
    print(f"Wrote {out_path}")


def main() -> None:
    main_guard()
    root = repo_root()
    req = root / "requirements-qtlz.txt"

    if not req.is_file():
        print(f"Missing {req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(QTLZ_ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    pip_install(QTLZ_ENV_NAME, "-r", str(req))

    print(f"Patching Ultralytics onnx2tf quant → per-channel in '{QTLZ_ENV_NAME}' ...")
    conda_run(QTLZ_ENV_NAME, "python", str(PATCH_SCRIPT.resolve()))

    export_conda_env_yaml(QTLZ_ENV_NAME, root / "results" / "conda_envs" / f"{QTLZ_ENV_NAME}.yml")

    print("Done.")
    print(f"- QTLZ env: {conda_activate_hint(QTLZ_ENV_NAME)}")
    conda_run(
        QTLZ_ENV_NAME,
        "python",
        "-c",
        "import sys; print('QTLZ Python', sys.version.split()[0])",
    )


if __name__ == "__main__":
    main()

