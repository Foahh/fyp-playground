#!/usr/bin/env python3
"""Create the ``fyp-ml`` conda env: training, dataset prep, Ultralytics export, INT8 TFLite.

Installs conda-forge ``ultralytics`` then repo-root ``requirements-ml.txt`` (PyTorch is not
pinned here; for CUDA wheels see README).

Default Python is 3.12 so TensorFlow / ``onnx2tf`` constraints match the quantization stack.
Override with ``FYP_ML_PYTHON`` (e.g. ``3.11``) if needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.conda.conda_setup_common import (
    conda_activate_hint,
    conda_env_spec_args,
    conda_install,
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

YOLO_ENV_NAME = os.environ.get("FYP_YOLO_ENV", "fyp-ml")
PYTHON_VERSION = os.environ.get("FYP_ML_PYTHON", "3.12")


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
    yolo_req = root / "requirements-ml.txt"

    if not yolo_req.is_file():
        print(f"Missing {yolo_req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(YOLO_ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    print(f"Installing Ultralytics into '{YOLO_ENV_NAME}' from conda-forge ...")
    conda_install(YOLO_ENV_NAME, "ultralytics", channels=("conda-forge",))

    print(f"Installing pip requirements from {yolo_req} ...")
    pip_install(YOLO_ENV_NAME, "-r", str(yolo_req))

    export_conda_env_yaml(
        YOLO_ENV_NAME, root / "results" / "conda_envs" / f"{YOLO_ENV_NAME}.yml"
    )

    print("Done.")
    print(f"- ML env: {conda_activate_hint(YOLO_ENV_NAME)}")
    conda_run(
        YOLO_ENV_NAME,
        "python",
        "-c",
        "import sys; print('ML Python', sys.version.split()[0])",
    )


if __name__ == "__main__":
    main()
