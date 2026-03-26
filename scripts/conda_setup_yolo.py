#!/usr/bin/env python3
"""Create conda env for YOLO training/export dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.conda.conda_setup_common import (
    conda_install,
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

YOLO_ENV_NAME = os.environ.get("ST_YOLO_ENV", "yolo")
PYTHON_VERSION = "3.12"
PYTORCH_WHL_INDEX = "https://download.pytorch.org/whl/cu128"


def main() -> None:
    main_guard()
    root = repo_root()
    yolo_req = root / "scripts" / "requirements-yolo.txt"

    if not yolo_req.is_file():
        print(f"Missing {yolo_req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(YOLO_ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    print(f"Installing Ultralytics into '{YOLO_ENV_NAME}' from conda-forge ...")
    conda_install(
        YOLO_ENV_NAME,
        "ultralytics",
        channels=("conda-forge",),
    )
    print(f"Installing PyTorch wheels into '{YOLO_ENV_NAME}' from {PYTORCH_WHL_INDEX} ...")
    pip_install(
        YOLO_ENV_NAME,
        "torch",
        "torchvision",
        "--index-url",
        PYTORCH_WHL_INDEX,
    )
    pip_install(YOLO_ENV_NAME, "-r", str(yolo_req))

    print("Done.")
    print(f"- YOLO env: conda activate {YOLO_ENV_NAME}")
    conda_run(
        YOLO_ENV_NAME,
        "python",
        "-c",
        "import sys; print('YOLO Python', sys.version.split()[0])",
    )


if __name__ == "__main__":
    main()
