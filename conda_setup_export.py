#!/usr/bin/env python3
"""Create conda env for YOLO export (PyTorch + ultralytics + TFLite patch)."""

from __future__ import annotations

import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.conda.conda_setup_common import (
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

ENV_NAME = os.environ.get("YOLO_EXPORT_ENV", "yolo-export")
TORCH_INDEX_URL = os.environ.get(
    "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128"
)


def main() -> None:
    main_guard()
    root = repo_root()
    ensure_conda_env(ENV_NAME, "3.10", "Python 3.10")

    pip_install(
        ENV_NAME,
        "torch",
        "torchvision",
        "--index-url",
        TORCH_INDEX_URL,
    )
    pip_install(ENV_NAME, "--upgrade", "ultralytics")
    conda_run(
        ENV_NAME,
        "python",
        str(root / "scripts" / "conda" / "patch_ultralytics_tflite_quant.py"),
    )

    print(f"Done. Activate with: conda activate {ENV_NAME}")
    conda_run(
        ENV_NAME,
        "python",
        "-c",
        "import ultralytics as u; print('ultralytics', u.__version__)",
    )


if __name__ == "__main__":
    main()
