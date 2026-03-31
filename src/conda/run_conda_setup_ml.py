#!/usr/bin/env python3
"""Create the ``fyp-ml`` conda env for training, INT8 TFLite quantization, and dataset prep.

Includes Ultralytics and TensorFlow (via repo-root ``requirements-ml.txt``) for
``src/ml/run_train_tinyissimo_coco_person.py``, ``src/ml/run_quantize.py``,
``src/dataset/run_download_coco_dataset.py``, and ``src/dataset/run_download_finetune_dataset.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PATCH_SCRIPT = Path(__file__).resolve().parent / "run_patch_ultralytics_per_channel_quant.py"

from src.conda.conda_setup_common import (
    conda_activate_hint,
    conda_install,
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

YOLO_ENV_NAME = os.environ.get("FYP_YOLO_ENV", "fyp-ml")
PYTHON_VERSION = "3.14"
PYTORCH_WHL_INDEX = "https://download.pytorch.org/whl/cu128"


def main() -> None:
    main_guard()
    root = repo_root()
    yolo_req = root / "requirements-ml.txt"

    if not yolo_req.is_file():
        print(f"Missing {yolo_req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(YOLO_ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    print(f"Installing PyTorch wheels into '{YOLO_ENV_NAME}' from {PYTORCH_WHL_INDEX} ...")
    pip_install(
        YOLO_ENV_NAME,
        "torch",
        "torchvision",
        "--index-url",
        PYTORCH_WHL_INDEX,
    )

    print(f"Installing Ultralytics into '{YOLO_ENV_NAME}' from conda-forge ...")
    conda_install(YOLO_ENV_NAME, "ultralytics", channels=("conda-forge",))

    pip_install(YOLO_ENV_NAME, "-r", str(yolo_req))

    print(f"Patching Ultralytics onnx2tf quant → per-channel in '{YOLO_ENV_NAME}' ...")
    conda_run(YOLO_ENV_NAME, "python", str(PATCH_SCRIPT.resolve()))

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
