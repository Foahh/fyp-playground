#!/usr/bin/env python3
"""Create conda env for YOLO training, INT8 TFLite quantization, and dataset prep.

Includes Ultralytics and TensorFlow (via ``requirements-yolo.txt``) for
``run_train_tinyissimo_coco_person.py``, ``run_quantize.py``,
``load_coco.py``, and ``load_finetune_data.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
PATCH_SCRIPT = Path(__file__).resolve().parent / "patch_ultralytics_per_channel_quant.py"
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

    print(f"Patching Ultralytics onnx2tf quant → per-channel in '{YOLO_ENV_NAME}' ...")
    conda_run(YOLO_ENV_NAME, "python", str(PATCH_SCRIPT.resolve()))

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
