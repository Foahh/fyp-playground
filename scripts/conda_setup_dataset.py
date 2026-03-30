#!/usr/bin/env python3
"""Create conda env for dataset download / prep (COCO, finetune, Dataset Ninja).

``load_coco.py`` and ``load_finetune_data.py`` run here. ``dataset-tools`` (Ninja)
requires Python 3.11 or earlier because it pins pandas<=1.5.2, which has no
binary wheels for Python 3.12.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.conda.conda_setup_common import (
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

DATASET_ENV_NAME = os.environ.get("ST_DATASET_ENV", "dataset")
PYTHON_VERSION = "3.11"


def main() -> None:
    main_guard()
    root = repo_root()
    req = root / "scripts" / "requirements-dataset.txt"

    if not req.is_file():
        print(f"Missing {req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(DATASET_ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    print(f"Installing dataset dependencies into '{DATASET_ENV_NAME}' ...")
    pip_install(DATASET_ENV_NAME, "-r", str(req))

    print("Done.")
    print(f"- Dataset env: conda activate {DATASET_ENV_NAME}")
    conda_run(
        DATASET_ENV_NAME,
        "python",
        "-c",
        "import sys; import numpy; print('Dataset Python', sys.version.split()[0], 'numpy', numpy.__version__)",
    )


if __name__ == "__main__":
    main()
