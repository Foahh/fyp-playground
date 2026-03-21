#!/usr/bin/env python3
"""Create conda env for TinyissimoYOLO training."""

from __future__ import annotations

import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.conda.conda_setup_common import ensure_conda_env, main_guard, pip_install, repo_root

ENV_NAME = os.environ.get("TINYISSIMO_TRAIN_ENV", "tinyissimo-train")
TORCH_INDEX_URL = os.environ.get(
    "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu126"
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
    pip_install(ENV_NAME, "-r", str(root / "external" / "TinyissimoYOLO" / "requirements.txt"))

    print(f"Done. Activate with: conda activate {ENV_NAME}")


if __name__ == "__main__":
    main()
