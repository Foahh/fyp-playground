#!/usr/bin/env python3
"""Create conda env for STM32 benchmark (CUDA, modelzoo deps, README metrics parser)."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.conda.conda_setup_common import (
    conda_install,
    conda_prefix,
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

ENV_NAME = os.environ.get("ST_BENCHMARK_ENV", "fyp")
PYTHON_VERSION = "3.12.9"
CUDA_VERSION = "12.8"


def main() -> None:
    main_guard()
    root = repo_root()
    req = root / "external" / "stm32ai-modelzoo-services" / "requirements.txt"
    if not req.is_file():
        print(f"Missing {req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    print("Installing NVIDIA CUDA runtime + cuDNN (conda-forge) ...")
    conda_install(ENV_NAME, f"cudatoolkit={CUDA_VERSION}", "cudnn", channels=("conda-forge",))

    if platform.system() == "Linux":
        activate_d = Path(conda_prefix(ENV_NAME)) / "etc" / "conda" / "activate.d"
        activate_d.mkdir(parents=True, exist_ok=True)
        script = activate_d / "st_benchmark_ld.sh"
        script.write_text(
            "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib/\n"
        )
        print(f"Wrote {script} (LD_LIBRARY_PATH for conda CUDA libs)")

    pip_install(ENV_NAME, "-r", str(req))
    pip_install(ENV_NAME, "-r", str(root / "requirements.txt"))

    print(f"Done. Activate with: conda activate {ENV_NAME}")
    conda_run(
        ENV_NAME,
        "python",
        "-c",
        "import sys; print('Python', sys.version.split()[0])",
    )


if __name__ == "__main__":
    main()
