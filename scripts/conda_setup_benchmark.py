#!/usr/bin/env python3
"""Create conda env for STM32 benchmarking dependencies."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.conda.conda_setup_common import (
    conda_prefix,
    conda_run,
    ensure_conda_env,
    main_guard,
    pip_install,
    repo_root,
)

BENCHMARK_ENV_NAME = os.environ.get("ST_BENCHMARK_ENV", "benchmark")
PYTHON_VERSION = "3.12.9"


def main() -> None:
    main_guard()
    root = repo_root()
    benchmark_req = root / "external" / "stm32ai-modelzoo-services" / "requirements.txt"
    benchmark_extra_req = root / "scripts" / "requirements-benchmark.txt"

    if not benchmark_req.is_file():
        print(f"Missing {benchmark_req}", file=sys.stderr)
        sys.exit(1)
    if not benchmark_extra_req.is_file():
        print(f"Missing {benchmark_extra_req}", file=sys.stderr)
        sys.exit(1)

    ensure_conda_env(BENCHMARK_ENV_NAME, PYTHON_VERSION, f"Python {PYTHON_VERSION}")

    if platform.system() == "Linux":
        activate_d = Path(conda_prefix(BENCHMARK_ENV_NAME)) / "etc" / "conda" / "activate.d"
        activate_d.mkdir(parents=True, exist_ok=True)
        script = activate_d / "st_benchmark_ld.sh"
        script.write_text(
            "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib/\n"
        )
        print(f"Wrote {script} (LD_LIBRARY_PATH for conda CUDA libs in '{BENCHMARK_ENV_NAME}')")

    pip_install(BENCHMARK_ENV_NAME, "-r", str(benchmark_req))
    pip_install(BENCHMARK_ENV_NAME, "-r", str(benchmark_extra_req))

    print("Done.")
    print(f"- Benchmark env: conda activate {BENCHMARK_ENV_NAME}")
    conda_run(
        BENCHMARK_ENV_NAME,
        "python",
        "-c",
        "import sys; print('Benchmark Python', sys.version.split()[0])",
    )


if __name__ == "__main__":
    main()
