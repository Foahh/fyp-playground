#!/usr/bin/env python3
"""Prepare an ST Model Zoo object-detection dataset for finetuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OD_ROOT = ROOT / "external" / "stm32ai-modelzoo-services" / "object_detection"

CONVERTER = OD_ROOT / "datasets" / "dataset_converter" / "converter.py"
CREATE_TFS = OD_ROOT / "datasets" / "dataset_create_tfs" / "dataset_create_tfs.py"
ANALYSIS = OD_ROOT / "datasets" / "dataset_analysis" / "dataset_analysis.py"


def _hydra_config_parts(config_file: Path) -> tuple[str, str]:
    resolved = config_file.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return str(resolved.parent), resolved.stem


def _run(script: Path, config_file: Path, overrides: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"Expected script not found: {script}")

    config_path, config_name = _hydra_config_parts(config_file)
    cmd = [
        sys.executable,
        str(script),
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        *overrides,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ST Model Zoo finetuning (convert -> TFS -> optional analysis)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the dataset config YAML used by ST Model Zoo dataset tools.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip converter.py and only run dataset_create_tfs.py.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run dataset_analysis.py after TFS creation.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override passed through to each invoked dataset tool. Repeat as needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = list(args.override)

    if not args.skip_convert:
        _run(CONVERTER, args.config, overrides)

    _run(CREATE_TFS, args.config, overrides)

    if args.analyze:
        _run(ANALYSIS, args.config, overrides)

    print("Dataset preparation done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
