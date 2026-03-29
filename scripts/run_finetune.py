#!/usr/bin/env python3
"""Run ST Model Zoo object-detection finetuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STM32AI_MAIN = (
    ROOT
    / "external"
    / "stm32ai-modelzoo-services"
    / "object_detection"
    / "stm32ai_main.py"
)


def _hydra_config_parts(config_file: Path) -> tuple[str, str]:
    resolved = config_file.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return str(resolved.parent), resolved.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ST Model Zoo finetuning/e2e chained modes.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to finetune YAML config (Hydra config).",
    )
    parser.add_argument(
        "--mode",
        choices=["training", "chain_tqe", "chain_tqeb"],
        default=None,
        help="Optional operation_mode override.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override(s), repeated as needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not STM32AI_MAIN.is_file():
        raise FileNotFoundError(f"Expected script not found: {STM32AI_MAIN}")

    config_path, config_name = _hydra_config_parts(args.config)
    overrides = list(args.override)
    if args.mode:
        overrides.append(f"operation_mode={args.mode}")

    cmd = [
        sys.executable,
        str(STM32AI_MAIN),
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        *overrides,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
