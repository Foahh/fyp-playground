#!/usr/bin/env python3
"""Run ST Model Zoo object-detection finetuning (TensorFlow / Hydra pipeline)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[2]
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


app = typer.Typer()


@app.command()
def main(
    config: Path = typer.Option(
        ..., help="Path to finetune YAML config (Hydra config)."
    ),
    mode: str | None = typer.Option(
        None, help="Optional operation_mode override (training, chain_tqe, chain_tqeb)."
    ),
    override: list[str] = typer.Option(
        [], help="Additional Hydra override(s), repeated as needed."
    ),
) -> int:
    if mode and mode not in ["training", "chain_tqe", "chain_tqeb"]:
        typer.echo(
            f"Error: mode must be one of [training, chain_tqe, chain_tqeb]", err=True
        )
        raise typer.Exit(1)
    if not STM32AI_MAIN.is_file():
        raise FileNotFoundError(f"Expected script not found: {STM32AI_MAIN}")

    config_path, config_name = _hydra_config_parts(config)
    overrides = list(override)
    if mode:
        overrides.append(f"operation_mode={mode}")

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
    raise SystemExit(app())
