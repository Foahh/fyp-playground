"""Host-side model evaluation (AP metrics) — decoupled from on-target benchmark.

Runs the STM32 Model Zoo evaluator for each registered model and writes
results to ``results/evaluation_result.csv``.

From repo root:
  python src/benchmark/run_evaluate.py [flags]
  python project.py evaluate [flags]
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import typer

from .core.config import build_eval_config
from .core.models import ModelEntry, load_models
from .io.parsing import parse_metrics
from .io.results import append_eval_result, load_eval_completed
from .paths import RESULTS_DIR, SERVICES_DIR
from .utils.logutil import configure_logging, get_logger, typer_install_exception_hook

DEFAULT_EVAL_CSV = RESULTS_DIR / "evaluation_result.csv"


def _run_evaluate(entry: ModelEntry) -> tuple[str, str, int]:
    """Run the Model Zoo evaluator for a single model entry."""
    build_eval_config(entry)

    cmd = [
        sys.executable,
        "stm32ai_main.py",
        "--config-path",
        ".",
        "--config-name",
        "_benchmark_temp_config",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["TQDM_DISABLE"] = "1"
    # Force TensorFlow host evaluation onto CPU to avoid CUDA runtime instability.
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SERVICES_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        return stdout, stderr, -1


eval_app = typer.Typer(
    help=__doc__,
    add_completion=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)


@eval_app.callback()
def run_evaluate(
    ctx: typer.Context,
    filter_substr: str = typer.Option(
        "",
        "--filter",
        help="Only evaluate variants whose name contains this string",
    ),
    output: Path = typer.Option(
        DEFAULT_EVAL_CSV,
        "--output",
        "-o",
        help="Output CSV path for evaluation results",
    ),
) -> None:
    """Run host-side evaluation for all registered models."""
    if getattr(ctx, "resilient_parsing", False):
        return

    configure_logging()
    typer_install_exception_hook()
    logger = get_logger("evaluate")

    entries = load_models()
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))

    if filter_substr:
        entries = [e for e in entries if filter_substr in e.variant]

    completed = load_eval_completed(output)
    total = len(entries)

    logger.info(
        "Evaluation started",
        total=total,
        completed=len(completed),
        filter=filter_substr or None,
        output_csv=str(output),
    )

    for i, entry in enumerate(entries, 1):
        key = (entry.variant, entry.fmt)

        if key in completed:
            logger.info(f"[{i}/{total}] Skipped (already evaluated)", variant=entry.variant)
            continue

        logger.info(
            f"[{i}/{total}] Evaluating",
            variant=entry.variant,
            fmt=entry.fmt,
            family=entry.family,
            dataset=entry.dataset,
            resolution=entry.resolution,
        )

        t0 = time.monotonic()
        try:
            stdout, stderr, rc = _run_evaluate(entry)
            elapsed = time.monotonic() - t0

            if rc != 0:
                logger.error(
                    f"[{i}/{total}] Evaluation failed",
                    variant=entry.variant,
                    rc=rc,
                    elapsed_s=f"{elapsed:.1f}",
                    stderr_tail=stderr[-500:] if stderr else "",
                )
                continue

            metrics = parse_metrics(stdout, stderr)
            ap_50 = metrics.get("ap_50", "")

            row = {
                "host_time_iso": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "model_family": entry.family,
                "model_variant": entry.variant,
                "hyperparameters": entry.hyperparameters,
                "dataset": entry.dataset,
                "format": entry.fmt,
                "resolution": entry.resolution,
                "ap_50": ap_50,
            }

            append_eval_result(row, output)

            logger.info(
                f"[{i}/{total}] Done",
                variant=entry.variant,
                ap_50=ap_50 or "N/A",
                elapsed_s=f"{elapsed:.1f}",
            )

        except Exception as exc:
            logger.error(
                f"[{i}/{total}] Exception",
                variant=entry.variant,
                error=str(exc),
            )

    logger.info("Evaluation complete", results_csv=str(output))


def evaluate_main(argv: list[str] | None = None) -> int:
    configure_logging()
    typer_install_exception_hook()
    args = [] if argv is None else argv
    try:
        eval_app(args=args)
    except typer.Exit as e:
        return int(e.exit_code) if e.exit_code is not None else 0
    return 0
