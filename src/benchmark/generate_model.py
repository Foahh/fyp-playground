"""Generate model artifacts once and store memory details for benchmark reuse."""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import typer

from .constants import GENERATE_CSV_COLUMNS
from .core.models import load_models
from .execution.workflow import generated_model_dir, get_stedgeai_version, run_generate_model
from .paths import GENERATE_LOG_PATH, GENERATE_RESULT_CSV_PATH, RESULTS_DIR
from .utils.logutil import (
    configure_logging,
    get_logger,
    log_model_done,
    log_model_fail,
    log_model_skip,
    log_model_start,
    typer_install_exception_hook,
)


def _append_generate_row(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=GENERATE_CSV_COLUMNS,
            quoting=csv.QUOTE_ALL,
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _load_completed_generate(csv_path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row.get("model_variant", ""), row.get("format", "")))
    return completed


def _to_results_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(RESULTS_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


generate_app = typer.Typer(
    help=__doc__,
    add_completion=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)


@generate_app.callback()
def generate_entry(
    ctx: typer.Context,
    filter_substr: str = typer.Option(
        "",
        "--filter",
        help="Only generate variants whose name contains this string",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate entries already present in generate_result.csv",
    ),
) -> None:
    """Generate all model artifacts and append memory stats CSV rows."""
    if getattr(ctx, "resilient_parsing", False):
        return

    configure_logging(audit_log_path=GENERATE_LOG_PATH)
    typer_install_exception_hook()

    entries = load_models()
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))
    if filter_substr:
        entries = [e for e in entries if filter_substr in e.variant]

    completed = _load_completed_generate(GENERATE_RESULT_CSV_PATH)
    total = len(entries)
    stedgeai_version = get_stedgeai_version(GENERATE_LOG_PATH)

    for i, entry in enumerate(entries, 1):
        key = (entry.variant, entry.fmt)
        if key in completed and not force:
            log_model_skip(i, total, entry.variant, "already in generate_result.csv")
            continue

        log_model_start(
            i,
            total,
            entry.variant,
            entry.fmt,
            family=entry.family,
            dataset=entry.dataset,
            resolution=entry.resolution,
        )
        try:
            metrics, rc = run_generate_model(entry, GENERATE_LOG_PATH)
            if rc != 0:
                log_model_fail(i, total, entry.variant, "generate", "stedgeai generate failed")
                continue

            row = {
                "host_time_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "stedgeai_version": stedgeai_version,
                "model_family": entry.family,
                "model_variant": entry.variant,
                "hyperparameters": entry.hyperparameters,
                "dataset": entry.dataset,
                "format": entry.fmt,
                "resolution": str(entry.resolution),
                "internal_ram_kib": metrics.get("internal_ram_kib", ""),
                "external_ram_kib": metrics.get("external_ram_kib", ""),
                "weights_flash_kib": metrics.get("weights_flash_kib", ""),
                "input_buffer_kib": metrics.get("input_buffer_kib", ""),
                "output_buffer_kib": metrics.get("output_buffer_kib", ""),
                "generated_model_dir": _to_results_relative(generated_model_dir(entry)),
            }
            _append_generate_row(GENERATE_RESULT_CSV_PATH, row)
            completed.add(key)
            log_model_done(i, total, entry.variant)
        except Exception as exc:
            log_model_fail(i, total, entry.variant, "exception", str(exc))

    get_logger("benchmark").info(
        "Generate-model complete",
        generated_csv=str(GENERATE_RESULT_CSV_PATH),
        generated_network_dir=str(generated_model_dir(entries[0]).parent) if entries else "",
    )


def generate_main(argv: list[str] | None = None) -> int:
    configure_logging(audit_log_path=GENERATE_LOG_PATH)
    typer_install_exception_hook()
    args = [] if argv is None else argv
    try:
        generate_app(args=args)
    except typer.Exit as e:
        return int(e.exit_code) if e.exit_code is not None else 0
    return 0

