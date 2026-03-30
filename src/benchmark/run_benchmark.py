"""CLI entry point for the benchmark package."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
import os
import re
import shlex
import time

import typer

from .paths import CSV_PATH, N6_WORKDIR, POWER_MEASURE_CSV_PATH
from .utils.logutil import (
    configure_logging,
    typer_install_exception_hook,
    log_benchmark_start,
    log_model_start,
    log_model_done,
    log_model_skip,
    log_model_fail,
)
from .core.models import load_models
from .io.parsing import parse_metrics
from .io.results import append_result, load_completed
from .utils.logutil import get_logger
from .execution.power_serial import (
    get_power_session_effective_port,
    start_power_session,
    stop_power_session,
)
from .execution import workflow
from .execution.workflow import _get_st_ai_output_dir, get_stedgeai_version

APP_CONFIG_PATH = (
    Path(os.environ["STEDGEAI_CORE_DIR"])
    / "Projects"
    / "STM32N6570-DK"
    / "Applications"
    / "NPU_Validation"
    / "Core"
    / "Inc"
    / "app_config.h"
)

_VALID_MODES = frozenset({"underdrive", "nominal", "overdrive", "override"})


def _ensure_no_ovd_clk400_commented(text: str) -> str:
    """Comment out #define NO_OVD_CLK400 (real nominal: 600 MHz no-overdrive path)."""
    if re.search(r"^\s*//\s*#define\s+NO_OVD_CLK400\b", text, flags=re.MULTILINE):
        return text
    pat = re.compile(r"^(\s*)(#define\s+NO_OVD_CLK400\b.*)$", re.MULTILINE)
    new, n = pat.subn(r"\1// \2", text, count=1)
    if n == 0:
        raise RuntimeError(
            f"Failed to comment NO_OVD_CLK400 in {APP_CONFIG_PATH}: active define not found"
        )
    return new


def _ensure_no_ovd_clk400_active(text: str) -> str:
    """Ensure #define NO_OVD_CLK400 is active (underdrive: 400 MHz when USE_OVERDRIVE is 0)."""
    for line in text.splitlines():
        s = line.lstrip()
        if re.match(r"#define\s+NO_OVD_CLK400\b", s):
            return text
    pat = re.compile(r"^(\s*)//\s*(#define\s+NO_OVD_CLK400\b.*)$", re.MULTILINE)
    new, n = pat.subn(r"\1\2", text, count=1)
    if n:
        return new
    raise RuntimeError(
        f"Failed to restore NO_OVD_CLK400 in {APP_CONFIG_PATH}: commented define not found"
    )


def _apply_benchmark_mode(mode: str) -> None:
    """Patch app_config.h for benchmark mode (USE_OVERDRIVE and NO_OVD_CLK400)."""
    m = mode.strip().lower()
    if m not in _VALID_MODES:
        raise ValueError(
            f"Invalid benchmark mode {mode!r}; expected one of {sorted(_VALID_MODES)}"
        )
    normalized = "overdrive" if m == "override" else m
    use_overdrive = "1" if normalized == "overdrive" else "0"
    if not APP_CONFIG_PATH.exists():
        raise FileNotFoundError(f"app_config.h not found: {APP_CONFIG_PATH}")

    text = APP_CONFIG_PATH.read_text(encoding="utf-8")
    pattern = r"(^\s*#define\s+USE_OVERDRIVE\s+)\d+"
    updated, count = re.subn(pattern, rf"\g<1>{use_overdrive}", text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(
            f"Failed to patch USE_OVERDRIVE in {APP_CONFIG_PATH}: define not found"
        )
    if normalized == "nominal":
        updated = _ensure_no_ovd_clk400_commented(updated)
    elif normalized == "underdrive":
        updated = _ensure_no_ovd_clk400_active(updated)

    if updated != text:
        APP_CONFIG_PATH.write_text(updated, encoding="utf-8")


bench_app = typer.Typer(
    help="Benchmark OD models on STM32N6570-DK",
    add_completion=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)


@bench_app.callback()
def run_benchmark(
    ctx: typer.Context,
    filter_substr: str = typer.Option(
        "",
        "--filter",
        help="Only run variants whose name contains this string",
    ),
    power_serial: str | None = typer.Option(
        None,
        "--power-serial",
        help="Serial port for INA228 (e.g. /dev/ttyUSB0); auto-detect if omitted",
    ),
    power_baud: int = typer.Option(
        921600,
        "--power-baud",
        help="Baud rate for power measurement serial",
    ),
    validation_count: int = typer.Option(
        50,
        "--validation-count",
        help="Number of inference runs for validation",
    ),
    mode: str = typer.Option(
        "all",
        "--mode",
        help="underdrive | nominal | overdrive | override | all (patch app_config.h)",
    ),
) -> None:
    if getattr(ctx, "resilient_parsing", False):
        return

    configure_logging()
    typer_install_exception_hook()

    N6_WORKDIR.mkdir(parents=True, exist_ok=True)

    if mode == "all":
        _run_all_modes(filter_substr, power_serial, power_baud, validation_count)
    else:
        _run_single_mode(mode, filter_substr, power_serial, power_baud, validation_count)


def _run_all_modes(
    filter_substr: str,
    power_serial: str | None,
    power_baud: int,
    validation_count: int,
) -> None:
    """Run benchmark in underdrive mode, pause, then overdrive mode."""
    _run_single_mode("underdrive", filter_substr, power_serial, power_baud, validation_count)
    get_logger("benchmark").info("Pausing for 5.0 seconds between modes...")
    time.sleep(5.0)
    _run_single_mode("overdrive", filter_substr, power_serial, power_baud, validation_count)


def _run_single_mode(
    mode: str,
    filter_substr: str,
    power_serial: str | None,
    power_baud: int,
    validation_count: int,
) -> None:
    """Run benchmark for a single mode."""
    _apply_benchmark_mode(mode)

    power_running = start_power_session(power_serial, power_baud)
    stedgeai_version = get_stedgeai_version()

    entries = load_models()
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))

    if filter_substr:
        entries = [e for e in entries if filter_substr in e.variant]

    completed = load_completed()
    total = len(entries)

    m = mode.strip().lower()
    mode_out = "overdrive" if m == "override" else mode
    log_benchmark_start(
        total=total,
        completed=len(completed),
        filter=filter_substr or None,
        mode=mode_out,
        validation_count=validation_count,
        stedgeai_version=stedgeai_version,
        power_active=power_running,
        power_port=get_power_session_effective_port(),
    )

    try:
        _run_benchmark_loop(
            entries,
            total,
            completed,
            power_running,
            validation_count,
            stedgeai_version,
        )
    finally:
        stop_power_session()


def _run_benchmark_loop(
    entries,
    total,
    completed,
    power_running: bool,
    validation_count: int,
    stedgeai_version: str,
):
    for i, entry in enumerate(entries, 1):
        key = (entry.variant, entry.fmt)
        tag = f"[{i}/{total}]"

        if key in completed:
            log_model_skip(i, total, entry.variant, "already in CSV")
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
            res = workflow.run_benchmark(entry, validation_count)

            if res.failed_step:
                log_model_fail(
                    i,
                    total,
                    entry.variant,
                    res.failed_step,
                    res.combined_stderr[-500:] if res.combined_stderr else "no stderr",
                )
                continue

            metrics = parse_metrics(res.combined_stdout, res.combined_stderr)

            missing = [
                k for k, v in {
                    "inference_time_ms": metrics.get("inference_time_ms"),
                    "inf_per_sec": metrics.get("inf_per_sec"),
                    "ap_50": metrics.get("ap_50"),
                    "pm_avg_inf_mW": res.pm_avg_inf_mW,
                }.items() if not v
            ]

            row = {
                "host_time_iso": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "stedgeai_version": stedgeai_version,
                "model_family": entry.family,
                "model_variant": entry.variant,
                "hyperparameters": entry.hyperparameters,
                "dataset": entry.dataset,
                "format": entry.fmt,
                "resolution": entry.resolution,
                "internal_ram_kib": metrics.get("internal_ram_kib", ""),
                "external_ram_kib": metrics.get("external_ram_kib", ""),
                "weights_flash_kib": metrics.get("weights_flash_kib", ""),
                "input_buffer_kib": metrics.get("input_buffer_kib", ""),
                "output_buffer_kib": metrics.get("output_buffer_kib", ""),
                "inference_time_ms": metrics.get("inference_time_ms", ""),
                "inf_per_sec": metrics.get("inf_per_sec", ""),
                "ap_50": metrics.get("ap_50", ""),
                "pm_avg_inf_mW": (
                    f"{res.pm_avg_inf_mW:.3f}"
                    if res.pm_avg_inf_mW is not None
                    else ""
                ),
                "pm_avg_idle_mW": (
                    f"{res.pm_avg_idle_mW:.3f}"
                    if res.pm_avg_idle_mW is not None
                    else ""
                ),
                "pm_avg_delta_mW": (
                    f"{res.pm_avg_delta_mW:.3f}"
                    if res.pm_avg_delta_mW is not None
                    else ""
                ),
                "pm_avg_inf_ms": (
                    f"{res.pm_avg_inf_ms:.3f}"
                    if res.pm_avg_inf_ms is not None
                    else ""
                ),
                "pm_avg_idle_ms": (
                    f"{res.pm_avg_idle_ms:.3f}"
                    if res.pm_avg_idle_ms is not None
                    else ""
                ),
                "pm_avg_inf_mJ": (
                    f"{res.pm_avg_inf_mJ:.3f}"
                    if res.pm_avg_inf_mJ is not None
                    else ""
                ),
                "pm_avg_idle_mJ": (
                    f"{res.pm_avg_idle_mJ:.3f}"
                    if res.pm_avg_idle_mJ is not None
                    else ""
                ),
            }

            append_result(row)

            log_model_done(
                i,
                total,
                entry.variant,
                ap_50=metrics.get("ap_50"),
                inference_ms=metrics.get("inference_time_ms"),
                pm_inf_mW=res.pm_avg_inf_mW,
                pm_delta_mW=res.pm_avg_delta_mW,
                missing=missing or None,
            )

        except Exception as exc:
            log_model_fail(i, total, entry.variant, "exception", str(exc))

    get_logger("benchmark").info(
        "Benchmark complete",
        results_csv=str(CSV_PATH),
        power_csv=str(POWER_MEASURE_CSV_PATH) if power_running else None,
    )


def main() -> None:
    configure_logging()
    typer_install_exception_hook()
    bench_app()


if __name__ == "__main__":
    main()
