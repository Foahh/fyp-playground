"""CLI entry point for the benchmark package."""

import datetime
import os
import re
import shlex
import sys
from pathlib import Path

import typer

from .constants import CSV_PATH, POWER_MEASURE_CSV_PATH, ensure_dirs
from .logutil import configure_logging, typer_install_exception_hook
from .models import load_models
from .parsing import parse_metrics, _parse_network_c_info
from .results import append_result, load_completed, log_error, log_stdout
from .power_serial import (
    get_power_session_effective_port,
    start_power_session,
    stop_power_session,
)
from .workflow import _get_st_ai_output_dir, get_stedgeai_version, run_evaluation

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


def _apply_benchmark_mode(mode: str) -> None:
    """Patch app_config.h USE_OVERDRIVE according to benchmark mode."""
    normalized = "nominal" if mode == "norminal" else mode
    normalized = "overdrive" if normalized == "override" else normalized
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
        "nominal",
        "--mode",
        help="nominal | norminal | overdrive | override (patch app_config.h)",
    ),
) -> None:
    if getattr(ctx, "resilient_parsing", False):
        return

    configure_logging()
    typer_install_exception_hook()

    ensure_dirs()
    _apply_benchmark_mode(mode)

    power_running = start_power_session(power_serial, power_baud)
    stedgeai_version = get_stedgeai_version()

    entries = load_models()
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))

    if filter_substr:
        entries = [e for e in entries if filter_substr in e.variant]

    completed = load_completed()
    total = len(entries)

    cmd_line = " ".join(shlex.quote(a) for a in sys.argv)
    eff_port = get_power_session_effective_port()
    power_port_line = (
        f"\n  power_serial_open: {eff_port}"
        if eff_port
        else ""
    )
    mode_out = "overdrive" if mode == "override" else ("nominal" if mode == "norminal" else mode)
    args_block = (
        f"Command: {cmd_line}\n"
        f"  --filter: {filter_substr or '(none)'}\n"
        f"  --power-serial: {power_serial or '(auto-detect)'}\n"
        f"  --power-baud: {power_baud}\n"
        f"  --validation-count: {validation_count}\n"
        f"  --mode: {mode_out}\n"
        f"  stedgeai_version: {stedgeai_version}\n"
        f"  app_config: {APP_CONFIG_PATH}\n"
        f"  power_measurement_active: {power_running}"
        f"{power_port_line}"
    )
    header = (
        f"\n{'='*60}\n"
        f"Benchmark run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total entries: {total}  |  Already completed: {len(completed)}\n"
        f"{args_block}\n"
        f"{'='*60}"
    )
    log_stdout(header)

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
            msg = f"{tag} SKIPPED: {entry.variant} ({entry.fmt}) — already in CSV"
            log_stdout(msg)
            continue

        model_header = (
            f"\n{'-'*60}\n"
            f"{tag} START\n"
            f"  Model   : {entry.variant}\n"
            f"  Family  : {entry.family}\n"
            f"  Format  : {entry.fmt}\n"
            f"  Dataset : {entry.dataset}\n"
            f"  Res     : {entry.resolution}\n"
            f"  Path    : {entry.model_path}\n"
            f"{'-'*60}"
        )
        log_stdout(model_header)

        try:
            res = run_evaluation(entry, validation_count)

            failed = res.failed_step
            if failed:
                err_msg = f"{tag} FAILED at {failed}: {entry.variant} ({entry.fmt})\n"
                err_msg += f"  stdout: {res.combined_stdout[-1000:]}\n"
                err_msg += f"  stderr: {res.combined_stderr[-1000:]}\n"
                log_error(err_msg)
                log_stdout(f"{tag} FAILED: {entry.variant} ({entry.fmt}) — {failed}")
                continue

            metrics = parse_metrics(res.combined_stdout, res.combined_stderr)

            cinfo_path = _get_st_ai_output_dir() / "network_c_info.json"
            if cinfo_path.exists():
                _parse_network_c_info(cinfo_path, metrics)

            missing = []
            if not metrics.get("inference_time_ms"):
                missing.append("inference_time_ms")
            if not metrics.get("inf_per_sec"):
                missing.append("inf_per_sec")
            if not metrics.get("ap_50"):
                missing.append("ap_50")
            if res.pm_avg_inf_mW is None:
                missing.append("pm_avg_inf_mW")

            if missing:
                warn_msg = f"  ⚠ WARNING: Missing metrics: {', '.join(missing)}"
                log_stdout(warn_msg)

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

            ap = metrics.get("ap_50", "N/A")
            inf = metrics.get("inference_time_ms", "N/A")
            power_parts: list[str] = []
            if res.pm_avg_inf_mW is not None:
                power_parts.append(f"pm_avg_inf={res.pm_avg_inf_mW:.1f}mW")
            if res.pm_avg_idle_mW is not None:
                power_parts.append(f"pm_avg_idle={res.pm_avg_idle_mW:.1f}mW")
            if res.pm_avg_delta_mW is not None:
                power_parts.append(f"pm_avg_delta={res.pm_avg_delta_mW:.1f}mW")

            power_str = f", {', '.join(power_parts)}" if power_parts else ""
            done_msg = f"{tag} DONE: ap_50={ap}, inference={inf}ms{power_str}"
            log_stdout(done_msg)

        except Exception as exc:
            err_msg = f"{tag} EXCEPTION: {entry.variant} ({entry.fmt}): {exc}\n"
            log_error(err_msg)
            log_stdout(f"{tag} ERROR: {entry.variant} ({entry.fmt}) — {exc}")

    footer = f"\nBenchmark complete. Results in: {CSV_PATH}"
    if power_running:
        footer += f"\nINA228 log: {POWER_MEASURE_CSV_PATH}"
    log_stdout(footer)


def main() -> None:
    configure_logging()
    typer_install_exception_hook()
    bench_app()


if __name__ == "__main__":
    main()
