"""CLI entry point for the benchmark package."""

import argparse
import datetime
import shlex
import sys

from .constants import CSV_PATH, POWER_MEASURE_CSV_PATH, ensure_dirs
from .models import load_models
from .parsing import parse_metrics, _parse_network_c_info
from .results import append_result, load_completed, log_error, log_stdout
from .power_serial import (
    get_power_session_effective_port,
    start_power_session,
    stop_power_session,
)
from .workflow import _get_st_ai_output_dir, run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Benchmark OD models on STM32N6570-DK")
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Only run variants whose name contains this string",
    )
    parser.add_argument(
        "--power-serial",
        type=str,
        default=None,
        help="Serial port for INA228 power measurement (e.g., /dev/ttyUSB0); auto-detects if not specified",
    )
    parser.add_argument(
        "--power-baud",
        type=int,
        default=921600,
        help="Baud rate for power measurement serial (default: 921600)",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=50,
        help="Number of inference runs for validation (default: 50)",
    )
    args = parser.parse_args()

    ensure_dirs()

    power_running = start_power_session(args.power_serial, args.power_baud)

    entries = load_models()

    # Sort for deterministic order: family, variant, format
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))

    if args.filter:
        entries = [e for e in entries if args.filter in e.variant]

    completed = load_completed()
    total = len(entries)

    cmd_line = " ".join(shlex.quote(a) for a in sys.argv)
    eff_port = get_power_session_effective_port()
    power_port_line = (
        f"\n  power_serial_open: {eff_port}"
        if eff_port
        else ""
    )
    args_block = (
        f"Command: {cmd_line}\n"
        f"  --filter: {args.filter or '(none)'}\n"
        f"  --power-serial: {args.power_serial or '(auto-detect)'}\n"
        f"  --power-baud: {args.power_baud}\n"
        f"  --validation-count: {args.validation_count}\n"
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
    print(header)
    log_stdout(header)

    try:
        _run_benchmark_loop(entries, total, completed, power_running, args.validation_count)
    finally:
        stop_power_session()


def _run_benchmark_loop(entries, total, completed, power_running: bool, validation_count: int):
    for i, entry in enumerate(entries, 1):
        key = (entry.variant, entry.fmt)
        tag = f"[{i}/{total}]"

        if key in completed:
            msg = f"{tag} SKIPPED: {entry.variant} ({entry.fmt}) — already in CSV"
            print(msg)
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
        print(model_header)
        log_stdout(model_header)

        try:
            res = run_evaluation(entry, validation_count)

            failed = res.failed_step
            if failed:
                err_msg = f"{tag} FAILED at {failed}: {entry.variant} ({entry.fmt})\n"
                err_msg += f"  stdout: {res.combined_stdout[-1000:]}\n"
                err_msg += f"  stderr: {res.combined_stderr[-1000:]}\n"
                print(f"{tag} FAILED: {entry.variant} ({entry.fmt}) — {failed}")
                log_error(err_msg)
                log_stdout(f"{tag} FAILED: {entry.variant} ({entry.fmt}) — {failed}")
                continue

            metrics = parse_metrics(res.combined_stdout, res.combined_stderr)

            # Also try network_c_info.json from the generate output
            cinfo_path = _get_st_ai_output_dir() / "network_c_info.json"
            if cinfo_path.exists():
                _parse_network_c_info(cinfo_path, metrics)

            # Warn about missing critical metrics
            missing = []
            if not metrics.get("inference_time_ms"):
                missing.append("inference_time_ms")
            if not metrics.get("inf_per_sec"):
                missing.append("inf_per_sec")
            if not metrics.get("ap_50"):
                missing.append("ap_50")
            if res.avg_power_inf_mW is None:
                missing.append("avg_power_inf_mW")

            if missing:
                warn_msg = f"  ⚠ WARNING: Missing metrics: {', '.join(missing)}"
                print(warn_msg)
                log_stdout(warn_msg)

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
                "internal_ram_kib": metrics.get("internal_ram_kib", ""),
                "external_ram_kib": metrics.get("external_ram_kib", ""),
                "weights_flash_kib": metrics.get("weights_flash_kib", ""),
                "inference_time_ms": metrics.get("inference_time_ms", ""),
                "inf_per_sec": metrics.get("inf_per_sec", ""),
                "ap_50": metrics.get("ap_50", ""),
                "avg_power_inf_mW": (
                    f"{res.avg_power_inf_mW:.3f}"
                    if res.avg_power_inf_mW is not None
                    else ""
                ),
                "avg_power_idle_mW": (
                    f"{res.avg_power_idle_mW:.3f}"
                    if res.avg_power_idle_mW is not None
                    else ""
                ),
                "avg_power_delta_mW": (
                    f"{res.avg_power_delta_mW:.3f}"
                    if res.avg_power_delta_mW is not None
                    else ""
                ),
                "avg_power_inf_ms": (
                    f"{res.avg_power_inf_ms:.3f}"
                    if res.avg_power_inf_ms is not None
                    else ""
                ),
                "avg_energy_inf_mJ": (
                    f"{res.avg_energy_inf_mJ:.3f}"
                    if res.avg_energy_inf_mJ is not None
                    else ""
                ),
            }

            append_result(row)

            ap = metrics.get("ap_50", "N/A")
            inf = metrics.get("inference_time_ms", "N/A")
            power_parts: list[str] = []
            if res.avg_power_inf_mW is not None:
                power_parts.append(f"avg_power_inf={res.avg_power_inf_mW:.1f}mW")
            if res.avg_power_idle_mW is not None:
                power_parts.append(f"avg_power_idle={res.avg_power_idle_mW:.1f}mW")
            if res.avg_power_delta_mW is not None:
                power_parts.append(f"avg_power_delta={res.avg_power_delta_mW:.1f}mW")

            power_str = f", {', '.join(power_parts)}" if power_parts else ""
            done_msg = f"{tag} DONE: ap_50={ap}, inference={inf}ms{power_str}"
            print(done_msg)
            log_stdout(done_msg)

        except Exception as exc:
            err_msg = f"{tag} EXCEPTION: {entry.variant} ({entry.fmt}): {exc}\n"
            print(f"{tag} ERROR: {entry.variant} ({entry.fmt}) — {exc}")
            log_error(err_msg)
            log_stdout(f"{tag} ERROR: {entry.variant} ({entry.fmt}) — {exc}")

    footer = f"\nBenchmark complete. Results in: {CSV_PATH}"
    if power_running:
        footer += f"\nINA228 log: {POWER_MEASURE_CSV_PATH}"
    print(footer)
    log_stdout(footer)


if __name__ == "__main__":
    main()
