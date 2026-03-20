"""CLI entry point for the benchmark package."""

import argparse

from .constants import CSV_PATH, ensure_dirs
from .models import discover_models
from .parsing import parse_metrics, _parse_network_c_info
from .results import append_result, load_completed, log_error
from .workflow import _get_st_ai_output_dir, run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Benchmark OD models on STM32N6570-DK")
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Only run variants whose name contains this string",
    )
    args = parser.parse_args()

    ensure_dirs()

    entries = discover_models()

    # Sort for deterministic order: family, variant, format
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))

    if args.filter:
        entries = [e for e in entries if args.filter in e.variant]

    completed = load_completed()
    total = len(entries)

    print(f"Discovered {total} model entries to benchmark.")
    print(f"Already completed: {len(completed)} entries in CSV.")
    print()

    for i, entry in enumerate(entries, 1):
        key = (entry.variant, entry.fmt)
        tag = f"[{i}/{total}]"

        if key in completed:
            print(f"{tag} SKIPPED: {entry.variant} ({entry.fmt}) — already in CSV")
            continue

        print(f"{tag} Running: {entry.variant} ({entry.fmt}) ...")

        try:
            res = run_evaluation(entry)

            failed = res.failed_step
            if failed:
                err_msg = f"{tag} FAILED at {failed}: {entry.variant} ({entry.fmt})\n"
                err_msg += f"  stdout: {res.combined_stdout[-1000:]}\n"
                err_msg += f"  stderr: {res.combined_stderr[-1000:]}\n"
                print(f"{tag} FAILED: {entry.variant} ({entry.fmt}) — {failed}")
                log_error(err_msg)
                continue

            metrics = parse_metrics(res.combined_stdout, res.combined_stderr)

            # Also try network_c_info.json from the generate output
            cinfo_path = _get_st_ai_output_dir() / "network_c_info.json"
            if cinfo_path.exists():
                _parse_network_c_info(cinfo_path, metrics)

            row = {
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
            }

            append_result(row)

            ap = metrics.get("ap_50", "N/A")
            inf = metrics.get("inference_time_ms", "N/A")
            print(f"{tag} DONE: ap_50={ap}, inference={inf}ms")

        except Exception as exc:
            err_msg = f"{tag} EXCEPTION: {entry.variant} ({entry.fmt}): {exc}\n"
            print(f"{tag} ERROR: {entry.variant} ({entry.fmt}) — {exc}")
            log_error(err_msg)

    print("\nBenchmark complete. Results in:", CSV_PATH)


if __name__ == "__main__":
    main()
