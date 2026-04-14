#!/usr/bin/env python3
"""Estimate battery life from benchmark power measurements.

Assumes a continuous object-detection pipeline: idle → inference → idle → …
Inference duration and powers come from the benchmark CSV; the gap between
frames (idle segment) is a fixed application value (default 1 ms), not the
longer idle windows used during power measurement.

From repo root:
    ./project.py estimate-battery --mah 500
    ./project.py estimate-battery --mah 1000 --mode underdrive
    ./project.py estimate-battery --mah 500 --table-format ai
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.benchmark.paths import benchmark_paths_for_mode
from src.benchmark.utils.table_format import (
    ai_markdown_table,
    normalize_table_format,
)

_MODES = ("underdrive", "nominal", "overdrive")

_REQUIRED_COLS = (
    "model_variant",
    "format",
    "resolution",
    "pm_avg_inf_mW",
    "pm_avg_idle_mW",
    "pm_avg_inf_ms",
)

DEFAULT_VOLTAGE = 5.0
DEFAULT_IDLE_MS = 1.0


def _estimate(
    df: pd.DataFrame,
    battery_mah: float,
    voltage: float,
    *,
    idle_ms: float,
) -> pd.DataFrame:
    """Return a DataFrame with battery-life estimates appended."""
    out = df[list(_REQUIRED_COLS)].copy()

    inf_mw = pd.to_numeric(out["pm_avg_inf_mW"], errors="coerce")
    idle_mw = pd.to_numeric(out["pm_avg_idle_mW"], errors="coerce")
    inf_ms = pd.to_numeric(out["pm_avg_inf_ms"], errors="coerce")
    idle_ms_series = float(idle_ms)

    cycle_ms = inf_ms + idle_ms_series
    avg_mw = (inf_mw * inf_ms + idle_mw * idle_ms) / cycle_ms
    duty_pct = inf_ms / cycle_ms * 100.0
    battery_mwh = battery_mah * voltage
    life_h = battery_mwh / avg_mw

    out["duty_cycle_pct"] = duty_pct
    out["avg_power_mW"] = avg_mw
    out["battery_life_h"] = life_h

    return out.sort_values("battery_life_h", ascending=False).reset_index(drop=True)


def _render_table(
    estimates: pd.DataFrame,
    *,
    mode: str,
    battery_mah: float,
    voltage: float,
    idle_ms: float,
) -> Table:
    title = (
        f"Battery Life Estimates — {mode} mode "
        f"| {battery_mah:.0f} mAh @ {voltage:.2f} V "
        f"({battery_mah * voltage:.0f} mWh) "
        f"| inter-frame idle {idle_ms:g} ms"
    )
    table = Table(title=title, show_lines=False, header_style="bold cyan")

    table.add_column("#", justify="right", style="dim")
    table.add_column("Model Variant", min_width=20)
    table.add_column("fmt", justify="center")
    table.add_column("res", justify="right")
    table.add_column("Idle mW", justify="right")
    table.add_column("Inf mW", justify="right")
    table.add_column("Avg mW", justify="right")
    table.add_column("Duty %", justify="right")
    table.add_column("Life (h)", justify="right", style="bold green")
    table.add_column("Life (d+h)", justify="right", style="green")

    for i, row in estimates.iterrows():
        life_h = row["battery_life_h"]
        days = int(life_h // 24)
        rem_h = life_h % 24
        dhm = f"{days}d {rem_h:.1f}h" if days else f"{rem_h:.1f}h"

        table.add_row(
            str(int(i) + 1),
            str(row["model_variant"]),
            str(row["format"]),
            str(int(row["resolution"])),
            f"{row['pm_avg_idle_mW']:.1f}",
            f"{row['pm_avg_inf_mW']:.1f}",
            f"{row['avg_power_mW']:.1f}",
            f"{row['duty_cycle_pct']:.1f}",
            f"{life_h:.2f}",
            dhm,
        )

    return table


def _render_ai_table(
    estimates: pd.DataFrame,
    *,
    mode: str,
    battery_mah: float,
    voltage: float,
    idle_ms: float,
) -> str:
    title = (
        f"Battery Life Estimates — {mode} mode "
        f"| {battery_mah:.0f} mAh @ {voltage:.2f} V "
        f"({battery_mah * voltage:.0f} mWh) "
        f"| inter-frame idle {idle_ms:g} ms"
    )
    headers = [
        "#",
        "Model Variant",
        "fmt",
        "res",
        "Idle_mW",
        "Inf_mW",
        "Avg_mW",
        "Duty_pct",
        "Life_h",
        "Life_d+h",
    ]
    rows: list[list[str]] = []
    for i, row in estimates.iterrows():
        life_h = row["battery_life_h"]
        days = int(life_h // 24)
        rem_h = life_h % 24
        dhm = f"{days}d {rem_h:.1f}h" if days else f"{rem_h:.1f}h"
        rows.append(
            [
                str(int(i) + 1),
                str(row["model_variant"]),
                str(row["format"]),
                str(int(row["resolution"])),
                f"{row['pm_avg_idle_mW']:.1f}",
                f"{row['pm_avg_inf_mW']:.1f}",
                f"{row['avg_power_mW']:.1f}",
                f"{row['duty_cycle_pct']:.1f}",
                f"{life_h:.2f}",
                dhm,
            ]
        )
    return title + "\n\n" + ai_markdown_table(headers, rows).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate battery life assuming continuous idle↔inference alternation."
        ),
    )
    parser.add_argument(
        "--mah",
        type=float,
        required=True,
        help="Battery capacity in milliamp-hours.",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=DEFAULT_VOLTAGE,
        help=f"Supply voltage in volts for mAh→mWh conversion (default: {DEFAULT_VOLTAGE} V — matches benchmarking).",
    )
    parser.add_argument(
        "--mode",
        choices=list(_MODES),
        default="nominal",
        help="Benchmark mode whose measurements to use (default: nominal).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter to model variants matching this substring.",
    )
    parser.add_argument(
        "--idle-ms",
        type=float,
        default=DEFAULT_IDLE_MS,
        help=(
            "Assumed idle time between inferences in ms (continuous pipeline; "
            f"default: {DEFAULT_IDLE_MS})."
        ),
    )
    parser.add_argument(
        "--table-format",
        type=str,
        default="rich",
        help="Tabular output: 'rich' (terminal) or 'ai' (compact Markdown, smaller context).",
    )
    args = parser.parse_args(argv)
    if args.idle_ms <= 0:
        print("Error: --idle-ms must be positive.", file=sys.stderr)
        return 1

    try:
        tf = normalize_table_format(args.table_format)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    csv_path = benchmark_paths_for_mode(args.mode).csv_path
    if not csv_path.is_file():
        print(f"Error: benchmark CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"Error: missing columns in {csv_path}: {missing}", file=sys.stderr)
        return 1

    if args.model:
        mask = df["model_variant"].str.contains(args.model, case=False, na=False)
        df = df[mask]
        if df.empty:
            print(f"No model variants match {args.model!r}", file=sys.stderr)
            return 1

    estimates = _estimate(df, args.mah, args.voltage, idle_ms=args.idle_ms)
    if tf == "ai":
        print(
            _render_ai_table(
                estimates,
                mode=args.mode,
                battery_mah=args.mah,
                voltage=args.voltage,
                idle_ms=args.idle_ms,
            )
        )
        print()
    else:
        term_width = shutil.get_terminal_size((140, 25)).columns
        width = max(term_width, 140)
        console = Console(width=width, _environ={"COLUMNS": str(width)})
        console.print(
            _render_table(
                estimates,
                mode=args.mode,
                battery_mah=args.mah,
                voltage=args.voltage,
                idle_ms=args.idle_ms,
            )
        )
        console.print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
