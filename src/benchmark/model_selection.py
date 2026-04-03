"""Model selection / filtering for STM32N6570-DK edge AI deployment.

Loads benchmark CSVs, applies format and family exclusions, hardware
constraint gates, and derived memory and efficiency-curve metrics
(``efficiency_geo``). Prints candidate models grouped by family (no
composite ranking score).

From repo root:
  ./project.py select-model [flags]
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table

from .paths import RESULTS_DIR
from .utils.logutil import configure_logging, typer_install_exception_hook

DEFAULT_EVAL_CSV = RESULTS_DIR / "evaluation_result.csv"
DEFAULT_MEMORY_CSV = RESULTS_DIR / "generate_result.csv"

# ── Hardware constants (STM32N6570-DK) ──────────────────────────────────────

NPU_RAM_ACTIVATION_KIB = 4 * 448  # npuRAM3–6
CPU_RAM_ACTIVATION_KIB = 512  # cpuRAM2
MAX_LATENCY_MS = 66  # 15 FPS ceiling
MAX_WEIGHTS_FLASH_KIB = 10240  # 10 MiB octoFlash (for glass deployment, altough 60 MiB is given on the STM32N6570-DK)

# Quantisation formats dropped before gating and listing.
SELECTION_EXCLUDED_FORMATS: frozenset[str] = frozenset({"W4A8"})

# Model families dropped before gating and listing (by ``model_family`` prefix).
SELECTION_EXCLUDED_FAMILY_PREFIXES: tuple[str, ...] = ("ssdlite_",)

DEFAULT_CSV = RESULTS_DIR / "benchmark_underdrive" / "benchmark_results.csv"

_MIN_WIDTH = 160
_MIN_HEIGHT = 50
_console = Console(width=_MIN_WIDTH, height=_MIN_HEIGHT)

# ── Data loading ─────────────────────────────────────────────────────────────

_BENCH_NUMERIC_COLS = [
    "resolution",
    "inference_time_ms",
    "inf_per_sec",
    "ap_50",
    "pm_avg_inf_mW",
    "pm_avg_inf_ms",
    "pm_avg_inf_mJ",
]

_MEMORY_NUMERIC_COLS = [
    "resolution",
    "internal_ram_kib",
    "external_ram_kib",
    "weights_flash_kib",
    "input_buffer_kib",
    "output_buffer_kib",
]


_EVAL_JOIN_KEY = [
    "model_family",
    "model_variant",
    "hyperparameters",
    "dataset",
    "format",
    "resolution",
]


def _load_csv_with_numeric(path: Path, numeric_cols: list[str]) -> pd.DataFrame:
    """Load CSV and convert specified columns to numeric."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_benchmark(path: Path) -> pd.DataFrame:
    return _load_csv_with_numeric(path, _BENCH_NUMERIC_COLS)


def load_memory_results(path: Path) -> pd.DataFrame:
    return _load_csv_with_numeric(path, _MEMORY_NUMERIC_COLS)


def load_ap_results(path: Path) -> pd.DataFrame:
    """Load the host-side AP evaluation CSV."""
    return _load_csv_with_numeric(path, ["ap_50", "resolution"])


def merge_benchmark_with_ap(
    bench: pd.DataFrame,
    ap_csv: Path,
) -> pd.DataFrame:
    """Join ap_50 from the host-side AP CSV into the benchmark DataFrame."""
    if not ap_csv.is_file():
        _console.print(
            f"[yellow]Warning: AP results CSV not found: {ap_csv} — "
            "ap_50 will be NaN for all rows.[/yellow]"
        )
        bench["ap_50"] = float("nan")
        return bench

    ev = load_ap_results(ap_csv)
    ev_dedup = ev.drop_duplicates(subset=_EVAL_JOIN_KEY, keep="last")
    merged = bench.merge(
        ev_dedup[_EVAL_JOIN_KEY + ["ap_50"]],
        on=_EVAL_JOIN_KEY,
        how="left",
    )
    n_missing = merged["ap_50"].isna().sum()
    if n_missing:
        _console.print(
            f"[yellow]Warning: {n_missing} benchmark row(s) have no matching "
            f"AP result — ap_50 will be NaN.[/yellow]"
        )
    return merged


def merge_benchmark_with_memory(
    bench: pd.DataFrame,
    memory_csv: Path,
) -> pd.DataFrame:
    """Join generation-time memory columns into benchmark rows."""
    mem_cols = [
        "internal_ram_kib",
        "external_ram_kib",
        "weights_flash_kib",
        "input_buffer_kib",
        "output_buffer_kib",
    ]
    if not memory_csv.is_file():
        _console.print(
            f"[yellow]Warning: memory CSV not found: {memory_csv} — "
            "memory-derived metrics will be NaN.[/yellow]"
        )
        for c in mem_cols:
            bench[c] = float("nan")
        return bench

    mem = load_memory_results(memory_csv)
    mem_dedup = mem.drop_duplicates(subset=_EVAL_JOIN_KEY, keep="last")
    merged = bench.merge(
        mem_dedup[_EVAL_JOIN_KEY + mem_cols],
        on=_EVAL_JOIN_KEY,
        how="left",
    )
    n_missing = merged["weights_flash_kib"].isna().sum()
    if n_missing:
        _console.print(
            f"[yellow]Warning: {n_missing} benchmark row(s) have no matching "
            "memory result — memory-derived metrics will be NaN.[/yellow]"
        )
    return merged


# ── Derived metrics ──────────────────────────────────────────────────────────


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    total_act = out["internal_ram_kib"] + out["external_ram_kib"]
    io_buf = out["input_buffer_kib"] + out["output_buffer_kib"]
    out["activations_without_io"] = total_act - io_buf
    out["npu_spill_kib"] = (
        out["activations_without_io"] - NPU_RAM_ACTIVATION_KIB
    ).clip(lower=0)

    out["acc_per_ms"] = out["ap_50"] / out["inference_time_ms"]

    has_energy = out["pm_avg_inf_mJ"].notna() & (out["pm_avg_inf_mJ"] > 0)
    out["acc_per_mj"] = float("nan")
    out.loc[has_energy, "acc_per_mj"] = (
        out.loc[has_energy, "ap_50"] / out.loc[has_energy, "pm_avg_inf_mJ"]
    )

    valid_eff = (
        (out["acc_per_ms"] > 0) & out["acc_per_mj"].notna() & (out["acc_per_mj"] > 0)
    )
    out["efficiency_geo"] = float("nan")
    out.loc[valid_eff, "efficiency_geo"] = (
        out.loc[valid_eff, "acc_per_ms"] * out.loc[valid_eff, "acc_per_mj"]
    ).apply(math.sqrt)

    return out


# ── Constraint gating ────────────────────────────────────────────────────────


def gate_constraints(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (passing, excluded)."""
    mask = (df["inference_time_ms"] <= MAX_LATENCY_MS) & (
        df["weights_flash_kib"] <= MAX_WEIGHTS_FLASH_KIB
    )
    return df[mask].copy(), df[~mask].copy()


_FAMILY_KEY = ["model_family", "hyperparameters"]


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Stable order for reporting: family, hyperparameters, resolution, variant."""
    return df.sort_values(
        _FAMILY_KEY + ["resolution", "model_variant"],
        kind="mergesort",
    ).reset_index(drop=True)


# ── Flags helper ─────────────────────────────────────────────────────────────


def _memory_flags(row: pd.Series) -> str:
    """Placement tier: NPU-only (no tag), cpuRAM2 spill, or external RAM."""
    act = row.get("activations_without_io", float("nan"))
    spill = row.get("npu_spill_kib", float("nan"))
    if pd.isna(act) or pd.isna(spill):
        return ""
    budget_npu_cpu = NPU_RAM_ACTIVATION_KIB + CPU_RAM_ACTIVATION_KIB
    if act > budget_npu_cpu:
        return "[ExtRAM]"
    if spill > 0:
        return "[cpuRAM]"
    return ""


def _flags(row: pd.Series) -> str:
    return _memory_flags(row)


# ── Report printing ──────────────────────────────────────────────────────────


def _f(v: float, fmt: str = ".1f") -> str:
    if pd.isna(v):
        return "—"
    return f"{v:{fmt}}"


def _print_column_abbreviations_legend() -> None:
    """Short column header glossary for tables printed in this report."""
    _console.print()
    _console.print("  [bold]Column abbreviations[/bold]")
    legend_width = min(92, max(72, (_console.width or 92) - 4))
    legend = Table(
        box=None,
        show_header=True,
        header_style="bold dim",
        pad_edge=False,
        expand=False,
        width=legend_width,
    )
    legend.add_column("Abbr", style="cyan", no_wrap=True, width=14)
    legend.add_column("Meaning", overflow="fold")
    legend.add_row("AP", "Primary accuracy: AP@50 ([bold]ap_50[/bold]).")
    legend.add_row("ms", "Inference time ([bold]inference_time_ms[/bold]).")
    legend.add_row("mJ", "Energy per inference ([bold]pm_avg_inf_mJ[/bold]).")
    legend.add_row(
        "eff_geo",
        "Efficiency curve: √(AP/ms × AP/mJ) ([bold]efficiency_geo[/bold]); "
        "needs valid AP, latency, and mJ.",
    )
    legend.add_row(
        "fmt", "Quantisation format (W4A8 excluded from the candidate pool)."
    )
    legend.add_row("res", "Input resolution in pixels.")
    legend.add_row(
        "act_no_io",
        "Activation KiB without input/output buffers (Issues 4–5).",
    )
    legend.add_row(
        "spill",
        "Spill KiB past NPU activation budget ([bold]npu_spill_kib[/bold]).",
    )
    legend.add_row(
        "Reason",
        "Hardware gate failure(s): latency above ceiling and/or weights flash over budget.",
    )
    _console.print(Padding(legend, (0, 0, 0, 2)))
    _console.print()


def print_section0(df: pd.DataFrame, filter_summary: str | None = None) -> None:
    _print_column_abbreviations_legend()
    _console.print(Rule("[bold]Section 0 — Assumptions & Substituted Values[/bold]"))
    _console.print()
    if filter_summary:
        _console.print(f"  Filter summary: [bold]{escape(filter_summary)}[/bold]")
        _console.print()

    power_cols = ["pm_avg_inf_mW", "pm_avg_inf_mJ"]
    any_missing = any(df[c].isna().all() for c in power_cols if c in df.columns)
    if any_missing:
        _console.print(
            "[yellow]Power columns missing — [bold]efficiency_geo[/bold] may be NaN "
            "where mJ is unavailable.[/yellow]"
        )
    else:
        _console.print("All benchmark columns present. No placeholders substituted.")
    _console.print()


def print_section1(
    all_df: pd.DataFrame,
    candidates: pd.DataFrame,
    excluded: pd.DataFrame,
) -> None:
    _console.print(Rule("[bold]Section 1 — Data Cleaning & Anomaly Report[/bold]"))
    _console.print()

    issues = [
        (
            "Issue 1 — Fine-Tuning & Safety-Critical Context",
            "Pre-trained AP indicates feature-extraction capacity for "
            "fine-tuning to small-object tasks (hand/tool detection). "
            "Efficiency curve (eff_geo) is computed per row from AP, latency, and energy.",
        ),
        (
            "Issue 2 — Family-Level Grouping",
            "All variants that pass gates are listed in Section 3, grouped by "
            "(model_family, hyperparameters). Optional --min-size gate excludes "
            "resolutions too low for small-object detection before listing.",
        ),
        (
            "Issue 3 — Quantisation Format Selection",
            "W4A8 variants are excluded from the candidate pool. Other formats "
            "(e.g. Int8) appear if they pass hardware gates.",
        ),
        (
            "Issue 4 — Buffer Placement Correction",
            "activations_without_io = (internal + external) − (input_buf + output_buf). "
            "Drives placement flags. Models with large input buffers benefit most "
            "from firmware PSRAM placement.",
        ),
        (
            "Issue 5 — HyperRAM Spillover",
            "activations_without_io and npu_spill_kib computed for every row;\n"
            "    [cpuRAM] when npu_spill_kib > 0 (activations spill past NPU budget into cpuRAM2);\n"
            f"    [extRAM] when activations exceed all internal RAM (> {NPU_RAM_ACTIVATION_KIB + CPU_RAM_ACTIVATION_KIB} KiB).",
        ),
    ]
    for title, desc in issues:
        _console.print(f"  [bold]{title}[/bold]")
        _console.print(f"    {escape(desc)}")
        _console.print()

    if excluded.empty:
        _console.print(
            "  [dim]Disposition table: no hardware gate failures "
            "(all filtered rows are candidates).[/dim]"
        )
    else:
        table = Table(
            title="Disposition Table (excluded only)",
            show_header=True,
            header_style="bold",
            show_lines=False,
        )
        table.add_column("Model Variant", no_wrap=True)
        table.add_column("fmt", min_width=4)
        table.add_column("res", min_width=3)
        table.add_column("act_no_io", min_width=8, justify="right")
        table.add_column("spill", min_width=6, justify="right")
        table.add_column("Flags", min_width=8)
        table.add_column("Reason", no_wrap=True)

        excluded_sorted = excluded.sort_values(
            ["model_variant", "resolution", "format"], kind="mergesort"
        )
        for _, row in excluded_sorted.iterrows():
            reasons: list[str] = []
            if row["inference_time_ms"] > MAX_LATENCY_MS:
                reasons.append(f"lat={row['inference_time_ms']:.1f}ms")
            if row["weights_flash_kib"] > MAX_WEIGHTS_FLASH_KIB:
                reasons.append(f"flash={row['weights_flash_kib']:.0f}KiB")
            flag_str = _memory_flags(row)
            table.add_row(
                str(row["model_variant"]),
                str(row["format"]),
                str(int(row["resolution"])),
                _f(row["activations_without_io"]),
                _f(row["npu_spill_kib"]),
                escape(flag_str) if flag_str else "",
                ", ".join(reasons),
            )

        _console.print(table)
    _console.print()
    _console.print(
        f"  Total rows: {len(all_df)} | Candidates: {len(candidates)} | "
        f"Excluded: {len(excluded)}"
    )
    _console.print()


def print_section3(candidates: pd.DataFrame) -> None:
    _console.print(
        Rule("[bold]Section 3 — Candidate Models (grouped by family)[/bold]")
    )
    _console.print()
    _console.print(
        "  [dim]Each table lists all gated-pass variants for one "
        "(model_family, hyperparameters) group. "
        "[bold]eff_geo[/bold] = √(AP/ms × AP/mJ) when energy data is valid.[/dim]"
    )
    _console.print()

    for key, grp in candidates.groupby(_FAMILY_KEY, sort=False):
        hyp_label = f" ({key[1]})" if key[1] else ""
        grp_sorted = grp.sort_values(
            ["resolution", "model_variant", "format"], kind="mergesort"
        )

        table = Table(
            title=f"{key[0]}{hyp_label} — {len(grp_sorted)} variant(s)",
            show_header=True,
            header_style="bold",
            show_lines=False,
        )
        table.add_column("Variant", no_wrap=True)
        table.add_column("fmt", min_width=4)
        table.add_column("res", min_width=4, justify="right")
        table.add_column("AP", min_width=5, justify="right")
        table.add_column("ms", min_width=5, justify="right")
        table.add_column("mJ", min_width=5, justify="right")
        table.add_column("eff_geo", min_width=8, justify="right")
        table.add_column("act_no_io", min_width=8, justify="right")
        table.add_column("spill", min_width=6, justify="right")
        table.add_column("Flags", min_width=8)

        for _, row in grp_sorted.iterrows():
            mem_flag = _flags(row)
            table.add_row(
                str(row["model_variant"]),
                str(row["format"]),
                str(int(row["resolution"])),
                _f(row["ap_50"]),
                _f(row["inference_time_ms"]),
                _f(row.get("pm_avg_inf_mJ", float("nan"))),
                _f(row["efficiency_geo"], ".4f"),
                _f(row["activations_without_io"]),
                _f(row["npu_spill_kib"]),
                escape(mem_flag) if mem_flag else "",
            )

        _console.print(table)
        _console.print()


# ── Main pipeline ────────────────────────────────────────────────────────────


_DATASET_LABELS: dict[str, str] = {
    "80": "COCO-80",
    "person": "COCO-Person",
}


def run_selection(
    csv_path: Path,
    *,
    dataset_filter: str | None = None,
    min_size: int | None = None,
    output_csv: Path | None = None,
    ap_csv: Path = DEFAULT_EVAL_CSV,
    memory_csv: Path = DEFAULT_MEMORY_CSV,
) -> pd.DataFrame:
    """Run filtering and print candidate models grouped by family.

    *dataset_filter* restricts candidates to a single dataset
    ("80" for COCO-80, "person" for COCO-Person).  When set, the
    deduplication step is skipped since only one dataset is present.

    *min_size* drops rows whose resolution is below this value before
    gating — useful when low resolutions are unsuitable for small-object
    detection.

    Rows whose *format* is in ``SELECTION_EXCLUDED_FORMATS`` (currently W4A8),
    or whose *model_family* starts with a prefix in
    ``SELECTION_EXCLUDED_FAMILY_PREFIXES``, are removed before constraint
    gating.
    """
    option_label = "gates: latency & flash; memory flags: [cpuRAM]/[extRAM]"

    df = load_benchmark(csv_path)
    df = merge_benchmark_with_memory(df, memory_csv)
    df = merge_benchmark_with_ap(df, ap_csv)
    df = add_derived(df)

    before_fmt = len(df)
    df = df[~df["format"].isin(SELECTION_EXCLUDED_FORMATS)].copy()
    dropped_fmt = before_fmt - len(df)
    if dropped_fmt:
        shown = ", ".join(sorted(SELECTION_EXCLUDED_FORMATS))
        option_label += f" | excluded format(s) {shown} ({dropped_fmt} rows dropped)"

    before_fam = len(df)
    fam_series = df["model_family"].astype(str)
    df = df[~fam_series.str.startswith(SELECTION_EXCLUDED_FAMILY_PREFIXES)].copy()
    dropped_fam = before_fam - len(df)
    if dropped_fam:
        shown = ", ".join(SELECTION_EXCLUDED_FAMILY_PREFIXES)
        option_label += (
            f" | excluded family prefix(es) {shown} ({dropped_fam} rows dropped)"
        )

    if min_size is not None:
        before = len(df)
        df = df[df["resolution"] >= min_size].copy()
        dropped = before - len(df)
        if dropped:
            option_label += f" | min-size={min_size} ({dropped} rows dropped)"

    if dataset_filter is not None:
        ds_label = _DATASET_LABELS[dataset_filter]
        before = len(df)
        df = df[df["dataset"] == ds_label].copy()
        option_label += f" | dataset={ds_label} ({before - len(df)} rows filtered)"

    passing, excluded = gate_constraints(df)
    candidates = sort_candidates(passing)

    print_section0(df, option_label)
    print_section1(df, candidates, excluded)
    print_section3(candidates)

    if output_csv is not None:
        out_cols = [
            "model_family",
            "model_variant",
            "hyperparameters",
            "dataset",
            "format",
            "resolution",
            "inference_time_ms",
            "ap_50",
            "pm_avg_inf_mW",
            "pm_avg_inf_mJ",
            "activations_without_io",
            "npu_spill_kib",
            "weights_flash_kib",
            "acc_per_ms",
            "acc_per_mj",
            "efficiency_geo",
        ]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates[out_cols].to_csv(
            output_csv,
            index=False,
            float_format="%.4f",
            quoting=csv.QUOTE_ALL,
        )
        _console.print(f"Candidate results written to [bold]{output_csv}[/bold]")

    return candidates


# ── CLI ──────────────────────────────────────────────────────────────────────

_err = Console(stderr=True, width=_MIN_WIDTH, height=_MIN_HEIGHT)

app = typer.Typer(
    help=__doc__,
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)


@app.callback()
def _cli_entry(
    ctx: typer.Context,
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Filter to a single dataset: '80' for COCO-80, 'person' for COCO-Person.",
    ),
    min_size: int | None = typer.Option(
        None,
        "--min-size",
        help="Minimum input resolution (px). Models below this are dropped.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write candidate results CSV to this path",
    ),
) -> None:
    """List gated-pass benchmark candidates for STM32N6570-DK deployment."""
    if getattr(ctx, "resilient_parsing", False):
        return

    configure_logging()
    typer_install_exception_hook()

    if dataset is not None and dataset not in _DATASET_LABELS:
        _err.print(
            f"[red]Error: --dataset must be '80' or 'person', got '{dataset}'[/red]"
        )
        raise typer.Exit(2)

    if not DEFAULT_CSV.is_file():
        _err.print(f"[red]Error: CSV not found: {DEFAULT_CSV}[/red]")
        raise typer.Exit(2)

    run_selection(
        DEFAULT_CSV,
        dataset_filter=dataset,
        min_size=min_size,
        output_csv=output,
    )


def selection_main(argv: list[str] | None = None) -> int:
    configure_logging()
    typer_install_exception_hook()
    args = [] if argv is None else argv
    try:
        app(args=args)
    except typer.Exit as e:
        return int(e.exit_code) if e.exit_code is not None else 0
    return 0
