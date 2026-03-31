"""Model selection scoring for STM32N6570-DK edge AI deployment.

Implements the multi-criteria scoring framework defined in
docs/model-selection-prompt.md — loads benchmark CSVs, applies
hardware constraint gates, computes derived memory/efficiency
metrics, scores candidates via a weighted composite, and prints
a full ranked report with per-family analysis.

From repo root:
  python src/benchmark/run_model_selection.py [flags]
  python project.py select-model [flags]
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

NPU_RAM_ACTIVATION_KIB = 4 * 448 # npuRAM3–6
CPU_RAM_ACTIVATION_KIB = 512     # cpuRAM2
MAX_LATENCY_MS = 66.666666667    # 15 FPS ceiling
MAX_WEIGHTS_FLASH_KIB = 10240    # 10 MiB octoFlash (for glass deployment, altough 60 MiB is given on the STM32N6570-DK)

# Quantisation formats dropped before gating and scoring.
SELECTION_EXCLUDED_FORMATS: frozenset[str] = frozenset({"W4A8"})

# Model families dropped before gating and scoring (by ``model_family`` prefix).
SELECTION_EXCLUDED_FAMILY_PREFIXES: tuple[str, ...] = ("ssdlite_",)

# ── Default scoring weights ──────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_acc": 0.40,
    "w_energy": 0.30,
    "w_eff": 0.20,
    "w_lat": 0.10,
}

WEIGHT_RATIONALE: dict[str, str] = {
    "w_acc": (
        "Pre-trained AP signals feature-extraction capacity; "
        "critical for fine-tuning to small-object tasks (hand/tool detection)."
    ),
    "w_energy": (
        "Battery life matters for wearable but secondary to detection "
        "reliability in safety-critical applications."
    ),
    "w_eff": (
        "Per-resolution-tier efficiency avoids systematic low-resolution "
        "bias; measures resource utilisation within each resolution class."
    ),
    "w_lat": (
        "Beyond 15 FPS gate, additional latency margin is a minor "
        "differentiator."
    ),
}

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


# ── Normalisation helpers ────────────────────────────────────────────────────


def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def _minmax_inv(s: pd.Series) -> pd.Series:
    """Min-max for lower-is-better metrics: normalise 1/x."""
    inv = 1.0 / s.replace(0, float("nan"))
    return _minmax(inv)


# ── Composite scoring ────────────────────────────────────────────────────────


def score_candidates(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Compute composite scores. Returns df sorted descending by score."""
    out = df.copy()

    # Accuracy — normalised within each dataset cohort
    out["s_acc"] = 0.0
    for cohort in out["dataset"].unique():
        mask = out["dataset"] == cohort
        if mask.sum() < 2:
            out.loc[mask, "s_acc"] = 0.5
        else:
            out.loc[mask, "s_acc"] = _minmax(out.loc[mask, "ap_50"])

    # Energy — lower is better
    has_e = out["pm_avg_inf_mJ"].notna() & (out["pm_avg_inf_mJ"] > 0)
    out["s_energy"] = 0.5
    if has_e.sum() >= 2:
        out.loc[has_e, "s_energy"] = _minmax_inv(out.loc[has_e, "pm_avg_inf_mJ"])
        if (~has_e).any():
            out.loc[~has_e, "s_energy"] = out.loc[has_e, "s_energy"].median()

    # Efficiency curve — geometric mean of AP/ms and AP/mJ,
    # normalised within each resolution tier so that higher-resolution
    # models are not systematically penalised by their inherently
    # larger compute cost.
    has_eff = out["efficiency_geo"].notna() & (out["efficiency_geo"] > 0)
    out["s_eff"] = 0.0
    if has_eff.sum() >= 2:
        for res in out["resolution"].unique():
            tier = has_eff & (out["resolution"] == res)
            if tier.sum() >= 2:
                out.loc[tier, "s_eff"] = _minmax(out.loc[tier, "efficiency_geo"])
            elif tier.sum() == 1:
                out.loc[tier, "s_eff"] = 0.5
        no_eff_rows = (~has_eff) & out["acc_per_ms"].notna()
        if no_eff_rows.sum() >= 2:
            out.loc[no_eff_rows, "s_eff"] = _minmax(out.loc[no_eff_rows, "acc_per_ms"])
        elif no_eff_rows.sum() == 1:
            out.loc[no_eff_rows, "s_eff"] = 0.5
    else:
        out["s_eff"] = _minmax(out["acc_per_ms"])

    # Latency margin — lower is better
    out["s_lat"] = _minmax_inv(out["inference_time_ms"])

    # Composite (activation placement flags — see _memory_flags)
    out["score"] = (
        weights["w_acc"] * out["s_acc"]
        + weights["w_energy"] * out["s_energy"]
        + weights["w_eff"] * out["s_eff"]
        + weights["w_lat"] * out["s_lat"]
    )

    return out.sort_values("score", ascending=False).reset_index(drop=True)


# ── Family-level aggregation ──────────────────────────────────────────────────

_FAMILY_KEY = ["model_family", "hyperparameters"]


def aggregate_families(scored: pd.DataFrame) -> pd.DataFrame:
    """Pick the best-scoring variant per (model_family, hyperparameters).

    Returns one row per family, sorted by score descending, with extra
    columns: n_variants, res_min, res_max, ap_max.
    """
    best_idx = scored.groupby(_FAMILY_KEY)["score"].idxmax()
    families = scored.loc[best_idx].copy()

    agg = scored.groupby(_FAMILY_KEY).agg(
        n_variants=("score", "size"),
        res_min=("resolution", "min"),
        res_max=("resolution", "max"),
        ap_max=("ap_50", "max"),
    )
    families = families.set_index(_FAMILY_KEY).join(agg).reset_index()

    for col in ("n_variants", "res_min", "res_max"):
        families[col] = families[col].astype(int)

    return families.sort_values("score", ascending=False).reset_index(drop=True)


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
        "Res Range",
        "Min–max input resolution (px) across variants in the family.",
    )
    legend.add_row("fmt", "Quantisation format (W4A8 excluded from the scoring pool).")
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
        "Disposition",
        "Scored or Excluded (gate failure reasons in parentheses).",
    )
    _console.print(Padding(legend, (0, 0, 0, 2)))
    _console.print()


def print_section0(df: pd.DataFrame) -> None:
    _print_column_abbreviations_legend()
    _console.print(Rule("[bold]Section 0 — Assumptions & Substituted Values[/bold]"))
    _console.print()

    power_cols = ["pm_avg_inf_mW", "pm_avg_inf_mJ"]
    any_missing = any(df[c].isna().all() for c in power_cols if c in df.columns)
    if any_missing:
        _console.print(
            "[yellow]Power columns missing in some rows — energy terms will use "
            "median fill for affected candidates.[/yellow]"
        )
    else:
        _console.print("All benchmark columns present. No placeholders substituted.")
    _console.print()


def print_section1(
    all_df: pd.DataFrame,
    scored: pd.DataFrame,
    excluded: pd.DataFrame,
) -> None:
    _console.print(Rule("[bold]Section 1 — Data Cleaning & Anomaly Report[/bold]"))
    _console.print()

    issues = [
        (
            "Issue 1 — Fine-Tuning & Safety-Critical Context",
            "Accuracy is the highest-weighted criterion — pre-trained AP signals "
            "feature-extraction capacity critical for fine-tuning to small-object "
            "tasks (hand/tool detection). Efficiency normalised within resolution "
            "tiers to avoid systematic low-resolution bias.",
        ),
        (
            "Issue 2 — Family-Level Selection",
            "All passing variants scored independently, then aggregated per "
            "(model_family, hyperparameters). Each family represented by its "
            "best-scoring variant. Optional --min-size gate excludes resolutions "
            "too low for small-object detection before aggregation.",
        ),
        (
            "Issue 3 — Quantisation Format Selection",
            "W4A8 variants are excluded from the scoring pool. Remaining formats "
            "(e.g. Int8) are ranked on the composite score.",
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

    table = Table(
        title="Disposition Table",
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
    table.add_column("Disposition", no_wrap=True)

    def _disp_row(row: pd.Series, disposition: str) -> None:
        flag_str = _memory_flags(row)
        table.add_row(
            str(row["model_variant"]),
            str(row["format"]),
            str(int(row["resolution"])),
            _f(row["activations_without_io"]),
            _f(row["npu_spill_kib"]),
            escape(flag_str) if flag_str else "",
            disposition,
        )

    for _, row in scored.iterrows():
        _disp_row(row, "Scored")
    for _, row in excluded.iterrows():
        reasons: list[str] = []
        if row["inference_time_ms"] > MAX_LATENCY_MS:
            reasons.append(f"lat={row['inference_time_ms']:.1f}ms")
        if row["weights_flash_kib"] > MAX_WEIGHTS_FLASH_KIB:
            reasons.append(f"flash={row['weights_flash_kib']:.0f}KiB")
        _disp_row(row, f"Excluded ({', '.join(reasons)})")

    _console.print(table)
    _console.print()
    _console.print(
        f"  Total rows: {len(all_df)} | Scored: {len(scored)} | "
        f"Excluded: {len(excluded)}"
    )
    _console.print()


def print_section2(weights: dict[str, float], option_label: str) -> None:
    _console.print(Rule("[bold]Section 2 — Criteria Framework Table[/bold]"))
    _console.print()
    _console.print(f"  Scoring option: [bold]{escape(option_label)}[/bold]")
    _console.print()

    table = Table(show_header=True, header_style="bold", show_lines=False)
    for c in ["Criterion", "Tier", "Metric", "Weight", "Rationale"]:
        table.add_column(c, no_wrap=c == "Criterion")

    table.add_row(
        "Real-time latency",
        "🔴 Hard",
        f"inference_time_ms ≤ {MAX_LATENCY_MS}",
        "Gate",
        "Non-negotiable 15 FPS floor",
    )
    table.add_row(
        "Weights flash",
        "🔴 Hard",
        f"weights_flash_kib ≤ {MAX_WEIGHTS_FLASH_KIB}",
        "Gate",
        "octoFlash capacity",
    )

    criteria = [
        ("w_acc", "Architecture capacity", "🟡 Primary", "norm_cohort(ap_50)"),
        ("w_energy", "Energy per inference", "🟢 Secondary", "norm(1/pm_avg_inf_mJ)"),
        (
            "w_eff",
            "Efficiency curve",
            "🟡 Primary",
            "norm_per_res_tier(geomean(AP/ms, AP/mJ))",
        ),
        ("w_lat", "Latency margin", "🟢 Secondary", "norm(1/inference_time_ms)"),
    ]
    for key, name, tier, metric in criteria:
        table.add_row(name, tier, metric, f"{weights[key]:.2f}", WEIGHT_RATIONALE[key])

    _console.print(table)
    total = sum(weights.values())
    check = " ✓" if abs(total - 1.0) < 1e-6 else " [red]≠ 1.0[/red]"
    _console.print(f"\n  Weight sum: {total:.2f}{check}")
    _console.print()


def print_section3(
    families: pd.DataFrame,
    scored: pd.DataFrame,
    weights: dict[str, float],
) -> None:
    _console.print(Rule("[bold]Section 3 — Family Ranking[/bold]"))
    _console.print()
    _console.print(
        f"  [dim]score = {weights['w_acc']:.2f}·acc + {weights['w_energy']:.2f}·energy"
        f" + {weights['w_eff']:.2f}·eff + {weights['w_lat']:.2f}·lat[/dim]"
    )
    _console.print(
        "  [dim]Each family represented by its best-scoring variant.[/dim]"
    )
    _console.print()

    # ── Family ranking table ──────────────────────────────────────────────

    table = Table(
        title="Family Ranking",
        show_header=True,
        header_style="bold",
        show_lines=False,
    )
    table.add_column("#", style="bold", min_width=2, justify="right")
    table.add_column("Family", no_wrap=True)
    table.add_column("Best Variant", no_wrap=True)
    table.add_column("res", min_width=5, justify="center")
    table.add_column("AP", min_width=5, justify="right")
    table.add_column("ms", min_width=5, justify="right")
    table.add_column("mJ", min_width=5, justify="right")
    table.add_column("Score", style="bold cyan", min_width=6, justify="right")
    table.add_column("count", min_width=3, justify="right")
    table.add_column("Res Range", min_width=7, justify="center")
    table.add_column("Flags", no_wrap=True)

    for rank, (_, row) in enumerate(families.iterrows(), 1):
        hyp_label = f" ({row['hyperparameters']})" if row["hyperparameters"] else ""
        family_label = f"{row['model_family']}{hyp_label}"
        score_str = f"{row['score']:.4f}"
        if rank <= 3:
            score_str = f"[bold]{score_str}[/bold]"
        res_min = int(row["res_min"])
        res_max = int(row["res_max"])
        res_range = f"{res_min}–{res_max}" if res_min != res_max else str(res_min)
        mem_flag = _flags(row)
        table.add_row(
            str(rank),
            family_label,
            str(row["model_variant"]),
            str(int(row["resolution"])),
            _f(row["ap_50"]),
            _f(row["inference_time_ms"]),
            _f(row.get("pm_avg_inf_mJ", float("nan"))),
            score_str,
            str(int(row["n_variants"])),
            res_range,
            escape(mem_flag) if mem_flag else "",
        )

    _console.print(table)
    _console.print()

    # ── Per-family variant details ────────────────────────────────────────

    _console.print(Rule("[bold]Per-Family Variant Details[/bold]", style="dim"))
    _console.print()

    variant_groups = {
        key: grp.sort_values("score", ascending=False)
        for key, grp in scored.groupby(_FAMILY_KEY)
    }

    for _, fam_row in families.iterrows():
        key = (fam_row["model_family"], fam_row["hyperparameters"])
        hyp_label = f" ({key[1]})" if key[1] else ""
        group = variant_groups[key]

        _console.print(f"  [bold]{key[0]}{hyp_label}[/bold]")

        for i, (_, row) in enumerate(group.iterrows()):
            marker = "★" if i == 0 else " "
            flags = _flags(row)
            flag_str = f"  {escape(flags)}" if flags else ""
            subs = (
                f"acc={row['s_acc']:.2f} nrg={row['s_energy']:.2f} "
                f"eff={row['s_eff']:.2f} lat={row['s_lat']:.2f}"
            )
            _console.print(
                f"    {marker} {int(row['resolution'])}×{int(row['resolution'])} "
                f"{row['format']}: "
                f"score={row['score']:.4f} | "
                f"AP={_f(row['ap_50'])} | "
                f"{row['inference_time_ms']:.1f} ms | "
                f"{_f(row.get('pm_avg_inf_mJ', float('nan')))} mJ"
                f"{flag_str}"
            )
            _console.print(f"      [dim]{subs}[/dim]")
        _console.print()


# ── Main pipeline ────────────────────────────────────────────────────────────


_DATASET_LABELS: dict[str, str] = {
    "80": "COCO-80",
    "person": "COCO-Person",
}


def run_selection(
    csv_path: Path,
    weights: dict[str, float] | None = None,
    *,
    dataset_filter: str | None = None,
    min_size: int | None = None,
    output_csv: Path | None = None,
    ap_csv: Path = DEFAULT_EVAL_CSV,
    memory_csv: Path = DEFAULT_MEMORY_CSV,
) -> pd.DataFrame:
    """Run the full model-family selection pipeline and print the report.

    *dataset_filter* restricts candidates to a single dataset
    ("80" for COCO-80, "person" for COCO-Person).  When set, the
    deduplication step is skipped since only one dataset is present.

    *min_size* drops candidates whose resolution is below this value
    before family aggregation — ensuring families are only evaluated on
    variants that can resolve small objects.

    Rows whose *format* is in ``SELECTION_EXCLUDED_FORMATS`` (currently W4A8),
    or whose *model_family* starts with a prefix in
    ``SELECTION_EXCLUDED_FAMILY_PREFIXES``, are removed before constraint
    gating and scoring.
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    option_label = "multi-criteria composite (memory: [cpuRAM]/[extRAM] flags only)"

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
    df = df[
        ~fam_series.str.startswith(SELECTION_EXCLUDED_FAMILY_PREFIXES)
    ].copy()
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
    scored = score_candidates(passing, w)
    families = aggregate_families(scored)

    print_section0(df)
    print_section1(df, scored, excluded)
    print_section2(w, option_label)
    print_section3(families, scored, w)

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
            "efficiency_geo",
            "s_acc",
            "s_energy",
            "s_eff",
            "s_lat",
            "score",
            "n_variants",
            "res_min",
            "res_max",
            "ap_max",
        ]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        families[out_cols].to_csv(
            output_csv,
            index=False,
            float_format="%.4f",
            quoting=csv.QUOTE_ALL,
        )
        _console.print(f"Scored results written to [bold]{output_csv}[/bold]")

    return families


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
        help="Write scored results CSV to this path",
    ),
    w_acc: float = typer.Option(DEFAULT_WEIGHTS["w_acc"], "--w-acc"),
    w_energy: float = typer.Option(DEFAULT_WEIGHTS["w_energy"], "--w-energy"),
    w_eff: float = typer.Option(DEFAULT_WEIGHTS["w_eff"], "--w-eff"),
    w_lat: float = typer.Option(DEFAULT_WEIGHTS["w_lat"], "--w-lat"),
) -> None:
    """Score and rank benchmark candidates for STM32N6570-DK deployment."""
    if getattr(ctx, "resilient_parsing", False):
        return

    configure_logging()
    typer_install_exception_hook()

    if dataset is not None and dataset not in _DATASET_LABELS:
        _err.print(
            f"[red]Error: --dataset must be '80' or 'person', got '{dataset}'[/red]"
        )
        raise typer.Exit(2)

    weights = {
        "w_acc": w_acc,
        "w_energy": w_energy,
        "w_eff": w_eff,
        "w_lat": w_lat,
    }

    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        _err.print(
            f"[yellow]Warning: weights sum to {total:.3f}, not 1.0. "
            "Normalising.[/yellow]"
        )
        weights = {k: v / total for k, v in weights.items()}

    if not DEFAULT_CSV.is_file():
        _err.print(f"[red]Error: CSV not found: {DEFAULT_CSV}[/red]")
        raise typer.Exit(2)

    run_selection(
        DEFAULT_CSV,
        weights,
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
