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
from rich.rule import Rule
from rich.table import Table

from .paths import RESULTS_DIR
from .utils.logutil import configure_logging, typer_install_exception_hook

DEFAULT_EVAL_CSV = RESULTS_DIR / "evaluation_result.csv"
DEFAULT_MEMORY_CSV = RESULTS_DIR / "generate_result.csv"

# ── Hardware constants (STM32N6570-DK) ──────────────────────────────────────

ON_CHIP_ACTIVATION_KIB = 2304  # 512 (cpuRAM2) + 4×448 (npuRAM3–6)
MAX_LATENCY_MS = 66.666666667  # 15 FPS ceiling
MAX_WEIGHTS_FLASH_KIB = 10240  # 10 MiB octoFlash (for glass deployment, altough 60 MiB is given on the STM32N6570-DK)

# ── Architecture modernity classification ────────────────────────────────────
# 1.0 = anchor-free decoupled head
# 0.0 = legacy anchor-based

ARCHITECTURE_MODERNITY: dict[str, float] = {
    "ssdlite_mobilenetv1_pt": 0.0,
    "ssdlite_mobilenetv2_pt": 0.0,
    "ssdlite_mobilenetv3large_pt": 0.0,
    "ssdlite_mobilenetv3small_pt": 0.0,
    "st_yolodv2milli_pt": 1.0,
    "st_yolodv2tiny_pt": 1.0,
    "st_yoloxn": 1.0,
    "tinyissimoyolo_v8": 1.0,
    "yolo11n": 1.0,
    "yolo26": 1.0,
    "yolov8n": 1.0,
}

# ── Default scoring weights (Option A) ──────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_acc": 0.30,
    "w_energy": 0.10,
    "w_eff": 0.15,
    "w_lat": 0.05,
    "w_mem": 0.20,
    "w_modern": 0.20,
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
    "w_mem": (
        "Full score when activation budget fits on-chip (no spill); "
        "HyperRAM spill scored lower by spill amount."
    ),
    "w_modern": (
        "Anchor-free decoupled heads fine-tune more reliably "
        "on small custom datasets; critical for hand/tool domain."
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
    out["estimated_spill_kib"] = (
        out["activations_without_io"] - ON_CHIP_ACTIVATION_KIB
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

    out["arch_modernity"] = (
        out["model_family"].map(ARCHITECTURE_MODERNITY).fillna(0.5)
    )

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


# ── Dataset deduplication ────────────────────────────────────────────────────

_DEDUP_KEY = ["model_family", "hyperparameters", "format", "resolution"]


def deduplicate_datasets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer COCO-Person over COCO-80. Returns (kept, superseded)."""
    person_configs: set[tuple] = set()
    for _, row in df[df["dataset"] == "COCO-Person"].iterrows():
        person_configs.add(tuple(row[c] for c in _DEDUP_KEY))

    superseded_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        if row["dataset"] == "COCO-80":
            if tuple(row[c] for c in _DEDUP_KEY) in person_configs:
                superseded_mask.at[idx] = True

    return df[~superseded_mask].copy(), df[superseded_mask].copy()


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
    *,
    option_b: bool = False,
    w_head_frac: float = 0.6,
) -> pd.DataFrame:
    """Compute composite scores. Returns df sorted descending by score."""
    out = df.copy()

    # Accuracy — within-cohort normalisation (Issue 1)
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

    # Memory: full normalised score when activations fit on-chip (no spill);
    # spilling models are ranked by spill volume, strictly below on-chip.
    on_chip = out["estimated_spill_kib"] <= 0
    if option_b:
        s_head = _minmax_inv(out["activations_without_io"])
        spill_denom = 1.0 + out["estimated_spill_kib"]
        s_spill = _minmax(1.0 / spill_denom)
        out["s_mem"] = w_head_frac * s_head + (1 - w_head_frac) * s_spill
        out.loc[on_chip, "s_mem"] = 1.0
    else:
        spillers = ~on_chip
        if on_chip.all():
            out["s_mem"] = 1.0
        elif spillers.all():
            out["s_mem"] = _minmax_inv(out["estimated_spill_kib"])
        else:
            out["s_mem"] = 0.0
            out.loc[on_chip, "s_mem"] = 1.0
            sub = _minmax_inv(out.loc[spillers, "estimated_spill_kib"])
            out.loc[spillers, "s_mem"] = sub * (1.0 - 1e-9)

    # Architecture modernity (already on 0–1 scale)
    out["s_modern"] = out["arch_modernity"]

    # Composite
    out["score"] = (
        weights["w_acc"] * out["s_acc"]
        + weights["w_energy"] * out["s_energy"]
        + weights["w_eff"] * out["s_eff"]
        + weights["w_lat"] * out["s_lat"]
        + weights["w_mem"] * out["s_mem"]
        + weights["w_modern"] * out["s_modern"]
    )

    return out.sort_values("score", ascending=False).reset_index(drop=True)


# ── Flags helper ─────────────────────────────────────────────────────────────


def _flags(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("estimated_spill_kib", 0) > 0:
        parts.append("[ExtRAM]")
    if row["dataset"] == "COCO-80":
        parts.append("[80only]")
    return " ".join(parts)


# ── Report printing ──────────────────────────────────────────────────────────


def _f(v: float, fmt: str = ".1f") -> str:
    if pd.isna(v):
        return "—"
    return f"{v:{fmt}}"


def print_section0(df: pd.DataFrame) -> None:
    _console.print()
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
    superseded: pd.DataFrame,
) -> None:
    _console.print(Rule("[bold]Section 1 — Data Cleaning & Anomaly Report[/bold]"))
    _console.print()

    issues = [
        (
            "Issue 1 — Dataset Incomparability",
            "COCO-Person AP used as primary. COCO-80 rows with Person counterparts "
            "marked 'Superseded'. Accuracy normalised within each cohort separately.",
        ),
        (
            "Issue 2 — Fine-Tuning & Safety-Critical Context",
            "Accuracy weighted 0.30 (highest) — pre-trained AP signals feature-extraction "
            "capacity critical for fine-tuning to small-object tasks (hand/tool detection). "
            "Efficiency normalised within resolution tiers to avoid systematic "
            "low-resolution bias.",
        ),
        (
            "Issue 3 — Resolution Variants",
            "All passing resolutions scored independently. Best per-family identified "
            "post-scoring. Variants within 5% composite listed as alternatives. "
            "Optional --min-size gate excludes resolutions too low for small-object "
            "detection. Optional --min-ap gate excludes models with insufficient "
            "baseline accuracy for fine-tuning viability.",
        ),
        (
            "Issue 4 — Quantisation Format Selection",
            "Both Int8 and W4A8 variants scored. Int8 preferred for fine-tuning "
            "unless W4A8 shows decisive multi-dimensional advantage.",
        ),
        (
            "Issue 5 — HyperRAM Spillover",
            "activations_without_io and estimated_spill_kib computed for every row. "
            "[ExtRAM] flag added. Spill penalised via memory score + implicit "
            "latency/energy cost.",
        ),
        (
            "Issue 6 — Buffer Placement Correction",
            "activations_without_io = (internal + external) − (input_buf + output_buf). "
            "Used for memory scoring. Models with large input buffers benefit most "
            "from firmware PSRAM placement.",
        ),
    ]
    for title, desc in issues:
        _console.print(f"  [bold]{title}[/bold]")
        _console.print(f"    {desc}")
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
        flags: list[str] = []
        if row.get("estimated_spill_kib", 0) > 0:
            flags.append("[ExtRAM]")
        if disposition == "Superseded":
            flags.append("[COCO-80]")
        table.add_row(
            str(row["model_variant"]),
            str(row["format"]),
            str(int(row["resolution"])),
            _f(row["activations_without_io"]),
            _f(row["estimated_spill_kib"]),
            " ".join(flags),
            disposition,
        )

    for _, row in scored.iterrows():
        _disp_row(row, "Scored")
    for _, row in superseded.iterrows():
        _disp_row(row, "Superseded")
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
        f"Superseded: {len(superseded)} | Excluded: {len(excluded)}"
    )
    _console.print()


def print_section2(weights: dict[str, float], option_label: str) -> None:
    _console.print(Rule("[bold]Section 2 — Criteria Framework Table[/bold]"))
    _console.print()
    _console.print(f"  Scoring option: [bold]{option_label}[/bold]")
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
        (
            "w_mem",
            "Activation memory",
            "🟡 Primary",
            "1.0 if no spill else norm(1/estimated_spill_kib)",
        ),
        ("w_modern", "Arch. modernity", "🟡 Primary", "anchor_free_score ∈ {0,0.5,1}"),
    ]
    for key, name, tier, metric in criteria:
        table.add_row(name, tier, metric, f"{weights[key]:.2f}", WEIGHT_RATIONALE[key])

    _console.print(table)
    total = sum(weights.values())
    check = " ✓" if abs(total - 1.0) < 1e-6 else " [red]≠ 1.0[/red]"
    _console.print(f"\n  Weight sum: {total:.2f}{check}")
    _console.print()


def print_section3(scored: pd.DataFrame, weights: dict[str, float]) -> None:
    _console.print(Rule("[bold]Section 3 — Composite Scoring & Ranked Table[/bold]"))
    _console.print()
    _console.print(
        f"  [dim]score = {weights['w_acc']:.2f}·acc + {weights['w_energy']:.2f}·energy"
        f" + {weights['w_eff']:.2f}·eff + {weights['w_lat']:.2f}·lat"
        f" + {weights['w_mem']:.2f}·mem + {weights['w_modern']:.2f}·modern[/dim]"
    )
    _console.print()

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("#", style="bold", min_width=2, justify="right")
    table.add_column("Model Variant", no_wrap=True)
    table.add_column("fmt", min_width=4)
    table.add_column("res", min_width=3)
    table.add_column("DS", min_width=4)
    for h in ("acc", "nrg", "eff", "lat", "mem"):
        table.add_column(h, min_width=5, justify="right")
    table.add_column("mod", min_width=3, justify="right")
    table.add_column("inf_mW", min_width=6, justify="right")
    table.add_column("Score", style="bold cyan", min_width=6, justify="right")
    table.add_column("Flags", no_wrap=True)

    for rank, (_, row) in enumerate(scored.iterrows(), 1):
        flags = _flags(row)
        inf_mw = _f(row.get("pm_avg_inf_mW", float("nan")))
        ds = "Pers" if row["dataset"] == "COCO-Person" else "C80"
        score_str = f"{row['score']:.4f}"
        if rank <= 3:
            score_str = f"[bold]{score_str}[/bold]"
        table.add_row(
            str(rank),
            str(row["model_variant"]),
            str(row["format"]),
            str(int(row["resolution"])),
            ds,
            f"{row['s_acc']:.3f}",
            f"{row['s_energy']:.3f}",
            f"{row['s_eff']:.3f}",
            f"{row['s_lat']:.3f}",
            f"{row['s_mem']:.3f}",
            f"{row['s_modern']:.1f}",
            inf_mw,
            score_str,
            flags,
        )

    _console.print(table)
    _console.print()
    _console.print("  [bold]Abbreviations[/bold] (table columns)")
    _abbr_lines: list[tuple[str, str]] = [
        ("fmt", "Quantisation format (e.g. Int8, W4A8)."),
        ("res", "Input resolution (px)."),
        ("DS", "Dataset: Pers = COCO-Person, C80 = COCO-80."),
        ("acc", "Normalised accuracy subscore (cohort AP@0.5)."),
        ("nrg", "Normalised energy subscore (lower mJ/infer → higher)."),
        ("eff", "Normalised efficiency (geom. mean AP/ms & AP/mJ)."),
        ("lat", "Normalised latency subscore (faster → higher)."),
        ("mem", "Normalised activation-memory (on-chip vs spill)."),
        (
            "mod",
            "Architecture modernity: 0.0 legacy anchor, 0.5 modern backbone, "
            "1.0 anchor-free decoupled.",
        ),
        ("inf_mW", "Mean inference power draw (pm_avg_inf_mW)."),
        ("Score", "Weighted composite (see formula above)."),
        (
            "Flags",
            "[ExtRAM] activations exceed on-chip budget; [80only] COCO-80 row.",
        ),
    ]
    col_w = max(len(a) for a, _ in _abbr_lines)
    for abbr, desc in _abbr_lines:
        _console.print(
            f"    [bold]{abbr:<{col_w}}[/bold]  [dim]{desc}[/dim]"
        )
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
    option_b: bool = False,
    dataset_filter: str | None = None,
    min_size: int | None = None,
    min_ap: float | None = None,
    output_csv: Path | None = None,
    ap_csv: Path = DEFAULT_EVAL_CSV,
    memory_csv: Path = DEFAULT_MEMORY_CSV,
) -> pd.DataFrame:
    """Run the full model selection pipeline and print the report.

    *dataset_filter* restricts candidates to a single dataset
    ("80" for COCO-80, "person" for COCO-Person).  When set, the
    deduplication step is skipped since only one dataset is present.

    *min_size* drops candidates whose resolution is below this value.

    *min_ap* drops candidates whose AP@0.5 is below this value,
    filtering out models with insufficient feature-extraction capacity
    for fine-tuning to harder tasks.
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    option_label = "B (split memory: head + spill)" if option_b else "A (unified memory)"

    df = load_benchmark(csv_path)
    df = merge_benchmark_with_memory(df, memory_csv)
    df = merge_benchmark_with_ap(df, ap_csv)
    df = add_derived(df)

    if min_size is not None:
        before = len(df)
        df = df[df["resolution"] >= min_size].copy()
        dropped = before - len(df)
        if dropped:
            option_label += f" | min-size={min_size} ({dropped} rows dropped)"

    if min_ap is not None:
        before = len(df)
        df = df[(df["ap_50"].notna()) & (df["ap_50"] >= min_ap)].copy()
        dropped = before - len(df)
        if dropped:
            option_label += f" | min-ap={min_ap} ({dropped} rows dropped)"

    if dataset_filter is not None:
        ds_label = _DATASET_LABELS[dataset_filter]
        before = len(df)
        df = df[df["dataset"] == ds_label].copy()
        option_label += f" | dataset={ds_label} ({before - len(df)} rows filtered)"

    passing, excluded = gate_constraints(df)

    if dataset_filter is not None:
        scoring_pool, superseded = passing, pd.DataFrame(columns=passing.columns)
    else:
        scoring_pool, superseded = deduplicate_datasets(passing)

    scored = score_candidates(scoring_pool, w, option_b=option_b)

    print_section0(df)
    print_section1(df, scored, excluded, superseded)
    print_section2(w, option_label)
    print_section3(scored, w)

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
            "estimated_spill_kib",
            "weights_flash_kib",
            "efficiency_geo",
            "arch_modernity",
            "s_acc",
            "s_energy",
            "s_eff",
            "s_lat",
            "s_mem",
            "s_modern",
            "score",
        ]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        scored[out_cols].to_csv(
            output_csv,
            index=False,
            float_format="%.4f",
            quoting=csv.QUOTE_ALL,
        )
        _console.print(f"Scored results written to [bold]{output_csv}[/bold]")

    return scored


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
    option_b: bool = typer.Option(
        False,
        "--option-b",
        help="Use Option B scoring (split memory into head + spill terms)",
    ),
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
    min_ap: float | None = typer.Option(
        None,
        "--min-ap",
        help="Minimum AP@0.5. Models below this baseline accuracy are dropped.",
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
    w_mem: float = typer.Option(DEFAULT_WEIGHTS["w_mem"], "--w-mem"),
    w_modern: float = typer.Option(DEFAULT_WEIGHTS["w_modern"], "--w-modern"),
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
        "w_mem": w_mem,
        "w_modern": w_modern,
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
        option_b=option_b,
        dataset_filter=dataset,
        min_size=min_size,
        min_ap=min_ap,
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
