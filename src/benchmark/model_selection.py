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
# 0.5 = anchor-based + modern backbone
# 0.0 = legacy anchor-based

ARCHITECTURE_MODERNITY: dict[str, float] = {
    "ssdlite_mobilenetv1_pt": 0.0,
    "ssdlite_mobilenetv2_pt": 0.0,
    "ssdlite_mobilenetv3large_pt": 0.0,
    "ssdlite_mobilenetv3small_pt": 0.0,
    "st_yolodv2milli_pt": 0.5,
    "st_yolodv2tiny_pt": 0.5,
    "st_yololcv1": 1.0,
    "st_yoloxn": 1.0,
    "tinyissimoyolo_v8": 1.0,
    "yolo11n": 1.0,
    "yolo26": 1.0,
    "yolov8n": 1.0,
}

# ── Default scoring weights (Option A) ──────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_acc": 0.10,
    "w_energy": 0.20,
    "w_eff": 0.25,
    "w_lat": 0.10,
    "w_mem": 0.20,
    "w_modern": 0.15,
}

WEIGHT_RATIONALE: dict[str, str] = {
    "w_acc": (
        "Pre-trained AP is a capacity proxy; fine-tuning closes gaps. "
        "De-emphasised per Issue 2."
    ),
    "w_energy": (
        "Battery life critical for wearable safety device. "
        "Integrates power × duration."
    ),
    "w_eff": (
        "AP-per-unit-cost best predicts fine-tuning ROI on fixed "
        "hardware budget (Issue 2)."
    ),
    "w_lat": (
        "Beyond 15 FPS gate, lower latency improves responsiveness "
        "but is secondary."
    ),
    "w_mem": (
        "Full score when activation budget fits on-chip (no spill); "
        "HyperRAM spill scored lower by spill amount."
    ),
    "w_modern": (
        "Anchor-free decoupled heads fine-tune more reliably "
        "on small custom datasets."
    ),
}

# ── Fine-tuning profile (hand + hazardous-tool detection) ────────────────
#
# When selecting a base model to fine-tune for a NEW multi-class task,
# backbone capacity and transfer readiness dominate.  Hardware fit is
# still gated but the soft-score emphasis shifts.

FINETUNE_WEIGHTS: dict[str, float] = {
    "w_acc": 0.20,
    "w_energy": 0.15,
    "w_eff": 0.15,
    "w_lat": 0.05,
    "w_mem": 0.15,
    "w_modern": 0.30,
}

FINETUNE_WEIGHT_RATIONALE: dict[str, str] = {
    "w_acc": (
        "Backbone feature richness is the strongest predictor of "
        "transfer learning success. Higher pre-trained AP indicates "
        "more discriminative features for hand and tool detection."
    ),
    "w_energy": (
        "Battery life remains critical for a wearable safety device "
        "but is slightly de-emphasised vs. transfer potential."
    ),
    "w_eff": (
        "AP-per-unit-cost still relevant but less decisive than "
        "raw backbone capacity for base model selection."
    ),
    "w_lat": (
        "Beyond the 15 FPS hard gate, latency margin is the least "
        "important factor when choosing a fine-tuning base."
    ),
    "w_mem": (
        "On-chip activation fit still matters; HyperRAM spill penalty "
        "applies equally to the fine-tuned model."
    ),
    "w_modern": (
        "Transfer readiness: combines architecture modernity (anchor-free "
        "heads are replaceable for new class counts) with pre-training "
        "diversity (COCO-80 backbones learn richer features for multi-class "
        "hand + hazardous-tool detection than COCO-Person)."
    ),
}

# Dataset diversity scores for transfer readiness.
# COCO-80 pre-training teaches the backbone to separate 80 categories
# (including scissors, knife, person) — superior starting point for
# multi-class fine-tuning.  COCO-Person is still useful because body/hand
# features overlap strongly with the "hand" class.
_DATASET_TRANSFER_SCORE: dict[str, float] = {
    "COCO-80": 1.0,
    "COCO-Person": 0.4,
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
    *,
    prefer_multiclass: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate configs where both COCO-Person and COCO-80 exist.

    Default (prefer_multiclass=False): keep COCO-Person, supersede COCO-80.
    Fine-tuning (prefer_multiclass=True): keep COCO-80, supersede COCO-Person,
    because multi-class pre-training produces richer backbone features for
    transfer to a new multi-class task (hand + hazardous-tool detection).
    """
    if prefer_multiclass:
        preferred_ds, superseded_ds = "COCO-80", "COCO-Person"
    else:
        preferred_ds, superseded_ds = "COCO-Person", "COCO-80"

    preferred_configs: set[tuple] = set()
    for _, row in df[df["dataset"] == preferred_ds].iterrows():
        preferred_configs.add(tuple(row[c] for c in _DEDUP_KEY))

    superseded_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        if row["dataset"] == superseded_ds:
            if tuple(row[c] for c in _DEDUP_KEY) in preferred_configs:
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
    finetune: bool = False,
) -> pd.DataFrame:
    """Compute composite scores. Returns df sorted descending by score.

    When *finetune* is True the "modernity" sub-score is expanded into a
    **transfer readiness** signal that blends architecture modernity with
    pre-training dataset diversity (COCO-80 > COCO-Person for multi-class
    fine-tuning).
    """
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

    # Efficiency curve — geometric mean of AP/ms and AP/mJ
    has_eff = out["efficiency_geo"].notna() & (out["efficiency_geo"] > 0)
    if has_eff.sum() >= 2:
        out["s_eff"] = 0.0
        out.loc[has_eff, "s_eff"] = _minmax(out.loc[has_eff, "efficiency_geo"])
        if (~has_eff).any():
            out.loc[~has_eff, "s_eff"] = _minmax(out.loc[~has_eff, "acc_per_ms"])
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

    # Architecture modernity / transfer readiness
    if finetune:
        arch = out["arch_modernity"]
        diversity = out["dataset"].map(_DATASET_TRANSFER_SCORE).fillna(0.5)
        out["s_modern"] = 0.5 * arch + 0.5 * diversity
    else:
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
    *,
    finetune: bool = False,
) -> None:
    _console.print(Rule("[bold]Section 1 — Data Cleaning & Anomaly Report[/bold]"))
    _console.print()

    if finetune:
        issue1 = (
            "Issue 1 — Dataset Selection for Fine-Tuning",
            "COCO-80 pre-trained models preferred: multi-class pre-training produces "
            "richer backbone features for transfer to hand + hazardous-tool detection. "
            "COCO-Person rows with COCO-80 counterparts marked 'Superseded'. "
            "Accuracy normalised within each cohort separately.",
        )
        issue2 = (
            "Issue 2 — Fine-Tuning Base Selection",
            "Transfer readiness (arch modernity + pre-training diversity) weighted 0.30 — "
            "highest single weight. Backbone capacity (AP) weighted 0.20. "
            "Hardware-fit criteria de-emphasised but hard gates still enforced.",
        )
    else:
        issue1 = (
            "Issue 1 — Dataset Incomparability",
            "COCO-Person AP used as primary. COCO-80 rows with Person counterparts "
            "marked 'Superseded'. Accuracy normalised within each cohort separately.",
        )
        issue2 = (
            "Issue 2 — Fine-Tuning Context",
            "Efficiency curve (geometric mean of AP/ms and AP/mJ) weighted 0.25 — "
            "highest single weight. Raw AP weighted 0.10 only.",
        )

    issues = [
        issue1,
        issue2,
        (
            "Issue 3 — Resolution Variants",
            "All passing resolutions scored independently. Best per-family identified "
            "post-scoring. Variants within 5% composite listed as alternatives.",
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
            flags.append(f"[{row['dataset']}]")
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


def print_section2(
    weights: dict[str, float],
    option_label: str,
    *,
    rationale: dict[str, str] | None = None,
) -> None:
    rat = rationale or WEIGHT_RATIONALE
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

    is_finetune = rat is FINETUNE_WEIGHT_RATIONALE
    modern_name = "Transfer readiness" if is_finetune else "Arch. modernity"
    modern_metric = (
        "0.5·arch_modernity + 0.5·dataset_diversity"
        if is_finetune
        else "anchor_free_score ∈ {0,0.5,1}"
    )

    criteria = [
        ("w_acc", "Architecture capacity", "🟡 Primary", "norm_cohort(ap_50)"),
        ("w_energy", "Energy per inference", "🟡 Primary", "norm(1/pm_avg_inf_mJ)"),
        ("w_eff", "Efficiency curve", "🟡 Primary", "norm(geomean(AP/ms, AP/mJ))"),
        ("w_lat", "Latency margin", "🟢 Secondary", "norm(1/inference_time_ms)"),
        (
            "w_mem",
            "Activation memory",
            "🟡 Primary",
            "1.0 if no spill else norm(1/estimated_spill_kib)",
        ),
        ("w_modern", modern_name, "🟡 Primary" if is_finetune else "🟢 Secondary", modern_metric),
    ]
    for key, name, tier, metric in criteria:
        table.add_row(name, tier, metric, f"{weights[key]:.2f}", rat[key])

    _console.print(table)
    total = sum(weights.values())
    check = " ✓" if abs(total - 1.0) < 1e-6 else " [red]≠ 1.0[/red]"
    _console.print(f"\n  Weight sum: {total:.2f}{check}")
    _console.print()


def print_section3(
    scored: pd.DataFrame,
    weights: dict[str, float],
    *,
    finetune: bool = False,
) -> None:
    mod_label = "xfer" if finetune else "modern"
    _console.print(Rule("[bold]Section 3 — Composite Scoring & Ranked Table[/bold]"))
    _console.print()
    _console.print(
        f"  [dim]score = {weights['w_acc']:.2f}·acc + {weights['w_energy']:.2f}·energy"
        f" + {weights['w_eff']:.2f}·eff + {weights['w_lat']:.2f}·lat"
        f" + {weights['w_mem']:.2f}·mem + {weights['w_modern']:.2f}·{mod_label}[/dim]"
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
    table.add_column(mod_label[:4], min_width=4, justify="right")
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
    if finetune:
        mod_abbr = (
            "xfer",
            "Transfer readiness: 0.5·arch_modernity + 0.5·dataset_diversity "
            "(COCO-80=1.0, COCO-Person=0.4).",
        )
    else:
        mod_abbr = (
            "mod",
            "Architecture modernity: 0.0 legacy anchor, 0.5 modern backbone, "
            "1.0 anchor-free decoupled.",
        )
    _abbr_lines: list[tuple[str, str]] = [
        ("fmt", "Quantisation format (e.g. Int8, W4A8)."),
        ("res", "Input resolution (px)."),
        ("DS", "Dataset: Pers = COCO-Person, C80 = COCO-80."),
        ("acc", "Normalised accuracy subscore (cohort AP@0.5)."),
        ("nrg", "Normalised energy subscore (lower mJ/infer → higher)."),
        ("eff", "Normalised efficiency (geom. mean AP/ms & AP/mJ)."),
        ("lat", "Normalised latency subscore (faster → higher)."),
        ("mem", "Normalised activation-memory (on-chip vs spill)."),
        mod_abbr,
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


def print_section4(scored: pd.DataFrame, *, finetune: bool = False) -> None:
    title = (
        "Section 4 — Interpretation (Fine-Tuning Base Selection)"
        if finetune
        else "Section 4 — Interpretation"
    )
    _console.print(Rule(f"[bold]{title}[/bold]"))
    _console.print()

    if finetune:
        _console.print(
            "  [dim]Target task: hand + hazardous-tool detection (multi-class).[/dim]"
        )
        _console.print(
            "  [dim]Selecting backbone + architecture for subsequent fine-tuning.[/dim]"
        )
        _console.print()

    top_n = min(10, len(scored))
    for rank in range(top_n):
        row = scored.iloc[rank]
        _console.print(
            Rule(
                f"[bold]Rank {rank + 1}: {row['model_variant']}[/bold]",
                style="dim",
            )
        )

        res = int(row["resolution"])
        _console.print(
            f"  [bold]Configuration:[/bold] {row['model_family']} / "
            f"{row['model_variant']} | {row['format']} | {res}×{res} | "
            f"{row['dataset']}"
        )

        mod_key = "transfer" if finetune else "modernity"
        components = {
            "accuracy": row["s_acc"],
            "energy": row["s_energy"],
            "efficiency": row["s_eff"],
            "latency": row["s_lat"],
            "memory": row["s_mem"],
            mod_key: row["s_modern"],
        }
        top_drivers = sorted(components.items(), key=lambda x: -x[1])[:3]
        drivers_str = ", ".join(f"{n}={v:.3f}" for n, v in top_drivers)
        _console.print(f"  [bold]Score drivers:[/bold] {drivers_str}")

        act = row["activations_without_io"]
        spill = row["estimated_spill_kib"]
        pct = min(100, 100 * ON_CHIP_ACTIVATION_KIB / act) if act > 0 else 100
        _console.print(
            f"  [bold]Memory:[/bold] activations_without_io = {act:.1f} KiB "
            f"({pct:.0f}% on-chip) | spill = {spill:.1f} KiB"
        )

        caveats: list[str] = []
        if spill > 0:
            caveats.append(f"HyperRAM spill of {spill:.0f} KiB")
        if row["dataset"] == "COCO-80" and not finetune:
            caveats.append(
                "COCO-80 only — conservative accuracy signal; "
                "fine-tuning uplift may be greater"
            )
        if row["dataset"] == "COCO-Person" and finetune:
            caveats.append(
                "COCO-Person only — single-class pre-training; backbone features "
                "less diverse than COCO-80 for multi-class transfer"
            )
        if row["inference_time_ms"] > 50:
            margin = MAX_LATENCY_MS - row["inference_time_ms"]
            caveats.append(f"Tight latency margin ({margin:.1f} ms to deadline)")
        if not caveats:
            caveats.append("None significant")
        _console.print(f"  [bold]Caveats:[/bold] {'; '.join(caveats)}")

        energy_str = _f(row.get("pm_avg_inf_mJ", float("nan")))
        _console.print(
            f"  [bold]Operating point:[/bold] {row['format']} @ {res} "
            f"({row['inference_time_ms']:.1f} ms, {energy_str} mJ)"
        )

        arch = row["arch_modernity"]
        if finetune:
            ds_note = (
                "COCO-80 backbone (rich multi-class features)"
                if row["dataset"] == "COCO-80"
                else "COCO-Person backbone (strong body/hand priors)"
            )
            if arch >= 1.0:
                ft = (
                    f"Anchor-free decoupled head + {ds_note} — replace head "
                    "with 2-class (hand, hazardous_tool) and fine-tune"
                )
            elif arch >= 0.5:
                ft = (
                    f"Modern backbone + {ds_note} — requires anchor "
                    "recalibration for hand + tool aspect ratios"
                )
            else:
                ft = (
                    f"Legacy SSD + {ds_note} — anchor priors need "
                    "significant recalibration; least recommended for fine-tuning"
                )
        else:
            if arch >= 1.0:
                ft = (
                    "Anchor-free decoupled head — straightforward fine-tuning "
                    "with standard YOLO pipelines"
                )
            elif arch >= 0.5:
                ft = (
                    "Anchor-based + modern backbone — requires anchor tuning "
                    "for custom dataset"
                )
            else:
                ft = (
                    "Legacy SSD architecture — anchor priors may need "
                    "significant recalibration"
                )
        _console.print(f"  [bold]Fine-tuning:[/bold] {ft}")
        _console.print()

    # ── Format comparisons (Int8 vs W4A8 at same config) ─────────────────

    _console.print(Rule("[bold]Format Comparisons (Int8 vs W4A8)[/bold]", style="dim"))
    _console.print()

    any_fmt_pair = False
    fmt_groups = scored.groupby(["model_family", "hyperparameters", "resolution"])
    for (family, hypers, res), group in fmt_groups:
        formats = group["format"].unique()
        if "Int8" not in formats or "W4A8" not in formats:
            continue
        any_fmt_pair = True
        i8 = group[group["format"] == "Int8"].iloc[0]
        w4 = group[group["format"] == "W4A8"].iloc[0]

        i8_wins: list[str] = []
        w4_wins: list[str] = []
        for metric, lower_better in [
            ("inference_time_ms", True),
            ("pm_avg_inf_mJ", True),
            ("weights_flash_kib", True),
            ("ap_50", False),
        ]:
            vi, vw = i8[metric], w4[metric]
            if pd.isna(vi) or pd.isna(vw):
                continue
            label = metric.split("_")[0] if "_" in metric else metric
            if lower_better:
                (i8_wins if vi < vw else w4_wins).append(label)
            else:
                (i8_wins if vi >= vw else w4_wins).append(label)

        delta_score = i8["score"] - w4["score"]
        hyp_label = f" ({hypers})" if hypers else ""
        _console.print(
            f"  [bold]{family}{hyp_label} @ {int(res)}:[/bold]"
        )
        _console.print(
            f"    Int8 score={i8['score']:.4f} | W4A8 score={w4['score']:.4f} | "
            f"Δ={delta_score:+.4f}"
        )
        _console.print(
            f"    Int8 wins: {', '.join(i8_wins) or '—'} | "
            f"W4A8 wins: {', '.join(w4_wins) or '—'}"
        )

        if delta_score >= 0:
            verdict = "→ Int8 preferred (fine-tuning stability)"
        elif abs(delta_score) < 0.02:
            verdict = (
                "→ Int8 preferred (marginal W4A8 advantage insufficient "
                "for fine-tuning risk)"
            )
        else:
            verdict = "→ W4A8 preferred (decisive efficiency advantage)"
        _console.print(f"    {verdict}")
        _console.print()

    if not any_fmt_pair:
        _console.print("  No families with both Int8 and W4A8 at same resolution.\n")

    # ── Resolution variants per family ───────────────────────────────────

    _console.print(
        Rule("[bold]Resolution Variants (per family)[/bold]", style="dim")
    )
    _console.print()

    res_groups = scored.groupby(["model_family", "hyperparameters", "format"])
    any_multi_res = False
    for (family, hypers, fmt), group in res_groups:
        if len(group) < 2:
            continue
        any_multi_res = True
        group_s = group.sort_values("score", ascending=False)
        best_score = group_s.iloc[0]["score"]
        hyp_label = f" ({hypers})" if hypers else ""
        _console.print(f"  [bold]{family}{hyp_label} {fmt}:[/bold]")

        for i, (_, row) in enumerate(group_s.iterrows()):
            marker = "★" if i == 0 else " "
            delta = (row["score"] - best_score) / best_score if best_score > 0 else 0
            within_5 = abs(delta) < 0.05 and i > 0
            alt = " [dim][viable alternative][/dim]" if within_5 else ""
            _console.print(
                f"    {marker} {int(row['resolution'])}×{int(row['resolution'])}: "
                f"score={row['score']:.4f} | {row['inference_time_ms']:.1f} ms | "
                f"AP={row['ap_50']:.1f}{alt}"
            )
        _console.print()

    if not any_multi_res:
        _console.print(
            "  No families with multiple resolution variants in scoring pool.\n"
        )


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_selection(
    csv_path: Path,
    weights: dict[str, float] | None = None,
    *,
    option_b: bool = False,
    finetune: bool = False,
    output_csv: Path | None = None,
    ap_csv: Path = DEFAULT_EVAL_CSV,
    memory_csv: Path = DEFAULT_MEMORY_CSV,
) -> pd.DataFrame:
    """Run the full model selection pipeline and print the report.

    When *finetune* is True the algorithm shifts to base-model selection
    for fine-tuning to hand + hazardous-tool detection:
    - COCO-80 pre-trained models are preferred over COCO-Person
    - Transfer readiness (architecture + pre-training diversity) is
      weighted more heavily
    - Backbone capacity (accuracy) is emphasised
    """
    w = weights or (FINETUNE_WEIGHTS.copy() if finetune else DEFAULT_WEIGHTS.copy())
    rationale = FINETUNE_WEIGHT_RATIONALE if finetune else WEIGHT_RATIONALE
    option_label = "B (split memory: head + spill)" if option_b else "A (unified memory)"
    if finetune:
        option_label += " | fine-tune profile (hand + hazardous-tool)"

    df = load_benchmark(csv_path)
    df = merge_benchmark_with_memory(df, memory_csv)
    df = merge_benchmark_with_ap(df, ap_csv)
    df = add_derived(df)

    passing, excluded = gate_constraints(df)
    scoring_pool, superseded = deduplicate_datasets(
        passing, prefer_multiclass=finetune
    )

    scored = score_candidates(
        scoring_pool, w, option_b=option_b, finetune=finetune
    )

    print_section0(df)
    print_section1(df, scored, excluded, superseded, finetune=finetune)
    print_section2(w, option_label, rationale=rationale)
    print_section3(scored, w, finetune=finetune)
    print_section4(scored, finetune=finetune)

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
    finetune: bool = typer.Option(
        False,
        "--finetune",
        help=(
            "Fine-tuning base selection mode: prefer COCO-80 pre-training, "
            "boost transfer readiness weight, emphasise backbone capacity."
        ),
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

    cli_weights = {
        "w_acc": w_acc,
        "w_energy": w_energy,
        "w_eff": w_eff,
        "w_lat": w_lat,
        "w_mem": w_mem,
        "w_modern": w_modern,
    }

    # When --finetune is set and no --w-* flags were explicitly provided,
    # let run_selection use FINETUNE_WEIGHTS instead of the CLI defaults.
    weights_match_defaults = all(
        abs(cli_weights[k] - DEFAULT_WEIGHTS[k]) < 1e-9 for k in DEFAULT_WEIGHTS
    )
    weights: dict[str, float] | None
    if finetune and weights_match_defaults:
        weights = None
    else:
        weights = cli_weights
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
        finetune=finetune,
        output_csv=output,
        ap_csv=DEFAULT_EVAL_CSV,
        memory_csv=DEFAULT_MEMORY_CSV,
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
