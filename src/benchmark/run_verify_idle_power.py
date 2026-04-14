#!/usr/bin/env python3
"""Audit stability of ``pm_avg_idle_mW`` per benchmark mode CSV.

Reads ``results/benchmark_{underdrive,nominal,overdrive}_results.csv``
and reports basic spread stats (mean, std, CV, percentiles, Tukey outliers).

Usage:
    ./project.py verify-idle-power
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.paths import benchmark_paths_for_mode

PRIMARY_METRIC = "pm_avg_idle_mW"
IDLE_OUTLIER_CONTEXT = ("pm_avg_idle_ms", "pm_avg_idle_mJ")
_MODES = ("underdrive", "nominal", "overdrive")


def _tukey_fences(x: np.ndarray) -> tuple[float, float]:
    q25, q75 = np.percentile(x.astype(float), [25, 75]).astype(float)
    iqr = float(q75 - q25)
    return float(q25 - 1.5 * iqr), float(q75 + 1.5 * iqr)


def _unit_for(metric: str) -> str:
    if metric.endswith("_mW"):
        return "mW"
    if metric.endswith("_ms"):
        return "ms"
    if metric.endswith("_mJ"):
        return "mJ"
    return ""


def _print_outliers(
    df: pd.DataFrame,
    metric: str,
    s: pd.Series,
    *,
    max_rows: int,
    companion_cols: tuple[str, ...] = (),
) -> None:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    x = vals.to_numpy(dtype=float)
    if x.size < 4:
        return

    lo, hi = _tukey_fences(x)
    out = vals[(vals < lo) | (vals > hi)]
    if out.empty:
        return

    unit = _unit_for(metric)
    unit_s = f" {unit}" if unit else ""
    print(f"  Outliers (Tukey 1.5*IQR; fences {lo:.3f}..{hi:.3f}{unit_s}): {len(out)}")

    cols_pref = ["model_variant", "format", "resolution", "dataset"]
    cols = [c for c in cols_pref if c in df.columns]
    if not cols:
        cols = [
            c for c in ("model_family", "model_variant", "format") if c in df.columns
        ]

    shown = out.sort_values().head(max_rows)
    for idx, v in shown.items():
        ident = ""
        if cols:
            row = df.loc[idx, cols]
            if isinstance(row, pd.Series):
                ident = " | " + ", ".join(f"{c}={row[c]!s}" for c in cols)
        extra = ""
        if companion_cols:
            parts: list[str] = []
            for c in companion_cols:
                if c not in df.columns:
                    continue
                raw = df.loc[idx, c]
                cv = pd.to_numeric(raw, errors="coerce")
                if pd.isna(cv):
                    continue
                u = _unit_for(c)
                u_s = f" {u}" if u else ""
                parts.append(f"{c}={float(cv):.3f}{u_s}")
            if parts:
                extra = " | " + ", ".join(parts)
        print(f"    - row {int(idx)}: {metric}={float(v):.3f}{unit_s}{extra}{ident}")

    if len(out) > len(shown):
        print(
            f"    ... {len(out) - len(shown)} more outliers not shown (use --max-outliers to raise)"
        )


def _audit_values(values: np.ndarray) -> dict[str, float | int]:
    x = values.astype(float)
    n = int(x.size)
    if n == 0:
        return {"n": 0}

    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    cv = float(std / mean) if mean != 0 else float("nan")
    mn, mx = float(x.min()), float(x.max())
    pcts = np.percentile(x, [5, 25, 50, 75, 95]).astype(float)
    q25, q50, q75 = float(pcts[1]), float(pcts[2]), float(pcts[3])
    iqr = q75 - q25
    lo = q25 - 1.5 * iqr
    hi = q75 + 1.5 * iqr
    outliers = int(((x < lo) | (x > hi)).sum())
    mad = float(np.median(np.abs(x - q50)))
    robust_sigma = float(1.4826 * mad)

    idx = np.arange(n, dtype=float)
    drift = float(np.corrcoef(idx, x)[0, 1]) if n > 2 else float("nan")

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": mn,
        "max": mx,
        "range": mx - mn,
        "p05": float(pcts[0]),
        "p25": q25,
        "p50": q50,
        "p75": q75,
        "p95": float(pcts[4]),
        "iqr": iqr,
        "tukey_outliers": outliers,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "corr_index": drift,
    }


def _format_report(
    mode: str, metric: str, csv_path: Path, stats: dict[str, float | int]
) -> str:
    unit = _unit_for(metric)
    unit_s = f" {unit}" if unit else ""
    lines = [
        f"## {mode} — {csv_path}",
        f"  metric: {metric}",
        f"  rows (non-null): {stats['n']}",
    ]
    if stats["n"] == 0:
        return "\n".join(lines)

    lines.extend(
        [
            f"  mean: {stats['mean']:.3f}{unit_s}",
            f"  std:  {stats['std']:.3f}{unit_s}",
            f"  CV:   {stats['cv']:.4f}",
            f"  min / max: {stats['min']:.3f} / {stats['max']:.3f}{unit_s} (range {stats['range']:.3f}{unit_s})",
            f"  p05 / p50 / p95: {stats['p05']:.3f} / {stats['p50']:.3f} / {stats['p95']:.3f}{unit_s}",
            f"  IQR: {stats['iqr']:.3f}{unit_s} | Tukey outliers: {stats['tukey_outliers']}",
            f"  MAD: {stats['mad']:.3f}{unit_s} | robust σ (~): {stats['robust_sigma']:.3f}{unit_s}",
            f"  corr(metric, row_index): {stats['corr_index']:.4f}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit pm_avg_idle_mW per benchmark mode CSV."
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=list(_MODES),
        dest="modes",
        help=f"Benchmark mode (repeatable). Default: all ({', '.join(_MODES)}).",
    )
    parser.add_argument(
        "--max-outliers",
        type=int,
        default=20,
        help="Maximum outlier rows to print per mode (default: 20).",
    )
    args = parser.parse_args()
    modes = tuple(args.modes) if args.modes else _MODES

    failures = 0
    for mode in modes:
        csv_path = benchmark_paths_for_mode(mode).csv_path
        if not csv_path.is_file():
            print(f"## {mode} — MISSING {csv_path}", file=sys.stderr)
            failures += 1
            continue

        df = pd.read_csv(csv_path)
        metric = PRIMARY_METRIC
        if metric not in df.columns:
            print(f"## {mode} — {csv_path}: missing column {metric!r}", file=sys.stderr)
            failures += 1
            continue

        s = pd.to_numeric(df[metric], errors="coerce")
        vals = s.dropna().to_numpy()
        null_ct = int(s.isna().sum())
        stats = _audit_values(vals)
        print(_format_report(mode, metric, csv_path, stats))
        if null_ct:
            print(f"  warning: {null_ct} null/non-numeric {metric} rows ignored")
        _print_outliers(
            df,
            metric,
            s,
            max_rows=int(args.max_outliers),
            companion_cols=IDLE_OUTLIER_CONTEXT,
        )
        print()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
