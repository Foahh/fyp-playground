"""Compare benchmark / README metrics with configurable modes.

The official ST Model Zoo NPU figures parsed from object_detection README tables are
measured in **overdrive mode** as part of the **default STM32Cube.AI configuration**
(see each family README under *Performances → Metrics*, e.g. input/output allocated).
That matches the high-performance supply/setup used for STM32N6570-DK reference
benchmarking, not a separate “nominal vs overdrive” split on the README side.

Modes (see ``--mode``):

- **readme-overdrive** — ``benchmark_parsed.csv`` vs ``benchmark_overdrive/benchmark_results.csv``
  (delta = measured − readme). Parsed rows with no metrics are skipped; empty parsed
  cells are not compared.

- **readme-nominal** — same, but measured file is ``benchmark_nominal/benchmark_results.csv``.

- **nominal-overdrive** — nominal vs overdrive ``benchmark_results.csv`` (delta = overdrive − nominal)
  for every shared config and metric column; ``stedgeai_version`` is listed when present on either side.

For readme modes, ``stedgeai_version`` is also listed per matched row (README-parsed vs measured
``benchmark_results``); values are compared as strings (delta column shows ``≠`` when they differ).

Output is grouped by ``model_variant`` with plain-text tables (no pass/fail).

Use ``--min-abs-delta-pct PCT`` to hide metric rows whose numeric ``|delta_pct|`` is below PCT;
rows without a numeric ``delta_pct`` (e.g. ``stedgeai_version``, baseline zero) stay listed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import BASE_DIR, CSV_COLUMNS_NO_POWER


RESULTS_DIR = BASE_DIR / "results"
DEFAULT_PARSED_CSV = RESULTS_DIR / "benchmark_parsed.csv"
DEFAULT_NOMINAL_CSV = RESULTS_DIR / "benchmark_nominal" / "benchmark_results.csv"
DEFAULT_OVERDRIVE_CSV = RESULTS_DIR / "benchmark_overdrive" / "benchmark_results.csv"
# Back-compat alias for readme-overdrive default second file
DEFAULT_BENCHMARK_CSV = DEFAULT_OVERDRIVE_CSV

COMPARE_MODES = ("readme-overdrive", "readme-nominal", "nominal-overdrive")

IDENTITY_COLS = (
    "model_family",
    "model_variant",
    "hyperparameters",
    "dataset",
    "format",
    "resolution",
)

METRIC_COLS = tuple(
    c
    for c in CSV_COLUMNS_NO_POWER
    if c not in IDENTITY_COLS and c not in ("host_time_iso", "stedgeai_version")
)

# Per-variant subtable (model name is printed as a section heading above).
PER_VARIANT_COLUMNS_README = (
    "dataset",
    "format",
    "res",
    "metric",
    "readme",
    "measured",
    "delta",
    "delta_pct",
)
PER_VARIANT_COLUMNS_BENCH_PAIR = (
    "dataset",
    "format",
    "res",
    "metric",
    "nominal",
    "overdrive",
    "delta",
    "delta_pct",
)


@dataclass
class ComparisonResult:
    mode: str = "readme-overdrive"
    headline: str = ""
    table_columns: tuple[str, ...] = field(default_factory=lambda: PER_VARIANT_COLUMNS_README)
    skipped_parsed_no_readme: int = 0
    matched_rows: int = 0
    missing_in_benchmark: list[str] = field(default_factory=list)
    missing_in_parsed: list[str] = field(default_factory=list)
    duplicate_benchmark_keys: list[str] = field(default_factory=list)
    duplicate_nominal_keys: list[str] = field(default_factory=list)
    missed_numeric_compare: int = 0
    delta_rows: list[dict[str, str]] = field(default_factory=list)


def _blank(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == ""


def _norm_key_part(v: Any) -> str:
    return ("" if v is None else str(v)).strip()


def _resolution_int(v: Any) -> int:
    s = _norm_key_part(v)
    if not s:
        raise ValueError("empty resolution")
    return int(round(float(s)))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        _norm_key_part(row.get("model_family")),
        _norm_key_part(row.get("model_variant")),
        _norm_key_part(row.get("hyperparameters")),
        _norm_key_part(row.get("dataset")),
        _norm_key_part(row.get("format")),
        str(_resolution_int(row.get("resolution"))),
    )


def key_label(k: tuple[str, ...]) -> str:
    return "|".join(k)


def parsed_row_has_any_metric(row: dict[str, str]) -> bool:
    return any(not _blank(row.get(c)) for c in METRIC_COLS)


def _parse_float(cell: str) -> float | None:
    s = (cell or "").strip()
    if s == "":
        return None
    return float(s)


def _external_ram_kib_float(raw: str) -> float:
    """Treat blanks and null-like tokens as 0; non-numeric garbage → 0 (same as missing)."""
    if _blank(raw):
        return 0.0
    s = (raw or "").strip().lower()
    if s in ("null", "none", "n/a", "na", "-"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def metric_floats(col: str, parsed_raw: str, bench_raw: str) -> tuple[float | None, float | None]:
    """Numeric readme vs measured; external RAM treats blank as 0."""
    if col == "external_ram_kib":
        return _external_ram_kib_float(parsed_raw), _external_ram_kib_float(bench_raw)
    pr = _parse_float(parsed_raw)
    br = _parse_float(bench_raw)
    return pr, br


def _fmt_num(x: float | None) -> str:
    if x is None:
        return ""
    if abs(x) >= 1000 or (abs(x) >= 1 and abs(x - round(x)) < 1e-4):
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4g}"


def _delta_cells(pr: float | None, br: float | None) -> tuple[str, str]:
    if pr is None:
        return "", ""
    if br is None:
        return "", ""
    d = br - pr
    ds = _fmt_num(d)
    if pr == 0:
        if br == 0:
            return ds, "+0.00%"
        return ds, "—"
    pct = 100.0 * (br - pr) / pr
    return ds, f"{pct:+.2f}%"


def _res_sort_key(res_s: str) -> int:
    s = (res_s or "").strip()
    return int(s) if s.isdigit() else 0


def _delta_sort_key(delta_s: str) -> float:
    s = (delta_s or "").strip()
    if not s:
        return float("inf")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _parse_delta_pct_value(pct_str: str) -> float | None:
    """Parse a ``delta_pct`` cell like ``+1.23%``; return None for ``—`` or non-numeric."""
    s = (pct_str or "").strip()
    if not s or s == "—":
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        v = float(s)
        if v != v:  # NaN
            return None
        return v
    except ValueError:
        return None


def filter_rows_by_abs_delta_pct(
    rows: list[dict[str, str]],
    min_abs_pct: float,
) -> list[dict[str, str]]:
    """Keep rows with no numeric ``delta_pct``, or whose ``|delta_pct|`` is >= ``min_abs_pct``."""
    if min_abs_pct < 0:
        raise ValueError("min_abs_pct must be non-negative")
    out: list[dict[str, str]] = []
    for r in rows:
        v = _parse_delta_pct_value(r.get("delta_pct", ""))
        if v is None or abs(v) >= min_abs_pct:
            out.append(r)
    return out


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"No header in {path}")
        rows = list(r)
        return list(r.fieldnames), rows


def compare_readme_to_bench(
    parsed_path: Path,
    bench_path: Path,
    *,
    mode: str,
    headline: str,
) -> ComparisonResult:
    _, parsed_rows = load_csv(parsed_path)
    _, bench_rows = load_csv(bench_path)

    out = ComparisonResult(
        mode=mode,
        headline=headline,
        table_columns=PER_VARIANT_COLUMNS_README,
    )

    bench_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in bench_rows:
        k = row_key(row)
        if k in bench_by_key:
            out.duplicate_benchmark_keys.append(key_label(k))
        else:
            bench_by_key[k] = row

    parsed_keys_with_data: set[tuple[str, ...]] = set()

    for prow in parsed_rows:
        if not parsed_row_has_any_metric(prow):
            out.skipped_parsed_no_readme += 1
            continue
        k = row_key(prow)
        parsed_keys_with_data.add(k)
        brow = bench_by_key.get(k)
        if brow is None:
            out.missing_in_benchmark.append(key_label(k))
            continue
        out.matched_rows += 1
        res_s = str(_resolution_int(prow.get("resolution")))
        for col in METRIC_COLS:
            if _blank(prow.get(col)):
                continue
            pr, br = metric_floats(col, prow.get(col, ""), brow.get(col, ""))
            if pr is None or br is None:
                out.missed_numeric_compare += 1
            d_str, pct_str = _delta_cells(pr, br)
            out.delta_rows.append(
                {
                    "model_variant": _norm_key_part(prow.get("model_variant")),
                    "dataset": _norm_key_part(prow.get("dataset")),
                    "format": _norm_key_part(prow.get("format")),
                    "res": res_s,
                    "metric": col,
                    "readme": _fmt_num(pr) if pr is not None else prow.get(col, "").strip(),
                    "measured": _fmt_num(br) if br is not None else "",
                    "delta": d_str,
                    "delta_pct": pct_str,
                }
            )
        pv = _norm_key_part(prow.get("stedgeai_version"))
        bv = _norm_key_part(brow.get("stedgeai_version"))
        if pv or bv:
            out.delta_rows.append(
                {
                    "model_variant": _norm_key_part(prow.get("model_variant")),
                    "dataset": _norm_key_part(prow.get("dataset")),
                    "format": _norm_key_part(prow.get("format")),
                    "res": res_s,
                    "metric": "stedgeai_version",
                    "readme": pv,
                    "measured": bv,
                    "delta": "" if pv == bv else "≠",
                    "delta_pct": "—",
                }
            )

    for k in bench_by_key:
        if k not in parsed_keys_with_data:
            out.missing_in_parsed.append(key_label(k))

    return out


def compare_nominal_to_overdrive(nominal_path: Path, overdrive_path: Path) -> ComparisonResult:
    _, nominal_rows = load_csv(nominal_path)
    _, over_rows = load_csv(overdrive_path)

    out = ComparisonResult(
        mode="nominal-overdrive",
        headline="Nominal vs overdrive (delta = overdrive − nominal)",
        table_columns=PER_VARIANT_COLUMNS_BENCH_PAIR,
    )

    nominal_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in nominal_rows:
        k = row_key(row)
        if k in nominal_by_key:
            out.duplicate_nominal_keys.append(key_label(k))
        else:
            nominal_by_key[k] = row

    over_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in over_rows:
        k = row_key(row)
        if k in over_by_key:
            out.duplicate_benchmark_keys.append(key_label(k))
        else:
            over_by_key[k] = row

    nominal_keys = set(nominal_by_key.keys())

    for k, nrow in nominal_by_key.items():
        orow = over_by_key.get(k)
        if orow is None:
            out.missing_in_benchmark.append(key_label(k))
            continue
        out.matched_rows += 1
        res_s = str(_resolution_int(nrow.get("resolution")))
        for col in METRIC_COLS:
            if _blank(nrow.get(col)) and _blank(orow.get(col)):
                continue
            pr, br = metric_floats(col, nrow.get(col, ""), orow.get(col, ""))
            if pr is None or br is None:
                out.missed_numeric_compare += 1
            d_str, pct_str = _delta_cells(pr, br)
            out.delta_rows.append(
                {
                    "model_variant": _norm_key_part(nrow.get("model_variant")),
                    "dataset": _norm_key_part(nrow.get("dataset")),
                    "format": _norm_key_part(nrow.get("format")),
                    "res": res_s,
                    "metric": col,
                    "nominal": _fmt_num(pr) if pr is not None else nrow.get(col, "").strip(),
                    "overdrive": _fmt_num(br) if br is not None else orow.get(col, "").strip(),
                    "delta": d_str,
                    "delta_pct": pct_str,
                }
            )
        nv = _norm_key_part(nrow.get("stedgeai_version"))
        ov = _norm_key_part(orow.get("stedgeai_version"))
        if nv or ov:
            out.delta_rows.append(
                {
                    "model_variant": _norm_key_part(nrow.get("model_variant")),
                    "dataset": _norm_key_part(nrow.get("dataset")),
                    "format": _norm_key_part(nrow.get("format")),
                    "res": res_s,
                    "metric": "stedgeai_version",
                    "nominal": nv,
                    "overdrive": ov,
                    "delta": "" if nv == ov else "≠",
                    "delta_pct": "—",
                }
            )

    for k in over_by_key:
        if k not in nominal_keys:
            out.missing_in_parsed.append(key_label(k))

    return out


def compare(parsed_path: Path, benchmark_path: Path) -> ComparisonResult:
    """Backward-compatible: readme vs a single measured CSV."""
    return compare_readme_to_bench(
        parsed_path,
        benchmark_path,
        mode="readme-overdrive",
        headline="README vs measured (delta = measured − readme)",
    )


def _row_sort_key(r: dict[str, str]) -> tuple:
    return (
        r["dataset"],
        r["format"],
        _res_sort_key(r["res"]),
        r["metric"],
        _delta_sort_key(r["delta"]),
    )


def _section_rule_width(variant: str) -> int:
    return max(56, min(88, len(variant) + 10))


def _print_section_header(variant: str) -> None:
    w = _section_rule_width(variant)
    rule = "═" * w
    print(rule)
    print(variant)
    print(rule)


def _print_table(
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
    *,
    indent: str = "",
) -> None:
    if not rows:
        print(f"{indent}(no rows)")
        return
    widths = {c: len(c) for c in columns}
    for r in rows:
        for c in columns:
            widths[c] = max(widths[c], len(r.get(c, "")))
    sep = "  "
    pad = indent
    header = pad + sep.join(c.ljust(widths[c]) for c in columns)
    rule = pad + sep.join("─" * widths[c] for c in columns)
    print(header)
    print(rule)
    for r in rows:
        print(pad + sep.join(r.get(c, "").ljust(widths[c]) for c in columns))


def print_comparison_report(
    result: ComparisonResult,
    *,
    delta_rows_before_filter: int | None = None,
    min_abs_delta_pct: float | None = None,
) -> None:
    print(result.headline)
    if min_abs_delta_pct is not None and delta_rows_before_filter is not None:
        print(
            f"(filtered: |Δ%| ≥ {min_abs_delta_pct:g}% — "
            f"{len(result.delta_rows)} of {delta_rows_before_filter} row(s))"
        )
    print()

    if not result.delta_rows:
        print("No overlapping metric cells (nothing to tabulate).")
    else:
        by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in result.delta_rows:
            by_variant[r["model_variant"]].append(r)

        for i, variant in enumerate(sorted(by_variant.keys())):
            rows = by_variant[variant]
            rows.sort(key=_row_sort_key)
            if i:
                print()
            _print_section_header(variant)
            print()
            _print_table(rows, result.table_columns, indent="  ")

    print()
    parts = [
        f"{result.matched_rows} config(s) matched",
        f"{len(result.delta_rows)} metric cell(s) listed",
    ]
    if result.mode.startswith("readme"):
        parts.insert(
            0,
            f"{result.skipped_parsed_no_readme} parsed row(s) skipped (no readme metrics)",
        )
    print("Summary: " + "; ".join(parts) + ".")

    if result.duplicate_nominal_keys:
        print(
            f"Note: {len(result.duplicate_nominal_keys)} duplicate key(s) in nominal CSV "
            f"(first row kept): {', '.join(sorted(set(result.duplicate_nominal_keys)))}"
        )
    if result.duplicate_benchmark_keys:
        label = (
            "overdrive CSV"
            if result.mode == "nominal-overdrive"
            else "measured benchmark CSV"
        )
        print(
            f"Note: {len(result.duplicate_benchmark_keys)} duplicate key(s) in {label} "
            f"(first row kept): {', '.join(sorted(set(result.duplicate_benchmark_keys)))}"
        )
    if result.missing_in_benchmark:
        keys = ", ".join(sorted(set(result.missing_in_benchmark)))
        if result.mode == "nominal-overdrive":
            msg = (
                f"Note: {len(result.missing_in_benchmark)} nominal config(s) have no overdrive row "
                f"(compare missed; not in table): {keys}"
            )
        else:
            msg = (
                f"Note: {len(result.missing_in_benchmark)} readme config(s) with metrics have no measured row "
                f"(compare missed; not in table): {keys}"
            )
        print(msg)
    if result.missing_in_parsed:
        keys = ", ".join(sorted(set(result.missing_in_parsed)))
        if result.mode == "nominal-overdrive":
            msg = (
                f"Note: {len(result.missing_in_parsed)} overdrive row(s) have no matching nominal row "
                f"(compare missed; not in table): {keys}"
            )
        else:
            msg = (
                f"Note: {len(result.missing_in_parsed)} measured row(s) have no matching readme metrics row "
                f"(compare missed; not in table): {keys}"
            )
        print(msg)
    if result.missed_numeric_compare:
        print(
            f"Note: {result.missed_numeric_compare} metric cell(s) have no delta "
            f"(missing or non-numeric value on one side)."
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=COMPARE_MODES,
        default="readme-overdrive",
        help=(
            "readme-overdrive: benchmark_parsed vs overdrive results; "
            "readme-nominal: benchmark_parsed vs nominal results; "
            "nominal-overdrive: nominal vs overdrive benchmark_results only."
        ),
    )
    p.add_argument(
        "--parsed",
        type=Path,
        default=DEFAULT_PARSED_CSV,
        help=f"README-parsed CSV for readme-* modes (default: {DEFAULT_PARSED_CSV})",
    )
    p.add_argument(
        "--nominal",
        type=Path,
        default=DEFAULT_NOMINAL_CSV,
        help=f"Nominal benchmark_results.csv (default: {DEFAULT_NOMINAL_CSV})",
    )
    p.add_argument(
        "--overdrive",
        type=Path,
        default=DEFAULT_OVERDRIVE_CSV,
        help=f"Overdrive benchmark_results.csv (default: {DEFAULT_OVERDRIVE_CSV})",
    )
    p.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help=(
            "Alias for the measured CSV in readme-* modes only (overrides --overdrive for "
            "readme-overdrive, and --nominal for readme-nominal)."
        ),
    )
    p.add_argument(
        "--min-abs-delta-pct",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Hide metric rows whose numeric |delta_pct| is below PCT (e.g. 5 keeps |Δ|≥5%%). "
            "Rows without a numeric Δ%% are always shown."
        ),
    )
    args = p.parse_args(argv)

    if args.mode == "readme-overdrive":
        bench = args.benchmark if args.benchmark is not None else args.overdrive
        if not args.parsed.is_file():
            print(f"error: parsed CSV not found: {args.parsed}", file=sys.stderr)
            return 2
        if not bench.is_file():
            print(f"error: overdrive/measured CSV not found: {bench}", file=sys.stderr)
            return 2
        result = compare_readme_to_bench(
            args.parsed,
            bench,
            mode="readme-overdrive",
            headline="README vs overdrive (delta = measured − readme)",
        )
    elif args.mode == "readme-nominal":
        bench = args.benchmark if args.benchmark is not None else args.nominal
        if not args.parsed.is_file():
            print(f"error: parsed CSV not found: {args.parsed}", file=sys.stderr)
            return 2
        if not bench.is_file():
            print(f"error: nominal/measured CSV not found: {bench}", file=sys.stderr)
            return 2
        result = compare_readme_to_bench(
            args.parsed,
            bench,
            mode="readme-nominal",
            headline="README vs nominal (delta = measured − readme)",
        )
    else:
        if args.benchmark is not None:
            print(
                "error: --benchmark is only valid for readme-overdrive or readme-nominal",
                file=sys.stderr,
            )
            return 2
        if not args.nominal.is_file():
            print(f"error: nominal CSV not found: {args.nominal}", file=sys.stderr)
            return 2
        if not args.overdrive.is_file():
            print(f"error: overdrive CSV not found: {args.overdrive}", file=sys.stderr)
            return 2
        result = compare_nominal_to_overdrive(args.nominal, args.overdrive)

    before_filter: int | None = None
    if args.min_abs_delta_pct is not None:
        if args.min_abs_delta_pct < 0:
            print("error: --min-abs-delta-pct must be >= 0", file=sys.stderr)
            return 2
        before_filter = len(result.delta_rows)
        result.delta_rows = filter_rows_by_abs_delta_pct(
            result.delta_rows,
            args.min_abs_delta_pct,
        )

    print_comparison_report(
        result,
        delta_rows_before_filter=before_filter,
        min_abs_delta_pct=args.min_abs_delta_pct,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
