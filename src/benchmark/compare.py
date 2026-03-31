"""Compare two datasources (README-parsed metrics and/or benchmark result CSVs).

The official ST Model Zoo NPU figures parsed from object_detection README tables are
measured in **overdrive mode** as part of the **default STM32Cube.AI configuration**
(see each family README under *Performances → Metrics*, e.g. input/output allocated).
That matches the high-performance supply/setup used for STM32N6570-DK reference
benchmarking, not a separate “underdrive vs overdrive” split on the README side.

Datasources (select via ``--left/--right``):

- **readme**: parsed CSV (default ``results/benchmark_parsed.csv``)
- **underdrive**: benchmark CSV (default ``results/benchmark_underdrive/benchmark_results.csv``)
- **nominal**: benchmark CSV (default ``results/benchmark_nominal/benchmark_results.csv``)
- **overdrive**: benchmark CSV (default ``results/benchmark_overdrive/benchmark_results.csv``)

Delta is always defined as ``right − left``.

For readme comparisons, empty README metric cells are ignored (not compared). For
``stedgeai_version``, values are compared as strings (delta column shows ``≠`` when they
differ), and rows with the same value on both sides are omitted.

Output is grouped by ``(model_variant, format)`` with plain-text tables (no pass/fail).

Use ``--delta-pct PCT`` to hide metric rows whose numeric ``|delta_pct|`` is below PCT. Rows
without a numeric ``delta_pct`` (e.g. baseline zero) stay listed; matching ``stedgeai_version``
rows are omitted.

Use ``--delta METRIC:THRESH`` (repeatable) to keep only listed metrics whose numeric ``|Δ|`` is
at least ``THRESH`` (raw delta, not percent). Rows without a numeric ``Δ`` stay listed; empty
``stedgeai_version`` match rows are omitted (same as ``--delta-pct``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from .constants import CSV_COLUMNS, CSV_COLUMNS_NO_POWER
from .paths import RESULTS_DIR
from .utils.logutil import configure_logging, typer_install_exception_hook

DEFAULT_PARSED_CSV = RESULTS_DIR / "benchmark_parsed.csv"
DEFAULT_UNDERDRIVE_CSV = RESULTS_DIR / "benchmark_underdrive" / "benchmark_results.csv"
DEFAULT_NOMINAL_CSV = RESULTS_DIR / "benchmark_nominal" / "benchmark_results.csv"
DEFAULT_OVERDRIVE_CSV = RESULTS_DIR / "benchmark_overdrive" / "benchmark_results.csv"
DEFAULT_EVALUATE_CSV = RESULTS_DIR / "evaluation_result.csv"

IDENTITY_COLS = (
    "model_family",
    "model_variant",
    "hyperparameters",
    "dataset",
    "format",
    "resolution",
)

KEY_MERGE = (
    "model_family",
    "model_variant",
    "hyperparameters",
    "dataset",
    "format",
    "_res",
)

_COMPARE_METADATA_COLS = frozenset(
    ("host_time_iso", "stedgeai_version", "cpu_mhz", "npu_mhz")
)

METRIC_COLS = tuple(
    c
    for c in CSV_COLUMNS_NO_POWER
    if c not in IDENTITY_COLS and c not in _COMPARE_METADATA_COLS
)

_BENCH_PAIR_OMIT_POWER = frozenset(
    (
        "pm_avg_idle_mW",
        "pm_avg_idle_ms",
        "pm_avg_idle_mJ",
        "pm_avg_delta_mW",
    )
)
METRIC_COLS_BENCH_PAIR = tuple(
    c
    for c in CSV_COLUMNS
    if c not in IDENTITY_COLS
    and c not in _COMPARE_METADATA_COLS
    and c not in _BENCH_PAIR_OMIT_POWER
)

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
    "underdrive",
    "overdrive",
    "delta",
    "delta_pct",
)


_DATASOURCE_CHOICES = ("readme", "underdrive", "nominal", "overdrive", "evaluate")


def _ds_norm(s: str) -> str:
    return (s or "").strip().casefold()


def _ds_error_choices() -> str:
    return ", ".join(_DATASOURCE_CHOICES)


def _require_file_exists(path: Path, *, label: str) -> None:
    if not path.is_file():
        _err_console.print(f"[red]error: {label} CSV not found: {path}[/red]")
        raise typer.Exit(2)


def _bench_path_for_ds(ds: str, ds_paths: dict[str, Path]) -> Path:
    dsn = _ds_norm(ds)
    if dsn in ds_paths:
        return ds_paths[dsn]
    raise ValueError(f"not a benchmark datasource: {ds!r}")


def _columns_for_readme_pair(left_label: str, right_label: str) -> tuple[str, ...]:
    return (
        "dataset",
        "format",
        "res",
        "metric",
        left_label,
        right_label,
        "delta",
        "delta_pct",
    )


def _columns_for_bench_pair(left_label: str, right_label: str) -> tuple[str, ...]:
    return (
        "dataset",
        "format",
        "res",
        "metric",
        left_label,
        right_label,
        "delta",
        "delta_pct",
    )


@dataclass
class ComparisonResult:
    left: str = ""
    right: str = ""
    headline: str = ""
    table_columns: tuple[str, ...] = field(default_factory=lambda: PER_VARIANT_COLUMNS_README)
    skipped_parsed_no_readme: int = 0
    matched_rows: int = 0
    missing_in_right: list[str] = field(default_factory=list)
    missing_in_left: list[str] = field(default_factory=list)
    duplicate_left_keys: list[str] = field(default_factory=list)
    duplicate_right_keys: list[str] = field(default_factory=list)
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
    s = (pct_str or "").strip()
    if not s or s == "—":
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        v = float(s)
        if v != v:
            return None
        return v
    except ValueError:
        return None


def _parse_delta_abs_value(delta_str: str) -> float | None:
    s = (delta_str or "").strip()
    if not s or s == "—" or s == "≠":
        return None
    try:
        v = float(s)
        if v != v:
            return None
        return v
    except ValueError:
        return None


def _parse_delta_cli_items(
    items: list[str],
) -> tuple[list[tuple[str, float]], dict[str, float]] | str:
    """Returns (ordered display pairs, metric_casefold → min |Δ| threshold) or an error message."""
    ordered: list[tuple[str, float]] = []
    thresh_by_metric: dict[str, float] = {}
    for raw in items:
        s = (raw or "").strip()
        if not s:
            continue
        if ":" not in s:
            return f"invalid --delta {raw!r}; expected METRIC:NUMBER"
        name, num_s = s.rsplit(":", 1)
        name = name.strip()
        num_s = num_s.strip()
        if not name:
            return f"invalid --delta {raw!r}; metric name before ':' is empty"
        if not num_s:
            return f"invalid --delta {raw!r}; threshold after ':' is empty"
        try:
            thr = float(num_s)
        except ValueError:
            return f"invalid --delta threshold in {raw!r}; expected a number"
        if thr < 0:
            return f"invalid --delta {raw!r}; threshold must be >= 0"
        key = name.casefold()
        ordered.append((name, thr))
        thresh_by_metric[key] = max(thresh_by_metric.get(key, 0.0), thr)
    return ordered, thresh_by_metric


def filter_rows_by_delta_spec(
    rows: list[dict[str, str]],
    thresh_by_metric: dict[str, float],
) -> list[dict[str, str]]:
    if not thresh_by_metric:
        return rows
    out: list[dict[str, str]] = []
    for r in rows:
        m = (r.get("metric") or "").strip().casefold()
        if m not in thresh_by_metric:
            continue
        if r.get("metric") == "stedgeai_version" and not (r.get("delta") or "").strip():
            continue
        thr = thresh_by_metric[m]
        v = _parse_delta_abs_value(r.get("delta", ""))
        if v is None or abs(v) >= thr:
            out.append(r)
    return out


def filter_rows_by_abs_delta_pct(
    rows: list[dict[str, str]],
    min_abs_pct: float,
) -> list[dict[str, str]]:
    if min_abs_pct < 0:
        raise ValueError("min_abs_pct must be non-negative")
    out: list[dict[str, str]] = []
    for r in rows:
        if r.get("metric") == "stedgeai_version" and not (r.get("delta") or "").strip():
            continue
        v = _parse_delta_pct_value(r.get("delta_pct", ""))
        if v is None or abs(v) >= min_abs_pct:
            out.append(r)
    return out


def filter_rows_by_metric(
    rows: list[dict[str, str]],
    metrics: list[str],
) -> list[dict[str, str]]:
    want = {(m or "").strip().casefold() for m in metrics if (m or "").strip()}
    if not want:
        return rows
    return [r for r in rows if (r.get("metric") or "").strip().casefold() in want]


def load_csv_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if df.shape[1] == 0 or df.columns[0] == "":
        raise SystemExit(f"No header in {path}")
    return df


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Backward-compatible CSV loader (dict rows)."""
    df = load_csv_df(path)
    return list(df.columns), df.to_dict("records")


def _normalize_identity_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["model_family", "model_variant", "hyperparameters", "dataset", "format"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip()
    if "resolution" not in out.columns:
        out["resolution"] = ""
    out["_res"] = out["resolution"].map(lambda v: str(_resolution_int(v)))
    return out


def _row_dict_for_metrics(r: pd.Series, original_columns: list[str]) -> dict[str, str]:
    d: dict[str, str] = {}
    idx = r.index
    for c in original_columns:
        if c in idx:
            v = r[c]
            d[c] = "" if pd.isna(v) else str(v)
        else:
            d[c] = ""
    return d


def _append_duplicate_labels(
    size_by_key: pd.Series,
    out: ComparisonResult,
    *,
    left: bool,
) -> None:
    for key_tuple, cnt in size_by_key.items():
        if cnt <= 1:
            continue
        kt = tuple(str(x) for x in (key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)))
        lbl = key_label(kt)
        dup_list = out.duplicate_left_keys if left else out.duplicate_right_keys
        dup_list.extend([lbl] * (cnt - 1))


def _merged_side(row: pd.Series, col: str, left: bool) -> str:
    tag = "_l" if left else "_r"
    for key in (f"{col}{tag}", col):
        if key in row.index:
            v = row[key]
            if pd.isna(v):
                return ""
            return str(v).strip()
    return ""


def compare_readme_to_bench(
    parsed_path: Path,
    bench_path: Path,
    *,
    left_label: str,
    right_label: str,
    left_is_readme: bool,
    headline: str,
) -> ComparisonResult:
    parsed_df = load_csv_df(parsed_path)
    bench_df = load_csv_df(bench_path)
    parsed_cols = list(parsed_df.columns)

    out = ComparisonResult(
        left=left_label,
        right=right_label,
        headline=headline,
        table_columns=_columns_for_readme_pair(left_label, right_label),
    )

    parsed_n = _normalize_identity_df(parsed_df)
    bench_n = _normalize_identity_df(bench_df)

    g_sizes = bench_n.groupby(list(KEY_MERGE), sort=False).size()
    _append_duplicate_labels(g_sizes, out, left=False)

    bench_first = bench_n.drop_duplicates(subset=list(KEY_MERGE), keep="first")

    parsed_rows_kept: list[pd.Series] = []
    for _, r in parsed_n.iterrows():
        rd = _row_dict_for_metrics(r, parsed_cols)
        if not parsed_row_has_any_metric(rd):
            out.skipped_parsed_no_readme += 1
            continue
        parsed_rows_kept.append(r)

    if not parsed_rows_kept:
        parsed_with = pd.DataFrame(columns=parsed_n.columns)
    else:
        parsed_with = pd.DataFrame(parsed_rows_kept)

    parsed_key_set: set[tuple[str, ...]] = set()
    if not parsed_with.empty:
        for _, r in parsed_with.iterrows():
            parsed_key_set.add(tuple(str(r[c]) for c in KEY_MERGE))

    left_df_m, right_df_m = (parsed_with, bench_first) if left_is_readme else (bench_first, parsed_with)
    left_m = pd.merge(
        left_df_m,
        right_df_m,
        on=list(KEY_MERGE),
        how="left",
        indicator=True,
        suffixes=("_l", "_r"),
    )

    out.matched_rows = int((left_m["_merge"] == "both").sum())

    for _, rrow in left_m[left_m["_merge"] == "left_only"].iterrows():
        k = tuple(str(rrow[c]) for c in KEY_MERGE)
        out.missing_in_right.append(key_label(k))

    if left_is_readme:
        for _, rrow in bench_first.iterrows():
            k = tuple(str(rrow[c]) for c in KEY_MERGE)
            if k not in parsed_key_set:
                out.missing_in_left.append(key_label(k))
    else:
        bench_key_set = {tuple(str(rrow[c]) for c in KEY_MERGE) for _, rrow in bench_first.iterrows()}
        for k in parsed_key_set:
            if k not in bench_key_set:
                out.missing_in_left.append(key_label(k))

    delta_src = left_m[left_m["_merge"] == "both"].drop(columns=["_merge"], errors="ignore")
    for _, mrow in delta_src.iterrows():
        res_s = str(mrow["_res"])
        for col in METRIC_COLS:
            lraw = _merged_side(mrow, col, True)
            rraw = _merged_side(mrow, col, False)
            readme_raw = lraw if left_is_readme else rraw
            if _blank(readme_raw):
                continue
            pr, br = metric_floats(col, lraw, rraw)
            if pr is None or br is None:
                out.missed_numeric_compare += 1
            d_str, pct_str = _delta_cells(pr, br)
            out.delta_rows.append(
                {
                    "model_variant": _merged_side(mrow, "model_variant", True),
                    "dataset": _merged_side(mrow, "dataset", True),
                    "format": _merged_side(mrow, "format", True),
                    "res": res_s,
                    "metric": col,
                    left_label: _fmt_num(pr) if pr is not None else lraw.strip(),
                    right_label: _fmt_num(br) if br is not None else rraw.strip(),
                    "delta": d_str,
                    "delta_pct": pct_str,
                }
            )

        lv = _merged_side(mrow, "stedgeai_version", True)
        rv = _merged_side(mrow, "stedgeai_version", False)
        if (lv or rv) and lv != rv:
            out.delta_rows.append(
                {
                    "model_variant": _merged_side(mrow, "model_variant", True),
                    "dataset": _merged_side(mrow, "dataset", True),
                    "format": _merged_side(mrow, "format", True),
                    "res": res_s,
                    "metric": "stedgeai_version",
                    left_label: lv,
                    right_label: rv,
                    "delta": "≠",
                    "delta_pct": "—",
                }
            )

    return out


def compare_bench_to_bench(
    left_label: str,
    left_path: Path,
    right_label: str,
    right_path: Path,
) -> ComparisonResult:
    left_df = load_csv_df(left_path)
    right_df = load_csv_df(right_path)

    out = ComparisonResult(
        left=left_label,
        right=right_label,
        headline=f"{left_label} vs {right_label} (delta = {right_label} − {left_label})",
        table_columns=_columns_for_bench_pair(left_label, right_label),
    )

    left_n = _normalize_identity_df(left_df)
    right_n = _normalize_identity_df(right_df)

    _append_duplicate_labels(
        left_n.groupby(list(KEY_MERGE), sort=False).size(),
        out,
        left=True,
    )
    _append_duplicate_labels(
        right_n.groupby(list(KEY_MERGE), sort=False).size(),
        out,
        left=False,
    )

    left_first = left_n.drop_duplicates(subset=list(KEY_MERGE), keep="first")
    right_first = right_n.drop_duplicates(subset=list(KEY_MERGE), keep="first")

    left_key_set = {tuple(str(r[c]) for c in KEY_MERGE) for _, r in left_first.iterrows()}

    left_m = pd.merge(
        left_first,
        right_first,
        on=list(KEY_MERGE),
        how="left",
        indicator=True,
        suffixes=("_l", "_r"),
    )

    out.matched_rows = int((left_m["_merge"] == "both").sum())

    for _, r in left_m[left_m["_merge"] == "left_only"].iterrows():
        k = tuple(str(r[c]) for c in KEY_MERGE)
        out.missing_in_right.append(key_label(k))

    for _, r in right_first.iterrows():
        k = tuple(str(r[c]) for c in KEY_MERGE)
        if k not in left_key_set:
            out.missing_in_left.append(key_label(k))

    delta_src = left_m[left_m["_merge"] == "both"].drop(columns=["_merge"], errors="ignore")
    for _, mrow in delta_src.iterrows():
        res_s = str(mrow["_res"])
        for col in METRIC_COLS_BENCH_PAIR:
            lraw = _merged_side(mrow, col, True)
            rraw = _merged_side(mrow, col, False)
            if _blank(lraw) and _blank(rraw):
                continue
            pr, br = metric_floats(col, lraw, rraw)
            if pr is None or br is None:
                out.missed_numeric_compare += 1
            d_str, pct_str = _delta_cells(pr, br)
            out.delta_rows.append(
                {
                    "model_variant": _merged_side(mrow, "model_variant", True),
                    "dataset": _merged_side(mrow, "dataset", True),
                    "format": _merged_side(mrow, "format", True),
                    "res": res_s,
                    "metric": col,
                    left_label: _fmt_num(pr) if pr is not None else lraw.strip(),
                    right_label: _fmt_num(br) if br is not None else rraw.strip(),
                    "delta": d_str,
                    "delta_pct": pct_str,
                }
            )

        lv = _merged_side(mrow, "stedgeai_version", True)
        rv = _merged_side(mrow, "stedgeai_version", False)
        if (lv or rv) and lv != rv:
            out.delta_rows.append(
                {
                    "model_variant": _merged_side(mrow, "model_variant", True),
                    "dataset": _merged_side(mrow, "dataset", True),
                    "format": _merged_side(mrow, "format", True),
                    "res": res_s,
                    "metric": "stedgeai_version",
                    left_label: lv,
                    right_label: rv,
                    "delta": "≠",
                    "delta_pct": "—",
                }
            )

    return out


def _row_sort_key(r: dict[str, str]) -> tuple:
    return (
        r["dataset"],
        r["format"],
        _res_sort_key(r["res"]),
        r["metric"],
        _delta_sort_key(r["delta"]),
    )


def print_comparison_report(
    result: ComparisonResult,
    *,
    delta_rows_before_filter: int | None = None,
    delta_pct: float | None = None,
    delta_abs_before: int | None = None,
    delta_abs_desc: str | None = None,
) -> None:
    console = Console()
    console.print(result.headline)
    if delta_abs_desc is not None and delta_abs_before is not None:
        console.print(
            f"(filtered: {delta_abs_desc} — "
            f"{len(result.delta_rows)} of {delta_abs_before} row(s))"
        )
    if delta_pct is not None and delta_rows_before_filter is not None:
        console.print(
            f"(filtered: |Δ%| ≥ {delta_pct:g}% — "
            f"{len(result.delta_rows)} of {delta_rows_before_filter} row(s))"
        )
    console.print()

    if not result.delta_rows:
        console.print("No overlapping metric cells (nothing to tabulate).")
    else:
        by_variant_and_format: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for r in result.delta_rows:
            by_variant_and_format[(r["model_variant"], r["format"])].append(r)

        for i, (variant, fmt) in enumerate(sorted(by_variant_and_format.keys())):
            rows = by_variant_and_format[(variant, fmt)]
            rows.sort(key=_row_sort_key)
            if i:
                console.print()
            console.print(Rule(title=f"{variant} | {fmt}", style="bold"))
            console.print()
            table = Table(show_header=True, header_style="bold", show_lines=False)
            for c in result.table_columns:
                col_kw = {"overflow": "fold", "no_wrap": c == "metric"}
                if c == "metric":
                    col_kw["min_width"] = 22
                table.add_column(c, **col_kw)
            for r in rows:
                table.add_row(*[r.get(c, "") for c in result.table_columns])
            console.print(table)

    console.print()
    parts = [
        f"{result.matched_rows} config(s) matched",
        f"{len(result.delta_rows)} metric cell(s) listed",
    ]
    if result.left == "readme" or result.right == "readme":
        parts.insert(
            0,
            f"{result.skipped_parsed_no_readme} parsed row(s) skipped (no readme metrics)",
        )
    console.print("Summary: " + "; ".join(parts) + ".")

    if result.duplicate_left_keys:
        console.print(
            f"Note: {len(result.duplicate_left_keys)} duplicate key(s) in {result.left} CSV "
            f"(first row kept): {', '.join(sorted(set(result.duplicate_left_keys)))}"
        )
    if result.duplicate_right_keys:
        console.print(
            f"Note: {len(result.duplicate_right_keys)} duplicate key(s) in {result.right} CSV "
            f"(first row kept): {', '.join(sorted(set(result.duplicate_right_keys)))}"
        )
    if result.missing_in_right:
        keys = ", ".join(sorted(set(result.missing_in_right)))
        msg = (
            f"Note: {len(result.missing_in_right)} {result.left} config(s) have no {result.right} row "
            f"(compare missed; not in table): {keys}"
        )
        console.print(msg)
    if result.missing_in_left:
        keys = ", ".join(sorted(set(result.missing_in_left)))
        msg = (
            f"Note: {len(result.missing_in_left)} {result.right} row(s) have no matching {result.left} row "
            f"(compare missed; not in table): {keys}"
        )
        console.print(msg)
    if result.missed_numeric_compare:
        console.print(
            f"Note: {result.missed_numeric_compare} metric cell(s) have no delta "
            f"(missing or non-numeric value on one side)."
        )


_err_console = Console(stderr=True)


compare_app = typer.Typer(
    help=__doc__,
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)


@compare_app.callback()
def compare_entry(
    ctx: typer.Context,
    left: str = typer.Option(
        "readme",
        "-l",
        "--left",
        help=f"Datasource ({_ds_error_choices()})",
    ),
    right: str = typer.Option(
        "overdrive",
        "-r",
        "--right",
        help=f"Datasource ({_ds_error_choices()})",
    ),
    readme: Path = typer.Option(
        DEFAULT_PARSED_CSV,
        "--readme",
        help=f"README-parsed CSV (default: {DEFAULT_PARSED_CSV})",
    ),
    nominal: Path = typer.Option(
        DEFAULT_NOMINAL_CSV,
        "--nominal",
        help=f"Nominal benchmark_results.csv (default: {DEFAULT_NOMINAL_CSV})",
    ),
    underdrive: Path = typer.Option(
        DEFAULT_UNDERDRIVE_CSV,
        "--underdrive",
        help=f"Underdrive benchmark_results.csv (default: {DEFAULT_UNDERDRIVE_CSV})",
    ),
    overdrive: Path = typer.Option(
        DEFAULT_OVERDRIVE_CSV,
        "--overdrive",
        help=f"Overdrive benchmark_results.csv (default: {DEFAULT_OVERDRIVE_CSV})",
    ),
    evaluate: Path = typer.Option(
        DEFAULT_EVALUATE_CSV,
        "--evaluate",
        help=f"Evaluation result CSV (default: {DEFAULT_EVALUATE_CSV})",
    ),
    delta_pct: float | None = typer.Option(
        None,
        "--delta-pct",
        help=(
            "Hide metric rows whose numeric |delta_pct| is below PCT; rows without numeric Δ%% stay; "
            "matching stedgeai_version rows omitted."
        ),
    ),
    delta_metric: list[str] = typer.Option(
        [],
        "--delta",
        help=(
            "Keep only these metric(s) with numeric |Δ| ≥ THRESH; format METRIC:THRESH (raw delta). "
            "Repeat for multiple. Rows without numeric Δ stay; empty stedgeai_version rows omitted."
        ),
    ),
    metric: list[str] = typer.Option(
        [],
        "--metric",
        help="Only include these metric(s). Repeat flag to include multiple (case-insensitive exact match).",
    ),
) -> None:
    """Compare any two datasources (delta = right − left)."""
    if getattr(ctx, "resilient_parsing", False):
        return
    l = _ds_norm(left)
    r = _ds_norm(right)
    if l not in _DATASOURCE_CHOICES:
        _err_console.print(
            f"[red]error: invalid --left {left!r}; expected one of: {_ds_error_choices()}[/red]"
        )
        raise typer.Exit(2)
    if r not in _DATASOURCE_CHOICES:
        _err_console.print(
            f"[red]error: invalid --right {right!r}; expected one of: {_ds_error_choices()}[/red]"
        )
        raise typer.Exit(2)
    if l == r:
        _err_console.print("[red]error: --left and --right must be different[/red]")
        raise typer.Exit(2)

    configure_logging()
    typer_install_exception_hook()

    ds_paths = dict(nominal=nominal, underdrive=underdrive, overdrive=overdrive, evaluate=evaluate)

    result: ComparisonResult | None = None
    before_filter: int | None = None
    before_delta_abs: int | None = None
    delta_abs_desc: str | None = None
    thresh_by_metric: dict[str, float] = {}

    if delta_metric:
        parsed = _parse_delta_cli_items(delta_metric)
        if isinstance(parsed, str):
            _err_console.print(f"[red]error: {parsed}[/red]")
            raise typer.Exit(2)
        ordered_spec, thresh_by_metric = parsed
        if not thresh_by_metric:
            _err_console.print("[red]error: --delta requires at least one METRIC:NUMBER[/red]")
            raise typer.Exit(2)
        delta_abs_desc = "; ".join(f"{n} |Δ| ≥ {t:g}" for n, t in ordered_spec)

    if l == "readme" or r == "readme":
        _require_file_exists(readme, label="--readme")
        if l == "readme":
            bench_path = _bench_path_for_ds(r, ds_paths)
            _require_file_exists(bench_path, label=r)
            result = compare_readme_to_bench(
                readme,
                bench_path,
                left_label=l,
                right_label=r,
                left_is_readme=True,
                headline=f"{l} vs {r} (delta = {r} − {l})",
            )
        else:
            bench_path = _bench_path_for_ds(l, ds_paths)
            _require_file_exists(bench_path, label=l)
            result = compare_readme_to_bench(
                readme,
                bench_path,
                left_label=l,
                right_label=r,
                left_is_readme=False,
                headline=f"{l} vs {r} (delta = {r} − {l})",
            )
    else:
        left_path = _bench_path_for_ds(l, ds_paths)
        right_path = _bench_path_for_ds(r, ds_paths)
        _require_file_exists(left_path, label=l)
        _require_file_exists(right_path, label=r)
        result = compare_bench_to_bench(l, left_path, r, right_path)

    assert result is not None
    if metric:
        result.delta_rows = filter_rows_by_metric(result.delta_rows, metric)
    if thresh_by_metric:
        before_delta_abs = len(result.delta_rows)
        result.delta_rows = filter_rows_by_delta_spec(result.delta_rows, thresh_by_metric)
    if delta_pct is not None:
        if delta_pct < 0:
            _err_console.print("[red]error: --delta-pct must be >= 0[/red]")
            raise typer.Exit(2)
        before_filter = len(result.delta_rows)
        result.delta_rows = filter_rows_by_abs_delta_pct(result.delta_rows, delta_pct)

    print_comparison_report(
        result,
        delta_rows_before_filter=before_filter,
        delta_pct=delta_pct,
        delta_abs_before=before_delta_abs,
        delta_abs_desc=delta_abs_desc,
    )


def compare_main(argv: list[str] | None = None) -> int:
    configure_logging()
    typer_install_exception_hook()
    args = [] if argv is None else argv
    try:
        compare_app(args=args)
    except typer.Exit as e:
        return int(e.exit_code) if e.exit_code is not None else 0
    return 0
