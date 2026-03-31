"""Compare benchmark / README metrics with configurable modes.

The official ST Model Zoo NPU figures parsed from object_detection README tables are
measured in **overdrive mode** as part of the **default STM32Cube.AI configuration**
(see each family README under *Performances → Metrics*, e.g. input/output allocated).
That matches the high-performance supply/setup used for STM32N6570-DK reference
benchmarking, not a separate “underdrive vs overdrive” split on the README side.

Modes (see ``--mode``):

- **readme-overdrive** — ``benchmark_parsed.csv`` vs ``benchmark_overdrive/benchmark_results.csv``
  (delta = measured − readme). Parsed rows with no metrics are skipped; empty parsed
  cells are not compared.

- **readme-underdrive** — same, but measured file is ``benchmark_underdrive/benchmark_results.csv``.

- **underdrive-overdrive** — underdrive vs overdrive ``benchmark_results.csv`` (delta = overdrive − underdrive)
  for every shared config and metric column, **including inference power** (``pm_avg_inf_*``) when present;
  ``pm_avg_idle_*`` and ``pm_avg_delta_mW`` are omitted (idle/delta splits are not comparable across runs).
  ``stedgeai_version`` is listed when present on either side.

For readme modes, ``stedgeai_version`` is also listed per matched row (README-parsed vs measured
``benchmark_results``); values are compared as strings (delta column shows ``≠`` when they differ).

Output is grouped by ``model_variant`` with plain-text tables (no pass/fail).

Use ``--delta-pct PCT`` to hide metric rows whose numeric ``|delta_pct|`` is below PCT. Rows
without a numeric ``delta_pct`` (e.g. baseline zero) stay listed; ``stedgeai_version`` rows with
the same value on both sides are omitted.
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
from .paths import BASE_DIR
from .utils.logutil import configure_logging, typer_install_exception_hook


RESULTS_DIR = BASE_DIR / "results"
DEFAULT_PARSED_CSV = RESULTS_DIR / "benchmark_parsed.csv"
DEFAULT_UNDERDRIVE_CSV = RESULTS_DIR / "benchmark_underdrive" / "benchmark_results.csv"
DEFAULT_OVERDRIVE_CSV = RESULTS_DIR / "benchmark_overdrive" / "benchmark_results.csv"
DEFAULT_BENCHMARK_CSV = DEFAULT_OVERDRIVE_CSV

COMPARE_MODES = ("readme-overdrive", "readme-underdrive", "underdrive-overdrive")

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
    duplicate_underdrive_keys: list[str] = field(default_factory=list)
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
    underdrive: bool,
) -> None:
    for key_tuple, cnt in size_by_key.items():
        if cnt <= 1:
            continue
        kt = tuple(str(x) for x in (key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)))
        lbl = key_label(kt)
        dup_list = out.duplicate_underdrive_keys if underdrive else out.duplicate_benchmark_keys
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
    mode: str,
    headline: str,
) -> ComparisonResult:
    parsed_df = load_csv_df(parsed_path)
    bench_df = load_csv_df(bench_path)
    parsed_cols = list(parsed_df.columns)

    out = ComparisonResult(
        mode=mode,
        headline=headline,
        table_columns=PER_VARIANT_COLUMNS_README,
    )

    parsed_n = _normalize_identity_df(parsed_df)
    bench_n = _normalize_identity_df(bench_df)

    g_sizes = bench_n.groupby(list(KEY_MERGE), sort=False).size()
    _append_duplicate_labels(g_sizes, out, underdrive=False)

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

    left_m = pd.merge(
        parsed_with,
        bench_first,
        on=list(KEY_MERGE),
        how="left",
        indicator=True,
        suffixes=("_l", "_r"),
    )

    out.matched_rows = int((left_m["_merge"] == "both").sum())

    for _, r in left_m[left_m["_merge"] == "left_only"].iterrows():
        k = tuple(str(r[c]) for c in KEY_MERGE)
        out.missing_in_benchmark.append(key_label(k))

    for _, r in bench_first.iterrows():
        k = tuple(str(r[c]) for c in KEY_MERGE)
        if k not in parsed_key_set:
            out.missing_in_parsed.append(key_label(k))

    delta_src = left_m[left_m["_merge"] == "both"].drop(columns=["_merge"], errors="ignore")
    for _, mrow in delta_src.iterrows():
        res_s = str(mrow["_res"])
        for col in METRIC_COLS:
            praw = _merged_side(mrow, col, True)
            if _blank(praw):
                continue
            braw = _merged_side(mrow, col, False)
            pr, br = metric_floats(col, praw, braw)
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
                    "readme": _fmt_num(pr) if pr is not None else praw.strip(),
                    "measured": _fmt_num(br) if br is not None else "",
                    "delta": d_str,
                    "delta_pct": pct_str,
                }
            )
        pv = _merged_side(mrow, "stedgeai_version", True)
        bv = _merged_side(mrow, "stedgeai_version", False)
        if pv or bv:
            out.delta_rows.append(
                {
                    "model_variant": _merged_side(mrow, "model_variant", True),
                    "dataset": _merged_side(mrow, "dataset", True),
                    "format": _merged_side(mrow, "format", True),
                    "res": res_s,
                    "metric": "stedgeai_version",
                    "readme": pv,
                    "measured": bv,
                    "delta": "" if pv == bv else "≠",
                    "delta_pct": "—",
                }
            )

    return out


def compare_underdrive_to_overdrive(
    underdrive_path: Path, overdrive_path: Path
) -> ComparisonResult:
    underdrive_df = load_csv_df(underdrive_path)
    over_df = load_csv_df(overdrive_path)

    out = ComparisonResult(
        mode="underdrive-overdrive",
        headline="Underdrive vs overdrive (delta = overdrive − underdrive)",
        table_columns=PER_VARIANT_COLUMNS_BENCH_PAIR,
    )

    underdrive_n = _normalize_identity_df(underdrive_df)
    over_n = _normalize_identity_df(over_df)

    _append_duplicate_labels(
        underdrive_n.groupby(list(KEY_MERGE), sort=False).size(),
        out,
        underdrive=True,
    )
    _append_duplicate_labels(
        over_n.groupby(list(KEY_MERGE), sort=False).size(),
        out,
        underdrive=False,
    )

    underdrive_first = underdrive_n.drop_duplicates(subset=list(KEY_MERGE), keep="first")
    over_first = over_n.drop_duplicates(subset=list(KEY_MERGE), keep="first")

    underdrive_key_set = {
        tuple(str(r[c]) for c in KEY_MERGE) for _, r in underdrive_first.iterrows()
    }

    left_m = pd.merge(
        underdrive_first,
        over_first,
        on=list(KEY_MERGE),
        how="left",
        indicator=True,
        suffixes=("_l", "_r"),
    )

    out.matched_rows = int((left_m["_merge"] == "both").sum())

    for _, r in left_m[left_m["_merge"] == "left_only"].iterrows():
        k = tuple(str(r[c]) for c in KEY_MERGE)
        out.missing_in_benchmark.append(key_label(k))

    for _, r in over_first.iterrows():
        k = tuple(str(r[c]) for c in KEY_MERGE)
        if k not in underdrive_key_set:
            out.missing_in_parsed.append(key_label(k))

    delta_src = left_m[left_m["_merge"] == "both"].drop(columns=["_merge"], errors="ignore")
    for _, mrow in delta_src.iterrows():
        res_s = str(mrow["_res"])
        for col in METRIC_COLS_BENCH_PAIR:
            nraw = _merged_side(mrow, col, True)
            oraw = _merged_side(mrow, col, False)
            if _blank(nraw) and _blank(oraw):
                continue
            pr, br = metric_floats(col, nraw, oraw)
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
                    "underdrive": _fmt_num(pr) if pr is not None else nraw.strip(),
                    "overdrive": _fmt_num(br) if br is not None else oraw.strip(),
                    "delta": d_str,
                    "delta_pct": pct_str,
                }
            )
        nv = _merged_side(mrow, "stedgeai_version", True)
        ov = _merged_side(mrow, "stedgeai_version", False)
        if nv or ov:
            out.delta_rows.append(
                {
                    "model_variant": _merged_side(mrow, "model_variant", True),
                    "dataset": _merged_side(mrow, "dataset", True),
                    "format": _merged_side(mrow, "format", True),
                    "res": res_s,
                    "metric": "stedgeai_version",
                    "underdrive": nv,
                    "overdrive": ov,
                    "delta": "" if nv == ov else "≠",
                    "delta_pct": "—",
                }
            )

    return out


def compare(parsed_path: Path, benchmark_path: Path) -> ComparisonResult:
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


def print_comparison_report(
    result: ComparisonResult,
    *,
    delta_rows_before_filter: int | None = None,
    delta_pct: float | None = None,
) -> None:
    console = Console()
    console.print(result.headline)
    if delta_pct is not None and delta_rows_before_filter is not None:
        console.print(
            f"(filtered: |Δ%| ≥ {delta_pct:g}% — "
            f"{len(result.delta_rows)} of {delta_rows_before_filter} row(s))"
        )
    console.print()

    if not result.delta_rows:
        console.print("No overlapping metric cells (nothing to tabulate).")
    else:
        by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in result.delta_rows:
            by_variant[r["model_variant"]].append(r)

        for i, variant in enumerate(sorted(by_variant.keys())):
            rows = by_variant[variant]
            rows.sort(key=_row_sort_key)
            if i:
                console.print()
            console.print(Rule(title=variant, style="bold"))
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
    if result.mode.startswith("readme"):
        parts.insert(
            0,
            f"{result.skipped_parsed_no_readme} parsed row(s) skipped (no readme metrics)",
        )
    console.print("Summary: " + "; ".join(parts) + ".")

    if result.duplicate_underdrive_keys:
        console.print(
            f"Note: {len(result.duplicate_underdrive_keys)} duplicate key(s) in underdrive CSV "
            f"(first row kept): {', '.join(sorted(set(result.duplicate_underdrive_keys)))}"
        )
    if result.duplicate_benchmark_keys:
        label = (
            "overdrive CSV"
            if result.mode == "underdrive-overdrive"
            else "measured benchmark CSV"
        )
        console.print(
            f"Note: {len(result.duplicate_benchmark_keys)} duplicate key(s) in {label} "
            f"(first row kept): {', '.join(sorted(set(result.duplicate_benchmark_keys)))}"
        )
    if result.missing_in_benchmark:
        keys = ", ".join(sorted(set(result.missing_in_benchmark)))
        if result.mode == "underdrive-overdrive":
            msg = (
                f"Note: {len(result.missing_in_benchmark)} underdrive config(s) have no overdrive row "
                f"(compare missed; not in table): {keys}"
            )
        else:
            msg = (
                f"Note: {len(result.missing_in_benchmark)} readme config(s) with metrics have no measured row "
                f"(compare missed; not in table): {keys}"
            )
        console.print(msg)
    if result.missing_in_parsed:
        keys = ", ".join(sorted(set(result.missing_in_parsed)))
        if result.mode == "underdrive-overdrive":
            msg = (
                f"Note: {len(result.missing_in_parsed)} overdrive row(s) have no matching underdrive row "
                f"(compare missed; not in table): {keys}"
            )
        else:
            msg = (
                f"Note: {len(result.missing_in_parsed)} measured row(s) have no matching readme metrics row "
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
    mode: str = typer.Option(
        "readme-overdrive",
        "--mode",
        help="readme-overdrive | readme-underdrive | underdrive-overdrive",
    ),
    parsed: Path = typer.Option(
        DEFAULT_PARSED_CSV,
        "--parsed",
        help=f"README-parsed CSV for readme-* modes (default: {DEFAULT_PARSED_CSV})",
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
    benchmark: Path | None = typer.Option(
        None,
        "--benchmark",
        help=(
            "Alias for measured CSV in readme-* only (overrides --overdrive for readme-overdrive "
            "and --underdrive for readme-underdrive)."
        ),
    ),
    delta_pct: float | None = typer.Option(
        None,
        "--delta-pct",
        help=(
            "Hide metric rows whose numeric |delta_pct| is below PCT; rows without numeric Δ%% stay; "
            "matching stedgeai_version rows omitted."
        ),
    ),
) -> None:
    """Compare README-parsed metrics to benchmark CSVs or underdrive vs overdrive."""
    if getattr(ctx, "resilient_parsing", False):
        return
    if mode not in COMPARE_MODES:
        _err_console.print(f"[red]error: invalid mode {mode!r}[/red]")
        raise typer.Exit(2)

    configure_logging()
    typer_install_exception_hook()

    result: ComparisonResult | None = None
    before_filter: int | None = None

    if mode == "readme-overdrive":
        bench = benchmark if benchmark is not None else overdrive
        if not parsed.is_file():
            _err_console.print(f"[red]error: parsed CSV not found: {parsed}[/red]")
            raise typer.Exit(2)
        if not bench.is_file():
            _err_console.print(f"[red]error: overdrive/measured CSV not found: {bench}[/red]")
            raise typer.Exit(2)
        result = compare_readme_to_bench(
            parsed,
            bench,
            mode="readme-overdrive",
            headline="README vs overdrive (delta = measured − readme)",
        )
    elif mode == "readme-underdrive":
        bench = benchmark if benchmark is not None else underdrive
        if not parsed.is_file():
            _err_console.print(f"[red]error: parsed CSV not found: {parsed}[/red]")
            raise typer.Exit(2)
        if not bench.is_file():
            _err_console.print(f"[red]error: underdrive/measured CSV not found: {bench}[/red]")
            raise typer.Exit(2)
        result = compare_readme_to_bench(
            parsed,
            bench,
            mode="readme-underdrive",
            headline="README vs underdrive (delta = measured − readme)",
        )
    else:
        if benchmark is not None:
            _err_console.print(
                "[red]error: --benchmark is only valid for readme-overdrive or readme-underdrive[/red]"
            )
            raise typer.Exit(2)
        if not underdrive.is_file():
            _err_console.print(f"[red]error: underdrive CSV not found: {underdrive}[/red]")
            raise typer.Exit(2)
        if not overdrive.is_file():
            _err_console.print(f"[red]error: overdrive CSV not found: {overdrive}[/red]")
            raise typer.Exit(2)
        result = compare_underdrive_to_overdrive(underdrive, overdrive)

    assert result is not None
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
