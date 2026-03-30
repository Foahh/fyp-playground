"""Parse STM32 model zoo README metrics and/or compare them to benchmark_results.csv.

Official README NPU metrics assume **overdrive mode** under **default STM32Cube.AI
configuration** (per each object_detection family README *Performances → Metrics*).

From repo root:
  python src/benchmark/run_compare.py parse
  python src/benchmark/run_compare.py compare
  python src/benchmark/run_compare.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from benchmark.compare import (  # noqa: E402
    DEFAULT_OVERDRIVE_CSV,
    DEFAULT_PARSED_CSV,
    compare_readme_to_bench,
    filter_rows_by_abs_delta_pct,
    compare_main,
    print_comparison_report,
)
from benchmark.logutil import configure_logging, typer_install_exception_hook  # noqa: E402

_COMPARE_ONLY_FLAGS = frozenset(
    ("--mode", "--benchmark", "--nominal", "--delta-pct")
)

_err = Console(stderr=True)

app = typer.Typer(
    help=__doc__,
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command("parse")
def cmd_parse(
    out: Path | None = typer.Option(
        None,
        "--out",
        help=f"Output CSV (default: {DEFAULT_PARSED_CSV})",
    ),
) -> None:
    configure_logging()
    typer_install_exception_hook()
    from benchmark.parse_modelzoo_readme import write_metric_parsed_csv

    p, n = write_metric_parsed_csv(path=out)
    typer.echo(f"Wrote {p} ({n} rows)")


@app.command(
    "compare",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help='Tabulate deltas (same flags as ``python -m benchmark.compare`` with PYTHONPATH=src).',
)
def cmd_compare(ctx: typer.Context) -> None:
    configure_logging()
    typer_install_exception_hook()
    raise typer.Exit(compare_main(list(ctx.args)))


@app.command("all")
def cmd_all(
    parsed: Path = typer.Option(DEFAULT_PARSED_CSV, "--parsed"),
    overdrive: Path = typer.Option(DEFAULT_OVERDRIVE_CSV, "--overdrive"),
    delta_pct: float | None = typer.Option(None, "--delta-pct"),
) -> None:
    raise typer.Exit(_run_all(parsed, overdrive, delta_pct))


def _run_all(parsed: Path, overdrive: Path, delta_pct: float | None) -> int:
    configure_logging()
    typer_install_exception_hook()
    from benchmark.parse_modelzoo_readme import write_metric_parsed_csv

    out, n = write_metric_parsed_csv(path=parsed)
    typer.echo(f"Wrote {out} ({n} rows)\n")

    if not parsed.is_file():
        _err.print(f"[red]error: parsed CSV not found after write: {parsed}[/red]")
        return 2
    if not overdrive.is_file():
        _err.print(f"[red]error: overdrive CSV not found: {overdrive}[/red]")
        return 2

    result = compare_readme_to_bench(
        parsed,
        overdrive,
        mode="readme-overdrive",
        headline="README vs overdrive (delta = measured − readme)",
    )
    before_filter: int | None = None
    if delta_pct is not None:
        if delta_pct < 0:
            _err.print("[red]error: --delta-pct must be >= 0[/red]")
            return 2
        before_filter = len(result.delta_rows)
        result.delta_rows = filter_rows_by_abs_delta_pct(result.delta_rows, delta_pct)

    print_comparison_report(
        result,
        delta_rows_before_filter=before_filter,
        delta_pct=delta_pct,
    )
    return 0


def _run_all_argv(argv: list[str]) -> int:
    inner = typer.Typer(
        add_completion=False,
        invoke_without_command=True,
        help="Parse README metrics then compare to overdrive results.",
    )

    @inner.callback(invoke_without_command=True)
    def _all_wrap(
        parsed: Path = typer.Option(DEFAULT_PARSED_CSV, "--parsed"),
        overdrive: Path = typer.Option(DEFAULT_OVERDRIVE_CSV, "--overdrive"),
        delta_pct: float | None = typer.Option(None, "--delta-pct"),
    ) -> None:
        raise typer.Exit(_run_all(parsed, overdrive, delta_pct))

    try:
        inner(args=argv, prog_name="run_compare.py")
    except typer.Exit as e:
        return int(e.exit_code) if e.exit_code is not None else 0
    return 0


def main() -> int:
    argv = list(sys.argv[1:])
    if not argv:
        return _run_all(DEFAULT_PARSED_CSV, DEFAULT_OVERDRIVE_CSV, None)

    if argv[0] in ("-h", "--help"):
        typer.echo(__doc__)
        typer.echo("Commands: parse | compare | all (default: all)")
        typer.echo(f"  parse   — write README metrics CSV (default out: {DEFAULT_PARSED_CSV})")
        typer.echo(
            "  compare — tabulate deltas; --mode readme-overdrive|readme-nominal|nominal-overdrive "
            "(see --help on compare)"
        )
        typer.echo("  all     — parse then compare")
        typer.echo("\nWith no subcommand, default is ``all``. Flags starting with ``-`` may imply "
                   "``compare`` or ``all`` (see legacy behavior in script source).")
        return 0

    if argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        if any(token in _COMPARE_ONLY_FLAGS for token in argv):
            return compare_main(argv)
        return _run_all_argv(argv)

    if argv[0] in ("parse", "compare", "all"):
        try:
            app(args=argv, prog_name="run_compare.py")
        except typer.Exit as e:
            return int(e.exit_code) if e.exit_code is not None else 0
        return 0

    typer.echo(f"error: unknown command {argv[0]!r} (use parse, compare, or all)", err=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
