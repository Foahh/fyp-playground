"""CSV I/O and error logging."""

import csv
from pathlib import Path

from ..constants import BENCHMARK_CSV_COLUMNS, EVAL_CSV_COLUMNS
from ..paths import BenchmarkPaths
from ..utils.logutil import configure_logging, get_logger


def _load_completed_generic(csv_path: Path) -> set[tuple[str, str]]:
    """Read CSV and return set of (variant, format) keys already done."""
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("model_variant", ""), row.get("format", ""))
            completed.add(key)
    return completed


def _append_csv_row(csv_path: Path, row: dict, fieldnames: list[str]):
    """Append row to CSV, creating file + header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def load_completed(paths: BenchmarkPaths) -> set[tuple[str, str]]:
    """Read CSV and return set of (variant, format) keys already done."""
    return _load_completed_generic(paths.csv_path)


def append_result(row: dict, paths: BenchmarkPaths):
    """Append one benchmark row to CSV, creating file + header if needed."""
    _append_csv_row(paths.csv_path, row, BENCHMARK_CSV_COLUMNS)


def load_eval_completed(eval_csv: Path) -> set[tuple[str, str]]:
    """Read evaluation CSV and return set of (variant, format) keys already done."""
    return _load_completed_generic(eval_csv)


def append_eval_result(row: dict, eval_csv: Path):
    """Append one evaluation row to CSV, creating file + header if needed."""
    _append_csv_row(eval_csv, row, EVAL_CSV_COLUMNS)


def log_error(msg: str):
    """Log error to console (Rich), structlog pipeline, and BENCHMARK_LOG."""
    configure_logging()
    get_logger("benchmark").error(msg)


def log_stdout(msg: str):
    """Log to console, structlog pipeline, and BENCHMARK_LOG (info level)."""
    configure_logging()
    get_logger("benchmark").info(msg)
