"""CSV I/O and error logging."""

import csv

from ..constants import CSV_COLUMNS
from ..paths import BenchmarkPaths
from ..utils.logutil import configure_logging, get_logger


def load_completed(paths: BenchmarkPaths) -> set[tuple[str, str]]:
    """Read CSV and return set of (variant, format) keys already done."""
    csv_path = paths.csv_path
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("model_variant", ""), row.get("format", ""))
            completed.add(key)
    return completed


def append_result(row: dict, paths: BenchmarkPaths):
    """Append one row to CSV, creating file + header if needed."""
    csv_path = paths.csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def log_error(msg: str):
    """Log error to console (Rich), structlog pipeline, and BENCHMARK_LOG."""
    configure_logging()
    get_logger("benchmark").error(msg)


def log_stdout(msg: str):
    """Log to console, structlog pipeline, and BENCHMARK_LOG (info level)."""
    configure_logging()
    get_logger("benchmark").info(msg)
