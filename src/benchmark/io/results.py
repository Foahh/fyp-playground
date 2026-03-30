"""CSV I/O and error logging."""

import csv

from ..constants import CSV_COLUMNS
from ..paths import CSV_PATH
from ..utils.logutil import configure_logging, get_logger


def load_completed() -> set[tuple[str, str]]:
    """Read CSV and return set of (variant, format) keys already done."""
    completed = set()
    if not CSV_PATH.exists():
        return completed
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("model_variant", ""), row.get("format", ""))
            completed.add(key)
    return completed


def append_result(row: dict):
    """Append one row to CSV, creating file + header if needed."""
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
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
