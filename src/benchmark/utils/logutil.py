"""Structlog + Rich console and audit file logging for benchmark scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.logging import RichHandler

from ..paths import benchmark_paths_for_mode

_configured = False
_audit_formatter: Optional[object] = None


def _default_audit_log_path() -> Path:
    return benchmark_paths_for_mode("underdrive").benchmark_log


def _swap_benchmark_audit_handler(audit_path: Path) -> None:
    """Point the audit FileHandler at a new path (mode switch within one process)."""
    global _audit_formatter
    if _audit_formatter is None:
        return
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_fyp_benchmark_audit", False):
            h.close()
            root.removeHandler(h)
    fh = logging.FileHandler(audit_path, encoding="utf-8", mode="a")
    fh.setFormatter(_audit_formatter)
    fh._fyp_benchmark_audit = True
    root.addHandler(fh)


def configure_logging(
    *,
    rich_tracebacks: bool = True,
    audit_log_path: Path | None = None,
) -> None:
    """Configure structlog to emit through stdlib logging: Rich console + audit files.

    ``audit_log_path`` selects the on-disk audit log; defaults to underdrive benchmark.log
    when omitted (e.g. compare-runs before any benchmark).
    """
    global _configured, _audit_formatter

    audit_path = (
        audit_log_path if audit_log_path is not None else _default_audit_log_path()
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    if not _configured:
        timestamper = structlog.processors.TimeStamper(fmt="iso")

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.PositionalArgumentsFormatter(),
                timestamper,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        pre_chain = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            timestamper,
        ]
        # No ANSI in the message body
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=pre_chain,
            processor=structlog.dev.ConsoleRenderer(colors=False),
        )

        plain_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=pre_chain,
            processor=structlog.dev.ConsoleRenderer(colors=False),
        )

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.INFO)

        rich_handler = RichHandler(
            rich_tracebacks=rich_tracebacks,
            show_path=False,
            markup=False,
            show_time=False,
        )
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(formatter)

        _audit_formatter = plain_formatter
        file_audit = logging.FileHandler(audit_path, encoding="utf-8", mode="a")
        file_audit.setLevel(logging.INFO)
        file_audit.setFormatter(plain_formatter)
        file_audit._fyp_benchmark_audit = True

        root.addHandler(rich_handler)
        root.addHandler(file_audit)

        _configured = True
    else:
        _swap_benchmark_audit_handler(audit_path)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


def log_benchmark_start(total: int, completed: int, **kwargs) -> None:
    """Log benchmark run start with configuration."""
    logger = get_logger("benchmark")
    logger.info("Benchmark started", total=total, completed=completed, **kwargs)


def log_model_start(index: int, total: int, variant: str, fmt: str, **kwargs) -> None:
    """Log model evaluation start."""
    logger = get_logger("benchmark")
    logger.info(f"[{index}/{total}] Starting", variant=variant, format=fmt, **kwargs)


def log_model_done(index: int, total: int, variant: str, **kwargs) -> None:
    """Log model evaluation completion."""
    logger = get_logger("benchmark")
    logger.info(f"[{index}/{total}] Done", variant=variant, **kwargs)


def log_model_skip(index: int, total: int, variant: str, reason: str) -> None:
    """Log skipped model."""
    logger = get_logger("benchmark")
    logger.info(f"[{index}/{total}] Skipped", variant=variant, reason=reason)


def log_model_fail(index: int, total: int, variant: str, step: str, error: str) -> None:
    """Log model evaluation failure."""
    logger = get_logger("benchmark")
    logger.error(f"[{index}/{total}] Failed", variant=variant, step=step, error=error)


def typer_install_exception_hook() -> None:
    """Optional: richer tracebacks for uncaught exceptions in Typer CLIs."""
    try:
        from rich.traceback import install

        install(show_locals=False, width=min(120, getattr(sys.stderr, "columns", 120)))
    except Exception:
        pass
