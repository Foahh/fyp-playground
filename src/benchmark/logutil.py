"""Structlog + Rich console and audit file logging for benchmark scripts."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import structlog
from rich.logging import RichHandler

from .constants import ERROR_LOG, STDOUT_LOG

_configured = False


class _NotErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR


def configure_logging(*, rich_tracebacks: bool = True) -> None:
    """Configure structlog to emit through stdlib logging: Rich console + audit files."""
    global _configured
    if _configured:
        return

    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)

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
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain,
        processor=structlog.dev.ConsoleRenderer(colors=True),
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

    stdout_audit = logging.FileHandler(STDOUT_LOG, encoding="utf-8", mode="a")
    stdout_audit.setLevel(logging.INFO)
    stdout_audit.setFormatter(plain_formatter)
    stdout_audit.addFilter(_NotErrorFilter())

    err_audit = logging.FileHandler(ERROR_LOG, encoding="utf-8", mode="a")
    err_audit.setLevel(logging.ERROR)
    err_audit.setFormatter(plain_formatter)

    root.addHandler(rich_handler)
    root.addHandler(stdout_audit)
    root.addHandler(err_audit)

    _configured = True


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


def typer_install_exception_hook() -> None:
    """Optional: richer tracebacks for uncaught exceptions in Typer CLIs."""
    try:
        from rich.traceback import install

        install(show_locals=False, width=min(120, getattr(sys.stderr, "columns", 120)))
    except Exception:
        pass
