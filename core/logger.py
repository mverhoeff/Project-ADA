"""Structured logging setup for Project ADA.

Call :func:`configure_logging` once at application startup, then obtain a
logger anywhere via :func:`get_logger(__name__)`. Output is colour-rendered
on a TTY and JSON otherwise, so logs are human-readable during development
and machine-parsable in production.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog and the stdlib logging bridge.

    Args:
        level: Minimum log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    renderer: structlog.types.Processor
    if sys.stderr.isatty():
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name.

    Args:
        name: Logger name, conventionally ``__name__``.
    """
    return structlog.get_logger(name)
