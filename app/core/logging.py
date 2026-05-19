from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging for structured JSON output."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
    ]

    if os.getenv("ENVIRONMENT", "development") == "development":
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def log_latency(logger: structlog.stdlib.BoundLogger, event: str, **kwargs: Any) -> Iterator[None]:
    """Context manager that emits an event with elapsed milliseconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(event, elapsed_ms=elapsed_ms, **kwargs)
