"""Logging setup shared by the library, the CLI, and the scripts.

Library modules call :func:`get_logger` and never configure handlers. Entry points call
:func:`configure_logging` exactly once, which is the only place a handler is attached.
Importing this module has no side effects beyond creating the package logger object.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

PACKAGE_LOGGER_NAME: Final[str] = "scale_aware_compression"

CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
FILE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_CONFIGURED_MARKER: Final[str] = "_sajc_configured"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the package namespace.

    Args:
        name: Usually ``__name__``. A dotted module path outside the package namespace is
            reduced to its last component and nested under the package logger, so all
            project logs share one root and one level setting.

    Returns:
        The logger. No handlers are attached here.
    """
    if not name or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name.rsplit('.', 1)[-1]}")


def configure_logging(
    level: int | str = "INFO",
    *,
    log_file: str | Path | None = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Attach handlers to the package logger. Safe to call more than once.

    Args:
        level: Logging level as a name (``"DEBUG"``) or an integer.
        log_file: Optional file to mirror all records into at DEBUG-or-``level``
            granularity. Parent directories are created.
        console: Whether to log to stderr.
        force: Replace existing handlers instead of leaving them in place. Use when a
            second entry point takes over in the same process.

    Returns:
        The configured package logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    resolved_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unrecognised logging level: {level!r}")

    if force or not getattr(logger, _CONFIGURED_MARKER, False):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        if console:
            console_handler = logging.StreamHandler(stream=sys.stderr)
            console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(console_handler)

        if log_file is not None:
            destination = Path(log_file)
            destination.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(destination, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(file_handler)

        # Keep third-party log noise out of the project's stream by default.
        logger.propagate = False
        setattr(logger, _CONFIGURED_MARKER, True)

    logger.setLevel(resolved_level)
    return logger


def log_key_values(logger: logging.Logger, title: str, values: dict[str, Any]) -> None:
    """Log a small mapping as an aligned block, for run headers.

    Args:
        logger: Target logger.
        title: Heading printed above the block.
        values: Mapping to render. Nested values are logged with ``repr``.
    """
    logger.info("%s", title)
    if not values:
        logger.info("  (none)")
        return
    width = max(len(str(key)) for key in values)
    for key, value in values.items():
        logger.info("  %-*s : %s", width, key, value)


@contextmanager
def log_stage(logger: logging.Logger, stage: str) -> Iterator[None]:
    """Log entry and exit of a pipeline stage, including failures.

    Args:
        logger: Target logger.
        stage: Stage name, e.g. ``"pruning"``.

    Yields:
        ``None``; the wrapped block runs inside the context.
    """
    logger.info("--> %s", stage)
    try:
        yield
    except Exception:
        logger.exception("<-- %s FAILED", stage)
        raise
    else:
        logger.info("<-- %s done", stage)
