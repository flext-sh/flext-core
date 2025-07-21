"""Infrastructure adapters for Dependency Inversion Principle compliance.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Adapters that implement infrastructure protocols using external libraries,
allowing the domain to depend on abstractions while infrastructure provides concrete implementations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from flext_core.infrastructure.protocols import LoggingProtocol


class StandardLibraryLoggerAdapter:
    """Adapter for Python standard library logger to implement LoggingProtocol.

    DIP compliance - allows domain to depend on LoggingProtocol abstraction
    while infrastructure uses standard library logger implementation.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize adapter with standard library logger.

        Args:
            logger: Standard library logger instance

        """
        self._logger = logger

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log debug message with formatting support."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log info message with formatting support."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log warning message with formatting support."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log error message with formatting support."""
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback and formatting support."""
        self._logger.error(msg, *args, **kwargs)


def create_logger_adapter(logger_name: str) -> LoggingProtocol:
    """Create logger adapter.

    Args:
        logger_name: Name for the logger

    Returns:
        LoggingProtocol implementation using standard library

    """
    stdlib_logger = logging.getLogger(logger_name)
    return StandardLibraryLoggerAdapter(stdlib_logger)


__all__ = [
    "StandardLibraryLoggerAdapter",
    "create_logger_adapter",
]
