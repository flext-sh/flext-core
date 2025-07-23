"""FLEXT Core Logging System - Unified Logging Pattern.

Enterprise-grade logging system with standardized context,
structured logging, and comprehensive error tracking.
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

from flext_core.constants import FlextLogLevel
from flext_core.patterns.typedefs import FlextLoggerContext
from flext_core.patterns.typedefs import FlextLoggerName
from flext_core.patterns.typedefs import FlextLogTag

# =============================================================================
# LOG CONTEXT - Structured logging context
# =============================================================================


class FlextLogContext:
    """Structured logging context for enterprise applications."""

    def __init__(
        self,
        context_id: FlextLoggerContext | None = None,
        tags: list[FlextLogTag] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize log context.

        Args:
            context_id: Unique context identifier
            tags: List of tags for categorization
            metadata: Additional context metadata

        """
        self.context_id = context_id or FlextLoggerContext(
            f"context_{id(self)}",
        )
        self.tags = tags or []
        self.metadata = metadata or {}

    def add_tag(self, tag: FlextLogTag) -> None:
        """Add a tag to the context."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_metadata(self, key: str, value: object) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value

    def merge(self, other: FlextLogContext) -> None:
        """Merge another context into this one."""
        # Merge tags
        for tag in other.tags:
            self.add_tag(tag)

        # Merge metadata
        self.metadata.update(other.metadata)

    def to_dict(self) -> dict[str, object]:
        """Convert context to dictionary for logging."""
        return {
            "context_id": self.context_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# =============================================================================
# LOGGER INTERFACE - Abstract logger for standardization
# =============================================================================


class FlextLogger(ABC):
    """Base class for all FLEXT loggers.

    Provides standardized logging interface with context support,
    structured logging, and consistent formatting.
    """

    def __init__(
        self,
        logger_name: FlextLoggerName,
        context: FlextLogContext | None = None,
    ) -> None:
        """Initialize logger.

        Args:
            logger_name: Name of the logger
            context: Optional default context

        """
        self.logger_name = logger_name
        self.default_context = context or FlextLogContext()

    @abstractmethod
    def log(
        self,
        level: FlextLogLevel,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log a message at specified level.

        Args:
            level: Log level
            message: Message to log
            context: Optional context for this log entry
            **kwargs: Additional log data

        """

    def debug(
        self,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log debug message."""
        self.log(FlextLogLevel.DEBUG, message, context, **kwargs)

    def info(
        self,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log info message."""
        self.log(FlextLogLevel.INFO, message, context, **kwargs)

    def warning(
        self,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log warning message."""
        self.log(FlextLogLevel.WARNING, message, context, **kwargs)

    def error(
        self,
        message: str,
        context: FlextLogContext | None = None,
        exception: Exception | None = None,
        **kwargs: object,
    ) -> None:
        """Log error message with optional exception."""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "args": exception.args,
            }
        self.log(FlextLogLevel.ERROR, message, context, **kwargs)

    def critical(
        self,
        message: str,
        context: FlextLogContext | None = None,
        exception: Exception | None = None,
        **kwargs: object,
    ) -> None:
        """Log critical message with optional exception."""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "args": exception.args,
            }
        self.log(FlextLogLevel.CRITICAL, message, context, **kwargs)

    def _merge_context(
        self,
        context: FlextLogContext | None,
    ) -> FlextLogContext:
        """Merge provided context with default context."""
        merged = FlextLogContext(
            context_id=self.default_context.context_id,
            tags=list(self.default_context.tags),
            metadata=dict(self.default_context.metadata),
        )

        if context:
            merged.merge(context)

        return merged


# =============================================================================
# STANDARD LOGGER - Python logging integration
# =============================================================================


class FlextStandardLogger(FlextLogger):
    """Standard logger implementation using Python logging."""

    def __init__(
        self,
        logger_name: FlextLoggerName,
        context: FlextLogContext | None = None,
        python_logger: logging.Logger | None = None,
    ) -> None:
        """Initialize standard logger.

        Args:
            logger_name: Name of the logger
            context: Optional default context
            python_logger: Optional Python logger instance

        """
        super().__init__(logger_name, context)
        self.python_logger = python_logger or logging.getLogger(
            str(logger_name),
        )

    def log(
        self,
        level: FlextLogLevel,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log message using Python logging."""
        # Merge contexts
        final_context = self._merge_context(context)

        # Prepare log data
        log_data = {
            "message": message,
            "context": final_context.to_dict(),
            **kwargs,
        }

        # Convert FlextLogLevel to Python logging level
        python_level = self._convert_log_level(level)

        # Log with structured data
        self.python_logger.log(
            python_level,
            self._format_message(message, log_data),
            extra={"flext_data": log_data},
        )

    def _convert_log_level(self, level: FlextLogLevel) -> int:
        """Convert FlextLogLevel to Python logging level."""
        level_mapping = {
            FlextLogLevel.CRITICAL: logging.CRITICAL,
            FlextLogLevel.ERROR: logging.ERROR,
            FlextLogLevel.WARNING: logging.WARNING,
            FlextLogLevel.INFO: logging.INFO,
            FlextLogLevel.DEBUG: logging.DEBUG,
            FlextLogLevel.TRACE: logging.DEBUG,  # Map TRACE to DEBUG
        }
        return level_mapping.get(level, logging.INFO)

    def _format_message(self, message: str, log_data: dict[str, Any]) -> str:
        """Format log message with context."""
        context = log_data.get("context", {})
        context_id = context.get("context_id", "")
        tags = context.get("tags", [])

        formatted_tags = f"[{','.join(tags)}]" if tags else ""

        return f"[{context_id}]{formatted_tags} {message}"


# =============================================================================
# LOGGER FACTORY - Centralized logger creation
# =============================================================================


class FlextLoggerFactory:
    """Factory for creating standardized loggers."""

    def __init__(
        self,
        default_level: FlextLogLevel = FlextLogLevel.INFO,
        default_context: FlextLogContext | None = None,
    ) -> None:
        """Initialize logger factory.

        Args:
            default_level: Default log level for created loggers
            default_context: Default context for created loggers

        """
        self.default_level = default_level
        self.default_context = default_context or FlextLogContext()
        self._loggers: dict[str, FlextLogger] = {}

    def create_logger(
        self,
        logger_name: FlextLoggerName,
        context: FlextLogContext | None = None,
        logger_type: type[FlextLogger] = FlextStandardLogger,
    ) -> FlextLogger:
        """Create or get existing logger.

        Args:
            logger_name: Name of the logger
            context: Optional context for the logger
            logger_type: Type of logger to create

        Returns:
            FlextLogger instance

        """
        logger_key = str(logger_name)

        if logger_key not in self._loggers:
            # Merge contexts
            final_context = FlextLogContext(
                tags=list(self.default_context.tags),
                metadata=dict(self.default_context.metadata),
            )
            if context:
                final_context.merge(context)

            # Create logger
            self._loggers[logger_key] = logger_type(
                logger_name,
                final_context,
            )

        return self._loggers[logger_key]

    def get_logger(
        self,
        logger_name: FlextLoggerName,
    ) -> FlextLogger | None:
        """Get existing logger by name."""
        return self._loggers.get(str(logger_name))

    def configure_python_logging(
        self,
        level: FlextLogLevel = FlextLogLevel.INFO,
        format_string: str | None = None,
    ) -> None:
        """Configure Python logging for FLEXT loggers.

        Args:
            level: Logging level
            format_string: Optional custom format string

        """
        temp_logger = FlextStandardLogger(FlextLoggerName("temp"))
        python_level = temp_logger._convert_log_level(level)  # noqa: SLF001

        default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=python_level,
            format=format_string or default_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def get_all_loggers(self) -> dict[str, FlextLogger]:
        """Get all created loggers."""
        return dict(self._loggers)


# =============================================================================
# LOGGER UTILITIES - Helper functions and classes
# =============================================================================


class FlextLoggerMixin:
    """Mixin to add logging capabilities to any class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize logger mixin with class-specific logger."""
        super().__init__(*args, **kwargs)
        self._logger = FlextLoggerFactory().create_logger(
            FlextLoggerName(self.__class__.__name__),
            FlextLogContext(
                tags=[
                    FlextLogTag("class"),
                    FlextLogTag(self.__class__.__name__),
                ],
            ),
        )

    @property
    def logger(self) -> FlextLogger:
        """Get logger for this class."""
        return self._logger


def create_context_from_dict(data: dict[str, object]) -> FlextLogContext:
    """Create log context from dictionary data.

    Args:
        data: Dictionary with context, tags, and metadata

    Returns:
        FlextLogContext instance

    """
    context_id_raw = data.get("context_id")
    context_id = None
    if context_id_raw and isinstance(context_id_raw, str):
        context_id = FlextLoggerContext(context_id_raw)

    tags_raw = data.get("tags", [])
    tags = []
    if isinstance(tags_raw, list):
        tags = [FlextLogTag(tag) for tag in tags_raw if isinstance(tag, str)]

    metadata_raw = data.get("metadata", {})
    metadata = None
    if isinstance(metadata_raw, dict):
        metadata = metadata_raw

    return FlextLogContext(
        context_id=context_id,
        tags=tags,
        metadata=metadata,
    )


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "FlextLogContext",
    "FlextLogLevel",  # Re-export from constants
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggerMixin",
    "FlextStandardLogger",
    "create_context_from_dict",
]
