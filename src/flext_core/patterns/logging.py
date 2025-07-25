"""FLEXT Core Logging System - Unified Logging Pattern.

Enterprise-grade logging system with standardized context,
structured logging, and comprehensive error tracking.
"""

from __future__ import annotations

import inspect
import logging
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from contextlib import suppress
from typing import Any
from typing import Self
from typing import TypeVar

import structlog

from flext_core.constants import FlextLogLevel
from flext_core.patterns.typedefs import FlextLoggerContext
from flext_core.patterns.typedefs import FlextLoggerName
from flext_core.patterns.typedefs import FlextLogTag

# =============================================================================
# GLOBAL LOGGING CONFIGURATION - Centralized logging setup
# =============================================================================


class FlextGlobalLoggingConfig:
    """Global logging configuration for FLEXT applications.

    Provides centralized control over logging levels and configuration
    for all FLEXT modules and any code that uses this logging system.
    """

    _instance: FlextGlobalLoggingConfig | None = None
    _configured = False
    _default_level = FlextLogLevel.INFO
    _structlog_configured = False

    def __new__(cls) -> Self:
        """Singleton pattern for global configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # type: ignore[return-value]

    def __init__(self) -> None:
        """Initialize global logging configuration."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_environment()

    def _setup_environment(self) -> None:
        """Setup logging from environment variables."""
        # Check for FLEXT_LOG_LEVEL environment variable
        env_level = os.getenv("FLEXT_LOG_LEVEL", "").upper()
        if env_level:
            with suppress(ValueError):
                self._default_level = FlextLogLevel(env_level)

    def configure(
        self,
        level: FlextLogLevel | str | None = None,
        format_string: str | None = None,
        *,
        structlog_enabled: bool = True,
        structlog_format: str = "json",
        force_reconfigure: bool = False,
    ) -> None:
        """Configure global logging settings.

        Args:
            level: Logging level (FlextLogLevel, string, or None for default)
            format_string: Custom format string for Python logging
            structlog_enabled: Whether to enable structlog configuration
            structlog_format: structlog format
                ('json', 'console', 'development')
            force_reconfigure: Force reconfiguration even if already configured

        Example:
            >>> from flext_core.patterns.logging import FlextGlobalLoggingConfig
            >>> config = FlextGlobalLoggingConfig()
            >>> config.configure(level="DEBUG", structlog_format="console")
        """
        if self._configured and not force_reconfigure:
            return

        # Set level
        if level is not None:
            if isinstance(level, str):
                try:
                    self._default_level = FlextLogLevel(level.upper())
                except ValueError as err:
                    msg = f"Invalid log level: {level}"
                    raise ValueError(msg) from err
            else:
                self._default_level = level  # type: ignore[unreachable]

        # Configure Python logging
        self._configure_python_logging(format_string)

        # Configure structlog if enabled
        if structlog_enabled:
            self._configure_structlog(structlog_format)

        self._configured = True

    def _configure_python_logging(
        self,
        format_string: str | None = None,
    ) -> None:
        """Configure Python logging globally."""
        python_level = self._convert_flext_to_python_level(self._default_level)

        default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=python_level,
            format=format_string or default_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,  # Force reconfiguration
        )

        # Set level for all existing loggers
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.setLevel(python_level)

    def _configure_structlog(self, format_type: str) -> None:
        """Configure structlog based on format type."""
        if format_type == "json":
            FlextStructlogConfig.configure_json()
        elif format_type in {"console", "development"}:
            FlextStructlogConfig.configure_development()
        else:
            FlextStructlogConfig.configure_production()

        self._structlog_configured = True

    @staticmethod
    def _convert_flext_to_python_level(level: FlextLogLevel) -> int:
        """Convert FlextLogLevel to Python logging level."""
        level_mapping = {
            FlextLogLevel.CRITICAL: logging.CRITICAL,
            FlextLogLevel.ERROR: logging.ERROR,
            FlextLogLevel.WARNING: logging.WARNING,
            FlextLogLevel.INFO: logging.INFO,
            FlextLogLevel.DEBUG: logging.DEBUG,
            FlextLogLevel.TRACE: logging.DEBUG,  # TRACE maps to DEBUG
        }
        return level_mapping.get(level, logging.INFO)

    @property
    def level(self) -> FlextLogLevel:
        """Get current global log level."""
        return self._default_level

    @property
    def is_configured(self) -> bool:
        """Check if logging is configured."""
        return self._configured

    @property
    def is_structlog_configured(self) -> bool:
        """Check if structlog is configured."""
        return self._structlog_configured

    def set_level(self, level: FlextLogLevel | str) -> None:
        """Set global log level and reconfigure if needed.

        Args:
            level: New log level

        Example:
            >>> config = FlextGlobalLoggingConfig()
            >>> config.set_level("TRACE")
        """
        if isinstance(level, str):
            level = FlextLogLevel(level.upper())

        self._default_level = level

        if self._configured:
            # Reconfigure with new level
            self.configure(force_reconfigure=True)


# Global configuration instance
_global_config = FlextGlobalLoggingConfig()


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

    def trace(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log trace message (most verbose level).

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        self.log(FlextLogLevel.TRACE, formatted_message, context, **kwargs)

    def debug(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log debug message.

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        self.log(FlextLogLevel.DEBUG, formatted_message, context, **kwargs)

    def info(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log info message.

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        self.log(FlextLogLevel.INFO, formatted_message, context, **kwargs)

    def warning(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log warning message.

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        self.log(FlextLogLevel.WARNING, formatted_message, context, **kwargs)

    def error(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        exception: Exception | None = None,
        **kwargs: object,
    ) -> None:
        """Log error message with optional exception.

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            exception: Optional exception instance
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "args": exception.args,
            }
        self.log(FlextLogLevel.ERROR, formatted_message, context, **kwargs)

    def exception(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log exception message with stack trace.

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        self.log(FlextLogLevel.ERROR, formatted_message, context, **kwargs)

    def critical(
        self,
        message: str,
        *args: object,
        context: FlextLogContext | None = None,
        exception: Exception | None = None,
        **kwargs: object,
    ) -> None:
        """Log critical message with optional exception.

        Args:
            message: Message format string or plain message
            *args: Format arguments for message (printf-style)
            context: Optional context for this log entry
            exception: Optional exception instance
            **kwargs: Additional log data
        """
        # Handle printf-style formatting like standard Python logging
        formatted_message = message % args if args else message
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "args": exception.args,
            }
        self.log(FlextLogLevel.CRITICAL, formatted_message, context, **kwargs)

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
        log_data: dict[str, Any] = {
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
            FlextLogLevel.TRACE: logging.DEBUG,  # TRACE maps to DEBUG
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
# STRUCTLOG LOGGER - Structured logging with structlog
# =============================================================================


class FlextStructLogger(FlextLogger):
    """Structured logger implementation using structlog."""

    def __init__(
        self,
        logger_name: FlextLoggerName,
        context: FlextLogContext | None = None,
        struct_logger: structlog.BoundLogger | None = None,
    ) -> None:
        """Initialize structlog logger.

        Args:
            logger_name: Name of the logger
            context: Optional default context
            struct_logger: Optional structlog logger instance

        """
        super().__init__(logger_name, context)
        self.struct_logger = struct_logger or structlog.get_logger(
            str(logger_name),
        )

    def log(
        self,
        level: FlextLogLevel,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log message using structlog."""
        # Merge contexts
        final_context = self._merge_context(context)

        # Prepare structured data
        log_data = {
            "logger_name": str(self.logger_name),
            "context_id": str(final_context.context_id),
            "tags": [str(tag) for tag in final_context.tags],
            **final_context.metadata,
            **kwargs,
        }

        # Get bound logger with context
        bound_logger = self.struct_logger.bind(**log_data)

        # Convert FlextLogLevel to structlog level
        structlog_level = self._convert_log_level(level)

        # Log with structured data
        getattr(bound_logger, structlog_level)(message)

    def _convert_log_level(self, level: FlextLogLevel) -> str:
        """Convert FlextLogLevel to structlog level."""
        level_mapping = {
            FlextLogLevel.CRITICAL: "critical",
            FlextLogLevel.ERROR: "error",
            FlextLogLevel.WARNING: "warning",
            FlextLogLevel.INFO: "info",
            FlextLogLevel.DEBUG: "debug",
            FlextLogLevel.TRACE: "debug",  # TRACE maps to debug
        }
        return level_mapping.get(level, "info")

    def bind(self, **kwargs: object) -> FlextStructLogger:
        """Bind additional context to logger."""
        bound_struct_logger = self.struct_logger.bind(**kwargs)
        return FlextStructLogger(
            self.logger_name,
            self.default_context,
            bound_struct_logger,
        )

    def new(self, **kwargs: object) -> FlextStructLogger:
        """Create new logger with additional context."""
        new_struct_logger = self.struct_logger.new(**kwargs)
        return FlextStructLogger(
            self.logger_name,
            self.default_context,
            new_struct_logger,
        )


# =============================================================================
# STRUCTLOG CONFIGURATION - Configure structlog processors and renderers
# =============================================================================


class FlextStructlogConfig:
    """Configuration for structlog processors and renderers."""

    @staticmethod
    def configure_development() -> None:
        """Configure structlog for development environment."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @staticmethod
    def configure_production() -> None:
        """Configure structlog for production environment."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @staticmethod
    def configure_json() -> None:
        """Configure structlog for JSON output."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @staticmethod
    def configure_custom(
        processors: list[Any] | None = None,
        renderer: Any | None = None,  # noqa: ANN401
    ) -> None:
        """Configure structlog with custom processors and renderer.

        Args:
            processors: Custom list of processors
            renderer: Custom renderer (defaults to JSONRenderer)

        """
        default_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        if processors is not None:
            default_processors = processors

        if renderer is not None:
            default_processors.append(renderer)
        else:
            default_processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=default_processors,  # type: ignore[arg-type]
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


# =============================================================================
# LOGGER FACTORY - Centralized logger creation
# =============================================================================


class FlextLoggerFactory:
    """Factory for creating standardized loggers."""

    def __init__(
        self,
        default_level: FlextLogLevel | None = None,
        default_context: FlextLogContext | None = None,
        default_logger_type: type[FlextLogger] = FlextStructLogger,
    ) -> None:
        """Initialize logger factory.

        Args:
            default_level: Default log level for created loggers
                (uses global if None)
            default_context: Default context for created loggers
            default_logger_type: Default logger type to create

        """
        self.default_level = default_level or _global_config.level
        self.default_context = default_context or FlextLogContext()
        self.default_logger_type = default_logger_type
        self._loggers: dict[str, FlextLogger] = {}

    def create_logger(
        self,
        logger_name: FlextLoggerName,
        context: FlextLogContext | None = None,
        logger_type: type[FlextLogger] | None = None,
    ) -> FlextLogger:
        """Create or get existing logger.

        Args:
            logger_name: Name of the logger
            context: Optional context for the logger
            logger_type: Type of logger to create (defaults to factory default)

        Returns:
            FlextLogger instance

        """
        logger_key = str(logger_name)
        actual_logger_type = logger_type or self.default_logger_type

        if logger_key not in self._loggers:
            # Merge contexts
            final_context = FlextLogContext(
                tags=list(self.default_context.tags),
                metadata=dict(self.default_context.metadata),
            )
            if context:
                final_context.merge(context)

            # Create logger
            self._loggers[logger_key] = actual_logger_type(
                logger_name,
                final_context,
            )

        return self._loggers[logger_key]

    @staticmethod
    def _convert_log_level(level: FlextLogLevel) -> int:
        """Convert FlextLogLevel to Python logging level."""
        level_mapping = {
            FlextLogLevel.CRITICAL: logging.CRITICAL,
            FlextLogLevel.ERROR: logging.ERROR,
            FlextLogLevel.WARNING: logging.WARNING,
            FlextLogLevel.INFO: logging.INFO,
            FlextLogLevel.DEBUG: logging.DEBUG,
            FlextLogLevel.TRACE: logging.DEBUG,  # TRACE maps to DEBUG
        }
        return level_mapping.get(level, logging.INFO)

    @staticmethod
    def create_logger_name(name: str) -> FlextLoggerName:
        """Create a logger name from a string."""
        return FlextLoggerName(name)

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
        python_level = self._convert_log_level(level)

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


def create_context_from_dict(data: dict[str, Any]) -> FlextLogContext:
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
    tags: list[FlextLogTag] = []
    if isinstance(tags_raw, list):
        tags = [
            FlextLogTag(tag_item) for tag_item in tags_raw if isinstance(tag_item, str)
        ]

    metadata_raw = data.get("metadata", {})
    metadata: dict[str, Any] | None = None
    if isinstance(metadata_raw, dict):
        metadata = dict(metadata_raw)  # Create a copy with proper typing

    return FlextLogContext(
        context_id=context_id,
        tags=tags,
        metadata=metadata,
    )


# =============================================================================
# STRUCTLOG UTILITIES - Helper functions for structlog
# =============================================================================


def get_structlog_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structlog logger with the given name.

    Args:
        name: Logger name. If None, uses the caller's module name.

    Returns:
        structlog.BoundLogger instance

    Example:
        >>> from flext_core.patterns.logging import get_structlog_logger
        >>> logger = get_structlog_logger(__name__)
        >>> logger.info("Hello", user_id=123, action="login")
    """
    if name is None:
        # Get the caller's module name
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back if frame else None
            if caller_frame:
                name = caller_frame.f_globals.get("__name__", "unknown")
            else:
                name = "unknown"
        finally:
            del frame

    return structlog.get_logger(name if name is not None else "unknown")  # type: ignore[no-any-return]


def bind_context(
    logger: structlog.BoundLogger,
    **kwargs: object,
) -> structlog.BoundLogger:
    """Bind context to a structlog logger.

    Args:
        logger: structlog logger to bind context to
        **kwargs: Context key-value pairs

    Returns:
        Bound logger with context

    Example:
        >>> from flext_core.patterns.logging import get_structlog_logger, bind_context
        >>> logger = get_structlog_logger(__name__)
        >>> user_logger = bind_context(logger, user_id=123, session_id="abc")
        >>> user_logger.info("User action", action="login")
    """
    return logger.bind(**kwargs)


def new_context(
    logger: structlog.BoundLogger,
    **kwargs: object,
) -> structlog.BoundLogger:
    """Create new context for a structlog logger.

    Args:
        logger: structlog logger to create new context for
        **kwargs: Context key-value pairs

    Returns:
        New logger with context

    Example:
        >>> from flext_core.patterns.logging import get_structlog_logger, new_context
        >>> logger = get_structlog_logger(__name__)
        >>> request_logger = new_context(logger, request_id="req-123", method="GET")
        >>> request_logger.info("Request processed", status_code=200)
    """
    return logger.new(**kwargs)


# =============================================================================
# LOGGER HELPERS - Simple utility functions
# =============================================================================

# Global default factory instance for convenience
_default_factory = FlextLoggerFactory()


def get_logger(name: str | None = None) -> FlextLogger:
    """Get a logger with the given name.

    Args:
        name: Logger name. If None, uses the caller's module name.

    Returns:
        FlextLogger instance

    Example:
        >>> from flext_core.patterns.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger = get_logger()  # Uses caller module
    """
    if name is None:
        # Get the caller's module name
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back if frame else None
            if caller_frame:
                name = caller_frame.f_globals.get("__name__", "unknown")
            else:
                name = "unknown"
        finally:
            del frame

    # Ensure name is not None for the FlextLoggerName constructor
    logger_name = name if name is not None else "unknown"
    return _default_factory.create_logger(FlextLoggerName(logger_name))


def configure_logging(
    level: FlextLogLevel | str | None = None,
    format_string: str | None = None,
    *,
    structlog_enabled: bool = True,
    structlog_format: str = "json",
) -> None:
    """Configure global logging with FLEXT defaults.

    Args:
        level: Logging level to use
            (FlextLogLevel, string, or None for default)
        format_string: Optional custom format string
        structlog_enabled: Whether to enable structlog configuration
        structlog_format: structlog format ('json', 'console', 'development')

    Example:
        >>> from flext_core.patterns.logging import (
        ...     configure_logging,
        ...     FlextLogLevel,
        ... )
        >>> configure_logging(FlextLogLevel.DEBUG)
        >>> configure_logging("TRACE", structlog_format="console")
    """
    _global_config.configure(
        level=level,
        format_string=format_string,
        structlog_enabled=structlog_enabled,
        structlog_format=structlog_format,
    )


def set_log_level(level: FlextLogLevel | str) -> None:
    """Set global log level.

    Args:
        level: New log level (FlextLogLevel or string)

    Example:
        >>> from flext_core.patterns.logging import set_log_level, FlextLogLevel
        >>> set_log_level("TRACE")
        >>> set_log_level(FlextLogLevel.DEBUG)
    """
    _global_config.set_level(level)


def get_global_config() -> FlextGlobalLoggingConfig:
    """Get the global logging configuration instance.

    Returns:
        FlextGlobalLoggingConfig instance

    Example:
        >>> from flext_core.patterns.logging import get_global_config
        >>> config = get_global_config()
        >>> print(f"Current level: {config.level}")
        >>> print(f"Configured: {config.is_configured}")
    """
    return _global_config


# =============================================================================
# DECORATORS - Convenient decorators for logging
# =============================================================================


F = TypeVar("F", bound=Callable[..., Any])


def log_function_call(
    level: FlextLogLevel = FlextLogLevel.DEBUG,
    *,
    include_args: bool = True,
    include_result: bool = True,
    include_exception: bool = True,
) -> Callable[[F], F]:
    """Decorator to log function calls with arguments and results.

    Args:
        level: Log level for function call logs
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        include_exception: Whether to log exceptions

    Example:
        >>> from flext_core.patterns.logging import log_function_call, FlextLogLevel
        >>> @log_function_call(level=FlextLogLevel.INFO)
        >>> def process_data(data: dict) -> dict:
        ...     return {"processed": True, **data}
        >>> @log_function_call(include_args=False, include_result=False)
        >>> def simple_function():
        ...     pass
    """

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            logger = get_logger(func.__module__)

            # Log function call
            func_name = func.__name__
            if include_args:
                logger.log(
                    level,
                    "Calling %s",
                    func_name=func_name,
                    args=args,
                    kwargs=kwargs,
                )
            else:
                logger.log(level, "Calling %s", func_name=func_name)

            try:
                result = func(*args, **kwargs)

                # Log result
                if include_result:
                    logger.log(
                        level,
                        "Function %s completed successfully",
                        func_name=func_name,
                        result=result,
                    )
                else:
                    logger.log(
                        level,
                        "Function %s completed successfully",
                        func_name=func_name,
                    )
                return result  # noqa: TRY300

            except Exception as e:
                # Log exception
                if include_exception:
                    logger.exception(
                        "Function %s failed with exception",
                        func_name=func_name,
                        exception=e,
                    )
                else:
                    logger.exception(
                        "Function %s failed",
                        func_name=func_name,
                    )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


C = TypeVar("C")


def log_class_methods(
    level: FlextLogLevel = FlextLogLevel.DEBUG,
    *,
    include_args: bool = True,
    include_result: bool = True,
    include_exception: bool = True,
) -> Callable[[type[C]], type[C]]:
    """Decorator to log all method calls in a class.

    Args:
        level: Log level for method call logs
        include_args: Whether to log method arguments
        include_result: Whether to log method result
        include_exception: Whether to log exceptions

    Example:
        >>> from flext_core.patterns.logging import log_class_methods, FlextLogLevel
        >>> @log_class_methods(level=FlextLogLevel.INFO)
        >>> class DataProcessor:
        ...     def process(self, data):
        ...         return {"processed": True, **data}
        >>> @log_class_methods(include_args=False)
        >>> class SimpleClass:
        ...     def method(self):
        ...         pass
    """

    def decorator(cls: type[C]) -> type[C]:
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith("_"):
                # Decorate the method
                setattr(
                    cls,
                    attr_name,
                    log_function_call(
                        level=level,
                        include_args=include_args,
                        include_result=include_result,
                        include_exception=include_exception,
                    )(attr),
                )
        return cls

    return decorator


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Global configuration
    "FlextGlobalLoggingConfig",
    # Core classes
    "FlextLogContext",
    "FlextLogLevel",  # Re-export from constants
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggerMixin",
    # Logger implementations
    "FlextStandardLogger",
    "FlextStructLogger",
    # Structlog configuration
    "FlextStructlogConfig",
    "bind_context",
    # Utility functions
    "configure_logging",
    "create_context_from_dict",
    "get_global_config",
    "get_logger",
    # Structlog utilities
    "get_structlog_logger",
    "log_class_methods",
    # Decorators
    "log_function_call",
    "new_context",
    "set_log_level",
    # Re-export structlog for convenience
    "structlog",
]
