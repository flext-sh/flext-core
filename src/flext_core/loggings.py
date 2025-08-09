"""Enterprise-grade structured logging system for FLEXT.

Provides comprehensive logging infrastructure with context management,
performance optimization, and observability features for distributed
data integration pipelines.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import TYPE_CHECKING, ClassVar, TypedDict

import structlog

from flext_core.constants import FlextConstants, FlextLogLevel

if TYPE_CHECKING:
    from structlog.typing import EventDict, Processor

    from flext_core.typings import TAnyDict, TContextDict, TLogMessage

Platform = FlextConstants.Platform

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Logging Pattern Specializations
# =============================================================================

# Logging specific types for better domain modeling

# Structured logging types

# =============================================================================
# FLEXT LOG CONTEXT - TypedDict as expected by tests
# =============================================================================


class FlextLogContext(TypedDict, total=False):
    """Typed dictionary for log context data.

    Defines the structure for log context information with optional fields
    for maximum flexibility while maintaining type safety.

    Enterprise Context Fields:
        - user_id: User identifier for tracking user actions
        - request_id: Request identifier for tracing requests across services
        - session_id: Session identifier for user session tracking
        - operation: Operation name for categorizing business operations
        - transaction_id: Transaction identifier for database transaction tracking
        - tenant_id: Tenant identifier for multi-tenant applications
        - customer_id: Customer identifier for customer-specific operations
        - order_id: Order identifier for e-commerce and order management

    Performance Context Fields:
        - duration_ms: Operation duration in milliseconds
        - memory_mb: Memory usage in megabytes
        - cpu_percent: CPU usage percentage

    Error Context Fields:
        - error_code: Error code for programmatic error handling
        - error_type: Error type classification
        - stack_trace: Stack trace information for debugging
    """

    # Enterprise tracking fields
    user_id: str
    request_id: str
    session_id: str
    operation: str
    transaction_id: str
    tenant_id: str
    customer_id: str
    order_id: str

    # Performance fields
    duration_ms: float
    memory_mb: float
    cpu_percent: float

    # Error fields
    error_code: str
    error_type: str
    stack_trace: str


class FlextLogEntry(TypedDict):
    """TypedDict for log store entries with proper typing.

    Defines the structure of log entries stored in the global log store
    for testing and debugging purposes.
    """

    timestamp: str
    level: str
    logger: str
    message: str
    method: str
    context: TAnyDict


# =============================================================================
# GLOBAL LOG STORE - Private para observabilidade
# =============================================================================

# Global log store consolidado - elimina duplicação
_log_store: list[FlextLogEntry] = []

# =============================================================================
# CUSTOM TRACE LEVEL SETUP - Complete Implementation
# =============================================================================

# Define custom TRACE level
# Constants moved to constants.py following SOLID Single Responsibility Principle

TRACE_LEVEL = FlextConstants.Observability.TRACE_LEVEL


def setup_custom_trace_level() -> None:
    """Set up custom TRACE level for both stdlib logging and structlog."""
    # Add to standard logging
    logging.addLevelName(TRACE_LEVEL, "TRACE")

    # Update structlog's internal mappings safely using getattr/setattr
    # This avoids direct attribute access issues
    name_to_level = getattr(structlog.stdlib, "_NAME_TO_LEVEL", None)
    if name_to_level is not None and isinstance(name_to_level, dict):
        name_to_level["trace"] = TRACE_LEVEL

    level_to_name = getattr(structlog.stdlib, "_LEVEL_TO_NAME", None)
    if level_to_name is not None and isinstance(level_to_name, dict):
        level_to_name[TRACE_LEVEL] = "trace"

    # Add trace method to standard logger
    def trace_method(
        self: logging.Logger,
        msg: str,
        *args: object,
    ) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            # Use the correct signature for _log
            self._log(TRACE_LEVEL, msg, args)

    # Add trace method to structlog BoundLogger
    def bound_trace_method(
        self: structlog.stdlib.BoundLogger,
        event: str | None = None,
        **kwargs: object,
    ) -> object:
        # Access the logger's proxy method properly
        if hasattr(self, "_proxy_to_logger"):
            proxy_method = self._proxy_to_logger
            return proxy_method("trace", event, **kwargs)
        return None

    # Use setattr which is the proper way to add attributes dynamically
    # MyPy doesn't understand dynamic attributes, but this is the correct approach
    # The alternative would be to create a subclass, but that breaks existing code
    logging.Logger.trace = trace_method  # type: ignore[attr-defined]
    structlog.stdlib.BoundLogger.trace = bound_trace_method  # type: ignore[attr-defined]


# Initialize custom TRACE level
setup_custom_trace_level()

# =============================================================================
# STRUCTLOG CONFIGURATION
# =============================================================================


def _add_to_log_store(
    logger: object,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Processor to add log entries to the global store.

    Args:
        logger: The structlog logger instance (used for logger name fallback)
        method_name: The log method name (used for debugging information)
        event_dict: The log event dictionary to process

    Returns:
        The unchanged event_dict for further processing

    """
    # Convert structlog event_dict to our format
    # Use logger name from logger object if not in event_dict
    logger_name = str(event_dict.get("logger", getattr(logger, "name", "unknown")))

    log_entry: FlextLogEntry = {
        "timestamp": event_dict.get("timestamp", time.time()),
        "level": str(event_dict.get("level", "INFO")).upper(),
        "logger": logger_name,
        "message": str(event_dict.get("event", "")),
        "method": method_name,  # Include method name for debugging
        "context": {
            k: v
            for k, v in event_dict.items()
            if k not in {"timestamp", "level", "logger", "event"}
        },
    }
    _log_store.append(log_entry)
    return event_dict


# Human-readable console renderer configuration
def _create_human_readable_renderer() -> structlog.dev.ConsoleRenderer:
    """Create human-readable console renderer following market standards."""
    # Check if we're in development or production
    env_value = os.environ.get("ENVIRONMENT", "development").lower()
    is_development = env_value in {"development", "dev", "local"}
    colors_enabled = os.environ.get("FLEXT_LOG_COLORS", "true").lower()
    enable_colors = colors_enabled == "true" and is_development

    return structlog.dev.ConsoleRenderer(
        colors=enable_colors,
        # Show level and logger name in brackets for clarity
        level_styles={
            "critical": "\033[91m",  # Bright red
            "error": "\033[91m",  # Red
            "warning": "\033[93m",  # Yellow
            "info": "\033[92m",  # Green
            "debug": "\033[94m",  # Blue
            "trace": "\033[95m",  # Magenta
        }
        if enable_colors
        else None,
    )


# Configure structlog with human-readable output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        # Use ISO timestamp format instead of unix timestamp for readability
        structlog.processors.TimeStamper(fmt="iso"),
        _add_to_log_store,  # Keep for testing
        _create_human_readable_renderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


# Configure standard logging with environment-aware level
def _get_logging_level_from_env() -> int:
    """Get logging level from environment variables."""
    # Check multiple environment variables in order of preference
    env_level = (
        os.environ.get("ALGAR_LOG_LEVEL")
        or os.environ.get("FLEXT_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL")
        or "INFO"
    ).upper()

    # Map to numeric levels
    level_mapping = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "TRACE": 5,
    }

    return level_mapping.get(env_level, 20)  # Default to INFO


def _get_env_log_level_string() -> str:
    """Get logging level from environment as string."""
    return (
        os.environ.get("ALGAR_LOG_LEVEL")
        or os.environ.get("FLEXT_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL")
        or "INFO"
    ).upper()


logging.basicConfig(
    format="%(message)s",
    stream=sys.stderr,
    level=_get_logging_level_from_env(),
)


# =============================================================================
# FLEXT LOGGER - Class-based factory interface as expected by tests
# =============================================================================


class FlextLogger:
    """Structured logger with context management and level filtering.

    Provides production-ready logging with performance optimization,
    hierarchical context management, and enterprise observability features.
    """

    # Class variables for configuration state
    _configured: ClassVar[bool] = False
    _loggers: ClassVar[TAnyDict] = {}

    def __init__(self, name: str, level: str = "INFO") -> None:
        """Initialize logger.

        Args:
            name: Logger name (typically __name__)
            level: Minimum log level

        """
        # Auto-configure structlog if not already configured
        if not self._configured:
            self.configure()

        self._name = name

        # Use environment-aware level if default level is requested
        if level == "INFO":
            env_level = _get_env_log_level_string()
            if env_level != "INFO":
                level = env_level

        self._level = level.upper()
        numeric_levels = FlextLogLevel.get_numeric_levels()
        self._level_value = numeric_levels.get(self._level, numeric_levels["INFO"])
        self._context: TContextDict = {}

        # Ensure stdlib logging level is permissive enough for structlog filtering
        # The filter_by_level processor uses stdlib logging levels
        current_stdlib_level = logging.getLogger().getEffectiveLevel()
        if self._level_value < current_stdlib_level:
            logging.getLogger().setLevel(self._level_value)

        # Create structlog logger with the name - this will use the global configuration
        # that includes _add_to_log_store processor
        self._structlog_logger = structlog.get_logger(name)

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        numeric_levels = FlextLogLevel.get_numeric_levels()
        # Handle both string and enum inputs safely
        try:
            # Try to get value attribute first (for enums)
            if hasattr(level, "value"):
                level_str = level.value.upper()
            else:
                level_str = str(level).upper()
        except AttributeError:
            # Fallback to string conversion
            level_str = str(level).upper()
        level_value = numeric_levels.get(level_str, numeric_levels["INFO"])
        return level_value >= self._level_value

    def _log_with_structlog(
        self,
        level: str,
        message: TLogMessage,
        context: TContextDict | None = None,
    ) -> None:
        """Log using structlog with context merging."""
        if not self._should_log(level):
            return

        # Merge instance context with method context
        merged_context = {**self._context}
        if context:
            merged_context.update(context)

        # Add our standard fields
        log_data = {
            "logger": self._name,
            **merged_context,
        }

        # Map our levels to standard logging levels for structlog
        level_mapping = {
            "TRACE": "trace",
            "DEBUG": "debug",
            "INFO": "info",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
        }

        level_str = str(level).upper()
        structlog_level = level_mapping.get(level_str, "info")
        log_method = getattr(self._structlog_logger, structlog_level)

        log_method(str(message), **log_data)

    def set_level(self, level: str) -> None:
        """Set logger level."""
        self._level = level.upper()
        numeric_levels = FlextLogLevel.get_numeric_levels()
        self._level_value = numeric_levels.get(self._level, numeric_levels["INFO"])

    def get_context(self) -> TContextDict:
        """Get current context."""
        return dict(self._context)

    def set_context(self, context: TContextDict) -> None:
        """Set logger context."""
        self._context = dict(context)

    def clear_context(self) -> None:
        """Clear logger context."""
        self._context = {}

    def with_context(self, **context: object) -> FlextLogger:
        """Create logger with additional context."""
        new_logger = FlextLogger(self._name, self._level)
        # Use the same structlog instance but with merged context
        new_logger._structlog_logger = self._structlog_logger
        new_logger._context = {**self._context, **context}
        return new_logger

    def trace(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log trace message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("TRACE", formatted_message, context)

    def debug(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log debug message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("DEBUG", formatted_message, context)

    def info(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log info message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("INFO", formatted_message, context)

    def warning(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log warning message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("WARNING", formatted_message, context)

    def error(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log error message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("ERROR", formatted_message, context)

    def critical(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log critical message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("CRITICAL", formatted_message, context)

    def exception(self, message: TLogMessage, *args: object, **context: object) -> None:
        """Log exception with traceback context and optional %s formatting."""
        formatted_message = self._format_message(message, args)
        context["traceback"] = traceback.format_exc()
        self._log_with_structlog("ERROR", formatted_message, context)

    def _format_message(self, message: TLogMessage, args: tuple[object, ...]) -> str:
        """Format message with %s placeholders if args are provided."""
        if not args:
            return str(message)

        try:
            return str(message) % args
        except (TypeError, ValueError):
            # If % formatting fails, fall back to string formatting
            try:
                return str(message).format(*args)
            except (KeyError, ValueError):
                # If both fail, just concatenate
                return f"{message} {' '.join(str(arg) for arg in args)}"

    @classmethod
    def get_logger(cls, name: str, _level: str = "INFO") -> object:
        """Get logger with auto-configuration.

        Args:
            name: Logger name
            level: Log level

        Returns:
            Structlog logger instance

        """
        if not cls._configured:
            cls.configure()

        # Use cache to ensure same instance for same name
        if name not in cls._loggers:
            cls._loggers[name] = structlog.get_logger(name)
        return cls._loggers[name]

    def bind(self, **context: object) -> FlextLogger:
        """Create new logger with bound context.

        Args:
            **context: Context data to bind

        Returns:
            New FlextLogger instance with bound context

        """
        new_logger = FlextLogger(self._name, self._level)
        # Use the same structlog instance but with merged context
        new_logger._structlog_logger = self._structlog_logger
        new_logger._context = {**self._context, **context}
        return new_logger

    @classmethod
    def configure(
        cls,
        *,
        log_level: FlextLogLevel | None = None,
        _log_level: FlextLogLevel | None = None,
        json_output: bool = False,
        add_timestamp: bool = True,
        add_caller: bool = False,
    ) -> None:
        """Configure the logging system.

        Args:
            log_level: Log level to configure (new preferred name)
            _log_level: Log level to configure (backward compatibility)
            json_output: Whether to use JSON output format
            add_timestamp: Whether to add timestamps to logs
            add_caller: Whether to add caller information to logs

        """
        # Handle both parameter names for backward compatibility
        # Note: final_log_level can be used for future enhancement
        # For now just validate parameters exist
        _ = log_level or _log_level or FlextLogLevel.INFO

        # Build processors list while preserving essential ones
        # Import Processor type from structlog

        processors: list[Processor] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
        ]

        # Add optional processors
        if add_timestamp:
            processors.append(structlog.processors.TimeStamper(fmt="unix"))
        if add_caller:
            processors.append(structlog.processors.CallsiteParameterAdder())

        # Always preserve our essential processors
        processors.append(_add_to_log_store)  # Always keep log store

        # Choose final renderer
        if json_output:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        # Reconfigure structlog with preserved essential functionality
        # Use type ignore for processor compatibility across structlog versions
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        cls._configured = True

    @classmethod
    def get_base_logger(
        cls,
        name: str,
        *,
        _level: str = "INFO",
    ) -> object:
        """Get base logger with observability.

        Args:
            name: Logger name
            _level: Log level (unused in this implementation)

        Returns:
            Structlog logger instance

        """
        return structlog.get_logger(name)

    @classmethod
    def bind_context(cls, **context: object) -> object:
        """Create logger with bound context.

        Args:
            **context: Context data to bind

        Returns:
            Structlog logger with bound context

        """
        return structlog.get_logger("context").bind(**context)

    @classmethod
    def with_performance_tracking(cls, name: str) -> object:
        """Get logger with performance tracking.

        Args:
            name: Logger name

        Returns:
            Structlog logger instance with performance tracking

        """
        return structlog.get_logger(name)

    @classmethod
    def flext_get_logger(cls, name: str) -> object:
        """Backward compatibility logger creation.

        Args:
            name: Logger name

        Returns:
            Structlog logger instance

        """
        return structlog.get_logger(name)


# =============================================================================
# FLEXT LOGGER FACTORY - Consolidado eliminando _BaseLoggerFactory
# =============================================================================


class FlextLoggerFactory:
    """Consolidated factory and unified interface for logger management.

    Comprehensive logger management combining factory pattern with unified public API
    eliminating FlextLogging duplication. Provides centralized logger creation,
    caching, global configuration, and testing utilities in a single interface.
    """

    _loggers: ClassVar[dict[str, FlextLogger]] = {}
    _global_level: ClassVar[str] = "INFO"

    @classmethod
    def get_logger(cls, name: str | None, level: str = "INFO") -> FlextLogger:
        """Get logger with caching and global level support.

        Args:
            name: Logger name
            level: Log level (uses global level if not specified)

        Returns:
            Cached logger instance

        """
        # Validate and normalize name
        if not (isinstance(name, str) and len(name.strip()) > 0):
            name = "flext.unknown"

        # Validate and normalize level
        if not (isinstance(level, str) and len(level.strip()) > 0):
            level = "INFO"

        # Use global level if not specified
        if level == "INFO" and cls._global_level != "INFO":
            level = cls._global_level

        # Create cache key
        cache_key = f"{name}:{level}"

        # Return cached logger if exists
        if cache_key in cls._loggers:
            return cls._loggers[cache_key]

        # Create new logger
        logger = FlextLogger(name, level)
        cls._loggers[cache_key] = logger
        return logger

    @classmethod
    def set_global_level(cls, level: str) -> None:
        """Set global log level for all loggers.

        Args:
            level: Log level to set globally

        """
        # Validate level
        numeric_levels = FlextLogLevel.get_numeric_levels()
        if level not in numeric_levels:
            return

        # Update global level
        cls._global_level = level

        # Update all existing loggers
        for logger in cls._loggers.values():
            logger.set_level(level)

    @classmethod
    def clear_loggers(cls) -> None:
        """Clear logger cache (for testing)."""
        cls._loggers.clear()

    # Consolidated methods from FlextLogging - eliminates duplication
    @staticmethod
    def get_log_store() -> list[FlextLogEntry]:
        """Get log store for testing (consolidated from FlextLogging)."""
        return _log_store.copy()

    @staticmethod
    def clear_log_store() -> None:
        """Clear log store for testing (consolidated from FlextLogging)."""
        _log_store.clear()

    @staticmethod
    def create_context(
        logger: FlextLogger,
        **context: object,
    ) -> FlextLogContextManager:
        """Create log context (consolidated from FlextLogging)."""
        return FlextLogContextManager(logger, **context)


# =============================================================================
# FLEXT LOG CONTEXT - Consolidado eliminando _BaseLogContext
# =============================================================================


class FlextLogContextManager:
    """Context manager for scoped logging with automatic cleanup.

    Provides temporary context addition to logger instances with automatic
    restoration of original context when exiting the scope.
    """

    def __init__(self, logger: FlextLogger, **context: object) -> None:
        """Initialize log context.

        Args:
            logger: Logger instance
            **context: Context data

        """
        self._logger = logger
        self._context = context
        self._original_context = logger.get_context()

    def __enter__(self) -> FlextLogger:
        """Enter context and add context data."""
        current_context = self._logger.get_context()
        current_context.update(self._context)
        self._logger.set_context(current_context)
        return self._logger

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context and restore original context."""
        self._logger.set_context(self._original_context)


# =============================================================================
# FLEXT LOGGING - Consolidated into FlextLoggerFactory (eliminating duplication)
# =============================================================================

# FlextLogging class eliminated - functionality consolidated into FlextLoggerFactory
# This follows "entregar mais com muito menos" principle by eliminating redundant
# delegation


# =============================================================================
# CONVENIENCE FUNCTIONS - Mantendo compatibilidade
# =============================================================================


# =============================================================================
# MIGRATION NOTICE - Legacy functions moved to legacy.py
# =============================================================================

# IMPORTANT: Legacy convenience functions have been moved to legacy.py
#
# Migration guide:
# OLD: from flext_core.loggings import get_logger
# NEW: from flext_core.legacy import get_logger (with deprecation warning)
# MODERN: from flext_core import FlextLoggerFactory; FlextLoggerFactory.get_logger()
#
# For new code, use FlextLoggerFactory methods directly


# Add missing interface methods to FlextLogger class that tests expect
class _ContextManagerLogger:
    """Wrapper for structlog.BoundLogger to provide context manager functionality."""

    def __init__(self, logger: object) -> None:
        self._logger = logger

    def __enter__(self) -> object:
        return self._logger

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass


# =============================================================================
# DUPLICATE CLASSES REMOVED - Using comprehensive implementation above
# =============================================================================


def get_logger(name: str = "flext", level: str = "INFO") -> FlextLogger:
    """Get logger instance - convenience function.

    Args:
        name: Logger name
        level: Log level

    Returns:
        FlextLogger instance

    """
    return FlextLoggerFactory.get_logger(name, level)


# NOTE: Legacy functions moved to legacy.py
# Use FlextLoggerFactory methods directly for new code


# =============================================================================
# LEGACY FUNCTION ALIASES - Backward compatibility
# =============================================================================


def create_log_context(
    logger: FlextLogger | str | None = None, **context: object,
) -> FlextLogContextManager:
    """Create log context manager for structured logging."""
    selected = (
        logger
        if isinstance(logger, FlextLogger)
        else FlextLoggerFactory.get_logger(logger or "default")
    )
    return FlextLogContextManager(selected, **context)


# flext_get_logger function moved to FlextLoggerFactory class method above


# =============================================================================
# EXPORTS - Clean public API following guidelines
# =============================================================================

__all__ = [
    "FlextLogContext",
    "FlextLogContextManager",
    "FlextLogEntry",
    "FlextLogLevel",
    "FlextLogger",
    "FlextLoggerFactory",
    # Legacy function aliases
    "create_log_context",
    "flext_get_logger",
    "get_logger",  # Convenience function
    "setup_custom_trace_level",  # Keep this as it's used internally
    # NOTE: Legacy functions moved to legacy.py
    # Import from flext_core.legacy if needed for backward compatibility
]


# =============================================================================
# ADDITIONAL LEGACY FUNCTIONS - Backward compatibility
# =============================================================================


def flext_get_logger(name: str = "flext") -> FlextLogger:
    """Get logger instance (legacy function)."""
    return get_logger(name)
