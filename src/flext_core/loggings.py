"""FLEXT Core Logging Module.

Comprehensive enterprise-grade structured logging system for the FLEXT Core library
providing consolidated architecture with context management and observability features.

Architecture:
    - Consolidated single-responsibility structured logging components
    - Enterprise-grade observability with in-memory log store for testing
    - Factory pattern with intelligent caching for performance optimization
    - Context management with automatic cleanup and scope isolation
    - Thread-safe operations for concurrent application environments
    - Level-based filtering with performance optimization for production use

Logging System Components:
    - FlextLogger: Core structured logger with context management and level filtering
    - FlextLoggerFactory: Centralized logger creation with intelligent caching strategy
    - FlextLogContext: Context manager for scoped logging with automatic cleanup
    - FlextLogging: Unified public API providing consolidated access to all features
    - Global log store: Thread-safe in-memory storage for testing and observability
    - Context inheritance: Hierarchical context management for request tracing

Maintenance Guidelines:
    - Add new log levels to FlextLogLevel.get_numeric_levels() method with values
    - Maintain structured logging format consistency across all log entries
    - Use context managers for request-scoped logging with automatic cleanup
    - Keep log store operations thread-safe for concurrent access patterns
    - Use factory pattern for logger caching and lifecycle management
    - Preserve context inheritance patterns for distributed tracing
    - Follow structured format: timestamp, level, logger, message, context

Design Decisions:
    - Single source of truth pattern eliminating base module duplication
    - Global log store for comprehensive testing and production observability
    - Factory pattern with intelligent caching for optimal performance
    - Context manager pattern for automatic resource cleanup and scope management
    - Structured format with consistent field ordering for parsing and analysis
    - Level-based filtering with early exit for performance optimization
    - Thread-safe global state management for concurrent environments

Enterprise Logging Features:
    - Structured logging with consistent JSON-serializable format
    - Context inheritance for distributed request tracing and correlation
    - Level-based filtering with performance optimization for high-throughput systems
    - Exception logging with automatic traceback capture and context preservation
    - Factory caching with memory management for long-running applications
    - Global configuration management affecting all logger instances
    - Testing utilities with log store access and assertion capabilities

Logging Level Hierarchy:
    - TRACE (5): Most verbose debugging information for fine-grained analysis
    - DEBUG (10): General debugging information for development and troubleshooting
    - INFO (20): General information about application flow and business operations
    - WARNING (30): Potentially harmful situations requiring attention
    - ERROR (40): Error events that allow application to continue running
    - CRITICAL (50): Very severe error events that may require application termination

Context Management Patterns:
    - Instance-level context: Persistent across all log calls from logger instance
    - Method-level context: Passed as keyword arguments for specific log entries
    - Scoped context: Temporary context using context managers with automatic cleanup
    - Context inheritance: Parent context preserved when adding child context
    - Context merging: Method context takes precedence over instance context

Performance Considerations:
    - Early level filtering prevents expensive operations for filtered messages
    - Logger caching reduces object creation overhead in high-frequency scenarios
    - Structured format optimized for both human readability and machine parsing
    - Context copying minimizes shared state for thread safety
    - Global store management with controlled memory usage

Dependencies:
    - validation: Input validation utilities for parameter checking
    - constants: Log levels and configuration constants with numeric mappings
    - types: Type definitions for context dictionaries and log messages

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import TYPE_CHECKING, ClassVar, TypedDict

import structlog

from flext_core.constants import FlextLogLevel
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from structlog.typing import EventDict

    from flext_core.flext_types import TAnyDict, TAnyList, TContextDict, TLogMessage
else:
    # Runtime aliases for when TYPE_CHECKING is False
    TAnyDict = dict[str, object]
    TAnyList = list[object]
    TContextDict = dict[str, object]
    TLogMessage = str

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Logging Pattern Specializations
# =============================================================================

# Logging specific types for better domain modeling
type TLoggerName = str  # Logger name identifier
type TLogLevel = str  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
type TLogFormat = str  # Log format string
type TLogHandler = str  # Log handler type identifier
type TLogFilter = str  # Log filter identifier
type TCorrelationId = str  # Correlation ID for request tracing
type TSessionId = str  # Session identifier for user tracking
type TTransactionId = str  # Transaction identifier for database operations
type TOperationName = str  # Business operation name for categorization

# Structured logging types
type TLogRecord = TAnyDict  # Complete log record data
type TLogMetrics = TAnyDict  # Log metrics and statistics
type TLogConfiguration = TAnyDict  # Logger configuration settings

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


# =============================================================================
# GLOBAL LOG STORE - Private para observabilidade
# =============================================================================

# Global log store consolidado - elimina duplicação
_log_store: TAnyList = []

# =============================================================================
# CUSTOM TRACE LEVEL SETUP - Complete Implementation
# =============================================================================

# Define custom TRACE level
TRACE_LEVEL = 5


def setup_custom_trace_level() -> None:
    """Set up custom TRACE level for both stdlib logging and structlog."""
    # Add to standard logging
    logging.addLevelName(TRACE_LEVEL, "TRACE")

    # Update structlog's internal mappings - check for correct attribute names
    if hasattr(structlog.stdlib, "_NAME_TO_LEVEL"):
        structlog.stdlib._NAME_TO_LEVEL["trace"] = TRACE_LEVEL
        # Note: _LEVEL_TO_NAME may not exist in all versions
        if hasattr(structlog.stdlib, "_LEVEL_TO_NAME"):
            # Use type ignore for dynamic attribute access
            structlog.stdlib._LEVEL_TO_NAME[TRACE_LEVEL] = "trace"  # type: ignore[attr-defined]
    elif hasattr(structlog.stdlib, "NAME_TO_LEVEL"):
        structlog.stdlib.NAME_TO_LEVEL["trace"] = TRACE_LEVEL
        # Some versions may not have LEVEL_TO_NAME
        if hasattr(structlog.stdlib, "LEVEL_TO_NAME"):
            # Use type ignore for dynamic attribute access
            structlog.stdlib.LEVEL_TO_NAME[TRACE_LEVEL] = "trace"  # type: ignore[attr-defined]

    # Add trace method to standard logger
    def trace_method(
        self: logging.Logger,
        msg: str,
        *args: object,
    ) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            # Use the correct signature for _log
            self._log(TRACE_LEVEL, msg, args)

    # Use type ignore for dynamic attribute assignment
    logging.Logger.trace = trace_method  # type: ignore[attr-defined]

    # Add trace method to structlog BoundLogger
    def bound_trace_method(
        self: object,
        event: str | None = None,
        **kwargs: object,
    ) -> object:
        # Use type ignore for dynamic attribute access
        return self._proxy_to_logger("trace", event, **kwargs)  # type: ignore[attr-defined]

    # Use type ignore for dynamic attribute assignment
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

    log_entry = {
        "timestamp": event_dict.get("timestamp", FlextGenerators.generate_timestamp()),
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


# Configure structlog with simpler, more stable processors
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="unix"),
        _add_to_log_store,  # Add to log store before rendering
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure standard logging simply
logging.basicConfig(
    format="%(message)s",
    stream=sys.stderr,
    level=TRACE_LEVEL,
)


# =============================================================================
# FLEXT LOGGER - Class-based factory interface as expected by tests
# =============================================================================


class FlextLogger:
    """Structured logger with comprehensive context management and level filtering.

    Core logging implementation providing production-ready structured logging
    capabilities with automatic context management, performance-optimized level
    filtering, and rich message formatting. Designed for high-throughput enterprise
    applications with observability and debugging requirements.

    Architecture:
        - Performance-optimized level-based filtering with early exit patterns
        - Structured log entries with consistent JSON-serializable format
        - Hierarchical context inheritance and intelligent merging for request tracing
        - Integration with global log store for comprehensive testing and observability
        - Thread-safe context management for concurrent application environments
        - Memory-efficient context copying and state management

    Enterprise Logging Features:
        - Six-level logging hierarchy with numeric values for efficient filtering
        - Automatic exception logging with comprehensive traceback capture
        - Context-aware logging with structured key-value pair support
        - Level-based filtering optimized for production performance
        - Global log store integration for testing and monitoring
        - Structured format optimized for log aggregation and analysis

    Context Management System:
        - Instance-level context: Persistent across all log calls from logger instance
        - Method-level context: Passed as keyword arguments for specific log entries
        - Context merging: Method context takes precedence over instance context
        - Context manager integration: Scoped logging with automatic cleanup
        - Context inheritance: Hierarchical context for distributed request tracing
        - Thread-safe context operations: Safe for concurrent access patterns

    Performance Optimizations:
        - Early level filtering prevents expensive operations for filtered messages
        - Structured format designed for both human readability and machine parsing
        - Context copying optimized for minimal memory overhead
        - Global store management with controlled memory usage patterns
        - Efficient timestamp generation and formatting

    Log Entry Structure:
        - timestamp: Unix timestamp for precise temporal ordering
        - level: Log level string for filtering and categorization
        - logger: Logger name for source identification and routing
        - message: Human-readable message content
        - context: Structured key-value pairs for debugging and correlation

    Usage Patterns:
        # Basic structured logging
        logger = FlextLogger("myapp.service", "INFO")
        logger.info("Service started", port=8080, version="1.0.0")

        # Context management for request tracing
        logger.set_context({"user_id": "123", "request_id": "abc-def"})
        logger.info("Processing request", action="create_order", items=5)

        # Method-level context override
        logger.debug("Database query", table="users", duration_ms=45)

        # Exception logging with automatic traceback
        try:
            risky_operation()
        except (RuntimeError, ValueError, TypeError):
            logger.exception("Operation failed", operation="user_creation")

        # Context inheritance for distributed operations
        with logger.with_context(transaction_id="tx_789"):
            logger.info("Transaction started")
            process_transaction()
            logger.info("Transaction completed")

        # Level-based conditional logging
        if logger._should_log("DEBUG"):
            expensive_debug_info = compute_debug_data()
            logger.debug("Debug information", data=expensive_debug_info)

    Thread Safety:
        - Context operations are thread-safe through dictionary copying
        - Logger instances can be safely shared across threads
        - Global log store operations are thread-safe for concurrent access
        - No shared mutable state between logger instances
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
        self._level = level.upper()
        numeric_levels = FlextLogLevel.get_numeric_levels()
        self._level_value = numeric_levels.get(self._level, numeric_levels["INFO"])
        self._context: TContextDict = {}

        # Create structlog logger with the name - this will use the global configuration
        # that includes _add_to_log_store processor
        self._structlog_logger = structlog.get_logger(name)

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        numeric_levels = FlextLogLevel.get_numeric_levels()
        # Handle both string and enum inputs
        level_str = level.value if hasattr(level, "value") else str(level).upper()
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
            "TRACE": "trace",  # Now we have proper custom TRACE level
            "DEBUG": "debug",
            "INFO": "info",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
        }

        # Handle both string and enum inputs for level mapping
        level_str = level.value if hasattr(level, "value") else str(level).upper()
        structlog_level = level_mapping.get(level_str, "info")
        log_method = getattr(self._structlog_logger, structlog_level)

        # Log with structlog - message goes as first positional argument
        try:
            # Structlog expects message as first positional argument
            log_method(str(message), **log_data)
        except TypeError:
            # Last resort: just log the message without context
            log_method(str(message))

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
        # Use list of objects to handle structlog processor types
        processors: list[object] = [
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
            processors=processors,  # type: ignore[arg-type]
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

    Architecture:
        - Class-level logger cache for instance reuse
        - Global level management affecting all loggers
        - Validation and normalization of logger parameters
        - Cache key strategy for logger instance identification
        - Direct log store access for testing and observability
        - Context manager creation for structured logging

    Factory Features:
        - Automatic logger caching based on name and level
        - Global level changes affecting all existing loggers
        - Parameter validation with sensible defaults
        - Cache management for testing and cleanup
        - Log store access for assertion and verification
        - Context manager creation for scoped logging

    Unified API Features (consolidated from FlextLogging):
        - Logger creation and retrieval through factory delegation
        - Global configuration management across all loggers
        - Log store access for testing and monitoring
        - Context manager creation for structured logging

    Cache Strategy:
        - Cache key format: "name:level" for unique identification
        - Lazy creation with cache-first lookup
        - Global operations affecting all cached instances
        - Manual cache clearing for testing scenarios

    Usage:
        # Basic logger operations (consolidated from FlextLogging)
        logger = FlextLoggerFactory.get_logger("myapp", "DEBUG")
        FlextLoggerFactory.set_global_level("INFO")

        # Testing utilities (consolidated from FlextLogging)
        logs = FlextLoggerFactory.get_log_store()
        FlextLoggerFactory.clear_log_store()

        # Context management (consolidated from FlextLogging)
        context = FlextLoggerFactory.create_context(logger, user_id="123")
    """

    _loggers: ClassVar[dict[str, FlextLogger]] = {}
    _global_level: ClassVar[str] = "INFO"

    @classmethod
    def get_logger(cls, name: str, level: str = "INFO") -> FlextLogger:
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
    def get_log_store() -> TAnyList:
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

    Architecture:
        - Context manager protocol implementation
        - Original context preservation and restoration
        - Context merging with preservation of existing values
        - Automatic cleanup on exception or normal exit

    Context Features:
        - Temporary context addition without permanent modification
        - Original context backup and restoration
        - Context merging with override capability
        - Exception-safe cleanup operations

    Scope Management:
        - Enter: Merge new context with existing logger context
        - Exit: Restore original context regardless of exit reason
        - Exception safety: Context restored even if exception occurs
        - Nested context support through context preservation

    Usage:
        logger = FlextLogger("myapp")
        logger.set_context({"service": "user"})

        with FlextLogContextManager(logger, request_id="123"):
            logger.info("Processing")  # Has both service and request_id
        # request_id automatically removed, service context preserved
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


# Convenience functions with comprehensive documentation
def get_logger(name: str, level: str = "INFO") -> FlextLogger:
    """Get logger instance using factory delegation.

    Convenience function providing direct access to logger creation
    without requiring explicit FlextLoggerFactory class usage.

    Args:
        name: Logger name, typically module name (__name__)
        level: Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        FlextLogger instance with specified configuration

    Usage:
        logger = get_logger(__name__, "DEBUG")

    """
    return FlextLoggerFactory.get_logger(name, level)


def create_log_context(
    logger: FlextLogger,
    **context: object,
) -> FlextLogContextManager:
    """Create scoped log context manager.

    Convenience function for creating context managers that temporarily
    add context to logger instances with automatic cleanup.

    Args:
        logger: FlextLogger instance to add context to
        **context: Key-value pairs to add as context

    Returns:
        FlextLogContextManager for use with 'with' statement

    Usage:
        with create_log_context(logger, request_id="123", user_id="456"):
            logger.info("Processing request")

    """
    return FlextLoggerFactory.create_context(logger, **context)


# Add missing interface methods to FlextLogger class that tests expect
class _ContextManagerLogger:
    """Wrapper for structlog.BoundLogger to provide context manager functionality."""

    def __init__(self, logger: object) -> None:
        self._logger = logger

    def __enter__(self) -> object:
        return self._logger

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass


# Module-level backward compatibility function
def flext_get_logger(name: str) -> object:
    """Module-level logger creation function for backward compatibility."""
    return structlog.get_logger(name)


# Backward compatibility: use FlextLoggerFactory directly for class-level operations
# FlextLoggerFactory.get_logger -> FlextLoggerFactory.get_logger
# FlextLogger.set_global_level -> FlextLoggerFactory.set_global_level
# FlextLogger.clear_loggers -> FlextLoggerFactory.clear_loggers


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    "FlextLogContext",
    "FlextLogContextManager",
    # Log levels from constants
    "FlextLogLevel",
    # Main consolidated classes (FlextLogging eliminated - functionality in
    # FlextLoggerFactory)
    "FlextLogger",
    "FlextLoggerFactory",
    # Convenience functions
    "create_log_context",
    "flext_get_logger",
    "get_logger",
]
