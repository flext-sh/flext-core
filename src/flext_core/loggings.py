"""Structured logging utilities enabling the context-first pillar for 1.0.0.

The module provides a thin wrapper around structlog with FLEXT-specific
context management and configuration.

Dependency Layer: 2 (Foundation Logging)
Dependencies: structlog, result, typings
Used by: All Flext modules requiring logging

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import traceback
import types
from collections.abc import Callable, Sequence
from typing import Self

from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

structlog = FlextRuntime.structlog()

# =============================================================================
# FLEXT LOGGER - THIN WRAPPER AROUND STRUCTLOG
# =============================================================================


class FlextLogger:
    """Advanced structured logging with context management and dependency injection.

    **FLEXT-CORE ADVANCED PATTERNS**:

    🚀 STRUCTLOG ADVANCED FEATURES
    - contextvars for automatic context propagation
    - Advanced processor chains (callsite, tracebacks, timestamps)
    - Bound logger factories with automatic context injection
    - Performance tracking and correlation IDs
    - Integration with FlextResult for automatic error logging

    🔧 DEPENDENCY INJECTION READY
    - Injectable via dependency_injector providers
    - Singleton and Factory patterns supported
    - Module-specific logger factories
    - Automatic service context binding

    **Function**: Enterprise-grade structured logging
        - Direct structlog integration with advanced processors
        - Automatic context propagation via contextvars
        - FlextResult-wrapped logging methods
        - Distributed tracing and correlation tracking
        - Performance monitoring integration
        - Clean Layer 2 dependency (no config imports)

    **Uses**: Advanced foundation patterns
        - structlog.contextvars for context propagation
        - structlog.processors for advanced processing
        - FlextResult for railway pattern error handling
        - FlextTypes for type definitions
        - dependency_injector providers (via FlextRuntime)

    **Example**: Advanced usage patterns
        ```python
        from flext_core import FlextLogger

        # Basic usage with automatic context
        logger = FlextLogger(__name__)
        logger.info("User logged in", user_id="123", action="login")

        # Context binding for request tracking
        FlextLogger.bind_global_context(request_id="req-456", correlation_id="cor-789")

        # Create bound logger with additional context
        request_logger = logger.bind(endpoint="/api/users", method="POST")
        request_logger.info("Processing request")

        # Factory pattern for module loggers
        service_logger = FlextLogger.create_service_logger(
            "user-service", version="1.0.0"
        )

        # Performance tracking
        with logger.track_performance("database_query"):
            # ... database operation
            pass
        ```
    """

    # =========================================================================
    # PRIVATE MEMBERS - Structlog configuration
    # =========================================================================

    _configured: bool = False
    _structlog_configured: bool = False

    @staticmethod
    def _configure_structlog_if_needed(
        log_level: int | None = None,
        *,
        console_enabled: bool = True,
        additional_processors: Sequence[Callable[..., object]] | None = None,
    ) -> None:
        """Configure structlog with advanced processor chain.

        Args:
            log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
            console_enabled: Use console renderer vs JSON renderer
            additional_processors: Extra processors to add to chain

        Note:
            Can be called with FlextConfig values to configure logging:
            >>> from flext_core import FlextConfig
            >>> config = FlextConfig()
            >>> log_level_int = getattr(logging, config.log_level.upper())
            >>> FlextLogger._configure_structlog_if_needed(log_level=log_level_int)

        """
        if FlextLogger._structlog_configured:
            return

        FlextRuntime.configure_structlog(
            log_level=log_level,
            console_renderer=console_enabled,
            additional_processors=additional_processors,
        )

        FlextLogger._structlog_configured = True
        FlextLogger._configured = True

    # =========================================================================
    # ADVANCED FEATURES - Global context management via contextvars
    # =========================================================================

    @classmethod
    def bind_global_context(cls, **context: object) -> FlextResult[None]:
        """Bind context globally using structlog contextvars.

        Context is automatically included in all subsequent log messages
        within the current execution context (thread, async task, etc.).

        Args:
            **context: Key-value pairs to bind globally

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> FlextLogger.bind_global_context(
            ...     request_id="req-123", user_id="usr-456", correlation_id="cor-789"
            ... )
            >>> logger = FlextLogger(__name__)
            >>> logger.info("User action")  # Automatically includes bound context

        """
        try:
            structlog.contextvars.bind_contextvars(**context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to bind global context: {e}")

    @classmethod
    def unbind_global_context(cls, *keys: str) -> FlextResult[None]:
        """Unbind specific keys from global context.

        Args:
            *keys: Context keys to unbind

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            structlog.contextvars.unbind_contextvars(*keys)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to unbind global context: {e}")

    @classmethod
    def clear_global_context(cls) -> FlextResult[None]:
        """Clear all globally bound context.

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            structlog.contextvars.clear_contextvars()
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to clear global context: {e}")

    @classmethod
    def get_global_context(cls) -> dict[str, object]:
        """Get current global context."""
        return dict(structlog.contextvars.get_contextvars())

    @classmethod
    def get_logger(cls) -> FlextLogger:
        """Get a logger instance."""
        return cls.create_module_logger("flext")

    # =========================================================================
    # FACTORY PATTERNS - DI-ready logger creation
    # =========================================================================

    @classmethod
    def create_service_logger(
        cls,
        service_name: str,
        *,
        version: str | None = None,
        correlation_id: str | None = None,
    ) -> FlextLogger:
        """Create logger with service context (DI Factory pattern).

        Args:
            service_name: Service name to include in logs
            version: Optional service version
            correlation_id: Optional correlation ID

        Returns:
            FlextLogger: Logger with service context bound

        Example:
            >>> logger = FlextLogger.create_service_logger(
            ...     "user-service", version="1.0.0"
            ... )

        """
        return cls(
            service_name,
            _service_name=service_name,
            _service_version=version,
            _correlation_id=correlation_id,
        )

    @classmethod
    def create_module_logger(cls, module_name: str) -> FlextLogger:
        """Create logger for Python module (DI Factory pattern).

        Args:
            module_name: Module name (typically __name__)

        Returns:
            FlextLogger: Logger for the module

        """
        return cls(module_name)

    def __init__(
        self,
        name: str,
        *,
        _level: str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize FlextLogger with context.

        Args:
            name: Logger name (typically __name__ or module path)
            _level: Optional log level override (currently unused, for future)
            _service_name: Optional service name override
            _service_version: Optional service version override
            _correlation_id: Optional correlation ID override
            _force_new: Force creation of new instance (for testing)

        """
        super().__init__()

        # Configure structlog if not already configured (NO config dependency)
        FlextLogger._configure_structlog_if_needed()

        # Store logger name for later use
        self._name = name

        # Build initial context
        context = {}
        if _service_name:
            context["service_name"] = _service_name
        if _service_version:
            context["service_version"] = _service_version
        if _correlation_id:
            context["correlation_id"] = _correlation_id

        # Create bound logger with initial context
        self.logger = structlog.get_logger(name).bind(**context)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    @classmethod
    def _create_bound_logger(cls, name: str, bound_logger: object) -> FlextLogger:
        """Internal factory for creating logger with pre-bound structlog instance.

        This factory method allows creating FlextLogger instances with an already
        configured structlog BoundLogger, avoiding the need to access private
        attributes directly.

        Args:
            name: Logger name
            bound_logger: Pre-configured bound structlog logger (object type used
                         as structlog.BoundLogger is not publicly exposed)

        Returns:
            FlextLogger instance with bound logger

        """
        instance = cls.__new__(cls)
        # Use setattr to initialize attributes without triggering descriptor protocol
        setattr(instance, "_name", name)
        setattr(instance, "logger", bound_logger)
        return instance

    def bind(self, **context: object) -> FlextLogger:
        """Bind additional context to the logger.

        Creates a new FlextLogger instance with additional context bound to the
        underlying structlog logger. The original logger remains unchanged.

        Args:
            **context: Context key-value pairs to bind

        Returns:
            New FlextLogger instance with additional context bound

        Example:
            >>> logger = FlextLogger(__name__)
            >>> request_logger = logger.bind(request_id="123", user_id="456")
            >>> request_logger.info("Processing request")

        """
        return FlextLogger._create_bound_logger(self.name, self.logger.bind(**context))

    # =============================================================================
    # LOGGING METHODS - DELEGATE TO STRUCTLOG
    # =============================================================================

    def trace(self, message: str, *args: object, **kwargs: object) -> FlextResult[None]:
        """Log trace message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"

            self.logger.debug(
                formatted_message, **kwargs
            )  # structlog doesn't have trace
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def debug(
        self, message: str, *args: object, **context: object
    ) -> FlextResult[None]:
        """Log debug message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.debug(formatted_message, **context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def info(self, message: str, *args: object, **context: object) -> FlextResult[None]:
        """Log info message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.info(formatted_message, **context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def warning(
        self, message: str, *args: object, **context: object
    ) -> FlextResult[None]:
        """Log warning message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.warning(formatted_message, **context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def error(self, message: str, *args: object, **kwargs: object) -> FlextResult[None]:
        """Log error message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.error(formatted_message, **kwargs)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def critical(
        self, message: str, *args: object, **kwargs: object
    ) -> FlextResult[None]:
        """Log critical message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.critical(formatted_message, **kwargs)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        **kwargs: object,
    ) -> FlextResult[None]:
        """Log exception message with stack trace - LoggerProtocol implementation."""
        try:
            # If a specific exception is provided, format its traceback
            if exception is not None:
                kwargs["stack_trace"] = "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                )
                kwargs["exception_type"] = type(exception).__name__
                kwargs["exception_message"] = str(exception)
            # Otherwise, if exc_info is True, get current exception info
            elif exc_info:
                kwargs["stack_trace"] = traceback.format_exc()
            self.logger.error(message, **kwargs)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    # =========================================================================
    # ADVANCED FEATURES - Performance tracking and result integration
    # =========================================================================

    def track_performance(self, operation_name: str) -> FlextLogger.PerformanceTracker:
        """Track operation performance with automatic logging.

        Returns context manager that automatically logs operation timing.

        Args:
            operation_name: Name of operation being tracked

        Returns:
            PerformanceTracker: Context manager for performance tracking

        Example:
            >>> logger = FlextLogger(__name__)
            >>> with logger.track_performance("database_query"):
            ...     # ... database operation
            ...     pass
            # Automatically logs: "database_query completed in 0.123s"

        """
        return FlextLogger.PerformanceTracker(self, operation_name)

    def log_result(
        self,
        result: FlextResult[object],
        *,
        operation: str | None = None,
        level: str = "info",
    ) -> FlextResult[None]:
        """Log FlextResult with automatic success/failure handling.

        Args:
            result: FlextResult to log
            operation: Optional operation name
            level: Log level for success case (error used for failures)

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> result = validate_user(data)
            >>> logger.log_result(result, operation="user_validation")

        """
        try:
            context: dict[str, object] = {}
            if operation:
                context["operation"] = operation

            if result.is_success:
                msg = f"{operation} succeeded" if operation else "Operation succeeded"
                log_method = getattr(self, level, self.info)
                log_method(msg, **context)
            else:
                msg = (
                    f"{operation} failed: {result.error}"
                    if operation
                    else f"Operation failed: {result.error}"
                )
                context["error_code"] = result.error_code
                context["error_data"] = result.error_data
                self.error(msg, **context)

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to log result: {e}")

    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: FlextLogger, operation_name: str) -> None:
            """Initialize performance tracker.

            Args:
                logger: FlextLogger instance
                operation_name: Name of operation being tracked

            """
            super().__init__()
            self.logger = logger
            self._operation_name = operation_name
            self._start_time: float = 0.0

        def __enter__(self) -> Self:
            """Start performance tracking."""
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            """Complete performance tracking and log results."""
            elapsed = time.time() - self._start_time

            if exc_type is None:
                # Success case
                self.logger.info(
                    f"{self._operation_name} completed",
                    duration_seconds=elapsed,
                    operation=self._operation_name,
                    status="success",
                )
            else:
                # Failure case
                self.logger.error(
                    f"{self._operation_name} failed",
                    duration_seconds=elapsed,
                    operation=self._operation_name,
                    status="failed",
                    exception_type=exc_type.__name__ if exc_type else None,
                    exception_message=str(exc_val) if exc_val else None,
                )


__all__: FlextTypes.StringList = [
    "FlextLogger",
]
