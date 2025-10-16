"""Structured logging with context propagation and dependency injection.

This module provides FlextLogger, a structured logging system built on
FlextRuntime.structlog() with automatic context propagation, dependency injection support,
and integration with the FLEXT ecosystem infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
import traceback
import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import ClassVar, Self

from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T

# =============================================================================
# FLEXT LOGGER - THIN WRAPPER AROUND FlextRuntime.structlog()
# =============================================================================


class FlextLogger:
    """Structured logging with context propagation and dependency injection.

    Architecture: Layer 4 (Infrastructure)
    ======================================
    Provides production-ready structured logging built on FlextRuntime.structlog() with
    automatic context propagation, dependency injection support, and integration with
    the FLEXT ecosystem infrastructure.

    Structural Typing and Protocol Compliance:
    ===========================================
    FlextLogger implements FlextProtocols.LoggerProtocol through structural typing by
    providing all required logging methods:
    - debug(message, *args, **context) -> FlextResult[None]
    - info(message, *args, **context) -> FlextResult[None]
    - warning(message, *args, **context) -> FlextResult[None]
    - error(message, *args, **context) -> FlextResult[None]
    - critical(message, *args, **context) -> FlextResult[None]
    - exception(message, *, exception=None, exc_info=True, **kwargs) -> FlextResult[None]
    - bind(**context) -> FlextLogger (new bound logger with context)
    - trace(message, *args, **kwargs) -> FlextResult[None]

    Core Features:
    ==============
    - Structured logging with automatic context propagation
    - Context variable binding and unbinding (global scope)
    - Three-tier scoped context management (application/request/operation)
    - Level-based context filtering (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    - Dependency injection integration via service/module logger factories
    - Performance tracking with automatic timing and duration logging
    - FlextResult integration for automatic success/failure handling
    - Service-specific logger factories for DI pattern
    - Module-specific logger creation with __name__ support
    - Global context management with thread-safe access
    - Context manager support for scoped operations
    - Exception tracking with stack trace capture
    - Lazy context binding with PerformanceTracker

    Architecture Layers:
    ====================
    - Uses FlextRuntime.structlog() - Bridge layer for external logging library
    - Returns FlextResult[None] for all operations - Railway pattern
    - Integrates with FlextContext for distributed tracing
    - Provides observability hooks for application layer

    Context Management Architecture:
    ================================
    Three-tier scoped context system:
    1. **Application Context** - Persists for entire application lifetime
       - Use for: app_name, app_version, environment, deployment_id
       - Example: FlextLogger.bind_application_context(app_name="client-a-oud-mig")

    2. **Request Context** - Persists for single request/command
       - Use for: correlation_id, command, user_id, tenant_id
       - Example: FlextLogger.bind_request_context(correlation_id="abc123")

    3. **Operation Context** - Persists for single service operation
       - Use for: operation, service_name, method, operation_duration
       - Example: FlextLogger.bind_operation_context(operation="migrate")

    Context managers for automatic cleanup:
    >>> with FlextLogger.scoped_context("request", correlation_id="abc123"):
    ...     # All logs include correlation_id
    ...     do_work()
    >>> # Context automatically cleared after block

    Level-Based Context Filtering:
    ==============================
    Bind context that only appears at specific log levels:
    - DEBUG-only context: FlextLogger.bind_context_for_level("DEBUG", config=config_dict)
    - ERROR-only context: FlextLogger.bind_context_for_level("ERROR", stack_trace=trace)
    - Prevents context noise in production logs

    Factory Patterns (Dependency Injection):
    ========================================
    Service Logger Factory:
    >>> logger = FlextLogger.create_service_logger(
    ...     "user-service", version="1.0.0", correlation_id="abc123"
    ... )

    Module Logger Factory (recommended):
    >>> logger = FlextLogger.create_module_logger(__name__)

    Performance Tracking:
    ====================
    Automatic timing with context managers:
    >>> with logger.track_performance("database_query"):
    ...     db.execute_query()
    # Automatically logs: "database_query completed in 0.123s"

    Result Integration:
    ==================
    Automatic success/failure logging:
    >>> result = validate_user(data)
    >>> logger.log_result(result, operation="user_validation")
    # Logs with error_code and error_data if failed

    FlextResult Integration (Railway Pattern):
    ===========================================
    All logging methods return FlextResult[None]:
    - Success: FlextResult[None].ok(None)
    - Failure: FlextResult[None].fail(error_message)
    - Enables functional composition of logging operations

    Global Context Management:
    ==========================
    Thread-safe context binding at application scope:
    >>> FlextLogger.bind_global_context(request_id="req-123", user_id="usr-456")
    >>> logger.info("Processing")  # Includes bound context
    >>> FlextLogger.unbind_global_context("request_id")  # Selective unbinding
    >>> FlextLogger.clear_global_context()  # Clear all global context

    Runtime Configuration:
    ======================
    No FlextConfig dependency - self-configuring:
    >>> FlextLogger._configure_structlog_if_needed(log_level=logging.DEBUG)
    Uses FlextRuntime.structlog() for consistent logging across ecosystem

    Integration with FLEXT Ecosystem:
    =================================
    - FlextContext: Automatic correlation ID and context propagation
    - FlextResult: Structured error and result logging
    - FlextService: Service-specific logger creation
    - FlextHandler: Handler operation logging
    - FlextBus: Message and event logging
    - All services can use FlextLogger for consistent observability

    Thread Safety:
    ==============
    - Thread-safe context variable management via FlextRuntime.structlog().contextvars
    - Global context safely shared across threads/async tasks
    - Scoped context per request/operation thread
    - Connection per logger instance ensures isolation

    Performance Characteristics:
    ===========================
    - O(1) logger creation
    - O(1) context binding/unbinding
    - O(1) logging operations
    - Minimal overhead via FlextRuntime.structlog() integration
    - Lazy context binding for deferred evaluation

    Advanced Patterns:
    ==================
    - Chain context binding: logger.bind(a=1).bind(b=2).info("msg")
    - Exception context: logger.exception("msg", exception=exc, exc_info=True)
    - Performance tracking: with logger.track_performance("op"): do_work()
    - Level-specific debugging: FlextLogger.bind_context_for_level("DEBUG", config=cfg)
    - Scoped context managers: with FlextLogger.scoped_context("operation"): ...

    Usage Patterns:
    ===============
        >>> from flext_core import FlextLogger
        >>>
        >>> # Create module logger (recommended)
        >>> logger = FlextLogger.create_module_logger(__name__)
        >>>
        >>> # Log with structured context
        >>> logger.info("User logged in", user_id="123", action="login")
        >>>
        >>> # Bind context globally for all messages
        >>> FlextLogger.bind_global_context(request_id="req-456")
        >>> logger.info("Processing request")  # Includes request_id automatically
        >>>
        >>> # Track operation performance
        >>> with logger.track_performance("database_query"):
        ...     db.execute()
        >>>
        >>> # Log FlextResult with automatic success/failure handling
        >>> result = validate_user(data)
        >>> logger.log_result(result, operation="user_validation")
    """

    # =========================================================================
    # PRIVATE MEMBERS - FlextRuntime.structlog() configuration
    # =========================================================================

    _configured: bool = False
    _structlog_configured: bool = False

    # Scoped context tracking
    # Format: {scope_name: {context_key: context_value}}
    _scoped_contexts: ClassVar[dict[str, dict[str, object]]] = {}

    # Level-based context tracking
    # Format: {log_level: {context_key: context_value}}
    _level_contexts: ClassVar[dict[str, dict[str, object]]] = {}

    @staticmethod
    def _configure_structlog_if_needed(
        log_level: int | None = None,
        *,
        console_enabled: bool = True,
        additional_processors: Sequence[Callable[..., object]] | None = None,
    ) -> None:
        """Configure FlextRuntime.structlog() with advanced processor chain.

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
        """Bind context globally using FlextRuntime.structlog() contextvars.

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
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
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
            FlextRuntime.structlog().contextvars.unbind_contextvars(*keys)
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
            FlextRuntime.structlog().contextvars.clear_contextvars()
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to clear global context: {e}")

    @classmethod
    def get_global_context(cls) -> FlextTypes.Dict:
        """Get current global context."""
        return dict[str, object](FlextRuntime.structlog().contextvars.get_contextvars())

    # =========================================================================
    # SCOPED CONTEXT MANAGEMENT - Three-tier context system
    # =========================================================================

    @classmethod
    def bind_application_context(cls, **context: object) -> FlextResult[None]:
        """Bind application-level context (persists for entire app lifetime).

        Application context persists for the entire application lifetime and is
        only cleared at application exit. Use for app name, version, environment.

        Args:
            **context: Application-level context variables

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> FlextLogger.bind_application_context(
            ...     app_name="client-a-oud-mig",
            ...     app_version="0.9.0",
            ...     environment="production",
            ... )
            >>> # All logs include app context until application exit

        """
        try:
            # Track in application scope
            if "application" not in cls._scoped_contexts:
                cls._scoped_contexts["application"] = {}
            cls._scoped_contexts["application"].update(context)

            # Bind globally
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to bind application context: {e}")

    @classmethod
    def bind_request_context(cls, **context: object) -> FlextResult[None]:
        """Bind request-level context (persists for single request/command).

        Request context persists for a single CLI command or API request.
        Cleared at command completion. Use for correlation_id, command, user_id.

        Args:
            **context: Request-level context variables

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> FlextLogger.bind_request_context(
            ...     correlation_id="flext-abc123", command="migrate", user_id="REDACTED_LDAP_BIND_PASSWORD"
            ... )
            >>> # All logs for this request include request context

        """
        try:
            # Track in request scope
            if "request" not in cls._scoped_contexts:
                cls._scoped_contexts["request"] = {}
            cls._scoped_contexts["request"].update(context)

            # Bind globally
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to bind request context: {e}")

    @classmethod
    def bind_operation_context(cls, **context: object) -> FlextResult[None]:
        """Bind operation-level context (persists for single service operation).

        Operation context persists for a single service operation.
        Cleared at operation completion. Use for operation, service_name, method.

        Args:
            **context: Operation-level context variables

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> FlextLogger.bind_operation_context(
            ...     operation="migrate",
            ...     service="client-aOudMigrationService",
            ...     method="execute",
            ... )
            >>> # All logs for this operation include operation context

        """
        try:
            # Track in operation scope
            if "operation" not in cls._scoped_contexts:
                cls._scoped_contexts["operation"] = {}
            cls._scoped_contexts["operation"].update(context)

            # Bind globally
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to bind operation context: {e}")

    @classmethod
    def clear_scope(cls, scope: str) -> FlextResult[None]:
        """Clear all context variables for a specific scope.

        Args:
            scope: Scope to clear ("application", "request", "operation")

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> FlextLogger.clear_scope("request")
            >>> # Clears all request-level context

        """
        try:
            if scope in cls._scoped_contexts:
                # Get keys to unbind
                keys = list(cls._scoped_contexts[scope].keys())

                # Unbind from structlog
                if keys:
                    FlextRuntime.structlog().contextvars.unbind_contextvars(*keys)

                # Clear from tracking
                cls._scoped_contexts[scope] = {}

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to clear scope {scope}: {e}")

    @classmethod
    @contextmanager
    def scoped_context(cls, scope: str, **context: object) -> Iterator[None]:
        """Context manager for automatic scoped context cleanup.

        Automatically binds context for the operation duration and clears it
        after completion. Prevents context accumulation.

        Args:
            scope: Scope identifier ("application", "request", "operation")
            **context: Context variables to bind

        Yields:
            None

        Example:
            >>> with FlextLogger.scoped_context(
            ...     "request", correlation_id="abc123", command="migrate"
            ... ):
            ...     # All logs include correlation_id and command
            ...     do_work()
            >>> # Context automatically cleared after block

        """
        # Bind context based on scope
        if scope == "application":
            result = cls.bind_application_context(**context)
        elif scope == "request":
            result = cls.bind_request_context(**context)
        elif scope == "operation":
            result = cls.bind_operation_context(**context)
        else:
            # Generic scoped binding
            if scope not in cls._scoped_contexts:
                cls._scoped_contexts[scope] = {}
            cls._scoped_contexts[scope].update(context)
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            result = FlextResult[None].ok(None)

        if result.is_failure:
            # If binding failed, still yield but log warning
            logger = cls.create_module_logger("flext_core.loggings")
            logger.warning(f"Failed to bind scoped context: {result.error}")

        try:
            yield
        finally:
            # Clear scope on exit
            cls.clear_scope(scope)

    # =========================================================================
    # LEVEL-BASED CONTEXT MANAGEMENT - Log level filtering
    # =========================================================================

    @classmethod
    def bind_context_for_level(cls, level: str, **context: object) -> FlextResult[None]:
        """Bind context that only appears at specific log level.

        Context variables are tracked and will be filtered by the
        LevelBasedContextFilter processor to only appear at the specified
        log level or higher.

        Args:
            level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            **context: Context variables to bind

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> # Config only appears in DEBUG logs
            >>> FlextLogger.bind_context_for_level("DEBUG", config=config_dict)
            >>>
            >>> # Stack trace only appears in ERROR/CRITICAL logs
            >>> FlextLogger.bind_context_for_level("ERROR", stack_trace=trace_str)

        Note:
            Requires LevelBasedContextFilter processor in structlog chain.

        """
        try:
            # Normalize level to uppercase
            level_upper = level.upper()

            # Track in level-specific context
            if level_upper not in cls._level_contexts:
                cls._level_contexts[level_upper] = {}
            cls._level_contexts[level_upper].update(context)

            # Bind globally with level prefix
            # The processor will filter based on this prefix
            prefixed_context = {
                f"_level_{level_upper.lower()}_{k}": v for k, v in context.items()
            }
            FlextRuntime.structlog().contextvars.bind_contextvars(**prefixed_context)

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to bind level context: {e}")

    @classmethod
    def unbind_context_for_level(cls, level: str, *keys: str) -> FlextResult[None]:
        """Unbind specific level-filtered context variables.

        Args:
            level: Log level the context was bound to
            *keys: Context keys to unbind

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            level_upper = level.upper()

            # Remove from tracking
            if level_upper in cls._level_contexts:
                for key in keys:
                    cls._level_contexts[level_upper].pop(key, None)

            # Unbind prefixed keys
            prefixed_keys = [f"_level_{level_upper.lower()}_{k}" for k in keys]
            if prefixed_keys:
                FlextRuntime.structlog().contextvars.unbind_contextvars(*prefixed_keys)

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to unbind level context: {e}")

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

        # Configure FlextRuntime.structlog() if not already configured (NO config dependency)
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
        self.logger = FlextRuntime.structlog().get_logger(name).bind(**context)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    @classmethod
    def _create_bound_logger(cls, name: str, bound_logger: object) -> FlextLogger:
        """Internal factory for creating logger with pre-bound FlextRuntime.structlog() instance.

        This factory method allows creating FlextLogger instances with an already
        configured FlextRuntime.structlog() BoundLogger, avoiding the need to access private
        attributes directly.

        Args:
            name: Logger name
            bound_logger: Pre-configured bound FlextRuntime.structlog() logger (object type used
                         as FlextRuntime.structlog().BoundLogger is not publicly exposed)

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
        underlying FlextRuntime.structlog() logger. The original logger remains unchanged.

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
    # LOGGING METHODS - DELEGATE TO FlextRuntime.structlog()
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
            )  # FlextRuntime.structlog() doesn't have trace
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
        result: FlextResult[T],
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
            context: FlextTypes.Dict = {}
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
        """Context manager for performance tracking with automatic logging utilities."""

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
