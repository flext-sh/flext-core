"""Structured logging with context propagation and dependency injection.

This module provides FlextLogger, a structured logging system built on
FlextRuntime.structlog() with automatic context propagation, dependency injection support,
and integration with the FLEXT ecosystem infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
import time
import traceback
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Literal, Self, Union, cast, overload

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T


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
    providing all required logging methods. Logging calls return None by default to
    align with conventional logger semantics, and callers can request structured
    FlextResult[bool] responses via the ``return_result`` flag or the
    :meth:`FlextLogger.with_result` adapter:
    - debug(message, *args, return_result=False, **context)
    - info(message, *args, return_result=False, **context)
    - warning(message, *args, return_result=False, **context)
    - error(message, *args, return_result=False, **context)
    - critical(message, *args, return_result=False, **context)
    - exception(message, *, exception=None, exc_info=True, return_result=False, **kwargs)
    - bind(**context) -> FlextLogger (new bound logger with context)
    - trace(message, *args, return_result=False, **kwargs)

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
    - Optional FlextResult[bool] outputs for all operations (via return_result or with_result)
    - Integrates with FlextContext for distributed tracing
    - Provides observability hooks for application layer

    Context Management Architecture:
    ================================
    Three-tier scoped context system:
    1. **Application Context** - Persists for entire application lifetime
       - Use for: app_name, app_version, environment, deployment_id
       - Example: FlextLogger.bind_application_context(app_name="algar-oud-mig")

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
    All logging methods return FlextResult[bool]:
    - Success: FlextResult[bool].ok(True)
    - Failure: FlextResult[bool].fail(error_message)
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
    - FlextDispatcher: Message and event logging
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
    #
    # NOTE: Configuration state is tracked by FlextRuntime._structlog_configured ONLY
    # FlextLogger no longer maintains its own redundant flags

    # Scoped context tracking
    # Format: {scope_name: {context_key: context_value}}
    _scoped_contexts: ClassVar[dict[str, dict[str, object]]] = {}

    # Level-based context tracking
    # Format: {log_level: {context_key: context_value}}
    _level_contexts: ClassVar[dict[str, dict[str, object]]] = {}

    # NOTE: _configure_structlog_if_needed() wrapper method REMOVED
    # Applications must call FlextRuntime.configure_structlog() explicitly at startup
    # This eliminates wrapper indirection and makes configuration responsibility clear

    # =========================================================================
    # ADVANCED FEATURES - Global context management via contextvars
    # =========================================================================

    @classmethod
    @overload
    def _context_operation(
        cls, operation: Literal["get"], **kwargs: object
    ) -> dict[str, object]: ...

    @classmethod
    @overload
    def _context_operation(
        cls, operation: Literal["bind", "unbind", "clear"], **kwargs: object
    ) -> FlextResult[bool]: ...

    @classmethod
    def _context_operation(
        cls, operation: str, **kwargs: object
    ) -> Union[FlextResult[bool], dict[str, object]]:
        """Generic context operation handler using mapping for DRY."""

        def bind_op() -> FlextResult[bool]:
            FlextRuntime.structlog().contextvars.bind_contextvars(**kwargs)
            return FlextResult[bool].ok(True)

        def unbind_op() -> FlextResult[bool]:
            keys = kwargs.get("keys", [])
            if isinstance(keys, list):
                FlextRuntime.structlog().contextvars.unbind_contextvars(*keys)
            return FlextResult[bool].ok(True)

        def clear_op() -> FlextResult[bool]:
            FlextRuntime.structlog().contextvars.clear_contextvars()
            return FlextResult[bool].ok(True)

        def get_op() -> dict[str, object]:
            return dict(FlextRuntime.structlog().contextvars.get_contextvars() or {})

        operations = {
            "bind": bind_op,
            "unbind": unbind_op,
            "clear": clear_op,
            "get": get_op,
        }

        try:
            result = operations[operation]()
            if operation == "get":
                return cast("dict[str, object]", result)
            return cast("FlextResult[bool]", result)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            error_msg = f"Failed to {operation} global context: {e}"
            if operation == "get":
                return {}
            return FlextResult[bool].fail(error_msg)

    @classmethod
    def bind_global_context(cls, **context: object) -> FlextResult[bool]:
        """Bind context globally using FlextRuntime.structlog() contextvars."""
        return cls._context_operation("bind", **context)

    @classmethod
    def unbind_global_context(cls, *keys: str) -> FlextResult[bool]:
        """Unbind specific keys from global context.

        Args:
            *keys: Context keys to unbind

        Returns:
            FlextResult[bool]: Success with True if unbound, failure with error details

        """
        return cls._context_operation("unbind", keys=keys)

    @classmethod
    def clear_global_context(cls) -> FlextResult[bool]:
        """Clear all globally bound context.

        Returns:
            FlextResult[bool]: Success with True if cleared, failure with error details

        """
        return cls._context_operation("clear")

    @classmethod
    def get_global_context(cls) -> dict[str, object]:
        """Get current global context."""
        return cls._context_operation("get")

    # =========================================================================
    # SCOPED CONTEXT MANAGEMENT - Three-tier context system
    # =========================================================================

    # Scoped context mapping for DRY binding
    _SCOPE_BINDERS: ClassVar[dict[str, str]] = {
        "application": "application",
        "request": "request",
        "operation": "operation",
    }

    @classmethod
    def _bind_context(cls, _scope: str, **context: object) -> FlextResult[bool]:
        """Internal method to bind context to a specific scope.

        Args:
            _scope: Scope name (application, request, operation)
            **context: Context variables to bind

        Returns:
            FlextResult[bool]: Success with True if bound, failure with error details

        """
        try:
            if _scope not in cls._scoped_contexts:
                cls._scoped_contexts[_scope] = {}
            cls._scoped_contexts[_scope].update(context)
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Failed to bind {_scope} context: {e}")

    @classmethod
    def bind_application_context(cls, **context: object) -> FlextResult[bool]:
        """Bind application-level context (persists for entire app lifetime).

        Application context persists for the entire application lifetime and is
        only cleared at application exit. Use for app name, version, environment.

        Args:
            **context: Application-level context variables

        Returns:
            FlextResult[bool]: Success with True if bound, failure with error details

        Example:
            >>> FlextLogger.bind_application_context(
            ...     app_name="algar-oud-mig",
            ...     app_version="0.9.0",
            ...     environment="production",
            ... )
            >>> # All logs include app context until application exit

        """
        return cls._bind_context("application", **context)

    @classmethod
    def bind_request_context(cls, **context: object) -> FlextResult[bool]:
        """Bind request-level context (persists for single request/command).

        Request context persists for a single CLI command or API request.
        Cleared at command completion. Use for correlation_id, command, user_id.

        Args:
            **context: Request-level context variables

        Returns:
            FlextResult[bool]: Success with True if bound, failure with error details

        Example:
            >>> FlextLogger.bind_request_context(
            ...     correlation_id="flext-abc123",
            ...     command="migrate",
            ...     user_id="admin",
            ... )
            >>> # All logs for this request include request context

        """
        return cls._bind_context("request", **context)

    @classmethod
    def bind_operation_context(cls, **context: object) -> FlextResult[bool]:
        """Bind operation-level context using mapping."""
        return cls._bind_context(cls._SCOPE_BINDERS["operation"], **context)

    @classmethod
    def clear_scope(cls, scope: str) -> FlextResult[bool]:
        """Clear all context variables for a specific scope.

        Args:
            scope: Scope to clear ("application", "request", "operation")

        Returns:
            FlextResult[bool]: Success with True if cleared, failure with error details

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

            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Failed to clear scope {scope}: {e}")

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
            result = FlextResult[bool].ok(True)

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
    def bind_context_for_level(cls, level: str, **context: object) -> FlextResult[bool]:
        """Bind context that only appears at specific log level.

        Context variables are tracked and will be filtered by the
        LevelBasedContextFilter processor to only appear at the specified
        log level or higher.

        Args:
            level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            **context: Context variables to bind

        Returns:
            FlextResult[bool]: Success with True if bound, failure with error details

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
            level_upper = level.upper()
            if level_upper not in cls._level_contexts:
                cls._level_contexts[level_upper] = {}
            cls._level_contexts[level_upper].update(context)

            # Use dict comprehension for DRY prefixing
            prefixed_context = {
                f"_level_{level_upper.lower()}_{k}": v for k, v in context.items()
            }
            FlextRuntime.structlog().contextvars.bind_contextvars(**prefixed_context)
            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Failed to bind level context: {e}")

    @classmethod
    def unbind_context_for_level(cls, level: str, *keys: str) -> FlextResult[bool]:
        """Unbind specific level-filtered context variables.

        Args:
            level: Log level the context was bound to
            *keys: Context keys to unbind

        Returns:
            FlextResult[bool]: Success with True if unbound, failure with error details

        """
        try:
            level_upper = level.upper()
            if level_upper in cls._level_contexts:
                for key in keys:
                    cls._level_contexts[level_upper].pop(key, None)

            # List comprehension for DRY prefixed keys
            prefixed_keys = [f"_level_{level_upper.lower()}_{k}" for k in keys]
            if prefixed_keys:
                FlextRuntime.structlog().contextvars.unbind_contextvars(*prefixed_keys)
            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Failed to unbind level context: {e}")

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
        config: object | None = None,
        _level: str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize FlextLogger with context.

        NEW (RECOMMENDED - Config Pattern):
            logger = FlextLogger(
                "my.module",
                config=FlextModels.Config.LoggerConfig(
                    service_name="my-service",
                    correlation_id="abc-123"
                )
            )

        OLD (Backward Compatible):
            logger = FlextLogger(
                "my.module",
                _service_name="my-service",
                _correlation_id="abc-123"
            )

        Args:
            name: Logger name (typically __name__ or module path)
            config: LoggerConfig instance (Pydantic v2) - NEW PATTERN
            _level: Optional log level override (backward compat)
            _service_name: Optional service name override (backward compat)
            _service_version: Optional service version override (backward compat)
            _correlation_id: Optional correlation ID override (backward compat)
            _force_new: Force creation of new instance (backward compat)

        """
        super().__init__()

        # Extract config values (config takes priority over individual params)
        if config is not None:
            _level = getattr(config, "level", _level)
            _service_name = getattr(config, "service_name", _service_name)
            _service_version = getattr(config, "service_version", _service_version)
            _correlation_id = getattr(config, "correlation_id", _correlation_id)
            _force_new = getattr(config, "force_new", _force_new)

        # DO NOT configure structlog here - should be done at application startup
        # Application must call FlextRuntime.configure_structlog() explicitly before creating loggers

        # Store logger name as public attribute (immutable after initialization)
        self.name = name

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

        # Initialize optional state variables
        self._context: dict[str, object] = {}
        self._tracking: dict[str, object] = {}

    @classmethod
    def _create_bound_logger(
        cls,
        name: str,
        bound_logger: FlextTypes.Logging.BoundLoggerType,
    ) -> FlextLogger:
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
        # Set attributes during __new__ - public attributes
        instance.name = name
        instance.logger = bound_logger
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

    def with_result(self) -> FlextLoggerResultAdapter:
        """Create a logger adapter that preserves FlextResult outputs."""
        return FlextLoggerResultAdapter(self)

    # =============================================================================
    # LOGGING METHODS - DELEGATE TO FlextRuntime.structlog()
    # =============================================================================

    @overload
    def trace(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[True],
        **kwargs: object,
    ) -> FlextResult[bool]: ...

    @overload
    def trace(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[False] = False,
        **kwargs: object,
    ) -> None: ...

    def trace(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: bool = False,
        **kwargs: object,
    ) -> FlextResult[bool] | None:
        """Log trace message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"

            self.logger.debug(
                formatted_message,
                **kwargs,
            )  # FlextRuntime.structlog() doesn't have trace
            result = FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            result = FlextResult[bool].fail(f"Logging failed: {e}")
        return result if return_result else None

    def _format_log_message(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType
    ) -> str:
        """Format log message with arguments.

        Args:
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting

        Returns:
            str: Formatted message

        """
        try:
            return message % args if args else message
        except (TypeError, ValueError):
            return f"{message} | args={args!r}"

    def _get_calling_frame(self) -> types.FrameType | None:
        """Get the calling frame 3 levels up the stack.

        Returns:
            types.FrameType | None: The calling frame, or None if unavailable.

        """
        frame = inspect.currentframe()
        if not frame:
            return None
        # Use range for DRY frame skipping
        for _ in range(4):
            frame = frame.f_back
            if not frame:
                return None
        return frame

    def _extract_class_name(self, frame: types.FrameType) -> str | None:
        """Extract class name from frame locals or qualname.

        Args:
            frame: The frame to extract class name from

        Returns:
            str | None: The class name if found, None otherwise

        """  # Check 'self' in locals
        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if hasattr(self_obj, "__class__"):
                return cast("str", self_obj.__class__.__name__)

        # Qualname extraction for Python 3.11+
        if hasattr(frame.f_code, "co_qualname"):
            qualname = frame.f_code.co_qualname
            if "." in qualname:
                parts = qualname.rsplit(".", 1)
                if len(parts) == FlextConstants.Validation.LEVEL_PREFIX_PARTS_COUNT:
                    potential_class = parts[0]
                    if potential_class and potential_class[0].isupper():
                        return potential_class
        return None

    def _get_caller_source_path(self) -> str | None:
        """Get source file path of the caller with line, class and method context.

        Returns:
            str | None: Formatted string like "file.py:3334" or "file.py:3334 ClassName.method_name",
                       None if fails

        """
        try:
            caller_frame = self._get_calling_frame()
            if not caller_frame:
                return None

            filename = caller_frame.f_code.co_filename
            file_path = self._convert_to_relative_path(filename)
            line_number = caller_frame.f_lineno + 1

            method_name = caller_frame.f_code.co_name
            class_name = self._extract_class_name(caller_frame)

            # Build source parts using conditional mapping
            source_parts = [f"{file_path}:{line_number}"]
            if class_name and method_name:
                source_parts.append(f"{class_name}.{method_name}")
            elif method_name and method_name != "<module>":
                source_parts.append(method_name)

            return " ".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        except Exception:
            return None

    def _convert_to_relative_path(self, filename: str) -> str:
        """Convert absolute path to relative path from workspace root.

        Args:
            filename: Absolute file path

        Returns:
            str: Relative path from workspace root or filename only

        """
        try:
            abs_path = Path(filename).resolve()
            workspace_root = self._find_workspace_root(abs_path)

            if workspace_root:
                try:
                    return str(abs_path.relative_to(workspace_root))
                except ValueError:
                    return Path(filename).name
            return Path(filename).name
        except Exception:
            return Path(filename).name

    def _find_workspace_root(self, abs_path: Path) -> Path | None:
        """Find workspace root by looking for common markers.

        Args:
            abs_path: Absolute path to start search from

        Returns:
            Path | None: Workspace root path or None if not found

        """
        current = abs_path.parent
        markers = ["pyproject.toml", ".git", "poetry.lock"]

        for _ in range(10):  # Max depth
            if any((current / marker).exists() for marker in markers):
                return current
            if current == current.parent:
                break
            current = current.parent
        return None

    def _log(
        self,
        _level: str,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        **context: object,
    ) -> FlextResult[bool]:
        """Internal logging method - consolidates all log level methods.

        Args:
            _level: Log level (debug, info, warning, error, critical)
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting
            **context: Additional context for structured logging

        Returns:
            FlextResult[bool]: Success with True if logged, failure with error details

        """
        try:
            formatted_message = self._format_log_message(message, *args)

            # Auto-add source if not provided
            if "source" not in context and (
                source_path := self._get_caller_source_path()
            ):
                context["source"] = source_path

            # Dynamic method call using getattr mapping
            getattr(self.logger, _level.lower())(formatted_message, **context)
            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Logging failed: {e}")

    @overload
    def debug(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[True],
        **context: object,
    ) -> FlextResult[bool]: ...

    @overload
    def debug(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[False] = False,
        **context: object,
    ) -> None: ...

    def debug(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: bool = False,
        **context: object,
    ) -> FlextResult[bool] | None:
        """Log debug message - LoggerProtocol implementation."""
        result = self._log("debug", message, *args, **context)
        return result if return_result else None

    @overload
    def info(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[True],
        **context: object,
    ) -> FlextResult[bool]: ...

    @overload
    def info(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[False] = False,
        **context: object,
    ) -> None: ...

    def info(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: bool = False,
        **context: object,
    ) -> FlextResult[bool] | None:
        """Log info message - LoggerProtocol implementation."""
        result = self._log("info", message, *args, **context)
        return result if return_result else None

    @overload
    def warning(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[True],
        **context: object,
    ) -> FlextResult[bool]: ...

    @overload
    def warning(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[False] = False,
        **context: object,
    ) -> None: ...

    def warning(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: bool = False,
        **context: object,
    ) -> FlextResult[bool] | None:
        """Log warning message - LoggerProtocol implementation."""
        result = self._log("warning", message, *args, **context)
        return result if return_result else None

    @overload
    def error(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[True],
        **context: object,
    ) -> FlextResult[bool]: ...

    @overload
    def error(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[False] = False,
        **context: object,
    ) -> None: ...

    def error(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: bool = False,
        **context: object,
    ) -> FlextResult[bool] | None:
        """Log error message - LoggerProtocol implementation."""
        result = self._log("error", message, *args, **context)
        return result if return_result else None

    @overload
    def critical(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[True],
        **context: object,
    ) -> FlextResult[bool]: ...

    @overload
    def critical(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: Literal[False] = False,
        **context: object,
    ) -> None: ...

    def critical(
        self,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        return_result: bool = False,
        **context: object,
    ) -> FlextResult[bool] | None:
        """Log critical message - LoggerProtocol implementation."""
        result = self._log("critical", message, *args, **context)
        return result if return_result else None

    @overload
    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        return_result: Literal[True],
        **kwargs: object,
    ) -> FlextResult[bool]: ...

    @overload
    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        return_result: Literal[False] = False,
        **kwargs: object,
    ) -> None: ...

    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        return_result: bool = False,
        **kwargs: object,
    ) -> FlextResult[bool] | None:
        """Log exception message with stack trace - LoggerProtocol implementation.

        Stack traces are conditionally included based on FlextConfig.log_level:
        - DEBUG mode: Full formatted stack trace included
        - INFO mode and above: Only exception type and message

        Args:
            message: Error message
            exception: Optional exception object to format
            exc_info: Include current exception info if no exception provided
            **kwargs: Additional context

        """
        try:
            # Determine stack trace inclusion using effective_log_level
            try:
                config = FlextConfig.get_global_instance()
                include_stack_trace = (
                    config.effective_log_level.upper() == FlextConstants.Logging.DEBUG
                )
            except Exception:
                include_stack_trace = True

            # Add exception details using conditional mapping
            if exception is not None:
                kwargs.update({
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                })
                if include_stack_trace:
                    kwargs["stack_trace"] = "".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    )
            elif exc_info and include_stack_trace:
                kwargs["stack_trace"] = traceback.format_exc()

            self.logger.error(message, **kwargs)
            result = FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            result = FlextResult[bool].fail(f"Logging failed: {e}")
        return result if return_result else None

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
    ) -> FlextResult[bool]:
        """Log FlextResult with automatic success/failure handling.

        Args:
            result: FlextResult to log
            operation: Optional operation name
            level: Log level for success case (error used for failures)

        Returns:
            FlextResult[bool]: Success with True if logged, failure with error details

        Example:
            >>> result = validate_user(data)
            >>> logger.log_result(result, operation="user_validation")

        """
        try:
            context: dict[str, object] = {}
            if operation is not None:
                context["operation"] = operation

            if result.is_success:
                msg = f"{operation} succeeded" if operation else "Operation succeeded"
                getattr(self, level, self.info)(msg, return_result=False, **context)
            else:
                msg = (
                    f"{operation} failed: {result.error}"
                    if operation
                    else f"Operation failed: {result.error}"
                )
                context["error_code"] = result.error_code
                context["error_data"] = result.error_data
                self.error(msg, return_result=False, **context)

            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Failed to log result: {e}")

    # =========================================================================
    # Protocol Implementations: ContextBinder, PerformanceTracker
    # =========================================================================

    def bind_context(self, context: dict[str, object]) -> FlextResult[bool]:
        """Bind context to logger (ContextBinder protocol)."""
        try:
            self._context.update(context)
            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(str(e))

    def get_context(self) -> FlextResult[dict[str, object]]:
        """Get context (ContextBinder protocol)."""
        try:
            return FlextResult[dict[str, object]].ok(self._context)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[dict[str, object]].fail(str(e))

    def start_tracking(self, _operation: str) -> FlextResult[bool]:
        """Start tracking operation (PerformanceTracker protocol)."""
        try:
            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(str(e))

    def stop_tracking(self, _operation: str) -> FlextResult[float]:
        """Stop tracking operation (PerformanceTracker protocol)."""
        return FlextResult[float].ok(0.0)

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
            """Start tracking."""
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            """Log results using status mapping."""
            elapsed = time.time() - self._start_time

            # Status mapping for DRY logging
            status_map = {
                True: ("success", self.logger.info),
                False: ("failed", self.logger.error),
            }

            is_success = exc_type is None
            status, log_method = status_map[is_success]

            context = {
                "duration_seconds": elapsed,
                "operation": self._operation_name,
                "status": status,
            }

            if not is_success:
                context.update({
                    "exception_type": exc_type.__name__ if exc_type else "",
                    "exception_message": str(exc_val) if exc_val else "",
                })

            log_method(
                f"{self._operation_name} {status}", return_result=False, **context
            )


class FlextLoggerResultAdapter:
    """Adapter ensuring FlextLogger methods return FlextResult outputs."""

    __slots__ = ("_base_logger",)

    def __init__(self, base_logger: FlextLogger) -> None:
        """Initialize adapter with base logger.

        Args:
            base_logger: FlextLogger instance to wrap

        """
        self._base_logger = base_logger

    def __getattr__(self, item: str) -> object:
        """Delegate attribute access."""
        return getattr(self._base_logger, item)

    def with_result(self) -> FlextLoggerResultAdapter:
        """Idempotent result adapter."""
        return self

    def bind(self, **context: object) -> FlextLoggerResultAdapter:
        """Bind context preserving adapter."""
        return FlextLoggerResultAdapter(self._base_logger.bind(**context))

    # Method mapping for DRY result logging
    _LOG_METHODS: ClassVar[dict[str, str]] = {
        "trace": "trace",
        "debug": "debug",
        "info": "info",
        "warning": "warning",
        "error": "error",
        "critical": "critical",
    }

    def _log_with_result(
        self,
        method: str,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        **kwargs: object,
    ) -> FlextResult[bool]:
        """Generic logging with result using method mapping."""
        return cast(
            "FlextResult[bool]",
            getattr(self._base_logger, method)(
                message, *args, return_result=True, **kwargs
            ),
        )

    def trace(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log trace message returning FlextResult.

        Args:
            message: Log message
            *args: Message format args
            **kwargs: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._log_with_result("trace", message, *args, **kwargs)

    def debug(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log debug message returning FlextResult.

        Args:
            message: Log message
            *args: Message format args
            **context: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._log_with_result("debug", message, *args, **kwargs)

    def info(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log info message returning FlextResult.

        Args:
            message: Log message
            *args: Message format args
            **context: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._log_with_result("info", message, *args, **kwargs)

    def warning(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log warning message returning FlextResult.

        Args:
            message: Log message
            *args: Message format args
            **context: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._log_with_result("warning", message, *args, **kwargs)

    def error(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log error message returning FlextResult.

        Args:
            message: Log message
            *args: Message format args
            **context: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._log_with_result("error", message, *args, **kwargs)

    def critical(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log critical message returning FlextResult.

        Args:
            message: Log message
            *args: Message format args
            **context: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._log_with_result("critical", message, *args, **kwargs)

    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        **kwargs: object,
    ) -> FlextResult[bool]:
        """Log exception with traceback returning FlextResult.

        Args:
            message: Log message
            exception: Exception object to log
            exc_info: Include exception info
            **kwargs: Context fields

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        kwargs_for_error = {k: v for k, v in kwargs.items() if k != "return_result"}
        return self._base_logger.error(
            message,
            exception=exception,
            exc_info=exc_info,
            return_result=True,
            **kwargs_for_error,
        )

    def track_performance(self, operation_name: str) -> FlextLogger.PerformanceTracker:
        """Track operation performance returning context manager.

        Args:
            operation_name: Name of operation being tracked

        Returns:
            PerformanceTracker: Context manager for performance tracking

        """
        return self._base_logger.track_performance(operation_name)

    def log_result(
        self,
        result: FlextResult[T],
        *,
        operation: str | None = None,
        level: str = "info",
    ) -> FlextResult[bool]:
        """Log FlextResult with automatic success/failure handling.

        Args:
            result: FlextResult to log
            operation: Optional operation name for context
            level: Log level (default: info)

        Returns:
            FlextResult[bool]: Success with True if logged

        """
        return self._base_logger.log_result(result, operation=operation, level=level)

    def bind_context(self, context: dict[str, object]) -> FlextResult[bool]:
        """Bind context to logger (ContextBinder protocol).

        Args:
            context: Context dictionary to bind

        Returns:
            FlextResult[bool]: Success with True if bound

        """
        return self._base_logger.bind_context(context)

    def get_context(self) -> FlextResult[dict[str, object]]:
        """Get current logger context.

        Returns:
            FlextResult[dict[str, object]]: Current context fields

        """
        return self._base_logger.get_context()

    def start_tracking(self, _operation: str) -> FlextResult[bool]:
        """Start tracking an operation.

        Args:
            _operation: Operation name to track

        Returns:
            FlextResult[bool]: Success with True if tracking started

        """
        return self._base_logger.start_tracking(_operation)

    def stop_tracking(self, _operation: str) -> FlextResult[float]:
        """Stop tracking an operation.

        Args:
            _operation: Operation name to stop tracking

        Returns:
            FlextResult[float]: Success with elapsed time

        """
        return self._base_logger.stop_tracking(_operation)


__all__: list[str] = [
    "FlextLogger",
    "FlextLoggerResultAdapter",
]
