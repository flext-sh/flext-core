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
from typing import ClassVar, Literal, Self, Union, overload

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T


class FlextLogger:
    """Structured logging with context propagation and FlextResult integration.

    Built on FlextRuntime.structlog() for thread-safe structured logging with:
    - Three-tier scoped contexts (application/request/operation)
    - Level-based context filtering (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    - DI factories: create_service_logger(), create_module_logger()
    - Performance tracking: track_performance() context manager
    - FlextResult integration: log_result(), with_result() adapter

    Logging methods (debug, info, warning, error, critical, exception) return
    None by default; use return_result=True or with_result() for FlextResult.

    Usage:
        logger = FlextLogger.create_module_logger(__name__)
        logger.info("Processing", user_id="123")

        with logger.track_performance("database_query"):
            db.execute()

        FlextLogger.bind_global_context(request_id="req-456")
        logger.log_result(result, operation="validation")
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
        try:
            return cls._execute_context_op(operation, kwargs)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return cls._handle_context_error(operation, e)

    @classmethod
    def _execute_context_op(
        cls, operation: str, kwargs: dict[str, object]
    ) -> Union[FlextResult[bool], dict[str, object]]:
        """Execute context operation by name."""
        if operation == "bind":
            FlextRuntime.structlog().contextvars.bind_contextvars(**kwargs)
            return FlextResult[bool].ok(True)
        if operation == "unbind":
            keys = kwargs.get("keys", [])
            if isinstance(keys, (list, tuple)):
                FlextRuntime.structlog().contextvars.unbind_contextvars(*keys)
            return FlextResult[bool].ok(True)
        if operation == "clear":
            FlextRuntime.structlog().contextvars.clear_contextvars()
            return FlextResult[bool].ok(True)
        if operation == "get":
            return dict(FlextRuntime.structlog().contextvars.get_contextvars() or {})
        return FlextResult[bool].fail(f"Unknown operation: {operation}")

    @classmethod
    def _handle_context_error(
        cls, operation: str, exc: Exception
    ) -> Union[FlextResult[bool], dict[str, object]]:
        """Handle context operation error."""
        if operation == "get":
            return {}
        return FlextResult[bool].fail(f"Failed to {operation} global context: {exc}")

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
            ...     app_name="client-a-oud-mig",
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
            ...     user_id="REDACTED_LDAP_BIND_PASSWORD",
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
        """Context manager for automatic scoped context cleanup."""
        # Use _bind_context for all scopes (handles known + generic scopes)
        result = cls._bind_context(scope, **context)

        if result.is_failure:
            cls.create_module_logger("flext_core.loggings").warning(
                f"Failed to bind scoped context: {result.error}"
            )

        try:
            yield
        finally:
            cls.clear_scope(scope)

    # =========================================================================
    # LEVEL-BASED CONTEXT MANAGEMENT - Log level filtering
    # =========================================================================

    @classmethod
    def bind_context_for_level(cls, level: str, **context: object) -> FlextResult[bool]:
        """Bind context that only appears at specific log level."""
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
        """Unbind specific level-filtered context variables."""
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
        """Create logger with service context (DI Factory pattern)."""
        return cls(
            service_name,
            _service_name=service_name,
            _service_version=version,
            _correlation_id=correlation_id,
        )

    @classmethod
    def create_module_logger(cls, module_name: str) -> FlextLogger:
        """Create logger for Python module (DI Factory pattern)."""
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
        """Initialize FlextLogger with name and optional context."""
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
        """Internal factory for creating logger with pre-bound structlog instance."""
        instance = cls.__new__(cls)
        instance.name = name
        instance.logger = bound_logger
        return instance

    def bind(self, **context: object) -> FlextLogger:
        """Bind additional context, returning new logger (original unchanged)."""
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
        """Format log message with % arguments."""
        try:
            return message % args if args else message
        except (TypeError, ValueError):
            return f"{message} | args={args!r}"

    def _get_calling_frame(self) -> types.FrameType | None:
        """Get the calling frame 4 levels up the stack."""
        frame = inspect.currentframe()
        if not frame:
            return None
        for _ in range(4):
            frame = frame.f_back
            if not frame:
                return None
        return frame

    def _extract_class_name(self, frame: types.FrameType) -> str | None:
        """Extract class name from frame locals or qualname."""
        # Check 'self' in locals
        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if hasattr(self_obj, "__class__"):
                class_name: str = self_obj.__class__.__name__
                return class_name

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
        """Get source file path with line, class and method context."""
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
        """Convert absolute path to relative path from workspace root."""
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
        """Find workspace root by looking for common markers."""
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
        """Internal logging method - consolidates all log level methods."""
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
        """Log exception with conditional stack trace (DEBUG only)."""
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
        """Track operation performance with automatic timing logs."""
        return FlextLogger.PerformanceTracker(self, operation_name)

    def log_result(
        self,
        result: FlextResult[T],
        *,
        operation: str | None = None,
        level: str = "info",
    ) -> FlextResult[bool]:
        """Log FlextResult with automatic success/failure handling."""
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
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: FlextLogger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
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
            """Log operation result with timing."""
            elapsed = time.time() - self._start_time
            is_success = exc_type is None
            status = "success" if is_success else "failed"
            log_method = self.logger.info if is_success else self.logger.error

            context: dict[str, object] = {
                "duration_seconds": elapsed,
                "operation": self._operation_name,
                "status": status,
            }

            if not is_success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""

            log_method(
                f"{self._operation_name} {status}", return_result=False, **context
            )


class FlextLoggerResultAdapter:
    """Adapter ensuring FlextLogger methods return FlextResult outputs.

    Uses __getattr__ for delegation - only overrides methods that need
    return_result=True behavior. Methods like track_performance, log_result,
    bind_context, get_context, start_tracking, stop_tracking are delegated
    automatically via __getattr__.
    """

    __slots__ = ("_base_logger",)

    def __init__(self, base_logger: FlextLogger) -> None:
        """Initialize adapter with base logger."""
        self._base_logger = base_logger

    def __getattr__(self, item: str) -> object:
        """Delegate attribute access to base logger."""
        return getattr(self._base_logger, item)

    def with_result(self) -> FlextLoggerResultAdapter:
        """Return self (idempotent)."""
        return self

    def bind(self, **context: object) -> FlextLoggerResultAdapter:
        """Bind context preserving adapter wrapper."""
        return FlextLoggerResultAdapter(self._base_logger.bind(**context))

    def _log_with_result(
        self,
        method: str,
        message: str,
        *args: FlextTypes.Logging.LoggingArgType,
        **kwargs: object,
    ) -> FlextResult[bool]:
        """Call logging method with return_result=True."""
        result = getattr(self._base_logger, method)(
            message, *args, return_result=True, **kwargs
        )
        return result if isinstance(result, FlextResult) else FlextResult[bool].ok(True)

    def trace(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log trace message returning FlextResult."""
        return self._log_with_result("trace", message, *args, **kwargs)

    def debug(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log debug message returning FlextResult."""
        return self._log_with_result("debug", message, *args, **kwargs)

    def info(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log info message returning FlextResult."""
        return self._log_with_result("info", message, *args, **kwargs)

    def warning(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log warning message returning FlextResult."""
        return self._log_with_result("warning", message, *args, **kwargs)

    def error(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log error message returning FlextResult."""
        return self._log_with_result("error", message, *args, **kwargs)

    def critical(
        self, message: str, *args: FlextTypes.Logging.LoggingArgType, **kwargs: object
    ) -> FlextResult[bool]:
        """Log critical message returning FlextResult."""
        return self._log_with_result("critical", message, *args, **kwargs)

    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        **kwargs: object,
    ) -> FlextResult[bool]:
        """Log exception with traceback returning FlextResult."""
        kwargs_for_error = {k: v for k, v in kwargs.items() if k != "return_result"}
        return self._base_logger.error(
            message,
            exception=exception,
            exc_info=exc_info,
            return_result=True,
            **kwargs_for_error,
        )


__all__: list[str] = [
    "FlextLogger",
    "FlextLoggerResultAdapter",
]
