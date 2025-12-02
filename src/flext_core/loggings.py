"""Structured logging with context propagation and dependency injection.

This module wraps ``structlog`` so dispatcher pipelines, handlers, and services
share context-aware logging that cooperates with ``FlextResult`` outcomes and
dependency-injector wiring. It keeps correlation data flowing alongside CQRS
operations without pulling higher-layer imports back into the foundation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
import time
import traceback
import types
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import ClassVar, Self, overload

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

# Use FlextTypes.GeneralValueType directly - no aliases


class FlextLogger:
    """Context-aware logger tuned for dispatcher-centric CQRS flows.

    FlextLogger layers structured logging on ``structlog`` with scoped contexts,
    dependency-injector factories, performance tracking helpers, and adapters for
    ``FlextResult`` so command/query handlers emit consistent telemetry without
    bespoke wrappers.
    """

    # =========================================================================
    # PRIVATE MEMBERS - FlextRuntime.structlog() configuration
    # =========================================================================
    #
    # NOTE: Configuration state is tracked by FlextRuntime._structlog_configured ONLY
    # FlextLogger no longer maintains its own redundant flags

    # Scoped context tracking
    # Format: {scope_name: {context_key: context_value}}
    _scoped_contexts: ClassVar[dict[str, dict[str, FlextTypes.GeneralValueType]]] = {}

    # Level-based context tracking
    # Format: {log_level: {context_key: context_value}}
    _level_contexts: ClassVar[dict[str, dict[str, FlextTypes.GeneralValueType]]] = {}

    # NOTE: _configure_structlog_if_needed() wrapper method REMOVED
    # Applications must call FlextRuntime.configure_structlog() explicitly at startup
    # This eliminates wrapper indirection and makes configuration responsibility clear

    # ═══════════════════════════════════════════════════════════════════
    # NESTED OPERATION GROUPS (Organization via Composition)
    # ═══════════════════════════════════════════════════════════════════

    class Context:
        """Context management: bind_context, bind_context_for_level, unbind_context_for_level."""

    class Factory:
        """Logger factory: create_service_logger, create_module_logger."""

    class Performance:
        """Performance tracking: start_tracking, stop_tracking, track_performance."""

    class Result:
        """Result logging: log_result, with_result."""

    # =========================================================================
    # ADVANCED FEATURES - Global context management via contextvars
    # =========================================================================

    @classmethod
    @overload
    def _context_operation(
        cls,
        operation: FlextConstants.Logging.ContextOperationGetLiteral,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextTypes.Types.ContextMetadataMapping: ...

    @classmethod
    @overload
    def _context_operation(
        cls,
        operation: FlextConstants.Logging.ContextOperationModifyLiteral,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @classmethod
    def _context_operation(
        cls,
        operation: str,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | FlextTypes.Types.ContextMetadataMapping:
        """Generic context operation handler using mapping for DRY."""
        try:
            return cls._execute_context_op(operation, kwargs)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return cls._handle_context_error(operation, e)

    @classmethod
    def _execute_context_op(
        cls,
        operation: str,
        kwargs: dict[str, FlextTypes.GeneralValueType],
    ) -> FlextProtocols.ResultProtocol[bool] | dict[str, FlextTypes.GeneralValueType]:
        """Execute context operation by name."""
        # Compare with StrEnum values directly - StrEnum comparison works with strings
        if operation == FlextConstants.Logging.ContextOperation.BIND:
            FlextRuntime.structlog().contextvars.bind_contextvars(**kwargs)
            return FlextRuntime.result_ok(True)
        if operation == FlextConstants.Logging.ContextOperation.UNBIND:
            keys = kwargs.get("keys", [])
            if isinstance(keys, Sequence):
                FlextRuntime.structlog().contextvars.unbind_contextvars(*keys)
            return FlextRuntime.result_ok(True)
        if operation == FlextConstants.Logging.ContextOperation.CLEAR:
            FlextRuntime.structlog().contextvars.clear_contextvars()
            return FlextRuntime.result_ok(True)
        if operation == FlextConstants.Logging.ContextOperation.GET:
            context_vars = FlextRuntime.structlog().contextvars.get_contextvars()
            return dict(context_vars) if context_vars else {}
        return FlextRuntime.result_fail(f"Unknown operation: {operation}")

    @classmethod
    def _handle_context_error(
        cls,
        operation: str,
        exc: Exception,
    ) -> FlextProtocols.ResultProtocol[bool] | FlextTypes.Types.ContextMetadataMapping:
        """Handle context operation error."""
        if operation == FlextConstants.Logging.ContextOperation.GET:
            return {}
        return FlextRuntime.result_fail(f"Failed to {operation} global context: {exc}")

    @classmethod
    def bind_global_context(
        cls,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Bind context globally using FlextRuntime.structlog() contextvars."""
        return cls._context_operation(
            FlextConstants.Logging.ContextOperation.BIND,
            **context,
        )

    @classmethod
    def clear_global_context(cls) -> FlextProtocols.ResultProtocol[bool]:
        """Clear all globally bound context.

        Returns:
            FlextProtocols.ResultProtocol[bool]: Success with True if cleared, failure with error details

        """
        return cls._context_operation(FlextConstants.Logging.ContextOperation.CLEAR)

    @classmethod
    def unbind_global_context(cls, *keys: str) -> FlextProtocols.ResultProtocol[bool]:
        """Unbind specific keys from global context.

        Args:
            *keys: Context keys to unbind

        Returns:
            FlextProtocols.ResultProtocol[bool]: Success with True if unbound, failure with error details

        """
        try:
            FlextRuntime.structlog().contextvars.unbind_contextvars(*keys)
            # Lazy import to avoid circular dependency

            return FlextRuntime.result_ok(True)
        except Exception as exc:
            return FlextRuntime.result_fail(f"Failed to unbind global context: {exc}")

    @classmethod
    def _get_global_context(cls) -> FlextTypes.Types.ContextMetadataMapping:
        """Get current global context (internal use only)."""
        return cls._context_operation(FlextConstants.Logging.ContextOperation.GET)

    # =========================================================================
    # SCOPED CONTEXT MANAGEMENT - Three-tier context system
    # =========================================================================

    # Scoped context mapping for DRY binding
    _SCOPE_BINDERS: ClassVar[dict[str, str]] = {
        FlextConstants.Context.SCOPE_APPLICATION: FlextConstants.Context.SCOPE_APPLICATION,
        FlextConstants.Context.SCOPE_REQUEST: FlextConstants.Context.SCOPE_REQUEST,
        FlextConstants.Context.SCOPE_OPERATION: FlextConstants.Context.SCOPE_OPERATION,
    }

    @classmethod
    def bind_context(
        cls,
        scope: str,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Bind context variables to a specific scope.

        This unified method replaces the separate bind_application_context,
        bind_request_context, and bind_operation_context methods.

        Args:
            scope: Scope name. Use FlextConstants.Context.SCOPE_* constants:
                   - SCOPE_APPLICATION: Persists for entire app lifetime
                   - SCOPE_REQUEST: Persists for single request/command
                   - SCOPE_OPERATION: Persists for single operation
            **context: Context variables to bind

        Returns:
            FlextResult[bool]: Success with True if bound, failure with error details

        Examples:
            >>> # Application-level context (app name, version, environment)
            >>> FlextLogger.bind_context(
            ...     FlextConstants.Context.SCOPE_APPLICATION,
            ...     app_name="algar-oud-mig",
            ...     app_version="0.9.0",
            ...     environment="production",
            ... )

            >>> # Request-level context (correlation_id, command, user_id)
            >>> FlextLogger.bind_context(
            ...     FlextConstants.Context.SCOPE_REQUEST,
            ...     correlation_id="flext-abc123",
            ...     command="migrate",
            ...     user_id="admin",
            ... )

            >>> # Operation-level context
            >>> FlextLogger.bind_context(
            ...     FlextConstants.Context.SCOPE_OPERATION,
            ...     operation="sync_users",
            ... )

        """
        try:
            if scope not in cls._scoped_contexts:
                cls._scoped_contexts[scope] = {}
            cls._scoped_contexts[scope].update(context)
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            # Lazy import to avoid circular dependency

            return FlextRuntime.result_ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextRuntime.result_fail(f"Failed to bind {scope} context: {e}")

    @classmethod
    def clear_scope(cls, scope: str) -> FlextProtocols.ResultProtocol[bool]:
        """Clear all context variables for a specific scope.

        Args:
            scope: Scope to clear (use FlextConstants.Context.SCOPE_* constants)

        Returns:
            FlextProtocols.ResultProtocol[bool]: Success with True if cleared, failure with error details

        Example:
            >>> FlextLogger.clear_scope(FlextConstants.Context.SCOPE_REQUEST)
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

            # Lazy import to avoid circular dependency

            return FlextRuntime.result_ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextRuntime.result_fail(f"Failed to clear scope {scope}: {e}")

    @classmethod
    @contextmanager
    def scoped_context(
        cls,
        scope: str,
        **context: FlextTypes.GeneralValueType,
    ) -> Iterator[None]:
        """Context manager for automatic scoped context cleanup."""
        # Use bind_context for all scopes (handles known + generic scopes)
        result = cls.bind_context(scope, **context)

        if result.is_failure:
            cls.create_module_logger("flext_core.loggings").warning(
                f"Failed to bind scoped context: {result.error}",
            )

        try:
            yield
        finally:
            _ = cls.clear_scope(scope)

    # =========================================================================
    # LEVEL-BASED CONTEXT MANAGEMENT - Log level filtering
    # =========================================================================

    @classmethod
    def bind_context_for_level(
        cls,
        level: str,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Bind context variables that only appear in logs at specified level or higher.

        Uses FlextRuntime for centralized logging management. Context variables
        are prefixed with `_level_{level}_` so they can be filtered by log level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - case insensitive
            **context: Context variables to bind

        Returns:
            FlextResult[bool]: Success with True if bound, failure with error details

        Example:
            >>> FlextLogger.bind_context_for_level("DEBUG", config="debug_config")
            >>> FlextLogger.bind_context_for_level("ERROR", stack_trace="trace_info")
            >>> # DEBUG logs will include 'config', ERROR logs will include 'stack_trace'

        """
        try:
            level_lower = level.lower()
            # Normalize level to standard format
            level_normalized = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "critical",
            }.get(level_lower, level_lower)

            # Track level contexts
            if level_normalized not in cls._level_contexts:
                cls._level_contexts[level_normalized] = {}

            # Bind context with level prefix for filtering
            prefixed_context: dict[str, FlextTypes.GeneralValueType] = {}
            for key, value in context.items():
                prefixed_key = f"_level_{level_normalized}_{key}"
                prefixed_context[prefixed_key] = value
                cls._level_contexts[level_normalized][key] = value

            # Use FlextRuntime for centralized logging management
            FlextRuntime.structlog().contextvars.bind_contextvars(**prefixed_context)

            return FlextRuntime.result_ok(True)
        except Exception as e:
            return FlextRuntime.result_fail(
                f"Failed to bind context for level {level}: {e}"
            )

    @classmethod
    def unbind_context_for_level(
        cls,
        level: str,
        *keys: str,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Unbind context variables that were bound for a specific log level.

        Uses FlextRuntime for centralized logging management.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - case insensitive
            *keys: Context keys to unbind

        Returns:
            FlextProtocols.ResultProtocol[bool]: Success with True if unbound, failure with error details

        Example:
            >>> FlextLogger.bind_context_for_level("DEBUG", config="debug_config")
            >>> FlextLogger.unbind_context_for_level("DEBUG", "config")
            >>> # 'config' will no longer appear in DEBUG logs

        """
        try:
            level_lower = level.lower()
            level_normalized = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "critical",
            }.get(level_lower, level_lower)

            # Build prefixed keys to unbind
            prefixed_keys: list[str] = []
            for key in keys:
                prefixed_key = f"_level_{level_normalized}_{key}"
                prefixed_keys.append(prefixed_key)
                # Remove from tracking
                if level_normalized in cls._level_contexts:
                    _ = cls._level_contexts[level_normalized].pop(key, None)

            # Use FlextRuntime for centralized logging management
            if prefixed_keys:
                FlextRuntime.structlog().contextvars.unbind_contextvars(*prefixed_keys)

            return FlextRuntime.result_ok(True)
        except Exception as e:
            return FlextRuntime.result_fail(
                f"Failed to unbind context for level {level}: {e}"
            )

    @classmethod
    def create_module_logger(cls, name: str = "flext") -> FlextLogger:
        """Create a logger instance for a module.

        Args:
            name: Module name (typically __name__). Defaults to "flext".

        Returns:
            FlextLogger: Logger instance for the module

        Example:
            >>> logger = FlextLogger.create_module_logger(__name__)
            >>> logger.info("Module initialized")

        Note:
            For backward compatibility, `get_logger()` calls are replaced by
            `create_module_logger()` with default name.

        """
        return cls(name)

    @classmethod
    def get_logger(cls, name: str | None = None) -> FlextProtocols.StructlogLogger:
        """Get structlog logger instance (alias for FlextRuntime.get_logger).

        This method provides compatibility with code that expects FlextLogger.get_logger()
        instead of FlextRuntime.get_logger().

        Args:
            name: Logger name (module name). Defaults to __name__ of caller.

        Returns:
            Logger: Typed structlog logger instance

        Example:
            >>> logger = FlextLogger.get_logger()
            >>> logger.debug("Debug message")

        """
        return FlextRuntime.get_logger(name)

    # =========================================================================
    # FACTORY PATTERNS - DI-ready logger creation
    # =========================================================================

    def __init__(
        self,
        name: str,
        *,
        config: FlextConfig | None = None,
        _level: FlextConstants.Literals.LogLevelLiteral | str | None = None,
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
        self.logger = FlextRuntime.get_logger(name).bind(**context)

        # Initialize optional state variables
        self._context: dict[str, FlextTypes.GeneralValueType] = {}
        self._tracking: dict[str, FlextTypes.GeneralValueType] = {}

    @classmethod
    def _create_bound_logger(
        cls,
        name: str,
        bound_logger: FlextProtocols.StructlogLogger,
    ) -> FlextLogger:
        """Internal factory for creating logger with pre-bound structlog instance."""
        instance = cls.__new__(cls)
        instance.name = name
        instance.logger = bound_logger
        return instance

    def bind(self, **context: FlextTypes.GeneralValueType) -> FlextLogger:
        """Bind additional context, returning new logger (original unchanged)."""
        return FlextLogger._create_bound_logger(self.name, self.logger.bind(**context))

    def with_result(self) -> FlextLogger.ResultAdapter:
        """Get a result-returning logger adapter.

        Returns a ResultAdapter that wraps all logging methods
        to return FlextProtocols.ResultProtocol[bool] indicating success/failure.

        Returns:
            ResultAdapter wrapping this logger

        """
        return FlextLogger.ResultAdapter(self)

    # =============================================================================
    # LOGGING METHODS - DELEGATE TO FlextRuntime.structlog()
    # =============================================================================

    @overload
    def trace(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def trace(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> None: ...

    def trace(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: bool = False,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
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
            result = FlextRuntime.result_ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            result = FlextRuntime.result_fail(f"Logging failed: {e}")
        return result if return_result else None

    @staticmethod
    def _format_log_message(
        message: str,
        *args: FlextTypes.GeneralValueType,
    ) -> str:
        """Format log message with % arguments."""
        try:
            return message % args if args else message
        except (TypeError, ValueError):
            return f"{message} | args={args!r}"

    @staticmethod
    def _get_calling_frame() -> types.FrameType | None:
        """Get the calling frame 4 levels up the stack."""
        frame = inspect.currentframe()
        if not frame:
            return None
        for _ in range(4):
            frame = frame.f_back
            if not frame:
                return None
        return frame

    @staticmethod
    def _extract_class_name(frame: types.FrameType) -> str | None:
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

    @staticmethod
    def _get_caller_source_path() -> str | None:
        """Get source file path with line, class and method context."""
        try:
            caller_frame = FlextLogger._get_calling_frame()
            if not caller_frame:
                return None

            filename = caller_frame.f_code.co_filename
            file_path = FlextLogger._convert_to_relative_path(filename)
            line_number = caller_frame.f_lineno + 1

            method_name = caller_frame.f_code.co_name
            class_name = FlextLogger._extract_class_name(caller_frame)

            # Build source parts using conditional mapping
            source_parts = [f"{file_path}:{line_number}"]
            if class_name and method_name:
                source_parts.append(f"{class_name}.{method_name}")
            elif method_name and method_name != "<module>":
                source_parts.append(method_name)

            return " ".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        except Exception:
            return None

    @staticmethod
    def _convert_to_relative_path(filename: str) -> str:
        """Convert absolute path to relative path from workspace root."""
        try:
            abs_path = Path(filename).resolve()
            workspace_root = FlextLogger._find_workspace_root(abs_path)

            if workspace_root:
                try:
                    return str(abs_path.relative_to(workspace_root))
                except ValueError:
                    return Path(filename).name
            return Path(filename).name
        except Exception:
            return Path(filename).name

    @staticmethod
    def _find_workspace_root(abs_path: Path) -> Path | None:
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
        _level: FlextConstants.Settings.LogLevel | str,
        message: str,
        *args: FlextTypes.GeneralValueType,
        **context: FlextTypes.GeneralValueType | Exception,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Internal logging method - consolidates all log level methods."""
        try:
            formatted_message = FlextLogger._format_log_message(message, *args)

            # Auto-add source if not provided
            if "source" not in context and (
                source_path := FlextLogger._get_caller_source_path()
            ):
                context["source"] = source_path

            # Use StrEnum directly - structlog accepts StrEnum values
            # Convert to lowercase string for method name lookup
            level_str = (
                _level.value
                if isinstance(_level, FlextConstants.Settings.LogLevel)
                else str(_level)
            ).lower()

            # Dynamic method call using getattr mapping
            getattr(self.logger, level_str)(formatted_message, **context)
            # Return success result directly
            return FlextRuntime.result_ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Return failure result directly
            return FlextRuntime.result_fail(f"Logging failed: {e}")

    def log(
        self,
        level: str,
        message: str,
        _context: Mapping[str, FlextTypes.FlexibleValue] | None = None,
    ) -> None:
        """Log message with specified level - LoggerProtocol implementation.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            _context: Optional context mapping

        """
        context_dict: dict[str, FlextTypes.GeneralValueType] = (
            dict(_context) if _context else {}
        )
        # Convert level string to LogLevel enum if possible
        level_enum: FlextConstants.Settings.LogLevel | str = level
        with suppress(ValueError, AttributeError):
            level_enum = FlextConstants.Settings.LogLevel(level.upper())

        # Use _log to handle the actual logging
        _ = self._log(level_enum, message, **context_dict)

    @overload
    def debug(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def debug(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **context: FlextTypes.GeneralValueType,
    ) -> None: ...

    def debug(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: bool = False,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
        """Log debug message - LoggerProtocol implementation."""
        result = self._log(
            FlextConstants.Settings.LogLevel.DEBUG,
            message,
            *args,
            **context,
        )
        return result if return_result else None

    @overload
    def info(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def info(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **context: FlextTypes.GeneralValueType,
    ) -> None: ...

    def info(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: bool = False,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
        """Log info message - LoggerProtocol implementation."""
        result = self._log(
            FlextConstants.Settings.LogLevel.INFO,
            message,
            *args,
            **context,
        )
        return result if return_result else None

    @overload
    def warning(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **context: FlextTypes.GeneralValueType | Exception,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def warning(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **context: FlextTypes.GeneralValueType | Exception,
    ) -> None: ...

    def warning(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: bool = False,
        **context: FlextTypes.GeneralValueType | Exception,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
        """Log warning message - LoggerProtocol implementation."""
        result = self._log(
            FlextConstants.Settings.LogLevel.WARNING,
            message,
            *args,
            **context,
        )
        return result if return_result else None

    @overload
    def error(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def error(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **context: FlextTypes.GeneralValueType,
    ) -> None: ...

    def error(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: bool = False,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
        """Log error message - LoggerProtocol implementation."""
        result = self._log(
            FlextConstants.Settings.LogLevel.ERROR,
            message,
            *args,
            **context,
        )
        return result if return_result else None

    @overload
    def critical(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def critical(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **context: FlextTypes.GeneralValueType,
    ) -> None: ...

    def critical(
        self,
        message: str,
        *args: FlextTypes.GeneralValueType,
        return_result: bool = False,
        **context: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
        """Log critical message - LoggerProtocol implementation."""
        result = self._log(
            FlextConstants.Settings.LogLevel.CRITICAL,
            message,
            *args,
            **context,
        )
        return result if return_result else None

    @overload
    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        return_result: FlextConstants.Logging.ReturnResultTrueLiteral,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]: ...

    @overload
    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        return_result: FlextConstants.Logging.ReturnResultFalseLiteral = False,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> None: ...

    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        return_result: bool = False,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool] | None:
        """Log exception with conditional stack trace (DEBUG only)."""
        try:
            # Determine stack trace inclusion using effective_log_level
            try:
                config = FlextConfig.get_global_instance()
                include_stack_trace = (
                    config.effective_log_level.upper()
                    == FlextConstants.Settings.LogLevel.DEBUG.value
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
                            type(exception),
                            exception,
                            exception.__traceback__,
                        ),
                    )
            elif exc_info and include_stack_trace:
                kwargs["stack_trace"] = traceback.format_exc()

            self.logger.error(message, **kwargs)
            result = FlextRuntime.result_ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            result = FlextRuntime.result_fail(f"Logging failed: {e}")
        return result if return_result else None

    # =========================================================================
    # ADVANCED FEATURES - Performance tracking and result integration
    # =========================================================================

    # =========================================================================
    # Protocol Implementations: PerformanceTracker
    # =========================================================================

    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: FlextLogger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
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
            """Log operation result with timing."""
            elapsed = time.time() - self._start_time
            is_success = exc_type is None
            status = "success" if is_success else "failed"
            log_method = self.logger.info if is_success else self.logger.error

            context: dict[str, FlextTypes.GeneralValueType] = {
                "duration_seconds": elapsed,
                "operation": self._operation_name,
                "status": status,
            }

            if not is_success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""

            log_method(
                f"{self._operation_name} {status}",
                return_result=False,
                **context,
            )

    class ResultAdapter:
        """Adapter ensuring FlextLogger methods return FlextResult outputs.

        Uses __getattr__ for delegation - only overrides methods that need
        return_result=True behavior. Methods like track_performance, log_result,
        bind_context, get_context, start_tracking, stop_tracking are delegated
        automatically via __getattr__.
        """

        __slots__ = ("_base_logger",)

        def __init__(self, base_logger: FlextLogger) -> None:
            """Initialize adapter with base logger."""
            super().__init__()
            self._base_logger = base_logger

        def __getattr__(self, item: str) -> object:
            """Delegate attribute access to base logger."""
            return getattr(self._base_logger, item)

        def with_result(self) -> FlextLogger.ResultAdapter:
            """Return self (idempotent)."""
            return self

        def bind(
            self, **context: FlextTypes.GeneralValueType
        ) -> FlextLogger.ResultAdapter:
            """Bind context preserving adapter wrapper."""
            return FlextLogger.ResultAdapter(self._base_logger.bind(**context))

        def _log_with_result(
            self,
            method: FlextConstants.Settings.LogLevel | str,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Call logging method with return_result=True."""
            # Convert StrEnum to string value if needed
            method_str = (
                method.value
                if isinstance(method, FlextConstants.Settings.LogLevel)
                else method
            ).lower()
            result = getattr(self._base_logger, method_str)(
                message,
                *args,
                return_result=True,
                **kwargs,
            )
            return (
                result
                if hasattr(result, "is_success")
                else FlextRuntime.result_ok(True)
            )

        def trace(
            self,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log trace message returning FlextResult."""
            return self._log_with_result("trace", message, *args, **kwargs)

        def debug(
            self,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log debug message returning FlextResult."""
            return self._log_with_result(
                FlextConstants.Settings.LogLevel.DEBUG,
                message,
                *args,
                **kwargs,
            )

        def info(
            self,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log info message returning FlextResult."""
            return self._log_with_result(
                FlextConstants.Settings.LogLevel.INFO,
                message,
                *args,
                **kwargs,
            )

        def warning(
            self,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log warning message returning FlextResult."""
            return self._log_with_result(
                FlextConstants.Settings.LogLevel.WARNING,
                message,
                *args,
                **kwargs,
            )

        def error(
            self,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log error message returning FlextResult."""
            return self._log_with_result(
                FlextConstants.Settings.LogLevel.ERROR,
                message,
                *args,
                **kwargs,
            )

        def critical(
            self,
            message: str,
            *args: FlextTypes.GeneralValueType,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log critical message returning FlextResult."""
            return self._log_with_result(
                FlextConstants.Settings.LogLevel.CRITICAL,
                message,
                *args,
                **kwargs,
            )

        def exception(
            self,
            message: str,
            *,
            exception: BaseException | None = None,
            exc_info: bool = True,
            **kwargs: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Log exception with traceback returning FlextResult."""
            # Convert exception to string for context if provided
            context: dict[str, FlextTypes.GeneralValueType] = kwargs
            if exception is not None:
                context["exception"] = str(exception)
                context["exception_type"] = type(exception).__name__
            if exc_info:
                context["exc_info"] = True

            # Filter out return_result if present and call error with return_result=True
            context_for_error = {
                k: v for k, v in context.items() if k != "return_result"
            }
            return self._base_logger.error(
                message,
                return_result=True,
                **context_for_error,
            )


__all__: list[str] = [
    "FlextLogger",
]
