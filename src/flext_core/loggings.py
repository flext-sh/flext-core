"""Structured logging with context propagation and dependency injection.

This module wraps ``structlog`` so dispatcher pipelines, handlers, and services
share context-aware logging that cooperates with ``r`` outcomes and
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
import warnings
from collections.abc import (
    Generator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import ClassVar, Self, override

from pydantic import BaseModel
from structlog.typing import Context

from flext_core import FlextSettings, c, p, r, t, u


class FlextLogger(u, p.Logger):
    """Context-aware logger tuned for dispatcher-centric CQRS flows.

    FlextLogger layers structured logging on ``structlog`` with scoped contexts,
    dependency-injector factories, performance tracking helpers, and adapters for
    ``r`` so command/query handlers emit consistent telemetry without
    bespoke wrappers.
    """

    _scoped_contexts: ClassVar[
        MutableMapping[str, MutableMapping[str, t.Container]]
    ] = {}
    _level_contexts: ClassVar[
        MutableMapping[str, MutableMapping[str, t.Container]]
    ] = {}
    _structlog_instance: p.Logger | None = None
    type _LogArg = t.RuntimeData | Exception

    def __init__(
        self,
        name: str,
        *,
        config: p.Settings | None = None,
        _bound_logger: p.Logger | None = None,
        _level: c.LogLevel | str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize FlextLogger with name and optional context."""
        super().__init__()
        self.name = name
        if _bound_logger is not None:
            self._structlog_instance = _bound_logger
            return
        if config is not None:
            _level = getattr(config, "level", _level)
            _service_name = getattr(config, c.KEY_SERVICE_NAME, _service_name)
            _service_version = getattr(
                config,
                c.KEY_SERVICE_VERSION,
                _service_version,
            )
            _correlation_id = getattr(
                config,
                c.KEY_CORRELATION_ID,
                _correlation_id,
            )
            _force_new = getattr(config, "force_new", _force_new)
        context = {}
        if _service_name:
            context[c.KEY_SERVICE_NAME] = _service_name
        if _service_version:
            context[c.KEY_SERVICE_VERSION] = _service_version
        if _correlation_id:
            context[c.KEY_CORRELATION_ID] = _correlation_id
        base_logger = u.get_logger(name)
        self._structlog_instance = (
            base_logger.bind(**context) if context else base_logger
        )

    def __call__(self) -> FlextLogger:
        """Return self to support factory-style DI registration."""
        return self

    @property
    @override
    def _context(self) -> Context:
        """Context mapping for BindableLogger protocol compliance."""
        return {}

    @property
    @override
    def logger(self) -> p.Logger:
        """Wrapped structlog logger instance."""
        instance = self._structlog_instance
        if instance is None:
            instance = u.get_logger(getattr(self, "name", __name__))
            self._structlog_instance = instance
        return instance

    @classmethod
    def _get_global_context(cls) -> t.ConfigMap:
        """Get current global context (internal use only)."""
        try:
            context_vars = u.structlog().contextvars.get_contextvars()
            context_map: t.FlatContainerMapping = (
                {
                    str(k): cls._to_container_value(v)
                    for k, v in dict(context_vars).items()
                }
                if context_vars
                else {}
            )
            context_obj: Mapping[str, t.ValueOrModel] = dict(context_map.items())
            return t.ConfigMap(root=context_obj)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return t.ConfigMap(root={})

    @classmethod
    def bind_context(cls, scope: str, **context: t.RuntimeData) -> r[bool]:
        """Bind context variables to a specific scope.

        Business Rule: Binds context variables to a specific scope (APPLICATION, REQUEST,
        or OPERATION) using structlog contextvars. Updates internal scoped context tracking
        and binds to structlog for automatic inclusion in log messages. This unified method
        replaces separate bind_application_context, bind_request_context, and bind_operation_context
        methods for DRY code organization.

        Audit Implication: Context binding ensures audit trail completeness by attaching
        context variables to log messages. All context variables are bound through this
        method, ensuring consistent context propagation across FLEXT. Used throughout
        FLEXT for structured logging with context.

        This unified method replaces the separate bind_application_context,
        bind_request_context, and bind_operation_context methods.

        Args:
            scope: Scope name. Use c.SCOPE_* constants:
                   - SCOPE_APPLICATION: Persists for entire app lifetime
                   - SCOPE_REQUEST: Persists for single request/command
                   - SCOPE_OPERATION: Persists for single operation
            **context: Context variables to bind

        Returns:
            r[bool]: Success with True if context bound, failure with error message otherwise.

        Examples:
            >>> # Application-level context (app name, version, environment)
            >>> FlextLogger.bind_context(
            ...     c.SCOPE_APPLICATION,
            ...     app_name="flext-oud-mig",
            ...     app_version="0.9.0",
            ...     environment="production",
            ... )

            >>> # Request-level context (correlation_id, command, user_id)
            >>> FlextLogger.bind_context(
            ...     c.SCOPE_REQUEST,
            ...     correlation_id="flext-abc123",
            ...     command="migrate",
            ...     user_id="REDACTED_LDAP_BIND_PASSWORD",
            ... )

            >>> # Operation-level context
            >>> FlextLogger.bind_context(
            ...     c.SCOPE_OPERATION,
            ...     operation="sync_users",
            ... )

        """
        try:
            if scope not in cls._scoped_contexts:
                cls._scoped_contexts[scope] = {}
            current_context: t.FlatContainerMapping = {
                key: cls._to_container_value(value)
                for key, value in cls._scoped_contexts[scope].items()
            }
            incoming_context: t.FlatContainerMapping = {
                key: cls._to_container_value(value) for key, value in context.items()
            }
            current_context_obj: t.ContainerMapping = dict(
                current_context.items(),
            )
            incoming_context_obj: t.ContainerMapping = dict(
                incoming_context.items(),
            )
            merge_result = u.merge_mappings(
                incoming_context_obj,
                current_context_obj,
                strategy="deep",
            )
            merged_value = merge_result.unwrap_or(current_context_obj)
            merged_context: t.MutableFlatContainerMapping = {}
            for key, value in merged_value.items():
                merged_context[str(key)] = cls._to_container_value(value)
            cls._scoped_contexts[scope] = merged_context
            u.structlog().contextvars.bind_contextvars(**context)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to bind context for scope '{scope}': {exc}")

    @classmethod
    def bind_context_for_level(cls, level: str, **context: t.RuntimeData) -> r[bool]:
        """Bind context variables that only appear in logs at specified level or higher.

        Business Rule: Binds context variables with level prefix (`_level_{level}_`) for
        conditional inclusion based on log level. Uses u for centralized logging
        management. Context variables are filtered by u.level_based_context_filter()
        processor based on log level hierarchy. Normalizes log level to standard format
        (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        Audit Implication: Level-based context binding ensures audit trail completeness by
        including verbose context only at appropriate log levels. All level-based context
        is filtered automatically by structlog processors, ensuring efficient log processing.

        Uses u for centralized logging management. Context variables
        are prefixed with `_level_{level}_` so they can be filtered by log level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - case insensitive
            **context: Context variables to bind

        Returns:
            r[bool]: Success with True if bound, failure with error details

        Example:
            >>> FlextLogger.bind_context_for_level("DEBUG", config="debug_config")
            >>> FlextLogger.bind_context_for_level("ERROR", stack_trace="trace_info")
            >>> # DEBUG logs will include 'config', ERROR logs will include 'stack_trace'

        """
        try:
            level_lower = level.lower()
            level_normalized = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                c.WarningLevel.ERROR: c.WarningLevel.ERROR,
                "critical": "critical",
            }.get(level_lower, level_lower)
            if level_normalized not in cls._level_contexts:
                cls._level_contexts[level_normalized] = {}
            normalized_context = cls._to_container_context(context)
            prefixed_context = {
                f"_level_{level_normalized}_{key}": value
                for key, value in normalized_context.items()
            }
            cls._level_contexts[level_normalized].update(normalized_context)
            u.structlog().contextvars.bind_contextvars(**prefixed_context)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Failed to bind context for level {level}: {e}")

    @classmethod
    def bind_global_context(cls, **context: t.RuntimeData) -> r[bool]:
        """Bind context globally using u.structlog() contextvars.

        Business Rule: Binds context variables globally using structlog contextvars,
        ensuring all subsequent log messages include these context variables automatically.
        Uses u for centralized logging management. Global context persists
        across all loggers until explicitly cleared or unbound.

        Audit Implication: Global context binding ensures audit trail completeness by
        attaching context variables to all log messages. All context variables are bound
        through this method, ensuring consistent context propagation across FLEXT.
        Critical for maintaining correlation IDs, user IDs, and other audit-relevant
        context throughout application execution.

        Args:
            **context: Context variables to bind globally

        Returns:
            r[bool]: Success with True if context bound, failure with error message otherwise.

        Example:
            >>> FlextLogger.bind_global_context(
            ...     correlation_id="flext-abc123",
            ...     user_id="REDACTED_LDAP_BIND_PASSWORD",
            ...     environment="production",
            ... )

        """
        try:
            normalized_context = cls._to_container_context(context)
            u.structlog().contextvars.bind_contextvars(**normalized_context)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to bind global context: {exc}")

    @classmethod
    def clear_global_context(cls) -> r[bool]:
        """Clear all globally bound context.

        Business Rule: Clears all globally bound context variables, removing them from
        all subsequent log messages. Uses u for centralized logging management.
        This operation is irreversible - all context must be rebound if needed.

        Audit Implication: Clearing global context removes audit trail context from
        log messages. Use with caution in production environments. Typically used
        during application shutdown or context reset scenarios. All context variables
        are cleared through this method, ensuring consistent context management.

        Example:
            >>> FlextLogger.clear_global_context()
            >>> # All global context cleared

        """
        try:
            u.structlog().contextvars.clear_contextvars()
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to clear global context: {exc}")

    @classmethod
    def clear_scope(cls, scope: str) -> r[bool]:
        """Clear all context variables for a specific scope.

        Business Rule: Clears all context variables bound to a specific scope (APPLICATION,
        REQUEST, or OPERATION). Unbinds context from structlog and clears internal tracking.
        This operation is irreversible - all context must be rebound if needed.

        Audit Implication: Clearing scope context removes audit trail context from
        log messages for that scope. Use with caution in production environments.
        Typically used during scope transitions (e.g., request completion) or context
        reset scenarios. All context variables are cleared through this method, ensuring
        consistent context management across FLEXT.

        Args:
            scope: Scope to clear (use c.SCOPE_* constants)

        Returns:
            r[bool]: Success with True if scope cleared, failure with error message otherwise.

        Example:
            >>> FlextLogger.clear_scope(c.SCOPE_REQUEST)
            >>> # Clears all request-level context

        """
        try:
            if scope in cls._scoped_contexts:
                keys = list(cls._scoped_contexts[scope].keys())
                if keys:
                    u.structlog().contextvars.unbind_contextvars(*keys)
                cls._scoped_contexts[scope] = {}
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to clear scope '{scope}': {exc}")

    @classmethod
    def create_bound_logger(cls, name: str, bound_logger: p.Logger) -> Self:
        """Internal factory for creating logger with pre-bound structlog instance."""
        return cls(name, _bound_logger=bound_logger)

    @classmethod
    def create_module_logger(cls, name: str = "flext") -> FlextLogger:
        """Create a logger instance for a module.

        Business Rule: Creates a FlextLogger instance for a specific module, using
        the module name for logger identification. Logger inherits global and scoped
        context automatically. Uses u for centralized logging management.
        Logger name is immutable after creation, ensuring consistent logger identity.

        Audit Implication: Module logger creation ensures audit trail completeness by
        attaching module name to log messages. All loggers are created through this
        method, ensuring consistent logger configuration across FLEXT. Module name
        is critical for tracing log messages to their source in audit trails.

        Args:
            name: Module name (typically __name__). Defaults to "flext".

        Returns:
            FlextLogger: Logger instance for the module

        Example:
            >>> logger = FlextLogger.create_module_logger(__name__)
            >>> logger.info("Module initialized")

        Note:
            `get_logger()` calls are replaced by `create_module_logger()` with default name.
            structlog is automatically configured on first logger creation.

        """
        u.ensure_structlog_configured()
        return cls(name)

    @classmethod
    def for_container(
        cls,
        container: p.Container,
        level: str | None = None,
        **context: t.RuntimeData,
    ) -> FlextLogger:
        """Create logger configured for a specific container.

        Creates a logger instance bound to a container's configuration and context.
        The logger inherits the container's configuration for log level and other
        settings, and can have additional context bound.

        Args:
            container: Container instance to bind logger to.
            level: Optional log level override. If not provided, uses container's
                config log_level.
            **context: Additional context variables to bind.

        Returns:
            FlextLogger: Logger instance configured for the container.

        Example:
            >>> logger = FlextLogger.for_container(
            ...     container, level="DEBUG", container_id="worker_1"
            ... )

        """
        if level is None:
            config = (
                container.config
                if hasattr(container, c.FIELD_CONFIG)
                else FlextSettings.get_global()
            )
            level = getattr(config, "log_level", "INFO")
        logger = cls.create_module_logger(f"container_{id(container)}")
        if context:
            _ = logger.bind_global_context(**context)
        return logger

    @classmethod
    @contextmanager
    def scoped_context(cls, scope: str, **context: t.RuntimeData) -> Generator[None]:
        """Context manager for automatic scoped context cleanup.

        Business Rule: Context manager that binds context variables to a specific scope
        and automatically clears them when exiting the context. Uses bind_context for
        binding and clear_scope for cleanup. Ensures context is always cleaned up even
        if exceptions occur, maintaining consistent context state.

        Audit Implication: Scoped context ensures audit trail completeness by attaching
        context variables to log messages within the scope. Automatic cleanup ensures
        context doesn't leak between scopes, maintaining audit trail integrity. All
        context variables are bound and cleared through this method, ensuring consistent
        context management across FLEXT.

        Args:
            scope: Scope name (use c.SCOPE_* constants)
            **context: Context variables to bind

        Yields:
            None: Context manager yields control to caller

        Example:
            >>> with FlextLogger.scoped_context(
            ...     c.SCOPE_OPERATION, operation="sync_users"
            ... ):
            ...     # Context automatically bound and cleared
            ...     logger.info("Operation started")

        """
        _ = cls.bind_context(scope, **context)
        try:
            yield
        finally:
            _ = cls.clear_scope(scope)

    @classmethod
    def unbind_context_for_level(cls, level: str, *keys: str) -> r[bool]:
        """Unbind context variables that were bound for a specific log level.

        Business Rule: Unbinds context variables that were bound for a specific log level,
        removing them from logs at that level or higher. Uses u for centralized
        logging management. Only specified keys are removed; other level-based context
        remains intact. Normalizes log level to standard format (DEBUG, INFO, WARNING,
        ERROR, CRITICAL).

        Audit Implication: Unbinding level-based context removes specific audit trail
        context from log messages at that level. Use with caution in production environments.
        Typically used when specific context variables are no longer relevant or need to
        be updated. All context variables are unbound through this method, ensuring
        consistent context management across FLEXT.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - case insensitive
            *keys: Context keys to unbind

        Returns:
            r[bool]: Success with True if unbound, failure with error details

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
                c.WarningLevel.ERROR: c.WarningLevel.ERROR,
                "critical": "critical",
            }.get(level_lower, level_lower)
            prefixed_keys: MutableSequence[str] = []
            for key in keys:
                prefixed_key = f"_level_{level_normalized}_{key}"
                prefixed_keys.append(prefixed_key)
                if level_normalized in cls._level_contexts:
                    _ = cls._level_contexts[level_normalized].pop(key, None)
            if prefixed_keys:
                u.structlog().contextvars.unbind_contextvars(*prefixed_keys)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Failed to unbind context for level {level}: {e}")

    @classmethod
    def unbind_global_context(cls, *keys: str) -> r[bool]:
        """Unbind specific keys from global context.

        Business Rule: Unbinds specific context keys from global context, removing them
        from all subsequent log messages. Uses u for centralized logging management.
        Only specified keys are removed; other global context remains intact.

        Audit Implication: Unbinding global context removes specific audit trail context
        from log messages. Use with caution in production environments. Typically used
        when specific context variables are no longer relevant or need to be updated.
        All context variables are unbound through this method, ensuring consistent
        context management.

        Args:
            *keys: Context keys to unbind

        Example:
            >>> FlextLogger.unbind_global_context("user_id", "session_id")
            >>> # Only 'user_id' and 'session_id' removed from global context

        """
        try:
            unbind_keys: Sequence[str] = [str(key) for key in keys]
            u.structlog().contextvars.unbind_contextvars(*unbind_keys)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to unbind global context: {exc}")

    @classmethod
    @contextmanager
    def with_container_context(
        cls,
        container: p.Container,
        level: c.LogLevel | str | None = None,
        **context: t.RuntimeData,
    ) -> Generator[FlextLogger]:
        """Context manager for container-scoped logging.

        Creates a logger bound to container context for the duration of the context
        manager. Context is automatically cleaned up when exiting the context.

        Args:
            container: Container instance to bind logger to.
            **context: Additional context variables to bind.

        Yields:
            FlextLogger: Logger instance configured for the container.

        Example:
            >>> with FlextLogger.with_container_context(
            ...     container, worker_id="1"
            ... ) as logger:
            ...     logger.info("Processing task")

        """
        resolved_level: str | None
        match level:
            case c.LogLevel() as enum_level:
                resolved_level = enum_level.value
            case None:
                resolved_level = None
            case _:
                resolved_level = level
        logger = cls.for_container(container, level=resolved_level)
        if context:
            _ = logger.bind_global_context(**context)
        try:
            yield logger
        finally:
            pass

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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            FlextLogger._report_internal_logging_failure(
                "convert_to_relative_path",
                exc,
            )
            return Path(filename).name

    @staticmethod
    def _extract_class_name(frame: types.FrameType) -> str | None:
        """Extract class name from frame locals or qualname."""
        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if hasattr(self_obj, "__class__"):
                class_name: str = self_obj.__class__.__name__
                return class_name
        if hasattr(frame.f_code, "co_qualname"):
            qualname = frame.f_code.co_qualname
            if "." in qualname:
                parts = qualname.rsplit(".", 1)
                if len(parts) == c.LEVEL_PREFIX_PARTS_COUNT:
                    potential_class = parts[0]
                    if potential_class and potential_class[0].isupper():
                        return potential_class
        return None

    @staticmethod
    def _find_workspace_root(abs_path: Path) -> Path | None:
        """Find workspace root by looking for common markers."""
        current = abs_path.parent
        markers = ["pyproject.toml", ".git", "poetry.lock"]
        for _ in range(10):
            if any((current / marker).exists() for marker in markers):
                return current
            if current == current.parent:
                break
            current = current.parent
        return None

    @staticmethod
    def _format_log_message(message: str, *args: _LogArg) -> str:
        """Format log message with % arguments."""
        try:
            return message % args if args else message
        except (TypeError, ValueError):
            return f"{message} | args={args!r}"

    @staticmethod
    def _to_container_value(
        value: _LogArg | t.Container | t.ValueOrModel,
    ) -> t.Container:
        """Normalize value to Container (internal helper)."""
        if isinstance(value, Exception):
            return str(value)
        if value is None:
            return ""
        if u.is_scalar(value) or isinstance(value, Path):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        normalized = u.normalize_to_container(value)
        if u.is_scalar(normalized) or isinstance(normalized, Path):
            return normalized
        return normalized.model_dump_json()

    @staticmethod
    def _to_scalar_value(
        value: _LogArg | t.Container | t.ValueOrModel | None,
    ) -> t.Scalar:
        if value is None:
            return ""
        if isinstance(value, Exception):
            return str(value)
        if isinstance(value, (list, tuple, dict, Mapping)):
            return str(value)
        if u.is_scalar(value):
            return value
        return str(value)

    @staticmethod
    def _to_container_context(
        context: Mapping[str, _LogArg | t.Container | t.ValueOrModel],
    ) -> t.FlatContainerMapping:
        """Convert mapping to container context using normalization."""
        return {
            key: FlextLogger._to_container_value(value)
            for key, value in context.items()
        }

    @classmethod
    def _to_scalar_context(
        cls,
        context: Mapping[str, _LogArg | t.Container | t.ValueOrModel | None],
    ) -> t.ScalarMapping:
        return {key: cls._to_scalar_value(value) for key, value in context.items()}

    @staticmethod
    def _get_caller_source_path() -> str | None:
        """Get source file path with line, class and method context."""
        try:
            caller_frame = FlextLogger._get_calling_frame()
            if caller_frame is None:
                return None
            filename = caller_frame.f_code.co_filename
            file_path = FlextLogger._convert_to_relative_path(filename)
            line_number = caller_frame.f_lineno + 1
            method_name = caller_frame.f_code.co_name
            class_name = FlextLogger._extract_class_name(caller_frame)
            source_parts = [f"{file_path}:{line_number}"]
            if class_name and method_name:
                source_parts.append(f"{class_name}.{method_name}")
            elif method_name and method_name != "<module>":
                source_parts.append(method_name)
            return " ".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            FlextLogger._report_internal_logging_failure("get_caller_source_path", exc)
            return None

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
    def _report_internal_logging_failure(operation: str, exc: Exception) -> None:
        with suppress(AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            u.structlog().get_logger("flext_core.loggings").warning(
                "Internal logger operation failed",
                operation=operation,
                error=exc,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
            )

    @staticmethod
    def _should_include_stack_trace() -> bool:
        try:
            config = FlextSettings.get_global()
            return config.effective_log_level.upper() == c.LogLevel.DEBUG.value
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            FlextLogger._report_internal_logging_failure(
                "should_include_stack_trace",
                exc,
            )
            return True

    @override
    @staticmethod
    def get_logger(name: str | None = None) -> p.Logger:
        """Get structlog logger instance (alias for u.get_logger)."""
        return u.get_logger(name)

    @override
    def bind(self, **context: t.RuntimeData) -> Self:
        """Bind additional context, returning new logger (original unchanged)."""
        bound_logger = self.logger.bind(**self._to_container_context(context))
        return self.__class__.create_bound_logger(self.name, bound_logger)

    def build_exception_context(
        self,
        *,
        exception: Exception | None,
        exc_info: bool,
        context: Mapping[str, t.RuntimeData | Exception],
    ) -> t.ConfigMap:
        """Build normalized context payload for exception/error logging."""
        include_stack_trace = self._should_include_stack_trace()
        context_dict: t.ConfigMap = t.ConfigMap(root={})
        if exception is not None:
            exception_data: t.ConfigMap = t.ConfigMap(
                root={
                    "exception_type": exception.__class__.__name__,
                    "exception_message": str(exception),
                },
            )
            merged_root: MutableMapping[str, t.ValueOrModel] = dict(context_dict.root)
            merged_root.update(dict(exception_data.root))
            context_dict = t.ConfigMap(root=merged_root)
            if include_stack_trace:
                context_dict["stack_trace"] = "".join(
                    traceback.format_exception(
                        exception.__class__,
                        exception,
                        exception.__traceback__,
                    ),
                )
        elif exc_info and include_stack_trace:
            context_dict["stack_trace"] = traceback.format_exc()
        for key, value in context.items():
            if not isinstance(value, BaseException):
                context_dict[key] = u.normalize_to_container(value)
        return context_dict

    @override
    def critical(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log critical message - Logger.Log implementation.

        Business Rule: Logs a critical-level message with optional context. Uses _log
        method for actual logging. Uses u for centralized logging management.

        Audit Implication: Critical logging ensures audit trail completeness by recording
        critical messages about severe failures. Critical messages are always included
        in production logs and critical for audit trail reconstruction and emergency
        response. All critical messages go through this method, ensuring consistent
        log formatting and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.LogLevel.CRITICAL, msg, *args, **kw)

    @override
    def debug(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log debug message - Logger.Log implementation.

        Business Rule: Logs a debug-level message with optional context. Uses _log
        method for actual logging. Uses u for centralized logging management.

        Audit Implication: Debug logging ensures audit trail completeness by recording
        detailed diagnostic information. Debug messages are typically filtered in
        production but critical for troubleshooting and audit trail reconstruction.
        All debug messages go through this method, ensuring consistent log formatting
        and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.LogLevel.DEBUG, msg, *args, **kw)

    @override
    def error(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log error message - Logger.Log implementation.

        Business Rule: Logs an error-level message with optional context. Uses _log
        method for actual logging. Uses u for centralized logging management.

        Audit Implication: Error logging ensures audit trail completeness by recording
        error messages about failures. Error messages are always included in production
        logs and critical for audit trail reconstruction and failure analysis. All
        error messages go through this method, ensuring consistent log formatting and
        context inclusion across FLEXT.
        """
        return self._log_standard_level(c.LogLevel.ERROR, msg, *args, **kw)

    @override
    def exception(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log exception with conditional stack trace (DEBUG only)."""
        message = str(msg)
        filtered_args: tuple[t.Scalar, ...] = tuple(
            FlextLogger._to_scalar_value(arg)
            for arg in args
            if not isinstance(arg, BaseException)
        )
        try:
            resolved_exception: Exception | None = (
                args[0] if args and isinstance(args[0], Exception) else None
            )
            raw_exception = kw.get("exception")
            exc_info_value = kw.get("exc_info", True)
            context_input: MutableMapping[str, t.Scalar | Exception] = {}
            for key, value in kw.items():
                if key in {"exception", "exc_info"}:
                    continue
                if isinstance(value, Exception):
                    context_input[key] = value
                else:
                    context_input[key] = FlextLogger._to_scalar_value(value)
            context_dict = self.build_exception_context(
                exception=resolved_exception,
                exc_info=bool(exc_info_value),
                context=context_input,
            )
            if resolved_exception is None and isinstance(raw_exception, BaseException):
                context_dict["exception_type"] = raw_exception.__class__.__name__
                context_dict["exception_message"] = str(raw_exception)
            _ = self.logger.error(
                message,
                *filtered_args,
                **FlextLogger._to_scalar_context(context_dict.root),
            )
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            FlextLogger._report_internal_logging_failure("exception", exc)
            return r[bool].fail(f"Exception logging failed: {exc}")

    def log(
        self,
        level: str,
        message: str,
        *args: _LogArg,
        **context: t.RuntimeData,
    ) -> r[bool]:
        """Log message with specified level - Logger.Log implementation.

        Business Rule: Logs a message with specified level, converting level string
        to LogLevel enum if possible. Uses _log method for actual logging. Context
        mapping is merged with logger's bound context. Uses u for centralized
        logging management.

        Audit Implication: Logging ensures audit trail completeness by recording
        messages with context. All log messages go through this method or specific
        level methods (debug, info, warning, error, critical), ensuring consistent
        log formatting and context inclusion across FLEXT. Log level is critical for
        filtering and prioritizing audit trail messages.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            *args: Positional args for printf-style message formatting
            **context: Keyword context to include in structured log

        """
        level_enum: c.LogLevel | str = level
        with suppress(ValueError, AttributeError):
            level_enum = c.LogLevel(level.upper())
        converted_args: tuple[t.Container, ...] = tuple(
            FlextLogger._to_scalar_value(arg) for arg in args
        )
        return self._log(level_enum, message, *converted_args, **context)

    @override
    def new(self, **context: t.RuntimeData) -> Self:
        """Create new logger with context - implements BindableLogger protocol."""
        return self.bind(**context)

    def trace(
        self,
        message: str,
        *args: _LogArg,
        **kwargs: t.RuntimeData,
    ) -> r[bool]:
        """Log trace message - Logger.Log implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self.logger.debug(
                formatted_message,
                **FlextLogger._to_scalar_context(kwargs),
            )
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            FlextLogger._report_internal_logging_failure("trace", exc)
            return r[bool].fail(f"Trace logging failed: {exc}")

    @override
    def unbind(self, *keys: str, safe: bool = False) -> Self:
        """Unbind keys from logger - implements BindableLogger protocol."""
        if safe:
            with suppress(KeyError, ValueError, AttributeError):
                bound_logger = self.logger.unbind(*keys)
                return self.__class__.create_bound_logger(self.name, bound_logger)
            return self
        bound_logger = self.logger.unbind(*keys)
        return self.__class__.create_bound_logger(self.name, bound_logger)

    @override
    def try_unbind(self, *keys: str) -> Self:
        """Unbind keys in safe mode (deprecated compatibility helper)."""
        warnings.warn(
            "FlextLogger.try_unbind is deprecated; use unbind(*keys, safe=True). "
            "Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.unbind(*keys, safe=True)

    @override
    def info(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log info message - Logger.Log implementation."""
        return self._log_standard_level(c.LogLevel.INFO, msg, *args, **kw)

    @override
    def warning(
        self,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Log warning message - Logger.Log implementation."""
        return self._log_standard_level(c.LogLevel.WARNING, msg, *args, **kw)

    def _log(
        self,
        _level: c.LogLevel | str,
        event: str,
        *args: t.RuntimeData,
        **context: t.RuntimeData | Exception,
    ) -> r[bool]:
        """Internal logging method - consolidates all log level methods.

        Business Rule: Internal method that consolidates all log level methods (debug,
        info, warning, error, critical) into a single implementation. Delegates to
        structlog logger. Uses u for centralized logging management.
        Returns r[bool] indicating success or failure.
        """
        try:
            if "source" not in context and (
                source_path := FlextLogger._get_caller_source_path()
            ):
                context["source"] = source_path
            for idx, arg in enumerate(args):
                context[f"arg_{idx}"] = arg
            match _level:
                case c.LogLevel() as enum_level:
                    level_raw: str = enum_level.value
                case _:
                    level_raw = str(_level)
            level_str = level_raw.lower()
            scalar_context = FlextLogger._to_scalar_context(context)
            getattr(self.logger, level_str)(event, **scalar_context)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Logging failed: {e}")

    def _log_standard_level(
        self,
        level: c.LogLevel,
        msg: str,
        *args: t.RuntimeData,
        **kw: t.RuntimeData | Exception,
    ) -> r[bool]:
        return self._log(level, msg, *args, **kw)

    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: p.Logger, operation_name: str) -> None:
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
            context: t.ConfigMap = t.ConfigMap(
                root={
                    "duration_seconds": elapsed,
                    c.HandlerType.OPERATION: self._operation_name,
                    c.FIELD_STATUS: status,
                },
            )
            if not is_success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            if is_success:
                _ = self.logger.info(
                    f"{self._operation_name} {status}",
                    **FlextLogger._to_container_context(context.root),
                )
            else:
                _ = self.logger.error(
                    f"{self._operation_name} {status}",
                    **FlextLogger._to_container_context(context.root),
                )


__all__: Sequence[str] = ["FlextLogger"]
