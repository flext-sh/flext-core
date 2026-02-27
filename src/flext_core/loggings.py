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
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import ClassVar, Literal, Self, cast, overload, override

from structlog.typing import Context

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import c
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import FlextResult, r
from flext_core.runtime import FlextRuntime
from flext_core.settings import FlextSettings
from flext_core.typings import t
from flext_core.utilities import u


class FlextLogger(FlextRuntime, p.Log.StructlogLogger):
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
    _scoped_contexts: ClassVar[
        MutableMapping[str, MutableMapping[str, t.MetadataAttributeValue]]
    ] = {}

    # Level-based context tracking
    # Format: {log_level: {context_key: context_value}}
    _level_contexts: ClassVar[
        MutableMapping[str, MutableMapping[str, t.MetadataAttributeValue]]
    ] = {}
    logger: p.Log.StructlogLogger

    # Protocol compliance: BindableLogger._context property
    @property
    def _context(self) -> Context:
        """Context mapping for BindableLogger protocol compliance."""
        return {}

    def __call__(self) -> FlextLogger:
        """Return self to support factory-style DI registration."""
        return self

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
        operation: Literal["bind", "unbind", "clear"],
        **kwargs: t.MetadataAttributeValue,
    ) -> r[bool]: ...

    @classmethod
    @overload
    def _context_operation(
        cls,
        operation: Literal["get"],
        **kwargs: t.MetadataAttributeValue,
    ) -> m.ConfigMap: ...

    @classmethod
    def _context_operation(
        cls,
        operation: Literal["get", "bind", "unbind", "clear"],
        **kwargs: t.MetadataAttributeValue,
    ) -> r[bool] | m.ConfigMap:
        """Generic context operation handler using mapping for DRY."""
        try:
            return cls._execute_context_op(operation, kwargs)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return cls._handle_context_error(operation, e)

    @classmethod
    def _execute_context_op(
        cls,
        operation: str,
        kwargs: Mapping[str, t.MetadataAttributeValue],
    ) -> r[bool] | m.ConfigMap:
        """Execute context operation by name."""
        # Compare with StrEnum values directly - StrEnum comparison works with strings
        if operation == c.Logging.ContextOperation.BIND:
            FlextRuntime.structlog().contextvars.bind_contextvars(**kwargs)
            return r[bool].ok(value=True)
        if operation == c.Logging.ContextOperation.UNBIND:
            keys_value = kwargs.get("keys")
            if isinstance(keys_value, str):
                unbind_keys = [keys_value]
            elif isinstance(keys_value, list | tuple):
                unbind_keys = [str(key) for key in keys_value]
            else:
                return r[bool].ok(value=True)

            if unbind_keys:
                FlextRuntime.structlog().contextvars.unbind_contextvars(*unbind_keys)
            return r[bool].ok(value=True)
        if operation == c.Logging.ContextOperation.CLEAR:
            FlextRuntime.structlog().contextvars.clear_contextvars()
            return r[bool].ok(value=True)
        if operation == c.Logging.ContextOperation.GET:
            context_vars = FlextRuntime.structlog().contextvars.get_contextvars()
            context_map = (
                {
                    str(k): FlextRuntime.normalize_to_general_value(v)
                    for k, v in dict(context_vars).items()
                }
                if context_vars
                else {}
            )
            return m.ConfigMap(root=context_map)
        return r[bool].fail(f"Unknown operation: {operation}")

    @classmethod
    def _handle_context_error(
        cls,
        operation: str,
        exc: Exception,
    ) -> r[bool] | m.ConfigMap:
        """Handle context operation error."""
        if operation == c.Logging.ContextOperation.GET:
            return m.ConfigMap(root={})
        return r[bool].fail(f"Failed to {operation} global context: {exc}")

    @classmethod
    def bind_global_context(
        cls,
        **context: t.MetadataAttributeValue,
    ) -> r[bool]:
        """Bind context globally using FlextRuntime.structlog() contextvars.

        Business Rule: Binds context variables globally using structlog contextvars,
        ensuring all subsequent log messages include these context variables automatically.
        Uses FlextRuntime for centralized logging management. Global context persists
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
        return cls._context_operation("bind", **context)

    @classmethod
    def clear_global_context(cls) -> r[bool]:
        """Clear all globally bound context.

        Business Rule: Clears all globally bound context variables, removing them from
        all subsequent log messages. Uses FlextRuntime for centralized logging management.
        This operation is irreversible - all context must be rebound if needed.

        Audit Implication: Clearing global context removes audit trail context from
        log messages. Use with caution in production environments. Typically used
        during application shutdown or context reset scenarios. All context variables
        are cleared through this method, ensuring consistent context management.

        Example:
            >>> FlextLogger.clear_global_context()
            >>> # All global context cleared

        """
        return cls._context_operation("clear")

    @classmethod
    def unbind_global_context(cls, *keys: str) -> r[bool]:
        """Unbind specific keys from global context.

        Business Rule: Unbinds specific context keys from global context, removing them
        from all subsequent log messages. Uses FlextRuntime for centralized logging management.
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
            unbind_keys: list[str] = [str(key) for key in keys]
            FlextRuntime.structlog().contextvars.unbind_contextvars(*unbind_keys)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to unbind global context: {exc}")

    @classmethod
    def _get_global_context(cls) -> m.ConfigMap:
        """Get current global context (internal use only)."""
        return cls._context_operation("get")

    # =========================================================================
    # SCOPED CONTEXT MANAGEMENT - Three-tier context system
    # =========================================================================

    # Scoped context mapping for DRY binding
    _SCOPE_BINDERS: ClassVar[m.ConfigMap] = m.ConfigMap(
        root={
            c.Context.SCOPE_APPLICATION: c.Context.SCOPE_APPLICATION,
            c.Context.SCOPE_REQUEST: c.Context.SCOPE_REQUEST,
            c.Context.SCOPE_OPERATION: c.Context.SCOPE_OPERATION,
        },
    )

    @classmethod
    def bind_context(
        cls,
        scope: str,
        **context: t.MetadataAttributeValue,
    ) -> r[bool]:
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
            scope: Scope name. Use c.Context.SCOPE_* constants:
                   - SCOPE_APPLICATION: Persists for entire app lifetime
                   - SCOPE_REQUEST: Persists for single request/command
                   - SCOPE_OPERATION: Persists for single operation
            **context: Context variables to bind

        Returns:
            r[bool]: Success with True if context bound, failure with error message otherwise.

        Examples:
            >>> # Application-level context (app name, version, environment)
            >>> FlextLogger.bind_context(
            ...     c.Context.SCOPE_APPLICATION,
            ...     app_name="flext-oud-mig",
            ...     app_version="0.9.0",
            ...     environment="production",
            ... )

            >>> # Request-level context (correlation_id, command, user_id)
            >>> FlextLogger.bind_context(
            ...     c.Context.SCOPE_REQUEST,
            ...     correlation_id="flext-abc123",
            ...     command="migrate",
            ...     user_id="REDACTED_LDAP_BIND_PASSWORD",
            ... )

            >>> # Operation-level context
            >>> FlextLogger.bind_context(
            ...     c.Context.SCOPE_OPERATION,
            ...     operation="sync_users",
            ... )

        """
        try:
            if scope not in cls._scoped_contexts:
                cls._scoped_contexts[scope] = {}
            current_context: dict[str, t.GuardInputValue] = {
                key: FlextRuntime.normalize_to_general_value(value)
                for key, value in cls._scoped_contexts[scope].items()
            }
            incoming_context: dict[str, t.GuardInputValue] = {
                key: FlextRuntime.normalize_to_general_value(value)
                for key, value in context.items()
            }
            merge_result = u.merge(
                current_context,
                incoming_context,
                strategy="deep",
            )
            if merge_result.is_success:
                # merge_result.value is compatible with m.ConfigMap payloads.
                merged_context: dict[str, t.MetadataAttributeValue] = {
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in dict(merge_result.value).items()
                }
                cls._scoped_contexts[scope] = merged_context
            FlextRuntime.structlog().contextvars.bind_contextvars(**context)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to bind context for scope '{scope}': {exc}")

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
            scope: Scope to clear (use c.Context.SCOPE_* constants)

        Returns:
            r[bool]: Success with True if scope cleared, failure with error message otherwise.

        Example:
            >>> FlextLogger.clear_scope(c.Context.SCOPE_REQUEST)
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
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return r[bool].fail(f"Failed to clear scope '{scope}': {exc}")

    @classmethod
    @contextmanager
    def scoped_context(
        cls,
        scope: str,
        **context: t.MetadataAttributeValue,
    ) -> Iterator[None]:
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
            scope: Scope name (use c.Context.SCOPE_* constants)
            **context: Context variables to bind

        Yields:
            None: Context manager yields control to caller

        Example:
            >>> with FlextLogger.scoped_context(
            ...     c.Context.SCOPE_OPERATION, operation="sync_users"
            ... ):
            ...     # Context automatically bound and cleared
            ...     logger.info("Operation started")

        """
        # Use bind_context for all scopes (handles known + generic scopes)
        _ = cls.bind_context(scope, **context)

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
        **context: t.MetadataAttributeValue,
    ) -> r[bool]:
        """Bind context variables that only appear in logs at specified level or higher.

        Business Rule: Binds context variables with level prefix (`_level_{level}_`) for
        conditional inclusion based on log level. Uses FlextRuntime for centralized logging
        management. Context variables are filtered by FlextRuntime.level_based_context_filter()
        processor based on log level hierarchy. Normalizes log level to standard format
        (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        Audit Implication: Level-based context binding ensures audit trail completeness by
        including verbose context only at appropriate log levels. All level-based context
        is filtered automatically by structlog processors, ensuring efficient log processing.

        Uses FlextRuntime for centralized logging management. Context variables
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

            # Bind context with level prefix - simple dict mapping
            prefixed_context = {
                f"_level_{level_normalized}_{key}": value
                for key, value in context.items()
            }
            # Update level contexts directly (simple dict update)
            cls._level_contexts[level_normalized].update(context)

            # Use FlextRuntime for centralized logging management
            FlextRuntime.structlog().contextvars.bind_contextvars(**prefixed_context)

            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Failed to bind context for level {level}: {e}")

    @classmethod
    def unbind_context_for_level(
        cls,
        level: str,
        *keys: str,
    ) -> r[bool]:
        """Unbind context variables that were bound for a specific log level.

        Business Rule: Unbinds context variables that were bound for a specific log level,
        removing them from logs at that level or higher. Uses FlextRuntime for centralized
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

            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Failed to unbind context for level {level}: {e}")

    @classmethod
    def for_container(
        cls,
        container: p.DI,
        level: str | None = None,
        **context: t.MetadataAttributeValue,
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
        # Get log level from container config or use provided level
        if level is None:
            config = (
                container.config
                if hasattr(container, "config")
                else FlextSettings.get_global_instance()
            )
            level = config.log_level if hasattr(config, "log_level") else "INFO"
        # Create logger with container context
        logger = cls.create_module_logger(f"container_{id(container)}")
        # Bind container context
        if context:
            _ = logger.bind_global_context(**context)
        return logger

    @classmethod
    @contextmanager
    def with_container_context(
        cls,
        container: p.DI,
        level: c.Settings.LogLevel | str | None = None,
        **context: t.MetadataAttributeValue,
    ) -> Iterator[FlextLogger]:
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
            case c.Settings.LogLevel() as enum_level:
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
            # Cleanup is handled by context manager exit
            pass

    @classmethod
    def create_module_logger(cls, name: str = "flext") -> FlextLogger:
        """Create a logger instance for a module.

        Business Rule: Creates a FlextLogger instance for a specific module, using
        the module name for logger identification. Logger inherits global and scoped
        context automatically. Uses FlextRuntime for centralized logging management.
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
        # Auto-configure structlog if not already configured
        FlextRuntime.ensure_structlog_configured()
        return cls(name)

    @override
    @staticmethod
    def get_logger(
        name: str | None = None,
    ) -> p.Log.StructlogLogger:
        """Get structlog logger instance (alias for FlextRuntime.get_logger)."""
        return cast("p.Log.StructlogLogger", FlextRuntime.get_logger(name))

    # =========================================================================
    # FACTORY PATTERNS - DI-ready logger creation
    # =========================================================================

    def __init__(
        self,
        name: str,
        *,
        config: p.Config | None = None,
        _level: c.Settings.LogLevel | str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize FlextLogger with name and optional context."""
        super().__init__()

        # Extract config values (config takes priority over individual params)
        if config is not None:
            _level = config.level if hasattr(config, "level") else _level
            _service_name = (
                config.service_name
                if hasattr(config, "service_name")
                else _service_name
            )
            _service_version = (
                config.service_version
                if hasattr(config, "service_version")
                else _service_version
            )
            _correlation_id = (
                config.correlation_id
                if hasattr(config, "correlation_id")
                else _correlation_id
            )
            _force_new = (
                config.force_new if hasattr(config, "force_new") else _force_new
            )

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
        base_logger = FlextRuntime.get_logger(name)
        self.logger = cast("p.Log.StructlogLogger", base_logger.bind(**context))

        # Initialize optional state variables
        # Note: _context and _tracking are initialized as needed by methods

    @classmethod
    def create_bound_logger(
        cls,
        name: str,
        bound_logger: p.Log.StructlogLogger,
    ) -> Self:
        """Internal factory for creating logger with pre-bound structlog instance."""
        instance = cls.__new__(cls)
        instance.name = name
        instance.logger = bound_logger
        return instance

    @override
    def bind(self, **context: t.MetadataAttributeValue) -> Self:
        """Bind additional context, returning new logger (original unchanged)."""
        bound_logger = self.logger.bind(**context)
        return self.__class__.create_bound_logger(self.name, bound_logger)

    @override
    def new(self, **context: t.MetadataAttributeValue) -> Self:
        """Create new logger with context - implements BindableLogger protocol."""
        return self.bind(**context)

    @override
    def unbind(self, *keys: str) -> Self:
        """Unbind keys from logger - implements BindableLogger protocol."""
        bound_logger = self.logger.unbind(*keys)
        return self.__class__.create_bound_logger(self.name, bound_logger)

    @override
    def try_unbind(self, *keys: str) -> Self:
        """Try to unbind keys from logger - implements BindableLogger protocol."""
        bound_logger = self.logger.try_unbind(*keys)
        return self.__class__.create_bound_logger(self.name, bound_logger)

    def with_result(self) -> FlextLogger.ResultAdapter:
        """Get a result-returning logger adapter.

        Returns a ResultAdapter that wraps all logging methods
        to return r[bool] indicating success/failure.

        Returns:
            ResultAdapter wrapping this logger

        """
        return FlextLogger.ResultAdapter(self)

    # =============================================================================
    # LOGGING METHODS - DELEGATE TO FlextRuntime.structlog()
    # =============================================================================

    def trace(
        self,
        message: str,
        *args: t.GuardInputValue,
        **kwargs: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log trace message - Logger.Log implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"

            if hasattr(self.logger, "bind") and hasattr(self.logger, "info"):
                self.logger.debug(
                    formatted_message,
                    **kwargs,
                )
            return r[bool].ok(value=True)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            # Logger internals must never raise into application flows.
            FlextLogger._report_internal_logging_failure("trace", exc)
            return r[bool].fail(f"Trace logging failed: {exc}")

    @staticmethod
    def _format_log_message(
        message: str,
        *args: t.GuardInputValue,
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
                if len(parts) == c.Validation.LEVEL_PREFIX_PARTS_COUNT:
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
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Caller metadata enrichment is best-effort and must not block logging.
            FlextLogger._report_internal_logging_failure("get_caller_source_path", exc)
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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            # Path normalization is best-effort and must not block logging.
            FlextLogger._report_internal_logging_failure(
                "convert_to_relative_path",
                exc,
            )
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

    @staticmethod
    def _report_internal_logging_failure(
        operation: str,
        exc: Exception,
    ) -> None:
        with suppress(AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            FlextRuntime.structlog().get_logger("flext_core.loggings").warning(
                "Internal logger operation failed",
                operation=operation,
                exception=exc,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
            )

    def _log(
        self,
        _level: c.Settings.LogLevel | str,
        message: str,
        *args: t.GuardInputValue,
        **context: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Internal logging method - consolidates all log level methods.

        Business Rule: Internal method that consolidates all log level methods (debug,
        info, warning, error, critical) into a single implementation. Formats message
        with % arguments, auto-adds source path if not provided, and delegates to
        structlog logger. Uses FlextRuntime for centralized logging management.
        Returns r[bool] indicating success or failure.

        Audit Implication: Internal logging ensures audit trail completeness by
        formatting messages and adding source context. All log messages go through
        this method, ensuring consistent log formatting and context inclusion across
        FLEXT. Source path is critical for tracing log messages to their source in
        audit trails.
        """
        try:
            formatted_message = FlextLogger._format_log_message(message, *args)

            # Auto-add source if not provided
            if "source" not in context and (
                source_path := FlextLogger._get_caller_source_path()
            ):
                context["source"] = source_path

            # Use StrEnum directly - structlog accepts StrEnum values
            # Convert to lowercase string for method name lookup
            match _level:
                case c.Settings.LogLevel() as enum_level:
                    level_raw: str = enum_level.value
                case _:
                    level_raw = str(_level)
            level_str = level_raw.lower()

            # Dynamic method call using getattr mapping
            getattr(self.logger, level_str)(formatted_message, **context)
            # Return success result directly
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Return failure result directly
            return r[bool].fail(f"Logging failed: {e}")

    def log(
        self,
        level: str,
        message: str,
        *args: t.GuardInputValue,
        **context: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log message with specified level - Logger.Log implementation.

        Business Rule: Logs a message with specified level, converting level string
        to LogLevel enum if possible. Uses _log method for actual logging. Context
        mapping is merged with logger's bound context. Uses FlextRuntime for centralized
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
        # Convert level string to LogLevel enum if possible
        level_enum: c.Settings.LogLevel | str = level
        with suppress(ValueError, AttributeError):
            level_enum = c.Settings.LogLevel(level.upper())

        # Use _log to handle the actual logging
        return self._log(level_enum, message, *args, **context)

    @override
    def debug(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue | Exception,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log debug message - Logger.Log implementation.

        Business Rule: Logs a debug-level message with optional context. Uses _log
        method for actual logging. Uses FlextRuntime for centralized logging management.

        Audit Implication: Debug logging ensures audit trail completeness by recording
        detailed diagnostic information. Debug messages are typically filtered in
        production but critical for troubleshooting and audit trail reconstruction.
        All debug messages go through this method, ensuring consistent log formatting
        and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.Settings.LogLevel.DEBUG, msg, *args, **kw)

    @override
    def info(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log info message - Logger.Log implementation.

        Business Rule: Logs an info-level message with optional context. Uses _log
        method for actual logging. Uses FlextRuntime for centralized logging management.

        Audit Implication: Info logging ensures audit trail completeness by recording
        informational messages about application flow. Info messages are typically
        included in production logs and critical for audit trail reconstruction.
        All info messages go through this method, ensuring consistent log formatting
        and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.Settings.LogLevel.INFO, msg, *args, **kw)

    @override
    def warning(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log warning message - Logger.Log implementation.

        Business Rule: Logs a warning-level message with optional context. Uses _log
        method for actual logging. Uses FlextRuntime for centralized logging management.

        Audit Implication: Warning logging ensures audit trail completeness by recording
        warning messages about potential issues. Warning messages are typically included
        in production logs and critical for audit trail reconstruction and issue
        identification. All warning messages go through this method, ensuring consistent
        log formatting and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.Settings.LogLevel.WARNING, msg, *args, **kw)

    @override
    def warn(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue,
        **kw: t.GuardInputValue,
    ) -> r[bool]:
        """Alias for warning() - implements p.Log.StructlogLogger protocol."""
        return self._log_standard_level(c.Settings.LogLevel.WARNING, msg, *args, **kw)

    @override
    def error(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log error message - Logger.Log implementation.

        Business Rule: Logs an error-level message with optional context. Uses _log
        method for actual logging. Uses FlextRuntime for centralized logging management.

        Audit Implication: Error logging ensures audit trail completeness by recording
        error messages about failures. Error messages are always included in production
        logs and critical for audit trail reconstruction and failure analysis. All
        error messages go through this method, ensuring consistent log formatting and
        context inclusion across FLEXT.
        """
        return self._log_standard_level(c.Settings.LogLevel.ERROR, msg, *args, **kw)

    @override
    def critical(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log critical message - Logger.Log implementation.

        Business Rule: Logs a critical-level message with optional context. Uses _log
        method for actual logging. Uses FlextRuntime for centralized logging management.

        Audit Implication: Critical logging ensures audit trail completeness by recording
        critical messages about severe failures. Critical messages are always included
        in production logs and critical for audit trail reconstruction and emergency
        response. All critical messages go through this method, ensuring consistent
        log formatting and context inclusion across FLEXT.
        """
        return self._log_standard_level(c.Settings.LogLevel.CRITICAL, msg, *args, **kw)

    def _log_standard_level(
        self,
        level: c.Settings.LogLevel,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue | Exception,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        message = str(msg)
        filtered_args: tuple[t.GuardInputValue, ...] = tuple(
            arg
            for arg in args
            # boundary: logging accepts arbitrary user args
            if not isinstance(arg, BaseException)
        )
        return self._log(level, message, *filtered_args, **kw)

    @staticmethod
    def _should_include_stack_trace() -> bool:
        try:
            config = FlextSettings.get_global_instance()
            return config.effective_log_level.upper() == c.Settings.LogLevel.DEBUG.value
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Stack-trace policy must fail open to preserve diagnostics.
            FlextLogger._report_internal_logging_failure(
                "should_include_stack_trace",
                exc,
            )
            return True

    def build_exception_context(
        self,
        *,
        exception: Exception | None,
        exc_info: bool,
        context: Mapping[str, t.GuardInputValue | Exception],
    ) -> m.ConfigMap:
        """Build normalized context payload for exception/error logging."""
        include_stack_trace = self._should_include_stack_trace()
        context_dict: m.ConfigMap = m.ConfigMap(root={})

        if exception is not None:
            exception_data: m.ConfigMap = m.ConfigMap(
                root={
                    "exception_type": exception.__class__.__name__,
                    "exception_message": str(exception),
                },
            )
            merge_result = u.merge(
                dict(context_dict.root),
                dict(exception_data.root),
                strategy="deep",
            )
            if merge_result.is_success:
                context_dict = m.ConfigMap(root=dict(merge_result.value))
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
            # boundary: logging accepts arbitrary user args
            if not isinstance(value, BaseException):
                context_dict[key] = FlextRuntime.normalize_to_general_value(value)

        return context_dict

    @override
    def exception(
        self,
        msg: str | t.GuardInputValue,
        *args: t.GuardInputValue,
        **kw: t.GuardInputValue | Exception,
    ) -> r[bool]:
        """Log exception with conditional stack trace (DEBUG only).

        Business Rule: Logs an exception with conditional stack trace inclusion based
        on effective log level. Stack trace is included only if effective log level is
        DEBUG. Exception details (type, message, stack trace) are added to context.
        Uses FlextRuntime for centralized logging management.

        Audit Implication: Exception logging ensures audit trail completeness by recording
        exception details and stack traces. Stack traces are critical for troubleshooting
        but may contain sensitive information, so they're only included at DEBUG level.
        All exception messages go through this method, ensuring consistent log formatting
        and context inclusion across FLEXT. Exception details are critical for audit
        trail reconstruction and failure analysis.
        """
        # Convert msg to string if needed for internal processing
        message = str(msg)
        # Filter out Exception types from args for format string interpolation
        filtered_args: tuple[t.GuardInputValue, ...] = tuple(
            arg
            for arg in args
            # boundary: logging accepts arbitrary user args
            if not isinstance(arg, BaseException)
        )

        try:
            exception_value = kw.get("exception")
            exc_info_value = kw.get("exc_info", True)
            context_input = {
                key: value
                for key, value in kw.items()
                if key not in {"exception", "exc_info"}
            }

            match exception_value:
                case Exception() as exc:
                    resolved_exception: Exception | None = exc
                case _:
                    resolved_exception = None

            context_dict = self.build_exception_context(
                exception=resolved_exception,
                exc_info=bool(exc_info_value),
                context=context_input,
            )

            if resolved_exception is None and isinstance(
                exception_value,
                BaseException,
            ):
                context_dict["exception_type"] = exception_value.__class__.__name__
                context_dict["exception_message"] = str(exception_value)

            error_method = self.logger.error if hasattr(self.logger, "error") else None
            if callable(error_method):
                _ = error_method(message, *filtered_args, **context_dict.root)
            return r[bool].ok(value=True)
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            # Logger internals must never raise into application flows.
            FlextLogger._report_internal_logging_failure("exception", exc)
            return r[bool].fail(f"Exception logging failed: {exc}")

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

            context: m.ConfigMap = m.ConfigMap(
                root={
                    "duration_seconds": elapsed,
                    "operation": self._operation_name,
                    "status": status,
                },
            )

            if not is_success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""

            _ = log_method(
                f"{self._operation_name} {status}",
                **context.root,
            )

    class ResultAdapter:
        """Adapter ensuring FlextLogger methods return FlextResult outputs.

        Provides explicit wrapper methods for common logger operations.
        For other methods, access _base_logger directly.
        """

        __slots__ = ("_base_logger",)

        def __init__(self, base_logger: FlextLogger) -> None:
            """Initialize adapter with base logger."""
            super().__init__()
            self._base_logger = base_logger

        def _call_optional(
            self,
            method_name: str,
            *args: t.MetadataAttributeValue | FlextResult[t.MetadataAttributeValue],
            **kwargs: t.GuardInputValue,
        ) -> t.GuardInputValue | None:
            method = (
                getattr(self._base_logger, method_name)
                if hasattr(self._base_logger, method_name)
                else None
            )
            if callable(method):
                result = method(*args, **kwargs)
                if FlextUtilitiesGuards.is_general_value_type(result):
                    return FlextRuntime.normalize_to_general_value(result)
                return str(result)
            return None

        def _log_and_wrap(
            self,
            method_name: str,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            log_method = getattr(self._base_logger, method_name)
            log_method(message, *args, **kwargs)
            return r[bool].ok(value=True)

        @property
        def name(self) -> str:
            """Get logger name - delegate to base logger."""
            return self._base_logger.name

        # Explicit wrapper methods for common operations
        # Use getattr for optional methods that may not exist on base logger
        def track_performance(
            self,
            operation_name: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> t.GuardInputValue | None:
            """Track performance metrics - delegate to base logger."""
            return self._call_optional(
                "track_performance",
                operation_name,
                *args,
                **kwargs,
            )

        def log_result(
            self,
            result: FlextResult[t.MetadataAttributeValue],
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> t.GuardInputValue | None:
            """Log result - delegate to base logger."""
            return self._call_optional("log_result", result, *args, **kwargs)

        def bind_context(
            self,
            **context: t.MetadataAttributeValue,
        ) -> t.GuardInputValue | None:
            """Bind context - delegate to base logger."""
            return self._call_optional("bind_context", "default", **context)

        def get_context(
            self,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> t.GuardInputValue | None:
            """Get context - delegate to base logger."""
            return self._call_optional("get_context", *args, **kwargs)

        def start_tracking(
            self,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> t.GuardInputValue | None:
            """Start tracking - delegate to base logger."""
            return self._call_optional("start_tracking", *args, **kwargs)

        def stop_tracking(
            self,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> t.GuardInputValue | None:
            """Stop tracking - delegate to base logger."""
            return self._call_optional("stop_tracking", *args, **kwargs)

        def with_result(self) -> FlextLogger.ResultAdapter:
            """Return self (idempotent)."""
            return self

        def bind(
            self,
            **context: t.MetadataAttributeValue,
        ) -> FlextLogger.ResultAdapter:
            """Bind context preserving adapter wrapper."""
            return FlextLogger.ResultAdapter(self._base_logger.bind(**context))

        def trace(
            self,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log trace message returning FlextResult."""
            return self._log_and_wrap("trace", message, *args, **kwargs)

        def debug(
            self,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log debug message returning FlextResult."""
            return self._log_and_wrap("debug", message, *args, **kwargs)

        def info(
            self,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log info message returning FlextResult."""
            return self._log_and_wrap("info", message, *args, **kwargs)

        def warning(
            self,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log warning message returning FlextResult."""
            return self._log_and_wrap("warning", message, *args, **kwargs)

        def error(
            self,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log error message returning FlextResult."""
            return self._log_and_wrap("error", message, *args, **kwargs)

        def critical(
            self,
            message: str,
            *args: t.MetadataAttributeValue,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log critical message returning FlextResult."""
            return self._log_and_wrap("critical", message, *args, **kwargs)

        def exception(
            self,
            message: str,
            *,
            exception: BaseException | None = None,
            exc_info: bool = True,
            **kwargs: t.MetadataAttributeValue,
        ) -> r[bool]:
            """Log exception with traceback returning FlextResult.

            Note: Uses error() with exception context as FlextLogger.exception()
            is typically used in adapter context where direct exception handling
            is done by the base logger.
            """
            match exception:
                case Exception() as exc:
                    resolved_exception: Exception | None = exc
                case _:
                    resolved_exception = None
            context_kwargs = kwargs
            match exception:
                case BaseException() as non_standard_exception:
                    if resolved_exception is None:
                        context_kwargs = {
                            **kwargs,
                            "exception_type": non_standard_exception.__class__.__name__,
                            "exception_message": str(non_standard_exception),
                        }
                case _:
                    pass

            context = self._base_logger.build_exception_context(
                exception=resolved_exception,
                exc_info=exc_info,
                context=context_kwargs,
            )
            self._base_logger.error(message, **context.root)
            return r[bool].ok(value=True)


__all__: list[str] = [
    "FlextLogger",
]
