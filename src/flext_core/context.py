"""Context and correlation utilities enabling the context-first pillar.

The helpers correspond to the observability commitments in ``README.md`` and
``docs/architecture.md`` for the FLEXT 1.0.0 release: correlation inheritance,
request metadata, and latency tracking that integrate directly with
``FlextDispatcher`` and ``FlextLogger``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import collections.abc
import contextlib
import json
import threading
import time
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import (
    Final,
    override,
)

import structlog
import structlog.contextvars

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextContext:
    """Hierarchical context manager for request-, service-, and perf-scopes.

    It is the single entry point referenced across the modernization plan:
    all dispatcher, container, and logging surfaces depend on context vars
    to propagate correlation IDs and structured metadata.

    **Function**: Context management for distributed systems
        - Correlation ID management for distributed tracing
        - Service identification context (name, version)
        - Request context with user and operation metadata
        - Performance tracking with timing operations
        - Scoped context storage (global, user, session)
        - Thread-safe context variable management
        - Context serialization for cross-service propagation
        - Hook system for context change notifications
        - Context cloning and merging capabilities
        - Statistics tracking for context operations

    **Uses**: Core infrastructure for context management
        - contextvars for thread-local context variables
        - threading.RLock for thread-safe operations
        - FlextTypes.Dict for type-safe dictionaries
        - FlextResult[T] for operation results
        - datetime for timestamp operations
        - uuid for correlation ID generation
        - json for context serialization
        - contextmanager for context scope management
        - collections.abc.Mapping for type checking
        - contextlib for context helpers

    **How to use**: Context management and propagation
        ```python
        from flext_core import FlextContext

        # Example 1: Create context instance
        context = FlextContext()
        context.set("user_id", "123")
        user_id = context.get("user_id")

        # Example 2: Correlation ID management
        with FlextContext.Correlation.new_correlation() as corr_id:
            # Correlation ID is automatically propagated
            print(f"Processing request {corr_id}")
            # All operations within this scope share correlation

        # Example 3: Service context
        with FlextContext.Service.service_context("user-service", version="1.0.0"):
            # Service name/version available to logging
            pass

        # Example 4: Request context with metadata
        with FlextContext.Request.request_context(
            user_id="user123", operation_name="create_user", request_id="req-abc"
        ):
            # Request metadata propagates automatically
            pass

        # Example 5: Timed performance tracking
        with FlextContext.Performance.timed_operation("database_query") as metrics:
            # Perform operation
            pass
            # metrics contains duration_seconds

        # Example 6: Cross-service propagation
        headers = FlextContext.Serialization.get_correlation_context()
        # Send headers to downstream service
        # Downstream service:
        FlextContext.Serialization.set_from_context(headers)

        # Example 7: Scoped storage
        context = FlextContext()
        context.set("key", "value", scope="user")
        value = context.get("key", scope="user")
        ```

    Attributes:
        Variables: Context variable definitions by domain.
        Correlation: Correlation ID management utilities.
        Service: Service identification context helpers.
        Request: Request-level context management.
        Performance: Performance tracking and timing.
        Serialization: Context serialization utilities.
        Utilities: Context utility methods and helpers.
        HandlerExecutionContext: Handler execution context.

    Note:
        All context variables are thread-safe using contextvars.
        Correlation IDs automatically propagate through scopes.
        Context can be serialized for cross-service communication.
        Statistics track all context operations. Global context
        uses singleton pattern with thread-safe access.

    Warning:
        Context must be cleared manually when no longer needed.
        Context hooks may impact performance on high-frequency ops.
        Suspended contexts do not respond to operations.
        Context serialization does not include hooks/statistics.

    Example:
        Complete context workflow with correlation and timing:

        >>> context = FlextContext()
        >>> with FlextContext.Correlation.new_correlation() as cid:
        ...     with FlextContext.Performance.timed_operation() as m:
        ...         context.set("processed", True)
        ...         print(cid)
        corr_12345678
        >>> print(context.get("processed"))
        True

    See Also:
        FlextLogger: For structured logging with context.
        FlextDispatcher: For context-aware message dispatch.
        FlextBus: For command/query execution with context.
        FlextConfig: For configuration management.

    """

    # =========================================================================
    # PRIVATE HELPERS - Internal data structures
    # =========================================================================

    @dataclass(slots=True)
    class _ContextData:
        """Lightweight container for initializing context state.

        Nested class for context-specific initialization data following
        SOLID principles. This is an implementation detail of FlextContext.
        """

        data: FlextTypes.Dict = field(default_factory=dict)
        metadata: FlextTypes.Dict = field(default_factory=dict)

    @dataclass(slots=True)
    class _ContextExport:
        """Typed snapshot returned by `export_snapshot`.

        Nested class for context-specific export data following
        SOLID principles. This is an implementation detail of FlextContext.
        """

        data: FlextTypes.Dict = field(default_factory=dict)
        metadata: FlextTypes.Dict = field(default_factory=dict)
        statistics: FlextTypes.Dict = field(default_factory=dict)

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    @override
    def __init__(
        self,
        initial_data: FlextContext._ContextData | FlextTypes.Dict | None = None,
    ) -> None:
        """Initialize FlextContext with optional initial data.

        Args:
            initial_data: Optional context data (dict or `_ContextData`)

        """
        if initial_data is None:
            context_data = FlextContext._ContextData()
        elif isinstance(initial_data, dict):
            context_data = FlextContext._ContextData(data=initial_data)
        else:
            context_data = initial_data

        self._data: FlextTypes.Dict = context_data.data
        self._metadata: FlextTypes.Dict = context_data.metadata
        self._hooks: FlextTypes.Context.HookRegistry = {}
        self._statistics: FlextTypes.Dict = {
            "operations": {
                "set": 0,
                "get": 0,
                "remove": 0,
                "clear": 0,
            },
            "sets": 0,
            "gets": 0,
            "removes": 0,
            "clears": 0,
        }
        self._active = True
        self._suspended = False
        self._lock = threading.RLock()
        # Scope-based storage
        self._scopes: FlextTypes.Context.ScopeRegistry = {
            "global": self._data,
            "user": {},
            "session": {},
        }

    # =========================================================================
    # Instance Methods - Core context operations
    # =========================================================================

    def set(self, key: str, value: object, scope: str = "global") -> None:
        """Set a value in the context.

        Args:
            key: The key to set
            value: The value to set
            scope: The scope for the value (global, user, session)

        """
        if not self._active:
            return

        # Validate key type; allow empty keys for compatibility with validation flow
        if not isinstance(key, str):
            msg = "Key must be a string"
            raise TypeError(msg)

        if not key:
            structlog.get_logger(__name__).warning(
                "Context key is empty; validation will fail",
                scope=scope,
            )

        # Validate value is serializable
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            msg = "Value must be serializable"
            raise TypeError(msg)

        with self._lock:
            # Ensure the scope exists in _scopes
            if scope not in self._scopes:
                self._scopes[scope] = {}

            scope_data = self._scopes[scope]
            scope_data[key] = value
            sets_count = self._statistics.get("sets", 0)
            if isinstance(sets_count, int):
                self._statistics["sets"] = sets_count + 1

            operations = self._statistics.get("operations", {})
            if isinstance(operations, dict) and "set" in operations:
                set_count = operations["set"]
                if isinstance(set_count, int):
                    operations["set"] = set_count + 1

            # Execute hooks
            if "set" in self._hooks:
                for hook in self._hooks["set"]:
                    with contextlib.suppress(Exception):
                        hook(key, value)

    def get(self, key: str, default: object = None, scope: str = "global") -> object:
        """Get a value from the context.

        Args:
            key: The key to get
            default: Default value if key not found
            scope: The scope to get from (global, user, session)

        Returns:
            The value or default

        """
        if not self._active:
            return default

        with self._lock:
            scope_data = self._scopes.get(scope, {})
            gets_count = self._statistics.get("gets", 0)
            if isinstance(gets_count, int):
                self._statistics["gets"] = gets_count + 1

            operations = self._statistics.get("operations", {})
            if isinstance(operations, dict) and "get" in operations:
                get_count = operations["get"]
                if isinstance(get_count, int):
                    operations["get"] = get_count + 1
            return scope_data.get(key, default)

    def has(self, key: str, scope: str = "global") -> bool:
        """Check if a key exists in the context.

        Args:
            key: The key to check
            scope: The scope to check (global, user, session)

        Returns:
            True if key exists, False otherwise

        """
        if not self._active:
            return False

        with self._lock:
            scope_data = self._scopes.get(scope, {})
            return key in scope_data

    def remove(self, key: str, scope: str = "global") -> None:
        """Remove a key from the context.

        Args:
            key: The key to remove
            scope: The scope to remove from (global, user, session)

        """
        if not self._active:
            return

        with self._lock:
            scope_data = self._scopes.get(scope, {})
            if key in scope_data:
                del scope_data[key]
                removes_count = self._statistics.get("removes", 0)
                if isinstance(removes_count, int):
                    self._statistics["removes"] = removes_count + 1

                operations = self._statistics.get("operations", {})
                if isinstance(operations, dict) and "remove" in operations:
                    remove_count = operations["remove"]
                    if isinstance(remove_count, int):
                        operations["remove"] = remove_count + 1

    def clear(self) -> None:
        """Clear all data from the context."""
        if not self._active:
            return

        with self._lock:
            # Clear all scopes
            for scope_data in self._scopes.values():
                scope_data.clear()
            clears_count = self._statistics.get("clears", 0)
            if isinstance(clears_count, int):
                self._statistics["clears"] = clears_count + 1

            operations = self._statistics.get("operations", {})
            if isinstance(operations, dict) and "clear" in operations:
                clear_count = operations["clear"]
                if isinstance(clear_count, int):
                    operations["clear"] = clear_count + 1

    def keys(self) -> FlextTypes.StringList:
        """Get all keys in the context.

        Returns:
            List of all keys

        """
        if not self._active:
            return []

        with self._lock:
            all_keys: set[str] = set()
            for scope_data in self._scopes.values():
                all_keys.update(scope_data.keys())
            return list(all_keys)

    def values(self) -> FlextTypes.List:
        """Get all values in the context.

        Returns:
            List of all values

        """
        if not self._active:
            return []

        with self._lock:
            all_values: FlextTypes.List = []
            for scope_data in self._scopes.values():
                all_values.extend(scope_data.values())
            return all_values

    def items(self) -> list[tuple[str, object]]:
        """Get all key-value pairs in the context.

        Returns:
            List of (key, value) tuples

        """
        if not self._active:
            return []

        with self._lock:
            all_items: list[tuple[str, object]] = []
            for scope_data in self._scopes.values():
                all_items.extend(scope_data.items())
            return all_items

    def merge(self, other: FlextContext | FlextTypes.Dict) -> FlextContext:
        """Merge another context or dictionary into this context.

        Args:
            other: Another FlextContext or dictionary to merge

        Returns:
            Self for chaining

        """
        if not self._active:
            return self

        with self._lock:
            if isinstance(other, FlextContext):
                # Merge all scopes from the other context
                other_scopes = other.get_all_scopes()
                for scope_name, scope_data in other_scopes.items():
                    if scope_name not in self._scopes:
                        self._scopes[scope_name] = {}
                    self._scopes[scope_name].update(scope_data)
            else:
                # Merge into global scope
                self._scopes["global"].update(other)
        return self

    def clone(self) -> FlextContext:
        """Create a clone of this context.

        Returns:
            A new FlextContext with the same data

        """
        with self._lock:
            cloned = FlextContext()
            # Clone all scopes
            cloned._scopes = {
                scope_name: scope_data.copy()
                for scope_name, scope_data in self._scopes.items()
            }
            cloned._metadata = self._metadata.copy()
            cloned._statistics = self._statistics.copy()
            return cloned

    def validate(self) -> FlextResult[None]:
        """Validate the context data.

        Returns:
            FlextResult indicating validation success or failure

        """
        if not self._active:
            return FlextResult[None].fail("Context is not active")

        with self._lock:
            # Check for empty keys in all scopes
            for scope_data in self._scopes.values():
                for key in scope_data:
                    if not key:
                        return FlextResult[None].fail("Invalid key found in context")
            return FlextResult[None].ok(None)

    def to_json(self) -> str:
        """Convert context to JSON string.

        Returns:
            JSON string representation of the context

        """
        with self._lock:
            # Combine all scopes into a single flat dictionary for backward compatibility
            all_data: FlextTypes.Dict = {}
            for scope_data in self._scopes.values():
                if isinstance(scope_data, dict):
                    all_data.update(scope_data)
            return json.dumps(all_data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> FlextContext:
        """Create context from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            New FlextContext instance

        """
        data = json.loads(json_str)
        context = cls()

        # Handle both old flat format and new scoped format
        if isinstance(data, dict):
            if all(isinstance(v, dict) for v in data.values()):
                # New scoped format
                context._scopes = data
            else:
                # Old flat format - put everything in global scope
                context._scopes["global"] = data

        return context

    def is_active(self) -> bool:
        """Check if context is active.

        Returns:
            True if context is active, False otherwise

        """
        return self._active and not self._suspended

    def suspend(self) -> None:
        """Suspend the context."""
        self._suspended = True

    def resume(self) -> None:
        """Resume the context."""
        self._suspended = False

    def destroy(self) -> None:
        """Destroy the context."""
        with self._lock:
            self._active = False
            # Clear all scopes
            for scope_data in self._scopes.values():
                scope_data.clear()
            self._metadata.clear()
            self._hooks.clear()

    def add_hook(self, event: str, hook: collections.abc.Callable[..., object]) -> None:
        """Add a hook for context events.

        Args:
            event: The event to hook (set, get, remove, clear)
            hook: The hook function to call

        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(hook)

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata for the context.

        Args:
            key: The metadata key
            value: The metadata value

        """
        with self._lock:
            self._metadata[key] = value

    def get_metadata(self, key: str, default: object = None) -> object:
        """Get metadata from the context.

        Args:
            key: The metadata key
            default: Default value if key not found

        Returns:
            The metadata value or default

        """
        with self._lock:
            return self._metadata.get(key, default)

    def get_all_metadata(self) -> FlextTypes.Dict:
        """Get all metadata from the context.

        Returns:
            Dictionary of all metadata

        """
        with self._lock:
            return self._metadata.copy()

    def get_all_data(self) -> FlextTypes.Dict:
        """Get all data from the context.

        Returns:
            Dictionary of all context data

        """
        with self._lock:
            return self._data.copy()

    def get_statistics(self) -> FlextTypes.Dict:
        """Get context statistics.

        Returns:
            Dictionary of context statistics

        """
        with self._lock:
            return self._statistics.copy()

    def cleanup(self) -> None:
        """Clean up the context."""
        with self._lock:
            # Clear all scopes
            for scope_data in self._scopes.values():
                scope_data.clear()
            self._metadata.clear()

    def get_all_scopes(self) -> FlextTypes.Context.ScopeRegistry:
        """Get all scopes.

        Returns:
            Dictionary of all scopes

        """
        with self._lock:
            return self._scopes.copy()

    def export(self) -> FlextTypes.Dict:
        """Export context data as a dictionary for compatibility consumers."""
        export_snapshot = self.export_snapshot()
        return dict(export_snapshot.data)

    def export_snapshot(self) -> FlextContext._ContextExport:
        """Return typed export snapshot including metadata and statistics."""
        with self._lock:
            all_data: FlextTypes.Dict = {}
            for scope_data in self._scopes.values():
                all_data.update(scope_data)

            return FlextContext._ContextExport(
                data=all_data,
                metadata=self._metadata.copy(),
                statistics=self._statistics.copy(),
            )

    def import_data(self, data: FlextTypes.Dict) -> None:
        """Import context data.

        Args:
            data: Dictionary containing context data

        """
        with self._lock:
            # Import data into global scope
            self._scopes["global"].update(data)

    # =========================================================================
    # Global Context Management - Static methods for global context
    # =========================================================================

    _global_context: FlextContext | None = None
    _global_lock = threading.RLock()

    @classmethod
    def get_global(cls) -> FlextContext:
        """REMOVED: Use direct instantiation or pass instance explicitly.

        Migration:
            # Old pattern
            context = FlextContext.get_global()

            # New pattern - create instance
            context = FlextContext()

            # Or for singleton pattern, manage explicitly
            _context_lock = threading.RLock()
            _context_instance: FlextContext | None = None

            with _context_lock:
                if _context_instance is None:
                    _context_instance = FlextContext()
                context = _context_instance

        """
        msg = (
            "FlextContext.get_global() has been removed. "
            "Use FlextContext() to create instances directly."
        )
        raise NotImplementedError(msg)

    @classmethod
    def reset_global(cls) -> None:
        """REMOVED: Manage singleton lifecycle explicitly if needed.

        Migration:
            # Old pattern
            FlextContext.reset_global()

            # New pattern - manage your own singleton
            _context_instance = None  # Reset in your own code

        """
        msg = (
            "FlextContext.reset_global() has been removed. "
            "Manage singleton lifecycle explicitly in your code."
        )
        raise NotImplementedError(msg)

    # =========================================================================
    # Static Methods - Context variables organized by functionality
    # =========================================================================

    class Variables:
        """Context variables organized by domain."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "correlation_id",
                default=None,
            )
            PARENT_CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "parent_correlation_id",
                default=None,
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[ContextVar[str | None]] = ContextVar(
                "service_name",
                default=None,
            )
            SERVICE_VERSION: Final[ContextVar[str | None]] = ContextVar(
                "service_version",
                default=None,
            )
            ENVIRONMENT: Final[ContextVar[str | None]] = ContextVar(
                "environment",
                default=None,
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[ContextVar[str | None]] = ContextVar("user_id", default=None)
            REQUEST_ID: Final[ContextVar[str | None]] = ContextVar(
                "request_id",
                default=None,
            )
            REQUEST_TIMESTAMP: Final[ContextVar[datetime | None]] = ContextVar(
                "request_timestamp",
                default=None,
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[ContextVar[str | None]] = ContextVar(
                "operation_name",
                default=None,
            )
            OPERATION_START_TIME: Final[ContextVar[datetime | None]] = ContextVar(
                "operation_start_time",
                default=None,
            )
            OPERATION_METADATA: Final[ContextVar[FlextTypes.Dict | None]] = ContextVar(
                "operation_metadata", default=None
            )

    # =========================================================================
    # Correlation Domain - Distributed tracing and correlation ID management
    # =========================================================================

    class Correlation:
        """Distributed tracing and correlation ID management."""

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get()

        @staticmethod
        def set_correlation_id(correlation_id: str) -> None:
            """Set correlation ID."""
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate unique correlation ID."""
            correlation_id = f"corr_{str(uuid.uuid4())[:8]}"
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID."""
            return FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()

        @staticmethod
        def set_parent_correlation_id(parent_id: str) -> None:
            """Set parent correlation ID."""
            FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create correlation context scope."""
            # Generate correlation ID if not provided
            if correlation_id is None:
                correlation_id = f"corr_{str(uuid.uuid4())[:8]}"

            # Save current context
            current_correlation = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
            )

            # Set new context
            correlation_token = FlextContext.Variables.Correlation.CORRELATION_ID.set(
                correlation_id,
            )

            # Set parent context
            parent_token: Token[str | None] | None = None
            if parent_id:
                parent_token = (
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                        parent_id,
                    )
                )
            elif current_correlation:
                # Current correlation becomes parent
                parent_token = (
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                        current_correlation,
                    )
                )

            try:
                yield correlation_id
            finally:
                # Restore previous context
                FlextContext.Variables.Correlation.CORRELATION_ID.reset(
                    correlation_token,
                )
                if parent_token:
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.reset(
                        parent_token,
                    )

        @staticmethod
        @contextmanager
        def inherit_correlation() -> Generator[str | None]:
            """Inherit or create correlation ID."""
            existing_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if existing_id:
                # Use existing correlation
                yield existing_id
            else:
                # Create new correlation context
                with FlextContext.Correlation.new_correlation() as new_id:
                    yield new_id

    # =========================================================================
    # Service Domain - Service identification and lifecycle context
    # =========================================================================

    class Service:
        """Service identification and lifecycle context management."""

        @staticmethod
        def get_service_name() -> str | None:
            """Get current service name."""
            return FlextContext.Variables.Service.SERVICE_NAME.get()

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set service name."""
            FlextContext.Variables.Service.SERVICE_NAME.set(service_name)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(service_name=service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get current service version."""
            return FlextContext.Variables.Service.SERVICE_VERSION.get()

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version."""
            FlextContext.Variables.Service.SERVICE_VERSION.set(version)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(service_version=version)

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service context scope."""
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Service.SERVICE_NAME.get()
            _ = FlextContext.Variables.Service.SERVICE_VERSION.get()

            # Set new context
            name_token = FlextContext.Variables.Service.SERVICE_NAME.set(service_name)
            version_token = None
            if version:
                version_token = FlextContext.Variables.Service.SERVICE_VERSION.set(
                    version,
                )

            try:
                yield
            finally:
                # Restore previous context
                FlextContext.Variables.Service.SERVICE_NAME.reset(name_token)
                if version_token:
                    FlextContext.Variables.Service.SERVICE_VERSION.reset(version_token)

    # =========================================================================
    # Request Domain - User and request metadata management
    # =========================================================================

    class Request:
        """Request-level context management for user and operation metadata."""

        @staticmethod
        def get_user_id() -> str | None:
            """Get current user ID."""
            return FlextContext.Variables.Request.USER_ID.get()

        @staticmethod
        def set_user_id(user_id: str) -> None:
            """Set user ID in context."""
            FlextContext.Variables.Request.USER_ID.set(user_id)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(user_id=user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context."""
            return FlextContext.Variables.Performance.OPERATION_NAME.get()

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context."""
            FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(operation_name=operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context."""
            return FlextContext.Variables.Request.REQUEST_ID.get()

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context."""
            FlextContext.Variables.Request.REQUEST_ID.set(request_id)
            # Also set in structlog context if available
            if structlog is not None:
                structlog.contextvars.bind_contextvars(request_id=request_id)

        @staticmethod
        @contextmanager
        def request_context(
            *,
            user_id: str | None = None,
            operation_name: str | None = None,
            request_id: str | None = None,
            metadata: FlextTypes.Dict | None = None,
        ) -> Generator[None]:
            """Create request metadata context scope with automatic cleanup."""
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Request.USER_ID.get()
            _ = FlextContext.Variables.Performance.OPERATION_NAME.get()
            _ = FlextContext.Variables.Request.REQUEST_ID.get()
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.get()

            # Set new context
            user_token = (
                FlextContext.Variables.Request.USER_ID.set(user_id) if user_id else None
            )
            operation_token = (
                FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)
                if operation_name
                else None
            )
            request_token = (
                FlextContext.Variables.Request.REQUEST_ID.set(request_id)
                if request_id
                else None
            )
            metadata_token = (
                FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)
                if metadata
                else None
            )

            try:
                yield
            finally:
                # Restore previous context
                if user_token is not None:
                    FlextContext.Variables.Request.USER_ID.reset(user_token)
                if operation_token is not None:
                    FlextContext.Variables.Performance.OPERATION_NAME.reset(
                        operation_token,
                    )
                if request_token is not None:
                    FlextContext.Variables.Request.REQUEST_ID.reset(request_token)
                if metadata_token is not None:
                    FlextContext.Variables.Performance.OPERATION_METADATA.reset(
                        metadata_token,
                    )

    # =========================================================================
    # Performance Domain - Operation timing and performance tracking
    # =========================================================================

    class Performance:
        """Performance monitoring and timing context management for operations."""

        @staticmethod
        def get_operation_start_time() -> datetime | None:
            """Get operation start time from context."""
            return FlextContext.Variables.Performance.OPERATION_START_TIME.get()

        @staticmethod
        def set_operation_start_time(start_time: datetime | None = None) -> None:
            """Set operation start time in context."""
            if start_time is None:
                start_time = datetime.now(UTC)
            FlextContext.Variables.Performance.OPERATION_START_TIME.set(start_time)

        @staticmethod
        def get_operation_metadata() -> FlextTypes.Dict | None:
            """Get operation metadata from context."""
            return FlextContext.Variables.Performance.OPERATION_METADATA.get()

        @staticmethod
        def set_operation_metadata(metadata: FlextTypes.Dict) -> None:
            """Set operation metadata in context."""
            FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)

        @staticmethod
        def add_operation_metadata(key: str, value: object) -> None:
            """Add single metadata entry to operation context."""
            current_metadata = (
                FlextContext.Variables.Performance.OPERATION_METADATA.get() or {}
            )
            current_metadata[key] = value
            FlextContext.Variables.Performance.OPERATION_METADATA.set(current_metadata)

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[FlextTypes.Dict]:
            """Create timed operation context with performance tracking."""
            start_time = datetime.now(UTC)
            operation_metadata: FlextTypes.Dict = {
                "start_time": start_time,
                "operation_name": operation_name,
            }

            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Performance.OPERATION_START_TIME.get()
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.get()
            _ = FlextContext.Variables.Performance.OPERATION_NAME.get()

            # Set new context
            start_token = FlextContext.Variables.Performance.OPERATION_START_TIME.set(
                start_time,
            )
            metadata_token = FlextContext.Variables.Performance.OPERATION_METADATA.set(
                operation_metadata,
            )
            operation_token = None
            if operation_name:
                operation_token = FlextContext.Variables.Performance.OPERATION_NAME.set(
                    operation_name,
                )

            try:
                yield operation_metadata
            finally:
                # Calculate duration
                end_time = datetime.now(UTC)
                duration = (end_time - start_time).total_seconds()
                operation_metadata.update(
                    {
                        "end_time": end_time,
                        "duration_seconds": duration,
                    },
                )

                # Restore previous context
                FlextContext.Variables.Performance.OPERATION_START_TIME.reset(
                    start_token,
                )
                FlextContext.Variables.Performance.OPERATION_METADATA.reset(
                    metadata_token,
                )
                if operation_token:
                    FlextContext.Variables.Performance.OPERATION_NAME.reset(
                        operation_token,
                    )

    # =========================================================================
    # Serialization Domain - Context serialization for cross-service communication
    # =========================================================================

    class Serialization:
        """Context serialization and deserialization for cross-service communication."""

        @staticmethod
        def get_full_context() -> FlextTypes.Dict:
            """Get current context as dictionary."""
            context_vars = FlextContext.Variables
            return {
                "correlation_id": context_vars.Correlation.CORRELATION_ID.get(),
                "parent_correlation_id": context_vars.Correlation.PARENT_CORRELATION_ID.get(),
                "service_name": context_vars.Service.SERVICE_NAME.get(),
                "service_version": context_vars.Service.SERVICE_VERSION.get(),
                "user_id": context_vars.Request.USER_ID.get(),
                "operation_name": context_vars.Performance.OPERATION_NAME.get(),
                "request_id": context_vars.Request.REQUEST_ID.get(),
                "operation_start_time": context_vars.Performance.OPERATION_START_TIME.get(),
                "operation_metadata": context_vars.Performance.OPERATION_METADATA.get(),
            }

        @staticmethod
        def get_correlation_context() -> FlextTypes.StringDict:
            """Get correlation context for cross-service propagation."""
            context: FlextTypes.StringDict = {}

            correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if correlation_id:
                context["X-Correlation-Id"] = str(correlation_id)

            parent_id = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()
            if parent_id:
                context["X-Parent-Correlation-Id"] = str(parent_id)

            service_name = FlextContext.Variables.Service.SERVICE_NAME.get()
            if service_name:
                context["X-Service-Name"] = str(service_name)

            return context

        @staticmethod
        def set_from_context(context: Mapping[str, object]) -> None:
            """Set context from dictionary (e.g., from HTTP headers)."""
            correlation_id = context.get("X-Correlation-Id") or context.get(
                "correlation_id",
            )
            if correlation_id and isinstance(correlation_id, str):
                FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

            parent_id = context.get("X-Parent-Correlation-Id") or context.get(
                "parent_correlation_id",
            )
            if parent_id and isinstance(parent_id, str):
                FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

            service_name = context.get("X-Service-Name") or context.get("service_name")
            if service_name and isinstance(service_name, str):
                FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

            user_id = context.get("X-User-Id") or context.get("user_id")
            if user_id and isinstance(user_id, str):
                FlextContext.Variables.Request.USER_ID.set(user_id)

    # =========================================================================
    # Utilities Domain - Context utility methods and helpers
    # =========================================================================

    class Utilities:
        """Utility methods for context management and helper operations."""

        @staticmethod
        def clear_context() -> None:
            """Clear all context variables."""
            # Clear string context variables
            for context_var in [
                FlextContext.Variables.Correlation.CORRELATION_ID,
                FlextContext.Variables.Correlation.PARENT_CORRELATION_ID,
                FlextContext.Variables.Service.SERVICE_NAME,
                FlextContext.Variables.Service.SERVICE_VERSION,
                FlextContext.Variables.Request.USER_ID,
                FlextContext.Variables.Request.REQUEST_ID,
                FlextContext.Variables.Performance.OPERATION_NAME,
            ]:
                with contextlib.suppress(LookupError):
                    context_var.set(None)

            # Clear typed context variables
            with contextlib.suppress(LookupError):
                FlextContext.Variables.Performance.OPERATION_START_TIME.set(None)
            with contextlib.suppress(LookupError):
                FlextContext.Variables.Performance.OPERATION_METADATA.set(None)
            with contextlib.suppress(LookupError):
                FlextContext.Variables.Request.REQUEST_TIMESTAMP.set(None)

            # Also clear structlog context if available
            if structlog is not None:
                structlog.contextvars.clear_contextvars()

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure correlation ID exists, creating one if needed."""
            correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if not correlation_id:
                correlation_id = FlextContext.Correlation.generate_correlation_id()
            return correlation_id

        @staticmethod
        def has_correlation_id() -> bool:
            """Check if correlation ID is set in context."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get() is not None

        @staticmethod
        def get_context_summary() -> str:
            """Get a human-readable context summary for debugging."""
            context = FlextContext.Serialization.get_full_context()
            parts: FlextTypes.StringList = []

            correlation_id = context["correlation_id"]
            if isinstance(correlation_id, str) and correlation_id:
                parts.append(f"correlation={correlation_id[:8]}...")

            service_name = context["service_name"]
            if isinstance(service_name, str) and service_name:
                parts.append(f"service={service_name}")

            operation_name = context["operation_name"]
            if isinstance(operation_name, str) and operation_name:
                parts.append(f"operation={operation_name}")

            user_id = context["user_id"]
            if isinstance(user_id, str) and user_id:
                parts.append(f"user={user_id}")

            return (
                f"FlextContext({', '.join(parts)})" if parts else "FlextContext(empty)"
            )

    class HandlerExecutionContext:
        """Handler execution context for FlextHandlers complexity reduction.

        Extracts execution context management from FlextHandlers to simplify
        handler execution and provide reusable context management patterns.
        """

        @override
        def __init__(self, handler_name: str, handler_mode: str) -> None:
            """Initialize handler execution context.

            Args:
                handler_name: Name of the handler
                handler_mode: Mode of the handler (command/query)

            """
            super().__init__()
            self.handler_name = handler_name
            self.handler_mode = handler_mode
            self._start_time: float | None = None
            self._metrics_state: FlextTypes.Dict | None = None

        def start_execution(self) -> None:
            """Start execution timing."""
            self._start_time = time.time()

        def get_execution_time_ms(self) -> float:
            """Get execution time in milliseconds.

            Returns:
                Execution time in milliseconds, or 0 if not started

            """
            if self._start_time is None:
                return 0.0

            elapsed = time.time() - self._start_time
            return round(elapsed * 1000, 2)

        def get_metrics_state(self) -> FlextTypes.Dict:
            """Get current metrics state.

            Returns:
                Dictionary containing metrics state

            """
            if self._metrics_state is None:
                self._metrics_state = {}
            return self._metrics_state

        def set_metrics_state(self, state: FlextTypes.Dict) -> None:
            """Set metrics state.

            Args:
                state: Metrics state to set

            """
            self._metrics_state = state

        def reset(self) -> None:
            """Reset execution context."""
            self._start_time = None
            self._metrics_state = None

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: str,
        ) -> FlextContext.HandlerExecutionContext:
            """Create execution context for a handler.

            Args:
                handler_name: Name of the handler
                handler_mode: Mode of the handler (command/query)

            Returns:
                New HandlerExecutionContext instance

            """
            return cls(handler_name, handler_mode)

    # =========================================================================
    # Enhanced flext-core Integration Methods
    # =========================================================================


__all__: FlextTypes.StringList = [
    "FlextContext",
]
