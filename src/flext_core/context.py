"""Hierarchical context management for distributed tracing and correlation.

This module provides FlextContext, a comprehensive context management system
for request metadata, correlation IDs, performance tracking, and distributed
tracing across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import collections.abc
import contextvars
import json
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import (
    Final,
    Self,
    cast,
)

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextContext:
    """Hierarchical context management for distributed tracing and correlation.

    Provides comprehensive context management for request metadata, correlation
    IDs, performance tracking, and distributed tracing across the FLEXT ecosystem.

    Features:
    - Hierarchical context scopes (global, request, session, transaction)
    - Correlation ID management for distributed tracing
    - Service identification context (name, version)
    - Request context with user and operation metadata
    - Performance tracking with timing operations
    - Context serialization for cross-service propagation
    - Thread-safe context variable management
    - Context cloning and merging capabilities
    - Automatic integration with FlextLogger

    Usage:
        >>> from flext_core import FlextContext
        >>>
        >>> context = FlextContext()
        >>> context.set("user_id", "123")
        >>> user_id = context.get("user_id")
        >>>
        >>> # Correlation ID management
        >>> with FlextContext.Correlation.new_correlation() as corr_id:
        ...     print(f"Processing request {corr_id}")
    """

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def __init__(
        self,
        initial_data: FlextModels.ContextData | FlextTypes.Dict | None = None,
    ) -> None:
        """Initialize FlextContext with optional initial data.

        ARCHITECTURAL NOTE: FlextContext now uses Python's contextvars for storage,
        completely independent of structlog. It delegates to FlextLogger for logging
        integration, maintaining clear separation of concerns.

        Args:
            initial_data: Optional context data (dict or `FlextModels.ContextData`)

        """
        super().__init__()
        if initial_data is None:
            context_data = FlextModels.ContextData()
        elif isinstance(initial_data, dict):
            context_data = FlextModels.ContextData(data=initial_data)
        else:
            context_data = initial_data

        # Initialize metadata and hooks (not stored in contextvars)
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

        # Create instance-specific contextvars for isolation (required for clone())
        # Note: Each instance gets its own contextvars to prevent state sharing
        # Note: Using None default per B039 - mutable defaults cause issues
        self._scope_vars: dict[str, contextvars.ContextVar[FlextTypes.Dict | None]] = {
            FlextConstants.Context.SCOPE_GLOBAL: contextvars.ContextVar(
                "flext_global_context",
                default=None,
            ),
            "user": contextvars.ContextVar(
                "flext_user_context",
                default=None,
            ),
            FlextConstants.Context.SCOPE_SESSION: contextvars.ContextVar(
                "flext_session_context",
                default=None,
            ),
        }

        # Initialize contextvars with initial data if provided
        # Note: No self._lock needed - contextvars are thread-safe by design
        if context_data.data:
            # Set initial data in global context
            self._set_in_contextvar(
                FlextConstants.Context.SCOPE_GLOBAL, context_data.data
            )

    # =========================================================================
    # PRIVATE HELPERS - Context variable management and FlextLogger delegation
    # =========================================================================

    def _get_or_create_scope_var(
        self, scope: str
    ) -> contextvars.ContextVar[FlextTypes.Dict | None]:
        """Get or create contextvar for scope.

        Args:
            scope: Scope name (global, user, session, or custom)

        Returns:
            ContextVar for the scope

        """
        if scope not in self._scope_vars:
            # Create new contextvar for dynamic scope
            # Note: Using None default per B039 - mutable defaults cause issues
            self._scope_vars[scope] = contextvars.ContextVar(
                f"flext_{scope}_context", default=None
            )
        return self._scope_vars[scope]

    def _set_in_contextvar(self, scope: str, data: FlextTypes.Dict) -> None:
        """Set multiple values in contextvar scope.

        Args:
            scope: Scope name
            data: Dictionary of key-value pairs to set

        """
        ctx_var = self._get_or_create_scope_var(scope)
        current = ctx_var.get() or {}  # Handle None default
        updated = {**current, **data}
        ctx_var.set(updated)

        # DELEGATION: Propagate global scope to FlextLogger for logging integration
        if scope == FlextConstants.Context.SCOPE_GLOBAL:
            FlextLogger.bind_global_context(**data)

    def _get_from_contextvar(self, scope: str) -> FlextTypes.Dict:
        """Get all values from contextvar scope.

        Args:
            scope: Scope name

        Returns:
            Dictionary of all key-value pairs in scope

        """
        ctx_var = self._get_or_create_scope_var(scope)
        return ctx_var.get() or {}  # Handle None default

    # =========================================================================
    # Instance Methods - Core context operations
    # =========================================================================

    def set(
        self,
        key: str,
        value: object,
        scope: str = FlextConstants.Context.SCOPE_GLOBAL,
    ) -> None:
        """Set a value in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            key: The key to set
            value: The value to set
            scope: The scope for the value (global, user, session)

        """
        if not self._active:
            return

        # Validate key is not None or empty
        if not key:
            msg = "Key must be a non-empty string"
            raise ValueError(msg)

        # Validate value is serializable
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            msg = "Value must be serializable"
            raise TypeError(msg)

        # Set in contextvar (thread-safe by design, no lock needed)
        ctx_var = self._get_or_create_scope_var(scope)
        current = ctx_var.get() or {}  # Handle None default
        updated = {**current, key: value}
        ctx_var.set(updated)

        # DELEGATION: Propagate global scope to FlextLogger for logging integration
        if scope == FlextConstants.Context.SCOPE_GLOBAL:
            FlextLogger.bind_global_context(**{key: value})

        # Update statistics
        sets_count = self._statistics.get("sets", 0)
        if isinstance(sets_count, int):
            self._statistics["sets"] = sets_count + 1

        operations = cast("dict[str, int]", self._statistics.get("operations", {}))
        if "set" in operations:
            set_count: int = operations["set"]
            operations["set"] = set_count + 1

        # Execute hooks
        # Note: Hooks should be designed to not raise exceptions.
        # If a hook raises an exception, it indicates a programming error
        # in the hook implementation that should be fixed by the caller.
        if "set" in self._hooks:
            for hook in self._hooks["set"]:
                hook(key, value)

    def get(
        self,
        key: str,
        default: object | None = None,
        scope: str = FlextConstants.Context.SCOPE_GLOBAL,
    ) -> object:
        """Get a value from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).
        No longer checks structlog - FlextLogger is independent.

        Args:
            key: The key to get
            default: Default value if key not found
            scope: The scope to get from (global, user, session)

        Returns:
            The value or default

        """
        if not self._active:
            return default

        # Get from contextvar (single source of truth)
        scope_data = self._get_from_contextvar(scope)
        value = scope_data.get(key, default)

        # Update statistics
        gets_count = self._statistics.get("gets", 0)
        if isinstance(gets_count, int):
            self._statistics["gets"] = gets_count + 1

        operations = cast("dict[str, int]", self._statistics.get("operations", {}))
        if "get" in operations:
            get_count: int = operations["get"]
            operations["get"] = get_count + 1

        return value

    def has(self, key: str, scope: str = FlextConstants.Context.SCOPE_GLOBAL) -> bool:
        """Check if a key exists in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Args:
            key: The key to check
            scope: The scope to check (global, user, session)

        Returns:
            True if key exists, False otherwise

        """
        if not self._active:
            return False

        # Check in contextvar (single source of truth)
        scope_data = self._get_from_contextvar(scope)
        return key in scope_data

    def remove(
        self, key: str, scope: str = FlextConstants.Context.SCOPE_GLOBAL
    ) -> None:
        """Remove a key from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            key: The key to remove
            scope: The scope to remove from (global, user, session)

        """
        if not self._active:
            return

        # Remove from contextvar
        ctx_var = self._get_or_create_scope_var(scope)
        current = ctx_var.get() or {}  # Handle None default
        if key in current:
            updated = {k: v for k, v in current.items() if k != key}
            ctx_var.set(updated)

            # DELEGATION: Propagate removal to FlextLogger for global scope
            if scope == FlextConstants.Context.SCOPE_GLOBAL:
                FlextLogger.unbind_global_context(key)

            # Update statistics
            removes_count = self._statistics.get("removes", 0)
            if isinstance(removes_count, int):
                self._statistics["removes"] = removes_count + 1

            operations = cast("dict[str, int]", self._statistics.get("operations", {}))
            remove_count: int = (
                operations.get("remove", 0) if "remove" in operations else 0
            )
            operations["remove"] = remove_count + 1

    def clear(self) -> None:
        """Clear all data from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        """
        if not self._active:
            return

        # Clear all contextvar scopes
        for scope_name, ctx_var in self._scope_vars.items():
            ctx_var.set({})  # Set to empty dict, not None

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == FlextConstants.Context.SCOPE_GLOBAL:
                FlextLogger.clear_global_context()

        # Update statistics
        clears_count = self._statistics.get("clears", 0)
        if isinstance(clears_count, int):
            self._statistics["clears"] = clears_count + 1

        operations = cast("dict[str, int]", self._statistics.get("operations", {}))
        if "clear" in operations:
            clear_count: int = operations["clear"]
            operations["clear"] = clear_count + 1

    def keys(self) -> FlextTypes.StringList:
        """Get all keys in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of all keys across all scopes

        """
        if not self._active:
            return []

        # Get keys from all contextvar scopes
        all_keys: set[str] = set()
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            all_keys.update(scope_data.keys())

        return list(all_keys)

    def values(self) -> FlextTypes.List:
        """Get all values in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of all values across all scopes

        """
        if not self._active:
            return []

        # Get values from all contextvar scopes
        all_values: FlextTypes.List = []
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            all_values.extend(scope_data.values())

        return all_values

    def items(self) -> list[tuple[str, object]]:
        """Get all key-value pairs in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of (key, value) tuples across all scopes

        """
        if not self._active:
            return []

        # Get items from all contextvar scopes
        all_items: list[tuple[str, object]] = []
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            all_items.extend(scope_data.items())

        return all_items

    def merge(self, other: FlextContext | FlextTypes.Dict) -> Self:
        """Merge another context or dictionary into this context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            other: Another FlextContext or dictionary to merge

        Returns:
            Self for chaining

        """
        if not self._active:
            return self

        if isinstance(other, FlextContext):
            # Merge all scopes from the other context
            other_scopes = other.get_all_scopes()
            for scope_name, scope_data in other_scopes.items():
                # Merge into contextvar
                ctx_var = self._get_or_create_scope_var(scope_name)
                current = ctx_var.get() or {}  # Handle None default
                updated = {**current, **scope_data}
                ctx_var.set(updated)

                # DELEGATION: Propagate global scope to FlextLogger
                if scope_name == FlextConstants.Context.SCOPE_GLOBAL:
                    FlextLogger.bind_global_context(**scope_data)
        else:
            # Merge dictionary into global scope
            self._set_in_contextvar(FlextConstants.Context.SCOPE_GLOBAL, other)

        return self

    def clone(self) -> Self:
        """Create a clone of this context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            A new FlextContext with the same data

        """
        cloned = FlextContext()

        # Clone all contextvar scopes from this instance
        for scope_name, ctx_var in self._scope_vars.items():
            scope_data = ctx_var.get() or {}  # Handle None default
            if scope_data:
                cloned_ctx_var = cloned._get_or_create_scope_var(scope_name)
                cloned_ctx_var.set(scope_data.copy())

        # Clone metadata and statistics
        cloned._metadata = self._metadata.copy()
        cloned._statistics = self._statistics.copy()

        return cast("Self", cloned)

    def get_all_scopes(self) -> FlextTypes.Context.ScopeRegistry:
        """Get all scopes.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary of all scopes with their data

        """
        scopes: FlextTypes.Context.ScopeRegistry = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_data = ctx_var.get() or {}  # Handle None default
            if scope_data:  # Only include non-empty scopes
                scopes[scope_name] = scope_data.copy()
        return scopes

    def validate(self) -> FlextResult[None]:
        """Validate the context data.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            FlextResult indicating validation success or failure

        """
        if not self._active:
            return FlextResult[None].fail("Context is not active")

        # Check for empty keys in all contextvar scopes
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            for key in scope_data:
                if not key:
                    return FlextResult[None].fail("Invalid key found in context")
        return FlextResult[None].ok(None)

    def to_json(self) -> str:
        """Convert context to JSON string.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            JSON string representation of the context

        """
        # Combine all contextvar scopes into a single flat dictionary for backward compatibility
        all_data: FlextTypes.Dict = {}
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            all_data.update(scope_data)
        return json.dumps(all_data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> FlextContext:
        """Create context from JSON string.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            json_str: JSON string to parse

        Returns:
            New FlextContext instance

        """
        data = json.loads(json_str)
        context = cls()

        # Handle both old flat format and new scoped format
        if isinstance(data, dict):
            data_values = cast("FlextTypes.Dict", data).values()
            if all(isinstance(v, dict) for v in data_values):
                # New scoped format - restore all scopes
                for scope_name, scope_data in cast(
                    "dict[str, FlextTypes.Dict]", data
                ).items():
                    context._set_in_contextvar(scope_name, scope_data)
            else:
                # Old flat format - put everything in global scope
                context._set_in_contextvar(FlextConstants.Context.SCOPE_GLOBAL, data)

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
        """Destroy the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        """
        self._active = False

        # Clear all contextvar scopes
        for scope_name, ctx_var in self._scope_vars.items():
            ctx_var.set({})  # Set to empty dict, not None

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == FlextConstants.Context.SCOPE_GLOBAL:
                FlextLogger.clear_global_context()

        # Clear metadata and hooks
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
        self._metadata[key] = value

    def get_metadata(self, key: str, default: object | None = None) -> object:
        """Get metadata from the context.

        Args:
            key: The metadata key
            default: Default value if key not found

        Returns:
            The metadata value or default

        """
        return self._metadata.get(key, default)

    def get_all_metadata(self) -> FlextTypes.Dict:
        """Get all metadata from the context.

        Returns:
            Dictionary of all metadata

        """
        return self._metadata.copy()

    def get_all_data(self) -> FlextTypes.Dict:
        """Get all data from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary of all context data across all scopes

        """
        # Combine all contextvar scopes
        all_data: FlextTypes.Dict = {}
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            all_data.update(scope_data)
        return all_data

    def get_statistics(self) -> FlextTypes.Dict:
        """Get context statistics.

        Returns:
            Dictionary of context statistics

        """
        return self._statistics.copy()

    def cleanup(self) -> None:
        """Clean up the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        """
        # Clear all contextvar scopes
        for scope_name, ctx_var in self._scope_vars.items():
            ctx_var.set({})  # Set to empty dict, not None

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == FlextConstants.Context.SCOPE_GLOBAL:
                FlextLogger.clear_global_context()

        # Clear metadata
        self._metadata.clear()

    def export(self) -> FlextTypes.Dict:
        """Export context data as a dictionary for compatibility consumers.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        """
        export_snapshot = self.export_snapshot()
        return dict[str, object](export_snapshot.data)

    def export_snapshot(self) -> FlextModels.ContextExport:
        """Return typed export snapshot including metadata and statistics.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        """
        # Combine all contextvar scopes
        all_data: FlextTypes.Dict = {}
        for ctx_var in self._scope_vars.values():
            scope_data = ctx_var.get() or {}  # Handle None default
            all_data.update(scope_data)

        return FlextModels.ContextExport(
            data=all_data,
            metadata=self._metadata.copy(),
            statistics=self._statistics.copy(),
        )

    def import_data(self, data: FlextTypes.Dict) -> None:
        """Import context data.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            data: Dictionary containing context data

        """
        # Import data into global scope using FlextConstants.Context
        self._set_in_contextvar(FlextConstants.Context.SCOPE_GLOBAL, data)

    # =========================================================================
    # Container integration for dependency injection
    # =========================================================================

    _container: FlextContainer | None = None

    @classmethod
    def get_container(cls) -> FlextContainer:
        """Get global container with lazy initialization.

        Returns:
            Global FlextContainer instance for dependency injection

        Example:
            >>> container = FlextContext.get_container()
            >>> container.register("my_service", MyService())
            >>> service_result = container.get("my_service")

        """
        if cls._container is None:
            cls._container = FlextContainer.get_global()
        return cls._container

    # ==========================================================================
    # Variables - Context Variables using structlog as Single Source of Truth
    # ==========================================================================

    # Use FlextModels.StructlogProxyContextVar (defined in models.py)
    StructlogProxyContextVar = FlextModels.StructlogProxyContextVar

    class Variables:
        """Context variables using structlog as single source of truth."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str](
                    "correlation_id", default=None
                )
            )
            PARENT_CORRELATION_ID: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str](
                    "parent_correlation_id", default=None
                )
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str]("service_name", default=None)
            )
            SERVICE_VERSION: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str](
                    "service_version", default=None
                )
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str]("user_id", default=None)
            )
            REQUEST_ID: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str]("request_id", default=None)
            )
            REQUEST_TIMESTAMP: Final[FlextModels.StructlogProxyContextVar[datetime]] = (
                FlextModels.StructlogProxyContextVar[datetime](
                    "request_timestamp", default=None
                )
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[FlextModels.StructlogProxyContextVar[str]] = (
                FlextModels.StructlogProxyContextVar[str](
                    "operation_name", default=None
                )
            )
            OPERATION_START_TIME: Final[
                FlextModels.StructlogProxyContextVar[datetime]
            ] = FlextModels.StructlogProxyContextVar[datetime](
                "operation_start_time", default=None
            )
            OPERATION_METADATA: Final[
                FlextModels.StructlogProxyContextVar[FlextTypes.Dict]
            ] = FlextModels.StructlogProxyContextVar[FlextTypes.Dict](
                "operation_metadata", default=None
            )

    # =========================================================================
    # Correlation Domain - Distributed tracing and correlation ID management
    # =========================================================================

    class Correlation:
        """Distributed tracing and correlation ID management utilities."""

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get()

        @staticmethod
        def set_correlation_id(correlation_id: str) -> None:
            """Set correlation ID.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            """
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate unique correlation ID.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            Uses FlextConstants.Context configuration for prefix and length.
            """
            # Generate correlation ID using centralized constants
            random_suffix = str(uuid.uuid4()).replace("-", "")[
                : FlextConstants.Context.CORRELATION_ID_LENGTH
            ]
            correlation_id = (
                f"{FlextConstants.Context.CORRELATION_ID_PREFIX}{random_suffix}"
            )
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
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
            """Create correlation context scope.

            Uses FlextConstants.Context configuration for correlation ID generation.
            """
            # Generate correlation ID if not provided
            if correlation_id is None:
                random_suffix = str(uuid.uuid4()).replace("-", "")[
                    : FlextConstants.Context.CORRELATION_ID_LENGTH
                ]
                correlation_id = (
                    f"{FlextConstants.Context.CORRELATION_ID_PREFIX}{random_suffix}"
                )

            # Save current context
            current_correlation = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
            )

            # Set new context
            correlation_token = FlextContext.Variables.Correlation.CORRELATION_ID.set(
                correlation_id,
            )

            # Set parent context
            parent_token = None
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
        """Service identification and lifecycle context management utilities."""

        @staticmethod
        def get_service_name() -> str | None:
            """Get current service name."""
            return FlextContext.Variables.Service.SERVICE_NAME.get()

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set service name.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            """
            FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get current service version."""
            return FlextContext.Variables.Service.SERVICE_VERSION.get()

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            """
            FlextContext.Variables.Service.SERVICE_VERSION.set(version)

        @staticmethod
        def get_service(service_name: str) -> FlextResult[object]:
            """Resolve service from global container using FlextResult.

            Provides unified service resolution pattern across the ecosystem
            by integrating FlextContainer with FlextContext.

            Args:
                service_name: Name of the service to retrieve

            Returns:
                FlextResult containing the service instance or error

            Example:
                >>> result = FlextContext.Service.get_service("logger")
                >>> if result.is_success:
                ...     logger = result.unwrap()
                ...     logger.info("Service retrieved")

            """
            container = FlextContext.get_container()
            return container.get(service_name)

        @staticmethod
        def register_service(
            service_name: str,
            service: object,
        ) -> FlextResult[None]:
            """Register service in global container using FlextResult.

            Provides unified service registration pattern across the ecosystem
            by integrating FlextContainer with FlextContext.

            Args:
                service_name: Name to register the service under
                service: Service instance to register

            Returns:
                FlextResult indicating registration success or failure

            Example:
                >>> result = FlextContext.Service.register_service(
                ...     "logger",
                ...     FlextLogger(__name__),
                ... )
                >>> if result.is_failure:
                ...     print(f"Registration failed: {result.error}")

            """
            container = FlextContext.get_container()
            return container.register(service_name, service)

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
        """Request-level context management for user and operation metadata utilities."""

        @staticmethod
        def get_user_id() -> str | None:
            """Get current user ID."""
            return FlextContext.Variables.Request.USER_ID.get()

        @staticmethod
        def set_user_id(user_id: str) -> None:
            """Set user ID in context.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            """
            FlextContext.Variables.Request.USER_ID.set(user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context."""
            return FlextContext.Variables.Performance.OPERATION_NAME.get()

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            """
            FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context."""
            return FlextContext.Variables.Request.REQUEST_ID.get()

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context.

            Note: Uses structlog as single source of truth (via StructlogProxyContextVar).
            """
            FlextContext.Variables.Request.REQUEST_ID.set(request_id)

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
        """Performance monitoring and timing context management utilities."""

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
        """Context serialization and deserialization utilities."""

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
        """Context management utility methods."""

        @staticmethod
        def clear_context() -> None:
            """Clear all context variables.

            Note: ContextVar.set() does not raise LookupError - it always succeeds.
            Setting to None effectively clears the variable.
            """
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
                context_var.set(None)

            # Clear typed context variables
            FlextContext.Variables.Performance.OPERATION_START_TIME.set(None)
            FlextContext.Variables.Performance.OPERATION_METADATA.set(None)
            FlextContext.Variables.Request.REQUEST_TIMESTAMP.set(None)

            # Note: All variables use structlog as single source (via StructlogProxyContextVar)

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


__all__: FlextTypes.StringList = [
    "FlextContext",
]
