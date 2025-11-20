"""Hierarchical context management for distributed tracing and correlation.

This module provides FlextContext, a comprehensive context management system
for request metadata, correlation IDs, performance tracking, and distributed
tracing across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Final,
    Self,
    TypeAlias,
)

from flext_core._models.context import FlextModelsContext
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# Type aliases for MyPy compatibility
if TYPE_CHECKING:
    _StructlogProxyStr: TypeAlias = FlextModelsContext.StructlogProxyContextVar[str]
    _StructlogProxyDatetime: TypeAlias = FlextModelsContext.StructlogProxyContextVar[datetime]
    _StructlogProxyDict: TypeAlias = FlextModelsContext.StructlogProxyContextVar[dict[str, object]]
else:
    _StructlogProxyStr = FlextModels.StructlogProxyContextVar[str]
    _StructlogProxyDatetime = FlextModels.StructlogProxyContextVar[datetime]
    _StructlogProxyDict = FlextModels.StructlogProxyContextVar[dict[str, object]]


def _create_str_proxy(key: str, default: str | None = None) -> _StructlogProxyStr:
    """Helper to create StructlogProxyContextVar[str] instances."""
    return FlextModelsContext.StructlogProxyContextVar[str](key, default=default)


def _create_datetime_proxy(
    key: str,
    default: datetime | None = None,
) -> _StructlogProxyDatetime:
    """Helper to create StructlogProxyContextVar[datetime] instances."""
    return FlextModelsContext.StructlogProxyContextVar[datetime](
        key,
        default=default,
    )


def _create_dict_proxy(
    key: str,
    default: dict[str, object] | None = None,
) -> _StructlogProxyDict:
    """Helper to create StructlogProxyContextVar[dict[str, object]] instances."""
    return FlextModelsContext.StructlogProxyContextVar[dict[str, object]](
        key,
        default=default,
    )


class FlextContext:
    """Hierarchical context management for distributed tracing and correlation.

    Implements comprehensive context management for request metadata, correlation
    IDs, performance tracking, and distributed tracing across the FLEXT ecosystem.
    Uses Python's contextvars for thread-safe context variable storage with automatic
    FlextLogger integration for structured logging.

    Architecture:
        - Single-instance context per operation (scope-based isolation)
        - ContextVar storage for thread-safe access across async/threading boundaries
        - Hierarchical scopes (global, user, session) with scope-based access
        - Delegation to FlextLogger for structured logging integration
        - Container integration via FlextContext.Service for DI pattern
        - FlextModels.StructlogProxyContextVar for direct logging system integration

    Core Features:
        - Hierarchical context scopes (global, request, session, transaction)
        - Correlation ID management for distributed tracing (automatic generation)
        - Service identification context (name, version)
        - Request context with user and operation metadata
        - Performance tracking with timing operations (start/end, duration)
        - Context serialization for cross-service propagation (HTTP headers)
        - Thread-safe context variable management via Python contextvars
        - Context cloning and merging capabilities
        - Automatic integration with FlextLogger for logging
        - Extensible hook system for context events
        - Statistics tracking for context operations

    Nested Domains:
        - FlextContext.Variables - Context variable definitions
          (Correlation, Service, Request, Performance)
        - FlextContext.Correlation - Distributed tracing and correlation ID management
        - FlextContext.Service - Service identification and lifecycle context
        - FlextContext.Request - User and request metadata management
        - FlextContext.Performance - Operation timing and performance tracking
        - FlextContext.Serialization - Context serialization for
          cross-service communication
        - FlextContext.Utilities - Helper methods and utility operations

    Integration Patterns:
        - Container Integration - FlextContext.Service.get_service() /
          register_service()
        - Logger Delegation - Automatic FlextLogger.bind_global_context() on
          context changes
        - Cross-Service Propagation - HTTP header format
          (X-Correlation-Id, X-Service-Name)
        - FlextResult Integration - All service operations return FlextResult[T]

    Usage Example:
        >>> from flext_core import FlextContext, FlextResult
        >>>
        >>> # Create context instance
        >>> context = FlextContext()
        >>> context.set("user_id", "123")
        >>> user_id = context.get("user_id")
        >>>
        >>> # Correlation ID management for distributed tracing
        >>> with FlextContext.Correlation.new_correlation() as corr_id:
        ...     print(f"Processing request {corr_id}")
        >>>
        >>> # Service context management
        >>> with FlextContext.Service.service_context("my_service", "v1.0"):
        ...     result = FlextContext.Service.get_service("logger")
        ...     if result.is_success:
        ...         logger = result.unwrap()
        >>>
        >>> # Request context with metadata
        >>> with FlextContext.Request.request_context(
        ...     user_id="user123",
        ...     operation_name="process_payment",
        ... ):
        ...     # All logging will include context automatically
        ...     pass
        >>>
        >>> # Performance tracking with timing
        >>> with FlextContext.Performance.timed_operation("data_processing"):
        ...     # Operation metrics automatically tracked
        ...     process_data()
        >>>
        >>> # Cross-service context propagation
        >>> headers = FlextContext.Serialization.get_correlation_context()
        >>> # headers: {"X-Correlation-Id": "...", "X-Service-Name": "..."}

    Context Variable Compliance:
        - Uses Python contextvars for inherent thread-safety and async support
        - FlextModels.StructlogProxyContextVar wraps contextvars with
          structlog integration
        - No explicit protocol inheritance needed - implements context management patterns
        - Automatic FlextLogger delegation for logging consistency across ecosystem
    """

    # Instance attributes
    _metadata: FlextModels.ContextMetadata

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def __init__(
        self,
        initial_data: FlextModels.ContextData | dict[str, object] | None = None,
    ) -> None:
        """Initialize FlextContext with optional initial data.

        ARCHITECTURAL NOTE: FlextContext now uses Python's contextvars for storage,
        completely independent of structlog. It delegates to FlextLogger for logging
        integration, maintaining clear separation of concerns.

        Args:
            initial_data: Optional `FlextModels.ContextData` instance or dict

        """
        super().__init__()
        # Use Pydantic directly - NO redundant helpers (Pydantic validates dict/None/model)
        # Type narrowing: always create FlextModels.ContextData instance
        if FlextRuntime.is_dict_like(initial_data):
            context_data: FlextModels.ContextData = FlextModels.ContextData(
                data=initial_data
            )
        elif initial_data is not None and isinstance(
            initial_data, FlextModels.ContextData
        ):
            # Already a ContextData instance - explicit type check for MyPy
            context_data = initial_data
        else:
            # None or uninitialized - create empty ContextData
            context_data = FlextModels.ContextData()
        # Initialize context-specific metadata (separate from ContextData.metadata)
        # ContextData.metadata = generic creation/modification metadata (Metadata)
        # FlextContext._metadata = context-specific tracing metadata (ContextMetadata)
        self._metadata = FlextModels.ContextMetadata()

        self._hooks: FlextTypes.HookRegistry = {}
        self._statistics: FlextModels.ContextStatistics = (
            FlextModels.ContextStatistics()
        )
        self._active = True
        self._suspended = False

        # Create instance-specific contextvars for isolation (required for clone())
        # Note: Each instance gets its own contextvars to prevent state sharing
        # Note: Using None default per B039 - mutable defaults cause issues
        self._scope_vars: dict[
            str,
            contextvars.ContextVar[dict[str, object] | None],
        ] = {
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
        # Type narrowing: context_data is always FlextModels.ContextData at this point
        if context_data.data:
            # Set initial data in global context
            self._set_in_contextvar(
                FlextConstants.Context.SCOPE_GLOBAL,
                context_data.data,
            )

    # =========================================================================
    # PRIVATE HELPERS - Context variable management and FlextLogger delegation
    # =========================================================================

    def _get_or_create_scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[dict[str, object] | None]:
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
                f"flext_{scope}_context",
                default=None,
            )
        return self._scope_vars[scope]

    def _set_in_contextvar(self, scope: str, data: dict[str, object]) -> None:
        """Set multiple values in contextvar scope.

        Args:
            scope: Scope name
            data: Dictionary of key-value pairs to set

        """
        ctx_var = self._get_or_create_scope_var(scope)
        current_value = ctx_var.get()
        # Fast fail: contextvar must contain dict or None (uninitialized)
        if current_value is not None and not FlextRuntime.is_dict_like(current_value):
            msg = (
                f"Invalid contextvar value type in scope '{scope}': "
                f"{type(current_value).__name__}. Expected dict[str, object] | None"
            )
            raise TypeError(msg)
        # Initialize with empty dict if None (first use)
        current: dict[str, object] = (
            current_value if FlextRuntime.is_dict_like(current_value) else {}
        )
        updated = {**current, **data}
        ctx_var.set(updated)

        # DELEGATION: Propagate global scope to FlextLogger for logging integration
        if scope == FlextConstants.Context.SCOPE_GLOBAL:
            FlextLogger.bind_global_context(**data)

    def _get_from_contextvar(self, scope: str) -> dict[str, object]:
        """Get all values from contextvar scope.

        Args:
            scope: Scope name

        Returns:
            Dictionary of all key-value pairs in scope

        """
        ctx_var = self._get_or_create_scope_var(scope)
        value = ctx_var.get()
        # Fast fail: contextvar must contain dict or None (uninitialized)
        if value is not None and not FlextRuntime.is_dict_like(value):
            msg = (
                f"Invalid contextvar value type in scope '{scope}': "
                f"{type(value).__name__}. Expected dict[str, object] | None"
            )
            raise TypeError(msg)
        # Return empty dict if None (uninitialized scope)
        return value if FlextRuntime.is_dict_like(value) else {}

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper).

        Args:
            operation: Operation name ('set', 'get', 'remove', etc.)

        """
        # Update primary counter using attribute name
        counter_attr = f"{operation}s"
        if hasattr(self._statistics, counter_attr):
            current_value = getattr(self._statistics, counter_attr, 0)
            if isinstance(current_value, int):
                setattr(self._statistics, counter_attr, current_value + 1)

        # Update operations dict if exists
        if (
            self._statistics.operations is not None
            and operation in self._statistics.operations
        ):
            value = self._statistics.operations[operation]
            if isinstance(value, int):
                self._statistics.operations[operation] = value + 1

    def _execute_hooks(self, event: str, event_data: dict[str, object]) -> None:
        """Execute hooks for an event (DRY helper).

        Args:
            event: Event name ('set', 'get', 'remove', etc.)
            event_data: Data to pass to hooks

        """
        if event not in self._hooks:
            return

        hooks = self._hooks[event]
        if not FlextRuntime.is_list_like(hooks):
            return

        for hook in hooks:
            if callable(hook):
                # Note: Hooks should not raise exceptions
                # Any exception indicates a programming error in hook implementation
                hook(event_data)

    def _propagate_to_logger(self, key: str, value: object, scope: str) -> None:
        """Propagate context changes to FlextLogger (DRY helper).

        Args:
            key: Context key
            value: Context value
            scope: Context scope

        """
        if scope == FlextConstants.Context.SCOPE_GLOBAL:
            FlextLogger.bind_global_context(**{key: value})

    # =========================================================================
    # Instance Methods - Core context operations
    # =========================================================================

    def set(
        self,
        key: str,
        value: object,
        scope: str = FlextConstants.Context.SCOPE_GLOBAL,
    ) -> FlextResult[bool]:
        """Set a value in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            key: The key to set
            value: The value to set
            scope: The scope for the value (global, user, session)

        Returns:
            FlextResult[bool]: Success with True if set, failure with error message

        """
        if not self._active:
            return FlextResult[bool].fail("Context is not active")

        # Validate key is not None or empty
        if not key:
            return FlextResult[bool].fail("Key must be a non-empty string")

        # Validate value is serializable
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            return FlextResult[bool].fail("Value must be serializable")

        try:
            # Set in contextvar (thread-safe by design, no lock needed)
            ctx_var = self._get_or_create_scope_var(scope)
            current_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if current_value is not None and not FlextRuntime.is_dict_like(
                current_value
            ):
                msg = (
                    f"Invalid contextvar value type in scope '{scope}': "
                    f"{type(current_value).__name__}. Expected dict[str, object] | None"
                )
                return FlextResult[bool].fail(msg)
            # Initialize with empty dict if None (first use)
            current: dict[str, object] = (
                current_value if FlextRuntime.is_dict_like(current_value) else {}
            )
            updated = {**current, key: value}
            ctx_var.set(updated)

            # DELEGATION: Propagate to logger, update stats, execute hooks
            self._propagate_to_logger(key, value, scope)
            self._update_statistics("set")
            self._execute_hooks("set", {"key": key, "value": value})

            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(f"Failed to set context value: {e}")

    def get(
        self,
        key: str,
        scope: str = FlextConstants.Context.SCOPE_GLOBAL,
    ) -> FlextResult[object]:
        """Get a value from the context.

        Fast fail: Returns FlextResult[object] - fails if key not found.
        No fallback behavior - use FlextResult monadic operations for defaults.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).
        No longer checks structlog - FlextLogger is independent.

        Args:
            key: The key to get
            scope: The scope to get from (global, user, session)

        Returns:
            FlextResult[object]: Success with value, or failure if key not found

        Example:
            >>> context = FlextContext()
            >>> context.set("key", "value")
            >>> result = context.get("key")
            >>> if result.is_success:
            ...     value = result.unwrap()  # "value"
            >>>
            >>> # Key not found - fast fail
            >>> result = context.get("nonexistent")
            >>> assert result.is_failure
            >>>
            >>> # Use monadic operations for defaults
            >>> value = context.get("key").unwrap_or("default")

        """
        if not self._active:
            return FlextResult[object].fail(
                "Context is not active",
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
            )

        # Get from contextvar (single source of truth)
        scope_data = self._get_from_contextvar(scope)

        if key not in scope_data:
            return FlextResult[object].fail(
                f"Context key '{key}' not found in scope '{scope}'",
                error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
            )

        value = scope_data[key]

        # Update statistics
        self._update_statistics("get")

        # Handle None values - return failure since FlextResult.ok() cannot accept None
        if value is None:
            return FlextResult[object].fail(
                f"Context key '{key}' has None value in scope '{scope}'",
                error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
            )

        return FlextResult[object].ok(value)

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
        self,
        key: str,
        scope: str = FlextConstants.Context.SCOPE_GLOBAL,
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
        current_value = ctx_var.get()
        # Fast fail: contextvar must contain dict or None (uninitialized)
        if current_value is not None and not FlextRuntime.is_dict_like(current_value):
            msg = (
                f"Invalid contextvar value type in scope '{scope}': "
                f"{type(current_value).__name__}. Expected dict[str, object] | None"
            )
            raise TypeError(msg)
        # Initialize with empty dict if None (first use)
        current: dict[str, object] = (
            current_value if FlextRuntime.is_dict_like(current_value) else {}
        )
        if key in current:
            updated = {k: v for k, v in current.items() if k != key}
            ctx_var.set(updated)

            # DELEGATION: Propagate removal to FlextLogger for global scope
            if scope == FlextConstants.Context.SCOPE_GLOBAL:
                FlextLogger.unbind_global_context(key)

            # Update statistics
            self._update_statistics("remove")

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

        # Update statistics using model (type-safe, no .get() needed)
        self._statistics.clears += 1
        if (
            self._statistics.operations is not None
            and "clear" in self._statistics.operations
        ):
            clear_value = self._statistics.operations["clear"]
            if isinstance(clear_value, int):
                self._statistics.operations["clear"] = clear_value + 1

    def keys(self) -> list[str]:
        """Get all keys in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of all keys across all scopes

        """
        if not self._active:
            return []

        # Get keys from all contextvar scopes
        all_keys: set[str] = set()
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                all_keys.update(scope_value.keys())

        return list(all_keys)

    def values(self) -> list[object]:
        """Get all values in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of all values across all scopes

        """
        if not self._active:
            return []

        # Get values from all contextvar scopes
        all_values: list[object] = []
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                all_values.extend(scope_value.values())

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
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                all_items.extend(scope_value.items())

        return all_items

    def merge(self, other: FlextContext | dict[str, object]) -> Self:
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
                current_value = ctx_var.get()
                # Fast fail: contextvar must contain dict or None (uninitialized)
                if current_value is not None and not FlextRuntime.is_dict_like(
                    current_value
                ):
                    msg = (
                        f"Invalid contextvar value type in scope '{scope_name}': "
                        f"{type(current_value).__name__}. Expected dict[str, object] | None"
                    )
                    raise TypeError(msg)
                # Fast fail: scope_data must be dict
                if not FlextRuntime.is_dict_like(scope_data):
                    msg = (
                        f"Invalid scope_data type in scope '{scope_name}': "
                        f"{type(scope_data).__name__}. Expected dict[str, object]"
                    )
                    raise TypeError(msg)
                # Initialize with empty dict if None (first use)
                current_dict: dict[str, object] = (
                    dict(current_value)
                    if FlextRuntime.is_dict_like(current_value)
                    else {}
                )
                updated = {**current_dict, **scope_data}
                ctx_var.set(updated)

                # DELEGATION: Propagate global scope to FlextLogger
                if scope_name == FlextConstants.Context.SCOPE_GLOBAL:
                    FlextLogger.bind_global_context(**scope_data)
        else:
            # Merge dictionary into global scope
            self._set_in_contextvar(FlextConstants.Context.SCOPE_GLOBAL, other)

        return self

    def clone(self) -> FlextContext:
        """Create a clone of this context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            A new FlextContext with the same data

        """
        cloned = FlextContext()

        # Clone all contextvar scopes from this instance
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value) and scope_value:
                cloned_ctx_var = cloned._get_or_create_scope_var(scope_name)
                cloned_ctx_var.set(scope_value.copy())

        # Clone metadata and statistics
        cloned._metadata = (
            self._metadata.model_copy()
            if isinstance(self._metadata, FlextModels.ContextMetadata)
            else FlextModels.ContextMetadata()
        )
        cloned._statistics = self._statistics.model_copy()

        return cloned

    def get_all_scopes(self) -> FlextTypes.ScopeRegistry:
        """Get all scopes.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary of all scopes with their data

        """
        scopes: FlextTypes.ScopeRegistry = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            scope_data = scope_value if FlextRuntime.is_dict_like(scope_value) else {}
            if scope_data:  # Only include non-empty scopes
                scopes[scope_name] = scope_data.copy()
        return scopes

    def validate(self) -> FlextResult[bool]:
        """Validate the context data.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            FlextResult[bool]: Success with True if valid, failure with error details

        """
        if not self._active:
            return FlextResult[bool].fail("Context is not active")

        # Check for empty keys in all contextvar scopes
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                return FlextResult[bool].fail(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                for key in scope_value:
                    if not key:
                        return FlextResult[bool].fail("Invalid key found in context")
        return FlextResult[bool].ok(True)

    def to_json(self) -> str:
        """Convert context to JSON string.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            JSON string representation of the context

        """
        # Combine all contextvar scopes into a single flat dictionary for backward compatibility
        all_data: dict[str, object] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                all_data.update(scope_value)
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
        if FlextRuntime.is_dict_like(data):
            data_values = data.values()
            if all(FlextRuntime.is_dict_like(v) for v in data_values):
                # New scoped format - restore all scopes
                for scope_name, scope_data in data.items():
                    if FlextRuntime.is_dict_like(scope_data):
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
        self._metadata = FlextModels.ContextMetadata()  # Reset model
        self._hooks.clear()

    def add_hook(self, event: str, hook: FlextTypes.HookCallableType) -> None:
        """Add a hook for context events.

        Args:
            event: The event to hook (set, get, remove, clear)
            hook: The hook function to call

        """
        if event not in self._hooks:
            self._hooks[event] = []
        hooks_list = self._hooks[event]
        if FlextRuntime.is_list_like(hooks_list):
            hooks_list.append(hook)

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata for the context.

        Args:
            key: The metadata key
            value: The metadata value

        """
        if isinstance(self._metadata, FlextModels.ContextMetadata):
            # Directly update custom_fields dict to avoid deprecation warning
            # and object recreation
            self._metadata.custom_fields[key] = value

    def get_metadata(self, key: str) -> FlextResult[object]:
        """Get metadata from the context.

        Fast fail: Returns FlextResult[object] - fails if key not found.
        No fallback behavior - use FlextResult monadic operations for defaults.

        Args:
            key: The metadata key

        Returns:
            FlextResult[object]: Success with metadata value, or failure if key not found

        Example:
            >>> context = FlextContext()
            >>> context.set_metadata("key", "value")
            >>> result = context.get_metadata("key")
            >>> if result.is_success:
            ...     value = result.unwrap()  # "value"
            >>>
            >>> # Key not found - fast fail
            >>> result = context.get_metadata("nonexistent")
            >>> assert result.is_failure
            >>>
            >>> # Use monadic operations for defaults
            >>> value = context.get_metadata("key").unwrap_or("default")

        """
        if not isinstance(self._metadata, FlextModels.ContextMetadata):
            return FlextResult[object].fail(
                "Context metadata not initialized",
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
            )

        custom_fields = (
            self._metadata.custom_fields
            if FlextRuntime.is_dict_like(self._metadata.custom_fields)
            else {}
        )

        if key not in custom_fields:
            return FlextResult[object].fail(
                f"Metadata key '{key}' not found",
                error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
            )

        return FlextResult[object].ok(custom_fields[key])

    def get_all_metadata(self) -> dict[str, object]:
        """Get all metadata from the context.

        Returns:
            Dictionary of all metadata (flattened including custom_fields)

        """
        if isinstance(self._metadata, FlextModels.ContextMetadata):
            result = self._metadata.model_dump()
            # Flatten custom_fields into top-level dict for backward compatibility
            custom_fields = result.pop("custom_fields", {})
            result.update(custom_fields)
            return result
        return {}

    def get_all_data(self) -> dict[str, object]:
        """Get all data from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary of all context data across all scopes

        """
        # Combine all contextvar scopes
        all_data: dict[str, object] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                all_data.update(scope_value)
        return all_data

    def get_statistics(self) -> FlextModels.ContextStatistics:
        """Get context statistics.

        Returns:
            ContextStatistics model with operation counts

        """
        return self._statistics

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

        # Reset metadata model
        self._metadata = FlextModels.ContextMetadata()

    def export(self) -> dict[str, object]:
        """Export context data as a dictionary for compatibility consumers.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        """
        export_snapshot = self.export_snapshot()
        return export_snapshot.data

    def export_snapshot(self) -> FlextModels.ContextExport:
        """Return typed export snapshot including metadata and statistics.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.
        Pydantic v2 field_validator accepts models directly - no model_dump() needed.

        """
        # Combine all contextvar scopes
        all_data: dict[str, object] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_value = ctx_var.get()
            # Fast fail: contextvar must contain dict or None (uninitialized)
            if scope_value is not None and not FlextRuntime.is_dict_like(scope_value):
                msg = (
                    f"Invalid contextvar value type in scope '{scope_name}': "
                    f"{type(scope_value).__name__}. Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Skip None (uninitialized scopes)
            if FlextRuntime.is_dict_like(scope_value):
                all_data.update(scope_value)

        # Fast fail: metadata must be ContextMetadata instance
        if not isinstance(self._metadata, FlextModels.ContextMetadata):
            msg = (
                f"Invalid metadata type: {type(self._metadata).__name__}. "
                "Expected FlextModels.ContextMetadata"
            )
            raise TypeError(msg)

        # Fast fail: statistics must be ContextStatistics instance
        if not isinstance(self._statistics, FlextModels.ContextStatistics):
            msg = (
                f"Invalid statistics type: {type(self._statistics).__name__}. "
                "Expected FlextModels.ContextStatistics"
            )
            raise TypeError(msg)

        # Convert metadata to dict using get_all_metadata() for proper API usage
        metadata_dict = self.get_all_metadata()

        # Convert statistics to dict - use model_dump() for Pydantic model
        if isinstance(self._statistics, FlextModels.ContextStatistics):
            statistics_dict = self._statistics.model_dump()
        else:
            statistics_dict = {}

        # Fast fail: metadata and statistics must be dict
        if not FlextRuntime.is_dict_like(metadata_dict):
            msg = (
                f"Invalid metadata_dict type: {type(metadata_dict).__name__}. "
                "Expected dict[str, object]"
            )
            raise TypeError(msg)
        if not FlextRuntime.is_dict_like(statistics_dict):
            msg = (
                f"Invalid statistics_dict type: {type(statistics_dict).__name__}. "
                "Expected dict[str, object]"
            )
            raise TypeError(msg)

        return FlextModels.ContextExport(
            data=all_data,
            metadata=metadata_dict,
            statistics=statistics_dict,
        )

    def import_data(self, data: dict[str, object]) -> None:
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
            >>> container.with_service("my_service", MyService())
            >>> service_result = container.get("my_service")

        """
        if cls._container is None:
            cls._container = FlextContainer.get_global()
        return cls._container

    # ==========================================================================
    # Variables - Context Variables using structlog as Single Source of Truth
    # ==========================================================================

    class Variables:
        """Context variables using structlog as single source of truth."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[_StructlogProxyStr] = _create_str_proxy(
                "correlation_id", default=None
            )
            PARENT_CORRELATION_ID: Final[_StructlogProxyStr] = _create_str_proxy(
                "parent_correlation_id", default=None
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[_StructlogProxyStr] = _create_str_proxy(
                "service_name", default=None
            )
            SERVICE_VERSION: Final[_StructlogProxyStr] = _create_str_proxy(
                "service_version", default=None
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[_StructlogProxyStr] = _create_str_proxy(
                "user_id", default=None
            )
            REQUEST_ID: Final[_StructlogProxyStr] = _create_str_proxy(
                "request_id", default=None
            )
            REQUEST_TIMESTAMP: Final[_StructlogProxyDatetime] = _create_datetime_proxy(
                "request_timestamp", default=None
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[_StructlogProxyStr] = _create_str_proxy(
                "operation_name", default=None
            )
            OPERATION_START_TIME: Final[_StructlogProxyDatetime] = (
                _create_datetime_proxy("operation_start_time", default=None)
            )
            OPERATION_METADATA: Final[_StructlogProxyDict] = _create_dict_proxy(
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
            value = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            return value if isinstance(value, str) else None

        @staticmethod
        def set_correlation_id(correlation_id: str) -> None:
            """Set correlation ID.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate unique correlation ID.

            Note: Uses FlextUtilities.Generators.generate_correlation_id() for ID generation.
            Sets the correlation ID in context variables (via FlextModels.StructlogProxyContextVar).
            """
            correlation_id = FlextUtilities.Generators.generate_correlation_id()
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID."""
            value = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()
            return value if isinstance(value, str) else None

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
            # Generate correlation ID if not provided using FlextUtilities
            if correlation_id is None:
                correlation_id = FlextUtilities.Generators.generate_correlation_id()

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
            if isinstance(existing_id, str):
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
            value = FlextContext.Variables.Service.SERVICE_NAME.get()
            return value if isinstance(value, str) else None

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set service name.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get current service version."""
            value = FlextContext.Variables.Service.SERVICE_VERSION.get()
            return value if isinstance(value, str) else None

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
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
        ) -> FlextResult[bool]:
            """Register service in global container using FlextResult.

            Provides unified service registration pattern across the ecosystem
            by integrating FlextContainer with FlextContext.

            Args:
                service_name: Name to register the service under
                service: Service instance to register

            Returns:
                FlextResult[bool]: Success with True if registered, failure with error details

            Example:
                >>> result = FlextContext.Service.register_service(
                ...     "logger",
                ...     FlextLogger(__name__),
                ... )
                >>> if result.is_failure:
                ...     print(f"Registration failed: {result.error}")

            """
            container = FlextContext.get_container()
            try:
                container.with_service(service_name, service)
                return FlextResult[bool].ok(True)
            except ValueError as e:
                return FlextResult[bool].fail(str(e))

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

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            FlextContext.Variables.Request.USER_ID.set(user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context."""
            return FlextContext.Variables.Performance.OPERATION_NAME.get()

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context."""
            return FlextContext.Variables.Request.REQUEST_ID.get()

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            FlextContext.Variables.Request.REQUEST_ID.set(request_id)

        @staticmethod
        @contextmanager
        def request_context(
            *,
            user_id: str | None = None,
            operation_name: str | None = None,
            request_id: str | None = None,
            metadata: dict[str, object] | None = None,
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
        def set_operation_start_time(
            start_time: datetime | None = None,
        ) -> None:
            """Set operation start time in context."""
            if start_time is None:
                start_time = FlextUtilities.Generators.generate_datetime_utc()
            FlextContext.Variables.Performance.OPERATION_START_TIME.set(start_time)

        @staticmethod
        def get_operation_metadata() -> dict[str, object] | None:
            """Get operation metadata from context."""
            return FlextContext.Variables.Performance.OPERATION_METADATA.get()

        @staticmethod
        def set_operation_metadata(metadata: dict[str, object]) -> None:
            """Set operation metadata in context."""
            FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)

        @staticmethod
        def add_operation_metadata(key: str, value: object) -> None:
            """Add single metadata entry to operation context."""
            metadata_value = FlextContext.Variables.Performance.OPERATION_METADATA.get()
            # Fast fail: metadata must be dict or None (uninitialized)
            if metadata_value is not None and not FlextRuntime.is_dict_like(
                metadata_value
            ):
                msg = (
                    f"Invalid OPERATION_METADATA type: {type(metadata_value).__name__}. "
                    "Expected dict[str, object] | None"
                )
                raise TypeError(msg)
            # Initialize with empty dict if None (first use)
            current_metadata: dict[str, object] = (
                metadata_value if FlextRuntime.is_dict_like(metadata_value) else {}
            )
            current_metadata[key] = value
            FlextContext.Variables.Performance.OPERATION_METADATA.set(current_metadata)

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[dict[str, object]]:
            """Create timed operation context with performance tracking."""
            start_time = FlextUtilities.Generators.generate_datetime_utc()
            operation_metadata: dict[str, object] = {
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
                # Calculate duration with full precision
                end_time = FlextUtilities.Generators.generate_datetime_utc()
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
        def get_full_context() -> dict[str, object]:
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
        def get_correlation_context() -> dict[str, str]:
            """Get correlation context for cross-service propagation."""
            context: dict[str, str] = {}

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
            # Fast fail: use explicit checks instead of OR fallback
            correlation_id_value = context.get("X-Correlation-Id")
            if correlation_id_value is None:
                correlation_id_value = context.get("correlation_id")
            if correlation_id_value is not None and isinstance(
                correlation_id_value,
                str,
            ):
                FlextContext.Variables.Correlation.CORRELATION_ID.set(
                    correlation_id_value,
                )

            parent_id_value = context.get("X-Parent-Correlation-Id")
            if parent_id_value is None:
                parent_id_value = context.get("parent_correlation_id")
            if parent_id_value is not None and isinstance(parent_id_value, str):
                FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                    parent_id_value,
                )

            service_name_value = context.get("X-Service-Name")
            if service_name_value is None:
                service_name_value = context.get("service_name")
            if service_name_value is not None and isinstance(service_name_value, str):
                FlextContext.Variables.Service.SERVICE_NAME.set(service_name_value)

            user_id_value = context.get("X-User-Id")
            if user_id_value is None:
                user_id_value = context.get("user_id")
            if user_id_value is not None and isinstance(user_id_value, str):
                FlextContext.Variables.Request.USER_ID.set(user_id_value)

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

            # Note: All variables use structlog as single source (via FlextModels.StructlogProxyContextVar)

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure correlation ID exists, creating one if needed."""
            correlation_id_value = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
            )
            if isinstance(correlation_id_value, str) and correlation_id_value:
                return correlation_id_value
            return FlextContext.Correlation.generate_correlation_id()

        @staticmethod
        def has_correlation_id() -> bool:
            """Check if correlation ID is set in context."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get() is not None

        @staticmethod
        def get_context_summary() -> str:
            """Get a human-readable context summary for debugging."""
            context = FlextContext.Serialization.get_full_context()
            parts: list[str] = []

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


__all__: list[str] = [
    "FlextContext",
]
