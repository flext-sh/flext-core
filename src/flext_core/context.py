"""Context propagation utilities for dispatcher-coordinated workloads.

FlextContext tracks correlation metadata, request data, and timing information
through the dispatcher pipeline and into handlers, ensuring structured logs and
metrics remain consistent across threads and async boundaries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Generator, Mapping, MutableMapping
from contextlib import contextmanager
from datetime import datetime
from typing import Final, Self, overload

from pydantic import BaseModel, TypeAdapter

from flext_core._models.context import FlextModelsContext
from flext_core.constants import c
from flext_core.loggings import FlextLogger
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t
from flext_core.utilities import u

# Concrete value type for context storage
type ContextValue = t.ConfigMapValue


class FlextContext(FlextRuntime):
    """Context manager for correlation, request data, and timing metadata.

    The dispatcher and decorators rely on FlextContext to move correlation IDs,
    service metadata, and timing details through CQRS handlers without mutating
    function signatures. Data lives in ``contextvars`` to survive async hops and
    thread switches, and hooks keep ``FlextLogger`` in sync for structured logs.

    Highlights:
    - Correlation IDs and service identity helpers for cross-service tracing
    - Request and operation scopes for user/action metadata
    - Timing utilities that feed metrics and structured logging
    - Serialization helpers for propagating context via headers or payloads
    """

    def _protocol_name(self) -> str:
        """Return the protocol name for introspection."""
        return "FlextContext"

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: m.ConfigMap | Mapping[str, t.ConfigMapValue] | None,
    ) -> dict[str, t.ConfigMapValue]:
        """Return contextvar payload as ConfigMap with safe default."""
        if ctx_value is None:
            return {}
        if not hasattr(ctx_value, "items"):
            return {}
        try:
            normalized: dict[str, t.ConfigMapValue] = {}
            for key, value in ctx_value.items():
                if str(key) != key:
                    return {}
                normalized_value = FlextRuntime.normalize_to_general_value(value)
                normalized[key] = normalized_value
            return normalized
        except Exception:
            return {}

    # =========================================================================
    # HELPER METHODS (moved from module level for SRP compliance)
    # =========================================================================

    # Instance attributes
    # Using direct import for mypy compatibility with nested class aliases
    _metadata: m.Metadata

    # NOTE: _scope_vars is an instance attribute (see __init__)
    # No property accessor needed - direct access via self._scope_vars

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def __init__(
        self,
        initial_data: m.ContextData | Mapping[str, t.ConfigMapValue] | None = None,
    ) -> None:
        """Initialize FlextContext with optional initial data.

        ARCHITECTURAL NOTE: FlextContext now uses Python's contextvars for storage,
        completely independent of structlog. It delegates to FlextLogger for logging
        integration, maintaining clear separation of concerns.

        Args:
            initial_data: Optional `m.Context.ContextData` instance or dict

        """
        super().__init__()
        context_data = m.ContextData()
        if initial_data is not None:
            if hasattr(initial_data, "data") and hasattr(initial_data, "metadata"):
                context_data = m.ContextData.model_validate(initial_data)
            else:
                context_data = m.ContextData(
                    data=t.Dict(
                        root=dict(m.ConfigMap.model_validate(initial_data).items())
                    )
                )
        # Initialize context-specific metadata (separate from ContextData.metadata)
        # ContextData.metadata = generic creation/modification metadata (m.Metadata)
        # FlextContext._metadata = context-specific tracing metadata (ContextMetadata)
        self._metadata = m.Metadata()

        self._hooks: MutableMapping[str, list[t.HandlerCallable]] = {}
        # Use nested class from FlextModelsContext
        self._statistics: m.ContextStatistics = m.ContextStatistics()
        self._active: bool = True
        self._suspended: bool = False

        # Create instance-specific contextvars for isolation (required for clone())
        # Note: Each instance gets its own contextvars to prevent state sharing
        # Note: Using None default per B039 - mutable defaults cause issues
        self._scope_vars: MutableMapping[
            str,
            contextvars.ContextVar[m.ConfigMap | None],
        ] = {
            c.Context.SCOPE_GLOBAL: contextvars.ContextVar(
                "flext_global_context",
                default=None,
            ),
            c.Context.SCOPE_USER: contextvars.ContextVar(
                "flext_user_context",
                default=None,
            ),
            c.Context.SCOPE_SESSION: contextvars.ContextVar(
                "flext_session_context",
                default=None,
            ),
        }

        # Initialize contextvars with initial data if provided
        # Note: No self._lock needed - contextvars are thread-safe by design
        # Type narrowing: context_data is always m.Context.ContextData at this point
        if context_data.data:
            # Set initial data in global context
            self._set_in_contextvar(
                c.Context.SCOPE_GLOBAL,
                m.ConfigMap(root=dict(context_data.data.items())),
            )

    @overload
    @classmethod
    def create(
        cls,
        initial_data: m.ConfigMap | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: m.ConfigMap | None = None,
    ) -> Self: ...

    @classmethod
    def create(
        cls,
        initial_data: m.ConfigMap | None = None,
        *,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: m.ConfigMap | None = None,
        auto_correlation_id: bool = True,
    ) -> Self:
        """Factory method to create a new FlextContext instance.

        This is the preferred way to instantiate FlextContext. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Supports two overloads:
        1. Simple creation: create(initial_data=...)
        2. Metadata-based creation: create(operation_id=..., user_id=..., metadata=...)

        Auto-correlation_id generation enables zero-config context setup by
        automatically generating a correlation ID when operation_id is not provided.

        Args:
            initial_data: Optional initial context data (dict or ContextData).
            operation_id: Optional operation identifier (keyword-only).
            user_id: Optional user identifier (keyword-only).
            metadata: Optional additional metadata dictionary (keyword-only).
            auto_correlation_id: If True, auto-generate operation_id if not provided.
                Default: True (enables zero-config setup).

        Returns:
            New FlextContext instance.

        Example:
            >>> # Zero-config: auto-generates correlation_id
            >>> context = FlextContext.create()
            >>>
            >>> # With custom operation_id: disables auto-generation
            >>> context = FlextContext.create(operation_id="op-123", user_id="user-456")
            >>>
            >>> # Disable auto-correlation_id
            >>> context = FlextContext.create(auto_correlation_id=False)

        """
        # Handle overload: if operation_id/user_id/metadata are provided, use metadata-based creation
        if operation_id is not None or user_id is not None or metadata is not None:
            initial_data_dict: m.ConfigMap = m.ConfigMap()
            if operation_id is not None:
                initial_data_dict[c.Context.KEY_OPERATION_ID] = operation_id
            elif auto_correlation_id:
                # Auto-generate correlation_id when not provided and auto_correlation_id=True
                initial_data_dict[c.Context.KEY_OPERATION_ID] = u.generate(
                    "correlation",
                )
            if user_id is not None:
                initial_data_dict[c.Context.KEY_USER_ID] = user_id
            # Merge metadata into initial_data
            if metadata is not None:
                initial_data_dict.update(dict(metadata.items()))
            return cls(
                initial_data=m.ContextData(
                    data=t.Dict(root=dict(initial_data_dict.items()))
                )
            )
        # Default: use initial_data parameter
        # Auto-generate correlation_id for zero-config setup
        data_map = (
            m.ConfigMap.model_validate(initial_data)
            if initial_data is not None
            else m.ConfigMap()
        )
        if auto_correlation_id and c.Context.KEY_OPERATION_ID not in data_map:
            # Convert initial_data to dict if needed
            initial_data_dict_new: m.ConfigMap = data_map.model_copy()
            # Add auto-generated correlation_id
            initial_data_dict_new[c.Context.KEY_OPERATION_ID] = u.generate(
                "correlation",
            )
            return cls(
                initial_data=m.ContextData(
                    data=t.Dict(root=dict(initial_data_dict_new.items()))
                )
            )
        return cls(initial_data=m.ContextData(data=t.Dict(root=dict(data_map.items()))))

    # =========================================================================
    # PRIVATE HELPERS - Context variable management and FlextLogger delegation
    # =========================================================================

    def _get_or_create_scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[m.ConfigMap | None]:
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

    def iter_scope_vars(
        self,
    ) -> Mapping[
        str,
        contextvars.ContextVar[m.ConfigMap | None],
    ]:
        """Get scope context variables for iteration.

        This method provides read-only access to scope variables for merge/clone
        operations, avoiding SLF001 violations when accessing from other instances.

        Returns:
            Dictionary of scope names to their context variables.

        """
        return self._scope_vars

    def _set_in_contextvar(
        self,
        scope: str,
        data: m.ConfigMap,
    ) -> None:
        """Set multiple values in contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        # Use helper for type narrowing to ConfigurationDict
        current = m.ConfigMap.model_validate(
            self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        )
        updated = current.model_copy()
        updated.update(dict(data.items()))
        _ = ctx_var.set(updated)
        if scope == c.Context.SCOPE_GLOBAL:
            normalized_context: dict[str, t.MetadataAttributeValue] = {
                key: FlextRuntime.normalize_to_metadata_value(value)
                for key, value in data.items()
            }
            _ = FlextLogger.bind_global_context(**normalized_context)

    def _get_from_contextvar(
        self,
        scope: str,
    ) -> m.ConfigMap:
        """Get all values from contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        value = ctx_var.get()
        return m.ConfigMap.model_validate(
            FlextContext._narrow_contextvar_to_configuration_dict(value)
        )

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper).

        Args:
            operation: Operation name ('set', 'get', 'remove', etc')

        """
        # Update primary counter using attribute name
        counter_attr: str = f"{operation}s"
        if hasattr(self._statistics, counter_attr):
            current_value: int = getattr(self._statistics, counter_attr, 0)
            # Direct increment (statistics fields are always int)
            setattr(self._statistics, counter_attr, current_value + 1)

        # Update operations dict if exists
        operations = dict(self._statistics.operations.items())
        value = operations.get(operation)
        if isinstance(value, int):
            operations[operation] = value + 1
            self._statistics = self._statistics.model_copy(
                update={"operations": operations}
            )

    def _add_hook(
        self,
        event: str,
        hook: t.HandlerCallable,
    ) -> None:
        """Add a hook for an event.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Args:
            event: Event name ('set', 'get', 'remove', etc.)
            hook: Callable hook function

        """
        if event not in self._hooks:
            self._hooks[event] = []
        if callable(hook):
            self._hooks[event].append(hook)

    def _remove_hook(
        self,
        event: str,
        hook: t.HandlerCallable,
    ) -> None:
        """Remove a hook for an event.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Args:
            event: Event name ('set', 'get', 'remove', etc.)
            hook: Callable hook function to remove

        """
        if event in self._hooks and hook in self._hooks[event]:
            self._hooks[event].remove(hook)

    def _execute_hooks(
        self,
        event: str,
        event_data: ContextValue | m.ConfigMap,
    ) -> None:
        """Execute hooks for an event (DRY helper).

        Args:
            event: Event name ('set', 'get', 'remove', etc.)
            event_data: Data to pass to hooks

        """
        if event not in self._hooks:
            return

        hooks = self._hooks[event]
        # Type narrowing: hooks is list[HandlerCallable], which is Sequence[Callable]
        for hook in hooks:
            if callable(hook):
                # Note: Hooks should not raise exceptions
                # All exceptions indicate a programming error in hook implementation
                hook_data: t.ScalarValue
                if event_data is None or isinstance(
                    event_data,
                    (str, int, float, bool, datetime),
                ):
                    hook_data = event_data
                else:
                    hook_data = str(event_data)
                _ = hook(hook_data)

    @staticmethod
    def _propagate_to_logger(
        key: str,
        value: ContextValue,
        scope: str,
    ) -> None:
        """Propagate context changes to FlextLogger (DRY helper).

        Args:
            key: Context key
            value: Context value
            scope: Context scope

        """
        if scope == c.Context.SCOPE_GLOBAL:
            normalized = FlextRuntime.normalize_to_metadata_value(value)
            _ = FlextLogger.bind_global_context(**{key: normalized})

    # =========================================================================
    # Instance Methods - Core context operations
    # =========================================================================

    @staticmethod
    def _validate_set_inputs(
        key: str,
        value: ContextValue,
    ) -> r[bool]:
        """Validate inputs for set operation.

        Args:
            key: The key to validate
            value: The value to validate

        Returns:
            r[bool]: Success with True if valid, failure with error message

        """
        if not key:
            return r[bool].fail("Key must be a non-empty string")
        if value is None:
            return r[bool].fail("Value cannot be None")
        if (
            u.guard(
                value,
                (str, int, float, bool, list, dict, m.ConfigMap),
                return_value=True,
            )
            is None
        ):
            return r[bool].fail("Value must be serializable")
        return r[bool].ok(value=True)

    def set(
        self,
        key: str,
        value: ContextValue,
        scope: str = c.Context.SCOPE_GLOBAL,
    ) -> r[bool]:
        """Set a value in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            key: The key to set
            value: The value to set
            scope: The scope for the value (global, user, session)

        Returns:
            r[bool]: Success with True if set, failure with error message

        """
        if not self._active:
            return r[bool].fail("Context is not active")

        validation_result = FlextContext._validate_set_inputs(key, value)
        if validation_result.is_failure:
            return r[bool].fail(validation_result.error or "Validation failed")

        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = m.ConfigMap.model_validate(current)
            updated[key] = value
            _ = ctx_var.set(updated)
            FlextContext._propagate_to_logger(key, value, scope)
            self._update_statistics(c.Context.OPERATION_SET)
            self._execute_hooks(
                c.Context.OPERATION_SET,
                m.ConfigMap(root={"key": key, "value": value}),
            )
            return r[bool].ok(value=True)
        except (TypeError, Exception) as e:
            error_msg = (
                str(e)
                if isinstance(e, TypeError)
                else f"Failed to set context value: {e}"
            )
            return r[bool].fail(error_msg)

    def set_all(
        self,
        data: m.ConfigMap,
        scope: str = c.Context.SCOPE_GLOBAL,
    ) -> r[bool]:
        """Set multiple values in the context.

        Args:
            data: Dictionary of key-value pairs to set
            scope: The scope for the values (global, user, session)

        Returns:
            r[bool]: Success with True if all set, failure with error message

        """
        if not self._active:
            return r[bool].fail("Context is not active")

        if not data:
            return r[bool].ok(value=True)

        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = m.ConfigMap.model_validate(current)
            updated.update(dict(data.items()))
            _ = ctx_var.set(updated)
            self._update_statistics(c.Context.OPERATION_SET)
            self._execute_hooks(
                c.Context.OPERATION_SET,
                m.ConfigMap(root={"data": m.ConfigMap(root=dict(data.items()))}),
            )
            return r[bool].ok(value=True)
        except (TypeError, Exception) as e:
            error_msg = (
                str(e)
                if isinstance(e, TypeError)
                else f"Failed to set context values: {e}"
            )
            return r[bool].fail(error_msg)

    def get(
        self,
        key: str,
        scope: str = c.Context.SCOPE_GLOBAL,
    ) -> r[ContextValue]:
        """Get a value from the context.

        Fast fail: Returns r[ContextValue] - fails if key not found.
        No fallback behavior - use FlextResult monadic operations for defaults.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).
        No longer checks structlog - FlextLogger is independent.

        Args:
            key: The key to get
            scope: The scope to get from (global, user, session)

        Returns:
            r[ContextValue]: Success with value, or failure if key not found

        Example:
            >>> context = FlextContext()
            >>> context.set("key", "value")
            >>> result = context.get("key")
            >>> if result.is_success:
            ...     value = result.value  # "value"
            >>>
            >>> # Key not found - fast fail
            >>> result = context.get("nonexistent")
            >>> assert result.is_failure
            >>>
            >>> # Use monadic operations for defaults
            >>> value = context.get("key").unwrap_or("default")

        """
        if not self._active:
            return r[ContextValue].fail("Context is not active")

        # Get from contextvar (single source of truth)
        scope_data = self._get_from_contextvar(scope)

        if key not in scope_data:
            return r[ContextValue].fail(
                f"Context key '{key}' not found in scope '{scope}'",
            )

        value = scope_data[key]

        # Update statistics
        self._update_statistics(c.Context.OPERATION_GET)

        # Handle None values - return failure since FlextResult.ok() cannot accept None
        if value is None:
            return r[ContextValue].fail(
                f"Context key '{key}' has None value in scope '{scope}'",
            )

        def normalize_plain(raw_value):
            mapped_value = FlextRuntime.normalize_to_general_value(raw_value)
            try:
                normalized_map = m.ConfigMap.model_validate(mapped_value)
            except Exception:
                root_value = getattr(mapped_value, "root", None)
                try:
                    normalized_map = m.ConfigMap.model_validate(root_value)
                except Exception:
                    return mapped_value
            return {
                str(item_key): normalize_plain(item_value)
                for item_key, item_value in normalized_map.items()
            }

        return r[ContextValue].ok(normalize_plain(value))

    def has(self, key: str, scope: str = c.Context.SCOPE_GLOBAL) -> bool:
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
        scope: str = c.Context.SCOPE_GLOBAL,
    ) -> None:
        """Remove a key from the context."""
        if not self._active:
            return
        ctx_var = self._get_or_create_scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key in current:
            # Use filter_dict for concise key removal
            filtered = u.filter_dict(
                dict(current.items()),
                lambda k, _v: k != key,
            )
            _ = ctx_var.set(m.ConfigMap(root=dict(filtered.items())))
            # Note: ContextVar.set() already cleared the key, no need to unbind from logger
            # FlextLogger doesn't have unbind_global_context method
            self._update_statistics(c.Context.OPERATION_REMOVE)

    def clear(self) -> None:
        """Clear all data from the context including metadata.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        This method consolidates cleanup functionality - it clears all scope
        data, resets metadata, and updates statistics.

        """
        if not self._active:
            return

        # Clear all contextvar scopes
        for scope_name, ctx_var in self._scope_vars.items():
            _ = ctx_var.set(m.ConfigMap())

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == c.Context.SCOPE_GLOBAL:
                _ = FlextLogger.clear_global_context()

        # Reset metadata model (formerly in cleanup())
        self._metadata = m.Metadata()

        # Update statistics using model (type-safe, no .get() needed)
        self._statistics.clears += 1
        operations = dict(self._statistics.operations.items())
        clear_value = operations.get(c.Context.OPERATION_CLEAR)
        if isinstance(clear_value, int):
            operations[c.Context.OPERATION_CLEAR] = clear_value + 1
            self._statistics = self._statistics.model_copy(
                update={"operations": operations}
            )

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
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_keys.update(scope_dict.keys())
        return list(all_keys)

    def merge(
        self,
        other: FlextContext | Mapping[str, t.ConfigMapValue],
    ) -> Self:
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

        export_callable = getattr(other, "export", None)
        if callable(export_callable):
            exported = export_callable(as_dict=True)
            try:
                exported_map = m.ConfigMap.model_validate(exported)
            except Exception:
                exported_map = m.ConfigMap()

            for scope_name, scope_payload in exported_map.items():
                if scope_name not in {
                    c.Context.SCOPE_GLOBAL,
                    c.Context.SCOPE_USER,
                    c.Context.SCOPE_SESSION,
                }:
                    continue
                try:
                    scope_data = m.ConfigMap.model_validate(scope_payload)
                except Exception:
                    continue
                self._set_in_contextvar(scope_name, scope_data)
        else:
            self._set_in_contextvar(
                c.Context.SCOPE_GLOBAL,
                m.ConfigMap.model_validate(other),
            )

        return self

    def clone(self) -> Self:
        """Create a clone of this context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            A new FlextContext with the same data

        """
        cloned: Self = self.__class__()
        # Clone all scope data using public API
        for scope_name, ctx_var in self.iter_scope_vars().items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                # Set all key-value pairs in the cloned context for this scope
                result = cloned.set_all(
                    m.ConfigMap.model_validate(scope_dict), scope_name
                )
                if not result:
                    # If setting fails, log warning but continue cloning
                    pass
        # Clone metadata and statistics using public methods
        cloned.set_all_metadata_for_clone(self._metadata.model_copy())
        statistics_copy: m.ContextStatistics = self._statistics.model_copy()
        cloned.set_statistics_for_clone(statistics_copy)

        # Type narrowing: cloned is FlextContext which implements p.Context protocol
        # FlextContext structurally implements p.Context, so no cast needed
        return cloned

    def validate(self) -> r[bool]:
        """Validate the context data.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            r[bool]: Success with True if valid, failure with error details

        """
        if not self._active:
            return r[bool].fail("Context is not active")
        for ctx_var in self._scope_vars.values():
            try:
                scope_dict = self._narrow_contextvar_to_configuration_dict(
                    ctx_var.get()
                )
            except TypeError as e:
                return r[bool].fail(str(e))
            for key in scope_dict:
                if not key:
                    return r[bool].fail("Invalid key found in context")
        return r[bool].ok(value=True)

    def to_json(self) -> str:
        """Convert context to JSON string.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            JSON string representation of the context

        """
        all_data: m.ConfigMap = m.ConfigMap()
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_data.update(dict(scope_dict.items()))
        json_ready = {
            k: FlextRuntime.normalize_to_general_value(v) for k, v in all_data.items()
        }
        return json.dumps(json_ready, default=str)

    @classmethod
    def create_with_metadata(
        cls,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: m.ConfigMap | None = None,
    ) -> Self:
        """Create context with operation and user metadata.

        Factory method for creating FlextContext instances with common metadata.

        Args:
            operation_id: Optional operation identifier
            user_id: Optional user identifier
            metadata: Optional additional metadata dictionary

        Returns:
            New FlextContext instance with provided metadata

        """
        initial_data: m.ConfigMap = m.ConfigMap()
        if operation_id is not None:
            initial_data[c.Context.KEY_OPERATION_ID] = operation_id
        if user_id is not None:
            initial_data[c.Context.KEY_USER_ID] = user_id
        if metadata is not None:
            initial_data.update(dict(metadata.items()))
        return cls.create(initial_data=initial_data or None)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create context from JSON string.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Args:
            json_str: JSON string representation of the context

        Returns:
            New FlextContext instance with data from JSON

        Raises:
            ValueError: If JSON string is invalid

        """
        try:
            data_obj = json.loads(json_str)
            if not isinstance(data_obj, dict):
                msg = f"JSON must represent a dict, got {data_obj.__class__.__name__}"
                raise TypeError(msg)
            # Use u.map to normalize each value in dict to ensure ContextValue compatibility

            def normalize_value(value: ContextValue) -> ContextValue:
                """Normalize value to ContextValue."""
                return FlextRuntime.normalize_to_general_value(value)

            normalized_data_for_json: Mapping[str, ContextValue] = {
                str(key): normalize_value(
                    TypeAdapter(t.ConfigMapValue).validate_python(value),
                )
                for key, value in data_obj.items()
            }
            context_data_for_json: m.ContextData = m.ContextData(
                data=t.Dict(root=dict(normalized_data_for_json.items())),
            )
            # Type narrowing: cls(initial_data=context_data_for_json) returns FlextContext which implements p.Context protocol
            # FlextContext structurally implements p.Context, so no cast needed
            return cls(initial_data=context_data_for_json)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON string: {e}"
            raise ValueError(msg) from e

    def _import_data(
        self,
        data: m.ConfigMap,
    ) -> None:
        """Import data into context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            data: Dictionary of key-value pairs to import

        """
        if not self._active:
            return
        # Use FlextRuntime.normalize_to_general_value directly - no wrapper needed
        # Normalize each value in dict to ensure ContextValue compatibility
        normalized_data: m.ConfigMap = m.ConfigMap()
        for k, v in data.items():
            normalized_data[str(k)] = FlextRuntime.normalize_to_general_value(v)
        # Merge into global scope
        self._set_in_contextvar(
            c.Context.SCOPE_GLOBAL,
            normalized_data,
        )

    def items(self) -> list[tuple[str, ContextValue]]:
        """Get all items (key-value pairs) in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of (key, value) tuples across all scopes

        """
        if not self._active:
            return []
        all_items: list[tuple[str, ContextValue]] = []
        for ctx_var in self._scope_vars.values():
            # Use helper for type narrowing to ConfigurationDict
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_items.extend(scope_dict.items())
        return all_items

    def export(
        self,
        *,
        include_statistics: bool = False,
        include_metadata: bool = False,
        as_dict: bool = True,
    ) -> m.ContextExport | dict[str, t.ConfigMapValue]:
        """Export context data for serialization or debugging.

        Args:
            include_statistics: Include context statistics
            include_metadata: Include context metadata
            as_dict: If True, return as dict instead of ContextExport model

        Returns:
            ContextExport model or dict with all requested data

        """
        all_data: m.ConfigMap = m.ConfigMap()

        # Collect all scope data
        all_scopes = self._get_all_scopes()
        all_data.update(dict(all_scopes.items()))

        # Collect statistics if requested
        stats_dict_export: m.ConfigMap | None = None
        if include_statistics and self._statistics:
            stats_dict_export = m.ConfigMap(root=self._statistics.model_dump())

        # Collect metadata if requested
        metadata_dict_export: dict[str, t.ConfigMapValue] | None = None
        if include_metadata:
            metadata_dict_export = self._get_all_metadata()

        # Normalize metadata_dict values to MetadataAttributeDict
        metadata_for_model: m.ConfigMap | None = None
        if metadata_dict_export:
            # Convert ConfigurationDict to MetadataAttributeDict
            normalized_metadata_map: dict[str, t.ConfigMapValue] = {}
            for k, v in metadata_dict_export.items():
                metadata_value: t.ConfigMapValue = v
                if hasattr(v, "items") and callable(getattr(v, "items", None)):
                    metadata_value = m.ConfigMap.model_validate(v)
                normalized_metadata_map[k] = FlextRuntime.normalize_to_general_value(
                    FlextRuntime.normalize_to_metadata_value(metadata_value)
                )
            metadata_for_model = m.ConfigMap(root=normalized_metadata_map)

        # Create ContextExport model
        # statistics expects ContextMetadataMapping (Mapping[str, ContextValue])
        statistics_mapping: t.Dict = t.Dict(
            root=dict((stats_dict_export or m.ConfigMap()).items())
        )

        # Return as dict if requested
        if as_dict:
            result_dict: dict[str, t.ConfigMapValue] = {
                scope_name: scope_data for scope_name, scope_data in all_scopes.items()
            }
            if include_statistics and stats_dict_export:
                result_dict["statistics"] = stats_dict_export
            if include_metadata and metadata_dict_export:
                result_dict["metadata"] = metadata_dict_export
            return result_dict

        metadata_root: m.ConfigMap | None = (
            m.ConfigMap(
                root={
                    k: FlextRuntime.normalize_to_general_value(v)
                    for k, v in metadata_for_model.items()
                }
            )
            if metadata_for_model
            else None
        )

        return m.ContextExport(
            data=dict(all_data.items()),
            metadata=m.Metadata(
                attributes={
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in metadata_root.items()
                }
            )
            if metadata_root
            else None,
            statistics=dict(statistics_mapping.items()),
        )

    def values(self) -> list[ContextValue]:
        """Get all values in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of all values across all scopes

        """
        if not self._active:
            return []
        all_values: list[ContextValue] = []
        for ctx_var in self._scope_vars.values():
            # Use helper for type narrowing to ConfigurationDict
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_values.extend(scope_dict.values())
        return all_values

    def _suspend(self) -> None:
        """Suspend the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        """
        self._suspended = True

    def _resume(self) -> None:
        """Resume the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        """
        self._suspended = False

    def _set_suspended(self, *, suspended: bool) -> None:
        """Set the suspended state of the context.

        Args:
            suspended: True to suspend, False to resume

        Note:
            Use is_active property to check if context is active and not suspended.
            Replaces separate suspend() and resume() methods.

        """
        self._suspended = suspended

    def is_active(self) -> bool:
        """Check if context is active.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            True if context is active and not suspended, False otherwise

        """
        return self._active and not self._suspended

    def _destroy(self) -> None:
        """Destroy the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        """
        self._active = False

        # Clear all contextvar scopes
        for scope_name, ctx_var in self._scope_vars.items():
            _ = ctx_var.set(m.ConfigMap())

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == c.Context.SCOPE_GLOBAL:
                _ = FlextLogger.clear_global_context()

        # Clear metadata and hooks
        self._metadata = m.Metadata()  # Reset model
        self._hooks.clear()

    def set_metadata(self, key: str, value: ContextValue) -> None:
        """Set metadata for the context.

        Args:
            key: The metadata key
            value: The metadata value

        """
        # Normalize value to MetadataAttributeValue before setting
        normalized_value: t.MetadataAttributeValue = (
            FlextRuntime.normalize_to_metadata_value(value)
        )
        # Directly update attributes dict to avoid deprecation warning
        # and object recreation
        updated_attributes = dict(self._metadata.attributes.items())
        updated_attributes[key] = normalized_value
        self._metadata = self._metadata.model_copy(
            update={"attributes": updated_attributes}
        )

    def get_metadata(self, key: str) -> r[ContextValue]:
        """Get metadata from the context.

        Fast fail: Returns r[ContextValue] - fails if key not found.
        No fallback behavior - use FlextResult monadic operations for defaults.

        Args:
            key: The metadata key

        Returns:
            r[ContextValue]: Success with metadata value, or failure if key not found

        Example:
            >>> context = FlextContext()
            >>> context.set_metadata("key", "value")
            >>> result = context.get_metadata("key")
            >>> if result.is_success:
            ...     value = result.value  # "value"
            >>>
            >>> # Key not found - fast fail
            >>> result = context.get_metadata("nonexistent")
            >>> assert result.is_failure
            >>>
            >>> # Use monadic operations for defaults
            >>> value = context.get_metadata("key").unwrap_or("default")

        """
        # Access attributes directly - it's a dict[str, MetadataAttributeValue]
        if key not in self._metadata.attributes:
            return r[ContextValue].fail(f"Metadata key '{key}' not found")

        value: ContextValue = self._metadata.attributes[key]
        normalized_value: ContextValue = FlextRuntime.normalize_to_general_value(
            value,
        )
        return r[ContextValue].ok(normalized_value)

    def _get_all_data(self) -> dict[str, t.ConfigMapValue]:
        """Get all data from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary of all context data across all scopes

        """
        all_data: dict[str, t.ConfigMapValue] = {}
        for ctx_var in self._scope_vars.values():
            # Use helper for type narrowing to ConfigurationDict
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_data.update(scope_dict)
        return all_data

    def _get_statistics(self) -> m.ContextStatistics:
        """Get context statistics.

        Returns:
            ContextStatistics model with operation counts

        """
        return self._statistics

    def set_statistics_for_clone(
        self,
        statistics: m.ContextStatistics,
    ) -> None:
        """Set context statistics (used internally for cloning).

        Args:
            statistics: ContextStatistics model to set

        """
        self._statistics = statistics

    def set_all_metadata_for_clone(self, metadata: m.Metadata) -> None:
        """Set all metadata for the context (used internally for cloning).

        Args:
            metadata: Metadata model to set

        """
        self._metadata = metadata

    def _set_all_metadata(self, metadata: m.Metadata) -> None:
        """Set all metadata for the context (used internally for cloning).

        Args:
            metadata: Metadata model to set

        """
        self._metadata = metadata

    def _get_all_metadata(self) -> dict[str, t.ConfigMapValue]:
        """Get all metadata from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.
        Custom fields are flattened into the top-level dict for easy access.

        Returns:
            Dictionary of all metadata (with custom_fields flattened)

        """
        # Convert Pydantic model to dict
        data = dict(self._metadata.model_dump().items())
        custom_fields_raw = data.pop("custom_fields", {})
        custom_fields_dict: dict[str, t.ConfigMapValue] = {}
        try:
            custom_fields_dict = dict(
                m.ConfigMap.model_validate(custom_fields_raw).items()
            )
        except Exception:
            custom_fields_dict = {}
        result: dict[str, t.ConfigMapValue] = {}
        for k, v in data.items():
            if v is None or v == {}:
                continue
            if hasattr(v, "isoformat") and callable(getattr(v, "isoformat", None)):
                result[k] = v.isoformat()
            else:
                result[k] = v
        result.update(custom_fields_dict)
        return result

    def _get_all_scopes(self) -> dict[str, dict[str, t.ConfigMapValue]]:
        """Get all scope registrations.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary mapping scope names to their data dictionaries

        """
        if not self._active:
            return {}
        scopes: dict[str, dict[str, t.ConfigMapValue]] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            # Use helper for type narrowing to ConfigurationDict
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                scopes[scope_name] = dict(scope_dict.items())
        return scopes

    def _export_snapshot(self) -> m.ContextExport:
        """Export context snapshot.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            ContextExport model with complete context state

        """
        # Get all data
        all_data_dict = self._get_all_data()
        all_data: m.ConfigMap = m.ConfigMap.model_validate(all_data_dict)

        # Get metadata as dict
        metadata_dict = self._get_all_metadata()

        # Normalize metadata values to MetadataAttributeDict
        metadata_for_model: m.ConfigMap | None = None
        if metadata_dict:
            # Convert ConfigurationDict to MetadataAttributeDict
            metadata_for_model = m.ConfigMap(
                root={
                    k: FlextRuntime.normalize_to_general_value(
                        FlextRuntime.normalize_to_metadata_value(v)
                    )
                    for k, v in metadata_dict.items()
                }
            )

        # Get statistics as dict
        stats_dict_raw: m.ConfigMap = m.ConfigMap()
        if hasattr(self._statistics, "model_dump"):
            stats_dict_raw = m.ConfigMap(root=self._statistics.model_dump())

        # Create ContextExport model
        # statistics expects ContextMetadataMapping (Mapping[str, ContextValue])
        statistics_mapping: t.Dict = t.Dict(root=dict(stats_dict_raw.items()))
        metadata_root: m.ConfigMap | None = (
            m.ConfigMap(
                root={
                    k: FlextRuntime.normalize_to_general_value(v)
                    for k, v in metadata_for_model.items()
                }
            )
            if metadata_for_model
            else None
        )

        return m.ContextExport(
            data=dict(all_data.items()),
            metadata=m.Metadata(
                attributes={
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in metadata_root.items()
                }
            )
            if metadata_root
            else None,
            statistics=dict(statistics_mapping.items()),
        )

    # =========================================================================
    # Container integration for dependency injection
    # =========================================================================

    _container: p.DI | None = None

    @classmethod
    def get_container(cls) -> p.DI:
        """Get global container instance.

        The container must be set explicitly using `set_container()` before
        calling this method. This breaks the circular dependency by requiring
        explicit initialization.

        Returns:
            Global DI instance for dependency injection

        Raises:
            RuntimeError: If container was not set via `set_container()`.

        Example:
            >>> from flext_core import FlextContainer
            >>> FlextContext.set_container(FlextContainer.get_global())
            >>> container = FlextContext.get_container()
            >>> result = container.get("service_name")

        """
        if cls._container is None:
            msg = (
                "Container not initialized. Call FlextContext.set_container(container) "
                "before using get_container()."
            )
            raise RuntimeError(msg)
        return cls._container

    @classmethod
    def set_container(cls, container: p.DI) -> None:
        """Set the global container instance.

        This method must be called before using `get_container()`. It breaks
        the circular dependency by requiring explicit initialization.

        Args:
            container: Container instance to set as global.

        Example:
            >>> from flext_core import FlextContainer
            >>> container = FlextContainer.get_global()
            >>> FlextContext.set_container(container)

        """
        cls._container = container

    # ==========================================================================
    # Variables - Context Variables using structlog as Single Source of Truth
    # ==========================================================================

    class Variables:
        """Context variables using structlog as single source of truth."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(
                    c.Context.KEY_CORRELATION_ID,
                    default=None,
                )
            )
            PARENT_CORRELATION_ID: Final[
                FlextModelsContext.StructlogProxyContextVar[str]
            ] = u.create_str_proxy(
                c.Context.KEY_PARENT_CORRELATION_ID,
                default=None,
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(
                    c.Context.KEY_SERVICE_NAME,
                    default=None,
                )
            )
            SERVICE_VERSION: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(
                    "service_version",
                    default=None,
                )
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(
                    c.Context.KEY_USER_ID,
                    default=None,
                )
            )
            REQUEST_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(
                    "request_id",
                    default=None,
                )
            )
            REQUEST_TIMESTAMP: Final[
                FlextModelsContext.StructlogProxyContextVar[datetime]
            ] = u.create_datetime_proxy(
                "request_timestamp",
                default=None,
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(
                    c.Context.KEY_OPERATION_NAME,
                    default=None,
                )
            )
            OPERATION_START_TIME: Final[
                FlextModelsContext.StructlogProxyContextVar[datetime]
            ] = u.create_datetime_proxy("operation_start_time", default=None)
            OPERATION_METADATA: Final[
                FlextModelsContext.StructlogProxyContextVar[t.ConfigMap]
            ] = u.create_dict_proxy(
                "operation_metadata",
                default=None,
            )

        # =====================================================================
        # Convenience Aliases - Flat access to nested context variables
        # =====================================================================
        # These aliases provide flat access (Variables.CorrelationId) to the
        # nested structure (Variables.Correlation.CORRELATION_ID) for ergonomic
        # API usage in FlextContext methods.
        CorrelationId: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Correlation.CORRELATION_ID
        )
        ParentCorrelationId: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Correlation.PARENT_CORRELATION_ID
        )
        ServiceName: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Service.SERVICE_NAME
        )
        ServiceVersion: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Service.SERVICE_VERSION
        )
        UserId: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Request.USER_ID
        )
        RequestId: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Request.REQUEST_ID
        )
        RequestTimestamp: Final[
            FlextModelsContext.StructlogProxyContextVar[datetime]
        ] = Request.REQUEST_TIMESTAMP
        OperationName: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
            Performance.OPERATION_NAME
        )
        OperationStartTime: Final[
            FlextModelsContext.StructlogProxyContextVar[datetime]
        ] = Performance.OPERATION_START_TIME
        OperationMetadata: Final[
            FlextModelsContext.StructlogProxyContextVar[t.ConfigMap]
        ] = Performance.OPERATION_METADATA

    # =========================================================================
    # Correlation Domain - Distributed tracing and correlation ID management
    # =========================================================================

    class Correlation:
        """Distributed tracing and correlation ID management utilities."""

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID."""
            correlation_id = FlextContext.Variables.CorrelationId.get()
            return correlation_id if isinstance(correlation_id, str) else None

        @staticmethod
        def set_correlation_id(correlation_id: str | None) -> None:
            """Set correlation ID.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            Accepts ``None`` to explicitly clear the active correlation when needed.
            """
            _ = FlextContext.Variables.CorrelationId.set(correlation_id)

        @staticmethod
        def reset_correlation_id() -> None:
            """Clear correlation ID from context variables."""
            _ = FlextContext.Variables.CorrelationId.set(None)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate unique correlation ID.

            Note: Uses u.generate("correlation") for ID generation.
            Sets the correlation ID in context variables (via FlextModels.StructlogProxyContextVar).
            """
            correlation_id: str = u.generate("correlation")
            _ = FlextContext.Variables.CorrelationId.set(correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID."""
            parent_id = FlextContext.Variables.ParentCorrelationId.get()
            return parent_id if isinstance(parent_id, str) else None

        @staticmethod
        def set_parent_correlation_id(parent_id: str) -> None:
            """Set parent correlation ID."""
            _ = FlextContext.Variables.ParentCorrelationId.set(parent_id)

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create correlation context scope.

            Uses c.Context configuration for correlation ID generation.
            """
            # Generate correlation ID if not provided using u
            if correlation_id is None:
                correlation_id = u.generate("correlation")

            # Save current context
            current_correlation = FlextContext.Variables.CorrelationId.get()

            # Set new context
            correlation_token = FlextContext.Variables.CorrelationId.set(
                correlation_id,
            )

            # Set parent context
            parent_token = None
            if parent_id:
                parent_token = FlextContext.Variables.ParentCorrelationId.set(
                    parent_id,
                )
            elif current_correlation:
                # Current correlation becomes parent
                parent_token = FlextContext.Variables.ParentCorrelationId.set(
                    current_correlation,
                )

            try:
                yield correlation_id
            finally:
                # Restore previous context
                FlextContext.Variables.CorrelationId.reset(
                    correlation_token,
                )
                if parent_token:
                    FlextContext.Variables.ParentCorrelationId.reset(
                        parent_token,
                    )

        @staticmethod
        @contextmanager
        def inherit_correlation() -> Generator[str | None]:
            """Inherit or create correlation ID."""
            existing_id = FlextContext.Variables.CorrelationId.get()
            if existing_id is not None:
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
            service_name = FlextContext.Variables.ServiceName.get()
            return service_name if isinstance(service_name, str) else None

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set service name.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.ServiceName.set(service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get current service version."""
            service_version = FlextContext.Variables.ServiceVersion.get()
            return service_version if isinstance(service_version, str) else None

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.ServiceVersion.set(version)

        @staticmethod
        def get_service(
            service_name: str,
        ) -> r[ContextValue]:
            """Resolve service from global container using FlextResult.

            Provides unified service resolution pattern across the ecosystem
            by integrating FlextContainer with FlextContext.

            Args:
                service_name: Name of the service to retrieve

            Returns:
                Result containing the service instance or error.
                Use get_typed() or type identity / MRO checks for type narrowing.

            Example:
                >>> result = FlextContext.Service.get_service("logger")
                >>> if result.is_success and isinstance(result.value, FlextLogger):
                ...     result.value.info("Service retrieved")

            """
            # get_container is a classmethod on FlextContext, access via class
            container: p.DI = FlextContext.get_container()
            # Container.get returns p.ResultLike[RegisterableService]
            # We need to convert to r[PayloadValue]
            service_result: p.ResultLike[object] = container.get(service_name)
            # Convert protocol result to concrete FlextResult
            if service_result.is_success:
                # Service value might be RegisterableService
                service_value = service_result.value
                if service_value is None or isinstance(
                    service_value,
                    (str, int, float, bool, datetime),
                ):
                    return r[ContextValue].ok(service_value)
                return r[ContextValue].ok(str(service_value))
            return r[ContextValue].fail(
                service_result.error or "Service not found",
            )

        @staticmethod
        def register_service(
            service_name: str,
            service: ContextValue | BaseModel,
        ) -> r[bool]:
            """Register service in global container using FlextResult.

            Provides unified service registration pattern across the ecosystem
            by integrating FlextContainer with FlextContext.

            Args:
                service_name: Name to register the service under
                service: Service instance to register

            Returns:
                r[bool]: Success with True if registered, failure with error details

            Example:
                >>> result = FlextContext.Service.register_service(
                ...     "logger",
                ...     FlextLogger(__name__),
                ... )
                >>> if result.is_failure:
                ...     print(f"Registration failed: {result.error}")

            """
            # get_container is a classmethod on FlextContext, access via class
            container = FlextContext.get_container()
            try:
                # Use container.with_service for fluent API
                # with_service returns Self for fluent chaining, but we don't need the return value
                _ = container.with_service(service_name, service)
                return r[bool].ok(value=True)
            except ValueError as e:
                return r[bool].fail(str(e))

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service context scope."""
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.ServiceName.get()
            _ = FlextContext.Variables.ServiceVersion.get()

            # Set new context
            name_token = FlextContext.Variables.ServiceName.set(service_name)
            version_token = None
            if version:
                version_token = FlextContext.Variables.ServiceVersion.set(
                    version,
                )

            try:
                yield
            finally:
                # Restore previous context
                FlextContext.Variables.ServiceName.reset(name_token)
                if version_token:
                    FlextContext.Variables.ServiceVersion.reset(version_token)

    # =========================================================================
    # Request Domain - User and request metadata management
    # =========================================================================

    class Request:
        """Request-level context management for user and operation metadata utilities."""

        @staticmethod
        def get_user_id() -> str | None:
            """Get current user ID."""
            user_id = FlextContext.Variables.UserId.get()
            return str(user_id) if user_id is not None else None

        @staticmethod
        def set_user_id(user_id: str) -> None:
            """Set user ID in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.UserId.set(user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context."""
            operation_name = FlextContext.Variables.OperationName.get()
            return str(operation_name) if operation_name is not None else None

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.OperationName.set(operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context."""
            request_id = FlextContext.Variables.RequestId.get()
            return str(request_id) if request_id is not None else None

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.RequestId.set(request_id)

        @staticmethod
        @contextmanager
        def request_context(
            *,
            user_id: str | None = None,
            operation_name: str | None = None,
            request_id: str | None = None,
            metadata: m.ConfigMap | None = None,
        ) -> Generator[None]:
            """Create request metadata context scope with automatic cleanup."""
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.UserId.get()
            _ = FlextContext.Variables.OperationName.get()
            _ = FlextContext.Variables.RequestId.get()
            _ = FlextContext.Variables.OperationMetadata.get()

            # Set new context
            user_token = FlextContext.Variables.UserId.set(user_id) if user_id else None
            operation_token = (
                FlextContext.Variables.OperationName.set(operation_name)
                if operation_name
                else None
            )
            request_token = (
                FlextContext.Variables.RequestId.set(request_id) if request_id else None
            )
            metadata_token = (
                FlextContext.Variables.OperationMetadata.set(metadata)
                if metadata
                else None
            )

            try:
                yield
            finally:
                # Restore previous context
                if user_token is not None:
                    FlextContext.Variables.UserId.reset(user_token)
                if operation_token is not None:
                    FlextContext.Variables.OperationName.reset(
                        operation_token,
                    )
                if request_token is not None:
                    FlextContext.Variables.RequestId.reset(request_token)
                if metadata_token is not None:
                    FlextContext.Variables.OperationMetadata.reset(
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
            start_time = FlextContext.Variables.OperationStartTime.get()
            return start_time if isinstance(start_time, datetime) else None

        @staticmethod
        def set_operation_start_time(
            start_time: datetime | None = None,
        ) -> None:
            """Set operation start time in context."""
            if start_time is None:
                start_time = u.generate_datetime_utc()
            _ = FlextContext.Variables.OperationStartTime.set(start_time)

        @staticmethod
        def get_operation_metadata() -> dict[str, t.ConfigMapValue] | None:
            """Get operation metadata from context."""
            metadata_value = FlextContext.Variables.OperationMetadata.get()
            if metadata_value is None:
                return None
            try:
                metadata_map: dict[str, t.ConfigMapValue] = {
                    key: value for key, value in metadata_value.items()
                }
            except Exception:
                return None
            return metadata_map

        @staticmethod
        def set_operation_metadata(
            metadata: Mapping[str, t.ConfigMapValue],
        ) -> None:
            """Set operation metadata in context."""
            _ = FlextContext.Variables.OperationMetadata.set(
                t.ConfigMap(root=dict(metadata.items()))
            )

        @staticmethod
        def add_operation_metadata(
            key: str,
            value: ContextValue,
        ) -> None:
            """Add single metadata entry to operation context."""
            metadata_value = FlextContext.Variables.OperationMetadata.get()
            current_metadata: t.ConfigMap = (
                metadata_value.model_copy()
                if metadata_value is not None
                else t.ConfigMap()
            )
            current_metadata[key] = value
            _ = FlextContext.Variables.OperationMetadata.set(
                current_metadata,
            )

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[m.ConfigMap]:
            """Create timed operation context with performance tracking."""
            start_time = u.generate_datetime_utc()
            operation_metadata: m.ConfigMap = m.ConfigMap(
                root={
                    c.Context.METADATA_KEY_START_TIME: start_time.isoformat(),
                    c.Context.KEY_OPERATION_NAME: operation_name,
                }
            )

            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.OperationStartTime.get()
            _ = FlextContext.Variables.OperationMetadata.get()
            _ = FlextContext.Variables.OperationName.get()

            # Set new context
            start_token = FlextContext.Variables.OperationStartTime.set(
                start_time,
            )
            metadata_token = FlextContext.Variables.OperationMetadata.set(
                operation_metadata,
            )
            operation_token = None
            if operation_name:
                operation_token = FlextContext.Variables.OperationName.set(
                    operation_name,
                )

            try:
                yield operation_metadata
            finally:
                # Calculate duration with full precision
                end_time = u.generate_datetime_utc()
                duration = (end_time - start_time).total_seconds()
                operation_metadata.update(
                    {
                        c.Context.METADATA_KEY_END_TIME: end_time.isoformat(),
                        c.Context.METADATA_KEY_DURATION_SECONDS: duration,
                    },
                )

                # Restore previous context
                FlextContext.Variables.OperationStartTime.reset(
                    start_token,
                )
                FlextContext.Variables.OperationMetadata.reset(
                    metadata_token,
                )
                if operation_token:
                    FlextContext.Variables.OperationName.reset(
                        operation_token,
                    )

    # =========================================================================
    # Serialization Domain - Context serialization for cross-service communication
    # =========================================================================

    class Serialization:
        """Context serialization and deserialization utilities."""

        @staticmethod
        def get_full_context() -> dict[str, t.ConfigMapValue]:
            """Get current context as dictionary."""
            context_vars = FlextContext.Variables
            return {
                c.Context.KEY_CORRELATION_ID: context_vars.Correlation.CORRELATION_ID.get(),
                c.Context.KEY_PARENT_CORRELATION_ID: context_vars.Correlation.PARENT_CORRELATION_ID.get(),
                c.Context.KEY_SERVICE_NAME: context_vars.Service.SERVICE_NAME.get(),
                c.Context.KEY_SERVICE_VERSION: context_vars.Service.SERVICE_VERSION.get(),
                c.Context.KEY_USER_ID: context_vars.Request.USER_ID.get(),
                c.Context.KEY_OPERATION_NAME: context_vars.Performance.OPERATION_NAME.get(),
                c.Context.KEY_REQUEST_ID: context_vars.Request.REQUEST_ID.get(),
                c.Context.KEY_OPERATION_START_TIME: (
                    st.isoformat()
                    if (st := context_vars.Performance.OPERATION_START_TIME.get())
                    else None
                ),
                c.Context.KEY_OPERATION_METADATA: context_vars.Performance.OPERATION_METADATA.get(),
            }

        @staticmethod
        def get_correlation_context() -> dict[str, str]:
            """Get correlation context for cross-service propagation."""
            context: dict[str, str] = {}

            correlation_id = FlextContext.Variables.CorrelationId.get()
            if correlation_id:
                context[c.Context.HEADER_CORRELATION_ID] = str(correlation_id)

            parent_id = FlextContext.Variables.ParentCorrelationId.get()
            if parent_id:
                context[c.Context.HEADER_PARENT_CORRELATION_ID] = str(parent_id)

            service_name = FlextContext.Variables.ServiceName.get()
            if service_name:
                context[c.Context.HEADER_SERVICE_NAME] = str(service_name)

            return context

        @staticmethod
        def set_from_context(
            context: Mapping[str, t.ConfigMapValue],
        ) -> None:
            """Set context from dictionary (e.g., from HTTP headers)."""
            context_map = m.ConfigMap.model_validate(context)

            # Fast fail: use explicit checks instead of OR fallback
            correlation_id_value = context_map.root.get(c.Context.HEADER_CORRELATION_ID)
            if correlation_id_value is None:
                correlation_id_value = context_map.root.get(
                    c.Context.KEY_CORRELATION_ID
                )
            if correlation_id_value is not None:
                _ = FlextContext.Variables.CorrelationId.set(
                    str(correlation_id_value),
                )

            parent_id_value = context_map.root.get(
                c.Context.HEADER_PARENT_CORRELATION_ID
            )
            if parent_id_value is None:
                parent_id_value = context_map.root.get(
                    c.Context.KEY_PARENT_CORRELATION_ID
                )
            if parent_id_value is not None:
                _ = FlextContext.Variables.ParentCorrelationId.set(
                    str(parent_id_value),
                )

            service_name_value = context_map.root.get(c.Context.HEADER_SERVICE_NAME)
            if service_name_value is None:
                service_name_value = context_map.root.get(c.Context.KEY_SERVICE_NAME)
            if service_name_value is not None:
                _ = FlextContext.Variables.ServiceName.set(str(service_name_value))

            user_id_value = context_map.root.get(c.Context.HEADER_USER_ID)
            if user_id_value is None:
                user_id_value = context_map.root.get(c.Context.KEY_USER_ID)
            if user_id_value is not None:
                _ = FlextContext.Variables.UserId.set(str(user_id_value))

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
                FlextContext.Variables.CorrelationId,
                FlextContext.Variables.ParentCorrelationId,
                FlextContext.Variables.ServiceName,
                FlextContext.Variables.ServiceVersion,
                FlextContext.Variables.UserId,
                FlextContext.Variables.RequestId,
                FlextContext.Variables.OperationName,
            ]:
                _ = context_var.set(None)

            # Clear typed context variables
            _ = FlextContext.Variables.OperationStartTime.set(None)
            _ = FlextContext.Variables.OperationMetadata.set(None)
            _ = FlextContext.Variables.RequestTimestamp.set(None)

            # Note: All variables use structlog as single source (via FlextModels.StructlogProxyContextVar)

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure correlation ID exists, creating one if needed."""
            correlation_id_value = FlextContext.Variables.CorrelationId.get()
            if isinstance(correlation_id_value, str) and correlation_id_value:
                return correlation_id_value
            # Generate new correlation_id and set it in context
            new_correlation_id: str = u.generate("correlation")
            FlextContext.Correlation.set_correlation_id(new_correlation_id)
            return new_correlation_id

        @staticmethod
        def has_correlation_id() -> bool:
            """Check if correlation ID is set in context."""
            return FlextContext.Variables.CorrelationId.get() is not None

        @staticmethod
        def get_context_summary() -> str:
            """Get a human-readable context summary for debugging."""
            context = FlextContext.Serialization.get_full_context()
            parts: list[str] = []

            correlation_id = context.get(c.Context.KEY_CORRELATION_ID)
            if correlation_id:
                parts.append(f"correlation={str(correlation_id)[:8]}...")

            service_name = context.get(c.Context.KEY_SERVICE_NAME)
            if service_name:
                parts.append(f"service={service_name}")

            operation_name = context.get(c.Context.KEY_OPERATION_NAME)
            if operation_name:
                parts.append(f"operation={operation_name}")

            user_id = context.get(c.Context.KEY_USER_ID)
            if user_id:
                parts.append(f"user={user_id}")

            return (
                f"FlextContext({', '.join(parts)})" if parts else "FlextContext(empty)"
            )


__all__: list[str] = [
    "FlextContext",
]
