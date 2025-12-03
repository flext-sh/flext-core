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
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Final, Self, cast

from pydantic import BaseModel

from flext_core._models.base import FlextModelsBase
from flext_core._models.context import FlextModelsContext
from flext_core.constants import c
from flext_core.loggings import FlextLogger
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t
from flext_core.utilities import u


class FlextContext:  # noqa: PLR0904
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

    # =========================================================================
    # HELPER METHODS (moved from module level for SRP compliance)
    # =========================================================================

    # Instance attributes
    # Using direct import for mypy compatibility with nested class aliases
    _metadata: FlextModelsContext.ContextMetadata

    # NOTE: _scope_vars is an instance attribute (see __init__)
    # No property accessor needed - direct access via self._scope_vars

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def __init__(
        self,
        initial_data: FlextModelsContext.ContextData
        | dict[str, t.GeneralValueType]
        | None = None,
    ) -> None:
        """Initialize FlextContext with optional initial data.

        ARCHITECTURAL NOTE: FlextContext now uses Python's contextvars for storage,
        completely independent of structlog. It delegates to FlextLogger for logging
        integration, maintaining clear separation of concerns.

        Args:
            initial_data: Optional `FlextModelsContext.ContextData` instance or dict

        """
        super().__init__()
        # Use Pydantic directly - NO redundant helpers (Pydantic validates dict/None/model)
        # Type narrowing: always create FlextModelsContext.ContextData instance
        if u.guard(initial_data, dict, return_value=True) is not None:
            # Use u.transform to normalize dict values
            transform_result = u.transform(
                initial_data,
                normalize=True,
                strip_none=False,
            )
            normalized_data = (
                transform_result.value
                if transform_result.is_success
                else {
                    str(k): FlextRuntime.normalize_to_general_value(v)
                    for k, v in initial_data.items()
                }
            )
            context_data = FlextModelsContext.ContextData(data=normalized_data)
        elif initial_data is not None:
            # Already a ContextData instance (not dict, not None)
            context_data = initial_data
        else:
            # None or uninitialized - create empty ContextData
            context_data = FlextModelsContext.ContextData()
        # Initialize context-specific metadata (separate from ContextData.metadata)
        # ContextData.metadata = generic creation/modification metadata (FlextModelsBase.Metadata)
        # FlextContext._metadata = context-specific tracing metadata (ContextMetadata)
        self._metadata = FlextModelsContext.ContextMetadata()

        self._hooks: dict[str, list[t.HandlerAliases.HandlerCallable]] = {}
        self._statistics: FlextModelsContext.ContextStatistics = (
            FlextModelsContext.ContextStatistics()
        )
        self._active = True
        self._suspended = False

        # Create instance-specific contextvars for isolation (required for clone())
        # Note: Each instance gets its own contextvars to prevent state sharing
        # Note: Using None default per B039 - mutable defaults cause issues
        self._scope_vars: dict[
            str,
            contextvars.ContextVar[dict[str, t.GeneralValueType] | None],
        ] = {
            c.Context.SCOPE_GLOBAL: contextvars.ContextVar(
                "flext_global_context",
                default=None,
            ),
            "user": contextvars.ContextVar(
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
        # Type narrowing: context_data is always FlextModelsContext.ContextData at this point
        if context_data.data:
            # Set initial data in global context
            self._set_in_contextvar(
                c.Context.SCOPE_GLOBAL,
                context_data.data,
            )

    # =========================================================================
    # PRIVATE HELPERS - Context variable management and FlextLogger delegation
    # =========================================================================

    def _get_or_create_scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[dict[str, t.GeneralValueType] | None]:
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
    ) -> dict[
        str,
        contextvars.ContextVar[dict[str, t.GeneralValueType] | None],
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
        data: dict[str, t.GeneralValueType],
    ) -> None:
        """Set multiple values in contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        current = cast(
            "dict[str, t.GeneralValueType]",
            u.ensure(ctx_var.get(), target_type="dict", default={}),
        )
        _ = ctx_var.set({**current, **data})
        if scope == c.Context.SCOPE_GLOBAL:
            _ = FlextLogger.bind_global_context(**data)

    def _get_from_contextvar(
        self,
        scope: str,
    ) -> dict[str, t.GeneralValueType]:
        """Get all values from contextvar scope."""
        return cast(
            "dict[str, t.GeneralValueType]",
            u.ensure(
                self._get_or_create_scope_var(scope).get(),
                target_type="dict",
                default={},
            ),
        )

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper).

        Args:
            operation: Operation name ('set', 'get', 'remove', etc.)

        """
        # Update primary counter using attribute name
        counter_attr = f"{operation}s"
        if hasattr(self._statistics, counter_attr):
            current_value = getattr(self._statistics, counter_attr, 0)
            if u.guard(current_value, int).is_success:
                setattr(self._statistics, counter_attr, current_value + 1)

        # Update operations dict if exists
        if operation in self._statistics.operations:
            value = self._statistics.operations[operation]
            if u.guard(value, int).is_success:
                self._statistics.operations[operation] = value + 1

    def _add_hook(
        self,
        event: str,
        hook: t.HandlerAliases.HandlerCallable,
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
        hook: t.HandlerAliases.HandlerCallable,
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
        event_data: t.GeneralValueType,
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
                # Any exception indicates a programming error in hook implementation
                _ = hook(event_data)

    @staticmethod
    def _propagate_to_logger(
        key: str,
        value: t.GeneralValueType,
        scope: str,
    ) -> None:
        """Propagate context changes to FlextLogger (DRY helper).

        Args:
            key: Context key
            value: Context value
            scope: Context scope

        """
        if scope == c.Context.SCOPE_GLOBAL:
            _ = FlextLogger.bind_global_context(**{key: value})

    # =========================================================================
    # Instance Methods - Core context operations
    # =========================================================================

    @staticmethod
    def _validate_set_inputs(
        key: str,
        value: t.GeneralValueType,
    ) -> p.ResultProtocol[bool]:
        """Validate inputs for set operation.

        Args:
            key: The key to validate
            value: The value to validate

        Returns:
            FlextResult[bool]: Success with True if valid, failure with error message

        """
        if not key:
            return FlextRuntime.result_fail("Key must be a non-empty string")
        if value is None:
            return FlextRuntime.result_fail("Value cannot be None")
        if (
            u.guard(value, (str, int, float, bool, list, dict), return_value=True)
            is None
        ):
            return FlextRuntime.result_fail("Value must be serializable")
        return FlextRuntime.result_ok(True)

    def set(
        self,
        key: str,
        value: t.GeneralValueType,
        scope: str = c.Context.SCOPE_GLOBAL,
    ) -> p.ResultProtocol[bool]:
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
            return FlextRuntime.result_fail("Context is not active")

        validation_result = FlextContext._validate_set_inputs(key, value)
        if validation_result.is_failure:
            return FlextRuntime.result_fail(
                validation_result.error or "Validation failed"
            )

        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            _ = ctx_var.set({**current, key: value})
            FlextContext._propagate_to_logger(key, value, scope)
            self._update_statistics("set")
            self._execute_hooks("set", {"key": key, "value": value})
            return FlextRuntime.result_ok(True)
        except (TypeError, Exception) as e:
            error_msg = (
                str(e)
                if isinstance(e, TypeError)
                else f"Failed to set context value: {e}"
            )
            return FlextRuntime.result_fail(error_msg)

    def get(
        self,
        key: str,
        scope: str = c.Context.SCOPE_GLOBAL,
    ) -> p.ResultProtocol[t.GeneralValueType]:
        """Get a value from the context.

        Fast fail: Returns FlextResult[t.GeneralValueType] - fails if key not found.
        No fallback behavior - use FlextResult monadic operations for defaults.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).
        No longer checks structlog - FlextLogger is independent.

        Args:
            key: The key to get
            scope: The scope to get from (global, user, session)

        Returns:
            FlextResult[t.GeneralValueType]: Success with value, or failure if key not found

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
            return FlextRuntime.result_fail(
                "Context is not active",
            )

        # Get from contextvar (single source of truth)
        scope_data = self._get_from_contextvar(scope)

        if key not in scope_data:
            return FlextRuntime.result_fail(
                f"Context key '{key}' not found in scope '{scope}'",
            )

        value = scope_data[key]

        # Update statistics
        self._update_statistics("get")

        # Handle None values - return failure since FlextResult.ok() cannot accept None
        if value is None:
            return FlextRuntime.result_fail(
                f"Context key '{key}' has None value in scope '{scope}'",
            )

        return FlextRuntime.result_ok(value)

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
        current = cast(
            "dict[str, t.GeneralValueType]",
            u.ensure(ctx_var.get(), target_type="dict", default={}),
        )
        if key in current:
            # Use filter_dict for concise key removal
            filtered = cast(
                "dict[str, t.GeneralValueType]",
                u.filter(current, lambda k, _: k != key),
            )
            _ = ctx_var.set(filtered)
            # Note: ContextVar.set() already cleared the key, no need to unbind from logger
            # FlextLogger doesn't have unbind_global_context method
            self._update_statistics("remove")

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
            _ = ctx_var.set({})  # Set to empty dict, not None

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == c.Context.SCOPE_GLOBAL:
                _ = FlextLogger.clear_global_context()

        # Reset metadata model (formerly in cleanup())
        self._metadata = FlextModelsContext.ContextMetadata()

        # Update statistics using model (type-safe, no .get() needed)
        self._statistics.clears += 1
        if "clear" in self._statistics.operations:
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
        for ctx_var in self._scope_vars.values():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            all_keys.update(scope_dict.keys())
        return list(all_keys)

    def merge(
        self,
        other: FlextContext | dict[str, t.GeneralValueType],
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

        if isinstance(other, FlextContext):
            # Merge all scopes from the other context
            # Iterate through all scope variables in other context
            for scope_name, other_ctx_var in other.iter_scope_vars().items():
                # Get scope data from other context
                other_scope_dict = cast(
                    "dict[str, t.GeneralValueType]",
                    u.ensure(other_ctx_var.get(), target_type="dict", default={}),
                )
                if other_scope_dict:
                    # Merge into this context's scope
                    ctx_var = self._get_or_create_scope_var(scope_name)
                    current_dict = cast(
                        "dict[str, t.GeneralValueType]",
                        u.ensure(ctx_var.get(), target_type="dict", default={}),
                    )
                    # Merge scope_data into current_dict
                    merge_result = u.merge(
                        current_dict, other_scope_dict, strategy="deep"
                    )
                    if merge_result.is_success:
                        current_dict = merge_result.value
                    _ = ctx_var.set(current_dict)

                    # DELEGATION: Propagate global scope to FlextLogger
                    if scope_name == c.Context.SCOPE_GLOBAL:
                        # Use current_dict which already has merged data
                        _ = FlextLogger.bind_global_context(**current_dict)
        else:
            # Merge dictionary into global scope
            self._set_in_contextvar(c.Context.SCOPE_GLOBAL, other)

        return self

    def clone(self) -> FlextContext:
        """Create a clone of this context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            A new FlextContext with the same data

        """
        cloned = FlextContext()
        for scope_name, ctx_var in self._scope_vars.items():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            if scope_dict:
                _ = cloned._get_or_create_scope_var(scope_name).set(scope_dict.copy())
        # Clone metadata and statistics
        cloned._metadata = self._metadata.model_copy()
        cloned._statistics = self._statistics.model_copy()

        return cloned

    def validate(self) -> p.ResultProtocol[bool]:
        """Validate the context data.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            FlextResult[bool]: Success with True if valid, failure with error details

        """
        if not self._active:
            return FlextRuntime.result_fail("Context is not active")
        for ctx_var in self._scope_vars.values():
            try:
                scope_dict = cast(
                    "dict[str, t.GeneralValueType]",
                    u.ensure(ctx_var.get(), target_type="dict", default={}),
                )
            except TypeError as e:
                return FlextRuntime.result_fail(str(e))
            for key in scope_dict:
                if not key:
                    return FlextRuntime.result_fail("Invalid key found in context")
        return FlextRuntime.result_ok(True)

    def to_json(self) -> str:
        """Convert context to JSON string.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            JSON string representation of the context

        """
        all_data: dict[str, t.GeneralValueType] = {}
        for ctx_var in self._scope_vars.values():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            merge_result = u.merge(all_data, scope_dict, strategy="deep")
            if merge_result.is_success:
                all_data = merge_result.value
        return json.dumps(all_data, default=str)

    @classmethod
    def create(
        cls,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: Mapping[str, t.GeneralValueType] | None = None,
    ) -> FlextContext:
        """Create context with operation and user metadata.

        Factory method for creating FlextContext instances with common metadata.

        Args:
            operation_id: Optional operation identifier
            user_id: Optional user identifier
            metadata: Optional additional metadata dictionary

        Returns:
            New FlextContext instance with provided metadata

        """
        initial_data: dict[str, t.GeneralValueType] = {}
        if operation_id is not None:
            initial_data["operation_id"] = operation_id
        if user_id is not None:
            initial_data["user_id"] = user_id
        if metadata is not None:
            merge_result = u.merge(initial_data, metadata, strategy="deep")
            if merge_result.is_success:
                initial_data = merge_result.value
        return cls(initial_data=initial_data or None)

    @classmethod
    def from_json(cls, json_str: str) -> FlextContext:
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
            data = json.loads(json_str)
            if u.guard(data, dict, return_value=True) is None:
                msg = f"JSON must represent a dict, got {type(data).__name__}"
                raise TypeError(msg)
            # Use u.map to normalize each value in dict to ensure GeneralValueType compatibility
            normalized_data = u.map(
                data,
                mapper=lambda _k, v: FlextRuntime.normalize_to_general_value(v),
            )
            context_data = FlextModelsContext.ContextData(data=normalized_data)
            return cls(initial_data=context_data)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON string: {e}"
            raise ValueError(msg) from e

    def _import_data(
        self,
        data: dict[str, t.GeneralValueType],
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
        # Normalize each value in dict to ensure GeneralValueType compatibility
        normalized_data: dict[str, t.GeneralValueType] = {}
        for k, v in data.items():
            normalized_data[str(k)] = FlextRuntime.normalize_to_general_value(v)
        # Merge into global scope
        self._set_in_contextvar(
            c.Context.SCOPE_GLOBAL,
            normalized_data,
        )

    def items(self) -> list[tuple[str, t.GeneralValueType]]:
        """Get all items (key-value pairs) in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of (key, value) tuples across all scopes

        """
        if not self._active:
            return []
        all_items: list[tuple[str, t.GeneralValueType]] = []
        for ctx_var in self._scope_vars.values():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            all_items.extend(scope_dict.items())
        return all_items

    def export(
        self,
        *,
        include_statistics: bool = False,
        include_metadata: bool = False,
        as_dict: bool = True,
    ) -> FlextModelsContext.ContextExport | dict[str, t.GeneralValueType]:
        """Export context data for serialization or debugging.

        Args:
            include_statistics: Include context statistics
            include_metadata: Include context metadata
            as_dict: If True, return as dict instead of ContextExport model

        Returns:
            ContextExport model or dict with all requested data

        """
        all_data: dict[str, t.GeneralValueType] = {}
        stats_dict: dict[str, t.GeneralValueType] | None = None
        metadata_dict: dict[str, t.GeneralValueType] | None = None

        # Collect all scope data
        all_scopes = self._get_all_scopes()
        all_data = dict(all_scopes)

        # Collect statistics if requested
        if include_statistics and self._statistics:
            stats_dict = self._statistics.model_dump()

        # Collect metadata if requested
        if include_metadata:
            metadata_dict = self._get_all_metadata()

        # Normalize metadata_dict values to t.MetadataAttributeValue
        normalized_metadata: dict[str, t.MetadataAttributeValue] | None = None
        if metadata_dict:
            normalized_metadata = {
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in metadata_dict.items()
            }

        # Create ContextExport model
        # statistics expects ContextMetadataMapping (Mapping[str, GeneralValueType])
        statistics_mapping: t.Types.ContextMetadataMapping = stats_dict or {}

        # Return as dict if requested
        if as_dict:
            result_dict: dict[str, t.GeneralValueType] = all_data.copy()
            if include_statistics and stats_dict:
                result_dict["statistics"] = stats_dict
            if include_metadata and metadata_dict:
                result_dict["metadata"] = metadata_dict
            return result_dict

        return FlextModelsContext.ContextExport(
            data=all_data,
            metadata=FlextModelsBase.Metadata(attributes=normalized_metadata)
            if normalized_metadata
            else None,
            statistics=statistics_mapping,
        )

    def values(self) -> list[t.GeneralValueType]:
        """Get all values in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of all values across all scopes

        """
        if not self._active:
            return []
        all_values: list[t.GeneralValueType] = []
        for ctx_var in self._scope_vars.values():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
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

    def _cleanup(self) -> None:
        """Cleanup context (alias for clear for backward compatibility).

        ARCHITECTURAL NOTE: Delegates to clear() for consistency.

        """
        self.clear()

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
            _ = ctx_var.set({})  # Set to empty dict, not None

            # DELEGATION: Clear FlextLogger for global scope
            if scope_name == c.Context.SCOPE_GLOBAL:
                _ = FlextLogger.clear_global_context()

        # Clear metadata and hooks
        self._metadata = FlextModelsContext.ContextMetadata()  # Reset model
        self._hooks.clear()

    def set_metadata(self, key: str, value: t.GeneralValueType) -> None:
        """Set metadata for the context.

        Args:
            key: The metadata key
            value: The metadata value

        """
        # Directly update custom_fields dict to avoid deprecation warning
        # and object recreation
        self._metadata.custom_fields[key] = value

    def get_metadata(self, key: str) -> p.ResultProtocol[t.GeneralValueType]:
        """Get metadata from the context.

        Fast fail: Returns FlextResult[t.GeneralValueType] - fails if key not found.
        No fallback behavior - use FlextResult monadic operations for defaults.

        Args:
            key: The metadata key

        Returns:
            FlextResult[t.GeneralValueType]: Success with metadata value, or failure if key not found

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
        custom_fields = (
            self._metadata.custom_fields
            if FlextRuntime.is_dict_like(self._metadata.custom_fields)
            else {}
        )

        if key not in custom_fields:
            return FlextRuntime.result_fail(
                f"Metadata key '{key}' not found",
            )

        # Use FlextRuntime.normalize_to_general_value directly - no wrapper needed
        value = custom_fields[key]
        normalized_value = FlextRuntime.normalize_to_general_value(value)
        return FlextRuntime.result_ok(normalized_value)

    def _get_all_data(self) -> dict[str, t.GeneralValueType]:
        """Get all data from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary of all context data across all scopes

        """
        all_data: dict[str, t.GeneralValueType] = {}
        for ctx_var in self._scope_vars.values():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            all_data.update(scope_dict)
        return all_data

    def _get_statistics(self) -> FlextModelsContext.ContextStatistics:
        """Get context statistics.

        Returns:
            ContextStatistics model with operation counts

        """
        return self._statistics

    def _get_all_metadata(self) -> dict[str, t.GeneralValueType]:
        """Get all metadata from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.
        Custom fields are flattened into the top-level dict for easy access.

        Returns:
            Dictionary of all metadata (with custom_fields flattened)

        """
        # Convert Pydantic model to dict
        data = self._metadata.model_dump()
        # Extract and flatten custom_fields into result
        custom_fields = data.pop("custom_fields", {}) or {}
        result: dict[str, t.GeneralValueType] = {
            k: v for k, v in data.items() if v is not None and v != {}
        }
        # Merge custom_fields at top level (custom_fields take precedence)
        result.update(custom_fields)
        return result

    def _get_all_scopes(self) -> dict[str, dict[str, t.GeneralValueType]]:
        """Get all scope registrations.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary mapping scope names to their data dictionaries

        """
        if not self._active:
            return {}
        scopes: dict[str, dict[str, t.GeneralValueType]] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_dict = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(ctx_var.get(), target_type="dict", default={}),
            )
            if scope_dict:
                scopes[scope_name] = scope_dict
        return scopes

    def _export_snapshot(self) -> FlextModelsContext.ContextExport:
        """Export context snapshot.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            ContextExport model with complete context state

        """
        # Get all data
        all_data = self._get_all_data()

        # Get metadata as dict
        metadata_dict = self._get_all_metadata()

        # Normalize metadata_dict values to t.MetadataAttributeValue
        normalized_metadata: dict[str, t.MetadataAttributeValue] | None = None
        if metadata_dict:
            normalized_metadata = {}
            for k, v in metadata_dict.items():
                # Normalize GeneralValueType to t.MetadataAttributeValue
                normalized_metadata[k] = FlextRuntime.normalize_to_metadata_value(v)

        # Get statistics as dict
        stats_dict: dict[str, t.GeneralValueType] = {}
        if hasattr(self._statistics, "model_dump"):
            stats_dict = self._statistics.model_dump()

        # Create ContextExport model
        # statistics expects ContextMetadataMapping (Mapping[str, GeneralValueType])
        statistics_mapping: t.Types.ContextMetadataMapping = stats_dict or {}
        return FlextModelsContext.ContextExport(
            data=all_data,
            metadata=FlextModelsBase.Metadata(attributes=normalized_metadata)
            if normalized_metadata
            else None,
            statistics=statistics_mapping,
        )

    # =========================================================================
    # Container integration for dependency injection
    # =========================================================================

    _container: p.ContainerProtocol | None = None

    @classmethod
    def get_container(cls) -> p.ContainerProtocol:
        """Get global container with lazy initialization.

        Returns:
            Global ContainerProtocol instance for dependency injection

        Example:
            >>> container = FlextContext.get_container()
            >>> result = container.get("service_name")

        """
        if cls._container is None:
            cls._container = FlextRuntime.create_container()
        return cls._container

    # ==========================================================================
    # Variables - Context Variables using structlog as Single Source of Truth
    # ==========================================================================

    class Variables:
        """Context variables using structlog as single source of truth."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.Context.create_str_proxy(
                    "correlation_id",
                    default=None,
                )
            )
            PARENT_CORRELATION_ID: Final[
                FlextModelsContext.StructlogProxyContextVar[str]
            ] = u.Context.create_str_proxy(
                "parent_correlation_id",
                default=None,
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.Context.create_str_proxy(
                    "service_name",
                    default=None,
                )
            )
            SERVICE_VERSION: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.Context.create_str_proxy(
                    "service_version",
                    default=None,
                )
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.Context.create_str_proxy(
                    "user_id",
                    default=None,
                )
            )
            REQUEST_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.Context.create_str_proxy(
                    "request_id",
                    default=None,
                )
            )
            REQUEST_TIMESTAMP: Final[
                FlextModelsContext.StructlogProxyContextVar[datetime]
            ] = u.Context.create_datetime_proxy(
                "request_timestamp",
                default=None,
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[FlextModelsContext.StructlogProxyContextVar[str]] = (
                u.Context.create_str_proxy(
                    "operation_name",
                    default=None,
                )
            )
            OPERATION_START_TIME: Final[
                FlextModelsContext.StructlogProxyContextVar[datetime]
            ] = u.Context.create_datetime_proxy("operation_start_time", default=None)
            OPERATION_METADATA: Final[
                FlextModelsContext.StructlogProxyContextVar[
                    dict[str, t.GeneralValueType]
                ]
            ] = u.Context.create_dict_proxy(
                "operation_metadata",
                default=None,
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
        def set_correlation_id(correlation_id: str | None) -> None:
            """Set correlation ID.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            Accepts ``None`` to explicitly clear the active correlation when needed.
            """
            _ = FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

        @staticmethod
        def reset_correlation_id() -> None:
            """Clear correlation ID from context variables."""
            _ = FlextContext.Variables.Correlation.CORRELATION_ID.set(None)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate unique correlation ID.

            Note: Uses u.Generators.generate_correlation_id() for ID generation.
            Sets the correlation ID in context variables (via FlextModels.StructlogProxyContextVar).
            """
            correlation_id = u.generate("correlation")
            _ = FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID."""
            value = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()
            return value if isinstance(value, str) else None

        @staticmethod
        def set_parent_correlation_id(parent_id: str) -> None:
            """Set parent correlation ID."""
            _ = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

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
            _ = FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

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
            _ = FlextContext.Variables.Service.SERVICE_VERSION.set(version)

        @staticmethod
        def get_service(
            service_name: str,
        ) -> p.ResultProtocol[t.GeneralValueType]:
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
            # get_container is a classmethod on FlextContext, access via class
            container = FlextContext.get_container()
            # Protocol returns ResultProtocol[T]
            result: p.ResultProtocol[t.GeneralValueType] = container.get(service_name)
            return result

        @staticmethod
        def register_service(
            service_name: str,
            service: t.GeneralValueType | BaseModel,
        ) -> p.ResultProtocol[bool]:
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
            # get_container is a classmethod on FlextContext, access via class
            container = FlextContext.get_container()
            try:
                # Cast service to FlexibleValue for protocol compatibility
                # FlextContainer.with_service accepts GeneralValueType internally
                service_flexible = cast("t.FlexibleValue", service)
                _ = container.with_service(service_name, service_flexible)
                # Import locally to avoid circular dependency

                return FlextRuntime.result_ok(True)
            except ValueError as e:
                return FlextRuntime.result_fail(str(e))

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
            _ = FlextContext.Variables.Request.USER_ID.set(user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context."""
            return FlextContext.Variables.Performance.OPERATION_NAME.get()

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context."""
            return FlextContext.Variables.Request.REQUEST_ID.get()

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.Request.REQUEST_ID.set(request_id)

        @staticmethod
        @contextmanager
        def request_context(
            *,
            user_id: str | None = None,
            operation_name: str | None = None,
            request_id: str | None = None,
            metadata: dict[str, t.GeneralValueType] | None = None,
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
                start_time = u.Generators.generate_datetime_utc()
            _ = FlextContext.Variables.Performance.OPERATION_START_TIME.set(start_time)

        @staticmethod
        def get_operation_metadata() -> dict[str, t.GeneralValueType] | None:
            """Get operation metadata from context."""
            return FlextContext.Variables.Performance.OPERATION_METADATA.get()

        @staticmethod
        def set_operation_metadata(
            metadata: dict[str, t.GeneralValueType],
        ) -> None:
            """Set operation metadata in context."""
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)

        @staticmethod
        def add_operation_metadata(
            key: str,
            value: t.GeneralValueType,
        ) -> None:
            """Add single metadata entry to operation context."""
            current_metadata = cast(
                "dict[str, t.GeneralValueType]",
                u.ensure(
                    FlextContext.Variables.Performance.OPERATION_METADATA.get(),
                    target_type="dict",
                    default={},
                ),
            )
            current_metadata[key] = value
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.set(
                current_metadata
            )

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[dict[str, t.GeneralValueType]]:
            """Create timed operation context with performance tracking."""
            start_time = u.Generators.generate_datetime_utc()
            operation_metadata: dict[str, t.GeneralValueType] = {
                "start_time": start_time.isoformat(),
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
                end_time = u.Generators.generate_datetime_utc()
                duration = (end_time - start_time).total_seconds()
                operation_metadata.update(
                    {
                        "end_time": end_time.isoformat(),
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
        def get_full_context() -> dict[str, t.GeneralValueType | None]:
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
                "operation_start_time": (
                    st.isoformat()
                    if (st := context_vars.Performance.OPERATION_START_TIME.get())
                    else None
                ),
                "operation_metadata": context_vars.Performance.OPERATION_METADATA.get(),
            }

        @staticmethod
        def get_correlation_context() -> dict[str, str | None]:
            """Get correlation context for cross-service propagation."""
            context: dict[str, str | None] = {}

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
        def set_from_context(
            context: Mapping[str, t.GeneralValueType],
        ) -> None:
            """Set context from dictionary (e.g., from HTTP headers)."""
            # Fast fail: use explicit checks instead of OR fallback
            correlation_id_value = context.get("X-Correlation-Id")
            if correlation_id_value is None:
                correlation_id_value = context.get("correlation_id")
            if correlation_id_value is not None and isinstance(
                correlation_id_value,
                str,
            ):
                _ = FlextContext.Variables.Correlation.CORRELATION_ID.set(
                    correlation_id_value,
                )

            parent_id_value = context.get("X-Parent-Correlation-Id")
            if parent_id_value is None:
                parent_id_value = context.get("parent_correlation_id")
            if parent_id_value is not None and isinstance(parent_id_value, str):
                _ = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                    parent_id_value,
                )

            service_name_value = context.get("X-Service-Name")
            if service_name_value is None:
                service_name_value = context.get("service_name")
            if service_name_value is not None and isinstance(service_name_value, str):
                _ = FlextContext.Variables.Service.SERVICE_NAME.set(service_name_value)

            user_id_value = context.get("X-User-Id")
            if user_id_value is None:
                user_id_value = context.get("user_id")
            if user_id_value is not None and isinstance(user_id_value, str):
                _ = FlextContext.Variables.Request.USER_ID.set(user_id_value)

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
                _ = context_var.set(None)

            # Clear typed context variables
            _ = FlextContext.Variables.Performance.OPERATION_START_TIME.set(None)
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.set(None)
            _ = FlextContext.Variables.Request.REQUEST_TIMESTAMP.set(None)

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
