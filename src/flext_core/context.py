"""Context propagation utilities for dispatcher-coordinated workloads.

FlextContext tracks correlation metadata, request data, and timing information
through the dispatcher pipeline and into handlers, ensuring structured logs and
metrics remain consistent across threads and async boundaries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
from collections.abc import Generator, Mapping, MutableMapping
from contextlib import contextmanager
from datetime import datetime
from typing import Final, Self, overload

from pydantic import BaseModel

from flext_core import FlextLogger, FlextRuntime, c, m, p, r, t, u
from flext_core._models.context import FlextModelsContext

_logger = FlextLogger(__name__)

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
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            _logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
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
                    data=t.Dict(root=m.ConfigMap.model_validate(initial_data).root),
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
                m.ConfigMap(root=context_data.data.root),
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
            initial_data_dict: m.ConfigMap = m.ConfigMap(root={})
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
                initial_data=m.ContextData(data=t.Dict(root=initial_data_dict.root)),
            )
        # Default: use initial_data parameter
        # Auto-generate correlation_id for zero-config setup
        data_map = (
            m.ConfigMap.model_validate(initial_data)
            if initial_data is not None
            else m.ConfigMap(root={})
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
                    data=t.Dict(root=initial_data_dict_new.root),
                ),
            )
        return cls(initial_data=m.ContextData(data=t.Dict(root=data_map.root)))

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
            self._narrow_contextvar_to_configuration_dict(ctx_var.get()),
        )
        updated = current.model_copy()
        updated.update(data.root)
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
            FlextContext._narrow_contextvar_to_configuration_dict(value),
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
                update={"operations": operations},
            )

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
            updated.update(data.root)
            _ = ctx_var.set(updated)
            self._update_statistics(c.Context.OPERATION_SET)
            self._execute_hooks(
                c.Context.OPERATION_SET,
                m.ConfigMap(root={"data": m.ConfigMap(root=data.root)}),
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

        def normalize_plain(raw_value: t.ConfigMapValue) -> t.ConfigMapValue:
            mapped_value = FlextRuntime.normalize_to_general_value(raw_value)
            try:
                normalized_map = m.ConfigMap.model_validate(mapped_value)
            except (TypeError, ValueError, AttributeError) as exc:
                _logger.debug(
                    "Context value is not a valid ConfigMap on first validation pass",
                    exc_info=exc,
                )
                root_value = getattr(mapped_value, "root", None)
                try:
                    normalized_map = m.ConfigMap.model_validate(root_value)
                except (TypeError, ValueError, AttributeError) as root_exc:
                    _logger.debug(
                        "Context value root payload is not a valid ConfigMap",
                        exc_info=root_exc,
                    )
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
                current,
                lambda k, _v: k != key,
            )
            _ = ctx_var.set(m.ConfigMap(root=filtered))
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
            _ = ctx_var.set(m.ConfigMap(root={}))

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
                update={"operations": operations},
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
            except (TypeError, ValueError, AttributeError) as exc:
                _logger.debug(
                    "Context export payload validation failed",
                    exc_info=exc,
                )
                exported_map = m.ConfigMap(root={})

            for scope_name, scope_payload in exported_map.items():
                if scope_name not in {
                    c.Context.SCOPE_GLOBAL,
                    c.Context.SCOPE_USER,
                    c.Context.SCOPE_SESSION,
                }:
                    continue
                try:
                    scope_data = m.ConfigMap.model_validate(scope_payload)
                except (TypeError, ValueError, AttributeError) as exc:
                    _logger.debug(
                        "Context scope payload validation failed",
                        exc_info=exc,
                    )
                    scope_data = None
                if scope_data is not None:
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
                    m.ConfigMap.model_validate(scope_dict),
                    scope_name,
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
                    ctx_var.get(),
                )
            except TypeError as e:
                return r[bool].fail(str(e))
            for key in scope_dict:
                if not key:
                    return r[bool].fail("Invalid key found in context")
        return r[bool].ok(value=True)

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
        all_data: m.ConfigMap = m.ConfigMap(root={})

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
                    FlextRuntime.normalize_to_metadata_value(metadata_value),
                )
            metadata_for_model = m.ConfigMap(root=normalized_metadata_map)

        # Create ContextExport model
        # statistics expects ContextMetadataMapping (Mapping[str, ContextValue])
        statistics_mapping: t.Dict = t.Dict(
            root=dict((stats_dict_export or m.ConfigMap(root={})).items()),
        )

        # Return as dict if requested
        if as_dict:
            result_dict: dict[str, t.ConfigMapValue] = dict(all_scopes.items())
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
                },
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
                },
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
            update={"attributes": updated_attributes},
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
                m.ConfigMap.model_validate(custom_fields_raw).items(),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            _logger.debug("Custom metadata field normalization failed", exc_info=exc)
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
        """Set the global container instance."""
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

    # =========================================================================
    # Service Domain - Service identification and lifecycle context
    # =========================================================================

    class Service:
        """Service identification and lifecycle context management utilities."""

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

    # =========================================================================
    # Performance Domain - Operation timing and performance tracking
    # =========================================================================

    class Performance:
        """Performance monitoring and timing context management utilities."""

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
                },
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


__all__: list[str] = [
    "FlextContext",
]
