"""Context propagation utilities for dispatcher-coordinated workloads.

FlextContext tracks correlation metadata, request data, and timing information
through the dispatcher pipeline and into handlers, ensuring structured logs and
metrics remain consistent across threads and async boundaries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
import time
from collections.abc import (
    Generator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Final, Self, overload

from pydantic import BaseModel, Field, PrivateAttr

from flext_core import FlextLogger, c, m, p, r, t, u


class FlextContext(m.ArbitraryTypesModel):
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

    _logger: ClassVar[p.Logger] = FlextLogger.get_logger(__name__)

    @staticmethod
    def _to_normalized(value: t.ValueOrModel) -> t.RecursiveContainer:
        """Narrow ``Container | BaseModel`` to ``RecursiveContainer``.

        BaseModel instances are converted via ``model_dump()`` so the result
        is always a plain ``RecursiveContainer`` (no BaseModel).
        """
        if u.is_pydantic_model(value):
            raw = value.model_dump()
            result: t.MutableContainerMapping = {}
            for k, v in raw.items():
                container_val = u.normalize_to_container(v)
                if u.is_pydantic_model(container_val):
                    result[str(k)] = str(container_val)
                else:
                    result[str(k)] = container_val
            return result
        return value

    @staticmethod
    def _empty_hooks() -> t.ContextHookMap:
        return {}

    initial_data: Annotated[
        m.ContextData | t.ConfigMap | None,
        Field(default=None, description="Initial data for context scopes."),
    ]

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: t.ConfigMap | t.ContainerMapping | BaseModel | None,
    ) -> t.ContainerMapping:
        """Return contextvar payload as ConfigMap with safe default."""
        if ctx_value is None:
            empty: t.ContainerMapping = {}
            return empty

        payload: Mapping[str, t.ValueOrModel] | t.ContainerMapping
        if isinstance(ctx_value, (t.ConfigMap, t.Dict)):
            payload = ctx_value.root
        elif u.is_pydantic_model(ctx_value):
            payload = ctx_value.model_dump()
        elif u.is_mapping(ctx_value):
            payload = ctx_value
        else:
            empty_fallback: t.ContainerMapping = {}
            return empty_fallback

        try:
            normalized: t.MutableContainerMapping = {}
            mapping_value: Mapping[str, t.ValueOrModel] = dict(
                payload.items(),
            )
            for key, value in mapping_value.items():
                if str(key) != key:
                    empty_key: t.ContainerMapping = {}
                    return empty_key
                if value is None:
                    normalized[key] = None
                    continue
                normalized_value = u.normalize_to_container(value)
                if u.is_pydantic_model(normalized_value):
                    metadata_normalized = u.normalize_to_container(
                        u.normalize_to_metadata(normalized_value),
                    )
                    if isinstance(metadata_normalized, (*t.CONTAINER_TYPES,)):
                        normalized[key] = metadata_normalized
                    else:
                        normalized[key] = str(metadata_normalized)
                else:
                    normalized[key] = normalized_value
            return normalized
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            FlextContext._logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
            empty_err: t.ContainerMapping = {}
            return empty_err

    _metadata: m.Metadata = PrivateAttr()
    _hooks: t.ContextHookMap = PrivateAttr(
        default_factory=_empty_hooks,
    )
    _statistics: m.ContextStatistics = PrivateAttr(
        default_factory=lambda: m.ContextStatistics(),
    )
    _active: bool = PrivateAttr(default=True)
    _suspended: bool = PrivateAttr(default=False)
    _scope_vars: MutableMapping[str, contextvars.ContextVar[t.ConfigMap | None]] = (
        PrivateAttr()
    )

    def __init__(self, **data: t.ValueOrModel) -> None:
        """Initialize FlextContext with optional initial data.

        ARCHITECTURAL NOTE: FlextContext now uses Python's contextvars for storage,
        completely independent of structlog. It delegates to FlextLogger for logging
        integration, maintaining clear separation of concerns.

        """
        super().__init__(**data)
        context_data = m.ContextData()
        if self.initial_data is not None:
            if isinstance(self.initial_data, m.ContextData):
                context_data = self.initial_data
            else:
                context_data = m.ContextData(
                    data=t.Dict(
                        root=dict(self.initial_data),
                    ),
                )
        self._metadata = m.Metadata()
        self._hooks = {}
        self._statistics = m.ContextStatistics()
        self._active = True
        self._suspended = False
        self._scope_vars = {
            c.ContextScope.GLOBAL: contextvars.ContextVar(
                "flext_global_context",
                default=None,
            ),
            c.ContextScope.USER: contextvars.ContextVar(
                "flext_user_context",
                default=None,
            ),
            c.ContextScope.SESSION: contextvars.ContextVar(
                "flext_session_context",
                default=None,
            ),
        }
        if context_data.data:
            self._set_in_contextvar(
                c.ContextScope.GLOBAL,
                t.ConfigMap(root=context_data.data.root),
            )

    @overload
    @classmethod
    def create(cls, initial_data: t.ConfigMap | None = None) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: t.ConfigMap | None = None,
    ) -> Self: ...

    @classmethod
    def create(
        cls,
        initial_data: t.ConfigMap | None = None,
        *,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: t.ConfigMap | None = None,
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
        if operation_id is not None or user_id is not None or metadata is not None:
            initial_data_dict: t.ConfigMap = t.ConfigMap(root={})
            if operation_id is not None:
                initial_data_dict[c.ContextKey.OPERATION_ID] = operation_id
            elif auto_correlation_id:
                initial_data_dict[c.ContextKey.OPERATION_ID] = u.generate(
                    "correlation",
                )
            if user_id is not None:
                initial_data_dict[c.ContextKey.USER_ID] = user_id
            if metadata is not None:
                initial_data_dict.update(dict(metadata))
            return cls(
                initial_data=m.ContextData(data=t.Dict(root=initial_data_dict.root)),
            )
        data_map = (
            initial_data
            if isinstance(initial_data, t.ConfigMap)
            else t.ConfigMap(initial_data)
            if initial_data is not None
            else t.ConfigMap(root={})
        )
        if auto_correlation_id and c.ContextKey.OPERATION_ID not in data_map:
            initial_data_dict_new: t.ConfigMap = data_map.model_copy()
            initial_data_dict_new[c.ContextKey.OPERATION_ID] = u.generate(
                "correlation",
            )
            return cls(
                initial_data=m.ContextData(
                    data=t.Dict(root=initial_data_dict_new.root),
                ),
            )
        return cls(initial_data=m.ContextData(data=t.Dict(root=data_map.root)))

    @staticmethod
    def _propagate_to_logger(key: str, value: t.ValueOrModel, scope: str) -> None:
        """Propagate context changes to FlextLogger (DRY helper).

        Args:
            key: Context key
            value: Context value
            scope: Context scope

        """
        if scope == c.ContextScope.GLOBAL:
            normalized = u.normalize_to_container(value)
            FlextLogger.structlog().contextvars.bind_contextvars(**{key: normalized})

    @staticmethod
    def _validate_set_inputs(key: str, value: t.ValueOrModel) -> r[bool]:
        """Validate inputs for set operation.

        Args:
            key: The key to validate
            value: The value to validate

        Returns:
            r[bool]: Success with True if valid, failure with error message

        """
        if not key:
            return r[bool].fail(c.ERR_CONTEXT_KEY_NON_EMPTY_STRING_REQUIRED)
        if value is None:
            return r[bool].fail(c.ERR_CONTEXT_VALUE_CANNOT_BE_NONE)
        value_for_guard: t.RecursiveContainer = (
            FlextContext._to_normalized(
                u.normalize_to_container(
                    u.normalize_to_metadata(value),
                ),
            )
            if u.is_pydantic_model(value)
            else value
        )
        if not isinstance(
            value_for_guard,
            (str, int, float, bool, list, dict, t.ConfigMap),
        ):
            return r[bool].fail(c.ERR_CONTEXT_VALUE_NOT_SERIALIZABLE)
        return r[bool].ok(True)

    def clear(self) -> None:
        """Clear all data from the context including metadata.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        This method consolidates cleanup functionality - it clears all scope
        data, resets metadata, and updates statistics.

        """
        if not self._active:
            return
        for scope_name, ctx_var in self._scope_vars.items():
            _ = ctx_var.set(t.ConfigMap(root={}))
            if scope_name == c.ContextScope.GLOBAL:
                FlextLogger.structlog().contextvars.clear_contextvars()
        self._metadata = m.Metadata()
        self._statistics.clears += 1
        operations = dict(self._statistics.operations)
        clear_value = operations.get(c.OPERATION_CLEAR)
        if isinstance(clear_value, int):
            operations[c.OPERATION_CLEAR] = clear_value + 1
            self._statistics = self._statistics.model_copy(
                update={"operations": operations},
            )

    def clone(self) -> Self:
        """Create a clone of this context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            A new FlextContext with the same data

        """
        cloned: Self = self.__class__(initial_data=self.initial_data)
        for scope_name, ctx_var in self.iter_scope_vars().items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                cloned.set(
                    t.ConfigMap(root=dict(scope_dict)),
                    scope=scope_name,
                )
        cloned.set_all_metadata_for_clone(self._metadata.model_copy())
        statistics_copy: m.ContextStatistics = self._statistics.model_copy()
        cloned.set_statistics_for_clone(statistics_copy)
        return cloned

    def export(
        self,
        *,
        include_statistics: bool = False,
        include_metadata: bool = False,
        as_dict: bool = True,
    ) -> m.ContextExport | t.ContainerMapping:
        """Export context data for serialization or debugging.

        Args:
            include_statistics: Include context statistics
            include_metadata: Include context metadata
            as_dict: If True, return as dict instead of ContextExport model

        Returns:
            ContextExport model or dict with all requested data

        """
        all_data: t.ConfigMap = t.ConfigMap(root={})
        all_scopes = self._get_all_scopes()
        all_data.update(dict(all_scopes))
        stats_dict_export: t.ConfigMap | None = None
        if include_statistics and self._statistics:
            stats_dict_export = t.ConfigMap(root=self._statistics.model_dump())
        metadata_dict_export: t.ContainerMapping | None = None
        if include_metadata:
            metadata_dict_export = self._get_all_metadata()
        metadata_for_model: t.ConfigMap | None = None
        if metadata_dict_export:
            normalized_metadata_map: MutableMapping[str, t.ValueOrModel] = {}
            for k, v in metadata_dict_export.items():
                metadata_value: t.ValueOrModel = v
                if u.is_mapping(v):
                    metadata_value = t.ConfigMap(
                        root=dict(v),
                    )
                normalized_metadata_map[k] = u.normalize_to_container(
                    u.normalize_to_metadata(metadata_value),
                )
            metadata_for_model = t.ConfigMap(root=normalized_metadata_map)
        statistics_mapping: t.Dict = t.Dict(
            root=dict(stats_dict_export or t.ConfigMap(root={})),
        )
        if as_dict:
            result_dict: t.MutableContainerMapping = dict(all_scopes)
            if include_statistics and stats_dict_export:
                stats_items: t.ContainerMapping = {
                    sk: FlextContext._to_normalized(sv)
                    for sk, sv in stats_dict_export.items()
                }
                result_dict["statistics"] = stats_items
            if include_metadata and metadata_dict_export:
                metadata_container: t.ConfigMap = t.ConfigMap(
                    root=dict(metadata_dict_export),
                )
                meta_items: t.ContainerMapping = {
                    mk: FlextContext._to_normalized(mv)
                    for mk, mv in metadata_container.items()
                }
                result_dict[c.FIELD_METADATA] = meta_items
            return result_dict
        metadata_root: t.ConfigMap | None = (
            t.ConfigMap(
                root={
                    k: u.normalize_to_container(v)
                    for k, v in metadata_for_model.items()
                },
            )
            if metadata_for_model
            else None
        )
        return m.ContextExport(
            data=dict(all_data),
            metadata=m.Metadata(
                attributes={
                    key: u.normalize_to_metadata(value)
                    for key, value in metadata_root.items()
                },
            )
            if metadata_root
            else None,
            statistics={
                key: FlextContext._to_normalized(
                    u.normalize_to_container(
                        u.normalize_to_metadata(value),
                    ),
                )
                for key, value in statistics_mapping.items()
            },
        )

    def get(self, key: str, scope: str = c.ContextScope.GLOBAL) -> r[t.RuntimeAtomic]:
        """Get a value from the context.

        Fast fail: Returns r[t.Container] - fails if key not found.
        No fallback behavior - use r monadic operations for defaults.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).
        No longer checks structlog - FlextLogger is independent.

        Args:
            key: The key to get
            scope: The scope to get from (global, user, session)

        Returns:
            r[t.Container]: Success with value, or failure if key not found

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
            return r[t.RuntimeAtomic].fail(c.ERR_CONTEXT_NOT_ACTIVE)
        scope_data = self._get_from_contextvar(scope)
        if key not in scope_data:
            return r[t.RuntimeAtomic].fail(
                f"Context key '{key}' not found in scope '{scope}'",
            )
        value = scope_data[key]
        self._update_statistics(c.OPERATION_GET)
        if value is None:
            return r[t.RuntimeAtomic].fail(
                f"Context key '{key}' has None value in scope '{scope}'",
            )

        normalized = u.normalize_to_container(value)
        return r[t.RuntimeAtomic].ok(normalized)

    def resolve_metadata(self, key: str) -> r[t.RuntimeAtomic]:
        """Get metadata from the context.

        Fast fail: Returns r[t.Container | BaseModel] - fails if key not found.
        No fallback behavior - use r monadic operations for defaults.

        Args:
            key: The metadata key

        Returns:
            r[t.Container | BaseModel]: Success with metadata value, or failure if key not found

        Example:
            >>> context = FlextContext()
            >>> context.apply_metadata("key", "value")
            >>> result = context.resolve_metadata("key")
            >>> if result.success:
            ...     value = result.value  # "value"
            >>>
            >>> # Key not found - fast fail
            >>> result = context.resolve_metadata("nonexistent")
            >>> assert result.failure
            >>>
            >>> # Use monadic operations for defaults
            >>> value = context.resolve_metadata("key").unwrap_or("default")

        """
        if key not in self._metadata.attributes:
            return r[t.RuntimeAtomic].fail(f"Metadata key '{key}' not found")
        raw_value: t.MetadataValue = self._metadata.attributes[key]
        normalized_value: t.RuntimeAtomic = u.normalize_to_container(
            raw_value,
        )
        return r[t.RuntimeAtomic].ok(normalized_value)

    def has(self, key: str, scope: str = c.ContextScope.GLOBAL) -> bool:
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
        scope_data = self._get_from_contextvar(scope)
        return key in scope_data

    def items(self) -> Sequence[tuple[str, t.RecursiveContainer]]:
        """Get all items (key-value pairs) in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of (key, value) tuples across all scopes

        """
        if not self._active:
            return []
        all_items: Sequence[tuple[str, t.RecursiveContainer]] = [
            item
            for ctx_var in self._scope_vars.values()
            for item in self._narrow_contextvar_to_configuration_dict(
                ctx_var.get()
            ).items()
        ]
        return all_items

    def iter_scope_vars(
        self,
    ) -> Mapping[str, contextvars.ContextVar[t.ConfigMap | None]]:
        """Get scope context variables for iteration.

        This method provides read-only access to scope variables for merge/clone
        operations, avoiding SLF001 violations when accessing from other instances.

        Returns:
            Dictionary of scope names to their context variables.

        """
        return self._scope_vars

    def keys(self) -> t.StrSequence:
        """Get all keys in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of all keys across all scopes

        """
        if not self._active:
            return list[str]()
        all_keys: set[str] = set()
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_keys.update(scope_dict.keys())
        return list(all_keys)

    _MERGEABLE_SCOPES: ClassVar[frozenset[str]] = frozenset({
        c.ContextScope.GLOBAL,
        c.ContextScope.USER,
        c.ContextScope.SESSION,
    })

    @staticmethod
    def _as_config_map(
        source: Mapping[str, t.ValueOrModel] | t.ContainerMapping,
        label: str,
    ) -> t.ConfigMap | None:
        """Try to wrap a mapping as ConfigMap, logging on failure."""
        try:
            return t.ConfigMap(root=dict(source))
        except (TypeError, ValueError, AttributeError) as exc:
            FlextContext._logger.debug(
                f"Context {label} validation failed",
                exc_info=exc,
            )
            return None

    def _extract_config_map(
        self,
        other: p.Context | t.ConfigMap | t.ContainerMapping,
    ) -> t.ConfigMap | None:
        """Extract a ConfigMap from any supported merge source."""
        match other:
            case FlextContext():
                exported_result = other.export(as_dict=True)
                if u.is_pydantic_model(exported_result):
                    return None
                return self._as_config_map(exported_result, "export payload")
            case t.ConfigMap():
                return other
            case _ if isinstance(other, Mapping):
                return self._as_config_map(other, "export payload")
            case _:
                return None

    def _apply_scoped_merge(self, exported_map: t.ConfigMap) -> None:
        """Merge exported scopes from another FlextContext."""
        for scope_name, scope_payload in exported_map.items():
            if scope_name not in self._MERGEABLE_SCOPES:
                continue
            if not u.is_mapping(scope_payload):
                continue
            scope_data = self._as_config_map(scope_payload, "scope payload")
            if scope_data is not None:
                self._set_in_contextvar(scope_name, scope_data)

    def merge(
        self,
        other: p.Context | t.ConfigMap | t.ContainerMapping,
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
        exported_map = self._extract_config_map(other)
        if exported_map is None:
            return self
        if isinstance(other, FlextContext):
            self._apply_scoped_merge(exported_map)
        else:
            self._set_in_contextvar(c.ContextScope.GLOBAL, exported_map)
        return self

    def remove(self, key: str, scope: str = c.ContextScope.GLOBAL) -> None:
        """Remove a key from the context."""
        if not self._active:
            return
        ctx_var = self._get_or_create_scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key in current:
            filtered = {k: v for k, v in current.items() if k != key}
            try:
                _ = ctx_var.set(t.ConfigMap(root=dict(filtered)))
            except (TypeError, ValueError, AttributeError) as exc:
                FlextContext._logger.debug(
                    "Failed to validate context after removal",
                    exc_info=exc,
                )
            self._update_statistics(c.OPERATION_REMOVE)

    @overload
    def set(
        self,
        key_or_data: str,
        value: t.RuntimeAtomic,
        *,
        scope: str = ...,
    ) -> r[bool]: ...

    @overload
    def set(
        self,
        key_or_data: t.ConfigMap,
        value: None = ...,
        *,
        scope: str = ...,
    ) -> r[bool]: ...

    def set(
        self,
        key_or_data: str | t.ConfigMap,
        value: t.RuntimeAtomic | None = None,
        *,
        scope: str = c.ContextScope.GLOBAL,
    ) -> r[bool]:
        """Set one or many values in the context.

        Supports two calling conventions:
        - Single key-value: ``ctx.set("key", value)``
        - Bulk: ``ctx.set(config_map)``

        ARCHITECTURAL NOTE: Uses Python contextvars for storage, delegates to
        FlextLogger for logging integration (global scope only).

        Args:
            key_or_data: A string key for single-value mode, or a ConfigMap for bulk mode
            value: The value to set (required for single-key mode, omit for bulk mode)
            scope: The scope for the value (global, user, session)

        Returns:
            r[bool]: Success with True if set, failure with error message

        """
        if not self._active:
            return r[bool].fail(c.ERR_CONTEXT_NOT_ACTIVE)
        if isinstance(key_or_data, t.ConfigMap):
            return self._set_bulk(key_or_data, scope)
        return self._set_single(key_or_data, value, scope)

    def set_all_metadata_for_clone(self, metadata: m.Metadata) -> None:
        """Set all metadata for the context (used internally for cloning)."""
        self._metadata = metadata

    def apply_metadata(self, key: str, value: t.MetadataValue) -> None:
        """Set metadata for the context.

        Args:
            key: The metadata key
            value: The metadata value

        """
        normalized_value: t.MetadataValue = u.normalize_to_metadata(value)
        updated_attributes = dict(self._metadata.attributes)
        updated_attributes[key] = normalized_value
        self._metadata = self._metadata.model_copy(
            update={c.FIELD_ATTRIBUTES: updated_attributes},
        )

    def set_statistics_for_clone(self, statistics: m.ContextStatistics) -> None:
        """Set context statistics (used internally for cloning)."""
        self._statistics = statistics

    def validate_context(self) -> r[bool]:
        """Validate the context data.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            r[bool]: Success with True if valid, failure with error details

        """
        if not self._active:
            return r[bool].fail(c.ERR_CONTEXT_NOT_ACTIVE)
        for ctx_var in self._scope_vars.values():
            try:
                scope_dict = self._narrow_contextvar_to_configuration_dict(
                    ctx_var.get(),
                )
            except TypeError as e:
                return r[bool].fail(str(e))
            for key in scope_dict:
                if not key:
                    return r[bool].fail(c.ERR_CONTEXT_INVALID_KEY_FOUND)
        return r[bool].ok(True)

    def values(self) -> t.ContainerList:
        """Get all values in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of all values across all scopes

        """
        if not self._active:
            empty_values: t.ContainerList = []
            return empty_values
        all_values: t.MutableContainerList = []
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_values.extend(scope_dict.values())
        return all_values

    def _execute_hooks(
        self,
        event: str,
        event_data: t.RecursiveContainer | t.ConfigMap,
    ) -> None:
        """Execute hooks for an event (DRY helper).

        Args:
            event: Event name ('set', 'get', 'remove', etc.)
            event_data: Data to pass to hooks

        """
        if event not in self._hooks:
            return
        hooks = self._hooks[event]
        for hook in hooks:
            if callable(hook):
                hook_data: t.Scalar
                if event_data is None:
                    hook_data = ""
                elif u.is_scalar(event_data):
                    hook_data = event_data
                else:
                    hook_data = str(event_data)
                _ = hook(hook_data)

    def _get_all_metadata(self) -> t.ContainerMapping:
        """Get all metadata from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.
        Custom fields are flattened into the top-level dict for easy access.

        Returns:
            Dictionary of all metadata (with custom_fields flattened)

        """
        data = dict(self._metadata.model_dump())
        custom_fields_raw = data.pop("custom_fields", {})
        custom_fields_dict: t.MutableContainerMapping = {}
        try:
            cf_map = t.ConfigMap(root=dict(custom_fields_raw))
            for ck, cv in cf_map.items():
                custom_fields_dict[ck] = FlextContext._to_normalized(cv)
        except (TypeError, ValueError, AttributeError) as exc:
            FlextContext._logger.debug(
                "Custom metadata field normalization failed",
                exc_info=exc,
            )
            custom_fields_dict = {}
        result: t.MutableContainerMapping = {}
        for k, v in data.items():
            if v is None or v == {}:
                continue
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, (str, int, float, bool, list, dict, tuple)):
                result[k] = v
            elif u.is_pydantic_model(v):
                result[k] = FlextContext._to_normalized(v)
            else:
                result[k] = str(v)
        result.update(custom_fields_dict)
        return result

    def _get_all_scopes(self) -> Mapping[str, t.ContainerMapping]:
        """Get all scope registrations.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary mapping scope names to their data dictionaries

        """
        if not self._active:
            return {}
        scopes: MutableMapping[str, t.ContainerMapping] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                scopes[scope_name] = dict(scope_dict)
        return scopes

    def _get_from_contextvar(self, scope: str) -> t.ConfigMap:
        """Get all values from contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        value = ctx_var.get()
        return t.ConfigMap(
            root=dict(
                FlextContext._narrow_contextvar_to_configuration_dict(value).items(),
            ),
        )

    def _get_or_create_scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[t.ConfigMap | None]:
        """Get or create contextvar for scope.

        Args:
            scope: Scope name (global, user, session, or custom)

        Returns:
            ContextVar for the scope

        """
        if scope not in self._scope_vars:
            self._scope_vars[scope] = contextvars.ContextVar(
                f"flext_{scope}_context",
                default=None,
            )
        return self._scope_vars[scope]

    def _set_bulk(self, data: t.ConfigMap, scope: str) -> r[bool]:
        """Set multiple values in the context from a ConfigMap."""
        if not data:
            return r[bool].ok(True)
        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = t.ConfigMap(root=dict(current))
            updated.update(data.root)
            _ = ctx_var.set(updated)
            self._update_statistics(c.OPERATION_SET)
            self._execute_hooks(
                c.OPERATION_SET,
                t.ConfigMap(root={c.Directory.DATA: t.ConfigMap(root=data.root)}),
            )
            return r[bool].ok(True)
        except TypeError as e:
            return r[bool].fail(str(e))

    def _set_in_contextvar(self, scope: str, data: t.ConfigMap) -> None:
        """Set multiple values in contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        current = t.ConfigMap(
            root=dict(
                self._narrow_contextvar_to_configuration_dict(ctx_var.get()),
            ),
        )
        updated = current.model_copy()
        updated.update(data.root)
        _ = ctx_var.set(updated)
        if scope == c.ContextScope.GLOBAL:
            normalized_context: Mapping[str, t.RuntimeAtomic] = {
                key: u.normalize_to_container(value)
                for key, value in data.items()
                if value is not None
            }
            FlextLogger.structlog().contextvars.bind_contextvars(**normalized_context)

    def _set_single(
        self,
        key: str,
        value: t.ValueOrModel | None,
        scope: str,
    ) -> r[bool]:
        """Set a single key-value pair in the context."""
        if value is None:
            return r[bool].fail(c.ERR_CONTEXT_SINGLE_KEY_VALUE_REQUIRED)
        validation_result = FlextContext._validate_set_inputs(key, value)
        if validation_result.failure:
            return r[bool].fail(validation_result.error or c.ERR_VALIDATION_FAILED)
        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = t.ConfigMap(root=dict(current))
            updated[key] = value
            _ = ctx_var.set(updated)
            FlextContext._propagate_to_logger(key, value, scope)
            self._update_statistics(c.OPERATION_SET)
            self._execute_hooks(
                c.OPERATION_SET,
                t.ConfigMap(root={"key": key, "value": value}),
            )
            return r[bool].ok(True)
        except TypeError as e:
            return r[bool].fail(str(e))

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper).

        Args:
            operation: Operation name ('set', 'get', 'remove', etc')

        """
        counter_attr: str = f"{operation}s"
        update_fields: MutableMapping[str, t.ValueOrModel] = {}
        if counter_attr in m.ContextStatistics.model_fields:
            current_value = getattr(self._statistics, counter_attr, 0)
            if isinstance(current_value, int):
                update_fields[counter_attr] = current_value + 1
        operations = dict(self._statistics.operations)
        op_value = operations.get(operation)
        if isinstance(op_value, int):
            operations[operation] = op_value + 1
            update_fields["operations"] = operations
        if update_fields:
            self._statistics = self._statistics.model_copy(update=update_fields)

    _container: p.Container | None = None

    @classmethod
    def resolve_container(cls) -> p.Container:
        """Get global container instance.

        The container must be set explicitly using `configure_container()` before
        calling this method. This breaks the circular dependency by requiring
        explicit initialization.

        Returns:
            Global DI instance for dependency injection

        Raises:
            RuntimeError: If container was not set via `configure_container()`.

        Example:
            >>> from flext_core import FlextContainer
            >>> FlextContext.configure_container(FlextContainer.get_global())
            >>> container = FlextContext.resolve_container()
            >>> result = container.get("service_name")

        """
        if cls._container is None:
            msg = "Container not initialized. Call FlextContext.configure_container(container) before using resolve_container()."
            raise RuntimeError(msg)
        return cls._container

    @classmethod
    def configure_container(cls, container: p.Container) -> None:
        """Set the global container instance."""
        cls._container = container

    class Variables:
        """Context variables using structlog as single source of truth."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.CORRELATION_ID,
                default=None,
            )
            PARENT_CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(c.ContextKey.PARENT_CORRELATION_ID, default=None)
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.SERVICE_NAME,
                default=None,
            )
            SERVICE_VERSION: Final[m.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(c.ContextKey.SERVICE_VERSION, default=None)
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.USER_ID,
                default=None,
            )
            REQUEST_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.REQUEST_ID,
                default=None,
            )
            REQUEST_TIMESTAMP: Final[m.StructlogProxyContextVar[datetime]] = (
                u.create_datetime_proxy(c.ContextKey.REQUEST_TIMESTAMP, default=None)
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.OPERATION_NAME,
                default=None,
            )
            OPERATION_START_TIME: Final[m.StructlogProxyContextVar[datetime]] = (
                u.create_datetime_proxy(
                    c.ContextKey.OPERATION_START_TIME,
                    default=None,
                )
            )
            OPERATION_METADATA: Final[m.StructlogProxyContextVar[t.ConfigMap]] = (
                u.create_dict_proxy(c.ContextKey.OPERATION_METADATA, default=None)
            )

        CorrelationId: Final[m.StructlogProxyContextVar[str]] = (
            Correlation.CORRELATION_ID
        )
        ParentCorrelationId: Final[m.StructlogProxyContextVar[str]] = (
            Correlation.PARENT_CORRELATION_ID
        )
        ServiceName: Final[m.StructlogProxyContextVar[str]] = Service.SERVICE_NAME
        ServiceVersion: Final[m.StructlogProxyContextVar[str]] = Service.SERVICE_VERSION
        UserId: Final[m.StructlogProxyContextVar[str]] = Request.USER_ID
        RequestId: Final[m.StructlogProxyContextVar[str]] = Request.REQUEST_ID
        RequestTimestamp: Final[m.StructlogProxyContextVar[datetime]] = (
            Request.REQUEST_TIMESTAMP
        )
        OperationName: Final[m.StructlogProxyContextVar[str]] = (
            Performance.OPERATION_NAME
        )
        OperationStartTime: Final[m.StructlogProxyContextVar[datetime]] = (
            Performance.OPERATION_START_TIME
        )
        OperationMetadata: Final[m.StructlogProxyContextVar[t.ConfigMap]] = (
            Performance.OPERATION_METADATA
        )

    class Correlation:
        """Distributed tracing and correlation ID management utilities."""

        @staticmethod
        def resolve_correlation_id() -> str | None:
            """Get current correlation ID."""
            correlation_id = FlextContext.Variables.CorrelationId.get()
            return correlation_id if isinstance(correlation_id, str) else None

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create correlation context scope.

            Uses c configuration for correlation ID generation.
            """
            if correlation_id is None:
                correlation_id = u.generate("correlation")
            current_correlation = FlextContext.Variables.CorrelationId.get()
            correlation_token = FlextContext.Variables.CorrelationId.set(correlation_id)
            parent_token = None
            if parent_id:
                parent_token = FlextContext.Variables.ParentCorrelationId.set(parent_id)
            elif isinstance(current_correlation, str):
                parent_token = FlextContext.Variables.ParentCorrelationId.set(
                    current_correlation,
                )
            try:
                yield correlation_id
            finally:
                FlextContext.Variables.CorrelationId.reset(correlation_token)
                if parent_token:
                    FlextContext.Variables.ParentCorrelationId.reset(parent_token)

        @staticmethod
        def apply_correlation_id(correlation_id: str | None) -> None:
            """Set correlation ID.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            Accepts ``None`` to explicitly clear the active correlation when needed.
            """
            _ = FlextContext.Variables.CorrelationId.set(correlation_id)

    class Service:
        """Service identification and lifecycle context management utilities."""

        @staticmethod
        def fetch_service(service_name: str) -> r[t.Scalar]:
            """Resolve service from global container using r.

            Provides unified service resolution pattern across the ecosystem
            by integrating FlextContainer with FlextContext.

            Args:
                service_name: Name of the service to retrieve

            Returns:
                Result containing the service instance or error.
                Use get_typed() or type identity / MRO checks for type narrowing.

            Example:
                >>> result = FlextContext.Service.fetch_service("logger")
                >>> if result.success and isinstance(result.value, FlextLogger):
                ...     result.value.info("Service retrieved")

            """
            container: p.Container = FlextContext.resolve_container()
            service_result = container.get(service_name)
            if service_result.success:
                service_value = service_result.value
                if service_value is None:
                    return r[t.Scalar].ok("")
                if u.is_scalar(service_value):
                    return r[t.Scalar].ok(service_value)
                return r[t.Scalar].ok(str(service_value))
            return r[t.Scalar].fail(service_result.error or "Service not found")

        @staticmethod
        def register_service(
            service_name: str,
            service: t.RegisterableService,
        ) -> r[bool]:
            """Register service in global container using r.

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
            container = FlextContext.resolve_container()
            try:
                # Type ignoring register until Pydantic vs arbitrary type unification is complete
                _ = container.register(service_name, service)
                return r[bool].ok(True)
            except ValueError as e:
                return r[bool].fail(str(e))

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service context scope."""
            _ = FlextContext.Variables.ServiceName.get()
            _ = FlextContext.Variables.ServiceVersion.get()
            name_token = FlextContext.Variables.ServiceName.set(service_name)
            version_token = None
            if version:
                version_token = FlextContext.Variables.ServiceVersion.set(version)
            try:
                yield
            finally:
                FlextContext.Variables.ServiceName.reset(name_token)
                if version_token:
                    FlextContext.Variables.ServiceVersion.reset(version_token)

    class Request:
        """Request-level context management for user and operation metadata utilities."""

        @staticmethod
        def resolve_operation_name() -> str | None:
            """Get the current operation name from context."""
            operation_name = FlextContext.Variables.OperationName.get()
            return str(operation_name) if operation_name is not None else None

        @staticmethod
        def apply_operation_name(operation_name: str) -> None:
            """Set operation name in context.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            """
            _ = FlextContext.Variables.OperationName.set(operation_name)

    class Performance:
        """Performance monitoring and timing context management utilities."""

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[t.ConfigMap]:
            """Create timed operation context with performance tracking."""
            start_time = u.generate_datetime_utc()
            start_perf = time.perf_counter()
            operation_metadata: t.ConfigMap = t.ConfigMap(
                root={
                    c.MetadataKey.START_TIME: start_time.isoformat(),
                    c.ContextKey.OPERATION_NAME: operation_name,
                },
            )
            start_token = FlextContext.Variables.OperationStartTime.set(start_time)
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
                duration = time.perf_counter() - start_perf
                end_time = start_time + timedelta(seconds=duration)
                operation_metadata.update({
                    c.MetadataKey.END_TIME: end_time.isoformat(),
                    c.MetadataKey.DURATION_SECONDS: duration,
                })
                FlextContext.Variables.OperationStartTime.reset(start_token)
                FlextContext.Variables.OperationMetadata.reset(metadata_token)
                if operation_token:
                    FlextContext.Variables.OperationName.reset(operation_token)

    class Serialization:
        """Context serialization and deserialization utilities."""

        @staticmethod
        def export_full_context() -> t.ContainerMapping:
            """Get current context as dictionary."""
            context_vars = FlextContext.Variables
            operation_metadata_raw = context_vars.Performance.OPERATION_METADATA.get()
            operation_metadata_value: t.RecursiveContainer = ""
            if operation_metadata_raw is not None:
                operation_metadata_value = FlextContext._to_normalized(
                    u.normalize_to_container(
                        u.normalize_to_metadata(operation_metadata_raw),
                    ),
                )
            raw_ctx: Mapping[str, t.ValueOrModel | None] = {
                c.ContextKey.CORRELATION_ID: context_vars.Correlation.CORRELATION_ID.get(),
                c.ContextKey.PARENT_CORRELATION_ID: context_vars.Correlation.PARENT_CORRELATION_ID.get(),
                c.ContextKey.SERVICE_NAME: context_vars.Service.SERVICE_NAME.get(),
                c.ContextKey.SERVICE_VERSION: context_vars.Service.SERVICE_VERSION.get(),
                c.ContextKey.USER_ID: context_vars.Request.USER_ID.get(),
                c.ContextKey.OPERATION_NAME: context_vars.Performance.OPERATION_NAME.get(),
                c.ContextKey.REQUEST_ID: context_vars.Request.REQUEST_ID.get(),
                c.ContextKey.OPERATION_START_TIME: st.isoformat()
                if isinstance(
                    (st := context_vars.Performance.OPERATION_START_TIME.get()),
                    datetime,
                )
                else None,
                c.ContextKey.OPERATION_METADATA: operation_metadata_value,
            }
            return {
                k: FlextContext._to_normalized(v)
                for k, v in raw_ctx.items()
                if v is not None
            }

    class Utilities:
        """Context management utility methods."""

        @staticmethod
        def clear_context() -> None:
            """Clear all context variables.

            Note: ContextVar.set() does not raise LookupError - it always succeeds.
            Setting to None effectively clears the variable.
            """
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
            _ = FlextContext.Variables.OperationStartTime.set(None)
            _ = FlextContext.Variables.OperationMetadata.set(None)
            _ = FlextContext.Variables.RequestTimestamp.set(None)

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure correlation ID exists, creating one if needed."""
            correlation_id_value = FlextContext.Variables.CorrelationId.get()
            if isinstance(correlation_id_value, str) and correlation_id_value:
                return correlation_id_value
            new_correlation_id: str = u.generate("correlation")
            FlextContext.Correlation.apply_correlation_id(new_correlation_id)
            return new_correlation_id


__all__: t.StrSequence = ["FlextContext"]
