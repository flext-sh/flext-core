"""Context propagation utilities for dispatcher-coordinated workloads.

FlextContext tracks correlation metadata, request data, and timing information
through the dispatcher pipeline and into handlers, ensuring structured logs and
metrics remain consistent across threads and async boundaries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
import enum
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, ClassVar, Final, Self, overload, override

from pydantic import BaseModel, Field, PrivateAttr

from flext_core import FlextLogger, FlextRuntime, c, m, p, r, t, u


class _Sentinel(enum.Enum):
    """Sentinel enum for distinguishing 'not provided' from None."""

    MISSING = "__sentinel_missing__"


_SENTINEL: Final[str] = _Sentinel.MISSING.value


class FlextContext(m.ArbitraryTypesModel, FlextRuntime):
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

    _logger: ClassVar[FlextLogger] = FlextLogger(__name__)

    @staticmethod
    def _empty_hooks() -> dict[
        str, list[Callable[[t.Scalar], t.NormalizedValue | BaseModel | None]]
    ]:
        return {}

    initial_data: Annotated[
        m.ContextData | m.ConfigMap | None,
        Field(default=None, description="Initial data for context scopes."),
    ]

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: m.ConfigMap | Mapping[str, t.NormalizedValue] | BaseModel | None,
    ) -> Mapping[str, t.NormalizedValue]:
        """Return contextvar payload as ConfigMap with safe default."""
        if ctx_value is None:
            return {}

        if isinstance(ctx_value, BaseModel):
            ctx_value = getattr(ctx_value, "root", ctx_value.model_dump())
        elif u.is_mapping(ctx_value):
            pass
        else:
            return {}

        try:
            normalized: dict[str, t.NormalizedValue] = {}
            mapping_value: Mapping[str, t.NormalizedValue]
            if isinstance(ctx_value, BaseModel):
                mapping_value = dict(ctx_value.model_dump().items())
            else:
                mapping_value = dict(ctx_value.items())
            for key, value in mapping_value.items():
                if str(key) != key:
                    return {}
                normalized_value = FlextRuntime.normalize_to_container(value)
                if isinstance(normalized_value, BaseModel):
                    normalized[key] = FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(normalized_value)
                    )
                else:
                    normalized[key] = normalized_value
            return normalized
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            FlextContext._logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
            return {}

    _metadata: m.Metadata = PrivateAttr()
    _hooks: dict[
        str, list[Callable[[t.Scalar], t.NormalizedValue | BaseModel | None]]
    ] = PrivateAttr(default_factory=_empty_hooks)
    _statistics: m.ContextStatistics = PrivateAttr(default_factory=m.ContextStatistics)
    _active: bool = PrivateAttr(default=True)
    _suspended: bool = PrivateAttr(default=False)
    _scope_vars: dict[str, contextvars.ContextVar[m.ConfigMap | None]] = PrivateAttr()

    def __init__(self, **data: t.NormalizedValue | BaseModel) -> None:
        """Initialize FlextContext with optional initial data.

        ARCHITECTURAL NOTE: FlextContext now uses Python's contextvars for storage,
        completely independent of structlog. It delegates to FlextLogger for logging
        integration, maintaining clear separation of concerns.

        """
        super().__init__(**data)
        context_data = m.ContextData()
        if self.initial_data is not None:
            if hasattr(self.initial_data, "data") and hasattr(
                self.initial_data, "metadata"
            ):
                context_data = m.ContextData.model_validate(
                    self.initial_data, from_attributes=True
                )
            elif isinstance(self.initial_data, m.ConfigMap):
                context_data = m.ContextData(
                    data=m.Dict(
                        root=m.ConfigMap(root=dict(self.initial_data.items())).root
                    )
                )
        self._metadata = m.Metadata()
        self._hooks = {}
        self._statistics = m.ContextStatistics()
        self._active = True
        self._suspended = False
        self._scope_vars = {
            c.Context.SCOPE_GLOBAL: contextvars.ContextVar(
                "flext_global_context", default=None
            ),
            c.Context.SCOPE_USER: contextvars.ContextVar(
                "flext_user_context", default=None
            ),
            c.Context.SCOPE_SESSION: contextvars.ContextVar(
                "flext_session_context", default=None
            ),
        }
        if context_data.data:
            self._set_in_contextvar(
                c.Context.SCOPE_GLOBAL, m.ConfigMap(root=context_data.data.root)
            )

    @overload
    @classmethod
    def create(cls, initial_data: m.ConfigMap | None = None) -> Self: ...

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
        if operation_id is not None or user_id is not None or metadata is not None:
            initial_data_dict: m.ConfigMap = m.ConfigMap(root={})
            if operation_id is not None:
                initial_data_dict[c.Context.KEY_OPERATION_ID] = operation_id
            elif auto_correlation_id:
                initial_data_dict[c.Context.KEY_OPERATION_ID] = u.generate(
                    "correlation"
                )
            if user_id is not None:
                initial_data_dict[c.Context.KEY_USER_ID] = user_id
            if metadata is not None:
                initial_data_dict.update(dict(metadata.items()))
            return cls(
                initial_data=m.ContextData(data=m.Dict(root=initial_data_dict.root))
            )
        data_map = (
            initial_data
            if isinstance(initial_data, m.ConfigMap)
            else m.ConfigMap(initial_data)
            if initial_data is not None
            else m.ConfigMap(root={})
        )
        if auto_correlation_id and c.Context.KEY_OPERATION_ID not in data_map:
            initial_data_dict_new: m.ConfigMap = data_map.model_copy()
            initial_data_dict_new[c.Context.KEY_OPERATION_ID] = u.generate(
                "correlation"
            )
            return cls(
                initial_data=m.ContextData(data=m.Dict(root=initial_data_dict_new.root))
            )
        return cls(initial_data=m.ContextData(data=m.Dict(root=data_map.root)))

    @staticmethod
    def _propagate_to_logger(
        key: str, value: t.NormalizedValue | BaseModel, scope: str
    ) -> None:
        """Propagate context changes to FlextLogger (DRY helper).

        Args:
            key: Context key
            value: Context value
            scope: Context scope

        """
        if scope == c.Context.SCOPE_GLOBAL:
            normalized = FlextRuntime.normalize_to_container(value)
            _ = FlextLogger.bind_global_context(**{key: normalized})

    @staticmethod
    def _validate_set_inputs(key: str, value: t.NormalizedValue | BaseModel) -> r[bool]:
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
        value_for_guard: t.NormalizedValue = (
            FlextRuntime.normalize_to_container(
                FlextRuntime.normalize_to_metadata(value)
            )
            if isinstance(value, BaseModel)
            else value
        )
        guarded_value = u.guard(
            value_for_guard,
            (str, int, float, bool, list, dict, m.ConfigMap),
            return_value=True,
        )
        if isinstance(guarded_value, r):
            return r[bool].fail("Value must be serializable")
        return r[bool].ok(value=True)

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
            _ = ctx_var.set(m.ConfigMap(root={}))
            if scope_name == c.Context.SCOPE_GLOBAL:
                _ = FlextLogger.clear_global_context()
        self._metadata = m.Metadata()
        self._statistics.clears += 1
        operations = dict(self._statistics.operations.items())
        clear_value = operations.get(c.Context.OPERATION_CLEAR)
        if isinstance(clear_value, int):
            operations[c.Context.OPERATION_CLEAR] = clear_value + 1
            self._statistics = self._statistics.model_copy(
                update={"operations": operations}
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
                result = cloned.set(
                    m.ConfigMap(root=dict(scope_dict.items())), scope=scope_name
                )
                if not result:
                    pass
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
    ) -> m.ContextExport | Mapping[str, t.NormalizedValue]:
        """Export context data for serialization or debugging.

        Args:
            include_statistics: Include context statistics
            include_metadata: Include context metadata
            as_dict: If True, return as dict instead of ContextExport model

        Returns:
            ContextExport model or dict with all requested data

        """
        all_data: m.ConfigMap = m.ConfigMap(root={})
        all_scopes = self._get_all_scopes()
        all_data.update(dict(all_scopes.items()))
        stats_dict_export: m.ConfigMap | None = None
        if include_statistics and self._statistics:
            stats_dict_export = m.ConfigMap(root=self._statistics.model_dump())
        metadata_dict_export: Mapping[str, t.NormalizedValue] | None = None
        if include_metadata:
            metadata_dict_export = self._get_all_metadata()
        metadata_for_model: m.ConfigMap | None = None
        if metadata_dict_export:
            normalized_metadata_map: dict[str, t.NormalizedValue] = {}
            for k, v in metadata_dict_export.items():
                metadata_value: t.NormalizedValue | BaseModel = v
                if hasattr(v, "items") and callable(getattr(v, "items", None)):
                    metadata_value = m.ConfigMap(root=dict(v.items()))
                normalized_metadata_map[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(metadata_value)
                )
            metadata_for_model = m.ConfigMap(root=normalized_metadata_map)
        statistics_mapping: m.Dict = m.Dict(
            root=dict((stats_dict_export or m.ConfigMap(root={})).items())
        )
        if as_dict:
            result_dict: dict[str, t.NormalizedValue] = dict(all_scopes.items())
            if include_statistics and stats_dict_export:
                result_dict["statistics"] = dict(stats_dict_export.items())
            if include_metadata and metadata_dict_export:
                metadata_container: m.ConfigMap = m.ConfigMap(
                    root=dict(metadata_dict_export.items())
                )
                result_dict["metadata"] = dict(metadata_container.items())
            return result_dict
        metadata_root: m.ConfigMap | None = (
            m.ConfigMap(
                root={
                    k: FlextRuntime.normalize_to_container(v)
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
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in metadata_root.items()
                }
            )
            if metadata_root
            else None,
            statistics={
                key: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(value)
                )
                for key, value in statistics_mapping.items()
            },
        )

    def get(
        self, key: str, scope: str = c.Context.SCOPE_GLOBAL
    ) -> r[t.Container | BaseModel]:
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
            return r[t.Container | BaseModel].fail("Context is not active")
        scope_data = self._get_from_contextvar(scope)
        if key not in scope_data:
            return r[t.Container | BaseModel].fail(
                f"Context key '{key}' not found in scope '{scope}'"
            )
        value = scope_data[key]
        self._update_statistics(c.Context.OPERATION_GET)
        if value is None:
            return r[t.Container | BaseModel].fail(
                f"Context key '{key}' has None value in scope '{scope}'"
            )

        normalized = FlextRuntime.normalize_to_container(value)
        return r[t.Container | BaseModel].ok(normalized)

    def get_metadata(self, key: str) -> r[t.Container | BaseModel]:
        """Get metadata from the context.

        Fast fail: Returns r[t.Container | BaseModel] - fails if key not found.
        No fallback behavior - use r monadic operations for defaults.

        Args:
            key: The metadata key

        Returns:
            r[t.Container | BaseModel]: Success with metadata value, or failure if key not found

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
        if key not in self._metadata.attributes:
            return r[t.Container | BaseModel].fail(f"Metadata key '{key}' not found")
        raw_value: t.MetadataValue = self._metadata.attributes[key]
        normalized_value: t.Container | BaseModel = FlextRuntime.normalize_to_container(
            raw_value
        )
        return r[t.Container | BaseModel].ok(normalized_value)

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
        scope_data = self._get_from_contextvar(scope)
        return key in scope_data

    def items(self) -> list[tuple[str, t.NormalizedValue]]:
        """Get all items (key-value pairs) in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of (key, value) tuples across all scopes

        """
        if not self._active:
            return []
        all_items: list[tuple[str, t.NormalizedValue]] = []
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_items.extend(scope_dict.items())
        return all_items

    def iter_scope_vars(
        self,
    ) -> Mapping[str, contextvars.ContextVar[m.ConfigMap | None]]:
        """Get scope context variables for iteration.

        This method provides read-only access to scope variables for merge/clone
        operations, avoiding SLF001 violations when accessing from other instances.

        Returns:
            Dictionary of scope names to their context variables.

        """
        return self._scope_vars

    def keys(self) -> list[str]:
        """Get all keys in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage (single source of truth).

        Returns:
            List of all keys across all scopes

        """
        if not self._active:
            return []
        all_keys: set[str] = set()
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_keys.update(scope_dict.keys())
        return list(all_keys)

    def merge(
        self, other: FlextContext | m.ConfigMap | Mapping[str, t.NormalizedValue]
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
                if isinstance(exported, Mapping):
                    exported_map = m.ConfigMap(root=dict(exported.items()))
                else:
                    exported_map = m.ConfigMap(root={})
            except (TypeError, ValueError, AttributeError) as exc:
                FlextContext._logger.debug(
                    "Context export payload validation failed", exc_info=exc
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
                    if isinstance(scope_payload, Mapping):
                        scope_data = m.ConfigMap(root=dict(scope_payload.items()))
                    else:
                        scope_data = None
                except (TypeError, ValueError, AttributeError) as exc:
                    FlextContext._logger.debug(
                        "Context scope payload validation failed", exc_info=exc
                    )
                    scope_data = None
                if scope_data is not None:
                    self._set_in_contextvar(scope_name, scope_data)
        else:
            self._set_in_contextvar(
                c.Context.SCOPE_GLOBAL, m.ConfigMap(root=dict(other.items()))
            )
        return self

    def remove(self, key: str, scope: str = c.Context.SCOPE_GLOBAL) -> None:
        """Remove a key from the context."""
        if not self._active:
            return
        ctx_var = self._get_or_create_scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key in current:
            filtered = {k: v for k, v in current.items() if k != key}
            try:
                _ = ctx_var.set(m.ConfigMap(root=dict(filtered.items())))
            except (TypeError, ValueError, AttributeError) as exc:
                FlextContext._logger.debug(
                    "Failed to validate context after removal", exc_info=exc
                )
            self._update_statistics(c.Context.OPERATION_REMOVE)

    @overload
    def set(
        self, key_or_data: str, value: t.Container | BaseModel, *, scope: str = ...
    ) -> r[bool]: ...

    @overload
    def set(
        self, key_or_data: m.ConfigMap, value: None = ..., *, scope: str = ...
    ) -> r[bool]: ...

    def set(
        self,
        key_or_data: str | m.ConfigMap,
        value: t.Container | BaseModel | None = _SENTINEL,
        *,
        scope: str = c.Context.SCOPE_GLOBAL,
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
            return r[bool].fail("Context is not active")
        if isinstance(key_or_data, m.ConfigMap):
            return self._set_bulk(key_or_data, scope)
        return self._set_single(key_or_data, value, scope)

    def set_all_metadata_for_clone(self, metadata: m.Metadata) -> None:
        """Set all metadata for the context (used internally for cloning)."""
        self._metadata = metadata

    def set_metadata(self, key: str, value: t.MetadataValue) -> None:
        """Set metadata for the context.

        Args:
            key: The metadata key
            value: The metadata value

        """
        normalized_value: t.MetadataValue = FlextRuntime.normalize_to_metadata(value)
        updated_attributes = dict(self._metadata.attributes.items())
        updated_attributes[key] = normalized_value
        self._metadata = self._metadata.model_copy(
            update={"attributes": updated_attributes}
        )

    def set_statistics_for_clone(self, statistics: m.ContextStatistics) -> None:
        """Set context statistics (used internally for cloning)."""
        self._statistics = statistics

    def validate_input(self) -> r[bool]:
        """Validate context data integrity across all active scopes."""
        return self.validate_context()

    def validate_context(self) -> r[bool]:
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

    def values(self) -> list[t.NormalizedValue]:
        """Get all values in the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            List of all values across all scopes

        """
        if not self._active:
            return []
        all_values: list[t.NormalizedValue] = []
        for ctx_var in self._scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_values.extend(scope_dict.values())
        return all_values

    def _execute_hooks(
        self, event: str, event_data: t.NormalizedValue | m.ConfigMap
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
                elif isinstance(event_data, (str, int, float, bool, datetime)):
                    hook_data = event_data
                else:
                    hook_data = str(event_data)
                _ = hook(hook_data)

    def _get_all_metadata(self) -> Mapping[str, t.NormalizedValue]:
        """Get all metadata from the context.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.
        Custom fields are flattened into the top-level dict for easy access.

        Returns:
            Dictionary of all metadata (with custom_fields flattened)

        """
        data = dict(self._metadata.model_dump().items())
        custom_fields_raw = data.pop("custom_fields", {})
        custom_fields_dict: dict[str, t.NormalizedValue] = {}
        try:
            custom_fields_dict = dict(
                m.ConfigMap(root=dict(custom_fields_raw.items())).items()
            )
        except (TypeError, ValueError, AttributeError) as exc:
            FlextContext._logger.debug(
                "Custom metadata field normalization failed", exc_info=exc
            )
            custom_fields_dict = {}
        result: dict[str, t.NormalizedValue] = {}
        for k, v in data.items():
            if v is None or v == {}:
                continue
            if hasattr(v, "isoformat") and callable(getattr(v, "isoformat", None)):
                result[k] = v.isoformat()
            else:
                result[k] = v
        result.update(custom_fields_dict)
        return result

    def _get_all_scopes(self) -> Mapping[str, Mapping[str, t.NormalizedValue]]:
        """Get all scope registrations.

        ARCHITECTURAL NOTE: Uses Python contextvars for storage.

        Returns:
            Dictionary mapping scope names to their data dictionaries

        """
        if not self._active:
            return {}
        scopes: dict[str, dict[str, t.NormalizedValue]] = {}
        for scope_name, ctx_var in self._scope_vars.items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                scopes[scope_name] = dict(scope_dict.items())
        return scopes

    def _get_from_contextvar(self, scope: str) -> m.ConfigMap:
        """Get all values from contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        value = ctx_var.get()
        return m.ConfigMap(
            root=dict(
                FlextContext._narrow_contextvar_to_configuration_dict(value).items()
            )
        )

    def _get_or_create_scope_var(
        self, scope: str
    ) -> contextvars.ContextVar[m.ConfigMap | None]:
        """Get or create contextvar for scope.

        Args:
            scope: Scope name (global, user, session, or custom)

        Returns:
            ContextVar for the scope

        """
        if scope not in self._scope_vars:
            self._scope_vars[scope] = contextvars.ContextVar(
                f"flext_{scope}_context", default=None
            )
        return self._scope_vars[scope]

    def _set_bulk(self, data: m.ConfigMap, scope: str) -> r[bool]:
        """Set multiple values in the context from a ConfigMap."""
        if not data:
            return r[bool].ok(value=True)
        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = m.ConfigMap(root=dict(current.items()))
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

    def _set_in_contextvar(self, scope: str, data: m.ConfigMap) -> None:
        """Set multiple values in contextvar scope."""
        ctx_var = self._get_or_create_scope_var(scope)
        current = m.ConfigMap(
            root=dict(
                self._narrow_contextvar_to_configuration_dict(ctx_var.get()).items()
            )
        )
        updated = current.model_copy()
        updated.update(data.root)
        _ = ctx_var.set(updated)
        if scope == c.Context.SCOPE_GLOBAL:
            normalized_context: dict[str, t.Container | BaseModel] = {
                key: FlextRuntime.normalize_to_container(value)
                for key, value in data.items()
            }
            _ = FlextLogger.bind_global_context(**normalized_context)

    def _set_single(
        self, key: str, value: t.NormalizedValue | BaseModel | None, scope: str
    ) -> r[bool]:
        """Set a single key-value pair in the context."""
        if value is _SENTINEL or value is None:
            return r[bool].fail("Value is required for single-key set")
        validation_result = FlextContext._validate_set_inputs(key, value)
        if validation_result.is_failure:
            return r[bool].fail(validation_result.error or "Validation failed")
        try:
            ctx_var = self._get_or_create_scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = m.ConfigMap(root=dict(current.items()))
            updated[key] = value
            _ = ctx_var.set(updated)
            FlextContext._propagate_to_logger(key, value, scope)
            self._update_statistics(c.Context.OPERATION_SET)
            self._execute_hooks(
                c.Context.OPERATION_SET, m.ConfigMap(root={"key": key, "value": value})
            )
            return r[bool].ok(value=True)
        except (TypeError, Exception) as e:
            error_msg = (
                str(e)
                if isinstance(e, TypeError)
                else f"Failed to set context value: {e}"
            )
            return r[bool].fail(error_msg)

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper).

        Args:
            operation: Operation name ('set', 'get', 'remove', etc')

        """
        counter_attr: str = f"{operation}s"
        if hasattr(self._statistics, counter_attr):
            current_value: int = getattr(self._statistics, counter_attr, 0)
            setattr(self._statistics, counter_attr, current_value + 1)
        operations = dict(self._statistics.operations.items())
        value = operations.get(operation)
        if isinstance(value, int):
            operations[operation] = value + 1
            self._statistics = self._statistics.model_copy(
                update={"operations": operations}
            )

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
            msg = "Container not initialized. Call FlextContext.set_container(container) before using get_container()."
            raise RuntimeError(msg)
        return cls._container

    @classmethod
    def set_container(cls, container: p.DI) -> None:
        """Set the global container instance."""
        cls._container = container

    class Variables:
        """Context variables using structlog as single source of truth."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.Context.KEY_CORRELATION_ID, default=None
            )
            PARENT_CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(c.Context.KEY_PARENT_CORRELATION_ID, default=None)
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.Context.KEY_SERVICE_NAME, default=None
            )
            SERVICE_VERSION: Final[m.StructlogProxyContextVar[str]] = (
                u.create_str_proxy("service_version", default=None)
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.Context.KEY_USER_ID, default=None
            )
            REQUEST_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                "request_id", default=None
            )
            REQUEST_TIMESTAMP: Final[m.StructlogProxyContextVar[datetime]] = (
                u.create_datetime_proxy("request_timestamp", default=None)
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.Context.KEY_OPERATION_NAME, default=None
            )
            OPERATION_START_TIME: Final[m.StructlogProxyContextVar[datetime]] = (
                u.create_datetime_proxy("operation_start_time", default=None)
            )
            OPERATION_METADATA: Final[m.StructlogProxyContextVar[m.ConfigMap]] = (
                u.create_dict_proxy("operation_metadata", default=None)
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
        OperationMetadata: Final[m.StructlogProxyContextVar[m.ConfigMap]] = (
            Performance.OPERATION_METADATA
        )

    class Correlation:
        """Distributed tracing and correlation ID management utilities."""

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID."""
            correlation_id = FlextContext.Variables.CorrelationId.get()
            return correlation_id if isinstance(correlation_id, str) else None

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None, parent_id: str | None = None
        ) -> Generator[str]:
            """Create correlation context scope.

            Uses c.Context configuration for correlation ID generation.
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
                    current_correlation
                )
            try:
                yield correlation_id
            finally:
                FlextContext.Variables.CorrelationId.reset(correlation_token)
                if parent_token:
                    FlextContext.Variables.ParentCorrelationId.reset(parent_token)

        @staticmethod
        def set_correlation_id(correlation_id: str | None) -> None:
            """Set correlation ID.

            Note: Uses structlog as single source of truth (via FlextModels.StructlogProxyContextVar).
            Accepts ``None`` to explicitly clear the active correlation when needed.
            """
            _ = FlextContext.Variables.CorrelationId.set(correlation_id)

    class Service:
        """Service identification and lifecycle context management utilities."""

        @staticmethod
        def get_service(service_name: str) -> r[t.Scalar]:
            """Resolve service from global container using r.

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
            container: p.DI = FlextContext.get_container()
            service_result = container.get(service_name)
            if service_result.is_success:
                service_value = service_result.value
                if service_value is None:
                    return r[t.Scalar].ok("")
                if isinstance(service_value, (str, int, float, bool, datetime)):
                    return r[t.Scalar].ok(service_value)
                return r[t.Scalar].ok(str(service_value))
            return r[t.Scalar].fail(service_result.error or "Service not found")

        @staticmethod
        def register_service(
            service_name: str, service: t.RegisterableService
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
            container = FlextContext.get_container()
            try:
                # Type ignoring register until Pydantic vs arbitrary type unification is complete
                _ = container.register(service_name, service)
                return r[bool].ok(value=True)
            except ValueError as e:
                return r[bool].fail(str(e))

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str, version: str | None = None
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
                }
            )
            _ = FlextContext.Variables.OperationStartTime.get()
            _ = FlextContext.Variables.OperationMetadata.get()
            _ = FlextContext.Variables.OperationName.get()
            start_token = FlextContext.Variables.OperationStartTime.set(start_time)
            metadata_token = FlextContext.Variables.OperationMetadata.set(
                operation_metadata
            )
            operation_token = None
            if operation_name:
                operation_token = FlextContext.Variables.OperationName.set(
                    operation_name
                )
            try:
                yield operation_metadata
            finally:
                end_time = u.generate_datetime_utc()
                duration = (end_time - start_time).total_seconds()
                operation_metadata.update({
                    c.Context.METADATA_KEY_END_TIME: end_time.isoformat(),
                    c.Context.METADATA_KEY_DURATION_SECONDS: duration,
                })
                FlextContext.Variables.OperationStartTime.reset(start_token)
                FlextContext.Variables.OperationMetadata.reset(metadata_token)
                if operation_token:
                    FlextContext.Variables.OperationName.reset(operation_token)

    class Serialization:
        """Context serialization and deserialization utilities."""

        @staticmethod
        def get_full_context() -> Mapping[str, t.NormalizedValue]:
            """Get current context as dictionary."""
            context_vars = FlextContext.Variables
            operation_metadata_raw = context_vars.Performance.OPERATION_METADATA.get()
            operation_metadata_value: t.NormalizedValue = (
                FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(operation_metadata_raw)
                )
                if operation_metadata_raw is not None
                else ""
            )
            raw_context: dict[str, t.NormalizedValue] = {
                c.Context.KEY_CORRELATION_ID: context_vars.Correlation.CORRELATION_ID.get(),
                c.Context.KEY_PARENT_CORRELATION_ID: context_vars.Correlation.PARENT_CORRELATION_ID.get(),
                c.Context.KEY_SERVICE_NAME: context_vars.Service.SERVICE_NAME.get(),
                c.Context.KEY_SERVICE_VERSION: context_vars.Service.SERVICE_VERSION.get(),
                c.Context.KEY_USER_ID: context_vars.Request.USER_ID.get(),
                c.Context.KEY_OPERATION_NAME: context_vars.Performance.OPERATION_NAME.get(),
                c.Context.KEY_REQUEST_ID: context_vars.Request.REQUEST_ID.get(),
                c.Context.KEY_OPERATION_START_TIME: st.isoformat()
                if isinstance(
                    (st := context_vars.Performance.OPERATION_START_TIME.get()),
                    datetime,
                )
                else None,
                c.Context.KEY_OPERATION_METADATA: operation_metadata_value,
            }
            return {k: v for k, v in raw_context.items() if v is not None}

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
            FlextContext.Correlation.set_correlation_id(new_correlation_id)
            return new_correlation_id


__all__: list[str] = ["FlextContext"]
