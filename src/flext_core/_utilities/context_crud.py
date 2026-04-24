"""Context CRUD operations (get/set/has/remove/clear/items/keys/values).

MRO layer above FlextUtilitiesContextState providing the public contract that
FlextContext exposes to service consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, overload

from flext_core import (
    FlextRuntime,
    FlextUtilitiesContextState,
    c,
    e,
    m,
    p,
    r,
    t,
    u,
)


class FlextUtilitiesContextCrud(FlextUtilitiesContextState):
    """CRUD operations on context scopes for FlextContext."""

    logger: ClassVar[p.Logger]
    state: m.ContextRuntimeState

    def clear(self) -> None:
        """Clear all data from the context including metadata."""
        if not self.state.active:
            return
        for scope_name, ctx_var in self.state.scope_vars.items():
            _ = ctx_var.set(m.ConfigMap(root={}))
            if scope_name == c.ContextScope.GLOBAL:
                _ = u.clear_global_context()
        self.state = self.state.model_copy(
            update={"metadata": m.Metadata()},
        ).with_operation_update(c.ContextOperation.CLEAR.value)

    def get(
        self, key: str, scope: str = c.ContextScope.GLOBAL
    ) -> p.Result[t.JsonPayload]:
        """Get a value from the context (fail-fast, no default fallback)."""
        if not self.state.active:
            return r[t.JsonPayload].fail_op(
                "get context value", c.ERR_CONTEXT_NOT_ACTIVE
            )
        scope_data = self._contextvar_data(scope)
        if key not in scope_data:
            return r[t.JsonPayload].fail_op(
                "resolve context key",
                f"Context key '{key}' not found in scope '{scope}'",
            )
        value = scope_data[key]
        self._update_statistics(c.ContextOperation.GET.value)
        if value is None:
            return r[t.JsonPayload].fail_op(
                "resolve context key value",
                f"Context key '{key}' has None value in scope '{scope}'",
            )
        return r[t.JsonPayload].ok(FlextRuntime.normalize_to_container(value))

    def resolve_metadata(self, key: str) -> p.Result[t.JsonPayload]:
        """Get metadata from the context."""
        if key not in self.state.metadata.attributes:
            return r[t.JsonPayload].fail_op(
                "resolve context metadata",
                c.ERR_CONTEXT_METADATA_KEY_NOT_FOUND.format(key=key),
            )
        raw_value: t.JsonValue = self.state.metadata.attributes[key]
        return r[t.JsonPayload].ok(FlextRuntime.normalize_to_container(raw_value))

    def apply_metadata(self, key: str, value: t.JsonValue) -> None:
        """Set metadata via Pydantic immutable copy DSL."""
        meta = self.state.metadata
        self.state = self.state.model_copy(
            update={
                "metadata": meta.model_copy(
                    update={"attributes": {**meta.attributes, key: value}},
                ),
            }
        )

    def has(self, key: str, scope: str = c.ContextScope.GLOBAL) -> bool:
        """Check if a key exists in the context."""
        if not self.state.active:
            return False
        return key in self._contextvar_data(scope)

    def items(self) -> Sequence[tuple[str, t.JsonValue]]:
        """Get all items (key-value pairs) across scopes."""
        if not self.state.active:
            return []
        return [
            item
            for ctx_var in self.state.scope_vars.values()
            for item in self._narrow_contextvar_to_configuration_dict(
                ctx_var.get(),
            ).items()
        ]

    def keys(self) -> t.StrSequence:
        """Get all keys across scopes."""
        if not self.state.active:
            return list[str]()
        all_keys: set[str] = set()
        for ctx_var in self.state.scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_keys.update(scope_dict.keys())
        return list(all_keys)

    def values(self) -> t.JsonList:
        """Get all values across scopes."""
        if not self.state.active:
            empty_values: t.JsonList = []
            return empty_values
        all_values: list[t.JsonValue] = []
        for ctx_var in self.state.scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_values.extend(scope_dict.values())
        return all_values

    def remove(self, key: str, scope: str = c.ContextScope.GLOBAL) -> None:
        """Remove a key from the context."""
        if not self.state.active:
            return
        ctx_var = self._scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key in current:
            filtered = {k: v for k, v in current.items() if k != key}
            try:
                _ = ctx_var.set(m.ConfigMap(root=dict(filtered)))
            except (TypeError, ValueError, AttributeError) as exc:
                self.logger.debug(
                    "Failed to validate context after removal",
                    exc_info=exc,
                )
            self._update_statistics(c.ContextOperation.REMOVE.value)

    @overload
    def set(
        self,
        key_or_data: str,
        value: t.JsonPayload,
        *,
        scope: str = ...,
    ) -> p.Result[bool]: ...

    @overload
    def set(
        self,
        key_or_data: t.JsonMapping,
        value: None = ...,
        *,
        scope: str = ...,
    ) -> p.Result[bool]: ...

    def set(
        self,
        key_or_data: str | t.JsonMapping,
        value: t.JsonPayload | None = None,
        *,
        scope: str = c.ContextScope.GLOBAL,
    ) -> p.Result[bool]:
        """Set one or many values in the context."""
        if not self.state.active:
            return r[bool].fail_op("set context value", c.ERR_CONTEXT_NOT_ACTIVE)
        if not isinstance(key_or_data, str):
            return self._apply_bulk(key_or_data, scope)
        return self._apply_single(key_or_data, value, scope)

    def _apply_bulk(
        self,
        data: t.JsonMapping,
        scope: str,
    ) -> p.Result[bool]:
        """Set multiple values in the context from a ConfigMap."""
        if not data:
            return r[bool].ok(True)
        try:
            self._update_contextvar(scope, data)
            self._update_statistics(c.ContextOperation.SET.value)
            self._execute_hooks(
                c.ContextOperation.SET.value,
                {c.Directory.DATA: FlextRuntime.normalize_to_container(dict(data))},
            )
            return r[bool].ok(True)
        except TypeError as exc:
            return e.fail_operation("apply bulk context update", exc)

    def _apply_single(
        self,
        key: str,
        value: t.JsonPayload | None,
        scope: str,
    ) -> p.Result[bool]:
        """Set a single key-value pair in the context."""
        if value is None:
            return r[bool].fail_op(
                "set single context value",
                c.ERR_CONTEXT_SINGLE_KEY_VALUE_REQUIRED,
            )
        if not key:
            return r[bool].fail_op(
                "validate context key",
                c.ERR_CONTEXT_KEY_NON_EMPTY_STRING_REQUIRED,
            )
        normalized_value = FlextRuntime.normalize_to_container(value)
        if not isinstance(normalized_value, t.CONTAINER_AND_COLLECTION_TYPES):
            return r[bool].fail_op(
                "validate context value serializable",
                c.ERR_CONTEXT_VALUE_NOT_SERIALIZABLE,
            )
        try:
            self._update_contextvar(scope, {key: normalized_value})
            self._update_statistics(c.ContextOperation.SET.value)
            self._execute_hooks(
                c.ContextOperation.SET.value,
                {"key": key, "value": normalized_value},
            )
            return r[bool].ok(True)
        except TypeError as exc:
            return e.fail_operation("apply single context key", exc)


__all__: list[str] = ["FlextUtilitiesContextCrud"]
