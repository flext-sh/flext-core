"""Context CRUD operations (get/set/has/remove/clear/items/keys/values).

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
from collections.abc import Mapping
from typing import ClassVar, overload

from flext_core import FlextRuntime, c, e, m, p, r, t, u
from flext_core._utilities.context_scope import FlextUtilitiesContextScope


class FlextUtilitiesContextCrud(FlextUtilitiesContextScope):
    """CRUD operations on context scopes for FlextContext."""

    _logger: ClassVar[p.Logger]
    _state: m.ContextRuntimeState

    @staticmethod
    def _propagate_to_logger(
        key: str,
        value: t.RuntimeAtomic | t.ValueOrModel,
        scope: str,
    ) -> None:
        """Propagate context changes to the public logging DSL."""
        if scope == c.ContextScope.GLOBAL:
            normalized = u.normalize_to_container(value)
            _ = u.bind_global_context(**{key: normalized})

    @staticmethod
    def _validate_update_inputs(
        key: str,
        value: t.RuntimeAtomic | t.ValueOrModel | t.ConfigMap,
    ) -> p.Result[bool]:
        """Validate inputs for set operation."""
        if not key:
            return r[bool].fail_op(
                "validate context key",
                c.ERR_CONTEXT_KEY_NON_EMPTY_STRING_REQUIRED,
            )
        if value is None:
            return r[bool].fail_op(
                "validate context value",
                c.ERR_CONTEXT_VALUE_CANNOT_BE_NONE,
            )
        value_for_guard: t.RuntimeAtomic | t.RecursiveContainer | t.ConfigMap = (
            FlextUtilitiesContextCrud._to_normalized(
                u.normalize_to_container(
                    u.normalize_to_metadata(value),
                ),
            )
            if u.pydantic_model(value)
            else value
        )
        if not isinstance(
            value_for_guard,
            (str, int, float, bool, list, dict, t.ConfigMap),
        ):
            return r[bool].fail_op(
                "validate context value serializable",
                c.ERR_CONTEXT_VALUE_NOT_SERIALIZABLE,
            )
        return r[bool].ok(True)

    def clear(self) -> None:
        """Clear all data from the context including metadata."""
        if not self._state.active:
            return
        for scope_name, ctx_var in self._state.scope_vars.items():
            _ = ctx_var.set(t.ConfigMap(root={}))
            if scope_name == c.ContextScope.GLOBAL:
                _ = u.clear_global_context()
        self._state = self._state.model_copy(
            update={"metadata": m.Metadata()},
        ).with_operation_update(c.ContextOperation.CLEAR.value)

    def get(
        self, key: str, scope: str = c.ContextScope.GLOBAL
    ) -> p.Result[t.RuntimeAtomic]:
        """Get a value from the context.

        Fast fail: Returns r[t.Container] - fails if key not found.
        No fallback behavior - use r monadic operations for defaults.
        """
        if not self._state.active:
            return r[t.RuntimeAtomic].fail_op(
                "get context value", c.ERR_CONTEXT_NOT_ACTIVE
            )
        scope_data = self._contextvar_data(scope)
        if key not in scope_data:
            return r[t.RuntimeAtomic].fail_op(
                "resolve context key",
                f"Context key '{key}' not found in scope '{scope}'",
            )
        value = scope_data[key]
        self._update_statistics(c.ContextOperation.GET.value)
        if value is None:
            return r[t.RuntimeAtomic].fail_op(
                "resolve context key value",
                f"Context key '{key}' has None value in scope '{scope}'",
            )

        normalized = u.normalize_to_container(value)
        return r[t.RuntimeAtomic].ok(normalized)

    def resolve_metadata(self, key: str) -> p.Result[t.RuntimeAtomic]:
        """Get metadata from the context."""
        if key not in self._state.metadata.attributes:
            return r[t.RuntimeAtomic].fail_op(
                "resolve context metadata",
                c.ERR_CONTEXT_METADATA_KEY_NOT_FOUND.format(key=key),
            )
        raw_value: t.MetadataValue = self._state.metadata.attributes[key]
        normalized_value = u.normalize_to_container(raw_value)
        return r[t.RuntimeAtomic].ok(normalized_value)

    def has(self, key: str, scope: str = c.ContextScope.GLOBAL) -> bool:
        """Check if a key exists in the context."""
        if not self._state.active:
            return False
        scope_data = self._contextvar_data(scope)
        return key in scope_data

    def items(self) -> list[tuple[str, t.RecursiveContainer]]:
        """Get all items (key-value pairs) in the context."""
        if not self._state.active:
            return []
        return [
            item
            for ctx_var in self._state.scope_vars.values()
            for item in self._narrow_contextvar_to_configuration_dict(
                ctx_var.get(),
            ).items()
        ]

    def iter_scope_vars(
        self,
    ) -> Mapping[str, contextvars.ContextVar[t.ConfigMap | None]]:
        """Get scope context variables for iteration."""
        return self._state.scope_vars

    def keys(self) -> t.StrSequence:
        """Get all keys in the context."""
        if not self._state.active:
            return list[str]()
        all_keys: set[str] = set()
        for ctx_var in self._state.scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_keys.update(scope_dict.keys())
        return list(all_keys)

    def values(self) -> t.RecursiveContainerList:
        """Get all values in the context."""
        if not self._state.active:
            empty_values: t.RecursiveContainerList = []
            return empty_values
        all_values: t.MutableRecursiveContainerList = []
        for ctx_var in self._state.scope_vars.values():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            all_values.extend(scope_dict.values())
        return all_values

    def remove(self, key: str, scope: str = c.ContextScope.GLOBAL) -> None:
        """Remove a key from the context."""
        if not self._state.active:
            return
        ctx_var = self._scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key in current:
            filtered = {k: v for k, v in current.items() if k != key}
            try:
                _ = ctx_var.set(t.ConfigMap(root=dict(filtered)))
            except (TypeError, ValueError, AttributeError) as exc:
                self._logger.debug(
                    "Failed to validate context after removal",
                    exc_info=exc,
                )
            self._update_statistics(c.ContextOperation.REMOVE.value)

    @overload
    def set(
        self,
        key_or_data: str,
        value: t.RuntimeAtomic,
        *,
        scope: str = ...,
    ) -> p.Result[bool]: ...

    @overload
    def set(
        self,
        key_or_data: t.ConfigMap,
        value: None = ...,
        *,
        scope: str = ...,
    ) -> p.Result[bool]: ...

    def set(
        self,
        key_or_data: str | t.ConfigMap,
        value: t.RuntimeAtomic | None = None,
        *,
        scope: str = c.ContextScope.GLOBAL,
    ) -> p.Result[bool]:
        """Set one or many values in the context."""
        if not self._state.active:
            return r[bool].fail_op("set context value", c.ERR_CONTEXT_NOT_ACTIVE)
        if isinstance(key_or_data, t.ConfigMap):
            return self._apply_bulk(key_or_data, scope)
        return self._apply_single(key_or_data, value, scope)

    def apply_metadata(self, key: str, value: t.MetadataValue) -> None:
        """Set metadata for the context."""
        normalized_value: t.MetadataValue = u.normalize_to_metadata(value)
        updated_attributes = dict(self._state.metadata.attributes)
        updated_attributes[key] = normalized_value
        self._state = self._state.model_copy(
            update={
                "metadata": self._state.metadata.model_copy(
                    update={c.FIELD_ATTRIBUTES: updated_attributes},
                ),
            },
        )

    def validate_context(self) -> p.Result[bool]:
        """Validate the context data."""
        if not self._state.active:
            return r[bool].fail_op("validate context", c.ERR_CONTEXT_NOT_ACTIVE)
        for ctx_var in self._state.scope_vars.values():
            try:
                scope_dict = self._narrow_contextvar_to_configuration_dict(
                    ctx_var.get(),
                )
            except TypeError as exc:
                return e.fail_operation("validate context scope", exc)
            for key in scope_dict:
                if not key:
                    return r[bool].fail_op(
                        "validate context key",
                        c.ERR_CONTEXT_INVALID_KEY_FOUND,
                    )
        return r[bool].ok(True)

    def _apply_bulk(self, data: t.ConfigMap, scope: str) -> p.Result[bool]:
        """Set multiple values in the context from a ConfigMap."""
        if not data:
            return r[bool].ok(True)
        try:
            ctx_var = self._scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = t.ConfigMap(root=dict(current))
            updated.update(data.root)
            _ = ctx_var.set(updated)
            self._update_statistics(c.ContextOperation.SET.value)
            self._execute_hooks(
                c.ContextOperation.SET.value,
                t.ConfigMap(root={c.Directory.DATA: t.ConfigMap(root=data.root)}),
            )
            return r[bool].ok(True)
        except TypeError as exc:
            return e.fail_operation("apply bulk context update", exc)

    def _apply_single(
        self,
        key: str,
        value: t.RuntimeAtomic | None,
        scope: str,
    ) -> p.Result[bool]:
        """Set a single key-value pair in the context."""
        if value is None:
            return r[bool].fail_op(
                "set single context value",
                c.ERR_CONTEXT_SINGLE_KEY_VALUE_REQUIRED,
            )
        validation_result = FlextUtilitiesContextCrud._validate_update_inputs(
            key,
            value,
        )
        if validation_result.failure:
            return r[bool].fail_op(
                "validate context update inputs",
                validation_result.error or c.ERR_VALIDATION_FAILED,
            )
        try:
            ctx_var = self._scope_var(scope)
            current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            updated = t.ConfigMap(root=dict(current))
            normalized_value = u.normalize_to_container(value)
            if isinstance(normalized_value, (t.ConfigMap, t.Dict)):
                updated[key] = FlextRuntime.to_plain_container(normalized_value)
            else:
                updated[key] = FlextRuntime.to_plain_container(normalized_value)
            _ = ctx_var.set(updated)
            FlextUtilitiesContextCrud._propagate_to_logger(key, value, scope)
            self._update_statistics(c.ContextOperation.SET.value)
            self._execute_hooks(
                c.ContextOperation.SET.value,
                t.ConfigMap(root={"key": key, "value": value}),
            )
            return r[bool].ok(True)
        except TypeError as exc:
            return e.fail_operation("apply single context key", exc)


__all__: list[str] = ["FlextUtilitiesContextCrud"]
