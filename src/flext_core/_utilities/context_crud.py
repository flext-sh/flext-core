"""Context CRUD operations (get/set/has/remove/clear/items/keys/values).

MRO layer above FlextUtilitiesContextState providing the public contract that
FlextContext exposes to service consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, overload

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
    FlextUtilitiesContextState,
    FlextUtilitiesLoggingContext,
    FlextUtilitiesModel,
)


class FlextUtilitiesContextCrud(FlextUtilitiesContextState):
    """CRUD operations on context scopes for FlextContext."""

    logger: ClassVar[p.Logger]
    state: m.ContextRuntimeState

    def _iter_scoped_dicts(self) -> Iterator[t.JsonMapping]:
        for ctx_var in self.state.scope_vars.values():
            yield self._narrow_contextvar_to_configuration_dict(ctx_var.get())

    def clear(self) -> None:
        """Clear all data from the context including metadata."""
        if not self.state.active:
            return
        for scope_name, ctx_var in self.state.scope_vars.items():
            _ = ctx_var.set(m.ConfigMap(root={}))
            if scope_name == c.ContextScope.GLOBAL:
                _ = FlextUtilitiesLoggingContext.clear_global_context()
        self.state = self.state.model_copy(
            update={"metadata": m.Metadata()},
        ).with_operation_update(c.ContextOperation.CLEAR.value)

    def get(
        self, key: str, scope: str = c.ContextScope.GLOBAL
    ) -> p.Result[t.JsonPayload]:
        """Get a value from the context (fail-fast, no default fallback)."""
        if not self.state.active:
            return r[t.JsonPayload].fail_op(
                c.ContextCrudOperation.GET_VALUE, c.ERR_CONTEXT_NOT_ACTIVE
            )
        scope_data = self._contextvar_data(scope)
        if key not in scope_data:
            return r[t.JsonPayload].fail_op(
                c.ContextCrudOperation.RESOLVE_KEY,
                f"Context key '{key}' not found in scope '{scope}'",
            )
        value = scope_data[key]
        self._update_statistics(c.ContextOperation.GET.value)
        return r[t.JsonPayload].ok(FlextRuntime.normalize_to_container(value))

    def has(self, key: str, scope: str = c.ContextScope.GLOBAL) -> bool:
        """Check if a key exists in the context."""
        if not self.state.active:
            return False
        return key in self._contextvar_data(scope)

    def items(self) -> t.SequenceOf[t.Pair[str, t.JsonValue]]:
        """Get all items (key-value pairs) across scopes."""
        if not self.state.active:
            return []
        return [item for d in self._iter_scoped_dicts() for item in d.items()]

    def keys(self) -> t.StrSequence:
        """Get all keys across scopes."""
        if not self.state.active:
            return list[str]()
        return list({k for d in self._iter_scoped_dicts() for k in d})

    def values(self) -> t.JsonList:
        """Get all values across scopes."""
        if not self.state.active:
            empty_values: t.JsonList = []
            return empty_values
        return [v for d in self._iter_scoped_dicts() for v in d.values()]

    def remove(self, key: str, scope: str = c.ContextScope.GLOBAL) -> None:
        """Remove a key from the context."""
        if not self.state.active:
            return
        ctx_var = self._scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key not in current:
            return
        filtered = {k: v for k, v in current.items() if k != key}
        try:
            _ = ctx_var.set(m.ConfigMap(root=dict(filtered)))
        except c.EXC_BASIC_TYPE as exc:
            self.logger.debug(
                c.LOG_CONTEXT_REMOVAL_FAILED,
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
        operation_result = r[bool].ok(True)
        prepared_update: tuple[t.JsonMapping, t.JsonMapping] | None = None
        if not self.state.active:
            operation_result = r[bool].fail_op(
                c.ContextCrudOperation.SET_VALUE, c.ERR_CONTEXT_NOT_ACTIVE
            )
        else:
            match key_or_data, value:
                case str(), None:
                    operation_result = r[bool].fail_op(
                        c.ContextCrudOperation.SET_SINGLE_VALUE,
                        c.ERR_CONTEXT_SINGLE_KEY_VALUE_REQUIRED,
                    )
                case "", _:
                    operation_result = r[bool].fail_op(
                        c.ContextCrudOperation.VALIDATE_KEY,
                        c.ERR_CONTEXT_KEY_NON_EMPTY_STRING_REQUIRED,
                    )
                case str() as key, raw_value:
                    normalized_value_result: p.Result[t.JsonValue] = (
                        FlextUtilitiesModel.validate_value(
                            t.JsonValue,
                            FlextRuntime.normalize_to_container(raw_value),
                        )
                    )
                    if normalized_value_result.failure:
                        operation_result = r[bool].fail_op(
                            c.ContextCrudOperation.VALIDATE_VALUE,
                            c.ERR_CONTEXT_VALUE_NOT_SERIALIZABLE,
                        )
                    else:
                        normalized_value = normalized_value_result.value
                        prepared_update = (
                            {key: normalized_value},
                            {
                                "key": key,
                                "value": normalized_value,
                            },
                        )
                case data, _ if not data:
                    operation_result = r[bool].ok(True)
                case payload_mapping, _:
                    mapping_payload = t.json_mapping_adapter().validate_python(
                        payload_mapping
                    )
                    normalized_mapping_result: p.Result[t.JsonValue] = (
                        FlextUtilitiesModel.validate_value(
                            t.JsonValue,
                            FlextRuntime.normalize_to_container(dict(mapping_payload)),
                        )
                    )
                    if normalized_mapping_result.failure:
                        operation_result = r[bool].fail_op(
                            c.ContextCrudOperation.VALIDATE_PAYLOAD,
                            c.ERR_CONTEXT_VALUE_NOT_SERIALIZABLE,
                        )
                    else:
                        prepared_update = (
                            mapping_payload,
                            {
                                c.Directory.DATA.value: normalized_mapping_result.value,
                            },
                        )

        if prepared_update is not None and operation_result.success:
            payload, hook_context = prepared_update
            try:
                self._update_contextvar(scope, payload)
                self._update_statistics(c.ContextOperation.SET.value)
                self._execute_hooks(c.ContextOperation.SET.value, hook_context)
            except TypeError as exc:
                operation_result = e.fail_operation(
                    c.ContextCrudOperation.APPLY_UPDATE, exc
                )
        return operation_result


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContextCrud"]
