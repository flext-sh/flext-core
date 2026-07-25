"""Context ``set`` overloads and implementation (extracted for LOC cap)."""

from __future__ import annotations

from typing import overload

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
)
from flext_core._utilities.context_state import FlextUtilitiesContextState
from flext_core._utilities.model import FlextUtilitiesModel


class FlextUtilitiesContextCrudSetMixin(FlextUtilitiesContextState):
    """Bulk/single value assignment logic for FlextUtilitiesContextCrud."""

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
        operation_result: p.Result[bool] = r[bool].ok(True)
        prepared_update: tuple[t.JsonMapping, t.JsonMapping] | None = None
        if not self.state.active:
            operation_result = r[bool].fail_op(
                c.ContextCrudOperation.SET_VALUE,
                c.ERR_CONTEXT_NOT_ACTIVE,
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
                            {"key": key, "value": normalized_value},
                        )
                case data, _ if not data:
                    operation_result = r[bool].ok(True)
                case payload_mapping, _:
                    mapping_payload = t.json_mapping_adapter().validate_python(
                        payload_mapping,
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
                            {c.Directory.DATA.value: normalized_mapping_result.value},
                        )

        if prepared_update is not None and operation_result.success:
            payload, hook_context = prepared_update
            try:
                self._update_contextvar(scope, payload)
                self._update_statistics(c.ContextOperation.SET.value)
                self._execute_hooks(c.ContextOperation.SET.value, hook_context)
            except TypeError as exc:
                operation_result = e.fail_operation(
                    c.ContextCrudOperation.APPLY_UPDATE,
                    exc,
                )
        return operation_result


__all__: list[str] = ["FlextUtilitiesContextCrudSetMixin"]
