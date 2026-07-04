"""Context state primitives: normalization, scope access, metadata mapping.

Base MRO layer for FlextContext providing contextvar payload normalization,
scope variable access, statistics/hooks bookkeeping, and metadata projection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from flext_core import (
    FlextConstants as c,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesLoggingContext,
)

if TYPE_CHECKING:
    import contextvars
    from collections.abc import MutableMapping


class FlextUtilitiesContextState:
    """Normalization + scope access primitives used by FlextContext."""

    logger: ClassVar[p.Logger]
    state: m.ContextRuntimeState

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: m.ConfigMap | t.MappingKV[str, t.JsonPayload] | p.Model | None,
    ) -> t.JsonMapping:
        """Return contextvar payload as a flat container mapping with safe default."""
        try:
            normalized = FlextRuntime.normalize_model_input_mapping(ctx_value)
            if normalized is None:
                empty_normalized_context: t.JsonMapping = {}
                return empty_normalized_context
            return normalized
        except c.EXC_ATTR_KEY_TYPE_VALUE as exc:
            FlextUtilitiesContextState.logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
            empty_context: t.JsonMapping = {}
            return empty_context

    def _scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[m.ConfigMap | None]:
        """Get or create contextvar for scope."""
        state, scope_var = self.state.resolve_scope_var(scope)
        self.state = state
        resolved_scope_var: contextvars.ContextVar[m.ConfigMap | None] = scope_var
        return resolved_scope_var

    def _contextvar_data(self, scope: str) -> t.JsonMapping:
        """Get all values from contextvar scope."""
        ctx_var = self._scope_var(scope)
        value = ctx_var.get()
        return dict(
            FlextUtilitiesContextState._narrow_contextvar_to_configuration_dict(
                value,
            ).items(),
        )

    def _scope_payloads(self) -> t.MappingKV[str, t.JsonMapping]:
        """Get all scope registrations."""
        if not self.state.active:
            empty_scopes: dict[str, t.JsonMapping] = {}
            return empty_scopes
        scopes: MutableMapping[str, t.JsonMapping] = {}
        for scope_name, ctx_var in self.state.scope_vars.items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                scopes[scope_name] = dict(scope_dict)
        return scopes

    def _update_contextvar(
        self,
        scope: str,
        data: m.ConfigMap | t.JsonMapping,
    ) -> None:
        """Set multiple values in contextvar scope."""
        ctx_var = self._scope_var(scope)
        incoming = self._narrow_contextvar_to_configuration_dict(data)
        current = m.ConfigMap(
            root=dict(
                self._narrow_contextvar_to_configuration_dict(ctx_var.get()),
            ),
        )
        updated = current.model_copy(
            update={"root": {**current.root, **incoming}},
        )
        _ = ctx_var.set(updated)
        if scope == c.ContextScope.GLOBAL:
            normalized_context: t.MappingKV[str, t.JsonPayload] = {
                key: FlextRuntime.normalize_to_container(value)
                for key, value in incoming.items()
                if value is not None
            }
            _ = FlextUtilitiesLoggingContext.bind_global_context(
                **normalized_context,
            )

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper)."""
        self.state = self.state.with_operation_update(operation)

    def _execute_hooks(
        self,
        event: str,
        event_data: t.JsonPayload | t.MappingKV[str, t.JsonPayload],
    ) -> None:
        """Execute hooks for an event (DRY helper)."""
        if event not in self.state.hooks:
            return
        hooks = self.state.hooks[event]
        for hook in hooks:
            if callable(hook):
                hook_data: t.Scalar
                if event_data is None:
                    hook_data = ""
                elif isinstance(event_data, t.SCALAR_TYPES):
                    hook_data = event_data
                else:
                    hook_data = str(event_data)
                _ = hook(hook_data)

    def _metadata_map(self) -> t.MappingKV[str, t.JsonPayload]:
        """Get all metadata from the context."""
        data = self.state.metadata.model_dump()
        custom_fields_raw = data.pop("custom_fields", {})
        custom_fields_dict: dict[str, t.JsonPayload] = {}
        try:
            cf_map = m.ConfigMap.model_validate(custom_fields_raw)
            for ck, cv in cf_map.items():
                custom_fields_dict[ck] = FlextRuntime.normalize_to_container(cv)
        except c.EXC_BASIC_TYPE as exc:
            self.logger.debug(
                "Custom metadata field normalization failed",
                exc_info=exc,
            )
            custom_fields_dict = {}
        result: dict[str, t.JsonPayload] = {}
        for k, v in data.items():
            if v is None or (FlextUtilitiesGuardsTypeCore.mapping(v) and not v):
                continue
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, (str, int, float, bool, list, dict)):
                result[k] = v
            elif FlextUtilitiesGuardsTypeModel.pydantic_model(v):
                result[k] = FlextRuntime.normalize_to_container(v)
            else:
                result[k] = str(v)
        result.update(custom_fields_dict)
        return result

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
            },
        )


__all__: list[str] = ["FlextUtilitiesContextState"]
