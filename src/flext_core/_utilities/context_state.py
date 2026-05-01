"""Context state primitives: normalization, scope access, metadata mapping.

Base MRO layer for FlextContext providing contextvar payload normalization,
scope variable access, statistics/hooks bookkeeping, and metadata projection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
from collections.abc import MutableMapping
from datetime import datetime
from typing import ClassVar

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


class FlextUtilitiesContextState:
    """Normalization + scope access primitives used by FlextContext."""

    logger: ClassVar[p.Logger]
    state: m.ContextRuntimeState

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: m.ConfigMap | t.MappingKV[str, t.JsonPayload] | p.Model | None,
    ) -> t.JsonMapping:
        """Return contextvar payload as a flat container mapping with safe default."""
        if ctx_value is None:
            return {}

        payload: t.MappingKV[str, t.JsonPayload]
        if isinstance(ctx_value, m.ConfigMap):
            payload = ctx_value.root
        elif isinstance(ctx_value, p.Model):
            dumped = ctx_value.model_dump(mode="python")
            payload = t.json_mapping_adapter().validate_python(dumped)
        else:
            payload = ctx_value

        try:
            normalized: dict[str, t.JsonPayload] = {}
            for key, value in payload.items():
                normalized[key] = FlextRuntime.normalize_to_container(value)
            validated: t.JsonMapping = t.json_mapping_adapter().validate_python(
                normalized
            )
            return validated
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            FlextUtilitiesContextState.logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
            return {}

    def _scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[m.ConfigMap | None]:
        """Get or create contextvar for scope."""
        self.state, scope_var = self.state.resolve_scope_var(scope)
        return scope_var

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
            return {}
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
        updated = current.model_copy()
        updated.update(dict(incoming))
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
                elif isinstance(event_data, (str, int, float, bool, datetime)):
                    hook_data = event_data
                else:
                    hook_data = str(event_data)
                _ = hook(hook_data)

    def _metadata_map(self) -> t.MappingKV[str, t.JsonPayload]:
        """Get all metadata from the context."""
        data = dict(self.state.metadata.model_dump())
        custom_fields_raw = data.pop("custom_fields", {})
        custom_fields_dict: dict[str, t.JsonPayload] = {}
        try:
            cf_map = m.ConfigMap(root=dict(custom_fields_raw))
            for ck, cv in cf_map.items():
                custom_fields_dict[ck] = FlextRuntime.normalize_to_container(cv)
        except (TypeError, ValueError, AttributeError) as exc:
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
            }
        )


__all__: list[str] = ["FlextUtilitiesContextState"]
