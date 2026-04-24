"""Context state primitives: normalization, scope access, metadata mapping.

Base MRO layer for FlextContext providing contextvar payload normalization,
scope variable access, statistics/hooks bookkeeping, and metadata projection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from typing import ClassVar

from flext_core import FlextRuntime, c, m, p, t, u


class FlextUtilitiesContextState:
    """Normalization + scope access primitives used by FlextContext."""

    logger: ClassVar[p.Logger]
    state: m.ContextRuntimeState

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: m.ConfigMap | Mapping[str, t.JsonPayload] | p.Model | None,
    ) -> t.JsonMapping:
        """Return contextvar payload as a flat container mapping with safe default."""
        if ctx_value is None:
            return {}

        payload: Mapping[str, t.JsonPayload]
        if isinstance(ctx_value, m.ConfigMap):
            payload = ctx_value.root
        elif isinstance(ctx_value, p.Model):
            dumped = ctx_value.model_dump(mode="python")
            payload = t.flat_container_mapping_adapter().validate_python(dumped)
        else:
            payload = ctx_value

        try:
            normalized: dict[str, t.JsonPayload] = {}
            for key, value in payload.items():
                if str(key) != key:
                    return {}
                normalized[key] = FlextRuntime.normalize_to_container(value)
            validated: t.JsonMapping = (
                t.flat_container_mapping_adapter().validate_python(normalized)
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

    def _scope_payloads(self) -> Mapping[str, t.JsonMapping]:
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
            normalized_context: Mapping[str, t.JsonPayload] = {
                key: FlextRuntime.normalize_to_container(value)
                for key, value in incoming.items()
                if value is not None
            }
            _ = u.bind_global_context(**normalized_context)

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper)."""
        self.state = self.state.with_operation_update(operation)

    def _execute_hooks(
        self,
        event: str,
        event_data: t.JsonPayload | Mapping[str, t.JsonPayload],
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

    def _metadata_map(self) -> Mapping[str, t.JsonPayload]:
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
            if v is None or v == {}:
                continue
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, (str, int, float, bool, list, dict)):
                result[k] = v
            elif u.pydantic_model(v):
                result[k] = FlextRuntime.normalize_to_container(v)
            else:
                result[k] = str(v)
        result.update(custom_fields_dict)
        return result


__all__: list[str] = ["FlextUtilitiesContextState"]
