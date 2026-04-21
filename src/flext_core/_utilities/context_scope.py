"""Context scope and state management helpers.

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
from collections.abc import (
    Mapping,
    MutableMapping,
)
from datetime import datetime
from typing import ClassVar

from flext_core import (
    FlextRuntime,
    FlextUtilitiesContextNormalization,
    c,
    m,
    p,
    t,
    u,
)


class FlextUtilitiesContextScope(FlextUtilitiesContextNormalization):
    """Scope variable access, update, and statistics helpers for FlextContext."""

    _logger: ClassVar[p.Logger]
    _state: m.ContextRuntimeState

    def _scope_var(
        self,
        scope: str,
    ) -> contextvars.ContextVar[m.ConfigMap | None]:
        """Get or create contextvar for scope."""
        self._state, scope_var = self._state.resolve_scope_var(scope)
        return scope_var

    def _contextvar_data(self, scope: str) -> t.FlatContainerMapping:
        """Get all values from contextvar scope."""
        ctx_var = self._scope_var(scope)
        value = ctx_var.get()
        return dict(
            FlextUtilitiesContextScope._narrow_contextvar_to_configuration_dict(
                value,
            ).items(),
        )

    def _scope_payloads(self) -> Mapping[str, t.FlatContainerMapping]:
        """Get all scope registrations."""
        if not self._state.active:
            return {}
        scopes: MutableMapping[str, t.FlatContainerMapping] = {}
        for scope_name, ctx_var in self._state.scope_vars.items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                scopes[scope_name] = dict(scope_dict)
        return scopes

    def _update_contextvar(
        self,
        scope: str,
        data: m.ConfigMap | Mapping[str, t.Container],
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
            normalized_context: Mapping[str, t.RuntimeData] = {
                key: FlextRuntime.normalize_to_container(value)
                for key, value in incoming.items()
                if value is not None
            }
            _ = u.bind_global_context(**normalized_context)

    def _update_statistics(self, operation: str) -> None:
        """Update statistics counter for an operation (DRY helper)."""
        self._state = self._state.with_operation_update(operation)

    def _execute_hooks(
        self,
        event: str,
        event_data: t.RuntimeData | Mapping[str, t.RuntimeData],
    ) -> None:
        """Execute hooks for an event (DRY helper)."""
        if event not in self._state.hooks:
            return
        hooks = self._state.hooks[event]
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

    def _metadata_map(self) -> Mapping[str, t.RuntimeData]:
        """Get all metadata from the context."""
        data = dict(self._state.metadata.model_dump())
        custom_fields_raw = data.pop("custom_fields", {})
        custom_fields_dict: dict[str, t.RuntimeData] = {}
        try:
            cf_map = m.ConfigMap(root=dict(custom_fields_raw))
            for ck, cv in cf_map.items():
                custom_fields_dict[ck] = FlextRuntime.normalize_to_container(cv)
        except (TypeError, ValueError, AttributeError) as exc:
            self._logger.debug(
                "Custom metadata field normalization failed",
                exc_info=exc,
            )
            custom_fields_dict = {}
        result: dict[str, t.RuntimeData] = {}
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


__all__: list[str] = ["FlextUtilitiesContextScope"]
