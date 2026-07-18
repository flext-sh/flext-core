"""Context CRUD operations (get/set/has/remove/clear/items/keys/values).

MRO layer above FlextUtilitiesContextState providing the public contract that
FlextContext exposes to service consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from flext_core import (
    FlextConstants as c,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
)
from flext_core._utilities._logging_context_parts.logging_context_part_01 import (
    FlextUtilitiesLoggingContext,
)
from flext_core._utilities.context_state import FlextUtilitiesContextState

from ._context_crud_set import FlextUtilitiesContextCrudSetMixin


if TYPE_CHECKING:
    import contextvars
    from collections.abc import Iterator


class FlextUtilitiesContextCrud(
    FlextUtilitiesContextCrudSetMixin, FlextUtilitiesContextState
):
    """CRUD operations on context scopes for FlextContext."""

    logger: ClassVar[p.Logger]
    state: p.ContextRuntimeState

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
            update={"metadata": p.Metadata()}
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
        """Get all items across scopes."""
        if not self.state.active:
            empty_items: list[t.Pair[str, t.JsonValue]] = []
            return empty_items
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
        ctx_var: contextvars.ContextVar[p.ConfigMap | None] = self._scope_var(scope)
        current = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
        if key not in current:
            return
        filtered = {k: v for k, v in current.items() if k != key}
        try:
            _ = ctx_var.set(m.ConfigMap.model_validate(filtered))
        except c.EXC_BASIC_TYPE as exc:
            self.logger.debug(c.LOG_CONTEXT_REMOVAL_FAILED, exc_info=exc)
        self._update_statistics(c.ContextOperation.REMOVE.value)


__all__: list[str] = ["FlextUtilitiesContextCrud"]
