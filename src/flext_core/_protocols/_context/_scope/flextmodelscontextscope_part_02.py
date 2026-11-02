# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/_scope/flextmodelscontextscope_part_02
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core.protocols import p


class FlextProtocolsContextScopeFlextmodelscontextscopePart02:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextRuntimeState(Protocol):
        """Structural data contract projected from the owning model."""

        metadata: p.Metadata
        hooks: t.ContextHookMap
        statistics: p.ContextStatistics
        active: bool
        suspended: bool
        scope_vars: Mapping[str, ContextVar[p.MutableRootDict[t.JsonPayload] | None]]


__all__: list[str] = [
    "FlextProtocolsContextScopeFlextmodelscontextscopePart02",
]
