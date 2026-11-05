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

import flext_core
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContainer
    from flext_core._protocols._context._scope import (
        FlextProtocolsContextScopeFlextmodelscontextscopePart01,
    )


class FlextProtocolsContextScopeFlextmodelscontextscopePart02:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextRuntimeState(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def metadata(self) -> FlextProtocolsBase.Metadata:
            """Normalized metadata bound to the active context."""
            ...

        @property
        def hooks(self) -> flext_core.t.ContextHookMap:
            """Lifecycle hooks keyed by event name."""
            ...

        @property
        def statistics(
            self,
        ) -> FlextProtocolsContextScopeFlextmodelscontextscopePart01.ContextStatistics:
            """Operation counters for this context instance."""
            ...

        @property
        def active(self) -> bool:
            """Whether the context accepts operations."""
            ...

        @property
        def suspended(self) -> bool:
            """Whether the context is temporarily suspended."""
            ...

        @property
        def scope_vars(
            self,
        ) -> Mapping[
            str,
            ContextVar[
                FlextProtocolsContainer.MutableRootDict[flext_core.t.JsonPayload] | None
            ],
        ]:
            """ContextVar registry keyed by scope name."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextScopeFlextmodelscontextscopePart02",
]
