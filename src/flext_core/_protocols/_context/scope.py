# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/scope
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase, FlextProtocolsPydantic

if TYPE_CHECKING:
    from flext_core import t
    from flext_core._protocols import FlextProtocolsContainer


class FlextProtocolsContextScope:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextStatistics(FlextProtocolsPydantic.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def sets(self) -> t.NonNegativeInt:
            """Number of set operations."""
            ...

        @property
        def gets(self) -> t.NonNegativeInt:
            """Number of get operations."""
            ...

        @property
        def removes(self) -> t.NonNegativeInt:
            """Number of remove operations."""
            ...

        @property
        def clears(self) -> t.NonNegativeInt:
            """Number of clear operations."""
            ...

        @property
        def operations(self) -> t.JsonMapping:
            """Additional metric counters and timing values grouped by metric key."""
            ...

    @runtime_checkable
    class ContextRuntimeState(FlextProtocolsPydantic.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def metadata(self) -> FlextProtocolsBase.Metadata:
            """Normalized metadata bound to the active context."""
            ...

        @property
        def hooks(self) -> t.ContextHookMap:
            """Lifecycle hooks keyed by event name."""
            ...

        @property
        def statistics(self) -> FlextProtocolsContextScope.ContextStatistics:
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
            ContextVar[FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None],
        ]:
            """ContextVar registry keyed by scope name."""
            ...

    @runtime_checkable
    class ContextContainerState(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        container: FlextProtocolsContainer.Container | None

        @property
        def configured(self) -> bool:
            """Whether a container is configured for service access."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextScope",
]
