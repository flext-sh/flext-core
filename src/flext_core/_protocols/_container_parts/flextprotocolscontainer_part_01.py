"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flext_core import p, t


class FlextProtocolsContainer:
    """Protocols for DI container behavior."""

    @runtime_checkable
    class ServiceRegistrationSpec(Protocol):
        """Complete typed input validated by the canonical registration model."""

        settings: p.Settings | None
        context: p.Context | None
        dispatcher: p.Dispatcher | None
        services: t.ServiceMap | None
        factories: t.FactoryMap | None
        resources: t.ResourceMap | None

        @property
        def user_overrides(self) -> p.Model | t.UserOverridesMapping | None: ...

        @property
        def container_config(self) -> p.Model | None: ...

    @runtime_checkable
    class RootDict[RootValueT](Protocol):
        """Protocol for dict-like root model objects.

        Represents the structure of Pydantic RootModel and similar
        objects that wrap a dict with a root attribute.
        """

        @property
        def root(self) -> t.MappingKV[str, RootValueT]: ...

    @runtime_checkable
    class MutableRootDict[RootValueT](Protocol):
        """Structural contract for mutable validated root mappings."""

        @property
        def root(self) -> t.MutableMappingKV[str, RootValueT]: ...


__all__: list[str] = ["FlextProtocolsContainer"]
