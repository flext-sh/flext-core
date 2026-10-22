"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsSettings,
)
from flext_core._protocols._context_parts.flextprotocolscontext_part_01 import (
    FlextProtocolsContextPart01,
)


class FlextProtocolsContainerPart01:
    """Protocols for DI container behavior."""

    type RegistryCollaborator = (
        FlextProtocolsBase.Model
        | FlextProtocolsLogging.Logger
        | FlextProtocolsSettings.Settings
        | FlextProtocolsContextPart01.Context
        | FlextProtocolsHandler.Dispatcher
    )
    type ServiceValue = t.ServiceValue[RegistryCollaborator]
    type RegisterableService = t.RegisterableService[RegistryCollaborator]
    type FactoryCallable = t.FactoryCallable[RegistryCollaborator]
    type ResourceCallable = t.ResourceCallable[RegistryCollaborator]
    type ServiceMap = t.ServiceMap[RegistryCollaborator]
    type FactoryMap = t.FactoryMap[RegistryCollaborator]
    type ResourceMap = t.ResourceMap[RegistryCollaborator]

    @runtime_checkable
    class ServiceRegistrationSpec(Protocol):
        """Complete typed input validated by the canonical registration model."""

        settings: FlextProtocolsSettings.Settings | None
        context: FlextProtocolsContextPart01.Context | None
        dispatcher: FlextProtocolsHandler.Dispatcher | None
        services: FlextProtocolsContainerPart01.ServiceMap | None
        factories: FlextProtocolsContainerPart01.FactoryMap | None
        resources: FlextProtocolsContainerPart01.ResourceMap | None

        @property
        def user_overrides(
            self,
        ) -> FlextProtocolsBase.Model | t.UserOverridesMapping | None: ...

        @property
        def container_config(self) -> FlextProtocolsBase.Model | None: ...

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


__all__: list[str] = ["FlextProtocolsContainerPart01"]
