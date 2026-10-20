"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsSettings,
)


class FlextProtocolsContainer:
    """Protocols for DI container behavior."""

    type RegistryCollaborator = (
        FlextProtocolsBase.Model[t.JsonPayload]
        | FlextProtocolsLogging.Logger
        | FlextProtocolsSettings.Settings
        | FlextProtocolsContext.Context
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
        context: FlextProtocolsContext.Context | None
        dispatcher: FlextProtocolsHandler.Dispatcher | None
        services: FlextProtocolsContainer.ServiceMap | None
        factories: FlextProtocolsContainer.FactoryMap | None
        resources: FlextProtocolsContainer.ResourceMap | None

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


__all__: list[str] = ["FlextProtocolsContainer"]
