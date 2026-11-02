"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from flext_core import p


from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Protocol, Self, overload, override, runtime_checkable

from flext_core import t
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)


class FlextProtocolsContainer:
    """Public facade for FlextProtocolsContainer."""

    type RegistryCollaborator = t.Port[
        FlextProtocolsBase.Model
        | FlextProtocolsLogging.Logger
        | FlextProtocolsSettings.Settings
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
    ]
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
    class ServiceRegistration(Protocol):
        """Read-only registration record returned by the service container."""

        @property
        def name(self) -> str: ...

        @property
        def service(self) -> FlextProtocolsContainer.RegisterableService: ...

        @property
        def tags(self) -> Sequence[str]: ...

    @runtime_checkable
    class RootDict[RootValueT](Protocol):
        """Protocol for dict-like root model objects.

        Represents the structure of Pydantic RootModel and similar
        objects that wrap a dict with a root attribute.
        """

        @property
        def root(self) -> Mapping[str, RootValueT]: ...

    @runtime_checkable
    class MutableRootDict[RootValueT](Protocol):
        """Structural contract for mutable validated root mappings."""

        @property
        def root(self) -> MutableMapping[str, RootValueT]: ...

    @runtime_checkable
    class Container(
        FlextProtocolsSettings.Configurable,
        FlextProtocolsBase.Base,
        Protocol,
    ):
        """Dependency injection container protocol.

        Exposes a compact command-style DSL for DI registration, resolution,
        scoping, and runtime integration.
        """

        @property
        def settings(self) -> FlextProtocolsSettings.Settings:
            """Configuration bound to the container."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context:
            """Execution context bound to the container."""
            ...

        def clear(self) -> None:
            """Clear all services and factories."""
            ...

        def register_core_services(
            self,
            *,
            dispatcher: FlextProtocolsHandler.Dispatcher | None = None,
        ) -> None:
            """Register canonical core services in the container."""
            ...

        @override
        def apply(self, settings: t.UserOverridesMapping | None = None) -> Self:
            """Apply user configuration overrides to the container."""
            ...

        @overload
        def resolve[T: FlextProtocolsContainer.RegisterableService](
            self,
            name: str,
            *,
            type_cls: type[T],
        ) -> FlextProtocolsResult.Result[T]: ...

        @overload
        def resolve(
            self,
            name: str,
            *,
            type_cls: None = None,
        ) -> FlextProtocolsResult.Result[
            FlextProtocolsContainer.RegisterableService
        ]: ...

        def snapshot(self) -> p.ConfigMap:
            """Return the merged settings exposed by this container."""
            ...

        def has(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def names(self) -> Sequence[str]:
            """List all registered services."""
            ...

        def bind(
            self, name: str, impl: FlextProtocolsContainer.RegisterableService
        ) -> Self:
            """Bind a concrete service instance or value."""
            ...

        def factory(
            self, name: str, impl: FlextProtocolsContainer.FactoryCallable
        ) -> Self:
            """Bind a factory callable."""
            ...

        def resource(
            self, name: str, impl: FlextProtocolsContainer.ResourceCallable
        ) -> Self:
            """Bind a lifecycle-managed resource factory."""
            ...

        def drop(self, name: str) -> FlextProtocolsResult.Result[bool]:
            """Remove a service, factory, or resource by name."""
            ...

        def dispatcher(
            self,
        ) -> FlextProtocolsResult.Result[FlextProtocolsHandler.Dispatcher]:
            """Resolve the canonical command bus / dispatcher service."""
            ...

        def scope(
            self,
            *,
            subproject: str | None = None,
            registration: (
                FlextProtocolsContainer.ServiceRegistrationSpec | None
            ) = None,
        ) -> Self:
            """Create an isolated container scope with optional overrides."""
            ...

    @runtime_checkable
    class ContainerLifecycle(
        Container,
        Protocol,
    ):
        """Extended container contract for bootstrap and lifecycle operations."""

        def initialize_registrations(
            self,
            *,
            registration: FlextProtocolsContainer.ServiceRegistrationSpec | None = None,
        ) -> None:
            """Initialize explicit registrations and runtime-bound state."""
            ...

        @override
        def register_core_services(
            self, *, dispatcher: FlextProtocolsHandler.Dispatcher | None = None
        ) -> None:
            """Register the canonical core service set into the container."""
            ...

    @runtime_checkable
    class ContainerType[
        TContainer: (FlextProtocolsContainer.Container) = (
            FlextProtocolsContainer.Container
        ),
    ](Protocol):
        """Protocol for concrete container classes exposing canonical factories."""

        @classmethod
        def shared(
            cls,
            *,
            settings: FlextProtocolsSettings.Settings | None = None,
            context: FlextProtocolsContext.Context | None = None,
            auto_register_factories: bool = False,
        ) -> TContainer:
            """Return the process-global container instance."""
            ...

        @classmethod
        def reset_for_testing(cls) -> None:
            """Reset singleton container state for test/example isolation."""
            ...

    type ProtocolGuardInput = (
        t.JsonPayload
        | t.JsonMapping
        | Sequence[str]
        | t.TypeHintSpecifier
        | Callable[..., t.JsonPayload]
        | Container
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.Logger
        | FlextProtocolsResult.Result[t.JsonPayload]
        | FlextProtocolsSettings.Settings
        | FlextProtocolsService.Service[t.JsonPayload]
        | None
    )

    # The generic guard already builds registration values from these leaves.
    type GuardInput = t.GuardInput[
        FlextProtocolsContainer.RegistryCollaborator
        | Container
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.HasLogger
        | FlextProtocolsResult.HasModelDump
        | FlextProtocolsRegistry.Registry
        | FlextProtocolsResult.Result[FlextProtocolsContainer.ServiceValue]
    ]

    @runtime_checkable
    class RegistrationEnvelope(Protocol):
        """Structural data contract projected from the owning model."""

        registration_time: datetime
        metadata: p.Metadata | p.ConfigMap | None

    @runtime_checkable
    class FactoryRegistration(Protocol):
        """Structural data contract projected from the owning model."""

        registration_time: datetime
        metadata: p.Metadata | p.ConfigMap | None
        name: t.NonEmptyStr
        factory: p.FactoryCallable

    @runtime_checkable
    class ResourceRegistration(Protocol):
        """Structural data contract projected from the owning model."""

        registration_time: datetime
        metadata: p.Metadata | p.ConfigMap | None
        name: t.NonEmptyStr
        factory: p.ResourceCallable

    @runtime_checkable
    class ContainerConfig(Protocol):
        """Structural data contract projected from the owning model."""

        enable_singleton: bool
        enable_factory_caching: bool
        max_services: t.PositiveInt
        max_factories: t.PositiveInt
        enable_auto_registration: bool
        enable_lifecycle_hooks: bool
        lazy_loading: bool

    @runtime_checkable
    class FactoryDecoratorConfig(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> t.NonEmptyStr: ...
        @property
        def singleton(self) -> bool: ...
        @property
        def lazy(self) -> bool: ...


__all__: list[str] = ["FlextProtocolsContainer"]
