"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from flext_core._protocols import FlextProtocolsContainers


from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Protocol, Self, overload, override, runtime_checkable

import flext_core
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)


class FlextProtocolsContainer:
    """Public facade for FlextProtocolsContainer."""

    type RegistryCollaborator = flext_core.t.Port[
        FlextProtocolsPydantic.BaseModel
        | FlextProtocolsLogging.Logger
        | FlextProtocolsSettings.Settings
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
    ]
    type ServiceValue = flext_core.t.ServiceValue[RegistryCollaborator]
    type RegisterableService = flext_core.t.RegisterableService[RegistryCollaborator]
    type FactoryCallable = flext_core.t.FactoryCallable[RegistryCollaborator]
    type ResourceCallable = flext_core.t.ResourceCallable[RegistryCollaborator]
    type ServiceMap = flext_core.t.ServiceMap[RegistryCollaborator]
    type FactoryMap = flext_core.t.FactoryMap[RegistryCollaborator]
    type ResourceMap = flext_core.t.ResourceMap[RegistryCollaborator]

    @runtime_checkable
    class ServiceRegistrationSpec(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings | None:
            """Settings instance bound to the container runtime."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context | None:
            """Execution context attached to the container."""
            ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher | None:
            """Structural member projected from the owning model.

            Explicit command bus retained by this scope; otherwise a fresh bus is built.
            """
            ...

        @property
        def services(self) -> FlextProtocolsContainer.ServiceMap | None:
            """Pre-registered service instances for bootstrap."""
            ...

        @property
        def factories(self) -> FlextProtocolsContainer.FactoryMap | None:
            """Pre-registered factory callables for bootstrap."""
            ...

        @property
        def resources(self) -> FlextProtocolsContainer.ResourceMap | None:
            """Pre-registered resource factories for bootstrap."""
            ...

        @property
        def user_overrides(
            self,
        ) -> (
            FlextProtocolsPydantic.BaseModel | flext_core.t.UserOverridesMapping | None
        ):
            """User-level configuration overrides applied after defaults."""
            ...

        @property
        def container_config(self) -> FlextProtocolsPydantic.BaseModel | None:
            """Container configuration model controlling DI behavior."""
            ...

    @runtime_checkable
    class RegistrationEnvelope(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        registration_time: datetime

        @property
        def metadata(
            self,
        ) -> FlextProtocolsBase.Metadata | FlextProtocolsContainers.ConfigMap | None:
            """Additional service metadata (JSON-serializable)."""
            ...

    @runtime_checkable
    class ServiceRegistration(RegistrationEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        name: flext_core.t.NonEmptyStr

        @property
        def service(self) -> FlextProtocolsContainer.RegisterableService:
            """Service instance (protocols, models, callables)."""
            ...

        tags: Sequence[str]

        @property
        def service_type(self) -> str:
            """Type name of the registered service, derived from the service."""
            ...

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
        FlextProtocolsPydantic.Base,
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
        def apply(
            self, settings: flext_core.t.UserOverridesMapping | None = None
        ) -> Self:
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

        def snapshot(self) -> flext_core.p.ConfigMap:
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
        flext_core.t.JsonPayload
        | flext_core.t.JsonMapping
        | Sequence[str]
        | flext_core.t.TypeHintSpecifier
        | Callable[..., flext_core.t.JsonPayload]
        | Container
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.Logger
        | FlextProtocolsResult.Result[flext_core.t.JsonPayload]
        | FlextProtocolsSettings.Settings
        | FlextProtocolsService.Service[flext_core.t.JsonPayload]
        | None
    )

    # The generic guard already builds registration values from these leaves.
    type GuardInput = flext_core.t.GuardInput[
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
    class FactoryRegistration(RegistrationEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        name: flext_core.t.NonEmptyStr

        @property
        def factory(self) -> FlextProtocolsContainer.FactoryCallable:
            """Factory function that creates service instances."""
            ...

    @runtime_checkable
    class ResourceRegistration(RegistrationEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        name: flext_core.t.NonEmptyStr

        @property
        def factory(self) -> FlextProtocolsContainer.ResourceCallable:
            """Factory returning the lifecycle-managed resource."""
            ...

    @runtime_checkable
    class ContainerConfig(FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        enable_singleton: bool
        enable_factory_caching: bool
        max_services: flext_core.t.PositiveInt
        max_factories: flext_core.t.PositiveInt
        enable_auto_registration: bool
        enable_lifecycle_hooks: bool
        lazy_loading: bool

    @runtime_checkable
    class FactoryDecoratorConfig(FlextProtocolsBase.ImmutableValueModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> flext_core.t.NonEmptyStr:
            """Name to register this factory under in the container."""
            ...

        @property
        def singleton(self) -> bool:
            """Whether factory creates singleton instances."""
            ...

        @property
        def lazy(self) -> bool:
            """Whether to defer factory invocation until first use."""
            ...


__all__: list[str] = ["FlextProtocolsContainer"]
