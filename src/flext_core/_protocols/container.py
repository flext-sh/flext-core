"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.context import FlextProtocolsContext
from flext_core._protocols.logging import FlextProtocolsLogging
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._protocols.settings import FlextProtocolsSettings
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers

if TYPE_CHECKING:
    from flext_core._models.container import FlextModelsContainer
    from flext_core._typings.services import FlextTypesServices


class FlextProtocolsContainer:
    """Protocols for DI container behavior."""

    class RootDict[RootValueT](Protocol):
        """Protocol for dict-like root model objects.

        Represents the structure of Pydantic RootModel and similar
        objects that wrap a dict with a root attribute.
        """

        root: Mapping[str, RootValueT]

    @runtime_checkable
    class ProviderLike[T_co](Protocol):
        """DI-free abstraction for dependency injection providers.

        Provides a framework-independent contract for dependency injection
        providers. Real providers in ``FlextRuntime.DependencyIntegration``
        implement this Protocol structurally.

        Usage::

            provider: p.ProviderLike[MyService]
            service = provider()  # Returns MyService instance

        This Protocol avoids coupling the ``_protocols`` layer to
        ``dependency_injector``, keeping the architecture boundary clean.
        """

        def __call__(self) -> T_co:
            """Resolve and return the provided dependency."""
            ...

    @runtime_checkable
    class ContainerCreationOptions(FlextProtocolsBase.Base, Protocol):
        """Structural contract for DI container bootstrap options."""

        settings: FlextTypingContainers.ConfigMap | None
        services: Mapping[str, FlextTypesServices.RegisterableService] | None
        factories: Mapping[str, FlextTypesServices.FactoryCallable] | None
        resources: Mapping[str, FlextTypesServices.ResourceCallable] | None
        wire_modules: Sequence[ModuleType] | None
        wire_packages: FlextTypingBase.StrSequence | None
        wire_classes: Sequence[type] | None
        factory_cache: bool

    @runtime_checkable
    class ContainerCreationOptionsType(Protocol):
        """Protocol for concrete model classes that validate container options."""

        @classmethod
        def model_validate(
            cls,
            obj: FlextTypesServices.ModelInput,
        ) -> FlextProtocolsContainer.ContainerCreationOptions:
            """Validate arbitrary input into container creation options."""
            ...

    @runtime_checkable
    class Container(FlextProtocolsSettings.Configurable, Protocol):
        """Dependency injection container protocol.

        Extends FlextProtocolsSettings.Configurable to allow container configuration.
        Implements configure() method from FlextProtocolsSettings.Configurable protocol.
        """

        @property
        def settings(self) -> FlextProtocolsSettings.Settings:
            """Configuration bound to the container."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context:
            """Execution context bound to the container."""
            ...

        def clear_all(self) -> None:
            """Clear all services and factories."""
            ...

        @overload
        def get[T: FlextTypesServices.RegisterableService](
            self,
            name: str,
            *,
            type_cls: type[T],
        ) -> FlextProtocolsResult.Result[T]: ...

        @overload
        def get(
            self,
            name: str,
            *,
            type_cls: None = None,
        ) -> FlextProtocolsResult.Result[FlextTypesServices.RegisterableService]: ...

        def resolve_settings(self) -> FlextTypingContainers.ConfigMap:
            """Return the merged settings exposed by this container."""
            ...

        def has_service(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def list_services(self) -> FlextTypingBase.StrSequence:
            """List all registered services."""
            ...

        def register(
            self,
            name: str,
            impl: FlextTypesServices.RegisterableService,
            *,
            kind: str = "service",
        ) -> Self:
            """Register an implementation by kind."""
            ...

        def scoped(
            self,
            *,
            settings: FlextProtocolsSettings.Settings | None = None,
            context: FlextProtocolsContext.Context | None = None,
            subproject: str | None = None,
            services: Mapping[str, FlextTypesServices.RegisterableService]
            | None = None,
            factories: FlextTypesServices.FactoryMap | None = None,
            resources: FlextTypesServices.ResourceMap | None = None,
        ) -> Self:
            """Create an isolated container scope with optional overrides."""
            ...

        def wire_modules(
            self,
            *,
            modules: Sequence[ModuleType] | None = None,
            packages: FlextTypingBase.StrSequence | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules/packages to the DI bridge for @inject/Provide usage."""
            ...

    class ContainerLifecycle(Container, Protocol):
        """Extended container contract for bootstrap and lifecycle operations."""

        def create_module_logger(
            self,
            module_name: str | None = None,
            *,
            service_name: str | None = None,
            service_version: str | None = None,
            correlation_id: str | None = None,
        ) -> FlextProtocolsLogging.Logger:
            """Create a logger bound to the current container runtime."""
            ...

        def initialize_di_components(self) -> None:
            """Initialize DI bridge and backing containers."""
            ...

        def initialize_registrations(
            self,
            *,
            services: Mapping[str, FlextModelsContainer.ServiceRegistration]
            | None = None,
            factories: Mapping[str, FlextModelsContainer.FactoryRegistration]
            | None = None,
            resources: Mapping[str, FlextModelsContainer.ResourceRegistration]
            | None = None,
            global_config: FlextModelsContainer.ContainerConfig | None = None,
            user_overrides: FlextTypesServices.UserOverridesMapping
            | FlextTypingContainers.ConfigMap
            | None = None,
            settings: FlextProtocolsSettings.Settings | None = None,
            context: FlextProtocolsContext.Context | None = None,
        ) -> None:
            """Initialize explicit registrations and runtime-bound state."""
            ...

        def register_core_services(self) -> None:
            """Register the canonical core service set into the container."""
            ...

        def register_existing_providers(self) -> None:
            """Hydrate dependency providers from current registrations."""
            ...

        def sync_config_to_di(self) -> None:
            """Synchronize validated configuration into DI providers."""
            ...

        def unregister(self, name: str) -> FlextProtocolsResult.Result[bool]:
            """Remove a service, factory, or resource by name."""
            ...

    @runtime_checkable
    class ContainerType[TContainer: Container = Container](Protocol):
        """Protocol for concrete container classes exposing canonical factories."""

        @classmethod
        def create(
            cls,
            *,
            auto_register_factories: bool = False,
        ) -> TContainer:
            """Create or return the canonical container instance."""
            ...

        @classmethod
        def fetch_global(
            cls,
            *,
            settings: FlextProtocolsSettings.Settings | None = None,
            context: FlextProtocolsContext.Context | None = None,
        ) -> TContainer:
            """Return the process-global container instance."""
            ...

        @classmethod
        def reset_for_testing(cls) -> None:
            """Reset singleton container state for test/example isolation."""
            ...


__all__: list[str] = ["FlextProtocolsContainer"]
