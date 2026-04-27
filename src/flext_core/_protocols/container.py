"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, override, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.context import FlextProtocolsContext
from flext_core._protocols.handler import FlextProtocolsHandler
from flext_core._protocols.logging import FlextProtocolsLogging
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._protocols.settings import FlextProtocolsSettings

if TYPE_CHECKING:
    from flext_core.models import m
    from flext_core.typings import t


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

        settings: m.ConfigMap | None
        services: Mapping[str, t.RegisterableService] | None
        factories: Mapping[str, t.FactoryCallable] | None
        resources: Mapping[str, t.ResourceCallable] | None
        wire_modules: Sequence[ModuleType] | None
        wire_packages: t.StrSequence | None
        wire_classes: Sequence[type] | None
        factory_cache: bool

    @runtime_checkable
    class ContainerCreationOptionsType(Protocol):
        """Protocol for concrete model classes that validate container options."""

        @classmethod
        def model_validate(
            cls,
            obj: Mapping[str, t.JsonPayload],
        ) -> FlextProtocolsContainer.ContainerCreationOptions:
            """Validate arbitrary input into container creation options."""
            ...

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

        @property
        def provide(self) -> Callable[[str], t.RegisterableService]:
            """Return the dependency-injector Provide helper scoped to the bridge."""
            ...

        def clear(self) -> None:
            """Clear all services and factories."""
            ...

        @override
        def configure(
            self,
            settings: t.UserOverridesMapping | None = None,
        ) -> Self:
            """Configure the container with validated flat overrides."""
            ...

        def apply(
            self,
            settings: t.UserOverridesMapping | None = None,
        ) -> Self:
            """Apply user configuration overrides to the container."""
            ...

        @overload
        def resolve[T: t.RegisterableService](
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
        ) -> FlextProtocolsResult.Result[t.RegisterableService]: ...

        def snapshot(self) -> m.ConfigMap:
            """Return the merged settings exposed by this container."""
            ...

        def has(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def names(self) -> t.StrSequence:
            """List all registered services."""
            ...

        def bind(
            self,
            name: str,
            impl: t.RegisterableService,
        ) -> Self:
            """Bind a concrete service instance or value."""
            ...

        def factory(
            self,
            name: str,
            impl: t.FactoryCallable,
        ) -> Self:
            """Bind a factory callable."""
            ...

        def resource(
            self,
            name: str,
            impl: t.ResourceCallable,
        ) -> Self:
            """Bind a lifecycle-managed resource factory."""
            ...

        def drop(self, name: str) -> FlextProtocolsResult.Result[bool]:
            """Remove a service, factory, or resource by name."""
            ...

        def logger(
            self,
            module_name: str | None = None,
            *,
            service_name: str | None = None,
            service_version: str | None = None,
            correlation_id: str | None = None,
        ) -> FlextProtocolsLogging.Logger:
            """Create a logger bound to the current container runtime."""
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
            registration: m.ServiceRegistrationSpec | None = None,
        ) -> Self:
            """Create an isolated container scope with optional overrides."""
            ...

        def wire(
            self,
            *,
            modules: Sequence[ModuleType] | None = None,
            packages: t.StrSequence | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules/packages to the DI bridge for @inject/Provide usage."""
            ...

    class ContainerLifecycle(Container, Protocol):
        """Extended container contract for bootstrap and lifecycle operations."""

        def initialize_di_components(self) -> None:
            """Initialize DI bridge and backing containers."""
            ...

        def initialize_registrations(
            self,
            *,
            registration: m.ServiceRegistrationSpec | None = None,
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

    @runtime_checkable
    class ContainerType[TContainer: Container = Container](Protocol):
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


__all__: list[str] = ["FlextProtocolsContainer"]
