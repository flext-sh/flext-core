"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, overload, override, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.settings import FlextProtocolsSettings

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType

    from flext_core import FlextModels as m, FlextTypes as t
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.logging import FlextProtocolsLogging
    from flext_core._protocols.result import FlextProtocolsResult
from flext_core._protocols._container_parts.flextprotocolscontainer_part_01 import (
    FlextProtocolsContainer as FlextProtocolsContainerPart01,
)


class FlextProtocolsContainer(FlextProtocolsContainerPart01):
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
            """The dependency-injector Provide helper scoped to the bridge."""
            ...

        def clear(self) -> None:
            """Clear all services and factories."""
            ...

        def register_core_services(self) -> None:
            """Register canonical core services in the container."""
            ...

        @override
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
            """The merged settings exposed by this container."""
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
            modules: t.SequenceOf[ModuleType] | None = None,
            packages: t.StrSequence | None = None,
            classes: t.SequenceOf[type] | None = None,
        ) -> None:
            """Wire modules/packages to the DI bridge for @inject/Provide usage."""
            ...


__all__: list[str] = ["FlextProtocolsContainer"]
