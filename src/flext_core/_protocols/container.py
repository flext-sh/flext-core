"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from flext_core import FlextProtocolsConfig, FlextProtocolsContext

if TYPE_CHECKING:
    from flext_core import r, t


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
    class Container(FlextProtocolsConfig.Configurable, Protocol):
        """Dependency injection container protocol.

        Extends FlextProtocolsConfig.Configurable to allow container configuration.
        Implements configure() method from FlextProtocolsConfig.Configurable protocol.
        """

        @property
        def config(self) -> FlextProtocolsConfig.Settings:
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
        def get[T: t.RegisterableService](
            self,
            name: str,
            *,
            type_cls: type[T],
        ) -> r[T]: ...

        @overload
        def get(
            self,
            name: str,
            *,
            type_cls: None = None,
        ) -> r[t.RegisterableService]: ...

        def get_config(self) -> t.ConfigMap:
            """Return the merged configuration exposed by this container."""
            ...

        def has_service(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def list_services(self) -> Sequence[str]:
            """List all registered services."""
            ...

        def register(
            self,
            name: str,
            impl: t.RegisterableService,
            *,
            kind: str = "service",
        ) -> Self:
            """Register an implementation by kind."""
            ...

        def scoped(
            self,
            *,
            config: FlextProtocolsConfig.Settings | None = None,
            context: FlextProtocolsContext.Context | None = None,
            subproject: str | None = None,
            services: Mapping[str, t.RegisterableService] | None = None,
            factories: Mapping[str, Callable[..., t.RegisterableService]] | None = None,
            resources: Mapping[str, Callable[..., t.RegisterableService]] | None = None,
        ) -> Self:
            """Create an isolated container scope with optional overrides."""
            ...

        def wire_modules(
            self,
            *,
            modules: Sequence[ModuleType] | None = None,
            packages: Sequence[str] | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules/packages to the DI bridge for @inject/Provide usage."""
            ...


__all__ = ["FlextProtocolsContainer"]
