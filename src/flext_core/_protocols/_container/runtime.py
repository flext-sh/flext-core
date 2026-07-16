"""Container runtime protocol composed by the container facade."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, override, runtime_checkable

if TYPE_CHECKING:
    # NOTE (multi-agent, mro-wkii.17.26.2): protocol annotations must not load
    # root facades while FlextTypesServices is composing this protocol graph.
    from flext_core import p, t


class FlextProtocolsContainerRuntime:
    """Runtime operations exposed by a dependency container."""

    @runtime_checkable
    class Container(Protocol):
        """Dependency injection registration and resolution contract."""

        @property
        def settings(self) -> p.Settings:
            """Settings bound to the container."""
            ...

        @property
        def context(self) -> p.Context:
            """The execution context bound to the container."""
            ...

        @property
        def provide(self) -> Callable[[str], t.RegisterableService]:
            """The dependency provider resolver."""
            ...

        def clear(self) -> None:
            """Clear services and factories."""
            ...

        def register_core_services(self) -> None:
            """Register the canonical core service set."""
            ...

        def apply(self, settings: t.UserOverridesMapping | None = None) -> Self:
            """Apply validated configuration overrides."""
            ...

        @overload
        def resolve[T: t.RegisterableService](
            self, name: str, *, type_cls: type[T]
        ) -> p.Result[T]: ...

        @overload
        def resolve(
            self, name: str, *, type_cls: None = None
        ) -> p.Result[t.RegisterableService]: ...

        def snapshot(self) -> p.ConfigMap:
            """Merged settings exposed by the container."""
            ...

        def has(self, name: str) -> bool:
            """Whether a named service is registered."""
            ...

        def names(self) -> t.StrSequence:
            """The registered service names."""
            ...

        def bind(self, name: str, impl: t.RegisterableService) -> Self:
            """Bind a concrete service value."""
            ...

        def factory(self, name: str, impl: t.FactoryCallable) -> Self:
            """Bind a service factory."""
            ...

        def resource(self, name: str, impl: t.ResourceCallable) -> Self:
            """Bind a lifecycle-managed resource factory."""
            ...

        def drop(self, name: str) -> p.Result[bool]:
            """Remove a service, factory, or resource."""
            ...

        def logger(
            self,
            module_name: str,
            *,
            service_name: str | None = None,
            service_version: str | None = None,
            correlation_id: str | None = None,
        ) -> p.Logger:
            """Create a logger bound to the container runtime."""
            ...

        def dispatcher(self) -> p.Result[p.Dispatcher]:
            """Resolve the canonical dispatcher service."""
            ...

        def scope(
            self,
            *,
            subproject: str | None = None,
            registration: p.ServiceRegistrationSpec | None = None,
        ) -> Self:
            """Create an isolated container scope."""
            ...

        def wire(
            self,
            *,
            modules: t.SequenceOf[ModuleType] | None = None,
            packages: t.StrSequence | None = None,
            classes: t.SequenceOf[type] | None = None,
        ) -> None:
            """Wire modules, packages, and classes to the DI bridge."""
            ...

    @runtime_checkable
    class ContainerLifecycle(Container, Protocol):
        """Extended container contract for bootstrap and lifecycle operations."""

        def initialize_di_components(self) -> None:
            """Initialize the DI bridge and backing containers."""
            ...

        def initialize_registrations(
            self, *, registration: p.ServiceRegistrationSpec | None = None
        ) -> None:
            """Initialize registrations and runtime-bound state."""
            ...

        @override
        def register_core_services(self) -> None:
            """Register the canonical core service set."""
            ...

        def register_existing_providers(self) -> None:
            """Hydrate dependency providers from current registrations."""
            ...

        def sync_config_to_di(self) -> None:
            """Synchronize validated configuration into DI providers."""
            ...

    @runtime_checkable
    class ContainerType[TContainer: Container = Container](Protocol):
        """Contract for concrete container classes exposing canonical factories."""

        @classmethod
        def shared(
            cls,
            *,
            settings: p.Settings | None = None,
            context: p.Context | None = None,
            auto_register_factories: bool = False,
        ) -> TContainer:
            """Return the process-global container instance."""
            ...

        @classmethod
        def reset_for_testing(cls) -> None:
            """Reset singleton container state for test isolation."""
            ...


__all__: tuple[str, ...] = ("FlextProtocolsContainerRuntime",)
