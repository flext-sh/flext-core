"""Container option protocols composed by the container facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    # NOTE (multi-agent, mro-wkii.17.26.2): annotations must not reopen t while
    # FlextTypesServices is composing the handler/container protocol graph.
    from collections.abc import Iterable
    from types import ModuleType

    from flext_core import p, t


@runtime_checkable
class _MappingRoot(Protocol):
    """Shared structural API for validated mutable mapping roots."""

    @property
    def root(self) -> t.MutableMappingKV[str, t.JsonPayload]: ...

    def __getitem__(self, key: str) -> t.JsonPayload: ...

    def __setitem__(self, key: str, value: t.JsonPayload) -> None: ...

    def __contains__(self, key: str) -> bool: ...

    def keys(self) -> Iterable[str]: ...

    def items(self) -> Iterable[tuple[str, t.JsonPayload]]: ...

    def get(
        self, key: str, default: t.JsonPayload | None = None
    ) -> t.JsonPayload | None: ...

    def update(self, other: t.MappingKV[str, t.JsonPayload]) -> None: ...

    def model_dump(
        self,
        *,
        mode: str = "python",
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> t.JsonDict: ...

    def model_copy(
        self,
        *,
        update: t.MappingKV[str, t.JsonPayload] | None = None,
        deep: bool = False,
    ) -> Self: ...


class FlextProtocolsContainerOptions:
    """Structural options used to create and wire containers."""

    @runtime_checkable
    class ServiceRegistration(Protocol):
        """Registered service entry consumed by the container runtime."""

        @property
        def service(self) -> t.RegisterableService: ...

        def model_copy(self, *, deep: bool = False) -> Self: ...

    @runtime_checkable
    class FactoryRegistration(Protocol):
        """Registered factory entry consumed by the container runtime."""

        @property
        def factory(self) -> t.FactoryCallable: ...

        def model_copy(self, *, deep: bool = False) -> Self: ...

    @runtime_checkable
    class ResourceRegistration(Protocol):
        """Registered resource entry consumed by the container runtime."""

        @property
        def factory(self) -> t.ResourceCallable: ...

        def model_copy(self, *, deep: bool = False) -> Self: ...

    @runtime_checkable
    class ContainerConfig(Protocol):
        """Container behavior settings consumed by the DI runtime."""

        @property
        def enable_factory_caching(self) -> bool: ...

        def model_dump(self, *, mode: str = "python") -> t.JsonDict: ...

        def model_copy(
            self,
            *,
            update: t.MappingKV[str, t.JsonPayload] | None = None,
            deep: bool = False,
        ) -> Self: ...

    @runtime_checkable
    class ServiceRegistrationSpec(Protocol):
        """Validated inputs used to bootstrap a container scope."""

        @property
        def settings(self) -> p.Settings | None: ...

        @property
        def context(self) -> p.Context | None: ...

        @property
        def services(
            self,
        ) -> (
            t.MappingKV[str, FlextProtocolsContainerOptions.ServiceRegistration] | None
        ): ...

        @property
        def factories(
            self,
        ) -> (
            t.MappingKV[str, FlextProtocolsContainerOptions.FactoryRegistration] | None
        ): ...

        @property
        def resources(
            self,
        ) -> (
            t.MappingKV[str, FlextProtocolsContainerOptions.ResourceRegistration] | None
        ): ...

        @property
        def user_overrides(
            self,
        ) -> (
            FlextProtocolsContainerOptions.ConfigMap
            | t.MappingKV[
                str,
                FlextProtocolsContainerOptions.ConfigMap
                | t.SequenceOf[t.Scalar]
                | t.Scalar,
            ]
            | None
        ): ...

        @property
        def container_config(
            self,
        ) -> FlextProtocolsContainerOptions.ContainerConfig | None: ...

    @runtime_checkable
    class Dict(_MappingRoot, Protocol):
        """Validated mutable dictionary-root container."""

    @runtime_checkable
    class ConfigMap(_MappingRoot, Protocol):
        """Validated mutable configuration-root container."""

    @runtime_checkable
    class RootDict[RootValueT](Protocol):
        """Protocol for immutable dict-rooted validated objects."""

        @property
        def root(self) -> t.MappingKV[str, RootValueT]:
            """The validated root mapping."""
            ...

    @runtime_checkable
    class MutableRootDict[RootValueT](Protocol):
        """Protocol for mutable dict-rooted validated objects."""

        @property
        def root(self) -> t.MutableMappingKV[str, RootValueT]:
            """The mutable validated root mapping."""
            ...

    @runtime_checkable
    class ProviderLike[T_co](Protocol):
        """Framework-independent dependency-provider contract."""

        def __call__(self) -> T_co:
            """Resolve and return the provided dependency."""
            ...

    @runtime_checkable
    class ContainerCreationOptions(Protocol):
        """Structural contract for DI container bootstrap options."""

        @property
        def settings(
            self,
        ) -> FlextProtocolsContainerOptions.RootDict[t.JsonPayload] | None:
            """Validated settings supplied to the container."""
            ...

        @property
        def services(self) -> t.MappingKV[str, t.RegisterableService] | None:
            """Explicitly registered services."""
            ...

        @property
        def factories(self) -> t.MappingKV[str, t.FactoryCallable] | None:
            """Registered service factories."""
            ...

        @property
        def resources(self) -> t.MappingKV[str, t.ResourceCallable] | None:
            """Registered resource factories."""
            ...

        @property
        def wire_modules(self) -> t.SequenceOf[ModuleType] | None:
            """Modules selected for dependency wiring."""
            ...

        @property
        def wire_packages(self) -> t.StrSequence | None:
            """Packages selected for dependency wiring."""
            ...

        @property
        def wire_classes(self) -> t.SequenceOf[type] | None:
            """Classes selected for dependency wiring."""
            ...

        @property
        def factory_cache(self) -> bool:
            """Whether resolved factories are cached."""
            ...


__all__: tuple[str, ...] = ("FlextProtocolsContainerOptions",)
