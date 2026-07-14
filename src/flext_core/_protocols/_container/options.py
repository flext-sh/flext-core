"""Container option protocols composed by the container facade."""

from __future__ import annotations

from types import ModuleType
from typing import Protocol, runtime_checkable

from flext_core import t


class FlextProtocolsContainerOptions:
    """Structural options used to create and wire containers."""

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
