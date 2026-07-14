"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase

from types import ModuleType

if TYPE_CHECKING:
    # mro-wkii.17.26 (codex): this module is loaded by FlextTypesServices, so
    # its postponed aliases cannot eagerly request the composing root facade.
    from flext_core import FlextTypes as t


class FlextProtocolsContainer:
    """Protocols for DI container behavior."""

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

        @property
        def settings(
            self,
        ) -> FlextProtocolsContainer.RootDict[t.JsonPayload] | None: ...

        @property
        def services(self) -> t.MappingKV[str, t.RegisterableService] | None: ...

        @property
        def factories(self) -> t.MappingKV[str, t.FactoryCallable] | None: ...

        @property
        def resources(self) -> t.MappingKV[str, t.ResourceCallable] | None: ...

        @property
        def wire_modules(self) -> t.SequenceOf[ModuleType] | None: ...

        @property
        def wire_packages(self) -> t.StrSequence | None: ...

        @property
        def wire_classes(self) -> t.SequenceOf[type] | None: ...

        @property
        def factory_cache(self) -> bool: ...


__all__: list[str] = ["FlextProtocolsContainer"]
