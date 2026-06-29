"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core.models import FlextModels as m
    from flext_core.typings import FlextTypes as t


class FlextProtocolsContainer:
    """Protocols for DI container behavior."""

    @runtime_checkable
    class RootDict[RootValueT](Protocol):
        """Protocol for dict-like root model objects.

        Represents the structure of Pydantic RootModel and similar
        objects that wrap a dict with a root attribute.
        """

        root: t.MappingKV[str, RootValueT]

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
        services: t.MappingKV[str, t.RegisterableService] | None
        factories: t.MappingKV[str, t.FactoryCallable] | None
        resources: t.MappingKV[str, t.ResourceCallable] | None
        wire_modules: t.SequenceOf[ModuleType] | None
        wire_packages: t.StrSequence | None
        wire_classes: t.SequenceOf[type] | None
        factory_cache: bool

    @runtime_checkable
    class ContainerCreationOptionsType(Protocol):
        """Protocol for concrete model classes that validate container options."""

        @classmethod
        def model_validate(
            cls,
            obj: t.MappingKV[str, t.JsonPayload],
        ) -> FlextProtocolsContainer.ContainerCreationOptions:
            """Validate arbitrary input into container creation options."""
            ...


__all__: list[str] = ["FlextProtocolsContainer"]
