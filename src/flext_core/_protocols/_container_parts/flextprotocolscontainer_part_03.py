"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

if TYPE_CHECKING:
    from flext_core import FlextModels as m
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.settings import FlextProtocolsSettings
from flext_core._protocols._container_parts.flextprotocolscontainer_part_02 import (
    FlextProtocolsContainer as FlextProtocolsContainerPart02,
)


class FlextProtocolsContainer(FlextProtocolsContainerPart02):
    @runtime_checkable
    class ContainerLifecycle(FlextProtocolsContainerPart02.Container, Protocol):
        """Extended container contract for bootstrap and lifecycle operations."""

        def initialize_di_components(self) -> None:
            """Initialize DI bridge and backing containers."""
            ...

        def initialize_registrations(
            self, *, registration: m.ServiceRegistrationSpec | None = None
        ) -> None:
            """Initialize explicit registrations and runtime-bound state."""
            ...

        @override
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
    class ContainerType[
        TContainer: FlextProtocolsContainerPart02.Container = FlextProtocolsContainerPart02.Container
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


__all__: list[str] = ["FlextProtocolsContainer"]
