"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from flext_core._protocols._container import flextprotocolscontainer_part_02

if TYPE_CHECKING:
    from flext_core._protocols import (
        FlextProtocolsContext,
        FlextProtocolsHandler,
        FlextProtocolsSettings,
    )


class FlextProtocolsContainerPart03(
    flextprotocolscontainer_part_02.FlextProtocolsContainerPart02
):
    @runtime_checkable
    class ContainerLifecycle(
        flextprotocolscontainer_part_02.FlextProtocolsContainerPart02.Container,
        Protocol,
    ):
        """Extended container contract for bootstrap and lifecycle operations."""

        def initialize_registrations(
            self,
            *,
            registration: FlextProtocolsContainerPart03.ServiceRegistrationSpec
            | None = None,
        ) -> None:
            """Initialize explicit registrations and runtime-bound state."""
            ...

        @override
        def register_core_services(
            self, *, dispatcher: FlextProtocolsHandler.Dispatcher | None = None
        ) -> None:
            """Register the canonical core service set into the container."""
            ...

    @runtime_checkable
    class ContainerType[
        TContainer: (
            flextprotocolscontainer_part_02.FlextProtocolsContainerPart02.Container
        ) = (flextprotocolscontainer_part_02.FlextProtocolsContainerPart02.Container),
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


__all__: list[str] = ["FlextProtocolsContainerPart03"]
