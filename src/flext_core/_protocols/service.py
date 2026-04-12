"""FlextProtocolsService - service and repository protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import (
    FlextProtocolsBase,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsSettings,
    t,
)


class FlextProtocolsService:
    """Protocols for service execution and repository access."""

    @runtime_checkable
    class CloneableRuntime(Protocol):
        """Structural protocol for runtime instances that support cloning.

        Exposes dispatcher, registry, context, container, and settings as read/write
        properties for type-safe cloning without private member access.
        """

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher | None: ...

        @dispatcher.setter
        def dispatcher(
            self,
            value: FlextProtocolsHandler.Dispatcher | None,
            /,
        ) -> None: ...

        @property
        def registry(self) -> FlextProtocolsRegistry.Registry | None: ...

        @registry.setter
        def registry(
            self,
            value: FlextProtocolsRegistry.Registry | None,
            /,
        ) -> None: ...

        @property
        def context(self) -> FlextProtocolsContext.Context: ...

        @context.setter
        def context(
            self,
            value: FlextProtocolsContext.Context,
            /,
        ) -> None: ...

        @property
        def settings(self) -> FlextProtocolsSettings.Settings: ...

        @settings.setter
        def settings(
            self,
            value: FlextProtocolsSettings.Settings,
            /,
        ) -> None: ...

        @property
        def container(self) -> FlextProtocolsContainer.Container: ...

        @container.setter
        def container(
            self,
            value: FlextProtocolsContainer.Container,
            /,
        ) -> None: ...

    @runtime_checkable
    class Service[T](FlextProtocolsBase.Base, Protocol):
        """FlextProtocolsBase.Base domain service interface."""

        def execute(self) -> FlextProtocolsResult.Result[T]:
            """Execute domain service logic."""
            ...

        def service_info(self) -> t.FlatContainerMapping:
            """Get service metadata and configuration information."""
            ...

        def valid(self) -> bool:
            """Check if service is in valid state for execution."""
            ...

        def validate_business_rules(self) -> FlextProtocolsResult.Result[bool]:
            """Validate business rules with extensible validation pipeline.business rule validation without external command parameters."""
            ...

    @runtime_checkable
    class DispatchableService(Protocol):
        """Structural protocol for dispatch-capable service objects in the DI container."""

        def dispatch(
            self,
            message: FlextProtocolsBase.Model,
            /,
        ) -> FlextProtocolsBase.Model:
            """Dispatch a message and return the result."""
            ...


__all__ = ["FlextProtocolsService"]
