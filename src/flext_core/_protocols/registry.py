"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Mirrors the public surface of ``FlextRegistry`` so that ``p.Registry`` can be
used in type annotations everywhere instead of the concrete class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from flext_core._protocols import FlextProtocolsService


from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import flext_core
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsEntity,
    FlextProtocolsPydantic,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._protocols import (
        FlextProtocolsContainer,
        FlextProtocolsHandler,
        FlextProtocolsResult,
    )


class FlextProtocolsRegistry:
    """Protocols for handler registration and plugin management."""

    @runtime_checkable
    class RegistrySummary(FlextProtocolsEntity.Value, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def registered(self) -> Sequence[FlextProtocolsHandler.RegistrationDetails]:
            """Successfully registered handlers with registration details."""
            ...

        @property
        def skipped(self) -> Sequence[str]:
            """Handler identifiers that were skipped (already registered)."""
            ...

        @property
        def errors(self) -> MutableSequence[str]:
            """Error messages for failed registrations."""
            ...

        @property
        def failure(self) -> bool:
            """Indicate whether the batch registration had errors."""
            ...

        @property
        def success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            ...

    @runtime_checkable
    class Registry(FlextProtocolsPydantic.Base, Protocol):
        """Registry protocol for CQRS handler and plugin management.

        Mirrors the public instance API of ``FlextRegistry`` so consumers
        can depend on ``p.Registry`` for typing instead of the concrete class.
        """

        # --- fields ---

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher: ...

        # --- lifecycle ---

        def execute(self) -> FlextProtocolsResult.Result[bool]:
            """Validate registry is properly initialized."""
            ...

        # --- handler registration ---

        def register(
            self,
            name: str,
            service: FlextProtocolsContainer.RegisterableService,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a service component."""
            ...

        def register_handler(
            self,
            handler: FlextProtocolsHandler.DispatchableHandler,
        ) -> FlextProtocolsResult.Result[FlextProtocolsHandler.RegistrationDetails]:
            """Register a handler instance or callable."""
            ...

        def register_handlers(
            self,
            handlers: Sequence[FlextProtocolsHandler.DispatchableHandler],
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrySummary]:
            """Register multiple handlers in batch."""
            ...

        def register_bindings(
            self,
            bindings: Mapping[
                flext_core.t.RegistryBindingKey,
                FlextProtocolsHandler.DispatchableHandler,
            ],
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrySummary]:
            """Register message-to-handler bindings."""
            ...

        # --- plugin management ---

        def register_plugin(
            self,
            category: str,
            name: str,
            plugin: FlextProtocolsContainer.RegisterableService,
            *,
            validate: Callable[
                [FlextProtocolsContainer.RegisterableService],
                FlextProtocolsResult.Result[bool],
            ]
            | None = None,
            scope: flext_core.c.RegistrationScope = (
                flext_core.c.RegistrationScope.INSTANCE
            ),
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a plugin with optional validation."""
            ...

        def unregister_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: flext_core.c.RegistrationScope = (
                flext_core.c.RegistrationScope.INSTANCE
            ),
        ) -> FlextProtocolsResult.Result[bool]:
            """Unregister a plugin."""
            ...

        def fetch_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: flext_core.c.RegistrationScope = (
                flext_core.c.RegistrationScope.INSTANCE
            ),
        ) -> FlextProtocolsResult.Result[FlextProtocolsContainer.RegisterableService]:
            """Get a registered plugin by category and name."""
            ...

        def list_plugins(
            self,
            category: str,
            *,
            scope: flext_core.c.RegistrationScope = (
                flext_core.c.RegistrationScope.INSTANCE
            ),
        ) -> FlextProtocolsResult.Result[Sequence[str]]:
            """List all plugins in a category."""
            ...

    @runtime_checkable
    class RegistryServiceState(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class RegistryOptions(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def registry_runtime(self) -> FlextProtocolsService.ServiceRuntime:
            """Explicit runtime shared by registration and execution."""
            ...


__all__: list[str] = ["FlextProtocolsRegistry"]
