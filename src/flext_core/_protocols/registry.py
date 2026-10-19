"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Mirrors the public surface of ``FlextRegistry`` so that ``p.Registry`` can be
used in type annotations everywhere instead of the concrete class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import c
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import t
    from flext_core._protocols import (
        FlextProtocolsContainer,
        FlextProtocolsHandler,
        FlextProtocolsResult,
    )


class FlextProtocolsRegistry:
    """Protocols for handler registration and plugin management."""

    @runtime_checkable
    class RegistrationDetails(FlextProtocolsBase.Base, Protocol):
        """Observable registration identity and routing metadata."""

        @property
        def registration_id(self) -> str: ...

        @property
        def handler_mode(self) -> c.HandlerType: ...

        @property
        def status(self) -> c.Status: ...

    @runtime_checkable
    class RegistrySummary(FlextProtocolsBase.Base, Protocol):
        """Observable outcome of one atomic registration batch."""

        @property
        def registered(
            self,
        ) -> t.SequenceOf[FlextProtocolsRegistry.RegistrationDetails]: ...

        @property
        def skipped(self) -> t.StrSequence: ...

        @property
        def errors(self) -> t.StrSequence: ...

        @property
        def success(self) -> bool: ...

        @property
        def failure(self) -> bool: ...

    @runtime_checkable
    class Registry(FlextProtocolsBase.Base, Protocol):
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
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrationDetails]:
            """Register a handler instance or callable."""
            ...

        def register_handlers(
            self,
            handlers: t.SequenceOf[FlextProtocolsHandler.DispatchableHandler],
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrySummary]:
            """Register multiple handlers in batch."""
            ...

        def register_bindings(
            self,
            bindings: t.MappingKV[
                t.RegistryBindingKey, FlextProtocolsHandler.DispatchableHandler
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
            scope: c.RegistrationScope = (c.RegistrationScope.INSTANCE),
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a plugin with optional validation."""
            ...

        def unregister_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: c.RegistrationScope = (c.RegistrationScope.INSTANCE),
        ) -> FlextProtocolsResult.Result[bool]:
            """Unregister a plugin."""
            ...

        def fetch_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: c.RegistrationScope = (c.RegistrationScope.INSTANCE),
        ) -> FlextProtocolsResult.Result[FlextProtocolsContainer.RegisterableService]:
            """Get a registered plugin by category and name."""
            ...

        def list_plugins(
            self,
            category: str,
            *,
            scope: c.RegistrationScope = (c.RegistrationScope.INSTANCE),
        ) -> FlextProtocolsResult.Result[t.StrSequence]:
            """List all plugins in a category."""
            ...


__all__: list[str] = ["FlextProtocolsRegistry"]
