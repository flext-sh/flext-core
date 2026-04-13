"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import (
    FlextConstantsMixins,
    FlextProtocolsBase,
    FlextProtocolsResult,
    FlextTypesCore,
    FlextTypingBase,
)

if TYPE_CHECKING:
    from flext_core import (
        FlextModelsHandler,
        FlextModelsRegistry,
        FlextTypesServices,
    )


class FlextProtocolsRegistry:
    """Protocols for handler registration and plugin management."""

    @runtime_checkable
    class Registry(FlextProtocolsBase.Base, Protocol):
        """Registry protocol for CQRS handler and plugin management."""

        def execute(self) -> FlextProtocolsResult.Result[bool]:
            """Validate registry is properly initialized."""
            ...

        def register(
            self,
            name: str,
            service: FlextTypesServices.RegistrablePlugin,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a service component."""
            ...

        def register_handler(
            self,
            handler: FlextTypesServices.HandlerProtocolVariant,
        ) -> FlextProtocolsResult.Result[FlextModelsHandler.RegistrationDetails]:
            """Register a handler instance or callable."""
            ...

        def register_handlers(
            self,
            handlers: Sequence[FlextTypesServices.HandlerProtocolVariant],
        ) -> FlextProtocolsResult.Result[FlextModelsRegistry.RegistrySummary]:
            """Register multiple handlers in batch."""
            ...

        def register_bindings(
            self,
            bindings: Mapping[
                FlextTypesCore.RegistryBindingKey,
                FlextTypesServices.HandlerProtocolVariant,
            ],
        ) -> FlextProtocolsResult.Result[FlextModelsRegistry.RegistrySummary]:
            """Register message-to-handler bindings."""
            ...

        def register_plugin(
            self,
            category: str,
            name: str,
            plugin: FlextTypesServices.RegistrablePlugin,
            *,
            validate: Callable[
                [FlextTypesServices.RegistrablePlugin],
                FlextProtocolsResult.Result[bool],
            ]
            | None = None,
            scope: FlextConstantsMixins.RegistrationScope = FlextConstantsMixins.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a plugin with optional validation."""
            ...

        def unregister_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: FlextConstantsMixins.RegistrationScope = FlextConstantsMixins.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[bool]:
            """Unregister a plugin."""
            ...

        def fetch_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: FlextConstantsMixins.RegistrationScope = FlextConstantsMixins.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic | None]:
            """Get a registered plugin by category and name."""
            ...

        def list_plugins(
            self,
            category: str,
            *,
            scope: FlextConstantsMixins.RegistrationScope = FlextConstantsMixins.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[FlextTypingBase.StrSequence]:
            """List all plugins in a category."""
            ...


__all__: list[str] = ["FlextProtocolsRegistry"]
