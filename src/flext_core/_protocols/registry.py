"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Mirrors the public surface of ``FlextRegistry`` so that ``p.Registry`` can be
used in type annotations everywhere instead of the concrete class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.core import FlextTypesCore

if TYPE_CHECKING:
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.registry import FlextModelsRegistry
    from flext_core._models.service import FlextModelsService
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._typings.services import FlextTypesServices


class FlextProtocolsRegistry:
    """Protocols for handler registration and plugin management."""

    @runtime_checkable
    class Registry(FlextProtocolsBase.Base, Protocol):
        """Registry protocol for CQRS handler and plugin management.

        Mirrors the public instance API of ``FlextRegistry`` so consumers
        can depend on ``p.Registry`` for typing instead of the concrete class.
        """

        # --- fields ---

        dispatcher: FlextProtocolsHandler.Dispatcher | None

        # --- lifecycle ---

        @classmethod
        def create(
            cls,
            dispatcher: FlextProtocolsHandler.Dispatcher | None = None,
            *,
            runtime: FlextModelsService.ServiceRuntime | None = None,
            auto_discover_handlers: bool = False,
        ) -> Self:
            """Factory method to create a new registry instance."""
            ...

        def configure_runtime(
            self,
            runtime: FlextModelsService.ServiceRuntime,
            *,
            dispatcher: FlextProtocolsHandler.Dispatcher | None = None,
        ) -> Self:
            """Bind this registry to a pre-built runtime snapshot."""
            ...

        def execute(self) -> FlextProtocolsResult.Result[bool]:
            """Validate registry is properly initialized."""
            ...

        # --- handler registration ---

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

        # --- plugin management ---

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
