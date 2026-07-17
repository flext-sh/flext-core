"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Mirrors the public surface of ``FlextRegistry`` so that ``p.Registry`` can be
used in type annotations everywhere instead of the concrete class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable, TYPE_CHECKING

from flext_core._constants.mixins import FlextConstantsMixins as c


if TYPE_CHECKING:
    from flext_core._constants.status import FlextConstantsStatus as cs
    from flext_core._constants.cqrs import FlextConstantsCqrs as cc
    from .settings import FlextProtocolsSettings
    from .result import FlextProtocolsResult
    from .handler import FlextProtocolsHandler
    from .container import FlextProtocolsContainer
    from flext_core import t
    from .context import FlextProtocolsContext
    from collections.abc import Callable


class FlextProtocolsRegistry:
    """Protocols for handler registration and plugin management."""

    # mro-wkii.17.26 (codex): registry owns structural result/runtime contracts;
    # importing the concrete model facade here creates the p -> m -> p cycle.
    @runtime_checkable
    class RegistrationDetails(Protocol):
        """Observable handler registration details."""

        @property
        def registration_id(self) -> str:
            """Unique registration identifier."""
            ...

        @property
        def handler_mode(self) -> cc.HandlerType:
            """Registered handler mode."""
            ...

        @property
        def timestamp(self) -> str:
            """Registration timestamp."""
            ...

        @property
        def status(self) -> cs.Status:
            """Current registration status."""
            ...

    @runtime_checkable
    class RegistrySummary(Protocol):
        """Observable batch registration outcome."""

        @property
        def registered(
            self,
        ) -> t.SequenceOf[FlextProtocolsRegistry.RegistrationDetails]:
            """Successful registration details."""
            ...

        @property
        def skipped(self) -> t.StrSequence:
            """Skipped handler identifiers."""
            ...

        @property
        def errors(self) -> t.StrSequence:
            """Registration error messages."""
            ...

        @property
        def failure(self) -> bool:
            """Whether any registration failed."""
            ...

        @property
        def success(self) -> bool:
            """Whether every registration succeeded."""
            ...

    @runtime_checkable
    class Registry(Protocol):
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
            runtime: FlextProtocolsRegistry.ServiceRuntime | None = None,
            auto_discover_handlers: bool = False,
        ) -> Self:
            """Create a new registry instance."""
            ...

        def configure_runtime(
            self,
            runtime: FlextProtocolsRegistry.ServiceRuntime,
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
            self, name: str, service: t.RegistrablePlugin
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a service component."""
            ...

        def register_handler(
            self, handler: t.DispatchableHandler
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrationDetails]:
            """Register a handler instance or callable."""
            ...

        def register_handlers(
            self, handlers: t.SequenceOf[t.DispatchableHandler]
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrySummary]:
            """Register multiple handlers in batch."""
            ...

        def register_bindings(
            self, bindings: t.MappingKV[t.RegistryBindingKey, t.DispatchableHandler]
        ) -> FlextProtocolsResult.Result[FlextProtocolsRegistry.RegistrySummary]:
            """Register message-to-handler bindings."""
            ...

        # --- plugin management ---

        def register_plugin(
            self,
            category: str,
            name: str,
            plugin: t.RegistrablePlugin,
            *,
            validate: Callable[[t.RegistrablePlugin], FlextProtocolsResult.Result[bool]]
            | None = None,
            scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a plugin with optional validation."""
            ...

        def unregister_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[bool]:
            """Unregister a plugin."""
            ...

        def fetch_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[t.JsonPayload | None]:
            """Get a registered plugin by category and name."""
            ...

        def list_plugins(
            self,
            category: str,
            *,
            scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
        ) -> FlextProtocolsResult.Result[t.StrSequence]:
            """List all plugins in a category."""
            ...

    @runtime_checkable
    class ServiceRuntime(Protocol):
        """Runtime capabilities consumed by registry composition."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings: ...

        @property
        def context(self) -> FlextProtocolsContext.Context: ...

        @property
        def container(self) -> FlextProtocolsContainer.Container: ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher | None:
            """Dispatcher bound to the runtime."""
            ...

        def model_copy(
            self,
            *,
            update: t.MappingKV[
                str,
                FlextProtocolsHandler.Dispatcher
                | FlextProtocolsRegistry.Registry
                | None,
            ]
            | None = None,
            deep: bool = False,
        ) -> FlextProtocolsRegistry.ServiceRuntime:
            """Copy runtime state with registry-owned updates."""
            ...


__all__: tuple[str, ...] = ("FlextProtocolsRegistry",)
