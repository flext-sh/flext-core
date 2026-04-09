"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocolsBase, FlextProtocolsResult, t

if TYPE_CHECKING:
    from flext_core import m


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
            service: t.RegistrablePlugin,
            metadata: t.ConfigMap | m.Metadata | None = None,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a service component with optional metadata."""
            ...

        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            _metadata: t.ConfigMap | m.Metadata | None = None,
        ) -> FlextProtocolsResult.Result[m.RegistrationDetails]:
            """Register a handler instance or callable."""
            ...

        def register_handlers(
            self,
            handlers: Sequence[t.HandlerProtocolVariant],
        ) -> FlextProtocolsResult.Result[m.RegistrySummary]:
            """Register multiple handlers in batch."""
            ...

        def register_bindings(
            self,
            bindings: Mapping[t.RegistryBindingKey, t.HandlerProtocolVariant],
        ) -> FlextProtocolsResult.Result[m.RegistrySummary]:
            """Register message-to-handler bindings."""
            ...

        def register_plugin(
            self,
            category: str,
            name: str,
            plugin: t.RegistrablePlugin,
            *,
            validate: Callable[
                [t.RegistrablePlugin],
                FlextProtocolsResult.Result[bool],
            ]
            | None = None,
            scope: Literal["instance", "class"] = "instance",
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a plugin with optional validation."""
            ...

        def unregister_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: Literal["instance", "class"] = "instance",
        ) -> FlextProtocolsResult.Result[bool]:
            """Unregister a plugin."""
            ...

        def get_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: str = "instance",
        ) -> FlextProtocolsResult.Result[t.RuntimeAtomic | None]:
            """Get a registered plugin by category and name."""
            ...

        def list_plugins(
            self,
            category: str,
            *,
            scope: str = "instance",
        ) -> FlextProtocolsResult.Result[t.StrSequence]:
            """List all plugins in a category."""
            ...


__all__ = ["FlextProtocolsRegistry"]
