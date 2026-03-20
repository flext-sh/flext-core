"""FlextProtocolsRegistry - handler registration and plugin management protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase

if TYPE_CHECKING:
    from pydantic import BaseModel

    from flext_core import FlextProtocolsResult, m, r, t


class FlextProtocolsRegistry:
    """Protocols for handler registration and plugin management."""

    @runtime_checkable
    class Registry(FlextProtocolsBase.Base, Protocol):
        """Registry protocol for CQRS handler and plugin management.

        Structural typing interface for FlextRegistry. Consumers depend on
        this protocol instead of the concrete FlextRegistry class, enabling
        DIP compliance across the workspace.
        """

        def execute(self) -> r[bool]:
            """Validate registry is properly initialized."""
            ...

        def register(
            self,
            name: str,
            service: t.RegistrablePlugin,
            metadata: t.ConfigMap | m.Metadata | None = None,
        ) -> r[bool]:
            """Register a service component with optional metadata."""
            ...

        def register_handler(
            self,
            handler: t.HandlerLike,
            _metadata: t.ConfigMap | m.Metadata | None = None,
        ) -> r[m.RegistrationDetails]:
            """Register a handler instance or callable."""
            ...

        def register_handlers(
            self,
            handlers: Sequence[t.HandlerLike],
        ) -> FlextProtocolsResult.Result[BaseModel]:
            """Register multiple handlers in batch."""
            ...

        def register_bindings(
            self,
            bindings: Mapping[t.RegistryBindingKey, t.HandlerLike],
        ) -> FlextProtocolsResult.Result[BaseModel]:
            """Register message-to-handler bindings."""
            ...

        def register_plugin(
            self,
            category: str,
            name: str,
            plugin: t.RegistrablePlugin,
            *,
            validate: Callable[[t.RegistrablePlugin], r[bool]] | None = None,
            scope: Literal["instance", "class"] = "instance",
        ) -> r[bool]:
            """Register a plugin with optional validation."""
            ...

        def unregister_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: Literal["instance", "class"] = "instance",
        ) -> r[bool]:
            """Unregister a plugin."""
            ...

        def get_plugin(
            self,
            category: str,
            name: str,
            *,
            scope: str = "instance",
        ) -> r[t.RuntimeAtomic | None]:
            """Get a registered plugin by category and name."""
            ...

        def list_plugins(
            self,
            category: str,
            *,
            scope: str = "instance",
        ) -> r[list[str]]:
            """List all plugins in a category."""
            ...


__all__ = ["FlextProtocolsRegistry"]
