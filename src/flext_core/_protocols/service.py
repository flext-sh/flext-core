"""FlextProtocolsService - service, mixin infrastructure, and repository protocols.

Mirrors the public surface of ``FlextService``, ``FlextMixins``, and related
concrete classes so that ``p.*`` protocols can be used in type annotations
everywhere instead of concrete types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import (
    FlextProtocolsBase,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsSettings,
)

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsService:
    """Protocols for service execution, mixin infrastructure, and repository access."""

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

    # ------------------------------------------------------------------
    # MixinsInfrastructure — mirrors FlextMixins public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class MixinsInfrastructure(Protocol):
        """Structural protocol for the shared infrastructure provided by ``FlextMixins``.

        ``FlextMixins`` (alias ``x``) is the base class for Service, Handler, and
        Registry. This protocol exposes its public runtime-access surface so
        consumers can depend on the abstraction instead of the concrete.
        """

        @property
        def settings(self) -> FlextProtocolsSettings.Settings:
            """Runtime settings associated with this component."""
            ...

        @property
        def container(self) -> FlextProtocolsContainer.Container:
            """Global DI container instance."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context:
            """Execution context for context operations."""
            ...

        @property
        def logger(self) -> FlextProtocolsLogging.Logger:
            """Structured logger for this component."""
            ...

        def track(
            self,
            operation_name: str,
        ) -> AbstractContextManager[Mapping[str, t.RuntimeData]]:
            """Track operation performance with timing and context cleanup."""
            ...

    # ------------------------------------------------------------------
    # Service — mirrors FlextService public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class Service[T](FlextProtocolsBase.Base, Protocol):
        """Domain service interface.

        Mirrors the public instance API of ``FlextService[T]`` so consumers
        can depend on ``p.Service`` for typing instead of the concrete class.
        """

        # --- runtime access (from FlextMixins via MRO) ---

        @property
        def settings(self) -> FlextProtocolsSettings.Settings:
            """Service-scoped settings."""
            ...

        @property
        def container(self) -> FlextProtocolsContainer.Container:
            """Container bound to the service context/settings."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context:
            """Service-scoped execution context."""
            ...

        # --- core contract ---

        def execute(self) -> FlextProtocolsResult.Result[T]:
            """Execute domain service logic."""
            ...

        def service_info(self) -> t.JsonMapping:
            """Get service metadata and configuration information."""
            ...

        def valid(self) -> bool:
            """Check if service is in valid state for execution."""
            ...

        def validate_business_rules(self) -> FlextProtocolsResult.Result[bool]:
            """Validate business rules with extensible validation pipeline."""
            ...

        # --- result helpers ---

        def ok[V](self, value: V) -> FlextProtocolsResult.Result[V]:
            """Wrap a successful value into a result."""
            ...

        def fail_op(
            self,
            operation: str,
            exc: Exception | str | None = ...,
        ) -> FlextProtocolsResult.Result[T]:
            """Return a failure result for an operation that failed."""
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


__all__: list[str] = ["FlextProtocolsService"]
