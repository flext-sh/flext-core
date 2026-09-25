"""FlextProtocolsService - service, mixin infrastructure, and repository protocols.

Mirrors the public surface of ``FlextService``, ``FlextMixins``, and related
concrete classes so that ``p.*`` protocols can be used in type annotations
everywhere instead of concrete types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from .._typings.base import FlextTypingBase as tb
from .._typings.services import FlextTypesServices as ts
from .base import FlextProtocolsBase
from .container import FlextProtocolsContainer
from .context import FlextProtocolsContext
from .loggings import FlextProtocolsLogging
from .result import FlextProtocolsResult
from .settings import FlextProtocolsSettings


class FlextProtocolsService:
    """Protocols for service execution, mixin infrastructure, and repository access."""

    # ------------------------------------------------------------------
    # RuntimeBootstrapProvider — a service base that declares its runtime
    # ------------------------------------------------------------------

    @runtime_checkable
    class RuntimeBootstrapProvider(Protocol):
        """Structural contract of a service base that declares its runtime options.

        Project service bases implement ``runtime_bootstrap_options`` as a
        classmethod to bind their settings class once. The runtime reads the hook
        through this protocol, so ``FlextMixins`` never declares it and no base
        override needs a decorator.
        """

        @classmethod
        def runtime_bootstrap_options(
            cls,
        ) -> FlextProtocolsContext.RuntimeBootstrapOptions:
            """Return the runtime bootstrap options this service base declares."""
            ...

    # ------------------------------------------------------------------
    # MixinsInfrastructure — mirrors FlextMixins public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class MixinsInfrastructure(Protocol):
        """Structural protocol for the shared infrastructure provided by ``FlextMixins``.

        ``FlextMixins`` (alias ``x``) is the base class for Service, Handler, and
        Registry. This protocol exposes its public runtime seeds and runtime-access
        surface so consumers depend on the abstraction instead of the concrete.
        """

        settings_type: ts.SettingsClass | None
        runtime_settings: FlextProtocolsSettings.Settings | None
        settings_overrides: tb.ScalarMapping | None
        initial_context: FlextProtocolsContext.Context | None

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
            self, operation_name: str
        ) -> AbstractContextManager[Mapping[str, ts.JsonPayload]]:
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

        def service_info(self) -> tb.JsonMapping:
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
            self, operation: str, exc: Exception | str | None = ...
        ) -> FlextProtocolsResult.Result[T]:
            """Return a failure result for an operation that failed."""
            ...

    @runtime_checkable
    class DispatchableService(Protocol):
        """Structural protocol for dispatch-capable service objects in the DI container."""

        def dispatch(
            self, message: FlextProtocolsBase.Model, /
        ) -> FlextProtocolsBase.Model:
            """Dispatch a message and return the result."""
            ...


__all__: list[str] = ["FlextProtocolsService"]
