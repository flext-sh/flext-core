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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsPydantic,
    FlextProtocolsSettings,
)

if TYPE_CHECKING:
    from flext_core._protocols import (
        FlextProtocolsConfig,
        FlextProtocolsContainer,
        FlextProtocolsContext,
        FlextProtocolsHandler,
        FlextProtocolsLogging,
        FlextProtocolsResult,
    )


class FlextProtocolsService:
    """Protocols for service execution, mixin infrastructure, and repository access."""

    @runtime_checkable
    class ServiceOperationHandler(Protocol):
        """Original unbound operation called with its explicitly provided service."""

        def __call__(
            self,
            service: FlextProtocolsPydantic.Base,
            /,
            *requests: FlextProtocolsPydantic.Base,
        ) -> FlextProtocolsResult.Result[t.JsonPayload]: ...

    @runtime_checkable
    class ServiceOperation(FlextProtocolsBase.ContractModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Method name that implements the operation."""
            ...

        @property
        def summary(self) -> str:
            """First docstring line of the operation."""
            ...

        @property
        def request(self) -> t.ModelClass[t.BaseModelType] | None:
            """Pydantic request model, or None for an input-less operation."""
            ...

        @property
        def handler(self) -> FlextProtocolsService.ServiceOperationHandler:
            """Original unbound operation, invoked with the provided service."""
            ...

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
        ) -> FlextProtocolsService.RuntimeBootstrapOptions:
            """Return the runtime bootstrap options this service base declares."""
            ...

    # ------------------------------------------------------------------
    # MixinsInfrastructure — mirrors FlextMixins public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class MixinsInfrastructure(Protocol):
        """Structural protocol for shared ``FlextMixins`` infrastructure.

        ``FlextMixins`` (alias ``x``) is the base class for Service, Handler, and
        Registry. This protocol exposes its public runtime seeds and runtime-access
        surface so consumers depend on the abstraction instead of the concrete.
        """

        settings_type: FlextProtocolsSettings.SettingsType | None

        @property
        def runtime_settings(self) -> FlextProtocolsSettings.Settings | None:
            """Settings seed observed by the service runtime."""
            ...

        settings_overrides: t.JsonMapping | None
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
            self,
            operation_name: str,
        ) -> AbstractContextManager[Mapping[str, t.JsonPayload]]:
            """Track operation performance with timing and context cleanup."""
            ...

    # ------------------------------------------------------------------
    # Service — mirrors FlextService public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class Service[T](FlextProtocolsPydantic.Base, MixinsInfrastructure, Protocol):
        """Domain service interface: the runtime surface plus ``execute``.

        Every ``FlextService[T]`` satisfies it structurally. The runtime surface
        (settings, container, context, logger, track and the runtime seeds) is
        inherited from ``MixinsInfrastructure`` — declared once — and a service
        adds only its domain ``execute``. Capabilities some services offer
        (business-rule validation, metadata) are declared by the member protocols
        that consume them, never here.
        """

        def execute(self) -> FlextProtocolsResult.Result[T]:
            """Execute domain service logic."""
            ...

    @runtime_checkable
    class DispatchableService(Protocol):
        """Structural protocol for dispatch-capable DI container services."""

        def dispatch(
            self,
            message: FlextProtocolsPydantic.BaseModel,
            /,
        ) -> FlextProtocolsPydantic.BaseModel:
            """Dispatch a message and return the result."""
            ...

    @runtime_checkable
    class ServiceRuntime(FlextProtocolsPydantic.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings:
            """Service configuration settings for runtime behavior."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context:
            """Execution context carrying correlation and tracing metadata."""
            ...

        @property
        def container(self) -> FlextProtocolsContainer.Container:
            """Dependency injection container scoped to this runtime."""
            ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher:
            """Dispatcher resolved for CQRS routing in this runtime."""
            ...

        @property
        def logger(self) -> FlextProtocolsLogging.Logger:
            """Logging destination elected by the composition root."""
            ...

        @property
        def logger_injected(self) -> bool:
            """Keep an explicit logger when transport settings are rebound."""
            ...

    @runtime_checkable
    class RuntimeBootstrapOptions(FlextProtocolsPydantic.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings | None:
            """Settings source cloned once into the elected container scope."""
            ...

        @property
        def settings_type(self) -> FlextProtocolsSettings.SettingsType | None:
            """FlextSettings class used to load runtime settings."""
            ...

        @property
        def settings_overrides(self) -> t.JsonMapping | None:
            """Key-value overrides applied on top of the loaded configuration."""
            ...

        @property
        def context(self) -> FlextProtocolsContext.Context | None:
            """Pre-built execution context to inject into the runtime."""
            ...

        @property
        def container(self) -> FlextProtocolsContainer.Container | None:
            """Structural member projected from the owning model.

            Existing container whose public DI registrations the runtime preserves.
            """
            ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher | None:
            """Pre-built dispatcher injected into the runtime."""
            ...

        @property
        def logger(self) -> FlextProtocolsLogging.Logger | None:
            """Logging dependency; absent selects the runtime policy."""
            ...

        @property
        def logging_options(self) -> FlextProtocolsConfig.StructlogOptions | None:
            """Programmatic logging extensions elected by the root."""
            ...


__all__: list[str] = ["FlextProtocolsService"]
