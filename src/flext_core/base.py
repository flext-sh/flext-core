"""Unified foundation base class for FLEXT ecosystem domain libraries.

This module introduces :class:`FlextBase`, a convenience layer that exposes all
flext-core namespace classes (constants, models, protocols, utilities, etc.)
while providing ready-to-use infrastructure helpers (configuration, logging,
context propagation, dependency injection, metrics) backed by
:class:`flext_core.mixins.FlextMixins`.

Domain libraries can inherit from :class:`FlextBase` to eliminate boilerplate in
examples and services, override any namespace class to extend behaviour, and
benefit from automatic container registration, context binding, and logging.
"""

# ruff: noqa: D102, D107

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TypeVar, cast

from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

ResultType = TypeVar("ResultType")


class _ServiceAdapter(FlextMixins.Service):
    """Internal helper leveraging FlextMixins.Service for automation."""

    def __init__(
        self,
        *,
        service_name: str,
        config: FlextConfig | None,
        container: FlextContainer | None,
        logger: FlextLogger | None,
        logger_name: str | None,
        auto_register: bool,
    ) -> None:
        self._config_override = config
        self._container_override = container
        self._logger_override = logger
        self._logger_name_override = logger_name
        self._custom_logger: FlextLogger | None = None
        super().__init__()
        if auto_register:
            self._init_service(service_name)

    @property
    def config(self) -> FlextConfig:
        if self._config_override is not None:
            return self._config_override
        return super().config

    @property
    def container(self) -> FlextContainer:
        if self._container_override is not None:
            return self._container_override
        return super().container

    @property
    def logger(self) -> FlextLogger:
        if self._logger_override is not None:
            return self._logger_override
        if self._custom_logger is None:
            name = self._logger_name_override or self.__class__.__name__
            self._custom_logger = FlextLogger(name)
        return self._custom_logger

    def log(self, level: str, message: str, **extra: object) -> None:
        self._log_with_context(level, message, **extra)

    def bind_context(self, **context: object) -> None:
        self._enrich_context(**context)

    def track(self, operation_name: str) -> AbstractContextManager[FlextTypes.Dict]:
        return self._track_operation(operation_name)


class FlextBase:
    """Convenience base exposing flext-core namespaces and infrastructure."""

    # Namespace aliases – inherit to enable domain-specific extension.

    Result = FlextResult

    class Config(FlextConfig):
        """Pydantic settings namespace."""

    class Constants(FlextConstants):
        """Centralised constant definitions."""

    class Types(FlextTypes):
        """Type system namespace."""

    class Models(FlextModels):
        """Domain modelling namespace."""

    class Exceptions(FlextExceptions):
        """Exception hierarchy namespace."""

    class Protocols(FlextProtocols):
        """Protocol definitions namespace."""

    class Utilities(FlextUtilities):
        """Cross-cutting utility namespace."""

    class Mixins(FlextMixins):
        """Reusable behaviour mixins."""

    class Service[TDomainResult](FlextService[TDomainResult]):
        """Service base alias preserving generic typing."""

    class Container(FlextContainer):
        """Dependency injection container namespace."""

        @classmethod
        def get_global(cls) -> FlextContainer:
            return super().get_global()

    class Logger(FlextLogger):
        """Structured logging namespace."""

    class Runtime(FlextRuntime):
        """Runtime management namespace."""

    class Bus(FlextBus):
        """Messaging bus namespace."""

    class Context(FlextContext):
        """Context namespace."""

    class Registry(FlextRegistry):
        """Component registry namespace."""

    class Dispatcher(FlextDispatcher):
        """Message dispatcher namespace."""

    class Handlers[MessageT_contra, ResultT](FlextHandlers[MessageT_contra, ResultT]):
        """CQRS handler namespace."""

    class Processors(FlextProcessors):
        """Processing utilities namespace."""

    class Decorators(FlextDecorators):
        """Decorator utilities namespace."""

    def __init__(
        self,
        *,
        service_name: str | None = None,
        config: FlextConfig | None = None,
        container: FlextContainer | None = None,
        logger: FlextLogger | None = None,
        runtime: FlextRuntime | None = None,
        logger_name: str | None = None,
        auto_register: bool = True,
    ) -> None:
        super().__init__()
        self._runtime = runtime or self.Runtime()
        resolved_name = service_name or self.__class__.__name__
        config_override = config or self.Config(debug=False, trace=False)
        self._service = _ServiceAdapter(
            service_name=resolved_name,
            config=config_override,
            container=container,
            logger=logger,
            logger_name=logger_name,
            auto_register=auto_register,
        )

    # ------------------------------------------------------------------
    # Core helpers – configuration, container, logging, runtime
    # ------------------------------------------------------------------

    @property
    def config(self) -> FlextConfig:
        return self._service.config

    @property
    def container(self) -> FlextContainer:
        return self._service.container

    @property
    def logger(self) -> FlextLogger:
        return self._service.logger

    @property
    def context(self) -> FlextContext:
        return self._service.context

    @property
    def runtime(self) -> FlextRuntime:
        return self._runtime

    # ------------------------------------------------------------------
    # Namespace accessors – keep attribute discovery predictable.
    # ------------------------------------------------------------------

    @property
    def constants(self) -> type[FlextConstants]:
        return self.Constants

    @property
    def types(self) -> type[FlextTypes]:
        return self.Types

    @property
    def models(self) -> type[FlextModels]:
        return self.Models

    @property
    def protocols(self) -> type[FlextProtocols]:
        return self.Protocols

    @property
    def exceptions(self) -> type[FlextExceptions]:
        return self.Exceptions

    @property
    def utilities(self) -> type[FlextUtilities]:
        return self.Utilities

    @property
    def mixins(self) -> type[FlextMixins]:
        return self.Mixins

    @property
    def handlers(self) -> type[Handlers[object, object]]:
        return self.Handlers

    @property
    def result(self) -> type[FlextResult[object]]:
        return self.Result

    @property
    def decorators(self) -> type[FlextDecorators]:
        return self.Decorators

    @property
    def registry(self) -> type[FlextRegistry]:
        return self.Registry

    @property
    def dispatcher(self) -> type[FlextDispatcher]:
        return self.Dispatcher

    @property
    def bus(self) -> type[FlextBus]:
        return self.Bus

    @property
    def processors(self) -> type[FlextProcessors]:
        return self.Processors

    # ------------------------------------------------------------------
    # Convenience helpers leveraging the mixin-backed adapter.
    # ------------------------------------------------------------------

    @classmethod
    def ok(cls, value: ResultType) -> FlextResult[ResultType]:
        return FlextResult[ResultType].ok(value)

    @classmethod
    def ok_none(cls) -> FlextResult[None]:
        return FlextResult[None].ok(None)

    @classmethod
    def fail(
        cls,
        message: str,
        *,
        error_code: str | None = None,
        error_data: FlextTypes.Dict | None = None,
    ) -> FlextResult[object]:
        return FlextResult[object].fail(
            message,
            error_code=error_code,
            error_data=error_data,
        )

    def log(self, level: str, message: str, **extra: object) -> None:
        self._service.log(level, message, **extra)

    def info(self, message: str, **extra: object) -> None:
        self.log("info", message, **extra)

    def warning(self, message: str, **extra: object) -> None:
        self.log("warning", message, **extra)

    def error(self, message: str, **extra: object) -> None:
        self.log("error", message, **extra)

    def bind_context(self, **context: object) -> None:
        self._service.bind_context(**context)

    def track(self, operation_name: str) -> AbstractContextManager[FlextTypes.Dict]:
        return self._service.track(operation_name)

    def run_operation(
        self,
        operation_name: str,
        operation: Callable[..., ResultType | FlextResult[ResultType]],
        *args: object,
        **kwargs: object,
    ) -> FlextResult[ResultType]:
        with self.track(operation_name):
            try:
                outcome = operation(*args, **kwargs)
                if isinstance(outcome, FlextResult):
                    return cast(
                        "FlextResult[ResultType]", outcome
                    )  # pyrefly: ignore[redundant-cast]
                return FlextResult[ResultType].ok(outcome)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.error(
                    "Operation failed",
                    operation=operation_name,
                    error=str(exc),
                )
                return FlextResult[ResultType].fail(
                    str(exc),
                    error_code=self.Constants.Errors.UNKNOWN_ERROR,
                    error_data={
                        "operation": operation_name,
                        "exception": exc.__class__.__name__,
                    },
                )
