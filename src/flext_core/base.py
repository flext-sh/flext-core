"""Convenience base class with flext-core namespace integration.

This module provides FlextBase, a convenience base class that exposes all
flext-core namespace classes while providing ready-to-use infrastructure
helpers including configuration, logging, context propagation, dependency
injection, and metrics

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""
# ruff: noqa: D102
# pyright: basic

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, TypeAlias

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
from flext_core.typings import FlextTypes, T_co
from flext_core.utilities import FlextUtilities


class FlextBase(FlextMixins):
    """Convenience base class with flext-core namespace integration.

    Provides ready-to-use infrastructure helpers including configuration,
    logging, context propagation, dependency injection, and metrics.

    Features:
    - Access to all flext-core namespace classes (Constants, Models, etc.)
    - Infrastructure helpers for common patterns
    - Automatic service registration and context binding
    - Ready-to-use logging and configuration integration
    - Extension points for domain-specific behavior

    Usage:
        >>> from flext_core.base import FlextBase
        >>>
        >>> class MyService(FlextBase):
        ...     def __init__(self):
        ...         super().__init__(service_name="MyService")
        ...         self.logger.info("Service initialized")
    """

    # Type variables for generic operations
    ResultType = TypeVar("ResultType")

    # Namespace aliases – inherit to enable domain-specific extension.

    Result = FlextResult  # Direct generic alias

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
        """Reusable behaviour mixins - direct passthrough."""
        pass

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
        runtime: FlextRuntime | None = None,
        auto_register: bool = True,
    ) -> None:
        """Initialize FlextBase with optional service registration.

        Args:
            service_name: Optional service name for registration
            runtime: Optional runtime instance (defaults to FlextRuntime())
            auto_register: Whether to auto-register in container (default: True)

        """
        super().__init__()
        self._runtime = runtime or self.Runtime()

        if auto_register:
            resolved_name = service_name or self.__class__.__name__
            self._init_service(resolved_name)

            # Enrich context with base service metadata for observability
            self._enrich_context(
                service_type="base_service",
                service_name=resolved_name,
                runtime_type=type(self._runtime).__name__,
                auto_registered=True,
            )

    # ------------------------------------------------------------------
    # Core helpers – runtime
    # ------------------------------------------------------------------

    @property
    def runtime(self) -> FlextRuntime:
        """Access FlextRuntime instance."""
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
    def ok(cls, value: FlextBase.ResultType) -> FlextResult[FlextBase.ResultType]:
        return FlextResult[FlextBase.ResultType].ok(value)

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
        """Log message with context (delegates to FlextMixins)."""
        self._log_with_context(level, message, **extra)

    def info(self, message: str, **extra: object) -> None:
        """Log info message."""
        self.log("info", message, **extra)

    def warning(self, message: str, **extra: object) -> None:
        """Log warning message."""
        self.log("warning", message, **extra)

    def error(self, message: str, **extra: object) -> None:
        """Log error message."""
        self.log("error", message, **extra)

    def bind_context(self, **context: object) -> None:
        """Bind context data (delegates to FlextMixins)."""
        self._enrich_context(**context)

    def run_operation(
        self,
        operation_name: str,
        operation: Callable[
            ..., FlextBase.ResultType | FlextResult[FlextBase.ResultType]
        ],
        *args: object,
        **kwargs: object,
    ) -> FlextResult[FlextBase.ResultType]:
        with self.track(operation_name):
            try:
                outcome = operation(*args, **kwargs)
                if isinstance(outcome, FlextResult):
                    # Type check: outcome is already FlextResult[ResultType]
                    return outcome
                # Type check: outcome is ResultType, wrap in FlextResult
                return FlextResult[FlextBase.ResultType].ok(outcome)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.error(
                    "Operation failed",
                    operation=operation_name,
                    error=str(exc),
                )
                return FlextResult[FlextBase.ResultType].fail(
                    str(exc),
                    error_code=self.Constants.Errors.UNKNOWN_ERROR,
                    error_data={
                        "operation": operation_name,
                        "exception": exc.__class__.__name__,
                    },
                )


__all__ = [
    "FlextBase",
]
