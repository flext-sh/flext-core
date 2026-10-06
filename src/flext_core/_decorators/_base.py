"""Base decorator helpers and dependency injection decorator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import TYPE_CHECKING, ClassVar, TypeIs

from flext_core import FlextContainer, c, m
from flext_core._models.flext_context import FlextContext
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.container import FlextProtocolsContainer
from flext_core._protocols.context import FlextProtocolsContext
from flext_core._protocols.loggings import FlextProtocolsLogging
from flext_core._typings.services import FlextTypesServices
from flext_core.loggings import FlextUtilitiesLogging

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextDecoratorsBase:
    """Base helpers shared by concrete decorator namespaces."""

    type _LoggerCarrier = (
        FlextProtocolsLogging.HasLogger
        | FlextProtocolsLogging.Logger
        | FlextTypesServices.JsonPayload
        | m.BaseModel
    )
    _CAUGHT_EXCEPTIONS: tuple[type[Exception], ...] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
    )
    _container_type: ClassVar[FlextProtocolsContainer.ContainerType] = FlextContainer
    _context_type: ClassVar[FlextProtocolsContext.ContextType] = FlextContext

    @classmethod
    def _is_logger_carrier(
        cls,
        value: FlextProtocolsBase.AttributeProbe | None,
    ) -> TypeIs[_LoggerCarrier]:
        """Return whether value carries or can route logging context.

        Returns:
            Whether value carries or can route logging context.

        """
        _ = cls
        return isinstance(
            value,
            (
                FlextProtocolsLogging.Logger,
                FlextProtocolsLogging.HasLogger,
                m.BaseModel,
                *c.CONTAINER_TYPES,
            ),
        )

    @classmethod
    def _resolve_logger(
        cls,
        first_arg: FlextProtocolsLogging.Logger | _LoggerCarrier | None = None,
        *,
        func: FlextTypesServices.DispatchableHandler | None = None,
        func_module: str | None = None,
    ) -> FlextProtocolsLogging.Logger:
        """Resolve the logger associated with the decorated call.

        Returns:
            The resulting ``pl.Logger``.

        """
        _ = cls
        if isinstance(first_arg, FlextProtocolsLogging.Logger):
            return first_arg
        if isinstance(first_arg, FlextProtocolsLogging.HasLogger):
            return first_arg.logger
        module_name = (
            func_module
            if isinstance(func_module, str)
            else (func.__module__ if callable(func) else __name__)
        )
        logger: FlextProtocolsLogging.Logger = FlextUtilitiesLogging.fetch_logger(
            module_name,
        )
        return logger

    @staticmethod
    def deprecated[**PCallback, TResult](
        reason: str,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Mark callable as deprecated and emit ``DeprecationWarning`` on use.

        Returns:
            The resulting ``Callable[[Callable[PCallback, TResult]], Callable[PCallback,
                TResult]]``.

        """

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                warnings.warn(
                    f"{func.__name__} is deprecated: {reason}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def inject[**PCallback, TResult](
        cls,
        **dependencies: str,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Inject dependencies from the configured FLEXT container.

        Returns:
            The resulting ``Callable[[Callable[PCallback, TResult]], Callable[PCallback,
                TResult]]``.

        """

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                container = cls._container_type.shared()
                for name, service_key in dependencies.items():
                    if name not in kwargs:
                        result = container.resolve(service_key)
                        if result.success:
                            kwargs[name] = result.value
                return func(*args, **kwargs)

            return wrapper

        return decorator


__all__: list[str] = ["FlextDecoratorsBase"]
