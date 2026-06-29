"""Base decorator helpers and dependency injection decorator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import ClassVar, TypeIs

from flext_core._loggings_parts import FlextLogger
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols.base import FlextProtocolsBase as pb
from flext_core._protocols.container import FlextProtocolsContainer as pc
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.logging import FlextProtocolsLogging as pl
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.services import FlextTypesServices as ts
from flext_core.container import FlextContainer
from flext_core.context import FlextContext


class FlextDecoratorsBase:
    """Base helpers shared by concrete decorator namespaces."""

    type _LoggerCarrier = pl.HasLogger | pl.Logger | ts.JsonPayload | mp.BaseModel
    _CAUGHT_EXCEPTIONS: tuple[type[Exception], ...] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
    )
    _container_type: ClassVar[pc.ContainerType] = FlextContainer
    _context_type: ClassVar[pcx.ContextType] = FlextContext

    @classmethod
    def _is_logger_carrier(
        cls,
        value: pb.AttributeProbe | None,
    ) -> TypeIs[_LoggerCarrier]:
        """Return whether value carries or can route logging context."""
        _ = cls
        return isinstance(
            value, (pl.Logger, pl.HasLogger, mp.BaseModel, *tb.CONTAINER_TYPES)
        )

    @classmethod
    def _resolve_logger(
        cls,
        first_arg: pl.Logger | _LoggerCarrier | None = None,
        *,
        func: ts.DispatchableHandler | None = None,
        func_module: str | None = None,
    ) -> pl.Logger:
        """Resolve the logger associated with the decorated call."""
        _ = cls
        if isinstance(first_arg, pl.Logger):
            return first_arg
        if isinstance(first_arg, pl.HasLogger):
            return first_arg.logger
        module_name = (
            func_module
            if isinstance(func_module, str)
            else (func.__module__ if callable(func) else __name__)
        )
        logger: pl.Logger = FlextLogger.fetch_logger(module_name)
        return logger

    @staticmethod
    def deprecated[**PCallback, TResult](
        reason: str,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Mark callable as deprecated and emit ``DeprecationWarning`` on use."""

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
        """Inject dependencies from the configured FLEXT container."""

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
