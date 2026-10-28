"""Base decorator helpers and dependency injection decorator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import warnings
from functools import wraps
from typing import TYPE_CHECKING, ClassVar, TypeIs

from flext_core import FlextContainer, FlextContext, c, m, p, t, u

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextDecoratorsBase:
    """Base helpers shared by concrete decorator namespaces."""

    type _LoggerCarrier = p.HasLogger | p.Logger | t.JsonPayload | m.BaseModel
    _CAUGHT_EXCEPTIONS: tuple[type[Exception], ...] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
    )
    _container_type: ClassVar[p.ContainerType] = FlextContainer
    _context_type: ClassVar[p.ContextType] = FlextContext

    @classmethod
    def _is_logger_carrier(
        cls,
        value: p.AttributeProbe | None,
    ) -> TypeIs[_LoggerCarrier]:
        """Return whether value carries or can route logging context.

        Returns:
            Whether value carries or can route logging context.

        """
        return isinstance(
            value,
            (
                p.Logger,
                p.HasLogger,
                m.BaseModel,
                *c.CONTAINER_TYPES,
            ),
        )

    @classmethod
    def _resolve_logger(
        cls,
        first_arg: p.Logger | _LoggerCarrier | None = None,
        *,
        func: p.DispatchableHandler | None = None,
        func_module: str | None = None,
    ) -> p.Logger:
        """Resolve the logger associated with the decorated call.

        Returns:
            The resulting ``pl.Logger``.

        """
        if isinstance(first_arg, p.Logger):
            return first_arg
        if isinstance(first_arg, p.HasLogger):
            return first_arg.logger
        module_name = (
            func_module
            if isinstance(func_module, str)
            else (func.__module__ if callable(func) else __name__)
        )
        logger: p.Logger = u.fetch_logger(
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

        A keyword the caller omits is resolved from the container; an
        unresolvable dependency raises the container's failure instead of
        silently leaving the parameter default in place.

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
                        kwargs[name] = container.resolve(service_key).unwrap()
                return func(*args, **kwargs)

            return wrapper

        return decorator


__all__: list[str] = ["FlextDecoratorsBase"]
