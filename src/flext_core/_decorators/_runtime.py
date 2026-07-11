"""Factory registration and timeout decorators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING

from flext_core._constants.mixins import FlextConstantsMixins as cm
from flext_core._constants.timeout import FlextConstantsTimeout as ct
from flext_core._constants.validation import FlextConstantsValidation as cv
from flext_core._decorators._combined import FlextDecoratorsCombined
from flext_core._exceptions.types import FlextExceptionsTypes as et
from flext_core._models.container import FlextModelsContainer as mc

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextDecoratorsRuntime(FlextDecoratorsCombined):
    """Decorators for runtime factory metadata and timeout enforcement."""

    @staticmethod
    def factory[**P, T](
        name: str,
        *,
        singleton: bool = False,
        lazy: bool = True,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Mark functions as factories for DI container discovery."""

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            settings = mc.FactoryDecoratorConfig(
                name=name,
                singleton=singleton,
                lazy=lazy,
            )
            setattr(func, cm.FACTORY_ATTR, settings)
            return func

        return decorator

    @classmethod
    def timeout[**PCallback, TResult](
        cls,
        timeout_seconds: float | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Raise a FLEXT timeout error when an operation exceeds the duration."""
        max_duration = (
            timeout_seconds
            if timeout_seconds is not None
            else ct.DEFAULT_TIMEOUT_SECONDS
        )

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = (
                            f"Operation {func.__name__} exceeded timeout of "
                            f"{max_duration}s (took {duration:.2f}s)"
                        )
                        raise et.TimeoutError(
                            msg,
                            error_code=error_code or "OPERATION_TIMEOUT",
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            duration_seconds=duration,
                            original_error="",
                        )
                    return result
                except et.TimeoutError:
                    raise
                except cls._CAUGHT_EXCEPTIONS as exc:
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = (
                            f"Operation {func.__name__} exceeded timeout of "
                            f"{max_duration}s (took {duration:.2f}s) and raised "
                            f"{exc.__class__.__name__}"
                        )
                        raise et.TimeoutError(
                            msg,
                            error_code=error_code or cv.ErrorCode.TIMEOUT_ERROR.value,
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            duration_seconds=duration,
                            original_error=str(exc),
                        ) from exc
                    raise

            return wrapper

        return decorator


__all__: list[str] = ["FlextDecoratorsRuntime"]
