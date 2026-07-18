"""Combined decorator composition.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal, overload, TYPE_CHECKING

from flext_core._decorators._railway import FlextDecoratorsRailway
from flext_core._models.handler import FlextModelsHandler as mh
# mro-cqxy (ADR-011): runtime import — used in a runtime-evaluated annotation
from collections.abc import Callable
from flext_core._protocols.result import FlextProtocolsResult as pr


if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as tb


class FlextDecoratorsCombined(FlextDecoratorsRailway):
    """Compose multiple decorator behaviors in one public decorator."""

    @overload
    @classmethod
    def combined[**PCallback, TResult](
        cls,
        *,
        inject_deps: tb.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: Literal[False] = False,
        railway_error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]: ...

    @overload
    @classmethod
    def combined[**PCallback, TResult](
        cls,
        *,
        inject_deps: tb.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: Literal[True],
        railway_error_code: str | None = None,
    ) -> Callable[
        [Callable[PCallback, TResult]], Callable[PCallback, pr.Result[TResult]]
    ]: ...

    @classmethod
    def combined[**PCallback, TResult](
        cls,
        *,
        inject_deps: tb.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: bool = False,
        railway_error_code: str | None = None,
    ) -> Callable[
        [Callable[PCallback, TResult]],
        Callable[PCallback, TResult] | Callable[PCallback, pr.Result[TResult]],
    ]:
        """Apply injection, operation logging, and optional railway wrapping."""
        railway = mh.CombinedRailwayOptions.model_validate({
            "enabled": railway_enabled,
            "error_code": railway_error_code,
        })
        if railway.enabled:

            def railway_decorator(
                func: Callable[PCallback, TResult],
            ) -> Callable[PCallback, pr.Result[TResult]]:
                result = cls.railway(error_code=railway.error_code)(func)
                if inject_deps:
                    result = cls.inject(**inject_deps)(result)
                operation_logger: Callable[
                    [Callable[PCallback, pr.Result[TResult]]],
                    Callable[PCallback, pr.Result[TResult]],
                ] = cls.log_operation(
                    operation_name=operation_name, track_perf=track_perf
                )
                return operation_logger(result)

            return railway_decorator

        def standard_decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            result = func
            if inject_deps:
                result = cls.inject(**inject_deps)(result)
            operation_logger: Callable[
                [Callable[PCallback, TResult]], Callable[PCallback, TResult]
            ] = cls.log_operation(operation_name=operation_name, track_perf=track_perf)
            return operation_logger(result)

        return standard_decorator


__all__: list[str] = ["FlextDecoratorsCombined"]
