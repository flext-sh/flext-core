"""Combined decorator composition.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from flext_core import m
from flext_core._decorators import FlextDecoratorsRailway
from flext_core._protocols import FlextProtocolsResult
from flext_core._typings import FlextTypingBase

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextDecoratorsCombined(FlextDecoratorsRailway):
    """Compose multiple decorator behaviors in one public decorator."""

    @overload
    @classmethod
    def combined[**PCallback, TResult](
        cls,
        *,
        inject_deps: FlextTypingBase.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: Literal[False] = False,
        railway_error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]: ...

    @overload
    @classmethod
    def combined(
        cls,
        *,
        inject_deps: FlextTypingBase.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: Literal[True],
        railway_error_code: str | None = None,
    ) -> FlextProtocolsResult.RailwayDecorator: ...

    @classmethod
    def combined[**PCallback, TResult](
        cls,
        *,
        inject_deps: FlextTypingBase.StrMapping | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        railway_enabled: bool = False,
        railway_error_code: str | None = None,
    ) -> (
        Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]
        | FlextProtocolsResult.RailwayDecorator
    ):
        """Apply injection, operation logging, and optional railway wrapping.

        Returns:
            A return-preserving decorator, or the result-preserving railway decorator.

        """
        railway = m.CombinedRailwayOptions.model_validate({
            "enabled": railway_enabled,
            "error_code": railway_error_code,
        })
        if railway.enabled:

            @overload
            def railway_decorator[**PCall, TValue](
                func: Callable[PCall, FlextProtocolsResult.Result[TValue]],
                /,
            ) -> Callable[PCall, FlextProtocolsResult.Result[TValue]]: ...

            @overload
            def railway_decorator[**PCall, TValue](
                func: Callable[PCall, TValue | FlextProtocolsResult.Result[TValue]],
                /,
            ) -> Callable[PCall, FlextProtocolsResult.Result[TValue]]: ...

            def railway_decorator[**PCall, TValue](
                func: Callable[PCall, TValue | FlextProtocolsResult.Result[TValue]],
                /,
            ) -> Callable[PCall, FlextProtocolsResult.Result[TValue]]:
                result: Callable[PCall, FlextProtocolsResult.Result[TValue]] = (
                    cls.railway(error_code=railway.error_code)(func)
                )
                if inject_deps:
                    result = cls.inject(**inject_deps)(result)
                operation_logger: Callable[
                    [Callable[PCall, FlextProtocolsResult.Result[TValue]]],
                    Callable[PCall, FlextProtocolsResult.Result[TValue]],
                ] = cls.log_operation(
                    operation_name=operation_name,
                    track_perf=track_perf,
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
                [Callable[PCallback, TResult]],
                Callable[PCallback, TResult],
            ] = cls.log_operation(operation_name=operation_name, track_perf=track_perf)
            return operation_logger(result)

        return standard_decorator


__all__: list[str] = ["FlextDecoratorsCombined"]
