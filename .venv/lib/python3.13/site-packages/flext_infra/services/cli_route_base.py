"""Shared typed behavior for CLI route services."""

from __future__ import annotations

from collections.abc import Callable

from flext_infra import p, t


class CliRouteBase:
    """Provide the common result-value widening contract for route handlers."""

    @staticmethod
    def as_route_value(value: t.Cli.ResultValue) -> t.Cli.ResultValue:
        """Widen a concrete result payload to the CLI route contract value."""
        return value

    @staticmethod
    def result_handler[TParams, TResult: t.Cli.ResultValue](
        handler: Callable[[TParams], p.Result[TResult]],
    ) -> p.Cli.ResultRouteHandler:
        """Erase one concrete result payload at the heterogeneous route boundary."""

        def execute(params: TParams) -> p.Result[t.Cli.ResultValue]:
            return handler(params).map(CliRouteBase.as_route_value)

        return execute


__all__: list[str] = ["CliRouteBase"]
