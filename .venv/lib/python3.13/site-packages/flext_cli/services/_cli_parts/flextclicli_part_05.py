"""FLEXT CLI - Unified Typer abstraction service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_cli import c, r, settings, t, u
from flext_cli.services._cli_parts.flextclicli_part_04 import (
    FlextCliCli as FlextCliCliPart04,
)

if TYPE_CHECKING:
    from flext_cli import p


class FlextCliCli(FlextCliCliPart04):
    """Implementation part for FlextCliCli."""

    @classmethod
    def register_result_callback[M: t.Cli.ModelLike, TResult: t.Cli.ResultValue](
        cls,
        app: p.Cli.Application,
        *,
        handler: p.Cli.ResultCommandHandler[M, TResult],
        model_cls: t.ModelClass[M],
        settings: t.Cli.ModelLike | None = None,
        success_formatter: t.Cli.SuccessMessageFormatter[TResult] | None = None,
        success_message: str | None = None,
        success_type: c.Cli.MessageTypes = c.Cli.MessageTypes.SUCCESS,
    ) -> None:
        """Register one model/result handler as the application root callback."""
        execute = cls._build_result_executor(
            handler=handler,
            success_formatter=success_formatter,
            success_message=success_message,
            success_type=success_type,
        )
        cls.register_callback(
            app, command=cls.model_command(model_cls, execute, settings=settings)
        )

    @classmethod
    def register_result_command[M: t.Cli.ModelLike, TResult: t.Cli.ResultValue](
        cls,
        app: p.Cli.Application,
        *,
        handler: p.Cli.ResultCommandHandler[M, TResult],
        help_text: str,
        # mro-j47u (codex): route registration preserves the model protocol.
        model_cls: t.ModelClass[M],
        name: str,
        settings: t.Cli.ModelLike | None = None,
        success_formatter: t.Cli.SuccessMessageFormatter[TResult] | None = None,
        success_message: str | None = None,
        success_type: c.Cli.MessageTypes = c.Cli.MessageTypes.SUCCESS,
    ) -> None:
        """Register a model command that normalizes `r[...]` CLI handling."""
        execute = cls._build_result_executor(
            handler=handler,
            success_formatter=success_formatter,
            success_message=success_message,
            success_type=success_type,
        )
        cls.register_command(
            app,
            name=name,
            help_text=help_text,
            command=cls.model_command(model_cls, execute, settings=settings),
        )

    @classmethod
    def _build_result_executor[M: t.Cli.ModelLike, TResult: t.Cli.ResultValue](
        cls,
        *,
        handler: p.Cli.ResultCommandHandler[M, TResult],
        success_formatter: t.Cli.SuccessMessageFormatter[TResult] | None = None,
        success_message: str | None = None,
        success_type: c.Cli.MessageTypes = c.Cli.MessageTypes.SUCCESS,
    ) -> p.Cli.ModelCommandHandler[M]:
        """Build the shared executor used by single and batched route registration."""

        def _exit_with_failure(result: p.Result[TResult]) -> None:
            # NOTE (multi-agent): programmatic execution propagates the original
            # Result; direct framework execution finalizes it at this boundary.
            if not u.Cli.framework_exit_result(result):
                cls.exit(code=cls.finalize_result(result))

        def execute(params: M) -> t.JsonValue:
            result: p.Result[TResult] = handler(params)
            if result.failure:
                _exit_with_failure(result)
            result_value: TResult = result.value
            message = u.Cli.commands_resolve_success_message(
                result_value=result_value,
                success_message=success_message,
                success_formatter=success_formatter,
            )
            if message:
                u.Cli.commands_emit_success_message(message, success_type)
            return True

        return execute

    @classmethod
    def register_result_route(
        cls, app: p.Cli.Application, *, route: p.Cli.ResultCommandRoute
    ) -> None:
        """Register a declarative result route on a Typer app."""

        def route_execute(params: t.Cli.ModelLike) -> p.Result[t.Cli.ResultValue]:
            result = route.handler(params)
            if result.failure:
                return r[t.Cli.ResultValue].from_failure(result)
            return r[t.Cli.ResultValue].ok(result.value)

        cls.register_result_command(
            app,
            name=route.name,
            help_text=route.help_text,
            model_cls=route.model_cls,
            handler=route_execute,
            success_message=route.success_message,
            success_formatter=route.success_formatter,
            success_type=route.success_type,
        )

    @classmethod
    def register_result_routes(
        cls, app: p.Cli.Application, routes: t.SequenceOf[p.Cli.ResultCommandRoute]
    ) -> None:
        """Register multiple heterogeneous result routes in one call."""
        for route in routes:
            cls.register_result_route(app, route=route)

    @staticmethod
    def finalize_result[TResult: t.Cli.ResultValue](
        result: p.Result[TResult], *, failure_exit_code: int = c.Cli.EXIT_CODE_FAILURE
    ) -> int:
        """Finalize one public CLI Result into output/logging and an exit code."""
        if result.success:
            return c.Cli.EXIT_CODE_SUCCESS
        u.Cli.commands_emit_result_error(result, verbose=settings.cli_verbose)
        # NOTE (multi-agent): the outermost CLI owns its process contract while
        # FlextCliCli remains the single error-emission boundary.
        return failure_exit_code


__all__: list[str] = ["FlextCliCli"]
