"""CLI command helpers shared through ``u.Cli``."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

# mro-j47u (codex): formatter contracts are owned once by the t facade.
from flext_cli import c, r, t
from flext_cli._utilities.output import FlextCliUtilitiesOutput as uo
from flext_core import u

if TYPE_CHECKING:
    from flext_cli import p


class FlextCliUtilitiesCommands:
    """Helpers for result-command messaging in the public Typer DSL."""

    @staticmethod
    def commands_resolve_success_message[TResult: t.Cli.ResultValue](
        *,
        result_value: TResult,
        success_message: str | None,
        success_formatter: t.Cli.SuccessMessageFormatter[TResult] | None,
    ) -> str | None:
        """Resolve success message using formatter/value fallback order."""
        if success_formatter is not None:
            formatted: str = success_formatter(result_value)
            return formatted
        normalized_value: t.JsonValue = u.normalize_to_json_value(result_value)
        match normalized_value:
            case {c.Cli.DICT_KEY_MESSAGE: str() as candidate} if candidate:
                return candidate
            case str() as candidate if candidate:
                return candidate
            case _:
                return success_message

    @staticmethod
    def commands_emit_success_message(
        message: str, success_type: c.Cli.MessageTypes
    ) -> None:
        """Emit success output as raw payload or styled CLI message."""
        rendered = (
            message
            if message.lstrip().startswith(("{", "["))
            else uo.output_message_payload(message, success_type)[0]
        )
        uo.emit_raw(f"{rendered}\n")

    @staticmethod
    def commands_emit_result_error[TResult: t.Cli.ResultValue](
        result: p.Result[TResult], *, verbose: bool = False
    ) -> None:
        """Finalize one failed Result through structured logging and CLI output."""
        # NOTE (multi-agent): keep the canonical Result intact through every
        # service layer; only this outer CLI boundary exposes its failure state.
        error = r.require_error(result)
        logger = u.fetch_logger(__name__)
        if isinstance(result.exception, Exception):
            logger.error(
                error,
                error_code=result.error_code,
                error_data=result.error_data,
                exc_info=result.exception,
            )
        else:
            logger.error(
                error, error_code=result.error_code, error_data=result.error_data
            )
        uo.emit_raw(
            f"{uo.output_message_payload(error, c.Cli.MessageTypes.ERROR)[0]}\n"
        )
        if result.error_code:
            uo.emit_raw(f"   [{result.error_code}]\n")
        if verbose and result.exception is not None:
            detail = "".join(
                traceback.format_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )
            )
            uo.emit_raw(detail if detail.endswith("\n") else f"{detail}\n")


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesCommands"]
