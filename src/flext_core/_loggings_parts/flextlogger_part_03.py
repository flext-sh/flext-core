"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.logging import FlextConstantsLogging as cl
from flext_core._exceptions.factories import FlextExceptionsFactories as ef
from flext_core.result import FlextResult as r

from .flextlogger_part_02 import (
    FlextUtilitiesLogging as FlextUtilitiesLoggingPart02,
)

if TYPE_CHECKING:
    from flext_core import FlextTypes as t


class FlextUtilitiesLogging(FlextUtilitiesLoggingPart02):
    def unbind(self, *keys: str, safe: bool = False) -> Self:
        """Unbind keys from logger — implements BindableLogger protocol."""
        bound_logger = (
            self.logger.try_unbind(*keys) if safe else self.logger.unbind(*keys)
        )
        return self.__class__(self.name, _bound_logger=bound_logger)

    def try_unbind(self, *keys: str) -> Self:
        """Unbind keys while ignoring missing values."""
        return self.unbind(*keys, safe=True)

    def warning(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log warning message."""
        return self._log_standard_level(cl.LogLevel.WARNING, msg, *args, **kw)

    @staticmethod
    def _resolve_level_name(level: cl.LogLevel | str) -> str:
        match level:
            case cl.LogLevel() as enum_level:
                level_raw: str = enum_level.value
            case _:
                level_raw = level
        return level_raw.lower()

    @staticmethod
    def _resolve_log_context(
        args: t.SequenceOf[t.LogValue],
        context: t.MappingKV[str, t.LogValue],
    ) -> t.JsonMapping:
        resolved_context: dict[str, t.LogValue] = dict(context)
        if "source" not in resolved_context and (
            source_path := FlextUtilitiesLogging._caller_source_path()
        ):
            resolved_context["source"] = source_path
        for idx, arg in enumerate(args):
            resolved_context[f"arg_{idx}"] = arg
        return FlextUtilitiesLogging._to_scalar_context(resolved_context)

    def _log(
        self,
        level: cl.LogLevel | str,
        event: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Consolidate all log level methods into one internal logging path."""
        try:
            level_str = FlextUtilitiesLogging._resolve_level_name(level)
            scalar_context = FlextUtilitiesLogging._resolve_log_context(args, context)
            getattr(self.logger, level_str)(event, **scalar_context)
            return r[bool].ok(True)
        except ce.EXC_BROAD_RUNTIME as exc:
            return ef.fail_operation("logging", exc)

    def _log_standard_level(
        self,
        level: cl.LogLevel,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        return self._log(level, msg, *args, **kw)

    def critical(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log critical message."""
        return self._log_standard_level(cl.LogLevel.CRITICAL, msg, *args, **kw)

    def debug(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log debug message."""
        return self._log_standard_level(cl.LogLevel.DEBUG, msg, *args, **kw)

    def error(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log error message."""
        return self._log_standard_level(cl.LogLevel.ERROR, msg, *args, **kw)

    def info(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log info message."""
        return self._log_standard_level(cl.LogLevel.INFO, msg, *args, **kw)

    def log(
        self,
        level: str,
        message: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Log message with specified level."""
        level_enum: cl.LogLevel = cl.LogLevel(level.upper())
        converted_args: tuple[t.JsonValue, ...] = tuple(
            FlextUtilitiesLogging._to_container_value(arg) for arg in args
        )
        return self._log(level_enum, message, *converted_args, **context)

    def trace(
        self,
        message: str,
        *args: t.LogValue,
        **kwargs: t.JsonPayload,
    ) -> t.LogResult:
        """Log trace message."""
        try:
            try:
                formatted_message = message % args if args else message
            except ce.EXC_TYPE_VALIDATION:
                formatted_message = f"{message} | args={args!r}"
            self.logger.debug(
                formatted_message,
                **FlextUtilitiesLogging._to_scalar_context(kwargs),
            )
            return r[bool].ok(True)
        except ce.EXC_BROAD_RUNTIME as exc:
            FlextUtilitiesLogging._report_internal_logging_failure("trace", exc)
            return ef.fail_operation("trace logging", exc)


__all__: list[str] = ["FlextUtilitiesLogging"]
