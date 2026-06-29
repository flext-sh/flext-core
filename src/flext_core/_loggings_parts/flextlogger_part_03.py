"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from contextlib import suppress
from typing import Self

from flext_core import FlextConstants as c, FlextExceptions as e, FlextTypes as t
from flext_core.result import FlextResult as r

from .flextlogger_part_02 import (
    FlextLogger as FlextLoggerPart02,
)


class FlextLogger(FlextLoggerPart02):
    def unbind(self, *keys: str, safe: bool = False) -> Self:
        """Unbind keys from logger — implements BindableLogger protocol."""
        if safe:
            with suppress(KeyError, ValueError, AttributeError):
                bound_logger = self.logger.unbind(*keys)
                return self.__class__(self.name, _bound_logger=bound_logger)
            return self
        bound_logger = self.logger.unbind(*keys)
        return self.__class__(self.name, _bound_logger=bound_logger)

    def warning(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log warning message."""
        return self._log_standard_level(c.LogLevel.WARNING, msg, *args, **kw)

    @staticmethod
    def _resolve_level_name(level: c.LogLevel | str) -> str:
        match level:
            case c.LogLevel() as enum_level:
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
            source_path := FlextLogger._caller_source_path()
        ):
            resolved_context["source"] = source_path
        for idx, arg in enumerate(args):
            resolved_context[f"arg_{idx}"] = arg
        return FlextLogger._to_scalar_context(resolved_context)

    def _log(
        self,
        level: c.LogLevel | str,
        event: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Internal logging method — consolidates all log level methods."""
        try:
            level_str = FlextLogger._resolve_level_name(level)
            scalar_context = FlextLogger._resolve_log_context(args, context)
            getattr(self.logger, level_str)(event, **scalar_context)
            return r[bool].ok(True)
        except c.EXC_BROAD_RUNTIME as exc:
            return e.fail_operation("logging", exc)

    def _log_standard_level(
        self,
        level: c.LogLevel,
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
        return self._log_standard_level(c.LogLevel.CRITICAL, msg, *args, **kw)

    def debug(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log debug message."""
        return self._log_standard_level(c.LogLevel.DEBUG, msg, *args, **kw)

    def error(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log error message."""
        return self._log_standard_level(c.LogLevel.ERROR, msg, *args, **kw)

    def info(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log info message."""
        return self._log_standard_level(c.LogLevel.INFO, msg, *args, **kw)

    def log(
        self,
        level: str,
        message: str,
        *args: t.LogValue,
        **context: t.LogValue,
    ) -> t.LogResult:
        """Log message with specified level."""
        level_enum: c.LogLevel | str = level
        with suppress(ValueError, AttributeError):
            level_enum = c.LogLevel(level.upper())
        converted_args: tuple[t.JsonValue, ...] = tuple(
            FlextLogger._to_container_value(arg) for arg in args
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
            except c.EXC_TYPE_VALIDATION:
                formatted_message = f"{message} | args={args!r}"
            self.logger.debug(
                formatted_message,
                **FlextLogger._to_scalar_context(kwargs),
            )
            return r[bool].ok(True)
        except c.EXC_BROAD_RUNTIME as exc:
            FlextLogger._report_internal_logging_failure("trace", exc)
            return e.fail_operation("trace logging", exc)


__all__: list[str] = ["FlextLogger"]
