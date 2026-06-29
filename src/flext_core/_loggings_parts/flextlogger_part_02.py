"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import traceback

from flext_core import FlextConstants as c, FlextExceptions as e, FlextTypes as t
from flext_core.result import FlextResult as r

from .flextlogger_part_01 import (
    FlextLogger as FlextLoggerPart01,
)


class FlextLogger(FlextLoggerPart01):
    def _exception_context_from_inputs(
        self,
        resolved_exception: Exception | None,
        raw_exception: t.LogValue | None,
        exc_info_value: t.LogValue,
        extra_context: t.MappingKV[str, t.LogValue],
    ) -> t.JsonDict:
        include_stack_trace = self._should_include_stack_trace()
        context_dict: t.JsonDict = {}
        if resolved_exception is not None:
            context_dict["exception_type"] = resolved_exception.__class__.__name__
            context_dict["exception_message"] = str(resolved_exception)
            if include_stack_trace:
                context_dict["stack_trace"] = "".join(
                    traceback.format_exception(
                        resolved_exception.__class__,
                        resolved_exception,
                        resolved_exception.__traceback__,
                    ),
                )
        elif exc_info_value and include_stack_trace:
            context_dict["stack_trace"] = traceback.format_exc()
        for key, value in extra_context.items():
            if key in {"exception", "exc_info"}:
                continue
            if not isinstance(value, BaseException):
                context_dict[key] = FlextLogger._to_container_value(value)
        if resolved_exception is None and isinstance(raw_exception, BaseException):
            context_dict["exception_type"] = raw_exception.__class__.__name__
            context_dict["exception_message"] = str(raw_exception)
        return context_dict

    def exception(
        self,
        msg: str,
        *args: t.LogValue,
        **kw: t.LogValue,
    ) -> t.LogResult:
        """Log exception with conditional stack trace (DEBUG only)."""
        message = msg
        filtered_args: tuple[t.JsonValue, ...] = tuple(
            FlextLogger._to_container_value(arg)
            for arg in args
            if not isinstance(arg, BaseException)
        )
        try:
            resolved_exception: Exception | None = (
                args[0] if args and isinstance(args[0], Exception) else None
            )
            context_dict = self._exception_context_from_inputs(
                resolved_exception,
                kw.get("exception"),
                kw.get("exc_info", True),
                kw,
            )
            _ = self.logger.error(
                message,
                *filtered_args,
                **FlextLogger._to_scalar_context(context_dict),
            )
            return r[bool].ok(True)
        except c.EXC_BROAD_RUNTIME as exc:
            FlextLogger._report_internal_logging_failure("exception", exc)
            return e.fail_operation("exception logging", exc)

    def build_exception_context(
        self,
        *,
        exception: Exception | None,
        exc_info: bool,
        context: t.MappingKV[str, t.JsonPayload | Exception],
    ) -> t.JsonMapping:
        """Build normalized structured exception context for logging."""
        result: t.JsonDict = {
            k: str(v)
            if isinstance(v, Exception)
            else FlextLogger._to_container_value(v)
            for k, v in context.items()
        }
        if exception is not None:
            result["exception_type"] = exception.__class__.__name__
            result["exception_message"] = str(exception)
        if exc_info and self._should_include_stack_trace():
            result["stack_trace"] = traceback.format_exc()
        return result


__all__: list[str] = ["FlextLogger"]
