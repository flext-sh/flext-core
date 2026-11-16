"""Logging exception resolution, metadata and formatting.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from types import TracebackType

from flext_core import p, t


class FlextUtilitiesLoggingExceptionInfo:
    """Own native exception resolution and its sanitized metadata."""

    @staticmethod
    def resolve_logging_exception_info(
        value: p.LoggingEventValue | tuple[p.AttributeProbe, ...] | None,
    ) -> t.LoggingExceptionInfo | None:
        """Resolve native exception inputs once, retaining a saved traceback.

        Returns:
            The exception triple or None when exception capture is disabled.

        Raises:
            TypeError: If the native exception input is invalid.

        """
        if value is True:
            value = sys.exception()
        if isinstance(value, BaseException):
            return type(value), value, value.__traceback__
        if value is None or value is False:
            return None
        if not isinstance(value, tuple) or len(value) != len(sys.exc_info()):
            msg = "exc_info must be an exception, saved tuple, boolean or None"
            raise TypeError(msg)
        if (
            isinstance(value[0], type)
            and issubclass(value[0], BaseException)
            and isinstance(value[1], BaseException)
            and (value[2] is None or isinstance(value[2], TracebackType))
        ):
            return value[0], value[1], value[2]
        msg = "exc_info must be an exception, saved tuple, boolean or None"
        raise TypeError(msg)

    @staticmethod
    def _exception_metadata(
        logger: p.OutputLogger,
        method_name: str,
        event: p.LoggingEvent,
    ) -> p.LoggingEvent:
        """Keep causal metadata before the native processor consumes exc_info.

        Returns:
            The event with its original exception metadata.

        """
        _ = logger, method_name
        info = FlextUtilitiesLoggingExceptionInfo.resolve_logging_exception_info(
            event.get("exc_info")
        )
        if info is not None:
            exception = info[1]
            event["_original_exception"] = exception
            event["exception_type"] = type(exception).__name__
            event["exception_message"] = str(exception)
        return event

    @staticmethod
    def _format_exception(exc_info: t.LoggingExceptionInfo) -> str:
        """Render native exception information without expanding its traceback.

        Returns:
            The exception type and message.

        """
        return f"{exc_info[0].__name__}: {exc_info[1]}"


__all__: list[str] = ["FlextUtilitiesLoggingExceptionInfo"]
