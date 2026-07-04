"""Logging context binding and value normalization.

Extracted from FlextLogger as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from flext_core import (
    FlextConstants as c,
)

from .logging_context_part_01 import (
    FlextUtilitiesLoggingContext as FlextUtilitiesLoggingContextPart01,
)

if TYPE_CHECKING:
    import types


class FlextUtilitiesLoggingContext(FlextUtilitiesLoggingContextPart01):
    @staticmethod
    def _caller_source_path() -> str | None:
        """Get source file path with line, class and method context."""
        try:
            caller_frame = FlextUtilitiesLoggingContext._calling_frame()
            if caller_frame is None:
                return None
            return FlextUtilitiesLoggingContext._format_caller_source_path(caller_frame)
        except c.EXC_ATTR_RUNTIME_TYPE as exc:
            FlextUtilitiesLoggingContext._report_internal_logging_failure(
                c.LoggingOperation.GET_CALLER_SOURCE,
                exc,
            )
            return None

    @staticmethod
    def _calling_frame() -> types.FrameType | None:
        """Walk the stack backward and return the first frame outside the logging machinery.

        Generic: skips any frame whose source file path matches one of
        ``c.LOGGING_INTERNAL_PATH_FRAGMENTS``. The first frame outside is the
        true caller regardless of how many internal wrappers are involved.
        """
        frame = inspect.currentframe()
        if frame is None:
            return None
        skip = c.LOGGING_INTERNAL_PATH_FRAGMENTS
        while frame is not None:
            filename = frame.f_code.co_filename
            if not any(fragment in filename for fragment in skip):
                return frame
            frame = frame.f_back
        return None

    @staticmethod
    def _report_internal_logging_failure(operation: str, exc: Exception) -> None:
        with suppress(*c.CONTEXT_EXCEPTIONS):
            FlextUtilitiesLoggingContext.structlog().fetch_logger(
                c.LOGGER_NAME_FLEXT_CORE,
            ).warning(
                c.LOG_INTERNAL_OPERATION_FAILED,
                operation=operation,
                error=exc,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
            )

    @staticmethod
    def _should_include_stack_trace() -> bool:
        try:
            return logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        except c.EXC_ATTR_RUNTIME_TYPE as exc:
            FlextUtilitiesLoggingContext._report_internal_logging_failure(
                c.LoggingOperation.SHOULD_INCLUDE_STACK,
                exc,
            )
            return True


__all__: list[str] = ["FlextUtilitiesLoggingContext"]
