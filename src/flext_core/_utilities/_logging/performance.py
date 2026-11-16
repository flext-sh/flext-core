"""Performance tracking context manager over the canonical logger.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from types import TracebackType
from typing import Self

from flext_core import c, p


class FlextUtilitiesLoggingPerformance:
    """Own the performance-tracking context manager as a logging part."""

    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: p.Logger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
            super().__init__()
            self.logger = logger
            self._operation_name = operation_name
            self._start_time: float = 0.0

        def __enter__(self) -> Self:
            """Start timing this operation.

            Returns:
                This active performance tracker.

            """
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            _exc_tb: TracebackType | None,
        ) -> None:
            """Log timing while preserving the original business exception."""
            elapsed = time.time() - self._start_time
            success = exc_type is None
            status = c.Status.SUCCESS if success else c.Status.FAILED
            context: MutableMapping[str, p.LogValue] = {
                c.MetadataKey.DURATION_SECONDS: elapsed,
                c.HandlerType.OPERATION: self._operation_name,
                c.FIELD_STATUS: status,
            }
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            emit = self.logger.info if success else self.logger.error
            event = f"{self._operation_name} {status}"
            try:
                _ = emit(
                    event,
                    **context,
                )
            except BaseException as logging_error:
                if exc_val is not None and exc_val is not logging_error:
                    exc_val.add_note(
                        f"Logging also failed: {type(logging_error).__name__}"
                    )
                    raise exc_val.with_traceback(_exc_tb) from exc_val.__cause__
                raise


__all__: list[str] = ["FlextUtilitiesLoggingPerformance"]
