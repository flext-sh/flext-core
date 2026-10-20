"""Structured logging with context propagation and dependency injection.

Family module for ``FlextUtilitiesLogging``: the composed class lives inside
its private family package so canonical facade files import it from a
private-family origin (ENFORCE-046); the root ``loggings.py`` namespace
module re-exports the single canonical binding.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Self

import structlog

from flext_core import FlextBoundLogger, c, m, p, t
from flext_core._utilities import FlextUtilitiesLoggingContext


class FlextUtilitiesLogging(FlextUtilitiesLoggingContext):
    """Typed direct entrypoints over the native structlog composition.

    Composed via MRO from:
    - FlextUtilitiesLoggingConfig — structlog configuration, async writer, processors
    - FlextUtilitiesLoggingContext — native context binding and SDK value boundaries
    """

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
            context = m.ConfigMap(
                root={
                    c.MetadataKey.DURATION_SECONDS: elapsed,
                    c.HandlerType.OPERATION: self._operation_name,
                    c.FIELD_STATUS: status,
                },
            )
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            emit = self.logger.info if success else self.logger.error
            event = f"{self._operation_name} {status}"
            try:
                _ = emit(
                    event, **FlextUtilitiesLogging.to_container_context(context.root)
                )
            except BaseException as logging_error:
                if exc_val is not None and exc_val is not logging_error:
                    exc_val.add_note(
                        f"Logging also failed: {type(logging_error).__name__}"
                    )
                    raise exc_val.with_traceback(_exc_tb) from exc_val.__cause__
                raise

    @classmethod
    def _direct_logger(cls) -> p.Logger:
        """Let native structlog resolve the real caller on each direct emission.

        Returns:
            The caller's canonical logger.

        """
        native = structlog.stdlib.LoggerFactory(ignore_frame_names=[__name__])()
        return cls.create_module_logger(native.name)

    @classmethod
    def log(
        cls,
        message: str,
        *args: p.LogValue,
        level: str | None = None,
        **context: p.LogValue,
    ) -> p.LogResult:
        """Emit an ordinary event through the elected Core composition.

        Returns:
            The native logging result.

        """
        return cls._direct_logger().log(
            level if level is not None else c.LogLevel.INFO, message, *args, **context
        )

    @classmethod
    def debug(
        cls, message: str, *args: p.LogValue, **context: p.LogValue
    ) -> p.LogResult:
        """Emit a DEBUG event without allocating a consumer-owned logger.

        Returns:
            The native logging result.

        """
        return cls.log(message, *args, level=c.LogLevel.DEBUG, **context)

    @classmethod
    def trace(
        cls, message: str, *args: p.LogValue, **context: p.LogValue
    ) -> p.LogResult:
        """Emit opt-in high-volume detail at DEBUG, never another severity.

        Returns:
            The native logging result.

        """
        return cls._direct_logger().trace(message, *args, **context)

    @classmethod
    def warning(
        cls, message: str, *args: p.LogValue, **context: p.LogValue
    ) -> p.LogResult:
        """Emit a WARNING event through the same pipeline.

        Returns:
            The native logging result.

        """
        return cls.log(message, *args, level=c.LogLevel.WARNING, **context)

    @classmethod
    def failure(
        cls,
        message: str,
        *args: p.LogValue,
        exception: BaseException | None = None,
        **context: p.LogValue,
    ) -> p.LogResult:
        """Emit ERROR with original causal data; trace only changes its detail.

        Returns:
            The native logging result.

        """
        if exception is not None:
            context["exception"] = exception
        return cls._direct_logger().exception(message, *args, **context)

    @classmethod
    def fetch_logger(cls, name: str) -> p.Logger:
        """Construct the canonical lazy logger for an explicit name.

        Returns:
            The named canonical logger.

        """
        return cls.create_module_logger(name)

    @classmethod
    def create_module_logger(
        cls,
        name: str,
        *,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
    ) -> p.Logger:
        """Construct a lazy module logger without configuring a destination.

        Returns:
            The lazy logger with the supplied context.

        """
        # Foreign utility facades must construct the canonical logger, not `cls`.
        return FlextBoundLogger(name, context=context)


__all__: tuple[str, ...] = ("FlextUtilitiesLogging",)
