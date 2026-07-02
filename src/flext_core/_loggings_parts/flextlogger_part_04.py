"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
import types
from typing import Self

from flext_core import FlextConstants as c, FlextProtocols as p
from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug

from .flextlogger_part_03 import (
    FlextLogger as FlextLoggerPart03,
)


class FlextLogger(FlextLoggerPart03):
    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: p.Logger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
            super().__init__()
            self.logger = logger
            self._operation_name = operation_name
            self._start_time: float = 0.0

        def __enter__(self) -> Self:
            """Start tracking."""
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            """Log operation result with timing."""
            elapsed = time.time() - self._start_time
            success = exc_type is None
            status = "success" if success else "failed"
            context: mc.ConfigMap = mc.ConfigMap(
                root={
                    c.MetadataKey.DURATION_SECONDS: elapsed,
                    c.HandlerType.OPERATION: self._operation_name,
                    c.FIELD_STATUS: status,
                },
            )
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            if success:
                _ = self.logger.info(
                    f"{self._operation_name} {status}",
                    **FlextLogger.to_container_context(context.root),
                )
            else:
                _ = self.logger.error(
                    f"{self._operation_name} {status}",
                    **FlextLogger.to_container_context(context.root),
                )

    class Integration:
        """Application-layer integration helpers using structlog directly."""

        @staticmethod
        def setup_service_infrastructure(
            *,
            service_name: str,
            service_version: str | None = None,
            enable_context_correlation: bool = True,
        ) -> None:
            """Setup complete service infrastructure."""
            sl = FlextLogger.structlog()
            _ = sl.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                _ = sl.contextvars.bind_contextvars(
                    service_version=service_version,
                )
            if enable_context_correlation:
                correlation_id = f"flext-{ug.generate_id().replace('-', '')[:12]}"
                _ = sl.contextvars.bind_contextvars(
                    correlation_id=correlation_id,
                )
            sl.fetch_logger(__name__).info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: mc.ConfigMap | None = None,
        ) -> None:
            """Track domain event with context correlation."""
            sl = FlextLogger.structlog()
            context_vars = sl.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            sl.fetch_logger(__name__).info(
                "Domain event emitted",
                event_name=event_name,
                aggregate_id=aggregate_id,
                event_data=event_data,
                correlation_id=correlation_id,
            )

        @staticmethod
        def track_service_resolution(
            service_name: str,
            *,
            resolved: bool = True,
            error_message: str | None = None,
        ) -> None:
            """Track service resolution with context correlation."""
            sl = FlextLogger.structlog()
            context_vars = sl.contextvars.get_contextvars()
            correlation_id = context_vars.get(c.ContextKey.CORRELATION_ID)
            logger = sl.fetch_logger(__name__)
            if resolved:
                logger.info(
                    "Service resolved",
                    service_name=service_name,
                    correlation_id=correlation_id,
                )
            else:
                logger.error(
                    "Service resolution failed",
                    service_name=service_name,
                    error=error_message,
                    correlation_id=correlation_id,
                )


__all__: list[str] = ["FlextLogger"]
