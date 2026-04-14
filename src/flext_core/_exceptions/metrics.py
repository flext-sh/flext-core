"""Exception metrics tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core import m, t


class FlextExceptionsMetrics:
    """Exception occurrence metrics tracking."""

    _metrics_state: ClassVar[m.ExceptionMetricsState] = m.ExceptionMetricsState()

    @classmethod
    def record_exception(cls, exception_type: type[BaseException]) -> None:
        """Record an exception occurrence for metrics tracking."""
        cls._metrics_state = cls._metrics_state.record_exception(exception_type)

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._metrics_state = cls._metrics_state.clear()

    @classmethod
    def resolve_metrics_snapshot(cls) -> m.ExceptionMetricsSnapshot:
        """Get the typed public metrics snapshot."""
        return cls._metrics_state.snapshot()

    @classmethod
    def resolve_metrics(cls) -> t.ConfigMap:
        """Get exception metrics and statistics."""
        return cls.resolve_metrics_snapshot().to_config_map()


__all__: list[str] = ["FlextExceptionsMetrics"]
