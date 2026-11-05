"""Exception metrics tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, ClassVar

import flext_core

if TYPE_CHECKING:
    pass


class FlextExceptionsMetrics:
    """Exception occurrence metrics tracking."""

    _metrics_state: ClassVar[flext_core.p.ExceptionMetricsState] = (
        flext_core.m.ExceptionMetricsState()
    )

    @classmethod
    def record_exception(cls, exception_type: type[BaseException]) -> None:
        """Record an exception occurrence for metrics tracking."""
        cls._metrics_state = cls._metrics_state.record_exception(exception_type)

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._metrics_state = cls._metrics_state.clear()

    @classmethod
    def resolve_metrics_snapshot(cls) -> flext_core.p.ExceptionMetricsSnapshot:
        """Get the typed public metrics snapshot.

        Returns:
            The resulting ``m.ExceptionMetricsSnapshot``.

        """
        return cls._metrics_state.snapshot()

    @classmethod
    def resolve_metrics(cls) -> flext_core.t.JsonMapping:
        """Get exception metrics and statistics.

        Returns:
            The resulting ``t.JsonMapping``.

        """
        return cls.resolve_metrics_snapshot().to_config_map()


__all__: MutableSequence[str] = ["FlextExceptionsMetrics"]
