"""Reusable service mixins facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import flext_core

if TYPE_CHECKING:
    pass


class FlextMixins(flext_core.m.ServiceState):
    """Consume model-owned state and the canonical runtime through MRO."""

    @property
    def settings(self) -> flext_core.p.Settings:
        """Settings from this component's scoped runtime."""
        return self._get_runtime().settings

    @property
    def container(self) -> flext_core.p.Container:
        """Container from the same runtime as context and settings."""
        return self._get_runtime().container

    @property
    def context(self) -> flext_core.p.Context:
        """Context associated with this component."""
        return self._get_runtime().context

    @property
    def logger(self) -> flext_core.p.Logger:
        """Consume the logging dependency elected by this component's runtime."""
        return self._get_runtime().logger

    @contextmanager
    def track(self, operation_name: str) -> flext_core.t.OperationMetricsStream:
        """Track operation effects and always release the operation context.

        Yields:
            The mutable operation metrics.

        Raises:
            EXC_BROAD_RUNTIME: If the tracked operation fails.

        """
        stats_model = self._operation_stats.get(
            operation_name,
            flext_core.m.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0}
            ),
        )
        stats = stats_model.root
        stats["operation_count"] = (
            flext_core.u.to_int(stats.get("operation_count", 0)) + 1
        )
        try:
            with flext_core.FlextContext.timed_operation(operation_name) as metrics:
                metrics_map = metrics.root
                metrics_map["operation_count"] = stats["operation_count"]
                try:
                    yield metrics_map
                    if "duration_ms" in metrics_map:
                        total = flext_core.u.to_float(
                            stats.get("total_duration_ms", 0.0)
                        )
                        stats["total_duration_ms"] = total + flext_core.u.to_float(
                            metrics_map.get("duration_ms", 0.0)
                        )
                except flext_core.c.EXC_BROAD_RUNTIME as exc:
                    self.logger.debug(
                        flext_core.c.LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION,
                        exc_info=exc,
                    )
                    stats["error_count"] = (
                        flext_core.u.to_int(stats.get("error_count", 0)) + 1
                    )
                    raise
                finally:
                    count = flext_core.u.to_int(
                        stats.get("operation_count", 1), default=1
                    )
                    errors = flext_core.u.to_int(stats.get("error_count", 0))
                    stats["success_rate"] = (count - errors) / count
                    if count > 0:
                        stats["avg_duration_ms"] = (
                            flext_core.u.to_float(stats.get("total_duration_ms", 0.0))
                            / count
                        )
                    metrics_map["error_count"] = stats["error_count"]
                    metrics_map["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics_map["avg_duration_ms"] = stats["avg_duration_ms"]
                    self._operation_stats[operation_name] = stats_model
        finally:
            _ = flext_core.u.clear_scope(flext_core.c.ContextScope.OPERATION)
            flext_core.FlextContext.apply_operation_name("")

    def _get_runtime(self) -> flext_core.p.ServiceRuntime:
        """Build the runtime once, unless a service supplies its explicit graph.

        Returns:
            The component's service runtime.

        """
        if self._runtime is None:
            self._runtime = flext_core.FlextContainer.build_service_runtime(self)
        return self._runtime


x = FlextMixins

__all__: Sequence[str] = ("FlextMixins", "x")
