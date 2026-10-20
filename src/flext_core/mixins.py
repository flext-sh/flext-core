"""Reusable service mixins facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from contextlib import contextmanager

from flext_core import FlextContext, c, m, p, t, u
from flext_core._models import FlextMixins as FlextMixinsState


class FlextMixins(FlextMixinsState):
    """Consume model-owned state and the canonical runtime through MRO."""

    @property
    def settings(self) -> p.Settings:
        """Settings from this component's scoped runtime."""
        return self._get_runtime().settings

    @property
    def container(self) -> p.Container:
        """Container from the same runtime as context and settings."""
        return self._get_runtime().container

    @property
    def context(self) -> p.Context:
        """Context associated with this component."""
        return self._get_runtime().context

    @property
    def logger(self) -> p.Logger:
        """Reuse the kernel logger rather than bootstrap one per operation."""
        return self._get_or_create_logger()

    @classmethod
    def _get_or_create_logger(cls) -> p.Logger:
        """Reuse one logger per concrete component class.

        Returns:
            The component's cached logger.

        """
        logger_name = f"{cls.__module__}.{cls.__name__}"
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cls._logger_cache[logger_name]
        logger = u.fetch_logger(logger_name)
        with cls._cache_lock:
            cls._logger_cache[logger_name] = logger
        return logger

    @contextmanager
    def track(self, operation_name: str) -> t.OperationMetricsStream:
        """Track operation effects and always release the operation context.

        Yields:
            The mutable operation metrics.

        Raises:
            EXC_BROAD_RUNTIME: If the tracked operation fails.

        """
        stats_model = self._operation_stats.get(
            operation_name,
            m.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0}
            ),
        )
        stats = stats_model.root
        stats["operation_count"] = u.to_int(stats.get("operation_count", 0)) + 1
        try:
            with FlextContext.timed_operation(operation_name) as metrics:
                metrics_map: t.MutableMappingKV[str, t.JsonPayload] = {
                    key: u.normalize_to_container(value)
                    for key, value in metrics.root.items()
                }
                metrics_map["operation_count"] = stats["operation_count"]
                try:
                    yield metrics_map
                    if "duration_ms" in metrics_map:
                        total = u.to_float(stats.get("total_duration_ms", 0.0))
                        stats["total_duration_ms"] = total + u.to_float(
                            metrics_map.get("duration_ms", 0.0)
                        )
                except c.EXC_BROAD_RUNTIME as exc:
                    self.logger.debug(
                        c.LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION, exc_info=exc
                    )
                    stats["error_count"] = u.to_int(stats.get("error_count", 0)) + 1
                    raise
                finally:
                    count = u.to_int(stats.get("operation_count", 1), default=1)
                    errors = u.to_int(stats.get("error_count", 0))
                    stats["success_rate"] = (count - errors) / count
                    if count > 0:
                        stats["avg_duration_ms"] = (
                            u.to_float(stats.get("total_duration_ms", 0.0)) / count
                        )
                    metrics_map["error_count"] = stats["error_count"]
                    metrics_map["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics_map["avg_duration_ms"] = stats["avg_duration_ms"]
                    self._operation_stats[operation_name] = stats_model
        finally:
            _ = u.clear_scope(c.ContextScope.OPERATION)
            FlextContext.apply_operation_name("")

    def _get_runtime(self) -> p.ServiceRuntime:
        """Build the runtime once, unless a service supplies its explicit graph.

        Returns:
            The component's service runtime.

        """
        if self._runtime is None:
            self._runtime = u.build_service_runtime(self)
        return self._runtime


x = FlextMixins

__all__: t.StrSequence = ("FlextMixins", "x")
