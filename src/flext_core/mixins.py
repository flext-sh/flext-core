"""Reusable service mixins facade."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Annotated, ClassVar, TYPE_CHECKING


from flext_core import FlextContainer, FlextContext, c, m, p, t, u
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.pydantic import FlextTypesPydantic as tp


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, MutableMapping


class FlextMixins(m.ArbitraryTypesModel):
    """Composable behaviors for dispatcher-driven services and handlers."""

    settings_type: t.SettingsClass | None = mp.Field(
        default=None,
        validate_default=True,
        exclude=True,
        description="FlextSettings class used to initialize the service runtime.",
    )

    runtime_settings: Annotated[
        p.Settings | None,
        tp.SkipValidation,
        mp.Field(
            default=None,
            exclude=True,
            description="Pre-built settings instance used directly for the runtime.",
        ),
    ] = None

    settings_overrides: t.ScalarMapping | None = mp.Field(
        default=None,
        validate_default=True,
        exclude=True,
        description="Settings overrides applied at instantiation.",
    )

    initial_context: Annotated[
        p.Context | None,
        tp.SkipValidation,
        mp.Field(
            default=None,
            exclude=True,
            description="Initial context for the service scope.",
        ),
    ] = None

    _runtime: p.ServiceRuntime | None = mp.PrivateAttr(default=None)

    _operation_stats: t.MutableMappingKV[str, p.ConfigMap] = mp.PrivateAttr(
        default_factory=dict[str, p.ConfigMap]
    )

    _logger_cache: ClassVar[MutableMapping[str, p.Logger]] = {}

    _cache_lock: ClassVar[p.Lock] = threading.Lock()

    _container_type: ClassVar[p.ContainerType] = FlextContainer

    _context_type: ClassVar[p.ContextType] = FlextContext

    _auto_context_scope: ClassVar[bool] = True

    @property
    def settings(self) -> p.Settings:
        """Runtime settings associated with this component."""
        return self._get_runtime().settings

    @property
    def container(self) -> p.Container:
        """Global FlextContainer instance with lazy initialization."""
        return self._get_runtime().container

    @property
    def context(self) -> p.Context:
        """FlextContext instance for context operations."""
        return self._get_runtime().context

    @property
    def logger(self) -> p.Logger:
        """FlextUtilitiesLogging instance for this component."""
        return self._get_or_create_logger()

    @classmethod
    def _get_or_create_logger(cls) -> p.Logger:
        """Get or create a DI-injected logger with fallback to direct creation."""
        logger_name = f"{cls.__module__}.{cls.__name__}"
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cls._logger_cache[logger_name]
        logger = u.fetch_logger(logger_name)
        with cls._cache_lock:
            cls._logger_cache[logger_name] = logger
        return logger

    @contextmanager
    def track(self, operation_name: str) -> Generator[Mapping[str, t.JsonPayload]]:
        """Track operation performance with timing and automatic context cleanup."""
        stats: p.ConfigMap = self._operation_stats.get(
            operation_name,
            m.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0}
            ),
        )
        stats["operation_count"] = u.to_int(stats.get("operation_count", 0)) + 1
        try:
            with self._context_type.timed_operation(operation_name) as metrics:
                metrics_map: MutableMapping[str, t.JsonPayload] = {
                    k: u.normalize_to_container(v) for k, v in metrics.items()
                }
                metrics_map["operation_count"] = stats["operation_count"]
                try:
                    yield metrics_map
                    if "duration_ms" in metrics_map:
                        total_dur = u.to_float(stats.get("total_duration_ms", 0.0))
                        dur_ms = u.to_float(metrics_map.get("duration_ms", 0.0))
                        stats["total_duration_ms"] = total_dur + dur_ms
                except c.EXC_BROAD_RUNTIME as exc:
                    self.logger.debug(
                        c.LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION, exc_info=exc
                    )
                    stats["error_count"] = u.to_int(stats.get("error_count", 0)) + 1
                    raise
                finally:
                    op_count = u.to_int(stats.get("operation_count", 1), default=1)
                    err_count = u.to_int(stats.get("error_count", 0))
                    stats["success_rate"] = (op_count - err_count) / op_count
                    if op_count > 0:
                        total_dur_final = u.to_float(
                            stats.get("total_duration_ms", 0.0)
                        )
                        stats["avg_duration_ms"] = total_dur_final / op_count
                    metrics_map["error_count"] = stats["error_count"]
                    metrics_map["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics_map["avg_duration_ms"] = stats["avg_duration_ms"]
                    self._operation_stats[operation_name] = stats
        finally:
            _ = u.clear_scope(c.ContextScope.OPERATION)
            self._context_type.apply_operation_name("")

    def _get_runtime(self) -> p.ServiceRuntime:
        """Get or create a runtime triple shared across mixin consumers."""
        runtime = self._runtime
        if isinstance(runtime, m.ServiceRuntime):
            return runtime
        self._runtime = u.build_service_runtime(self)
        return self._runtime


x = FlextMixins

__all__: list[str] = ["FlextMixins", "x"]
