"""Reusable mixins for service infrastructure.

Provide shared behaviors for services and handlers including structured
logging, DI-backed context handling, and operation tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import (
    Generator,
    Mapping,
    MutableMapping,
)
from contextlib import contextmanager
from typing import (
    Annotated,
    ClassVar,
    Unpack,
)

from pydantic import ConfigDict, PrivateAttr

from flext_core import FlextContainer, FlextContext, c, e, m, p, r, t, u


class FlextMixins(m.ArbitraryTypesModel):
    """Composable behaviors for dispatcher-driven services and handlers."""

    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)
    _operation_stats: MutableMapping[str, m.ConfigMap] = PrivateAttr(
        default_factory=dict[str, m.ConfigMap],
    )
    _logger_cache: ClassVar[MutableMapping[str, p.Logger]] = {}
    _cache_lock: ClassVar[p.Lock] = threading.Lock()
    _container_type: ClassVar[type[p.Container]] = FlextContainer

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)

    @property
    def settings(self) -> p.Settings:
        """Return the runtime settings associated with this component."""
        return self._get_runtime().settings

    @property
    def container(self) -> p.Container:
        """Get global FlextContainer instance with lazy initialization."""
        return self._get_runtime().container

    @property
    def context(self) -> p.Context:
        """Get FlextContext instance for context operations."""
        return self._get_runtime().context

    @property
    def logger(self) -> p.Logger:
        """Get or create FlextLogger for this component."""
        return self._get_or_create_logger()

    @classmethod
    def _get_or_create_logger(cls) -> p.Logger:
        """Get or create DI-injected logger with fallback to direct creation."""
        logger_name = f"{cls.__module__}.{cls.__name__}"
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cls._logger_cache[logger_name]
        logger = u.fetch_logger(logger_name)
        with cls._cache_lock:
            cls._logger_cache[logger_name] = logger
        return logger

    settings_type: Annotated[
        type | None,
        m.Field(
            exclude=True,
            description="Settings class used to initialize the service.",
        ),
    ] = None
    runtime_settings: Annotated[
        p.Settings | None,
        m.Field(
            exclude=True,
            description="Pre-built settings instance used directly for the runtime.",
        ),
    ] = None
    settings_overrides: Annotated[
        Mapping[str, t.Container] | None,
        m.Field(
            exclude=True,
            description="Settings overrides applied at instantiation.",
        ),
    ] = None
    initial_context: Annotated[
        p.Context | None,
        m.Field(
            exclude=True,
            description="Initial context for the service scope.",
        ),
    ] = None

    @contextmanager
    def track(self, operation_name: str) -> Generator[Mapping[str, t.ValueOrModel]]:
        """Track operation performance with timing and automatic context cleanup."""
        stats: m.ConfigMap = self._operation_stats.get(
            operation_name,
            m.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0}
            ),
        )
        stats["operation_count"] = u.to_int(stats.get("operation_count", 0)) + 1
        try:
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                metrics_map: MutableMapping[str, t.ValueOrModel] = {
                    str(k): u.to_plain_container(u.normalize_to_container(v))
                    for k, v in metrics.items()
                }
                metrics_map["operation_count"] = stats["operation_count"]
                try:
                    yield metrics_map
                    if "duration_ms" in metrics_map:
                        total_dur = u.to_float(
                            stats.get("total_duration_ms", 0.0),
                        )
                        dur_ms = u.to_float(
                            metrics_map.get("duration_ms", 0.0),
                        )
                        stats["total_duration_ms"] = total_dur + dur_ms
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    self.logger.debug(
                        c.LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION,
                        exc_info=exc,
                    )
                    stats["error_count"] = u.to_int(stats.get("error_count", 0)) + 1
                    raise
                finally:
                    op_count = u.to_int(
                        stats.get("operation_count", 1),
                        default=1,
                    )
                    err_count = u.to_int(stats.get("error_count", 0))
                    stats["success_rate"] = (op_count - err_count) / op_count
                    if op_count > 0:
                        total_dur_final = u.to_float(
                            stats.get("total_duration_ms", 0.0),
                        )
                        stats["avg_duration_ms"] = total_dur_final / op_count
                    metrics_map["error_count"] = stats["error_count"]
                    metrics_map["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics_map["avg_duration_ms"] = stats["avg_duration_ms"]
                    self._operation_stats[operation_name] = stats
        finally:
            _ = u.clear_scope(c.ContextScope.OPERATION)
            FlextContext.Request.apply_operation_name("")

    def _get_runtime(self) -> m.ServiceRuntime:
        """Return or create a runtime triple shared across mixin consumers."""
        runtime = self._runtime if hasattr(self, "_runtime") else None
        if isinstance(runtime, m.ServiceRuntime):
            return runtime
        self._runtime = u.build_service_runtime(self)
        return self._runtime

    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic container registration."""
        effective_service_name: str = (
            service_name
            if service_name is not None and u.matches_type(service_name, str)
            else self.__class__.__name__
        )
        register_result = self._register_in_container(effective_service_name)
        if register_result.failure:
            error_msg = register_result.error
            if error_msg is None:
                error_msg = c.ERR_SERVICE_REGISTRATION_FAILED
            if "already registered" not in error_msg.lower():
                self.logger.warning(
                    c.LOG_SERVICE_REGISTRATION_FAILED,
                    service_name=effective_service_name,
                    error=register_result.error or c.ERR_SERVICE_REGISTRATION_FAILED,
                )

    def _register_in_container(self, service_name: str) -> p.Result[bool]:
        """Register self in global container for service discovery."""
        container = self.container
        was_registered = container.has(service_name)
        _ = container.bind(service_name, self)
        if was_registered or container.has(service_name):
            return r[bool].ok(True)
        operation = "register service in container"
        return r[bool].fail(
            e.render_error_template(
                c.ERR_TEMPLATE_FAILED_WITH_ERROR,
                operation=operation,
                error=c.ERR_SERVICE_REGISTRATION_FAILED,
                params=m.OperationErrorParams(
                    operation=operation,
                    reason=c.ERR_SERVICE_REGISTRATION_FAILED,
                ),
            ),
        )


x = FlextMixins

__all__: list[str] = ["FlextMixins", "x"]
