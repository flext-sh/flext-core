"""Reusable mixins for service infrastructure.

Provide shared behaviors for services and handlers including structured
logging, DI-backed context handling, and operation tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Unpack,
    cast,
)

from pydantic import ConfigDict, PrivateAttr

from flext_core import FlextContainer, FlextContext, c, m, p, t, u

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        MutableMapping,
    )

type _RuntimeBootstrapValue = (
    t.GuardInput | p.Settings | p.Context | t.SettingsClass | None
)


class FlextMixins(m.ArbitraryTypesModel):
    """Composable behaviors for dispatcher-driven services and handlers."""

    _runtime_bootstrap_field_names: ClassVar[frozenset[str]] = frozenset({
        "settings_type",
        "runtime_settings",
        "settings_overrides",
        "initial_context",
    })

    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)

    _operation_stats: MutableMapping[str, m.ConfigMap] = PrivateAttr(
        default_factory=dict[str, m.ConfigMap],
    )

    _logger_cache: ClassVar[MutableMapping[str, p.Logger]] = {}

    _cache_lock: ClassVar[p.Lock] = threading.Lock()

    _container_type: ClassVar[p.ContainerType] = FlextContainer

    _context_type: ClassVar[p.ContextType] = FlextContext

    _auto_context_scope: ClassVar[bool] = True

    _settings_type: t.SettingsClass | None = PrivateAttr(default=None)

    _runtime_settings: p.Settings | None = PrivateAttr(default=None)

    _settings_overrides: t.JsonMapping | None = PrivateAttr(default=None)

    _initial_context: p.Context | None = PrivateAttr(default=None)

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        _ = kwargs
        super().__init_subclass__()

    def __init__(self, **model_data: _RuntimeBootstrapValue) -> None:
        """Initialize Pydantic fields and private runtime bootstrap state."""
        runtime_state: dict[str, _RuntimeBootstrapValue] = {}
        for field_name in self._runtime_bootstrap_field_names:
            if field_name in model_data:
                runtime_state[field_name] = model_data.pop(field_name)
        self.__pydantic_validator__.validate_python(model_data, self_instance=self)
        self._apply_runtime_bootstrap_state(runtime_state)

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
        """The or create DI-injected logger with fallback to direct creation."""
        logger_name = f"{cls.__module__}.{cls.__name__}"
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cls._logger_cache[logger_name]
        logger = u.fetch_logger(logger_name)
        with cls._cache_lock:
            cls._logger_cache[logger_name] = logger
        return logger

    @staticmethod
    def _coerce_settings_type(value: _RuntimeBootstrapValue) -> t.SettingsClass | None:
        """Validate the explicit settings class accepted by service constructors."""
        if value is None:
            return None
        if (
            isinstance(value, type)
            and callable(getattr(value, "fetch_global", None))
            and callable(getattr(value, "model_copy", None))
        ):
            return cast("t.SettingsClass", value)
        msg = "settings_type must satisfy p.SettingsType"
        raise TypeError(msg)

    @staticmethod
    def _coerce_runtime_settings(value: _RuntimeBootstrapValue) -> p.Settings | None:
        """Validate an injected runtime settings instance."""
        if value is None:
            return None
        if isinstance(value, p.Settings):
            return value
        msg = "runtime_settings must satisfy p.Settings"
        raise TypeError(msg)

    @staticmethod
    def _coerce_settings_overrides(
        value: _RuntimeBootstrapValue,
    ) -> t.JsonMapping | None:
        """Validate runtime settings overrides without exposing them as model fields."""
        if value is None:
            return None
        if isinstance(value, Mapping):
            validated: t.JsonMapping = t.json_mapping_adapter().validate_python(value)
            return validated
        msg = "settings_overrides must be a JSON mapping"
        raise TypeError(msg)

    @staticmethod
    def _coerce_initial_context(value: _RuntimeBootstrapValue) -> p.Context | None:
        """Validate an injected runtime context instance."""
        if value is None:
            return None
        if isinstance(value, p.Context):
            return value
        msg = "initial_context must satisfy p.Context"
        raise TypeError(msg)

    @property
    def settings_type(self) -> t.SettingsClass | None:
        """Settings class used to initialize the service runtime."""
        return self._settings_type

    @settings_type.setter
    def settings_type(self, value: t.SettingsClass | None) -> None:
        self._settings_type = value

    @property
    def runtime_settings(self) -> p.Settings | None:
        """Pre-built settings instance used directly for the runtime."""
        return self._runtime_settings

    @runtime_settings.setter
    def runtime_settings(self, value: p.Settings | None) -> None:
        self._runtime_settings = value

    @property
    def settings_overrides(self) -> t.JsonMapping | None:
        """Settings overrides applied at instantiation."""
        return self._settings_overrides

    @settings_overrides.setter
    def settings_overrides(self, value: t.JsonMapping | None) -> None:
        self._settings_overrides = value

    @property
    def initial_context(self) -> p.Context | None:
        """Initial context for the service scope."""
        return self._initial_context

    @initial_context.setter
    def initial_context(self, value: p.Context | None) -> None:
        self._initial_context = value

    def _apply_runtime_bootstrap_state(
        self,
        runtime_state: t.MappingKV[str, _RuntimeBootstrapValue],
    ) -> None:
        """Apply constructor-provided runtime state to private attributes."""
        if "settings_type" in runtime_state:
            self.settings_type = self._coerce_settings_type(
                runtime_state["settings_type"],
            )
        if "runtime_settings" in runtime_state:
            self.runtime_settings = self._coerce_runtime_settings(
                runtime_state["runtime_settings"],
            )
        if "settings_overrides" in runtime_state:
            self.settings_overrides = self._coerce_settings_overrides(
                runtime_state["settings_overrides"],
            )
        if "initial_context" in runtime_state:
            self.initial_context = self._coerce_initial_context(
                runtime_state["initial_context"],
            )

    @contextmanager
    def track(self, operation_name: str) -> Generator[Mapping[str, t.JsonPayload]]:
        """Track operation performance with timing and automatic context cleanup."""
        stats: m.ConfigMap = self._operation_stats.get(
            operation_name,
            m.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0},
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
                        total_dur = u.to_float(
                            stats.get("total_duration_ms", 0.0),
                        )
                        dur_ms = u.to_float(
                            metrics_map.get("duration_ms", 0.0),
                        )
                        stats["total_duration_ms"] = total_dur + dur_ms
                except c.EXC_BROAD_RUNTIME as exc:
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
            self._context_type.apply_operation_name("")

    def _get_runtime(self) -> m.ServiceRuntime:
        """The or create a runtime triple shared across mixin consumers."""
        runtime = self._runtime
        if isinstance(runtime, m.ServiceRuntime):
            return runtime
        self._runtime = u.build_service_runtime(self)
        return self._runtime


__all__: list[str] = ["FlextMixins"]
