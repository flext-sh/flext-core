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
    Sequence,
)
from contextlib import contextmanager
from types import ModuleType
from typing import (
    Annotated,
    ClassVar,
    Unpack,
)

from pydantic import ConfigDict, Field, PrivateAttr, ValidationError

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextLogger,
    FlextSettings,
    c,
    m,
    p,
    r,
    t,
    u,
)


class FlextMixins(m.ArbitraryTypesModel):
    """Composable behaviors for dispatcher-driven services and handlers."""

    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)
    _operation_stats: MutableMapping[str, t.ConfigMap] = PrivateAttr(
        default_factory=dict[str, t.ConfigMap],
    )

    _logger_cache: ClassVar[MutableMapping[str, p.Logger]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)

    @property
    def config(self) -> p.Settings:
        """Return the runtime configuration associated with this component."""
        return self._get_runtime().config

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

    config_type: Annotated[
        type | None,
        Field(
            default=None,
            description="Configuration class to initialize the service.",
        ),
    ] = None
    config_overrides: Annotated[
        t.ContainerMapping | None,
        Field(
            default=None,
            description="Configuration overrides applied at instantiation.",
        ),
    ] = None
    initial_context: Annotated[
        p.Context | None,
        Field(default=None, description="Initial context for the service scope."),
    ] = None

    @staticmethod
    def _clear_operation_context() -> None:
        """Clear operation scope context (preserves request/application scopes)."""
        _ = FlextLogger.clear_scope(c.ContextScope.OPERATION)
        FlextContext.Request.apply_operation_name("")

    @staticmethod
    def _propagate_context(operation_name: str) -> None:
        """Propagate context for current operation using FlextContext."""
        FlextContext.Request.apply_operation_name(operation_name)
        _ = FlextContext.Utilities.ensure_correlation_id()

    @staticmethod
    def _with_operation_context(
        operation_name: str,
        **operation_data: t.Scalar,
    ) -> None:
        """Set operation context with level-based binding (DEBUG/ERROR/normal)."""
        FlextMixins._propagate_context(operation_name)
        if operation_data:
            debug_data: t.ConfigMap = t.ConfigMap(
                root={
                    k: v for k, v in operation_data.items() if k in c.DEBUG_CONTEXT_KEYS
                },
            )
            error_data: t.ConfigMap = t.ConfigMap(
                root={
                    k: v for k, v in operation_data.items() if k in c.ERROR_CONTEXT_KEYS
                },
            )
            normal_data: t.ConfigMap = t.ConfigMap(
                root={
                    k: v
                    for k, v in operation_data.items()
                    if k not in c.DEBUG_CONTEXT_KEYS and k not in c.ERROR_CONTEXT_KEYS
                },
            )
            all_context_data: t.ConfigMap = normal_data.model_copy()
            if debug_data:
                merged_debug: t.ConfigMap = all_context_data.model_copy()
                merged_debug.update(debug_data.root)
                all_context_data = merged_debug
            if error_data:
                merged_error: t.ConfigMap = all_context_data.model_copy()
                merged_error.update(error_data.root)
                all_context_data = merged_error
            if all_context_data:
                metadata_context: Mapping[str, t.RuntimeAtomic] = {
                    key: u.normalize_to_container(value)
                    for key, value in all_context_data.root.items()
                }
                _ = FlextLogger.bind_global_context(**metadata_context)
            if normal_data:
                normal_metadata_context: Mapping[str, t.RuntimeAtomic] = {
                    key: u.normalize_to_container(value)
                    for key, value in normal_data.root.items()
                }
                _ = FlextLogger.bind_context(
                    c.ContextScope.OPERATION,
                    **normal_metadata_context,
                )

    @contextmanager
    def track(self, operation_name: str) -> Generator[Mapping[str, t.ValueOrModel]]:
        """Track operation performance with timing and automatic context cleanup."""
        stats: t.ConfigMap = self._operation_stats.get(
            operation_name,
            t.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0},
            ),
        )
        stats["operation_count"] = u.to_int(stats.get("operation_count", 0)) + 1
        try:
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                metrics_map: MutableMapping[str, t.ValueOrModel] = {
                    str(k): u.normalize_to_container(v) for k, v in metrics.items()
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
            FlextMixins._clear_operation_context()

    def _enrich_context(self, **context_data: t.Scalar) -> None:
        """Log service information ONCE at initialization (not bound to context)."""
        service_context: t.ConfigMap = t.ConfigMap(
            root={
                c.ContextKey.SERVICE_NAME: self.__class__.__name__,
                c.ContextKey.SERVICE_MODULE: self.__class__.__module__,
                **context_data,
            },
        )
        self.logger.info(
            c.LOG_SERVICE_INITIALIZED,
            return_result=False,
            **u.normalize_log_payload(service_context.root),
        )

    def _get_runtime(self) -> m.ServiceRuntime:
        """Return or create a runtime triple shared across mixin consumers."""
        runtime = self._runtime if hasattr(self, "_runtime") else None
        if isinstance(runtime, m.ServiceRuntime):
            return runtime

        options = FlextMixins._resolve_bootstrap_options(self)
        config_type_raw = getattr(self, "config_type", None)
        overrides: t.ContainerMapping | None = None
        initial_ctx: p.Context | None = None
        bootstrap_services: t.ServiceMap | None = None
        bootstrap_factories: t.FactoryMap | None = None
        bootstrap_resources: t.ResourceMap | None = None
        bootstrap_wire_modules: Sequence[ModuleType] | None = None
        bootstrap_wire_packages: t.StrSequence | None = None
        bootstrap_wire_classes: Sequence[type] | None = None

        if options:
            config_type_raw, overrides, initial_ctx = (
                options.config_type or config_type_raw,
                options.config_overrides,
                options.context,
            )
            bootstrap_services = u.filter_registerable_services(options.services)
            bootstrap_factories = options.factories
            bootstrap_resources = options.resources
            bootstrap_wire_modules, bootstrap_wire_packages, bootstrap_wire_classes = (
                u.resolve_wire_targets(
                    options.wire_modules,
                    options.wire_packages,
                    options.wire_classes,
                )
            )

        config_cls_typed: type[p.Settings] = (
            config_type_raw if u.settings_type(config_type_raw) else FlextSettings
        )
        if overrides is None:
            overrides = getattr(self, "config_overrides", None)
        overrides_typed: t.ScalarMapping | None = (
            {k: v for k, v in overrides.items() if u.scalar(v)}
            if overrides is not None
            else None
        )
        runtime_config: p.Settings = config_cls_typed.fetch_global(
            overrides=overrides_typed,
        )
        if initial_ctx is None:
            initial_ctx = getattr(self, "initial_context", None)
        runtime_context: p.Context = (
            initial_ctx if isinstance(initial_ctx, p.Context) else FlextContext.create()
        )
        runtime_container = FlextContainer.create().scoped(
            config=runtime_config,
            context=runtime_context,
            services=bootstrap_services,
            factories=bootstrap_factories,
            resources=bootstrap_resources,
        )
        if bootstrap_wire_modules or bootstrap_wire_packages or bootstrap_wire_classes:
            runtime_container.wire_modules(
                modules=bootstrap_wire_modules,
                packages=bootstrap_wire_packages,
                classes=bootstrap_wire_classes,
            )

        self._runtime = m.ServiceRuntime(
            container=runtime_container,
            config=runtime_config,
            context=runtime_context,
        )
        return self._runtime

    @staticmethod
    def _resolve_bootstrap_options(
        instance: FlextMixins,
    ) -> m.RuntimeBootstrapOptions | None:
        """Load and validate bootstrap options from the mixin consumer."""
        try:
            options_raw: t.BootstrapInput = getattr(
                instance,
                "_runtime_bootstrap_options",
                lambda: None,
            )()
            if options_raw is None:
                return None
            match options_raw:
                case m.RuntimeBootstrapOptions():
                    return options_raw
                case dict() | Mapping():
                    try:
                        return m.RuntimeBootstrapOptions.model_validate(options_raw)
                    except ValidationError:
                        return None
                case _:
                    try:
                        return m.RuntimeBootstrapOptions.model_validate(
                            options_raw,
                            from_attributes=True,
                        )
                    except ValidationError:
                        return None
        except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
            FlextLogger.create_module_logger(__name__).warning(
                c.LOG_RUNTIME_BOOTSTRAP_OPTIONS_LOAD_FAILED,
                exc_info=exc,
            )
            return None

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

    def _log_config_once(
        self,
        config: t.ConfigMap,
        message: str = "Configuration loaded",
    ) -> None:
        """Log configuration ONCE without binding to context."""
        config_typed: t.ConfigMap = t.ConfigMap(root=dict(config))
        self.logger.info(
            message,
            **u.normalize_log_payload(config_typed.root),
        )

    def _log_with_context(self, level: str, message: str, **extra: t.Scalar) -> None:
        """Log message with automatic context data inclusion."""
        correlation_id = FlextContext.Correlation.resolve_correlation_id()
        operation_name = FlextContext.Request.resolve_operation_name()
        context_data: t.ConfigMap = t.ConfigMap(
            root={
                c.ContextKey.CORRELATION_ID: u.normalize_to_container(
                    correlation_id or "",
                ),
                c.ContextKey.OPERATION_NAME: u.normalize_to_container(
                    operation_name or "",
                ),
                **{k: u.normalize_to_container(v) for k, v in extra.items()},
            },
        )
        log_method = (
            getattr(self.logger, level)
            if hasattr(self.logger, level)
            else self.logger.info
        )
        _ = log_method(
            message,
            **u.normalize_log_payload(context_data.root),
        )

    def _register_in_container(self, service_name: str) -> r[bool]:
        """Register self in global container for service discovery."""
        try:
            container = self.container
            was_registered = container.has_service(service_name)
            _ = container.register(service_name, self)
            if was_registered or container.has_service(service_name):
                return r[bool].ok(True)
            msg = c.ERR_SERVICE_REGISTRATION_FAILED
            raise RuntimeError(msg)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            return r[bool].fail(str(e))


x = FlextMixins

__all__ = ["FlextMixins", "x"]
