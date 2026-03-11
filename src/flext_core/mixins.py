"""Reusable mixins for service infrastructure.

Provide shared behaviors for services and handlers including structured
logging, DI-backed context handling, and operation tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, suppress
from types import ModuleType
from typing import ClassVar, override

from pydantic import BaseModel, PrivateAttr

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextLogger,
    FlextRuntime,
    FlextSettings,
    c,
    m,
    p,
    r,
    t,
    u,
)
from flext_core._models.service import FlextModelsService


class FlextMixins(FlextRuntime):
    """Composable behaviors for dispatcher-driven services and handlers.

    These mixins centralize DI container access, structured logging, and
    context management so dispatcher-executed services can stay focused on
    domain work while still emitting `r` outcomes and metrics.

    Properties:
    - ``container``: Lazy ``FlextContainer`` singleton lookups for DI wiring.
    - ``logger``: Cached ``FlextLogger`` resolution for structured logs.
    - ``context``: Per-operation ``FlextContext`` for correlation metadata.
    - ``config``: Thread-safe ``FlextSettings`` access for runtime settings.

    Key methods:
    - ``track``: Context manager that records timing/err counts per operation.
    - ``_with_operation_context`` / ``_clear_operation_context``: Scoped
      context bindings used by dispatcher pipelines.
    - Delegated ``FlextRuntime``/``r`` helpers for railway flows.

    Example:
        class MyService(x):
            def process(
                self, data: m.ConfigMap
            ) -> r[m.ConfigMap]:
                with self.track("process"):
                    self.logger.info("Processing", size=len(data))
                    return self.ok({"status": "processed"})

    """

    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)

    @staticmethod
    def fail(
        error: str | None,
        error_code: str | None = None,
        error_data: m.ConfigMap | None = None,
    ) -> r[t.ContainerValue]:
        """Create failed result with error message."""
        fail_error_data: t.ConfigurationMapping = (
            {
                str(k): FlextRuntime.normalize_to_general_value(v)
                for k, v in error_data.root.items()
            }
            if error_data is not None
            else {}
        )
        return r[t.ContainerValue].fail(
            error, error_code=error_code, error_data=fail_error_data
        )

    @staticmethod
    def ok[T](value: T) -> r[T]:
        """Create successful result wrapping value."""
        return r[T].ok(value)

    traverse = staticmethod(r.traverse)
    accumulate_errors = staticmethod(r.accumulate_errors)

    @classmethod
    def to_dict(cls, obj: BaseModel | t.ConfigurationMapping | None) -> m.ConfigMap:
        """Convert BaseModel/dict to dict (None → empty dict). Use x.to_dict at call sites."""
        if obj is None:
            return m.ConfigMap(root={})
        if isinstance(obj, m.ConfigMap):
            return obj
        if isinstance(obj, BaseModel):
            model_dump_result = obj.model_dump()
            try:
                normalized_model_dump: dict[str, t.ContainerValue] = {}
                for key, value in model_dump_result.items():
                    normalized_value: t.ContainerValue
                    if value is None:
                        normalized_value = ""
                    elif isinstance(value, t.Primitives | BaseModel):
                        normalized_value = FlextRuntime.normalize_to_general_value(
                            value
                        )
                    else:
                        normalized_value = str(value)
                    normalized_model_dump[str(key)] = normalized_value
                return m.ConfigMap.model_validate(normalized_model_dump)
            except (TypeError, ValueError, AttributeError) as exc:
                cls._get_or_create_logger().debug(
                    "Model dump normalization fallback to string conversion",
                    exc_info=exc,
                )
                return m.ConfigMap(root={"value": str(model_dump_result)})
        try:
            normalized_mapping: dict[str, t.ContainerValue] = {}
            for key, value in obj.items():
                normalized_mapping_value: t.ContainerValue = (
                    FlextRuntime.normalize_to_general_value(value)
                    if isinstance(value, t.Primitives | type(None) | BaseModel)
                    else str(value)
                )
                normalized_mapping[str(key)] = normalized_mapping_value
            return m.ConfigMap.model_validate(normalized_mapping)
        except (TypeError, ValueError, AttributeError) as exc:
            cls._get_or_create_logger().debug(
                "Object-to-config-map normalization failed", exc_info=exc
            )
            return m.ConfigMap(root={})

    @staticmethod
    def ensure_result(
        value: t.ContainerValue | r[t.ContainerValue],
    ) -> r[t.ContainerValue]:
        """Wrap value in r if not already wrapped. Use x.ensure_result at call sites."""
        if isinstance(value, r):
            return value
        return r[t.ContainerValue].ok(value)

    _logger_cache: ClassVar[MutableMapping[str, FlextLogger]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(
        cls, **kwargs: t.Scalar | m.ConfigMap | Sequence[t.Scalar]
    ) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)

    @property
    def config(self) -> p.Config:
        """Return the runtime configuration associated with this component.

        Returns p.Config protocol type for type safety. In practice, this is
        FlextSettings - callers needing concrete attributes can use type
        identity or MRO checks for type narrowing.
        """
        return self._get_runtime().config

    @property
    def container(self) -> p.DI:
        """Get global FlextContainer instance with lazy initialization."""
        return self._get_runtime().container

    @property
    def context(self) -> p.Context:
        """Get FlextContext instance for context operations."""
        return self._get_runtime().context

    @property
    @override
    def logger(self) -> FlextLogger:
        """Get or create FlextLogger for this component."""
        return self._get_or_create_logger()

    @classmethod
    def _get_or_create_logger(cls) -> FlextLogger:
        """Get or create DI-injected logger with fallback to direct creation."""
        logger_name = f"{cls.__module__}.{cls.__name__}"
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cls._logger_cache[logger_name]
        try:
            container = FlextContainer.create()
            logger_key = f"logger:{logger_name}"
            logger_result = container.get(logger_key, type_cls=FlextLogger)
            if logger_result.is_success:
                logger: FlextLogger = logger_result.value
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger
                return logger
            logger = FlextLogger(logger_name)
            container_impl: p.DI = container
            with suppress(ValueError, TypeError):
                if hasattr(container_impl, "register_factory"):

                    def logger_factory() -> t.ContainerValue:
                        return logger_name

                    _ = container_impl.register(
                        logger_key, logger_factory, kind="factory"
                    )
                else:
                    _ = container_impl.register(logger_key, logger)
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger
            return logger
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            logger = FlextLogger(logger_name)
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger
            return logger

    @classmethod
    def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
        """Hook to customize runtime creation for mixin consumers.

        Returns:
            p.RuntimeBootstrapOptions: Pydantic model with optional runtime configuration options.

        """
        return FlextModelsService.RuntimeBootstrapOptions()

    @staticmethod
    def _clear_operation_context() -> None:
        """Clear operation scope context (preserves request/application scopes)."""
        _ = FlextLogger.clear_scope("operation")
        FlextContext.Request.set_operation_name("")

    @staticmethod
    def _propagate_context(operation_name: str) -> None:
        """Propagate context for current operation using FlextContext."""
        FlextContext.Request.set_operation_name(operation_name)
        _ = FlextContext.Utilities.ensure_correlation_id()

    @staticmethod
    def _with_operation_context(
        operation_name: str, **operation_data: t.ContainerValue
    ) -> None:
        """Set operation context with level-based binding (DEBUG/ERROR/normal)."""
        FlextMixins._propagate_context(operation_name)
        if operation_data:
            debug_keys = {"schema", "params"}
            error_keys = {"stack_trace", "exception", "traceback", "error_details"}
            debug_data: m.ConfigMap = m.ConfigMap(
                root={k: v for k, v in operation_data.items() if k in debug_keys}
            )
            error_data: m.ConfigMap = m.ConfigMap(
                root={k: v for k, v in operation_data.items() if k in error_keys}
            )
            normal_data: m.ConfigMap = m.ConfigMap(
                root={
                    k: v
                    for k, v in operation_data.items()
                    if k not in debug_keys and k not in error_keys
                }
            )
            all_context_data: m.ConfigMap = normal_data.model_copy()
            if debug_data:
                merged_debug: m.ConfigMap = all_context_data.model_copy()
                merged_debug.update(debug_data.root)
                all_context_data = merged_debug
            if error_data:
                merged_error: m.ConfigMap = all_context_data.model_copy()
                merged_error.update(error_data.root)
                all_context_data = merged_error
            if all_context_data:
                metadata_context: dict[str, t.MetadataValue] = {
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in all_context_data.root.items()
                }
                _ = FlextLogger.bind_global_context(**metadata_context)
            if normal_data:
                normal_metadata_context: dict[str, t.MetadataValue] = {
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in normal_data.root.items()
                }
                _ = FlextLogger.bind_context(
                    c.Context.SCOPE_OPERATION, **normal_metadata_context
                )

    @contextmanager
    def track(self, operation_name: str) -> Iterator[Mapping[str, t.ContainerValue]]:
        """Track operation performance with timing and automatic context cleanup."""
        stats_attr = f"_stats_{operation_name}"
        stats: m.ConfigMap = (
            getattr(self, stats_attr)
            if hasattr(self, stats_attr)
            else m.ConfigMap(
                root={"operation_count": 0, "error_count": 0, "total_duration_ms": 0.0}
            )
        )
        op_count_raw = u.get(stats, "operation_count", default=0)
        stats["operation_count"] = (
            int(op_count_raw) if isinstance(op_count_raw, (int, float, str)) else 0
        ) + 1
        try:
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                metrics_map: dict[str, t.ContainerValue] = (
                    {
                        str(k): FlextRuntime.normalize_to_general_value(v)
                        for k, v in metrics.items()
                    }
                    if hasattr(metrics, "items")
                    else {}
                )
                metrics_map["operation_count"] = stats["operation_count"]
                try:
                    yield metrics_map
                    if "duration_ms" in metrics_map:
                        total_dur_raw = u.get(stats, "total_duration_ms", default=0.0)
                        dur_ms_raw = u.get(metrics_map, "duration_ms", default=0.0)
                        total_dur = (
                            float(total_dur_raw)
                            if isinstance(total_dur_raw, (int, float, str))
                            else 0.0
                        )
                        dur_ms = (
                            float(dur_ms_raw)
                            if isinstance(dur_ms_raw, (int, float, str))
                            else 0.0
                        )
                        stats["total_duration_ms"] = total_dur + dur_ms
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    log_debug = getattr(self.logger, "debug", None)
                    if callable(log_debug):
                        log_debug(
                            "Tracked operation raised expected exception", exc_info=exc
                        )
                    err_raw = u.get(stats, "error_count", default=0)
                    stats["error_count"] = (
                        int(err_raw) if isinstance(err_raw, (int, float, str)) else 0
                    ) + 1
                    raise
                finally:
                    op_raw = u.get(stats, "operation_count", default=1)
                    err_raw2 = u.get(stats, "error_count", default=0)
                    op_count = (
                        int(op_raw) if isinstance(op_raw, (int, float, str)) else 1
                    )
                    err_count = (
                        int(err_raw2) if isinstance(err_raw2, (int, float, str)) else 0
                    )
                    stats["success_rate"] = (op_count - err_count) / op_count
                    if op_count > 0:
                        total_raw = u.get(stats, "total_duration_ms", default=0.0)
                        total_dur_final = (
                            float(total_raw)
                            if isinstance(total_raw, (int, float, str))
                            else 0.0
                        )
                        stats["avg_duration_ms"] = total_dur_final / op_count
                    metrics_map["error_count"] = stats["error_count"]
                    metrics_map["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics_map["avg_duration_ms"] = stats["avg_duration_ms"]
                    setattr(self, stats_attr, stats)
        finally:
            FlextMixins._clear_operation_context()

    def _enrich_context(self, **context_data: t.ContainerValue) -> None:
        """Log service information ONCE at initialization (not bound to context)."""
        service_context: m.ConfigMap = m.ConfigMap(
            root={
                "service_name": self.__class__.__name__,
                "service_module": self.__class__.__module__,
                **context_data,
            }
        )
        self.logger.info(
            "Service initialized", return_result=False, **service_context.root
        )

    def _get_runtime(self) -> m.ServiceRuntime:
        """Return or create a runtime triple shared across mixin consumers."""
        runtime = self._runtime if hasattr(self, "_runtime") else None
        match runtime:
            case m.ServiceRuntime() as service_runtime:
                return service_runtime
            case _:
                pass
        runtime_options_callable = (
            self._runtime_bootstrap_options
            if hasattr(self, "_runtime_bootstrap_options")
            else None
        )
        options_candidate = (
            runtime_options_callable()
            if callable(runtime_options_callable)
            else FlextModelsService.RuntimeBootstrapOptions()
        )
        try:
            options: p.RuntimeBootstrapOptions = (
                FlextModelsService.RuntimeBootstrapOptions.model_validate(
                    options_candidate
                )
            )
        except (TypeError, ValueError, AttributeError) as exc:
            self.logger.debug(
                "Runtime bootstrap options validation failed", exc_info=exc
            )
            options = FlextModelsService.RuntimeBootstrapOptions()
        config_type_raw = options.config_type
        config_cls_typed: type[FlextSettings]
        if config_type_raw is not None and issubclass(config_type_raw, FlextSettings):
            config_cls_typed = config_type_raw
        else:
            config_cls_typed = FlextSettings
        runtime_config = config_cls_typed.get_global(overrides=options.config_overrides)
        runtime_context: p.Context = (
            options.context if options.context is not None else FlextContext.create()
        )
        runtime_config_typed: FlextSettings = runtime_config
        runtime_container = FlextContainer.create().scoped(
            config=runtime_config_typed,
            context=runtime_context,
            subproject=options.subproject,
            services=options.services,
            factories=options.factories,
            resources=options.resources,
        )
        if options.container_overrides:
            runtime_container.configure(dict(options.container_overrides))
        wire_modules: Sequence[ModuleType] | None = options.wire_modules
        wire_packages: Sequence[str] | None = options.wire_packages
        wire_classes: Sequence[type] | None = options.wire_classes
        if wire_modules or wire_packages or wire_classes:
            runtime_container.wire_modules(
                modules=wire_modules, packages=wire_packages, classes=wire_classes
            )
        runtime = m.ServiceRuntime.model_construct(
            config=runtime_config, context=runtime_context, container=runtime_container
        )
        self._runtime = runtime
        return runtime

    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic container registration."""
        effective_service_name: str = (
            service_name
            if service_name is not None and u.is_type(service_name, str)
            else self.__class__.__name__
        )
        register_result = self._register_in_container(effective_service_name)
        if register_result.is_failure:
            error_msg = register_result.error
            if error_msg is None:
                error_msg = "Service registration failed"
            if "already registered" not in error_msg.lower():
                self.logger.warning(
                    f"Service registration failed: {register_result.error}",
                    service_name=effective_service_name,
                )

    def _log_config_once(
        self, config: m.ConfigMap, message: str = "Configuration loaded"
    ) -> None:
        """Log configuration ONCE without binding to context."""
        config_typed: m.ConfigMap = m.ConfigMap(root=dict(config.items()))
        self.logger.info(message, **config_typed.root)

    def _log_with_context(
        self, level: str, message: str, **extra: t.MetadataValue
    ) -> None:
        """Log message with automatic context data inclusion."""
        correlation_id = FlextContext.Correlation.get_correlation_id()
        operation_name = FlextContext.Request.get_operation_name()
        context_data: m.ConfigMap = m.ConfigMap(
            root={
                "correlation_id": FlextRuntime.normalize_to_general_value(
                    correlation_id or ""
                ),
                "operation": FlextRuntime.normalize_to_general_value(
                    operation_name or ""
                ),
                **{
                    k: FlextRuntime.normalize_to_general_value(v)
                    for k, v in extra.items()
                },
            }
        )
        log_method = (
            getattr(self.logger, level)
            if hasattr(self.logger, level)
            else self.logger.info
        )
        _ = log_method(message, **context_data.root)

    def _register_in_container(self, service_name: str) -> r[bool]:
        """Register self in global container for service discovery."""

        def register() -> bool:
            """Register service in container."""
            container = self.container
            was_registered = container.has_service(service_name)
            if isinstance(self, BaseModel):
                _ = container.register(service_name, self)
            else:
                _ = container.register(service_name, service_name)
            if was_registered:
                return True
            if container.has_service(service_name):
                return True
            msg = "Service registration failed"
            raise RuntimeError(msg)

        try:
            result = register()
            return r[bool].ok(result)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            return r[bool].fail(str(e))

    class CQRS:
        """CQRS utilities for handlers."""

        class MetricsTracker:
            """Tracks handler execution metrics."""

            _metrics: ClassVar[MutableMapping[str, t.ContainerValue]] = {}

            def __init__(
                self, *args: t.ContainerValue, **kwargs: t.ContainerValue
            ) -> None:
                """Initialize metrics tracker with empty metrics dict."""
                super().__init__(*args, **kwargs)
                vars(self)["_metrics"] = {}

            def get_metrics(self) -> r[m.ConfigMap]:
                """Get current metrics dictionary.

                Returns:
                    r[m.ConfigMap]: Success result with metrics collection

                """
                if not hasattr(self, "_metrics"):
                    vars(self)["_metrics"] = {}
                return r[m.ConfigMap].ok(m.ConfigMap(root=dict(self._metrics.items())))

            def record_metric(self, name: str, value: t.ContainerValue) -> r[bool]:
                """Record a metric value.

                Args:
                    name: Metric name
                    value: Metric value to record

                Returns:
                    r[bool]: Success result

                """
                if not hasattr(self, "_metrics"):
                    vars(self)["_metrics"] = {}
                self._metrics[name] = value
                return r[bool].ok(value=True)

        class ContextStack:
            """Manages execution context stack."""

            _stack: ClassVar[
                list[m.ExecutionContext | m.ConfigMap | dict[str, t.ContainerValue]]
            ] = []

            def __init__(
                self, *args: t.ContainerValue, **kwargs: t.ContainerValue
            ) -> None:
                """Initialize context stack with empty list."""
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "_stack", [])

            def current_context(self) -> t.ContainerValue | None:
                """Get current execution context without popping.

                Returns:
                    m.ExecutionContext | None: Current context or None if stack is empty

                """
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if self._stack:
                    top_item = self._stack[-1]
                    match top_item:
                        case m.ExecutionContext() as execution_ctx:
                            return execution_ctx
                        case _:
                            return None
                return None

            def pop_context(self) -> r[Mapping[str, t.ContainerValue]]:
                """Pop execution context from the stack.

                Returns:
                    r[m.ConfigMap]: Success result with popped context or empty dict

                """
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if self._stack:
                    popped = self._stack.pop()
                    match popped:
                        case m.ExecutionContext() as execution_ctx:
                            context_dict: m.ConfigMap = m.ConfigMap(
                                root={
                                    "handler_name": execution_ctx.handler_name,
                                    "handler_mode": execution_ctx.handler_mode,
                                }
                            )
                            return r[dict[str, t.ContainerValue]].ok(context_dict.root)
                        case m.ConfigMap() as popped_dict:
                            return r[dict[str, t.ContainerValue]].ok(popped_dict.root)
                        case dict() as popped_plain:
                            return r[dict[str, t.ContainerValue]].ok(dict(popped_plain))
                return r[dict[str, t.ContainerValue]].ok({})

            def push_context(self, ctx: t.ContainerValue) -> r[bool]:
                """Push execution context onto the stack.

                Args:
                    ctx: Execution context or context dict to push onto the stack

                Returns:
                    r[bool]: Success result

                """
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if isinstance(ctx, m.ExecutionContext):
                    self._stack.append(ctx)
                    return r[bool].ok(value=True)
                if not isinstance(ctx, Mapping):
                    return r[bool].fail("Unsupported context type for push_context")
                ctx_mapping = ctx
                handler_name_raw = ctx_mapping.get("handler_name", "unknown")
                handler_name: str = (
                    str(handler_name_raw) if handler_name_raw is not None else "unknown"
                )
                handler_mode_raw = ctx_mapping.get("handler_mode", "operation")
                handler_mode_str: str = (
                    str(handler_mode_raw)
                    if handler_mode_raw is not None
                    else "operation"
                )
                handler_mode_literal: c.Cqrs.HandlerTypeLiteral = (
                    "command"
                    if handler_mode_str == "command"
                    else "query"
                    if handler_mode_str == "query"
                    else "event"
                    if handler_mode_str == "event"
                    else "saga"
                    if handler_mode_str == "saga"
                    else "operation"
                )
                execution_ctx = m.ExecutionContext.create_for_handler(
                    handler_name=handler_name, handler_mode=handler_mode_literal
                )
                self._stack.append(execution_ctx)
                return r[bool].ok(value=True)

    class Validation:
        """Railway-oriented validation patterns with r composition."""

        @staticmethod
        def validate_with_result(
            data: t.ContainerValue,
            validators: list[Callable[[t.ContainerValue], r[bool]]],
        ) -> r[t.ContainerValue]:
            """Chain validators sequentially, returning first failure or data on success."""
            result: r[t.ContainerValue] = r[t.ContainerValue].ok(data)
            for validator in validators:

                def validate_and_preserve(
                    data: t.ContainerValue, v: Callable[[t.ContainerValue], r[bool]]
                ) -> r[t.ContainerValue]:
                    validation_result = v(data)
                    if validation_result.is_failure:
                        base_msg = "Validation failed"
                        error_msg = (
                            f"{base_msg}: {validation_result.error}"
                            if validation_result.error
                            else f"{base_msg} (validation rule failed)"
                        )
                        fail_error_data: t.ConfigurationMapping = (
                            dict(validation_result.error_data.root)
                            if validation_result.error_data is not None
                            else {}
                        )
                        return r[t.ContainerValue].fail(
                            error_msg,
                            error_code=validation_result.error_code,
                            error_data=fail_error_data,
                        )
                    if validation_result.value is not True:
                        return r[t.ContainerValue].fail(
                            f"Validator must return r[bool].ok(True) for success, got {validation_result.value!r}"
                        )
                    return r[t.ContainerValue].ok(data)

                if result.is_success:
                    result = validate_and_preserve(result.value, validator)
            return result

    class ProtocolValidation:
        """Runtime protocol compliance validation utilities."""

        @staticmethod
        def is_command_bus(obj: object) -> bool:
            """Check if *obj* satisfies ``p.CommandBus`` structurally."""
            return (
                hasattr(obj, "dispatch")
                and callable(getattr(obj, "dispatch", None))
                and hasattr(obj, "publish")
                and callable(getattr(obj, "publish", None))
                and hasattr(obj, "register_handler")
                and callable(getattr(obj, "register_handler", None))
            )

        @staticmethod
        def is_handler(obj: object) -> bool:
            """Check if *obj* satisfies ``p.Handler`` structurally."""
            return (
                hasattr(obj, "handle")
                and callable(getattr(obj, "handle", None))
                and hasattr(obj, "validate")
                and callable(getattr(obj, "validate", None))
            )

        @staticmethod
        def is_service(obj: object) -> bool:
            """Check if *obj* satisfies ``p.Service`` structurally."""
            return (
                hasattr(obj, "execute")
                and callable(getattr(obj, "execute", None))
                and hasattr(obj, "get_service_info")
                and callable(getattr(obj, "get_service_info", None))
                and hasattr(obj, "is_valid")
            )

        @staticmethod
        def validate_processor_protocol(obj: object) -> r[bool]:
            """Validate *obj* has ``model_dump``, ``process``, and ``validate``."""
            required_methods = ["model_dump", "process", "validate"]
            for method_name in required_methods:
                if not hasattr(obj, method_name):
                    methods_str = ", ".join(required_methods)
                    error_msg = f"Processor {obj.__class__.__name__} missing required method '{method_name}()'. Processors must implement: {methods_str}"
                    return r[bool].fail(error_msg)
                if not callable(getattr(obj, method_name, None)):
                    return r[bool].fail(
                        f"Processor {obj.__class__.__name__}.{method_name} is not callable"
                    )
            return r[bool].ok(value=True)

        @staticmethod
        def validate_protocol_compliance(obj: object, protocol_name: str) -> r[bool]:
            """Validate *obj* compliance with named protocol via duck-typing."""
            protocol_required_attrs: Mapping[str, Sequence[str]] = {
                "Handler": ["handle", "can_handle"],
                "Service": ["execute", "get_service_info", "is_valid"],
                "CommandBus": ["dispatch", "publish", "register_handler"],
                "Repository": ["get_by_id", "save", "delete", "find_all"],
                "Configurable": ["configure"],
            }
            if protocol_name not in protocol_required_attrs:
                supported = ", ".join(protocol_required_attrs.keys())
                return r[bool].fail(
                    f"Unknown protocol: {protocol_name}. Supported: {supported}"
                )
            for attr in protocol_required_attrs[protocol_name]:
                if not hasattr(obj, attr):
                    return r[bool].fail(
                        f"{obj.__class__.__name__} missing '{attr}' required by {protocol_name}"
                    )
            return r[bool].ok(value=True)


x = FlextMixins
__all__ = ["FlextMixins", "x"]
