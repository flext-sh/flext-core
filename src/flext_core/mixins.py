"""Reusable mixins for service infrastructure.

Provide shared behaviors for services and handlers including structured
logging, DI-backed context handling, and operation tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import Annotated, ClassVar, TypeGuard, Unpack, override

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

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


class FlextMixins(m.ArbitraryTypesModel, FlextRuntime):
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
                    return u.ok({"status": "processed"})

    """

    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)

    _logger_cache: ClassVar[dict[str, FlextLogger]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
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
        logger = FlextLogger(logger_name)
        with cls._cache_lock:
            cls._logger_cache[logger_name] = logger
        return logger

    config_type: Annotated[
        type[FlextSettings] | None,
        Field(
            default=None,
            description="Configuration class to initialize the service.",
        ),
    ]
    config_overrides: Annotated[
        dict[str, object] | None,
        Field(
            default=None,
            description="Configuration overrides applied at instantiation.",
        ),
    ]
    initial_context: Annotated[
        FlextContext | None,
        Field(default=None, description="Initial FlextContext for the service scope."),
    ]

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
        operation_name: str, **operation_data: t.Scalar
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
                metadata_context: dict[str, t.Container | BaseModel] = {
                    key: FlextRuntime.normalize_to_container(value)
                    for key, value in all_context_data.root.items()
                }
                _ = FlextLogger.bind_global_context(**metadata_context)
            if normal_data:
                normal_metadata_context: dict[str, t.Container | BaseModel] = {
                    key: FlextRuntime.normalize_to_container(value)
                    for key, value in normal_data.root.items()
                }
                _ = FlextLogger.bind_context(
                    c.Context.SCOPE_OPERATION, **normal_metadata_context
                )

    @staticmethod
    def _normalize_log_payload(
        payload: Mapping[str, object],
    ) -> dict[str, t.Container | BaseModel]:
        normalized_payload: dict[str, t.Container | BaseModel] = {}
        for key, value in payload.items():
            normalized_payload[str(key)] = FlextRuntime.normalize_to_container(value)
        return normalized_payload

    @contextmanager
    def track(self, operation_name: str) -> Iterator[Mapping[str, object]]:
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
                metrics_map: dict[str, object] = (
                    {
                        str(k): FlextRuntime.normalize_to_container(v)
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

    def _enrich_context(self, **context_data: t.Scalar) -> None:
        """Log service information ONCE at initialization (not bound to context)."""
        service_context: m.ConfigMap = m.ConfigMap(
            root={
                "service_name": self.__class__.__name__,
                "service_module": self.__class__.__module__,
                **context_data,
            }
        )
        self.logger.info(
            "Service initialized",
            return_result=False,
            **FlextMixins._normalize_log_payload(service_context.root),
        )

    def _get_runtime(self) -> m.ServiceRuntime:
        """Return or create a runtime triple shared across mixin consumers."""
        runtime = self._runtime if hasattr(self, "_runtime") else None
        match runtime:
            case m.ServiceRuntime() as service_runtime:
                return service_runtime
            case _:
                pass
        config_type_raw = getattr(self, "config_type", None)
        config_cls_typed: type[FlextSettings]

        overrides: Mapping[str, object] | None = None
        initial_ctx: p.Context | None = None
        bootstrap_services: Mapping[str, t.RegisterableService] | None = None
        bootstrap_factories: Mapping[str, t.FactoryCallable] | None = None
        bootstrap_resources: Mapping[str, t.ResourceCallable] | None = None
        bootstrap_wire_modules: Sequence[ModuleType] | None = None
        bootstrap_wire_packages: Sequence[str] | None = None
        bootstrap_wire_classes: Sequence[type[object]] | None = None

        if hasattr(self, "_runtime_bootstrap_options"):
            bootstrap_method = getattr(self, "_runtime_bootstrap_options")
            if callable(bootstrap_method):
                try:
                    options_raw = bootstrap_method()
                    options: m.RuntimeBootstrapOptions | None = None
                    if isinstance(options_raw, m.RuntimeBootstrapOptions):
                        options = options_raw
                    elif isinstance(options_raw, (dict, Mapping)):
                        options = m.RuntimeBootstrapOptions.model_validate(options_raw)
                    elif options_raw is not None:
                        # Handle duck-typed objects (e.g. SimpleNamespace)
                        try:
                            options = m.RuntimeBootstrapOptions.model_validate(
                                options_raw, from_attributes=True
                            )
                        except ValidationError:
                            options = None

                    if options:
                        if options.config_type is not None:
                            config_type_raw = options.config_type
                        if options.config_overrides is not None:
                            overrides = options.config_overrides
                        if options.context is not None:
                            initial_ctx = options.context
                        if options.services is not None:
                            services_typed: dict[str, t.RegisterableService] = {}
                            for key, value in options.services.items():
                                if u.is_registerable_service(value):
                                    services_typed[str(key)] = value
                            bootstrap_services = services_typed
                        bootstrap_factories = options.factories
                        bootstrap_resources = options.resources
                        if options.wire_modules is not None:
                            modules_list: list[ModuleType] = []
                            packages_list: list[str] = []
                            for item in options.wire_modules:
                                if isinstance(item, str):
                                    packages_list.append(item)
                                else:
                                    modules_list.append(item)
                            bootstrap_wire_modules = modules_list
                            if packages_list:
                                bootstrap_wire_packages = packages_list
                        if options.wire_packages is not None:
                            current_packages = list(bootstrap_wire_packages or [])
                            current_packages.extend(options.wire_packages)
                            bootstrap_wire_packages = current_packages
                        if options.wire_classes is not None:
                            bootstrap_wire_classes = options.wire_classes
                except Exception as exc:
                    FlextLogger.create_module_logger(__name__).warning(
                        "Failed to load runtime bootstrap options",
                        exc_info=exc,
                    )

        if FlextMixins._is_flext_settings_type(config_type_raw):
            config_cls_typed = config_type_raw
        else:
            config_cls_typed = FlextSettings

        if overrides is None:
            overrides = getattr(self, "config_overrides", None)

        overrides_typed: Mapping[str, t.Scalar] | None = None
        if overrides is not None:
            overrides_typed = {
                key: value for key, value in overrides.items() if u.is_scalar(value)
            }

        runtime_config = config_cls_typed.get_global(overrides=overrides_typed)

        if initial_ctx is None:
            initial_ctx = getattr(self, "initial_context", None)

        runtime_context: p.Context = (
            initial_ctx if isinstance(initial_ctx, p.Context) else FlextContext.create()
        )
        runtime_config_typed: FlextSettings = runtime_config
        runtime_container = FlextContainer.create().scoped(
            config=runtime_config_typed,
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
            config=runtime_config_typed,
            context=runtime_context,
        )
        return self._runtime

    @staticmethod
    def _is_flext_settings_type(candidate: object) -> TypeGuard[type[FlextSettings]]:
        return isinstance(candidate, type) and callable(
            getattr(candidate, "get_global", None)
        )

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
        self.logger.info(
            message,
            **FlextMixins._normalize_log_payload(config_typed.root),
        )

    def _log_with_context(self, level: str, message: str, **extra: t.Scalar) -> None:
        """Log message with automatic context data inclusion."""
        correlation_id = FlextContext.Correlation.get_correlation_id()
        operation_name = FlextContext.Request.get_operation_name()
        context_data: m.ConfigMap = m.ConfigMap(
            root={
                "correlation_id": FlextRuntime.normalize_to_container(
                    correlation_id or ""
                ),
                "operation": FlextRuntime.normalize_to_container(operation_name or ""),
                **{k: FlextRuntime.normalize_to_container(v) for k, v in extra.items()},
            }
        )
        log_method = (
            getattr(self.logger, level)
            if hasattr(self.logger, level)
            else self.logger.info
        )
        _ = log_method(
            message,
            **FlextMixins._normalize_log_payload(context_data.root),
        )

    def _register_in_container(self, service_name: str) -> r[bool]:
        """Register self in global container for service discovery."""
        try:
            container = self.container
            was_registered = container.has_service(service_name)
            _ = container.register(service_name, self)
            if was_registered or container.has_service(service_name):
                return r[bool].ok(True)
            msg = "Service registration failed"
            raise RuntimeError(msg)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            return r[bool].fail(str(e))

    class CQRS:
        """CQRS utilities for handlers."""

        class MetricsTracker:
            """Tracks handler execution metrics."""

            _metrics: ClassVar[dict[str, object]] = {}

            def __init__(self, *args: object, **kwargs: t.Scalar) -> None:
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

            def record_metric(self, name: str, value: object) -> r[bool]:
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
                list[m.ExecutionContext | m.ConfigMap | dict[str, object]]
            ] = []

            def __init__(self, *args: object, **kwargs: t.Scalar) -> None:
                """Initialize context stack with empty list."""
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "_stack", [])

            def current_context(self) -> object | None:
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

            def pop_context(self) -> r[Mapping[str, object]]:
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
                            return r[dict[str, object]].ok(context_dict.root)
                        case m.ConfigMap() as popped_dict:
                            return r[dict[str, object]].ok(popped_dict.root)
                        case dict() as popped_plain:
                            return r[dict[str, object]].ok(dict(popped_plain))
                return r[dict[str, object]].ok({})

            def push_context(self, ctx: object) -> r[bool]:
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
                if not u.is_mapping(ctx):
                    return r[bool].fail("Unsupported context type for push_context")
                ctx_mapping: dict[str, object] = {}
                for key, value in ctx.items():
                    ctx_mapping[str(key)] = value
                handler_name_raw: object = ctx_mapping.get("handler_name", "unknown")
                handler_name: str = (
                    str(handler_name_raw) if handler_name_raw is not None else "unknown"
                )
                handler_mode_raw: object = ctx_mapping.get("handler_mode", "operation")
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
        @staticmethod
        def validate_with_result[TValidation](
            data: TValidation,
            validators: Sequence[Callable[[TValidation], r[bool]]],
        ) -> r[TValidation]:
            """Chain validators sequentially, returning first failure or data on success."""
            result: r[TValidation] = r[TValidation].ok(data)
            for validator in validators:
                if result.is_success:
                    current_data = result.unwrap()
                    validation_result = validator(current_data)
                    if validation_result.is_failure:
                        base_msg = "Validation failed"
                        error_msg = (
                            f"{base_msg}: {validation_result.error}"
                            if validation_result.error
                            else f"{base_msg} (validation rule failed)"
                        )
                        fail_error_data: Mapping[str, t.Container] | None = {}
                        if validation_result.error_data is not None:
                            normalized_error_data: dict[str, t.Container] = {}
                            for key, value in validation_result.error_data.root.items():
                                normalized = FlextRuntime.normalize_to_container(value)
                                if isinstance(normalized, BaseModel):
                                    normalized_error_data[str(key)] = str(normalized)
                                else:
                                    normalized_error_data[str(key)] = normalized
                            fail_error_data = normalized_error_data
                        return r[TValidation].fail(
                            error_msg,
                            error_code=validation_result.error_code,
                            error_data=fail_error_data,
                        )
                    if validation_result.value is not True:
                        return r[TValidation].fail(
                            f"Validator must return r[bool].ok(True) for success, got {validation_result.value!r}"
                        )
                    result = r[TValidation].ok(current_data)
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

    @classmethod
    def fail(cls, error: str, *, error_code: str | None = None) -> r[bool]:
        """Create a failure result — convenience shortcut for mixin consumers.

        Delegates to ``r[bool].fail`` so callers can write ``x.fail("msg")``
        without importing ``r`` separately.

        Args:
            error: Human-readable error description.
            error_code: Optional machine-readable error code for routing.

        Returns:
            r[bool]: Failure result carrying the given error message.

        """
        return r[bool].fail(error, error_code=error_code)


x = FlextMixins
__all__ = ["FlextMixins", "x"]
