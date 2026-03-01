"""Reusable mixins for service infrastructure.

Provide shared behaviors for services and handlers including structured
logging, DI-backed context handling, and operation tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, suppress
from functools import partial
from types import ModuleType
from typing import ClassVar, cast

from pydantic import BaseModel, PrivateAttr

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextLogger,
    FlextResult,
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

_module_logger = logging.getLogger(__name__)


class FlextMixins(FlextRuntime):
    """Composable behaviors for dispatcher-driven services and handlers.

    These mixins centralize DI container access, structured logging, and
    context management so dispatcher-executed services can stay focused on
    domain work while still emitting `FlextResult` outcomes and metrics.

    Properties:
    - ``container``: Lazy ``FlextContainer`` singleton lookups for DI wiring.
    - ``logger``: Cached ``FlextLogger`` resolution for structured logs.
    - ``context``: Per-operation ``FlextContext`` for correlation metadata.
    - ``config``: Thread-safe ``FlextSettings`` access for runtime settings.

    Key methods:
    - ``track``: Context manager that records timing/err counts per operation.
    - ``_with_operation_context`` / ``_clear_operation_context``: Scoped
      context bindings used by dispatcher pipelines.
    - Delegated ``FlextRuntime``/``FlextResult`` helpers for railway flows.

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

    # Runtime helpers inherited from FlextRuntime via MRO; use runtime aliases (c, m, r, t, x, ...) at call sites; MRO protocol only.

    # =========================================================================
    # RESULT FACTORY UTILITIES (Delegated from FlextResult)
    # =========================================================================
    # All classes inheriting FlextMixins automatically have access to
    # r factory methods for railway-oriented programming

    # Factory methods - Use: self.ok(value) or self.fail("error")
    # These delegate to r for unified usage with proper type inference
    @staticmethod
    def ok[T](value: T) -> r[T]:
        """Create successful result wrapping value."""
        return r[T].ok(value)

    @staticmethod
    def fail(
        error: str | None,
        error_code: str | None = None,
        error_data: m.ConfigMap | None = None,
    ) -> r[t.ConfigMapValue]:
        """Create failed result with error message."""
        return r[t.ConfigMapValue].fail(
            error,
            error_code=error_code,
            error_data=error_data,
        )

    traverse = staticmethod(r.traverse)
    accumulate_errors = staticmethod(r.accumulate_errors)

    @staticmethod
    def to_dict(
        obj: BaseModel | Mapping[str, t.ConfigMapValue] | None,
    ) -> m.ConfigMap:
        """Convert BaseModel/dict to dict (None → empty dict). Use x.to_dict at call sites."""
        if obj is None:
            return m.ConfigMap(root={})

        if isinstance(obj, m.ConfigMap):
            return obj

        model_dump_callable = obj.model_dump if hasattr(obj, "model_dump") else None
        if callable(model_dump_callable):
            model_dump_result = model_dump_callable()
            try:
                normalized_model_dump: dict[str, t.ConfigMapValue] = {}
                if isinstance(model_dump_result, Mapping):
                    for key, value in model_dump_result.items():
                        normalized_value: t.ConfigMapValue = (
                            FlextRuntime.normalize_to_general_value(value)
                            if isinstance(
                                value,
                                str | int | float | bool | type(None) | BaseModel,
                            )
                            else str(value)
                        )
                        normalized_model_dump[str(key)] = normalized_value
                return m.ConfigMap(root=normalized_model_dump)
            except (TypeError, ValueError, AttributeError) as exc:
                _module_logger.debug(
                    "Model dump normalization fallback to string conversion",
                    exc_info=exc,
                )
                return m.ConfigMap(root={"value": str(model_dump_result)})

        try:
            if isinstance(obj, Mapping):
                normalized_mapping: dict[str, t.ConfigMapValue] = {}
                for key, value in obj.items():
                    normalized_mapping_value: t.ConfigMapValue = (
                        FlextRuntime.normalize_to_general_value(value)
                        if isinstance(
                            value,
                            str | int | float | bool | type(None) | BaseModel,
                        )
                        else str(value)
                    )
                    normalized_mapping[str(key)] = normalized_mapping_value
                return m.ConfigMap.model_validate(normalized_mapping)
            return m.ConfigMap.model_validate(obj)
        except (TypeError, ValueError, AttributeError) as exc:
            _module_logger.debug(
                "Object-to-config-map normalization failed",
                exc_info=exc,
            )
            return m.ConfigMap(root={})

    @staticmethod
    def ensure_result[T](value: T | r[T]) -> r[T]:
        """Wrap value in r if not already wrapped. Use x.ensure_result at call sites."""
        if isinstance(value, FlextResult):
            return value
        return r[T].ok(value)

    # =========================================================================
    # SERVICE INFRASTRUCTURE (Original FlextMixins functionality)
    # =========================================================================

    # Class-level cache for loggers to avoid repeated DI lookups
    # Uses FlextLogger directly for proper type checking (overloaded methods)
    _logger_cache: ClassVar[MutableMapping[str, FlextLogger]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(
        cls,
        **kwargs: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
    ) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)
        # Container is lazily initialized on first access

    @property
    def container(self) -> p.DI:
        """Get global FlextContainer instance with lazy initialization."""
        # _get_runtime().container returns p.DI from ServiceRuntime model
        return self._get_runtime().container

    @property
    def context(self) -> p.Context:
        """Get FlextContext instance for context operations."""
        return self._get_runtime().context

    @property
    def logger(self) -> FlextLogger:
        """Get FlextLogger instance (DI-backed with caching)."""
        return self._get_or_create_logger()

    @property
    def config(self) -> p.Config:
        """Return the runtime configuration associated with this component.

        Returns p.Config protocol type for type safety. In practice, this is
        FlextSettings - callers needing concrete attributes can use type
        identity or MRO checks for type narrowing.
        """
        return self._get_runtime().config

    @classmethod
    def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
        """Hook to customize runtime creation for mixin consumers.

        Returns:
            p.RuntimeBootstrapOptions: Pydantic model with optional runtime configuration options.

        """
        return FlextModelsService.RuntimeBootstrapOptions()

    def _get_runtime(self) -> m.ServiceRuntime:
        """Return or create a runtime triple shared across mixin consumers."""
        # Use getattr to safely access PrivateAttr before initialization
        # PrivateAttr may return the descriptor object if not initialized
        # Check if _runtime is actually a ServiceRuntime instance
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
                    options_candidate,
                )
            )
        except (TypeError, ValueError, AttributeError) as exc:
            _module_logger.debug(
                "Runtime bootstrap options validation failed",
                exc_info=exc,
            )
            options = FlextModelsService.RuntimeBootstrapOptions()

        config_type_raw = options.config_type
        config_cls_typed: type[FlextSettings]
        if config_type_raw is not None and issubclass(config_type_raw, FlextSettings):
            config_cls_typed = config_type_raw
        else:
            config_cls_typed = FlextSettings
        runtime_config = config_cls_typed.materialize(
            config_overrides=options.config_overrides,
        )

        runtime_context: p.Context = (
            options.context if options.context is not None else FlextContext.create()
        )

        # Use FlextSettings directly - it's guaranteed to be FlextSettings
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
                modules=wire_modules,
                packages=wire_packages,
                classes=wire_classes,
            )

        runtime = m.ServiceRuntime.model_construct(
            config=runtime_config,
            context=runtime_context,
            container=runtime_container,
        )
        self._runtime = runtime
        return runtime

    @contextmanager
    def track(
        self,
        operation_name: str,
    ) -> Iterator[Mapping[str, t.ConfigMapValue]]:
        """Track operation performance with timing and automatic context cleanup."""
        stats_attr = f"_stats_{operation_name}"
        stats: m.ConfigMap = (
            getattr(self, stats_attr)
            if hasattr(self, stats_attr)
            else m.ConfigMap(
                root={
                    "operation_count": 0,
                    "error_count": 0,
                    "total_duration_ms": 0.0,
                },
            )
        )

        op_count_raw = u.get(stats, "operation_count", default=0)
        stats["operation_count"] = int(op_count_raw) + 1

        try:
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                metrics_map: dict[str, t.ConfigMapValue] = (
                    dict(metrics.items()) if hasattr(metrics, "items") else {}
                )
                # Add aggregated stats to metrics for visibility
                metrics_map["operation_count"] = stats["operation_count"]
                try:
                    yield metrics_map
                    # Success - update stats
                    if "duration_ms" in metrics_map:
                        total_dur_raw = u.get(
                            stats,
                            "total_duration_ms",
                            default=0.0,
                        )
                        dur_ms_raw = u.get(
                            metrics_map,
                            "duration_ms",
                            default=0.0,
                        )
                        total_dur = float(total_dur_raw)
                        dur_ms = float(dur_ms_raw)
                        stats["total_duration_ms"] = total_dur + dur_ms
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    _module_logger.debug(
                        "Tracked operation raised expected exception",
                        exc_info=exc,
                    )
                    # Failure - increment error count
                    err_raw = u.get(stats, "error_count", default=0)
                    stats["error_count"] = int(err_raw) + 1
                    raise
                finally:
                    # Calculate success rate
                    op_raw = u.get(stats, "operation_count", default=1)
                    err_raw2 = u.get(stats, "error_count", default=0)
                    op_count = int(op_raw)
                    err_count = int(err_raw2)
                    stats["success_rate"] = (op_count - err_count) / op_count
                    if op_count > 0:
                        total_raw = u.get(
                            stats,
                            "total_duration_ms",
                            default=0.0,
                        )
                        total_dur_final = float(total_raw)
                        stats["avg_duration_ms"] = total_dur_final / op_count
                    # Update metrics with final stats
                    metrics_map["error_count"] = stats["error_count"]
                    metrics_map["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics_map["avg_duration_ms"] = stats["avg_duration_ms"]
                    # Store updated stats
                    setattr(self, stats_attr, stats)
        finally:
            # Auto-cleanup operation context
            FlextMixins._clear_operation_context()

    def _register_in_container(self, service_name: str) -> r[bool]:
        """Register self in global container for service discovery."""

        # Use r.create_from_callable() for unified error handling (DSL pattern)
        def register() -> bool:
            """Register service in container."""
            # Get container with explicit type for registration
            container = self.container
            if isinstance(self, BaseModel):
                result = container.register(service_name, self)
            else:
                result = container.register(service_name, service_name)
            # Use u.when() for conditional error handling (DSL pattern)
            if result.is_failure:
                error_msg = result.error or ""
                # Check if error is "already registered" (using native 'in' for string check)
                if "already registered" in error_msg.lower():
                    return True  # Already registered is success
                raise RuntimeError(error_msg or "Service registration failed")
            return True

        return r[bool].create_from_callable(register)

    @staticmethod
    def _propagate_context(operation_name: str) -> None:
        """Propagate context for current operation using FlextContext."""
        FlextContext.Request.set_operation_name(operation_name)
        _ = FlextContext.Utilities.ensure_correlation_id()

    @classmethod
    def _get_or_create_logger(cls) -> FlextLogger:
        """Get or create DI-injected logger with fallback to direct creation."""
        # Generate unique logger name based on module and class
        logger_name = f"{cls.__module__}.{cls.__name__}"

        # Check cache first (thread-safe)
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                # Cache stores FlextLogger - return directly
                return cls._logger_cache[logger_name]

        # Try to get from DI container
        try:
            container = FlextContainer.create()
            logger_key = f"logger:{logger_name}"

            # Attempt to retrieve logger from container
            logger_result = container.get_typed(logger_key, FlextLogger)

            if logger_result.is_success:
                # Use .value directly - FlextResult never returns None on success
                # Explicit annotation: get_typed returns r[FlextLogger] but Pyrefly tracks T generically
                logger: FlextLogger = cast("FlextLogger", logger_result.value)
                # Cache the result
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger
                # FlextLogger satisfies p.Log.StructlogLogger via structural typing
                return logger

            # Logger not in container - create and register
            logger = FlextLogger(logger_name)
            # Register concrete logger instance directly
            container_impl: p.DI = container
            with suppress(ValueError, TypeError):
                # Ignore if already registered (race condition)
                if hasattr(container_impl, "register_factory"):

                    def logger_factory() -> t.RegisterableService:
                        return {"logger": logger_name}

                    _ = container_impl.register_factory(
                        logger_key,
                        logger_factory,
                    )
                else:
                    _ = container_impl.register(logger_key, logger)

            # Cache the result
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger

            # FlextLogger satisfies p.Log.StructlogLogger via structural typing
            return logger

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            logger = FlextLogger(logger_name)
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger
            # FlextLogger satisfies p.Log.StructlogLogger via structural typing
            return logger

    def _log_with_context(
        self,
        level: str,
        message: str,
        **extra: t.MetadataAttributeValue,
    ) -> None:
        """Log message with automatic context data inclusion."""
        correlation_id = FlextContext.Correlation.get_correlation_id()
        operation_name = FlextContext.Request.get_operation_name()
        context_data: m.ConfigMap = m.ConfigMap(
            root={
                "correlation_id": FlextRuntime.normalize_to_general_value(
                    correlation_id,
                ),
                "operation": FlextRuntime.normalize_to_general_value(operation_name),
                **{
                    k: FlextRuntime.normalize_to_general_value(v)
                    for k, v in extra.items()
                },
            },
        )

        log_method = (
            getattr(self.logger, level)
            if hasattr(self.logger, level)
            else self.logger.info
        )
        _ = log_method(message, extra=context_data.root)

    # =========================================================================
    # SERVICE METHODS - Complete Infrastructure (inherited by x)
    # =========================================================================

    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic container registration."""
        # Fast fail: service_name must be str or None
        effective_service_name: str = (
            service_name
            if u.is_type(service_name, str) and service_name
            else self.__class__.__name__
        )

        register_result = self._register_in_container(effective_service_name)

        if register_result.is_failure:
            # Only log warning if it's not an "already registered" error
            # Fast fail: error must be str (r guarantees this)
            error_msg = register_result.error
            if error_msg is None:
                error_msg = "Service registration failed"
            if "already registered" not in error_msg.lower():
                self.logger.warning(
                    f"Service registration failed: {register_result.error}",
                    extra={"service_name": effective_service_name},
                )

    # =========================================================================
    # CONTEXT ENRICHMENT METHODS - Automatic Context Management
    # =========================================================================

    def _enrich_context(self, **context_data: t.ConfigMapValue) -> None:
        """Log service information ONCE at initialization (not bound to context)."""
        # Build service context for logging using correct types
        # Use dict for mutability
        service_context: m.ConfigMap = m.ConfigMap(
            root={
                "service_name": self.__class__.__name__,
                "service_module": self.__class__.__module__,
                **context_data,
            },
        )
        # Log service initialization ONCE instead of binding to all logs
        self.logger.info(
            "Service initialized",
            return_result=False,
            **service_context.root,
        )

    def _log_config_once(
        self,
        config: m.ConfigMap,
        message: str = "Configuration loaded",
    ) -> None:
        """Log configuration ONCE without binding to context."""
        # Convert config to t.ConfigMapValue for logging
        # ConfigurationMapping is Mapping[str, t.ConfigMapValue], convert to dict
        config_typed: m.ConfigMap = m.ConfigMap(root=dict(config.items()))
        # Log configuration as single event, not bound to context
        self.logger.info(message, config=config_typed.root)

    @staticmethod
    def _with_operation_context(
        operation_name: str,
        **operation_data: t.ConfigMapValue,
    ) -> None:
        """Set operation context with level-based binding (DEBUG/ERROR/normal)."""
        # Propagate context using inherited Context mixin method
        FlextMixins._propagate_context(operation_name)

        # Bind additional operation data with level filtering
        if operation_data:
            # Categorize data by log level
            # NOTE: 'config', 'configuration', 'settings' removed - use _log_config_once() instead
            debug_keys = {"schema", "params"}
            error_keys = {
                "stack_trace",
                "exception",
                "traceback",
                "error_details",
            }

            # Separate data by level - preserve t.ConfigMapValue from operation_data
            # Use dict for mutability
            debug_data: m.ConfigMap = m.ConfigMap(
                root={k: v for k, v in operation_data.items() if k in debug_keys},
            )
            error_data: m.ConfigMap = m.ConfigMap(
                root={k: v for k, v in operation_data.items() if k in error_keys},
            )
            normal_data: m.ConfigMap = m.ConfigMap(
                root={
                    k: v
                    for k, v in operation_data.items()
                    if k not in debug_keys and k not in error_keys
                },
            )

            # Bind context using bind_global_context - no level-specific binding available
            # Combine all context data for global binding
            all_context_data: m.ConfigMap = normal_data.model_copy()
            # Simple merge: deep strategy - new values override existing ones
            if debug_data:
                merged_debug: m.ConfigMap = all_context_data.model_copy()
                merged_debug.update(debug_data.root)
                all_context_data = merged_debug
            if error_data:
                merged_error: m.ConfigMap = all_context_data.model_copy()
                merged_error.update(error_data.root)
                all_context_data = merged_error
            if all_context_data:
                metadata_context: dict[str, t.MetadataAttributeValue] = {
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in all_context_data.root.items()
                }
                _ = FlextLogger.bind_global_context(**metadata_context)
            if normal_data:
                normal_metadata_context: dict[str, t.MetadataAttributeValue] = {
                    key: FlextRuntime.normalize_to_metadata_value(value)
                    for key, value in normal_data.root.items()
                }
                _ = FlextLogger.bind_context(
                    c.Context.SCOPE_OPERATION,
                    **normal_metadata_context,
                )

    @staticmethod
    def _clear_operation_context() -> None:
        """Clear operation scope context (preserves request/application scopes)."""
        # Clear operation scope only (preserves request and application scopes)
        _ = FlextLogger.clear_scope("operation")

        # Clear FlextContext operation name
        FlextContext.Request.set_operation_name("")

    class CQRS:
        """CQRS utilities for handlers."""

        class MetricsTracker:
            """Tracks handler execution metrics."""

            def _protocol_name(self) -> str:
                """Return protocol name for BaseProtocol compliance."""
                return "MetricsTracker"

            # Type annotation for type checker
            _metrics: ClassVar[MutableMapping[str, t.ConfigMapValue]] = {}

            def __init__(
                self,
                *args: t.ConfigMapValue,
                **kwargs: t.ConfigMapValue,
            ) -> None:
                """Initialize metrics tracker with empty metrics dict."""
                super().__init__(*args, **kwargs)
                # Initialize _metrics as instance attribute (not PrivateAttr for mixin compatibility)
                # Use vars() to bypass __setattr__ (works with Pydantic frozen models)
                vars(self)["_metrics"] = {}

            def record_metric(
                self,
                name: str,
                value: t.ConfigMapValue,
            ) -> r[bool]:
                """Record a metric value.

                Args:
                    name: Metric name
                    value: Metric value to record

                Returns:
                    r[bool]: Success result

                """
                # _metrics is initialized in __init__, but check for safety
                if not hasattr(self, "_metrics"):
                    vars(self)["_metrics"] = {}
                self._metrics[name] = value
                return r[bool].ok(value=True)

            def get_metrics(self) -> r[m.ConfigMap]:
                """Get current metrics dictionary.

                Returns:
                    r[m.ConfigMap]: Success result with metrics collection

                """
                # _metrics is initialized in __init__, but check for safety
                if not hasattr(self, "_metrics"):
                    vars(self)["_metrics"] = {}
                return r[m.ConfigMap].ok(m.ConfigMap(root=dict(self._metrics.items())))

        class ContextStack:
            """Manages execution context stack."""

            # Type annotation for type checker
            _stack: ClassVar[
                list[
                    m.Handler.ExecutionContext
                    | m.ConfigMap
                    | dict[str, t.ConfigMapValue]
                ]
            ] = []

            def __init__(
                self,
                *args: t.ConfigMapValue,
                **kwargs: t.ConfigMapValue,
            ) -> None:
                """Initialize context stack with empty list."""
                super().__init__(*args, **kwargs)
                # Initialize _stack as instance attribute (not PrivateAttr for mixin compatibility)
                object.__setattr__(self, "_stack", [])

            def _protocol_name(self) -> str:
                """Return the protocol name for introspection."""
                return "ContextStack"

            def push_context(
                self,
                ctx: t.GuardInputValue,
            ) -> r[bool]:
                """Push execution context onto the stack.

                Args:
                    ctx: Execution context or context dict to push onto the stack

                Returns:
                    r[bool]: Success result

                """
                # _stack is initialized in __init__, but check for safety
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if isinstance(ctx, m.Handler.ExecutionContext):
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
                execution_ctx = m.Handler.ExecutionContext.create_for_handler(
                    handler_name=handler_name,
                    handler_mode=handler_mode_literal,
                )
                self._stack.append(execution_ctx)
                return r[bool].ok(value=True)

            def pop_context(self) -> r[Mapping[str, t.ConfigMapValue]]:
                """Pop execution context from the stack.

                Returns:
                    r[m.ConfigMap]: Success result with popped context or empty dict

                """
                # _stack is initialized in __init__, but check for safety
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if self._stack:
                    popped = self._stack.pop()
                    match popped:
                        case m.Handler.ExecutionContext() as execution_ctx:
                            context_dict: m.ConfigMap = m.ConfigMap(
                                root={
                                    "handler_name": execution_ctx.handler_name,
                                    "handler_mode": execution_ctx.handler_mode,
                                },
                            )
                            return r[dict[str, t.ConfigMapValue]].ok(context_dict.root)
                        case m.ConfigMap() as popped_dict:
                            return r[dict[str, t.ConfigMapValue]].ok(popped_dict.root)
                        case dict() as popped_plain:
                            return r[dict[str, t.ConfigMapValue]].ok(
                                dict(popped_plain),
                            )
                return r[dict[str, t.ConfigMapValue]].ok({})

            def current_context(self) -> t.GuardInputValue | None:
                """Get current execution context without popping.

                Returns:
                    m.ExecutionContext | None: Current context or None if stack is empty

                """
                # _stack is initialized in __init__, but check for safety
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if self._stack:
                    top_item = self._stack[-1]
                    match top_item:
                        case m.Handler.ExecutionContext() as execution_ctx:
                            return execution_ctx
                        case _:
                            return None
                return None

    class Validation:
        """Railway-oriented validation patterns with r composition."""

        @staticmethod
        def validate_with_result(
            data: t.ConfigMapValue,
            validators: list[Callable[[t.ConfigMapValue], r[bool]]],
        ) -> r[t.ConfigMapValue]:
            """Chain validators sequentially, returning first failure or data on success."""
            result: r[t.ConfigMapValue] = r[t.ConfigMapValue].ok(data)

            for validator in validators:
                # Create helper function with proper closure to validate and preserve data
                def validate_and_preserve(
                    data: t.ConfigMapValue,
                    v: Callable[[t.ConfigMapValue], r[bool]],
                ) -> r[t.ConfigMapValue]:
                    validation_result = v(data)
                    if validation_result.is_failure:
                        base_msg = "Validation failed"
                        error_msg = (
                            f"{base_msg}: {validation_result.error}"
                            if validation_result.error
                            else f"{base_msg} (validation rule failed)"
                        )
                        return r[t.ConfigMapValue].fail(
                            error_msg,
                            error_code=validation_result.error_code,
                            error_data=validation_result.error_data,
                        )
                    # Check that validation returned True
                    if validation_result.value is not True:
                        return r[t.ConfigMapValue].fail(
                            f"Validator must return r[bool].ok(True) for success, got {validation_result.value!r}",
                        )
                    return r[t.ConfigMapValue].ok(data)

                # Use partial to bind validator while passing data through flat_map
                result = result.flat_map(partial(validate_and_preserve, v=validator))

            return result

    class ProtocolValidation:
        """Runtime protocol compliance validation utilities."""

        @staticmethod
        def is_handler(
            obj: t.ConfigMapValue,
        ) -> bool:
            """Check if object satisfies p.Handler protocol."""
            return (
                hasattr(obj, "handle")
                and callable(getattr(obj, "handle", None))
                and hasattr(obj, "validate")
                and callable(getattr(obj, "validate", None))
            )

        @staticmethod
        def is_service(
            _obj: p.Service[t.ConfigMapValue],
        ) -> bool:
            """Check if object satisfies p.Service protocol.

            Uses structural typing - any object implementing Service protocol
            will pass this check, including FlextService instances.
            """
            return True

        @staticmethod
        def is_command_bus() -> bool:
            """Check if object satisfies p.CommandBus protocol."""
            return True

        @staticmethod
        def validate_protocol_compliance(
            _obj: p.Handler[t.GuardInputValue, t.GuardInputValue]
            | p.Service[t.ConfigMapValue]
            | p.CommandBus
            | p.Repository[t.ConfigMapValue]
            | p.Configurable,
            protocol_name: str,
        ) -> r[bool]:
            """Validate object compliance with named protocol."""
            protocol_map = {
                "Handler": p.Handler,
                "Service": p.Service,
                "CommandBus": p.CommandBus,
                "Repository": p.Repository,
                "Configurable": p.Configurable,
            }

            # Check if protocol_name is in protocol_map (using native 'not in')
            if protocol_name not in protocol_map:
                supported = ", ".join(protocol_map.keys())
                return r[bool].fail(
                    f"Unknown protocol: {protocol_name}. Supported: {supported}",
                )

            # Type already guarantees protocol compliance
            return r[bool].ok(value=True)

        @staticmethod
        def validate_processor_protocol(
            obj: p.HasModelDump,
        ) -> r[bool]:
            """Validate object has required process() and validate() methods."""
            required_methods = ["process", "validate"]

            for method_name in required_methods:
                if not hasattr(obj, method_name):
                    methods_str = ", ".join(required_methods)
                    error_msg = (
                        f"Processor {obj.__class__.__name__} missing required "
                        f"method '{method_name}()'. "
                        f"Processors must implement: {methods_str}"
                    )
                    return r[bool].fail(error_msg)
                if not callable(u.get(obj, method_name, default=None)):
                    return r[bool].fail(
                        f"Processor {obj.__class__.__name__}.{method_name} is not callable",
                    )

            return r[bool].ok(value=True)


x = FlextMixins

__all__ = ["FlextMixins", "x"]
