"""Dispatcher-aware mixins for reusable service infrastructure.

Provide shared behaviors for services and handlers that rely on dispatcher-
first CQRS execution, structured logging, and DI-backed context handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime
from functools import partial
from typing import ClassVar, cast

from pydantic import BaseModel, PrivateAttr

from flext_core._models.handler import FlextModelsHandler
from flext_core.constants import c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.settings import FlextSettings
from flext_core.typings import t
from flext_core.utilities import u

# =============================================================================
# DRY HELPER FUNCTIONS (Python 3.13 PEP 695)
# =============================================================================


def require_initialized[T](value: T | None, name: str) -> T:
    """Guard function that raises RuntimeError if value is None.

    Eliminates repetitive null-check boilerplate across service classes.

    Args:
        value: The value to check (may be None).
        name: Human-readable name for error messages.

    Returns:
        The value if not None.

    Raises:
        RuntimeError: If value is None.

    Example:
        @property
        def context(self) -> p.Ctx:
            return require_initialized(self._context, "Context")

    """
    if value is None:
        msg = f"{name} not initialized"
        raise RuntimeError(msg)
    return value


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
                self, data: t.ContextMetadataMapping
            ) -> r[t.ContextMetadataMapping]:
                with self.track("process"):
                    self.logger.info("Processing", size=len(data))
                    return self.ok({"status": "processed"})

    """

    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)

    # =========================================================================
    # RUNTIME VALIDATION UTILITIES (Delegated from FlextRuntime)
    # =========================================================================
    # All classes inheriting FlextMixins automatically have access to
    # runtime validation utilities without explicit FlextRuntime import

    # Type guard utilities - access via FlextRuntime directly
    # These are inherited from FlextRuntime, no need to redeclare

    # Type introspection utilities
    is_sequence_type = staticmethod(FlextRuntime.is_sequence_type)
    safe_get_attribute = staticmethod(FlextRuntime.safe_get_attribute)
    extract_generic_args = staticmethod(FlextRuntime.extract_generic_args)

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
        error_data: t.ConfigurationMapping | None = None,
    ) -> r[t.GeneralValueType]:
        """Create failed result with error message.

        Returns a failed result with the error message. The type parameter
        is t.GeneralValueType to allow flexible error handling.
        """
        return r[t.GeneralValueType].fail(
            error,
            error_code=error_code,
            error_data=error_data,
        )

    traverse = staticmethod(r.traverse)
    parallel_map = staticmethod(r.parallel_map)
    accumulate_errors = staticmethod(r.accumulate_errors)

    # =========================================================================
    # MODEL CONVERSION UTILITIES (New in Phase 0 - Consolidation)
    # =========================================================================

    class ModelConversion:
        """BaseModel/dict conversion utilities (eliminates 32+ repetitive patterns)."""

        @staticmethod
        def to_dict(
            obj: (BaseModel | t.ContextMetadataMapping | t.ConfigurationMapping | None),
        ) -> t.ContextMetadataMapping:
            """Convert BaseModel/dict to dict (None → empty dict).

            Accepts BaseModel, dict with nested structures, or None.
            Nested Mapping/Sequence are preserved as-is in the output.
            """
            if obj is None:
                return {}
            if isinstance(obj, BaseModel):
                # BaseModel.model_dump() returns t.ConfigurationDict, normalize to t.GeneralValueType
                dumped = obj.model_dump()
                # Recursively normalize to ensure t.GeneralValueType compliance
                normalized = FlextRuntime.normalize_to_general_value(dumped)
                # Use isinstance for type checking - returns dict or None
                dict_result = normalized if u.is_type(normalized, dict) else None
                if dict_result is not None:
                    # Type narrowing: dict is a subtype of Mapping[str, t.GeneralValueType]
                    # ConfigurationDict (dict[str, t.GeneralValueType]) is compatible with ContextMetadataMapping
                    return cast("t.ContextMetadataMapping", dict_result)
                # Fallback: wrap scalar in dict (shouldn't happen for BaseModel.dump())
                return cast("t.ContextMetadataMapping", {"value": normalized})
            # For Mapping, use Collection.process() to normalize each value
            process_result = u.Collection.process(
                obj,
                lambda _k, v: FlextRuntime.normalize_to_general_value(v),
                on_error="skip",
            )
            if process_result.is_success:
                # Type narrowing: ConfigurationDict is dict[str, t.GeneralValueType]
                # ContextMetadataMapping is Mapping[str, t.GeneralValueType]
                # Since dict is a subtype of Mapping, ConfigurationDict is compatible with ContextMetadataMapping
                result_value = process_result.value
                if isinstance(result_value, dict):
                    return cast("t.ContextMetadataMapping", result_value)
            return cast("t.ContextMetadataMapping", {})

    # =========================================================================
    # RESULT HANDLING UTILITIES (New in Phase 0 - Consolidation)
    # =========================================================================

    class ResultHandling:
        """r wrapping utilities (eliminates 209+ repetitive patterns)."""

        @staticmethod
        def ensure_result[T](value: T | r[T]) -> r[T]:
            """Wrap value in r if not already wrapped."""
            # Check if value is already a r
            # Use type guard for proper type narrowing
            if isinstance(value, r):
                # Type narrowing: value is r[T] after isinstance check
                return value
            # Wrap non-result value in r.ok()
            # Type narrowing: value is T after isinstance check (not a Result)
            return r[T].ok(value)

    # =========================================================================
    # SERVICE INFRASTRUCTURE (Original FlextMixins functionality)
    # =========================================================================

    # Class-level cache for loggers to avoid repeated DI lookups
    _logger_cache: ClassVar[t.StringFlextLoggerDict] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(cls, **kwargs: t.GeneralValueType) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)
        # Container is lazily initialized on first access

    @property
    def container(self) -> p.DI:
        """Get global FlextContainer instance with lazy initialization."""
        # _get_runtime().container returns p.DI from ServiceRuntime model
        return self._get_runtime().container

    @property
    def context(self) -> p.Ctx:
        """Get FlextContext instance for context operations."""
        return self._get_runtime().context

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get FlextLogger instance (DI-backed with caching)."""
        return self._get_or_create_logger()

    @property
    def config(self) -> FlextSettings:
        """Return the runtime configuration associated with this component."""
        return cast("FlextSettings", self._get_runtime().config)

    @classmethod
    def _runtime_bootstrap_options(cls) -> t.RuntimeBootstrapOptions:
        """Hook to customize runtime creation for mixin consumers.

        Returns:
            RuntimeBootstrapOptions: TypedDict with optional runtime configuration options.

        """
        return {}

    def _get_runtime(self) -> m.ServiceRuntime:
        """Return or create a runtime triple shared across mixin consumers."""
        # Use getattr to safely access PrivateAttr before initialization
        # PrivateAttr may return the descriptor object if not initialized
        # Check if _runtime is actually a ServiceRuntime instance
        runtime = getattr(self, "_runtime", None)
        # Verify it's actually a ServiceRuntime, not the PrivateAttr descriptor
        if (
            runtime is not None
            and hasattr(runtime, "config")
            and hasattr(runtime, "container")
        ):
            return cast("m.ServiceRuntime", runtime)

        runtime_options_callable = getattr(self, "_runtime_bootstrap_options", None)
        # Call method and ensure result is t.RuntimeBootstrapOptions TypedDict
        # _runtime_bootstrap_options returns t.RuntimeBootstrapOptions per class definition
        options_raw: object = (
            runtime_options_callable() if callable(runtime_options_callable) else {}
        )
        # Type narrowing: ensure result is RuntimeBootstrapOptions TypedDict
        # Cast to t.RuntimeBootstrapOptions - object from callable is compatible
        options: t.RuntimeBootstrapOptions = cast(
            "t.RuntimeBootstrapOptions",
            options_raw if isinstance(options_raw, dict) else {},
        )
        # Use factory methods directly - Clean Architecture pattern
        # Each class knows how to instantiate itself
        config_type = (
            cast(
                "type[FlextSettings] | None",
                u.mapper().get(options, "config_type"),
            )
            if "config_type" in options
            else None
        )
        config_cls = config_type or FlextSettings
        config_overrides_raw = u.mapper().get(options, "config_overrides")
        config_overrides_typed: Mapping[str, t.FlexibleValue] | None = (
            cast("Mapping[str, t.FlexibleValue]", config_overrides_raw)
            if isinstance(config_overrides_raw, (dict, Mapping))
            else None
        )
        runtime_config = config_cls.materialize(
            config_overrides=config_overrides_typed,
        )

        context_option = (
            cast(
                "p.Ctx | None",
                u.mapper().get(options, "context"),
            )
            if "context" in options
            else None
        )
        runtime_context = (
            context_option if context_option is not None else FlextContext.create()
        )

        # Cast config to protocol for container.scoped() compatibility
        runtime_config_typed: p.Config = cast(
            "p.Config",
            runtime_config,
        )
        services_option = (
            cast(
                "Mapping[str, t.GeneralValueType | BaseModel | p.VariadicCallable[t.GeneralValueType]] | None",
                u.mapper().get(options, "services"),
            )
            if "services" in options
            else None
        )

        # Cast context to Ctx | None for scoped() compatibility
        context_typed: p.Ctx | None = (
            cast("p.Ctx", runtime_context)
            if isinstance(runtime_context, FlextContext)
            else runtime_context
        )
        factories_raw = options.get("factories")

        runtime_container = FlextContainer.create().scoped(
            config=runtime_config_typed,
            context=context_typed,
            # Cast needed: mapper().get() returns t.GeneralValueType, narrow to str | None
            subproject=cast("str | None", u.mapper().get(options, "subproject")),
            services=services_option,
            factories=cast(
                "Mapping[str, Callable[[], str | int | float | bool | datetime | Sequence[str | int | float | bool | datetime | None] | Mapping[str, str | int | float | bool | datetime | None] | None]] | None",
                factories_raw,
            )
            if isinstance(factories_raw, (dict, Mapping))
            else None,
            resources=cast(
                "Mapping[str, Callable[[], t.GeneralValueType]] | None",
                u.mapper().get(options, "resources"),
            )
            if isinstance(u.mapper().get(options, "resources"), (dict, Mapping))
            else None,
        )

        container_overrides = options.get("container_overrides")
        if container_overrides:
            runtime_container.configure(container_overrides)

        wire_modules = options.get("wire_modules")
        wire_packages = options.get("wire_packages")
        wire_classes = options.get("wire_classes")
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
    ) -> Iterator[t.ConfigurationDict]:
        """Track operation performance with timing and automatic context cleanup."""
        # Get or initialize stats storage for this operation
        stats_attr = f"_stats_{operation_name}"
        # Use correct type - stats values are all t.GeneralValueType (int, float)
        # Use dict for mutability (not Mapping)
        stats: t.ConfigurationDict = getattr(
            self,
            stats_attr,
            {
                "operation_count": 0,
                "error_count": 0,
                "total_duration_ms": 0.0,
            },
        )

        # Increment operation count - use cast for type safety
        op_count_raw = u.mapper().get(stats, "operation_count", default=0)
        stats["operation_count"] = (
            int(op_count_raw if isinstance(op_count_raw, (int, float, str)) else 0) + 1
        )

        try:
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                # Add aggregated stats to metrics for visibility
                metrics["operation_count"] = stats["operation_count"]
                try:
                    yield metrics
                    # Success - update stats
                    if "duration_ms" in metrics:
                        total_dur_raw = u.mapper().get(
                            stats, "total_duration_ms", default=0.0
                        )
                        dur_ms_raw = u.mapper().get(metrics, "duration_ms", default=0.0)
                        total_dur = float(
                            total_dur_raw
                            if isinstance(total_dur_raw, (int, float, str))
                            else 0.0,
                        )
                        dur_ms = float(
                            dur_ms_raw
                            if isinstance(dur_ms_raw, (int, float, str))
                            else 0.0,
                        )
                        stats["total_duration_ms"] = total_dur + dur_ms
                except Exception:
                    # Failure - increment error count
                    err_raw = u.mapper().get(stats, "error_count", default=0)
                    stats["error_count"] = (
                        int(err_raw if isinstance(err_raw, (int, float, str)) else 0)
                        + 1
                    )
                    raise
                finally:
                    # Calculate success rate
                    op_raw = u.mapper().get(stats, "operation_count", default=1)
                    err_raw2 = u.mapper().get(stats, "error_count", default=0)
                    op_count = int(
                        op_raw if isinstance(op_raw, (int, float, str)) else 1,
                    )
                    err_count = int(
                        err_raw2 if isinstance(err_raw2, (int, float, str)) else 0,
                    )
                    stats["success_rate"] = (op_count - err_count) / op_count
                    if op_count > 0:
                        total_raw = u.mapper().get(
                            stats, "total_duration_ms", default=0.0
                        )
                        total_dur_final = float(
                            total_raw
                            if isinstance(total_raw, (int, float, str))
                            else 0.0,
                        )
                        stats["avg_duration_ms"] = total_dur_final / op_count
                    # Update metrics with final stats
                    # stats values are already t.GeneralValueType (int, float)
                    metrics["error_count"] = stats["error_count"]
                    metrics["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics["avg_duration_ms"] = stats["avg_duration_ms"]
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
            service: t.GeneralValueType | BaseModel = cast(
                "t.GeneralValueType | BaseModel",
                self,
            )
            result = self.container.register(
                service_name,
                cast("t.FlexibleValue", service),
            )
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
    def _get_or_create_logger(cls) -> p.Log.StructlogLogger:
        """Get or create DI-injected logger with fallback to direct creation."""
        # Generate unique logger name based on module and class
        logger_name = f"{cls.__module__}.{cls.__name__}"

        # Check cache first (thread-safe)
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cast(
                    "p.Log.StructlogLogger",
                    cls._logger_cache[logger_name],
                )

        # Try to get from DI container
        try:
            container = FlextContainer.create()
            logger_key = f"logger:{logger_name}"

            # Attempt to retrieve logger from container
            logger_result = container.get_typed(logger_key, FlextLogger)

            if logger_result.is_success:
                # Use .value directly - FlextResult never returns None on success
                logger = logger_result.value
                # Cache the result
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger
                return cast("p.Log.StructlogLogger", logger)

            # Logger not in container - create and register
            logger = FlextLogger(logger_name)
            # FlextLogger is not BaseModel, so use register_factory to wrap it
            container_impl: p.DI = container
            # Register factory instead of instance (FlextLogger is not BaseModel or FlexibleValue)

            def _logger_factory() -> t.GeneralValueType:
                # Convert logger to dict-like representation for factory return
                # FlextLogger is not t.GeneralValueType, so convert to dict
                return {"logger": str(logger)}

            with suppress(ValueError, TypeError):
                # Ignore if already registered (race condition)
                _ = container_impl.register_factory(logger_key, _logger_factory)

            # Cache the result
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger

            return cast("p.Log.StructlogLogger", logger)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            # Fallback: create logger without DI if container unavailable
            logger = FlextLogger(logger_name)
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger
            return cast("p.Log.StructlogLogger", logger)

    def _log_with_context(
        self,
        level: str,
        message: str,
        **extra: t.GeneralValueType,
    ) -> None:
        """Log message with automatic context data inclusion."""
        # Normalize extra values to t.GeneralValueType for logging
        correlation_id = FlextContext.Correlation.get_correlation_id()
        operation_name = FlextContext.Request.get_operation_name()
        context_data: t.ConfigurationDict = {
            "correlation_id": FlextRuntime.normalize_to_general_value(correlation_id),
            "operation": FlextRuntime.normalize_to_general_value(operation_name),
            **{k: FlextRuntime.normalize_to_general_value(v) for k, v in extra.items()},
        }

        log_method = getattr(self.logger, level, self.logger.info)
        _ = log_method(message, extra=context_data)

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

    def _enrich_context(self, **context_data: t.GeneralValueType) -> None:
        """Log service information ONCE at initialization (not bound to context)."""
        # Build service context for logging using correct types
        # Use dict for mutability
        service_context: t.ConfigurationDict = {
            "service_name": self.__class__.__name__,
            "service_module": self.__class__.__module__,
            **context_data,
        }
        # Log service initialization ONCE instead of binding to all logs
        self.logger.info("Service initialized", return_result=False, **service_context)

    def _log_config_once(
        self,
        config: t.ConfigurationMapping,
        message: str = "Configuration loaded",
    ) -> None:
        """Log configuration ONCE without binding to context."""
        # Convert config to t.GeneralValueType for logging
        # ConfigurationMapping is Mapping[str, t.GeneralValueType], convert to dict
        config_typed: t.ConfigurationDict = dict(config.items())
        # Log configuration as single event, not bound to context
        self.logger.info(message, config=config_typed)

    @staticmethod
    def _with_operation_context(
        operation_name: str,
        **operation_data: t.GeneralValueType,
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

            # Separate data by level - preserve t.GeneralValueType from operation_data
            # Use dict for mutability
            debug_data: t.ConfigurationDict = {
                k: v for k, v in operation_data.items() if k in debug_keys
            }
            error_data: t.ConfigurationDict = {
                k: v for k, v in operation_data.items() if k in error_keys
            }
            normal_data: t.ConfigurationDict = {
                k: v
                for k, v in operation_data.items()
                if k not in debug_keys and k not in error_keys
            }

            # Bind context using bind_global_context - no level-specific binding available
            # Combine all context data for global binding
            all_context_data: t.ConfigurationDict = dict(normal_data)
            # Simple merge: deep strategy - new values override existing ones
            if debug_data:
                merged_debug: t.ConfigurationDict = dict(all_context_data)
                merged_debug.update(debug_data)
                all_context_data = merged_debug
            if error_data:
                merged_error: t.ConfigurationDict = dict(all_context_data)
                merged_error.update(error_data)
                all_context_data = merged_error
            if all_context_data:
                _ = FlextLogger.bind_global_context(**all_context_data)
            if normal_data:
                _ = FlextLogger.bind_context(
                    c.Context.SCOPE_OPERATION,
                    **normal_data,
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

            # Type annotation for type checker
            _metrics: t.ConfigurationDict

            def __init__(
                self, *args: t.GeneralValueType, **kwargs: t.GeneralValueType
            ) -> None:
                """Initialize metrics tracker with empty metrics dict."""
                super().__init__(*args, **kwargs)
                # Initialize _metrics as instance attribute (not PrivateAttr for mixin compatibility)
                # Use vars() to bypass __setattr__ (works with Pydantic frozen models)
                vars(self)["_metrics"] = {}

            def record_metric(
                self,
                name: str,
                value: t.GeneralValueType,
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
                return r[bool].ok(True)

            def get_metrics(self) -> r[t.ConfigurationDict]:
                """Get current metrics dictionary.

                Returns:
                    r[t.ConfigurationDict]: Success result with metrics collection

                """
                # _metrics is initialized in __init__, but check for safety
                if not hasattr(self, "_metrics"):
                    vars(self)["_metrics"] = {}
                return r[t.ConfigurationDict].ok(self._metrics.copy())

        class ContextStack:
            """Manages execution context stack."""

            # Type annotation for type checker
            _stack: list[FlextModelsHandler.ExecutionContext | t.ConfigurationDict]

            def __init__(
                self, *args: t.GeneralValueType, **kwargs: t.GeneralValueType
            ) -> None:
                """Initialize context stack with empty list."""
                super().__init__(*args, **kwargs)
                # Initialize _stack as instance attribute (not PrivateAttr for mixin compatibility)
                object.__setattr__(self, "_stack", [])

            def push_context(
                self,
                ctx: FlextModelsHandler.ExecutionContext | t.ConfigurationDict,
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
                # Convert dict to ExecutionContext if needed
                if isinstance(ctx, dict):
                    # For backward compatibility, accept dict but convert to ExecutionContext
                    # Create ExecutionContext from dict data
                    handler_name_raw = ctx.get("handler_name", "unknown")
                    handler_name: str = (
                        str(handler_name_raw)
                        if handler_name_raw is not None
                        else "unknown"
                    )
                    handler_mode_raw = ctx.get("handler_mode", "operation")
                    # Cast to Literal type expected by create_for_handler
                    handler_mode_str: str = (
                        str(handler_mode_raw)
                        if handler_mode_raw is not None
                        else "operation"
                    )
                    # Cast handler_mode to Literal type - runtime validated, type checker needs cast
                    handler_mode_literal = cast(
                        "c.Cqrs.HandlerTypeLiteral", handler_mode_str
                    )
                    execution_ctx = (
                        FlextModelsHandler.ExecutionContext.create_for_handler(
                            handler_name=handler_name,
                            handler_mode=handler_mode_literal,
                        )
                    )
                    self._stack.append(execution_ctx)
                else:
                    self._stack.append(ctx)
                return r[bool].ok(True)

            def pop_context(self) -> r[t.ConfigurationDict]:
                """Pop execution context from the stack.

                Returns:
                    r[t.ConfigurationDict]: Success result with popped context or empty dict

                """
                # _stack is initialized in __init__, but check for safety
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if self._stack:
                    popped = self._stack.pop()
                    # Convert ExecutionContext to dict for backward compatibility
                    if isinstance(popped, m.Handler.ExecutionContext):
                        context_dict: t.ConfigurationDict = {
                            "handler_name": popped.handler_name,
                            "handler_mode": popped.handler_mode,
                        }
                        return r[t.ConfigurationDict].ok(context_dict)
                    # If it's already a dict, return as-is
                    return r[t.ConfigurationDict].ok(
                        cast("t.ConfigurationDict", popped)
                    )
                return r[t.ConfigurationDict].ok({})

            def current_context(self) -> m.Handler.ExecutionContext | None:
                """Get current execution context without popping.

                Returns:
                    m.ExecutionContext | None: Current context or None if stack is empty

                """
                # _stack is initialized in __init__, but check for safety
                if not hasattr(self, "_stack"):
                    vars(self)["_stack"] = []
                if self._stack:
                    top_item = self._stack[-1]
                    # Type narrowing: _stack contains ExecutionContext | ConfigurationDict
                    # Return None if ConfigurationDict, otherwise ExecutionContext
                    if isinstance(top_item, m.Handler.ExecutionContext):
                        return top_item
                    return None
                return None

    class Validation:
        """Railway-oriented validation patterns with r composition."""

        @staticmethod
        def validate_with_result(
            data: t.GeneralValueType,
            validators: list[Callable[[t.GeneralValueType], r[bool]]],
        ) -> r[t.GeneralValueType]:
            """Chain validators sequentially, returning first failure or data on success."""
            result: r[t.GeneralValueType] = r[t.GeneralValueType].ok(data)

            for validator in validators:
                # Create helper function with proper closure to validate and preserve data
                def validate_and_preserve(
                    data: t.GeneralValueType,
                    v: Callable[[t.GeneralValueType], r[bool]],
                ) -> r[t.GeneralValueType]:
                    validation_result = v(data)
                    if validation_result.is_failure:
                        base_msg = "Validation failed"
                        error_msg = (
                            f"{base_msg}: {validation_result.error}"
                            if validation_result.error
                            else f"{base_msg} (validation rule failed)"
                        )
                        return r[t.GeneralValueType].fail(
                            error_msg,
                            error_code=validation_result.error_code,
                            error_data=validation_result.error_data,
                        )
                    # Check that validation returned True
                    if validation_result.value is not True:
                        return r[t.GeneralValueType].fail(
                            f"Validator must return r[bool].ok(True) for success, got {validation_result.value!r}",
                        )
                    return r[t.GeneralValueType].ok(data)

                # Use partial to bind validator while passing data through flat_map
                result = result.flat_map(partial(validate_and_preserve, v=validator))

            return result

    class ProtocolValidation:
        """Runtime protocol compliance validation utilities."""

        @staticmethod
        def is_handler(
            obj: p.Handler | Callable[..., t.GeneralValueType],
        ) -> bool:
            """Check if object satisfies p.Handler protocol."""
            return isinstance(obj, p.Handler)

        @staticmethod
        def is_service(
            _obj: p.Service[t.GeneralValueType],
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
            _obj: p.Handler
            | p.Service[t.GeneralValueType]
            | p.CommandBus
            | p.Repository[t.GeneralValueType]
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
            return r[bool].ok(True)

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
                        f"Processor {type(obj).__name__} missing required "
                        f"method '{method_name}()'. "
                        f"Processors must implement: {methods_str}"
                    )
                    return r[bool].fail(error_msg)
                if not callable(u.mapper().get(obj, method_name, default=None)):
                    return r[bool].fail(
                        f"Processor {type(obj).__name__}.{method_name} is not callable",
                    )

            return r[bool].ok(True)


# Alias for runtime compatibility
x = FlextMixins

__all__ = [
    "FlextMixins",
    "require_initialized",
    "x",
]
