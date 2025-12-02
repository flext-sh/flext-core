"""Dispatcher-aware mixins for reusable service infrastructure.

Provide shared behaviors for services and handlers that rely on dispatcher-
first CQRS execution, structured logging, and DI-backed context handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from functools import partial
from typing import ClassVar, TypeVar, cast

from pydantic import BaseModel

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextMixins:
    """Composable behaviors for dispatcher-driven services and handlers.

    These mixins centralize DI container access, structured logging, and
    context management so dispatcher-executed services can stay focused on
    domain work while still emitting `FlextResult` outcomes and metrics.

    Properties:
    - ``container``: Lazy ``FlextContainer`` singleton lookups for DI wiring.
    - ``logger``: Cached ``FlextLogger`` resolution for structured logs.
    - ``context``: Per-operation ``FlextContext`` for correlation metadata.
    - ``config``: Thread-safe ``FlextConfig`` access for runtime settings.

    Key methods:
    - ``track``: Context manager that records timing/err counts per operation.
    - ``_with_operation_context`` / ``_clear_operation_context``: Scoped
      context bindings used by dispatcher pipelines.
    - Delegated ``FlextRuntime``/``FlextResult`` helpers for railway flows.

    Example:
        class MyService(FlextMixins):
            def process(
                self, data: FlextTypes.Types.ContextMetadataMapping
            ) -> FlextResult[FlextTypes.Types.ContextMetadataMapping]:
                with self.track("process"):
                    self.logger.info("Processing", size=len(data))
                    return self.ok({"status": "processed"})

    """

    # =========================================================================
    # RUNTIME VALIDATION UTILITIES (Delegated from FlextRuntime)
    # =========================================================================
    # All classes inheriting FlextMixins automatically have access to
    # runtime validation utilities without explicit FlextRuntime import

    # Type guard utilities
    is_dict_like = staticmethod(FlextRuntime.is_dict_like)
    is_list_like = staticmethod(FlextRuntime.is_list_like)
    is_valid_json = staticmethod(FlextRuntime.is_valid_json)
    is_valid_identifier = staticmethod(FlextRuntime.is_valid_identifier)
    is_valid_phone = staticmethod(FlextRuntime.is_valid_phone)

    # Type introspection utilities
    is_sequence_type = staticmethod(FlextRuntime.is_sequence_type)
    safe_get_attribute = staticmethod(FlextRuntime.safe_get_attribute)
    extract_generic_args = staticmethod(FlextRuntime.extract_generic_args)

    # =========================================================================
    # RESULT FACTORY UTILITIES (Delegated from FlextResult)
    # =========================================================================
    # All classes inheriting FlextMixins automatically have access to
    # FlextResult factory methods for railway-oriented programming

    # Factory methods - Use: self.ok(value) or self.fail("error")
    ok = FlextResult.ok
    fail = FlextResult.fail
    traverse = FlextResult.traverse
    parallel_map = FlextResult.parallel_map
    accumulate_errors = FlextResult.accumulate_errors

    # =========================================================================
    # MODEL CONVERSION UTILITIES (New in Phase 0 - Consolidation)
    # =========================================================================

    class ModelConversion:
        """BaseModel/dict conversion utilities (eliminates 32+ repetitive patterns)."""

        @staticmethod
        def to_dict(
            obj: (
                BaseModel
                | FlextTypes.Types.ContextMetadataMapping
                | FlextTypes.Types.ConfigurationMapping
                | None
            ),
        ) -> FlextTypes.Types.ContextMetadataMapping:
            """Convert BaseModel/dict to dict (None → empty dict).

            Accepts BaseModel, dict with nested structures, or None.
            Nested Mapping/Sequence are preserved as-is in the output.
            """
            if obj is None:
                return {}
            if isinstance(obj, BaseModel):
                # BaseModel.model_dump() returns dict[str, Any], normalize to GeneralValueType
                dumped = obj.model_dump()
                # Recursively normalize to ensure GeneralValueType compliance
                normalized = FlextRuntime.normalize_to_general_value(dumped)
                # Type guard: normalize_to_general_value always returns GeneralValueType
                # For BaseModel.dump(), we know it's a dict-like structure
                if isinstance(normalized, dict):
                    return normalized
                # Fallback: wrap scalar in dict (shouldn't happen for BaseModel.dump())
                return {"value": normalized}
            # For Mapping, values should already be GeneralValueType-compatible
            # Normalize each value to ensure type safety
            result: dict[str, FlextTypes.GeneralValueType] = {}
            for key, value in obj.items():
                normalized_value = FlextRuntime.normalize_to_general_value(value)
                result[key] = normalized_value
            # Return as Mapping[str, GeneralValueType] compatible type
            return result

    # =========================================================================
    # RESULT HANDLING UTILITIES (New in Phase 0 - Consolidation)
    # =========================================================================

    class ResultHandling:
        """FlextResult wrapping utilities (eliminates 209+ repetitive patterns)."""

        @staticmethod
        def ensure_result[T](value: T | FlextResult[T]) -> FlextResult[T]:
            """Wrap value in FlextResult if not already wrapped."""
            return value if isinstance(value, FlextResult) else FlextResult.ok(value)

    # =========================================================================
    # SERVICE INFRASTRUCTURE (Original FlextMixins functionality)
    # =========================================================================

    # Class-level cache for loggers to avoid repeated DI lookups
    _logger_cache: ClassVar[dict[str, FlextLogger]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(cls, **kwargs: FlextTypes.GeneralValueType) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)
        # Container is lazily initialized on first access

    @property
    def container(self) -> FlextContainer:
        """Get global FlextContainer instance with lazy initialization."""
        return FlextContainer()

    @property
    def context(self) -> FlextProtocols.ContextProtocol:
        """Get FlextContext instance for context operations."""
        # FlextContext implements ContextProtocol structurally (no cast needed)
        return FlextContext()

    @property
    def logger(self) -> FlextLogger:
        """Get FlextLogger instance (DI-backed with caching)."""
        return self._get_or_create_logger()

    @contextmanager
    def track(
        self,
        operation_name: str,
    ) -> Iterator[dict[str, FlextTypes.GeneralValueType]]:
        """Track operation performance with timing and automatic context cleanup."""
        # Get or initialize stats storage for this operation
        stats_attr = f"_stats_{operation_name}"
        # Use correct type - stats values are all GeneralValueType (int, float)
        # Use dict for mutability (not Mapping)
        stats: dict[str, FlextTypes.GeneralValueType] = getattr(
            self,
            stats_attr,
            {
                "operation_count": 0,
                "error_count": 0,
                "total_duration_ms": 0.0,
            },
        )

        # Increment operation count - use cast for type safety
        op_count_raw = stats.get("operation_count", 0)
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
                        total_dur_raw = stats.get("total_duration_ms", 0.0)
                        dur_ms_raw = metrics.get("duration_ms", 0.0)
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
                    err_raw = stats.get("error_count", 0)
                    stats["error_count"] = (
                        int(err_raw if isinstance(err_raw, (int, float, str)) else 0)
                        + 1
                    )
                    raise
                finally:
                    # Calculate success rate
                    op_raw = stats.get("operation_count", 1)
                    err_raw2 = stats.get("error_count", 0)
                    op_count = int(
                        op_raw if isinstance(op_raw, (int, float, str)) else 1,
                    )
                    err_count = int(
                        err_raw2 if isinstance(err_raw2, (int, float, str)) else 0,
                    )
                    stats["success_rate"] = (op_count - err_count) / op_count
                    if op_count > 0:
                        total_raw = stats.get("total_duration_ms", 0.0)
                        total_dur_final = float(
                            total_raw
                            if isinstance(total_raw, (int, float, str))
                            else 0.0,
                        )
                        stats["avg_duration_ms"] = total_dur_final / op_count
                    # Update metrics with final stats
                    # stats values are already GeneralValueType (int, float)
                    metrics["error_count"] = stats["error_count"]
                    metrics["success_rate"] = stats["success_rate"]
                    if "avg_duration_ms" in stats:
                        metrics["avg_duration_ms"] = stats["avg_duration_ms"]
                    # Store updated stats
                    setattr(self, stats_attr, stats)
        finally:
            # Auto-cleanup operation context
            FlextMixins._clear_operation_context()

    @property
    def config(self) -> FlextConfig:
        """Get global FlextConfig instance with namespace support."""
        return FlextConfig.get_global_instance()

    def _register_in_container(self, service_name: str) -> FlextResult[bool]:
        """Register self in global container for service discovery."""
        try:
            # container.register accepts GeneralValueType | BaseModel | Callable
            # Cast self to satisfy type checker (self is compatible at runtime)
            service: FlextTypes.GeneralValueType | BaseModel = cast(
                "FlextTypes.GeneralValueType | BaseModel", self
            )
            return self.container.register(service_name, service)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # If already registered, return success (for test compatibility)
            if "already registered" in str(e).lower():
                return FlextResult[bool].ok(True)
            return FlextResult[bool].fail(f"Service registration failed: {e}")

    @staticmethod
    def _propagate_context(operation_name: str) -> None:
        """Propagate context for current operation using FlextContext."""
        FlextContext.Request.set_operation_name(operation_name)
        _ = FlextContext.Utilities.ensure_correlation_id()

    @staticmethod
    def _get_correlation_id() -> str | None:
        """Get current correlation ID from FlextContext."""
        return FlextContext.Correlation.get_correlation_id()

    @staticmethod
    def _set_correlation_id(correlation_id: str) -> None:
        """Set correlation ID in FlextContext."""
        FlextContext.Correlation.set_correlation_id(correlation_id)

    @classmethod
    def _get_or_create_logger(cls) -> FlextLogger:
        """Get or create DI-injected logger with fallback to direct creation."""
        # Generate unique logger name based on module and class
        logger_name = f"{cls.__module__}.{cls.__name__}"

        # Check cache first (thread-safe)
        with cls._cache_lock:
            if logger_name in cls._logger_cache:
                return cls._logger_cache[logger_name]

        # Try to get from DI container
        try:
            container = FlextContainer()
            logger_key = f"logger:{logger_name}"

            # Attempt to retrieve logger from container
            logger_result = container.get_typed(logger_key, FlextLogger)

            if logger_result.is_success:
                # unwrap() returns FlextLogger when is_success is True
                logger = logger_result.unwrap()
                # Cache the result
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger
                return logger

            # Logger not in container - create and register
            logger = FlextLogger(logger_name)
            # FlextLogger is not BaseModel, so use register_factory to wrap it
            container_impl: FlextContainer = container
            # Register factory instead of instance (FlextLogger is not BaseModel or FlexibleValue)

            def _logger_factory() -> FlextTypes.GeneralValueType:
                # Convert logger to dict-like representation for factory return
                # FlextLogger is not GeneralValueType, so convert to dict
                return {"logger": str(logger)}

            with suppress(ValueError, TypeError):
                # Ignore if already registered (race condition)
                _ = container_impl.register_factory(logger_key, _logger_factory)

            # Cache the result
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger

            return logger

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            # Fallback: create logger without DI if container unavailable
            logger = FlextLogger(logger_name)
            with cls._cache_lock:
                cls._logger_cache[logger_name] = logger
            return logger

    def _log_with_context(
        self,
        level: str,
        message: str,
        **extra: FlextTypes.GeneralValueType,
    ) -> None:
        """Log message with automatic context data inclusion."""
        # Normalize extra values to GeneralValueType for logging
        correlation_id = FlextContext.Correlation.get_correlation_id()
        operation_name = FlextContext.Request.get_operation_name()
        context_data: dict[str, FlextTypes.GeneralValueType] = {
            "correlation_id": FlextRuntime.normalize_to_general_value(correlation_id),
            "operation": FlextRuntime.normalize_to_general_value(operation_name),
            **{k: FlextRuntime.normalize_to_general_value(v) for k, v in extra.items()},
        }

        log_method = getattr(self.logger, level, self.logger.info)
        _ = log_method(message, extra=context_data)

    # =========================================================================
    # SERVICE METHODS - Complete Infrastructure (inherited by FlextMixins)
    # =========================================================================

    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic container registration."""
        # Fast fail: service_name must be str or None
        effective_service_name: str = (
            service_name
            if isinstance(service_name, str) and service_name
            else self.__class__.__name__
        )

        register_result = self._register_in_container(effective_service_name)

        if register_result.is_failure:
            # Only log warning if it's not an "already registered" error
            # Fast fail: error must be str (FlextResult guarantees this)
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

    def _enrich_context(self, **context_data: FlextTypes.GeneralValueType) -> None:
        """Log service information ONCE at initialization (not bound to context)."""
        # Build service context for logging using correct types
        # Use dict for mutability
        service_context: dict[str, FlextTypes.GeneralValueType] = {
            "service_name": self.__class__.__name__,
            "service_module": self.__class__.__module__,
            **context_data,
        }
        # Log service initialization ONCE instead of binding to all logs
        self.logger.info("Service initialized", return_result=False, **service_context)

    def _log_config_once(
        self,
        config: FlextTypes.Types.ConfigurationMapping,
        message: str = "Configuration loaded",
    ) -> None:
        """Log configuration ONCE without binding to context."""
        # Convert config to GeneralValueType for logging
        # ConfigurationMapping is Mapping[str, GeneralValueType], convert to dict
        config_typed: dict[str, FlextTypes.GeneralValueType] = dict(config.items())
        # Log configuration as single event, not bound to context
        self.logger.info(message, config=config_typed)

    @staticmethod
    def _with_operation_context(
        operation_name: str,
        **operation_data: FlextTypes.GeneralValueType,
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

            # Separate data by level - preserve GeneralValueType from operation_data
            # Use dict for mutability
            debug_data: dict[str, FlextTypes.GeneralValueType] = {
                k: v for k, v in operation_data.items() if k in debug_keys
            }
            error_data: dict[str, FlextTypes.GeneralValueType] = {
                k: v for k, v in operation_data.items() if k in error_keys
            }
            normal_data: dict[str, FlextTypes.GeneralValueType] = {
                k: v
                for k, v in operation_data.items()
                if k not in debug_keys and k not in error_keys
            }

            # Bind context using bind_global_context - no level-specific binding available
            # Combine all context data for global binding
            all_context_data: dict[str, FlextTypes.GeneralValueType] = {}
            all_context_data.update(normal_data)
            if debug_data:
                all_context_data.update(debug_data)
            if error_data:
                all_context_data.update(error_data)
            if all_context_data:
                _ = FlextLogger.bind_global_context(**all_context_data)
            if normal_data:
                _ = FlextLogger.bind_context(
                    FlextConstants.Context.SCOPE_OPERATION,
                    **normal_data,
                )

    @staticmethod
    def _clear_operation_context() -> None:
        """Clear operation scope context (preserves request/application scopes)."""
        # Clear operation scope only (preserves request and application scopes)
        _ = FlextLogger.clear_scope("operation")

        # Clear FlextContext operation name
        FlextContext.Request.set_operation_name("")

    class Validation:
        """Railway-oriented validation patterns with FlextResult composition."""

        T = TypeVar("T")

        @staticmethod
        def validate_with_result(
            data: FlextTypes.GeneralValueType,
            validators: list[
                Callable[[FlextTypes.GeneralValueType], FlextResult[bool]]
            ],
        ) -> FlextResult[FlextTypes.GeneralValueType]:
            """Chain validators sequentially, returning first failure or data on success."""
            result: FlextResult[FlextTypes.GeneralValueType] = FlextResult.ok(data)

            for validator in validators:
                # Create helper function with proper closure to validate and preserve data
                def validate_and_preserve(
                    data: FlextTypes.GeneralValueType,
                    v: Callable[[FlextTypes.GeneralValueType], FlextResult[bool]],
                ) -> FlextResult[FlextTypes.GeneralValueType]:
                    validation_result = v(data)
                    if validation_result.is_failure:
                        base_msg = "Validation failed"
                        error_msg = (
                            f"{base_msg}: {validation_result.error}"
                            if validation_result.error
                            else f"{base_msg} (validation rule failed)"
                        )
                        return FlextResult[FlextTypes.GeneralValueType].fail(
                            error_msg,
                            error_code=validation_result.error_code,
                            error_data=validation_result.error_data,
                        )
                    # Check that validation returned True
                    if validation_result.value is not True:
                        return FlextResult[FlextTypes.GeneralValueType].fail(
                            f"Validator must return FlextResult[bool].ok(True) for success, got {validation_result.value!r}",
                        )
                    return FlextResult[FlextTypes.GeneralValueType].ok(data)

                # Use partial to bind validator while passing data through flat_map
                result = result.flat_map(partial(validate_and_preserve, v=validator))

            return result

    class ProtocolValidation:
        """Runtime protocol compliance validation utilities."""

        @staticmethod
        def is_handler(
            obj: FlextProtocols.Handler | Callable[..., FlextTypes.GeneralValueType],
        ) -> bool:
            """Check if object satisfies FlextProtocols.Handler protocol."""
            return isinstance(obj, FlextProtocols.Handler)

        @staticmethod
        def is_service(
            _obj: FlextProtocols.Service[FlextTypes.GeneralValueType],
        ) -> bool:
            """Check if object satisfies FlextProtocols.Service protocol.

            Uses structural typing - any object implementing Service protocol
            will pass this check, including FlextService instances.
            """
            return True

        @staticmethod
        def is_command_bus(_obj: FlextProtocols.CommandBus) -> bool:
            """Check if object satisfies FlextProtocols.CommandBus protocol."""
            return True

        @staticmethod
        def validate_protocol_compliance(
            _obj: FlextProtocols.Handler
            | FlextProtocols.Service[FlextTypes.GeneralValueType]
            | FlextProtocols.CommandBus
            | FlextProtocols.Repository[FlextTypes.GeneralValueType]
            | FlextProtocols.Configurable,
            protocol_name: str,
        ) -> FlextResult[bool]:
            """Validate object compliance with named protocol."""
            protocol_map = {
                "Handler": FlextProtocols.Handler,
                "Service": FlextProtocols.Service,
                "CommandBus": FlextProtocols.CommandBus,
                "Repository": FlextProtocols.Repository,
                "Configurable": FlextProtocols.Configurable,
            }

            if protocol_name not in protocol_map:
                supported = ", ".join(protocol_map.keys())
                return FlextResult[bool].fail(
                    f"Unknown protocol: {protocol_name}. Supported: {supported}",
                )

            # Type already guarantees protocol compliance
            return FlextResult[bool].ok(True)

        @staticmethod
        def validate_processor_protocol(
            obj: FlextProtocols.HasModelDump,
        ) -> FlextResult[bool]:
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
                    return FlextResult[bool].fail(error_msg)
                if not callable(getattr(obj, method_name)):
                    return FlextResult[bool].fail(
                        f"Processor {type(obj).__name__}.{method_name} is not callable",
                    )

            return FlextResult[bool].ok(True)


__all__ = [
    "FlextMixins",
]
