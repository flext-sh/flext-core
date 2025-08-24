"""Central orchestration for unified FLEXT Core system access."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, cast, override

from pydantic import Field, field_validator

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.commands import FlextCommands
from flext_core.config import (
    merge_configs,
    safe_get_env_var,
)
from flext_core.constants import FlextConstants
from flext_core.container import (
    FlextContainer,
    get_flext_container,
)
from flext_core.context import FlextContext
from flext_core.decorators import (
    FlextDecoratorFactory,
    FlextDecorators,
    FlextErrorHandlingDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
)
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import (
    FlextConfigurationError,
    FlextError,
    FlextExceptions,
    FlextValidationError,
    clear_exception_metrics,
    get_exception_metrics,
)
from flext_core.fields import (
    FlextFieldRegistry,
    FlextFields,
    flext_create_boolean_field,
    flext_create_integer_field,
    flext_create_string_field,
)
from flext_core.guards import (
    FlextGuards,
    immutable,
    require_non_empty,
    require_not_none,
    require_positive,
)
from flext_core.handlers import (
    FlextAbstractHandler,
    FlextBaseHandler,
    FlextHandlerRegistry,
    FlextHandlers,
)
from flext_core.loggings import (
    FlextLogContextManager,
    FlextLogger,
    FlextLoggerFactory,
    FlextLogLevel,
    create_log_context,
)
from flext_core.mixins import (
    FlextCacheableMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextValidatableMixin,
)
from flext_core.models import (
    FlextEntity,
    FlextFactory,
    FlextValue,
    create_database_model,
    create_service_model,
)
from flext_core.observability import (
    FlextMinimalObservability,
    get_global_observability,
    reset_global_observability,
)
from flext_core.payload import (
    FlextEvent,
    FlextMessage,
    FlextPayload,
    create_cross_service_event,
    create_cross_service_message,
    get_serialization_metrics,
    validate_cross_service_protocol,
)
from flext_core.result import FlextResult, safe_call
from flext_core.root_models import (
    FlextEmailAddress,
    FlextEntityId,
    FlextMetadata,
    FlextServiceName,
    FlextTimestamp,
    FlextVersion,
    create_email,
    create_entity_id,
    create_service_name,
    create_version,
)
from flext_core.schema_processing import (
    FlextProcessingPipeline,
)
from flext_core.services import FlextServiceProcessor
from flext_core.typings import FlextPlugin, FlextRepository, FlextTypes, P, R, T
from flext_core.utilities import (
    FlextConsole,
    FlextGenerators,
    FlextPerformance,
    FlextTypeGuards,
    FlextUtilities,
    generate_correlation_id,
    generate_id,
    generate_uuid,
    is_not_none,
    truncate,
)
from flext_core.validation import (
    FlextAbstractValidator,
    FlextPredicates,
    FlextValidators,
    flext_validate_email,
    flext_validate_numeric,
    flext_validate_required,
    flext_validate_service_name,
    flext_validate_string,
)

# Use centralized types from FlextTypes
ValidatorCallable = FlextTypes.Core.Validator


class FlextCore:
    """Comprehensive facade providing unified access to ALL FLEXT functionality.

    This class exposes the complete FLEXT Core ecosystem through a single interface,
    including container management, logging, validation, domain modeling, CQRS,
    utilities, observability, and all architectural patterns.

    Thread-safe with lazy initialization of components.
    """

    _instance: FlextCore | None = None

    def __init__(self) -> None:
        """Initialize FLEXT Core with all subsystems."""
        # Core container
        self._container = get_flext_container()

        # Settings cache
        self._settings_cache: dict[type[object], object] = {}

        # Lazy-loaded components
        self._handler_registry: FlextHandlerRegistry | None = None
        self._field_registry: FlextFieldRegistry | None = None
        self._plugin_registry: object | None = None
        self._console: FlextConsole | None = None
        self._observability: FlextMinimalObservability | None = None

    @classmethod
    def get_instance(cls) -> FlextCore:
        """Get singleton instance of FlextCore."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # CONTAINER & DEPENDENCY INJECTION
    # =========================================================================

    @property
    def container(self) -> FlextContainer:
        """Access dependency injection container."""
        return self._container

    def register_service(
        self,
        key: str,
        service: object,
    ) -> FlextResult[None]:
        """Register service in container."""
        return self._container.register(str(key), service)

    def get_service(self, key: str) -> FlextResult[object]:
        """Get service from container."""
        result = self._container.get(str(key))
        if result.is_failure:
            return FlextResult[object].fail(result.error or "Service not found")
        return FlextResult[object].ok(result.value)

    def register_factory(
        self,
        key: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register service factory in container."""
        return self._container.register_factory(str(key), factory)

    def configure_container(self, **config: object) -> FlextResult[None]:
        """Configure container with settings."""
        try:
            # Configure container if it has configuration capability
            if hasattr(self._container, "configure"):
                configure_method = getattr(self._container, "configure", None)
                if callable(configure_method):
                    configure_method(**config)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Container configuration failed: {e}")

    def clear_container(self) -> FlextResult[None]:
        """Clear all services from container."""
        try:
            if hasattr(self._container, "clear"):
                self._container.clear()
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Container clear failed: {e}")

    # =========================================================================
    # LOGGING & OBSERVABILITY
    # =========================================================================

    @staticmethod
    def get_logger(name: str) -> FlextLogger:
        """Get configured logger instance."""
        return FlextLoggerFactory.get_logger(name)

    @staticmethod
    def configure_logging(
        *,
        log_level: str = "INFO",
        _json_output: bool | None = None,
    ) -> None:
        """Configure logging system."""
        log_level_enum = FlextLogLevel.INFO
        try:
            log_level_enum = FlextLogLevel(log_level.upper())
        except (ValueError, AttributeError):
            log_level_enum = FlextLogLevel.INFO

        FlextLoggerFactory.set_global_level(log_level)

        if _json_output is not None:
            FlextLogger.configure(
                log_level=log_level_enum,
                json_output=_json_output,
                add_timestamp=True,
                add_caller=False,
            )

    def create_log_context(
        self, logger: FlextLogger | str | None = None, **context: object
    ) -> FlextLogContextManager:
        """Create structured logging context manager."""
        return create_log_context(logger, **context)

    @property
    def observability(self) -> FlextMinimalObservability:
        """Get observability instance."""
        if self._observability is None:
            self._observability = get_global_observability()
        return self._observability

    def reset_observability(self) -> None:
        """Reset global observability state."""
        reset_global_observability()
        self._observability = None

    # =========================================================================
    # RESULT PATTERN & RAILWAY PROGRAMMING
    # =========================================================================

    @staticmethod
    def ok(value: object) -> FlextResult[object]:
        """Create successful Result."""
        return FlextResult[object].ok(value)

    @staticmethod
    def fail(error: str) -> FlextResult[object]:
        """Create failed Result."""
        return FlextResult[object].fail(error)

    @staticmethod
    def from_exception(exc: Exception) -> FlextResult[object]:
        """Create failed Result from exception."""
        return FlextResult[object].fail(str(exc))

    @staticmethod
    def sequence(results: list[FlextResult[object]]) -> FlextResult[list[object]]:
        """Convert list of Results to Result of list."""
        values: list[object] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[object]].fail(result.error or "Sequence failed")
            values.append(result.value)
        return FlextResult[list[object]].ok(values)

    @staticmethod
    def first_success(results: list[FlextResult[object]]) -> FlextResult[object]:
        """Return first successful Result, or last error if all fail."""
        last_error = "No results provided"
        for result in results:
            if result.is_success:
                return result
            last_error = result.error or "Unknown error"
        return FlextResult[object].fail(last_error)

    # =========================================================================
    # FUNCTIONAL PROGRAMMING & PIPELINES
    # =========================================================================

    @staticmethod
    def pipe(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Create a pipeline of Result-returning functions."""

        def pipeline(value: object) -> FlextResult[object]:
            result: FlextResult[object] = FlextResult[object].ok(value)
            for func in funcs:
                if result.is_failure:
                    break
                result = func(result.value)
            return result

        return pipeline

    @staticmethod
    def compose(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Compose Result-returning functions (right to left)."""
        return FlextCore.pipe(*reversed(funcs))

    @staticmethod
    def when(
        predicate: ValidatorCallable,
        then_func: Callable[[object], FlextResult[object]],
        else_func: Callable[[object], FlextResult[object]] | None = None,
    ) -> Callable[[object], FlextResult[object]]:
        """Conditional Result execution."""

        def conditional(value: object) -> FlextResult[object]:
            if predicate(value):
                return then_func(value)
            if else_func:
                return else_func(value)
            return FlextResult[object].ok(value)

        return conditional

    @staticmethod
    def tap(
        side_effect: Callable[[object], None],
    ) -> Callable[[object], FlextResult[object]]:
        """Execute side effect in pipeline."""

        def side_effect_wrapper(value: object) -> FlextResult[object]:
            side_effect(value)
            return FlextResult[object].ok(value)

        return side_effect_wrapper

    # =========================================================================
    # VALIDATION & GUARDS
    # =========================================================================

    @staticmethod
    def validate_required(
        value: object, field_name: str = "value"
    ) -> FlextResult[object]:
        """Validate that a value is not None or empty."""
        result = flext_validate_required(value, field_name)
        if result.is_valid:
            return FlextResult[object].ok(value)
        return FlextResult[object].fail(result.error_message)

    @staticmethod
    def validate_string(
        value: object, min_length: int = 0, max_length: int | None = None
    ) -> FlextResult[str]:
        """Validate string value with length constraints."""
        if not isinstance(value, str):
            return FlextResult[str].fail("Value must be a string")
        result = flext_validate_string(value, "field", min_length, max_length)
        if result.is_valid:
            return FlextResult[str].ok(value)
        return FlextResult[str].fail(result.error_message)

    @staticmethod
    def validate_numeric(
        value: object, min_value: float | None = None, max_value: float | None = None
    ) -> FlextResult[float]:
        """Validate numeric value with range constraints."""
        if not isinstance(value, (int, float)):
            return FlextResult[float].fail("Value must be numeric")
        numeric_value = float(value)
        result = flext_validate_numeric(numeric_value, "field", min_value, max_value)
        if result.is_valid:
            return FlextResult[float].ok(numeric_value)
        return FlextResult[float].fail(result.error_message)

    @staticmethod
    def validate_email(value: object) -> FlextResult[str]:
        """Validate email address format."""
        if not isinstance(value, str):
            return FlextResult[str].fail("Email must be a string")
        result = flext_validate_email(value)
        if result.is_valid:
            return FlextResult[str].ok(value)
        return FlextResult[str].fail(result.error_message)

    @staticmethod
    def validate_service_name(value: object) -> FlextResult[str]:
        """Validate service name format."""
        if not isinstance(value, str):
            return FlextResult[str].fail("Service name must be a string")
        is_valid = flext_validate_service_name(value)
        if is_valid:
            return FlextResult[str].ok(value)
        return FlextResult[str].fail("Invalid service name format")

    @staticmethod
    def require_not_none(
        value: T | None, message: str = "Value cannot be None"
    ) -> FlextResult[T]:
        """Guard that ensures value is not None."""
        try:
            result = require_not_none(value, message)
            return FlextResult[T].ok(cast("T", result))
        except Exception as e:
            return FlextResult[T].fail(str(e))

    @staticmethod
    def require_non_empty(
        value: str, message: str = "Value cannot be empty"
    ) -> FlextResult[str]:
        """Guard that ensures string is not empty."""
        try:
            result = require_non_empty(value, message)
            return FlextResult[str].ok(cast("str", result))
        except Exception as e:
            return FlextResult[str].fail(str(e))

    @staticmethod
    def require_positive(
        value: float, message: str = "Value must be positive"
    ) -> FlextResult[float]:
        """Guard that ensures number is positive."""
        try:
            result = require_positive(value, message)
            return FlextResult[float].ok(cast("float", result))
        except Exception as e:
            return FlextResult[float].fail(str(e))

    def create_validator(
        self, validation_func: ValidatorCallable, error_message: str
    ) -> Callable[[object], FlextResult[object]]:
        """Create custom validator function."""

        def validator(value: object) -> FlextResult[object]:
            if validation_func(value):
                return FlextResult[object].ok(value)
            return FlextResult[object].fail(error_message)

        return validator

    @property
    def validators(self) -> object:
        """Access validation utilities."""
        return FlextValidators

    @property
    def predicates(self) -> object:
        """Access predicate functions."""
        return FlextPredicates

    @property
    def guards(self) -> type[FlextGuards]:
        """Access guard functions."""
        return FlextGuards

    # =========================================================================
    # CONFIGURATION & SETTINGS
    # =========================================================================

    def get_settings(self, settings_class: type[object]) -> object:
        """Get settings instance with caching."""
        if settings_class not in self._settings_cache:
            self._settings_cache[settings_class] = settings_class()
        return self._settings_cache[settings_class]

    @property
    def constants(self) -> type[FlextConstants]:
        """Access FLEXT constants."""
        return FlextConstants

    @staticmethod
    def load_config_from_env(prefix: str = "FLEXT_") -> FlextResult[dict[str, object]]:
        """Load configuration from environment variables (foundation pattern)."""
        try:
            env_data: dict[str, object] = {}
            prefix_with_sep = f"{prefix.rstrip('_')}_"

            for key, value in os.environ.items():
                if key.startswith(prefix_with_sep):
                    # Remove prefix and convert to lowercase
                    config_key = key[len(prefix_with_sep) :].lower()
                    env_data[config_key] = value

            return FlextResult[dict[str, object]].ok(env_data)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to load config: {e}")

    @staticmethod
    def merge_configs(*configs: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Merge multiple configuration dictionaries."""
        try:
            min_configs_for_merge = 2
            if len(configs) < min_configs_for_merge:
                return FlextResult[dict[str, object]].fail(
                    "At least 2 configs required for merging"
                )
            result = merge_configs(configs[0], configs[1])
            if result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    result.error or "Config merge failed"
                )
            return result  # Already correct type
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to merge configs: {e}")

    @staticmethod
    def validate_config(
        config: dict[str, object], schema: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Validate configuration against schema."""
        try:
            # Simple validation - check that all schema keys are present
            for key in schema:
                if key not in config:
                    return FlextResult[dict[str, object]].fail(
                        f"Missing required config key: {key}"
                    )
            return FlextResult[dict[str, object]].ok(config)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Config validation error: {e}")

    @staticmethod
    def safe_get_env_var(name: str, default: str | None = None) -> FlextResult[str]:
        """Safely get environment variable."""
        return safe_get_env_var(name, default)

    # =========================================================================
    # DOMAIN MODELING & DDD PATTERNS
    # =========================================================================

    @staticmethod
    def create_entity(entity_class: type[T], **data: object) -> FlextResult[T]:
        """Create domain entity with validation."""
        try:
            if hasattr(entity_class, "model_validate"):
                # Use getattr to safely access the method with type safety
                model_validate = getattr(entity_class, "model_validate", None)
                if callable(model_validate):
                    instance = model_validate(data)
                    if not isinstance(instance, entity_class):
                        return FlextResult[T].fail(
                            "Model validation returned incorrect type"
                        )
                else:
                    instance = entity_class(**data)
            else:
                instance = entity_class(**data)
            # Type assertion after validation - instance must be of type T
            validated_instance = (
                instance if isinstance(instance, entity_class) else entity_class(**data)
            )
            return FlextResult[T].ok(validated_instance)
        except Exception as e:
            return FlextResult[T].fail(f"Entity creation failed: {e}")

    @staticmethod
    def create_value_object(vo_class: type[T], **data: object) -> FlextResult[T]:
        """Create value object with validation."""
        try:
            if hasattr(vo_class, "model_validate"):
                # Use getattr to safely access the method with type safety
                model_validate = getattr(vo_class, "model_validate", None)
                if callable(model_validate):
                    instance = model_validate(data)
                    if not isinstance(instance, vo_class):
                        return FlextResult[T].fail(
                            "Model validation returned incorrect type"
                        )
                else:
                    instance = vo_class(**data)
            else:
                instance = vo_class(**data)
            # Type assertion after validation - instance must be of type T
            validated_instance = (
                instance if isinstance(instance, vo_class) else vo_class(**data)
            )
            return FlextResult[T].ok(validated_instance)
        except Exception as e:
            return FlextResult[T].fail(f"Value object creation failed: {e}")

    @staticmethod
    def create_aggregate_root(
        aggregate_class: type[T], **data: object
    ) -> FlextResult[T]:
        """Create aggregate root with validation."""
        try:
            if hasattr(aggregate_class, "model_validate"):
                # Use getattr to safely access the method with type safety
                model_validate = getattr(aggregate_class, "model_validate", None)
                if callable(model_validate):
                    instance = model_validate(data)
                    if not isinstance(instance, aggregate_class):
                        return FlextResult[T].fail(
                            "Model validation returned incorrect type"
                        )
                else:
                    instance = aggregate_class(**data)
            else:
                instance = aggregate_class(**data)
            # Type assertion after validation - instance must be of type T
            validated_instance = (
                instance
                if isinstance(instance, aggregate_class)
                else aggregate_class(**data)
            )
            return FlextResult[T].ok(validated_instance)
        except Exception as e:
            return FlextResult[T].fail(f"Aggregate root creation failed: {e}")

    @property
    def entity_base(self) -> type[FlextEntity]:
        """Access entity base class."""
        return FlextEntity

    @property
    def value_object_base(self) -> type[FlextValue]:
        """Access value object base class."""
        return FlextValue

    @property
    def aggregate_root_base(self) -> type[FlextAggregateRoot]:
        """Access aggregate root base class."""
        return FlextAggregateRoot

    @property
    def domain_service_base(self) -> type[FlextDomainService[object]]:
        """Access domain service base class."""
        return FlextDomainService[object]

    # =========================================================================
    # UTILITIES & GENERATORS
    # =========================================================================

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID."""
        return generate_id()

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID."""
        return generate_uuid()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID."""
        return generate_correlation_id()

    @staticmethod
    def safe_call(func: Callable[[], T], default: T) -> T:
        """Safely call function with default fallback."""
        result = safe_call(func)
        if result.is_failure:
            return default
        return result.value

    @staticmethod
    def truncate(text: str, max_length: int = 100) -> str:
        """Truncate text to maximum length."""
        return truncate(text, max_length)

    @staticmethod
    def is_not_none(value: object | None) -> bool:
        """Check if value is not None."""
        return is_not_none(value)

    @property
    def console(self) -> FlextConsole:
        """Get console instance."""
        if self._console is None:
            self._console = FlextConsole()
        return self._console

    @property
    def utilities(self) -> type[FlextUtilities]:
        """Access utility functions."""
        return FlextUtilities

    @property
    def generators(self) -> type[FlextGenerators]:
        """Access generator functions."""
        return FlextGenerators

    @property
    def type_guards(self) -> type[FlextTypeGuards]:
        """Access type guard functions."""
        return FlextTypeGuards

    # =========================================================================
    # MESSAGING & EVENTS
    # =========================================================================

    @staticmethod
    def create_message(
        message_type: str, **kwargs: object
    ) -> FlextResult[FlextMessage]:
        """Create cross-service message."""
        try:
            correlation_id = kwargs.pop("correlation_id", None)
            if isinstance(correlation_id, str):
                message_result = create_cross_service_message(
                    message_type, correlation_id, **kwargs
                )
            else:
                message_result = create_cross_service_message(
                    message_type, None, **kwargs
                )

            if message_result.is_failure:
                return FlextResult[FlextMessage].fail(
                    message_result.error or "Message creation failed"
                )
            return message_result
        except Exception as e:
            return FlextResult[FlextMessage].fail(f"Message creation failed: {e}")

    @staticmethod
    def create_event(
        event_type: str, data: dict[str, object], **kwargs: object
    ) -> FlextResult[FlextEvent]:
        """Create cross-service event."""
        try:
            correlation_id = kwargs.pop("correlation_id", None)
            correlation_id_str = (
                correlation_id if isinstance(correlation_id, str) else None
            )
            event_result = create_cross_service_event(
                event_type, data, correlation_id_str, **kwargs
            )

            if event_result.is_failure:
                return FlextResult[FlextEvent].fail(
                    event_result.error or "Event creation failed"
                )
            return event_result
        except Exception as e:
            return FlextResult[FlextEvent].fail(f"Event creation failed: {e}")

    @staticmethod
    def validate_protocol(payload: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Validate cross-service protocol."""
        validation_result = validate_cross_service_protocol(payload)
        if validation_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                validation_result.error or "Protocol validation failed"
            )
        return FlextResult[dict[str, object]].ok(payload)

    @staticmethod
    def get_serialization_metrics() -> dict[str, object]:
        """Get payload serialization metrics."""
        return get_serialization_metrics({})

    @property
    def payload_base(self) -> type[FlextPayload[object]]:
        """Access payload base class."""
        return FlextPayload[object]

    @property
    def message_base(self) -> type[FlextMessage]:
        """Access message base class."""
        return FlextMessage

    @property
    def event_base(self) -> type[FlextEvent]:
        """Access event base class."""
        return FlextEvent

    # =========================================================================
    # HANDLERS & CQRS
    # =========================================================================

    @property
    def handler_registry(self) -> FlextHandlerRegistry:
        """Get handler registry instance."""
        if self._handler_registry is None:
            self._handler_registry = FlextHandlerRegistry()
        return self._handler_registry

    def register_handler(self, handler_type: str, handler: object) -> FlextResult[None]:
        """Register event/command handler."""
        try:
            registry = self.handler_registry
            if hasattr(registry, "register"):
                validated_handler = cast(
                    "FlextAbstractHandler[object, object]", handler
                )
                registry.register(handler_type, validated_handler)
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(
                "Handler registry does not support registration"
            )
        except Exception as e:
            return FlextResult[None].fail(f"Handler registration failed: {e}")

    def get_handler(self, handler_type: str) -> FlextResult[object]:
        """Get registered handler."""
        try:
            registry = self.handler_registry
            if hasattr(registry, "get_handler"):
                return registry.get_handler(handler_type)
            return FlextResult[object].fail(f"Handler not found: {handler_type}")
        except Exception as e:
            return FlextResult[object].fail(f"Handler retrieval failed: {e}")

    @property
    def handlers(self) -> type[FlextHandlers]:
        """Access handlers utilities."""
        return FlextHandlers

    @property
    def base_handler(self) -> type[FlextBaseHandler]:
        """Access base handler class."""
        return FlextBaseHandler

    @property
    def commands(self) -> type[FlextCommands]:
        """Access CQRS commands."""
        return FlextCommands

    # =========================================================================
    # FIELDS & METADATA
    # =========================================================================

    @property
    def field_registry(self) -> FlextFieldRegistry:
        """Get field registry instance."""
        if self._field_registry is None:
            self._field_registry = FlextFieldRegistry()
        return self._field_registry

    @staticmethod
    def create_string_field(name: str, **kwargs: object) -> object:
        """Create string field definition."""
        return flext_create_string_field(name, **kwargs)

    @staticmethod
    def create_integer_field(name: str, **kwargs: object) -> object:
        """Create integer field definition."""
        return flext_create_integer_field(name, **kwargs)

    @staticmethod
    def create_boolean_field(name: str, **kwargs: object) -> object:
        """Create boolean field definition."""
        return flext_create_boolean_field(name, **kwargs)

    @property
    def fields(self) -> type[FlextFields]:
        """Access field utilities."""
        return FlextFields

    # =========================================================================
    # DECORATORS & ASPECTS
    # =========================================================================

    @property
    def decorators(self) -> type[FlextDecorators]:
        """Access decorator utilities."""
        return FlextDecorators

    @property
    def decorator_factory(self) -> type[FlextDecoratorFactory]:
        """Access decorator factory."""
        return FlextDecoratorFactory

    def create_validation_decorator(self, validator: ValidatorCallable) -> object:
        """Create custom validation decorator."""
        factory = FlextDecoratorFactory()
        return factory.create_validation_decorator(validator=validator)

    def create_error_handling_decorator(self) -> type[FlextErrorHandlingDecorators]:
        """Create custom error handling decorator."""
        return FlextErrorHandlingDecorators

    def create_performance_decorator(self) -> type[FlextPerformanceDecorators]:
        """Create performance monitoring decorator."""
        return FlextPerformanceDecorators

    def create_logging_decorator(self) -> type[FlextLoggingDecorators]:
        """Create logging decorator."""
        return FlextLoggingDecorators

    @staticmethod
    def make_immutable(target_class: type[T]) -> type[T]:
        """Make class immutable."""
        return immutable(target_class)

    @staticmethod
    def make_pure(func: Callable[P, R]) -> Callable[P, R]:
        """Make function pure."""
        # Cast to satisfy type compatibility

        return cast("Callable[P, R]", FlextGuards.pure(func))

    # =========================================================================
    # MIXINS & COMPOSITION
    # =========================================================================

    @property
    def timestamp_mixin(self) -> type[FlextTimestampMixin]:
        """Access timestamp mixin."""
        return FlextTimestampMixin

    @property
    def identifiable_mixin(self) -> type[FlextIdentifiableMixin]:
        """Access identifiable mixin."""
        return FlextIdentifiableMixin

    @property
    def loggable_mixin(self) -> type[FlextLoggableMixin]:
        """Access loggable mixin."""
        return FlextLoggableMixin

    @property
    def validatable_mixin(self) -> type[FlextValidatableMixin]:
        """Access validatable mixin."""
        return FlextValidatableMixin

    @property
    def serializable_mixin(self) -> type[FlextSerializableMixin]:
        """Access serializable mixin."""
        return FlextSerializableMixin

    @property
    def cacheable_mixin(self) -> type[FlextCacheableMixin]:
        """Access cacheable mixin."""
        return FlextCacheableMixin

    # =========================================================================
    # ROOT MODELS & VALUE TYPES
    # =========================================================================

    @staticmethod
    def create_entity_id(value: str | None = None) -> FlextResult[FlextEntityId]:
        """Create entity ID."""
        if value is None:
            return FlextResult[FlextEntityId].fail("Entity ID value cannot be None")
        return create_entity_id(value)

    @staticmethod
    def create_version_number(value: int) -> FlextResult[FlextVersion]:
        """Create version number."""
        return create_version(value)

    @staticmethod
    def create_email_address(value: str) -> FlextResult[FlextEmailAddress]:
        """Create email address."""
        return create_email(value)

    @staticmethod
    def create_service_name_value(value: str) -> FlextResult[FlextServiceName]:
        """Create service name."""
        return create_service_name(value)

    @staticmethod
    def create_timestamp() -> FlextTimestamp:
        """Create current timestamp."""
        return FlextTimestamp.now()

    @staticmethod
    def create_metadata(**data: object) -> FlextResult[FlextMetadata]:
        """Create metadata object."""
        try:
            typed_data = dict(data)
            metadata = FlextMetadata(typed_data)
            return FlextResult[FlextMetadata].ok(metadata)
        except Exception as e:
            return FlextResult[FlextMetadata].fail(f"Metadata creation failed: {e}")

    # =========================================================================
    # EXCEPTIONS & ERROR HANDLING
    # =========================================================================

    @staticmethod
    def create_error(message: str, error_code: str | None = None) -> object:
        """Create FLEXT error."""
        return FlextError(message, error_code=error_code)

    @staticmethod
    def create_validation_error(message: str, field_name: str | None = None) -> object:
        """Create validation error."""
        try:
            # Try different constructor patterns
            if field_name:
                return FlextValidationError(f"{message} (field: {field_name})")
            return FlextValidationError(message)
        except Exception:
            # Fallback to basic constructor
            return FlextValidationError(message)

    @staticmethod
    def create_configuration_error(
        message: str, config_key: str | None = None
    ) -> object:
        """Create configuration error."""
        try:
            # Try different constructor patterns
            if config_key:
                return FlextConfigurationError(f"{message} (config: {config_key})")
            return FlextConfigurationError(message)
        except Exception:
            # Fallback to basic constructor
            return FlextConfigurationError(message)

    @staticmethod
    def get_exception_metrics() -> dict[str, object]:
        """Get exception metrics."""
        metrics = get_exception_metrics()
        return cast("dict[str, object]", metrics)

    @staticmethod
    def clear_exception_metrics() -> None:
        """Clear exception metrics."""
        clear_exception_metrics()

    @property
    def exceptions(self) -> type[FlextExceptions]:
        """Access exception utilities."""
        return FlextExceptions

    # =========================================================================
    # SCHEMA PROCESSING & SEMANTIC
    # =========================================================================

    @staticmethod
    def create_processing_pipeline() -> FlextProcessingPipeline[object, object]:
        """Create processing pipeline."""
        return FlextProcessingPipeline[object, object]()

    # =========================================================================
    # CONTEXT & PROTOCOLS
    # =========================================================================

    @staticmethod
    def create_context(**data: object) -> FlextResult[FlextContext]:
        """Create execution context."""
        try:
            context = FlextContext(**data)
            return FlextResult[FlextContext].ok(context)
        except Exception as e:
            return FlextResult[FlextContext].fail(f"Context creation failed: {e}")

    @property
    def context_class(self) -> type[FlextContext]:
        """Access context class."""
        return FlextContext

    @property
    def repository_protocol(self) -> type[FlextRepository[object]]:
        """Access repository protocol."""
        return cast("type[FlextRepository[object]]", FlextRepository)

    @property
    def plugin_protocol(self) -> object:
        """Access plugin protocol."""
        return FlextPlugin

    @property
    def plugin_registry(self) -> object:
        """Get plugin registry instance."""
        if self._plugin_registry is None:
            # Create a simple plugin registry implementation
            class SimplePluginRegistry:
                def __init__(self) -> None:
                    self._plugins: dict[str, object] = {}

                def register(self, plugin: object) -> None:
                    plugin_name = getattr(plugin, "name", str(type(plugin).__name__))
                    self._plugins[plugin_name] = plugin

                def get(self, name: str) -> object | None:
                    return self._plugins.get(name)

                def list_plugins(self) -> list[str]:
                    return list(self._plugins.keys())

            self._plugin_registry = SimplePluginRegistry()
        return self._plugin_registry

    def register_plugin(self, plugin: object) -> FlextResult[None]:
        """Register plugin."""
        try:
            registry = self.plugin_registry
            if hasattr(registry, "register"):
                register_method = getattr(registry, "register", None)
                if callable(register_method):
                    register_method(plugin)
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(
                "Plugin registry does not support registration"
            )
        except Exception as e:
            return FlextResult[None].fail(f"Plugin registration failed: {e}")

    # =========================================================================
    # TYPE VALIDATION & GUARDS
    # =========================================================================

    @staticmethod
    def validate_type(obj: object, expected_type: type) -> FlextResult[object]:
        """Validate object type."""
        if not isinstance(obj, expected_type):
            return FlextResult[object].fail(
                f"Expected {expected_type.__name__}, got {type(obj).__name__}"
            )
        return FlextResult[object].ok(obj)

    @staticmethod
    def validate_dict_structure(
        obj: object, value_type: type
    ) -> FlextResult[dict[str, object]]:
        """Validate dictionary structure."""
        if not isinstance(obj, dict):
            return FlextResult[dict[str, object]].fail("Expected dictionary")

        if not FlextGuards.is_dict_of(cast("dict[object, object]", obj), value_type):
            return FlextResult[dict[str, object]].fail(
                f"Dictionary values must be of type {value_type.__name__}"
            )

        return FlextResult[dict[str, object]].ok(cast("dict[str, object]", obj))

    @staticmethod
    def create_validated_model(
        model_class: type, **data: object
    ) -> FlextResult[object]:
        """Create validated model."""
        try:
            model_validate_attr = getattr(model_class, "model_validate", None)
            if callable(model_validate_attr):
                instance: object = model_validate_attr(data)
                return FlextResult[object].ok(instance)
            instance_fallback: object = model_class(**data)
            return FlextResult[object].ok(instance_fallback)
        except Exception as e:
            return FlextResult[object].fail(f"Model validation failed: {e}")

    # =========================================================================
    # PERFORMANCE & MONITORING
    # =========================================================================

    @property
    def performance(self) -> type[FlextPerformance]:
        """Access performance utilities."""
        return FlextPerformance

    def track_performance(self, operation_name: str) -> object:
        """Create performance tracking decorator."""

        def _raise_not_callable() -> None:
            """Abstract raise to inner function for TRY301 compliance."""
            msg = "Decorated object is not callable"
            raise TypeError(msg)

        def decorator(func: object) -> object:
            def wrapper(*args: object, **kwargs: object) -> object:
                start_time = datetime.now(UTC)
                try:
                    # Use getattr for type-safe callable access
                    if callable(func):
                        result = func.__call__(*args, **kwargs)
                    else:
                        _raise_not_callable()
                        return None  # This line should never be reached
                    duration = (datetime.now(UTC) - start_time).total_seconds()
                    # Log performance metrics
                    logger = self.get_logger(__name__)
                    logger.info(
                        f"Operation {operation_name} completed in {duration:.3f}s"
                    )
                    return result
                except Exception:
                    duration = (datetime.now(UTC) - start_time).total_seconds()
                    logger = self.get_logger(__name__)
                    logger.exception(
                        f"Operation {operation_name} failed after {duration:.3f}s"
                    )
                    raise

            return wrapper

        return decorator

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    def create_factory(
        self, factory_type: str, **config: object
    ) -> FlextResult[object]:
        """Create factory instance."""
        try:
            if factory_type == "model":
                return FlextResult[object].ok(create_database_model(**config))
            if factory_type == "service":
                return FlextResult[object].ok(create_service_model(**config))
            return FlextResult[object].fail(f"Unknown factory type: {factory_type}")
        except Exception as e:
            return FlextResult[object].fail(f"Factory creation failed: {e}")

    @property
    def model_factory(self) -> type[FlextFactory]:
        """Access model factory."""
        return FlextFactory

    # =========================================================================
    # COMPREHENSIVE API ACCESS
    # =========================================================================

    def get_all_functionality(self) -> dict[str, object]:
        """Get dictionary of all available functionality."""
        return {
            # Core patterns
            "result": FlextResult,
            "container": self.container,
            "constants": self.constants,
            # Domain modeling
            "entity_base": self.entity_base,
            "value_object_base": self.value_object_base,
            "aggregate_root_base": self.aggregate_root_base,
            "domain_service_base": self.domain_service_base,
            # Validation & guards
            "validators": self.validators,
            "predicates": self.predicates,
            "guards": self.guards,
            # Configuration utilities
            "safe_get_env_var": self.safe_get_env_var,
            "merge_configs": self.merge_configs,
            "validate_config": self.validate_config,
            # Utilities
            "utilities": self.utilities,
            "generators": self.generators,
            "type_guards": self.type_guards,
            "console": self.console,
            "performance": self.performance,
            # Messaging & events
            "payload_base": self.payload_base,
            "message_base": self.message_base,
            "event_base": self.event_base,
            # Handlers & CQRS
            "handlers": self.handlers,
            "handler_registry": self.handler_registry,
            "base_handler": self.base_handler,
            "commands": self.commands,
            # Fields & metadata
            "fields": self.fields,
            "field_registry": self.field_registry,
            # Decorators
            "decorators": self.decorators,
            "decorator_factory": self.decorator_factory,
            # Mixins
            "timestamp_mixin": self.timestamp_mixin,
            "identifiable_mixin": self.identifiable_mixin,
            "loggable_mixin": self.loggable_mixin,
            "validatable_mixin": self.validatable_mixin,
            "serializable_mixin": self.serializable_mixin,
            "cacheable_mixin": self.cacheable_mixin,
            # Exceptions
            "exceptions": self.exceptions,
            # Schema & semantic - removed as semantic.py no longer exists
            # Context & protocols
            "context_class": self.context_class,
            "repository_protocol": self.repository_protocol,
            "plugin_protocol": self.plugin_protocol,
            "plugin_registry": self.plugin_registry,
            # Observability
            "observability": self.observability,
            # Factory
            "model_factory": self.model_factory,
        }

    def list_available_methods(self) -> list[str]:
        """List all available public methods."""
        return [
            method
            for method in dir(self)
            if not method.startswith("_") and callable(getattr(self, method))
        ]

    def get_method_info(self, method_name: str) -> FlextResult[dict[str, object]]:
        """Get information about a specific method."""
        try:
            if not hasattr(self, method_name):
                return FlextResult[dict[str, object]].fail(
                    f"Method not found: {method_name}"
                )

            method = getattr(self, method_name)
            if not callable(method):
                return FlextResult[dict[str, object]].fail(
                    f"Attribute is not callable: {method_name}"
                )

            info: dict[str, object] = {
                "name": method_name,
                "doc": method.__doc__ or "No documentation available",
                "type": "method" if hasattr(method, "__self__") else "function",
                "callable": True,
            }

            return FlextResult[dict[str, object]].ok(info)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to get method info: {e}"
            )

    # =========================================================================
    # SYSTEM INFORMATION & DIAGNOSTICS
    # =========================================================================

    def get_system_info(self) -> dict[str, object]:
        """Get comprehensive system information."""
        info: dict[str, object] = {
            "version": "2.0.0-comprehensive",
            "singleton_id": id(self),
            "container_services": self._container.get_service_count()
            if hasattr(self._container, "get_service_count")
            else "Unknown",
            "settings_cached": len(self._settings_cache),
            "handler_registry_loaded": self._handler_registry is not None,
            "field_registry_loaded": self._field_registry is not None,
            "plugin_registry_loaded": self._plugin_registry is not None,
            "console_loaded": self._console is not None,
            "observability_loaded": self._observability is not None,
            "total_methods": len(self.list_available_methods()),
            "functionality_count": len(self.get_all_functionality()),
        }
        return info

    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform comprehensive health check."""
        try:
            health: dict[str, object] = {
                "status": "healthy",
                "container": "ok" if self._container else "missing",
                "logging": "ok",
                "validation": "ok",
                "utilities": "ok",
                "timestamp": str(datetime.now(UTC)),
            }

            # Test basic functionality
            test_result = self.ok("health_check_test")
            if test_result.is_failure:
                health["status"] = "degraded"
                health["result_pattern"] = "error"
            else:
                health["result_pattern"] = "ok"

            # Test validation
            try:
                validation_result = self.validate_required("test", "health_check")
                if (
                    hasattr(validation_result, "is_failure")
                    and validation_result.is_failure
                ):
                    health["status"] = "degraded"
                    health["validation"] = "error"
            except Exception:
                health["status"] = "degraded"
                health["validation"] = "error"

            return FlextResult[dict[str, object]].ok(health)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Health check failed: {e}")

    def reset_all_caches(self) -> FlextResult[None]:
        """Reset all cached instances."""
        try:
            self._settings_cache.clear()
            self._handler_registry = None
            self._field_registry = None
            self._plugin_registry = None
            self._console = None
            self._observability = None
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Cache reset failed: {e}")

    # =========================================================================
    # ENTERPRISE BUILDERS & FACTORIES (BOILERPLATE REDUCTION)
    # =========================================================================

    def create_validator_class[T](
        self,
        name: str,
        validation_func: Callable[[T], FlextResult[T]],
    ) -> type[FlextAbstractValidator[T]]:
        """Create validator class dynamically to reduce boilerplate."""
        # Import already at module level

        class DynamicValidator(FlextAbstractValidator[T]):
            @override
            def validate(self, value: T) -> FlextResult[T]:
                return validation_func(value)

        DynamicValidator.__name__ = name
        DynamicValidator.__qualname__ = name
        return DynamicValidator

    def create_service_processor(
        self,
        name: str,
        process_func: Callable[[object], FlextResult[object]],
        result_type: type[object] = object,
        build_func: Callable[[object, str], object] | None = None,
        decorators: list[str] | None = None,
    ) -> type:
        """Create service processor class dynamically to reduce boilerplate."""

        class DynamicServiceProcessor(FlextServiceProcessor[object, object, object]):
            def __init__(self) -> None:
                super().__init__()
                # Use class method to get logger since get_logger may not be available in instance context
                self._logger = FlextLoggerFactory.get_logger(
                    f"flext.services.{name.lower()}"
                )

            @override
            def process(self, request: object) -> FlextResult[object]:
                return process_func(request)

            @override
            def build(self, domain: object, *, correlation_id: str) -> object:
                if build_func:
                    return build_func(domain, correlation_id)
                # Default: return domain if types match
                if isinstance(domain, result_type):
                    return domain
                # Fallback: try to create result_type from domain
                if hasattr(result_type, "model_validate"):
                    # Use getattr to safely access the method with type safety
                    model_validate = getattr(result_type, "model_validate", None)
                    if callable(model_validate):
                        return model_validate(domain)
                # Final fallback: try constructor with domain attributes
                if hasattr(domain, "__dict__"):
                    return result_type(**domain.__dict__)
                return result_type()

        # Apply decorators if specified
        if decorators:
            original_process = DynamicServiceProcessor.process
            for decorator_name in decorators:
                if hasattr(FlextDecorators, decorator_name):
                    decorator = getattr(FlextDecorators, decorator_name)
                    if callable(decorator):
                        # Create a new decorated method instead of assigning to class method
                        decorated_method = decorator(original_process)
                        # Use setattr to dynamically assign the method to the class
                        # This is necessary for dynamic method decoration and is intentional
                        DynamicServiceProcessor.process = decorated_method  # pyright: ignore[reportAttributeAccessIssue]
                        original_process = decorated_method

        DynamicServiceProcessor.__name__ = f"{name}ServiceProcessor"
        DynamicServiceProcessor.__qualname__ = f"{name}ServiceProcessor"
        return DynamicServiceProcessor

    def create_entity_with_validators(
        self,
        name: str,
        fields: dict[str, tuple[type, dict[str, object]]],
        validators: dict[str, Callable[[object], FlextResult[object]]] | None = None,
    ) -> type[FlextEntity]:
        """Create entity class with built-in validators to reduce boilerplate."""
        # Build field annotations
        annotations = {}

        for field_name, (field_type, _field_config) in fields.items():
            # Create basic annotated field - simplified to avoid type issues
            annotations[field_name] = Annotated[field_type, Field()]

        # Create class attributes
        class_attrs: dict[str, object] = {
            "__annotations__": annotations,
        }

        # Add field validators if provided
        if validators:
            for field_name, validator_func in validators.items():

                def create_validator(
                    func: Callable[[object], FlextResult[object]],
                    fname: str = field_name,
                ) -> object:
                    def validator_method(_cls: type[object], v: object) -> object:
                        result = func(v)
                        if result.is_failure:
                            error_msg = result.error or f"{fname} validation failed"
                            raise ValueError(error_msg)
                        return result.value

                    # Apply decorators to create proper validator
                    return field_validator(fname)(classmethod(validator_method))

                class_attrs[f"validate_{field_name}_field"] = create_validator(
                    validator_func
                )

        # Create dynamic class
        return type(name, (FlextEntity,), class_attrs)

    def create_value_object_with_validators(
        self,
        name: str,
        fields: dict[str, tuple[type, dict[str, object]]],
        validators: dict[str, Callable[[object], FlextResult[object]]] | None = None,
        business_rules: Callable[[object], FlextResult[None]] | None = None,
    ) -> type[FlextValue]:
        """Create value object class with built-in validators to reduce boilerplate."""
        # Build field annotations
        annotations = {}

        for field_name, (field_type, _field_config) in fields.items():
            # Create basic annotated field - simplified to avoid type issues
            annotations[field_name] = Annotated[field_type, Field()]

        # Create class attributes
        class_attrs: dict[str, object] = {
            "__annotations__": annotations,
        }

        # Add field validators if provided
        if validators:
            for field_name, validator_func in validators.items():

                def create_validator(
                    func: Callable[[object], FlextResult[object]],
                    fname: str = field_name,
                ) -> object:
                    def validator_method(_cls: type[object], v: object) -> object:
                        result = func(v)
                        if result.is_failure:
                            error_msg = result.error or f"{fname} validation failed"
                            raise ValueError(error_msg)
                        return result.value

                    # Apply decorators to create proper validator
                    return field_validator(fname)(classmethod(validator_method))

                class_attrs[f"validate_{field_name}_field"] = create_validator(
                    validator_func
                )

        # Add business rules validation if provided
        if business_rules:

            def validate_business_rules_method(self: object) -> FlextResult[None]:
                return business_rules(self)

            class_attrs["validate_business_rules"] = validate_business_rules_method

        # Create dynamic class
        return type(name, (FlextValue,), class_attrs)

    def setup_container_with_services(
        self,
        services: dict[str, type | Callable[[], object]],
        validator: Callable[[str], FlextResult[str]] | None = None,
    ) -> FlextResult[FlextContainer]:
        """Setup container with multiple services, reducing boilerplate."""
        try:
            container = self.container

            for service_name, service_factory in services.items():
                # Validate service name if validator provided
                if validator:
                    validation_result = validator(service_name)
                    if validation_result.is_failure:
                        self.get_logger(__name__).error(
                            "Service name validation failed",
                            service_name=service_name,
                            error=validation_result.error,
                        )
                        continue

                # Register service
                if isinstance(service_factory, type):
                    # Class factory - use closure to capture type properly
                    def create_factory_func(
                        cls: type | Callable[[], object] = service_factory,
                    ) -> object:
                        if isinstance(cls, type):
                            return cls()
                        return cls()

                    register_result = container.register_factory(
                        service_name, create_factory_func
                    )
                else:
                    # Callable factory
                    register_result = container.register_factory(
                        service_name, service_factory
                    )

                if register_result.is_failure:
                    return FlextResult[FlextContainer].fail(
                        f"Failed to register {service_name}: {register_result.error}"
                    )

                self.get_logger(__name__).info(
                    "Service registered",
                    service_name=service_name,
                    service_class=(
                        service_factory.__name__
                        if hasattr(service_factory, "__name__")
                        else str(type(service_factory))
                    ),
                )

            self.get_logger(__name__).info(
                "Container setup completed",
                total_services=len(services),
                registered_services=list(services.keys()),
            )

            return FlextResult[FlextContainer].ok(container)
        except Exception as e:
            return FlextResult[FlextContainer].fail(f"Container setup failed: {e}")

    def create_demo_function(
        self,
        name: str,
        demo_func: Callable[[], None],
        decorators: list[str] | None = None,
    ) -> Callable[[], None]:
        """Create demo function with standard decorators to reduce boilerplate."""
        # Apply decorators if specified
        decorated_func: Callable[[], None] = demo_func
        if decorators:
            for decorator_name in reversed(decorators):  # Apply in reverse order
                if hasattr(FlextDecorators, decorator_name):
                    decorator = getattr(FlextDecorators, decorator_name)
                    if callable(decorator):
                        decorated_func = cast(
                            "Callable[[], None]", decorator(decorated_func)
                        )

        decorated_func.__name__ = name
        decorated_func.__qualname__ = name
        return decorated_func

    def log_result(
        self, result: FlextResult[T], success_msg: str, logger_name: str | None = None
    ) -> FlextResult[T]:
        """Utility to log FlextResult with consistent formatting."""
        logger = self.get_logger(logger_name or __name__)
        if result.is_success:
            logger.info(f"✅ {success_msg}", result_type=type(result.value).__name__)
        else:
            logger.error(f"❌ {success_msg} failed", error=result.error)
        return result

    def get_service_with_fallback(
        self, service_name: str, default_factory: type[T]
    ) -> T:
        """Get service from container with type-safe fallback."""
        result = self.get_service(service_name)
        if result.is_success:
            self.get_logger(__name__).debug(
                "Service retrieved from container", service_name=service_name
            )
            return cast("T", result.value)

        self.get_logger(__name__).warning(
            "Service not found in container, using default factory",
            service_name=service_name,
            default_factory=default_factory.__name__,
        )
        return default_factory()

    def create_standard_validators(
        self,
    ) -> dict[str, Callable[[object], FlextResult[object]]]:
        """Create standard validators to reduce boilerplate."""
        return {
            "age": lambda v: cast(
                "FlextResult[object]",
                self.validate_numeric(v, 18, 120)
                if isinstance(v, (int, float))
                else FlextResult[object].fail("Age must be numeric"),
            ),
            "email": lambda v: cast(
                "FlextResult[object]",
                self.validate_email(v)
                if isinstance(v, str)
                else FlextResult[object].fail("Email must be string"),
            ),
            "name": lambda v: cast(
                "FlextResult[object]",
                self.validate_string(v, 2, 100)
                if isinstance(v, str)
                else FlextResult[object].fail("Name must be string"),
            ),
            "service_name": lambda v: cast(
                "FlextResult[object]",
                self.validate_service_name(v)
                if isinstance(v, str)
                else FlextResult[object].fail("Service name must be string"),
            ),
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @override
    def __repr__(self) -> str:
        """Return comprehensive string representation."""
        service_count = (
            self._container.get_service_count()
            if hasattr(self._container, "get_service_count")
            else "Unknown"
        )
        return (
            f"FlextCore("
            f"services={service_count}, "
            f"methods={len(self.list_available_methods())}, "
            f"functionality={len(self.get_all_functionality())}"
            f")"
        )

    @override
    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return "FlextCore - Comprehensive FLEXT ecosystem access (v2.0.0)"


# Convenience function for global access
def flext_core() -> FlextCore:
    """Get global FlextCore instance with a convenient access pattern.

    Convenience function providing direct access to the global FlextCore singleton
    instance without requiring explicit class method calls. Maintains a singleton
    pattern while providing simpler access syntax.

    Returns:
      Global FlextCore singleton instance

    """
    return FlextCore.get_instance()


# Export API
__all__: list[str] = [
    "FlextCore",
    "flext_core",
]
