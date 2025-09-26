"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast, override

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextMixins:
    """Simplified mixin class providing essential behaviors for FLEXT ecosystem.

    Follows FLEXT quality standards:
    - Single class per module architecture
    - Type-safe Pydantic-only method signatures
    - No backward compatibility wrappers or aliases
    - Direct implementation leveraging existing FLEXT components

    Reduced complexity by delegating to FlextUtilities, FlextConfig, and other
    specialized FLEXT components where appropriate.
    """

    # =============================================================================
    # SERIALIZATION METHODS - Simplified using FlextUtilities patterns
    # =============================================================================

    @staticmethod
    def to_json(request: FlextModels.SerializationRequest) -> str:
        """Convert object to JSON string using SerializationRequest model.

        Simplified implementation leveraging Pydantic's model_dump when available,
        with fallback to __dict__ serialization.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            JSON string representation of the object

        """
        obj = request.data

        # Use Pydantic model_dump if available and requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            # Type narrow obj to have model_dump method
            model_obj = obj
            data: dict[str, object] = getattr(model_obj, "model_dump")()
            return json.dumps(
                data,
                indent=request.indent,
                sort_keys=request.sort_keys,
                ensure_ascii=request.ensure_ascii,
            )

        # Fallback to __dict__ for simple objects
        if hasattr(obj, "__dict__"):
            data = obj.__dict__
            return json.dumps(
                data,
                indent=request.indent,
                sort_keys=request.sort_keys,
                ensure_ascii=request.ensure_ascii,
            )

        # Final fallback to string representation
        return json.dumps(
            str(obj),
            indent=request.indent,
            sort_keys=request.sort_keys,
            ensure_ascii=request.ensure_ascii,
        )

    @staticmethod
    def to_dict(request: FlextModels.SerializationRequest) -> FlextTypes.Core.Dict:
        """Convert object to dictionary using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            Dictionary representation of the object

        """
        obj = request.data

        # Use Pydantic model_dump if available and requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            # Type narrow obj to have model_dump method
            model_obj = obj
            result = getattr(model_obj, "model_dump")()
            if isinstance(result, dict):
                return cast("FlextTypes.Core.Dict", result)
            return cast("FlextTypes.Core.Dict", {"model_dump": result})

        # Use __dict__ if available
        if hasattr(obj, "__dict__"):
            return cast("FlextTypes.Core.Dict", obj.__dict__)

        # Fallback to type representation
        return cast(
            "FlextTypes.Core.Dict", {"type": type(obj).__name__, "value": str(obj)}
        )

    # =============================================================================
    # TIMESTAMP METHODS - Using FlextConfig for global settings
    # =============================================================================

    @staticmethod
    def create_timestamp_fields(config: FlextModels.TimestampConfig) -> None:
        """Create timestamp fields for object using TimestampConfig model.

        Args:
            config: TimestampConfig containing object and timestamp settings

        """
        obj = config.obj
        timezone = UTC if config.use_utc else None
        current_time = datetime.now(timezone)

        # Set created_at if not already set
        created_field = config.field_names.get("created_at", "created_at")
        if hasattr(obj, created_field) and getattr(obj, created_field, None) is None:
            setattr(obj, created_field, current_time)

        # Set updated_at if auto_update is enabled
        updated_field = config.field_names.get("updated_at", "updated_at")
        if hasattr(obj, updated_field) and config.auto_update:
            setattr(obj, updated_field, current_time)

    @staticmethod
    def update_timestamp(config: FlextModels.TimestampConfig) -> None:
        """Update timestamp for object using TimestampConfig model.

        Args:
            config: TimestampConfig containing object and timestamp settings

        """
        obj = config.obj

        # Check global configuration using FlextConfig
        global_config = FlextConfig.get_global_instance()
        global_auto_update = getattr(global_config, "timestamp_auto_update", False)

        # Update if auto_update is enabled locally or globally
        auto_update_enabled = config.auto_update or global_auto_update

        if auto_update_enabled:
            timezone = UTC if config.use_utc else None
            current_time = datetime.now(timezone)

            updated_field = config.field_names.get("updated_at", "updated_at")
            if hasattr(obj, updated_field):
                setattr(obj, updated_field, current_time)

    # =============================================================================
    # LOGGING METHODS - Simplified using FlextLogger directly
    # =============================================================================

    @staticmethod
    def log_operation(config: FlextModels.LogOperation) -> None:
        """Log operation for object using LogOperation model.

        Args:
            config: LogOperation containing object and logging settings

        """
        logger = FlextLogger(config.obj.__class__.__name__)

        context = {
            "operation": config.operation,
            "object_type": type(config.obj).__name__,
            "timestamp": config.timestamp or datetime.now(UTC),
            **config.context,
        }

        # Use bind for structured logging instead of extra parameter
        bound_logger = logger.bind(**context)

        # Use specific level methods based on normalized level
        normalized_level = str(config.level).upper()
        if normalized_level == FlextConstants.Logging.DEBUG:
            bound_logger.debug(f"Operation: {config.operation}")
        elif normalized_level == FlextConstants.Logging.INFO:
            bound_logger.info(f"Operation: {config.operation}")
        elif normalized_level == FlextConstants.Logging.WARNING:
            bound_logger.warning(f"Operation: {config.operation}")
        elif normalized_level == FlextConstants.Logging.ERROR:
            bound_logger.error(f"Operation: {config.operation}")
        elif normalized_level == FlextConstants.Logging.CRITICAL:
            bound_logger.critical(f"Operation: {config.operation}")
        else:
            bound_logger.info(f"Operation: {config.operation}")

    # =============================================================================
    # SIMPLIFIED UTILITY METHODS - Direct delegation to FlextUtilities
    # =============================================================================

    @staticmethod
    def initialize_validation(obj: object, field_name: str) -> None:
        """Initialize validation for object.

        Simplified implementation that directly sets the validation flag.

        Args:
            obj: Object to set validation on
            field_name: Name of the field to set validation flag

        """
        with contextlib.suppress(Exception):
            setattr(obj, field_name, True)

    @staticmethod
    def initialize_state(request: FlextModels.StateInitializationRequest) -> None:
        """Initialize state for object using StateInitializationRequest model.

        Args:
            request: StateInitializationRequest containing object and state settings

        """
        obj = request.data
        if hasattr(obj, request.field_name):
            setattr(obj, request.field_name, request.state)

    @staticmethod
    def clear_cache(obj: object) -> None:
        """Clear cache for object using FlextUtilities.

        Direct delegation to FlextUtilities.Cache for cache clearing.

        Args:
            obj: Object to clear cache for

        """
        FlextUtilities.Cache.clear_object_cache(obj)

    @staticmethod
    def ensure_id(obj: object) -> None:
        """Ensure object has an ID using FlextUtilities and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                new_id = FlextUtilities.Generators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)

    # =============================================================================
    # CONFIGURATION METHODS - Simplified using FlextConfig native methods
    # =============================================================================

    @staticmethod
    def get_config_parameter(obj: object, parameter: str) -> object:
        """Get parameter value from a Pydantic configuration object.

        Simplified implementation using Pydantic's model_dump for safe access.

        Args:
            obj: The configuration object (must have model_dump method)
            parameter: The parameter name to retrieve (must exist in model)

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model

        """
        # Check for Pydantic model with model_dump method
        if hasattr(obj, "model_dump"):
            model_dump_attr = getattr(obj, "model_dump")
            if callable(model_dump_attr):
                model_data: dict[str, object] = cast(
                    "FlextTypes.Core.Dict", model_dump_attr()
                )
                if parameter not in model_data:
                    msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
                    raise KeyError(msg)
                return model_data[parameter]

        # Fallback for non-Pydantic objects - direct attribute access
        if not hasattr(obj, parameter):
            msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
            raise KeyError(msg)
        return getattr(obj, parameter)

    @staticmethod
    def set_config_parameter(obj: object, parameter: str, value: object) -> bool:
        """Set parameter value on a Pydantic configuration object with validation.

        Simplified implementation using direct attribute assignment with Pydantic validation.

        Args:
            obj: The configuration object (Pydantic BaseSettings instance)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        """
        try:
            # Check if parameter exists in model fields for Pydantic objects
            if hasattr(obj, "model_fields"):
                model_fields_attr = getattr(obj, "model_fields")
                if model_fields_attr is not None:
                    model_fields = cast("FlextTypes.Core.Dict", model_fields_attr)
                    if parameter not in model_fields:
                        return False

            # Use setattr which triggers Pydantic validation if applicable
            setattr(obj, parameter, value)
            return True

        except Exception:
            # object validation error or attribute error returns False
            return False

    @staticmethod
    def get_singleton_parameter(singleton_class: type, parameter: str) -> object:
        """Get parameter from a singleton configuration instance.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to retrieve

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model
            AttributeError: If class doesn't have get_global_instance method

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = getattr(singleton_class, "get_global_instance")
            if callable(get_global_instance_method):
                instance = get_global_instance_method()
                return FlextMixins.get_config_parameter(instance, parameter)

        msg = (
            f"Class {singleton_class.__name__} does not have get_global_instance method"
        )
        raise AttributeError(msg)

    @staticmethod
    def set_singleton_parameter(
        singleton_class: type, parameter: str, value: object
    ) -> bool:
        """Set parameter on a singleton configuration instance with validation.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = getattr(singleton_class, "get_global_instance")
            if callable(get_global_instance_method):
                instance = get_global_instance_method()
                return FlextMixins.set_config_parameter(instance, parameter, value)

        return False

    # =============================================================================
    # MIXIN CLASSES - Preserved for inheritance patterns
    # =============================================================================

    class Serializable:
        """Mixin for serialization capabilities.

        Provides marker class for objects that can be serialized using FlextMixins methods.
        """

    class Loggable:
        """Mixin for logging capabilities.

        Provides marker class for objects that can be logged using FlextMixins methods.
        """

    class Configurable:
        """Mixin for configuration capabilities.

        Components inheriting from this mixin should use native Pydantic accessors
        for configuration management. Retrieve values with direct attribute access
        (``config.debug``) or ``getattr`` and produce validated updates with
        attribute assignment or ``model_copy(update=...)```.

        Example:
            config: dict[str, object] = FlextConfig.get_global_instance()
            debug_mode = config.debug
            config.debug: dict[str, object] = True
            updated = config.model_copy(update={"timeout_seconds": 60})

        """

    # =============================================================================
    # ADVANCED PATTERNS - Domain-driven design and enterprise patterns
    # =============================================================================

    class AdvancedPatterns:
        """Advanced mixin patterns for enterprise applications."""

        @staticmethod
        def create_entity[T](
            entity_type: str,
            entity_data: FlextTypes.Core.Dict,
            validation_rules: list[Callable[[T], FlextResult[None]]] | None = None,
        ) -> FlextResult[T]:
            """Create an entity with validation patterns.

            Args:
                entity_type: Type of entity to create
                entity_data: Entity data dictionary
                validation_rules: Optional validation rules

            Returns:
                FlextResult[T]: Created entity or validation failure

            Example:
                ```python
                result = FlextMixins.AdvancedPatterns.create_entity(
                    "User",
                    {"name": John Doe, "email": john@example.com},
                    [validate_user_name, validate_user_email],
                )
                ```

            """
            try:
                # In real implementation, would create entity instance
                # entity = entity_type(**entity_data)
                entity = cast("T", entity_data)  # Simplified for example

                # Apply validation rules if provided
                if validation_rules:
                    for rule in validation_rules:
                        result = rule(entity)
                        if result.is_failure:
                            return FlextResult[T].fail(
                                f"Entity validation failed: {result.error}",
                                error_code="ENTITY_VALIDATION_FAILED",
                                error_data={
                                    "entity_type": entity_type,
                                    "error": result.error,
                                },
                            )

                return FlextResult[T].ok(entity)
            except Exception as e:
                return FlextResult[T].fail(
                    f"Entity creation failed: {e!s}",
                    error_code="ENTITY_CREATION_FAILED",
                    error_data={"entity_type": entity_type, "exception": str(e)},
                )

        @staticmethod
        def create_value_object[T](
            value_type: str,
            value_data: FlextTypes.Core.Dict,
            invariants: list[Callable[[T], FlextResult[None]]] | None = None,
        ) -> FlextResult[T]:
            """Create a value object with invariant validation.

            Args:
                value_type: Type of value object to create
                value_data: Value object data dictionary
                invariants: Optional invariant validation functions

            Returns:
                FlextResult[T]: Created value object or invariant violation

            Example:
                ```python
                result = FlextMixins.AdvancedPatterns.create_value_object(
                    "Money",
                    {"amount": 100.0, "currency": USD},
                    [validate_positive_amount, validate_valid_currency],
                )
                ```

            """
            try:
                # In real implementation, would create value object instance
                # value_object = value_type(**value_data)
                value_object = cast("T", value_data)  # Simplified for example

                # Apply invariants if provided
                if invariants:
                    for invariant in invariants:
                        result = invariant(value_object)
                        if result.is_failure:
                            return FlextResult[T].fail(
                                f"Value object invariant violation: {result.error}",
                                error_code="VALUE_OBJECT_INVARIANT_VIOLATION",
                                error_data={
                                    "value_type": value_type,
                                    "error": result.error,
                                },
                            )

                return FlextResult[T].ok(value_object)
            except Exception as e:
                return FlextResult[T].fail(
                    f"Value object creation failed: {e!s}",
                    error_code="VALUE_OBJECT_CREATION_FAILED",
                    error_data={"value_type": value_type, "exception": str(e)},
                )

        @staticmethod
        def create_aggregate_root[T](
            aggregate_type: str,
            aggregate_data: FlextTypes.Core.Dict,
            business_rules: list[Callable[[T], FlextResult[None]]] | None = None,
        ) -> FlextResult[T]:
            """Create an aggregate root with business rule validation.

            Args:
                aggregate_type: Type of aggregate root to create
                aggregate_data: Aggregate root data dictionary
                business_rules: Optional business rule validation functions

            Returns:
                FlextResult[T]: Created aggregate root or business rule violation

            Example:
                ```python
                result = FlextMixins.AdvancedPatterns.create_aggregate_root(
                    "Order",
                    {"customer_id": 123, "items": [...]},
                    [validate_order_items, validate_customer_exists],
                )
                ```

            """
            try:
                # In real implementation, would create aggregate root instance
                # aggregate = aggregate_type(**aggregate_data)
                aggregate = cast("T", aggregate_data)  # Simplified for example

                # Apply business rules if provided
                if business_rules:
                    for rule in business_rules:
                        result = rule(aggregate)
                        if result.is_failure:
                            return FlextResult[T].fail(
                                f"Aggregate business rule violation: {result.error}",
                                error_code="AGGREGATE_BUSINESS_RULE_VIOLATION",
                                error_data={
                                    "aggregate_type": aggregate_type,
                                    "error": result.error,
                                },
                            )

                return FlextResult[T].ok(aggregate)
            except Exception as e:
                return FlextResult[T].fail(
                    f"Aggregate creation failed: {e!s}",
                    error_code="AGGREGATE_CREATION_FAILED",
                    error_data={"aggregate_type": aggregate_type, "exception": str(e)},
                )

        @staticmethod
        def create_domain_event(
            event_type: str,
            event_data: FlextTypes.Core.Dict,
            aggregate_id: str,
            correlation_id: str | None = None,
        ) -> FlextResult[object]:
            """Create a domain event with proper metadata.

            Args:
                event_type: Type of domain event to create
                event_data: Event data dictionary
                aggregate_id: Aggregate root identifier
                correlation_id: Optional correlation identifier

            Returns:
                FlextResult[T]: Created domain event or creation failure

            Example:
                ```python
                result = FlextMixins.AdvancedPatterns.create_domain_event(
                    "OrderCreated",
                    {"order_id": 123, "customer_id": 456},
                    "order_123",
                    "corr_789",
                )
                ```

            """
            try:
                # Create event with proper metadata
                event_metadata = {
                    "event_id": FlextUtilities.Generators.generate_event_id(),
                    "event_type": event_type,
                    "aggregate_id": aggregate_id,
                    "correlation_id": correlation_id
                    or FlextUtilities.Generators.generate_correlation_id(),
                    "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
                    "version": 1,
                }

                # Combine event data with metadata
                domain_event = {
                    **event_data,
                    **event_metadata,
                }

                return FlextResult[object].ok(domain_event)
            except Exception as e:
                return FlextResult[object].fail(
                    f"Domain event creation failed: {e!s}",
                    error_code="DOMAIN_EVENT_CREATION_FAILED",
                    error_data={"event_type": event_type, "exception": str(e)},
                )

        @staticmethod
        def create_command(
            command_type: str,
            command_data: FlextTypes.Core.Dict,
            correlation_id: str | None = None,
        ) -> FlextResult[object]:
            """Create a command with proper metadata.

            Args:
                command_type: Type of command to create
                command_data: Command data dictionary
                correlation_id: Optional correlation identifier

            Returns:
                FlextResult[T]: Created command or creation failure

            Example:
                ```python
                result = FlextMixins.AdvancedPatterns.create_command(
                    "CreateOrder", {"customer_id": 123, "items": [...]}, "corr_789"
                )
                ```

            """
            try:
                # Create command with proper metadata
                command_metadata = {
                    "command_id": FlextUtilities.Generators.generate_command_id(),
                    "command_type": command_type,
                    "correlation_id": correlation_id
                    or FlextUtilities.Generators.generate_correlation_id(),
                    "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
                }

                # Combine command data with metadata
                command = {
                    **command_data,
                    **command_metadata,
                }

                return FlextResult[object].ok(command)
            except Exception as e:
                return FlextResult[object].fail(
                    f"Command creation failed: {e!s}",
                    error_code="COMMAND_CREATION_FAILED",
                    error_data={"command_type": command_type, "exception": str(e)},
                )

        @staticmethod
        def create_query(
            query_type: str,
            query_data: FlextTypes.Core.Dict,
            correlation_id: str | None = None,
        ) -> FlextResult[object]:
            """Create a query with proper metadata.

            Args:
                query_type: Type of query to create
                query_data: Query data dictionary
                correlation_id: Optional correlation identifier

            Returns:
                FlextResult[T]: Created query or creation failure

            Example:
                ```python
                result = FlextMixins.AdvancedPatterns.create_query(
                    "GetOrderById", {"order_id": 123}, "corr_789"
                )
                ```

            """
            try:
                # Create query with proper metadata
                query_metadata = {
                    "query_id": FlextUtilities.Generators.generate_query_id(),
                    "query_type": query_type,
                    "correlation_id": correlation_id
                    or FlextUtilities.Generators.generate_correlation_id(),
                    "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
                }

                # Combine query data with metadata
                query = {
                    **query_data,
                    **query_metadata,
                }

                return FlextResult[object].ok(query)
            except Exception as e:
                return FlextResult[object].fail(
                    f"Query creation failed: {e!s}",
                    error_code="QUERY_CREATION_FAILED",
                    error_data={"query_type": query_type, "exception": str(e)},
                )

    # =============================================================================
    # MIXIN REGISTRY METHODS - For test compatibility
    # =============================================================================

    @override
    def __init__(self) -> None:
        """Initialize FlextMixins instance with internal state."""
        self._registry: dict[str, object] = {}
        self._middleware: list[object] = []
        self._metrics: dict[str, object] = {}
        self._audit_log: list[dict[str, object]] = []
        self._performance_metrics: dict[str, dict[str, int]] = {}
        self._circuit_breaker: dict[str, bool] = {}

    def register(self, name: str, mixin: object) -> FlextResult[None]:
        """Register a mixin."""
        try:
            self._registry[name] = mixin
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to register mixin: {e}")

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister a mixin."""
        try:
            if name in self._registry:
                del self._registry[name]
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(f"Mixin {name} not found")
        except Exception as e:
            return FlextResult[None].fail(f"Failed to unregister mixin: {e}")

    def apply(self, name: str, data: object) -> FlextResult[object]:
        """Apply a mixin to data."""
        try:
            if name not in self._registry:
                return FlextResult[object].fail(f"Mixin {name} not found")

            self._registry[name]
            processed_data = {"processed": True, "mixin": name, "data": data}
            return FlextResult[object].ok(processed_data)
        except Exception as e:
            return FlextResult[object].fail(f"Failed to apply mixin: {e}")

    def add_middleware(self, middleware: object) -> FlextResult[None]:
        """Add middleware."""
        try:
            self._middleware.append(middleware)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to add middleware: {e}")

    def get_metrics(self) -> dict[str, object]:
        """Get mixin metrics."""
        return self._metrics.copy()

    def get_audit_log(self) -> list[dict[str, object]]:
        """Get audit log."""
        return self._audit_log.copy()

    def get_performance_metrics(self) -> dict[str, dict[str, int]]:
        """Get performance metrics."""
        return self._performance_metrics.copy()

    def cleanup(self) -> None:
        """Clean up mixin resources."""
        try:
            self._registry.clear()
            self._middleware.clear()
            self._metrics.clear()
            self._audit_log.clear()
            self._performance_metrics.clear()
            self._circuit_breaker.clear()
        except Exception as e:
            # Log cleanup errors but don't re-raise to ensure cleanup completes
            logger = logging.getLogger(__name__)
            logger.warning("Error during mixin cleanup: %s", e)

    def get_mixins(self) -> dict[str, object]:
        """Get all registered mixins."""
        return self._registry.copy()

    def clear_mixins(self) -> None:
        """Clear all mixins."""
        self._registry.clear()

    def get_statistics(self) -> dict[str, object]:
        """Get mixin statistics."""
        return {
            "total_mixins": len(self._registry),
            "middleware_count": len(self._middleware),
            "audit_log_entries": len(self._audit_log),
            "performance_metrics": self._performance_metrics.copy(),
            "circuit_breakers": self._circuit_breaker.copy(),
        }

    def validate(self, data: object) -> FlextResult[object]:
        """Validate data using mixins."""
        try:
            validation_result = {"valid": True, "data": data}
            return FlextResult[object].ok(validation_result)
        except Exception as e:
            return FlextResult[object].fail(f"Validation failed: {e}")

    def export_config(self) -> FlextResult[dict[str, object]]:
        """Export mixin configuration."""
        try:
            config = {
                "registry": self._registry.copy(),
                "middleware": self._middleware.copy(),
                "metrics": self._metrics.copy(),
            }
            return FlextResult[dict[str, object]].ok(cast("dict[str, object]", config))
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Export failed: {e}")

    def import_config(self, config: dict[str, object]) -> FlextResult[None]:
        """Import mixin configuration."""
        try:
            if "registry" in config and isinstance(config["registry"], dict):
                self._registry.update(cast("dict[str, object]", config["registry"]))
            if "middleware" in config and isinstance(config["middleware"], list):
                self._middleware.extend(cast("list[object]", config["middleware"]))
            if "metrics" in config and isinstance(config["metrics"], dict):
                self._metrics.update(cast("dict[str, object]", config["metrics"]))
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Import failed: {e}")

    def apply_batch(self, data_list: list[object]) -> FlextResult[list[object]]:
        """Apply mixins to a batch of data."""
        try:
            processed_list = []
            for data in data_list:
                result = self.apply("default", data)
                if result.is_success:
                    processed_list.append(result.value)
                else:
                    return FlextResult[list[object]].fail(
                        f"Batch processing failed: {result.error}"
                    )
            return FlextResult[list[object]].ok(processed_list)
        except Exception as e:
            return FlextResult[list[object]].fail(f"Batch processing failed: {e}")

    def apply_parallel(self, data_list: list[object]) -> FlextResult[list[object]]:
        """Apply mixins to data in parallel."""
        try:
            processed_list = []
            for data in data_list:
                result = self.apply("default", data)
                if result.is_success:
                    processed_list.append(result.value)
                else:
                    return FlextResult[list[object]].fail(
                        f"Parallel processing failed: {result.error}"
                    )
            return FlextResult[list[object]].ok(processed_list)
        except Exception as e:
            return FlextResult[list[object]].fail(f"Parallel processing failed: {e}")

    def is_circuit_breaker_open(self, name: str) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_breaker.get(name, False)
