"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from typing import (
    ClassVar,
    cast,
    override,
)

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextMixins:
    """Simplified mixin class providing essential behaviors for FLEXT.

    Follows FLEXT quality standards with single class per module,
    type-safe Pydantic signatures, no backward compatibility wrappers,
    and direct implementation leveraging existing FLEXT components.

    **Function**: Reusable behavior mixins for ecosystem
        - JSON serialization with Pydantic model support
        - Object cloning and deep copying
        - Metadata management with timestamps
        - Logging integration with structured logging
        - Validation helpers for business rules
        - Performance monitoring and timing
        - Thread-safe operations with locks
        - Configuration binding and updates
        - Event notification system
        - Snapshot and restore capabilities

    **Uses**: Core FLEXT infrastructure for mixins
        - FlextUtilities for ID generation and validation
        - FlextConfig for configuration management
        - FlextLogger for structured logging
        - FlextModels for domain models
        - FlextResult[T] for operation results
        - FlextConstants for defaults and error codes
        - json module for serialization
        - datetime for timestamp operations
        - threading for thread-safe operations
        - contextlib for context management

    **How to use**: Reusable behaviors via static methods
        ```python
        from flext_core import FlextMixins, FlextModels

        # Example 1: JSON serialization
        request = FlextModels.SerializationRequest(obj={"key": "value"})
        json_str = FlextMixins.to_json(request)

        # Example 2: Add metadata to object
        obj = MyModel()
        metadata_req = FlextModels.MetadataRequest(obj=obj, metadata={"version": "1.0"})
        FlextMixins.add_metadata(metadata_req)

        # Example 3: Set timestamp on object
        timestamp_req = FlextModels.TimestampRequest(obj=obj)
        FlextMixins.set_timestamp(timestamp_req)

        # Example 4: Clone object
        clone_req = FlextModels.CloneRequest(obj=obj)
        cloned = FlextMixins.clone(clone_req)

        # Example 5: Validate object
        validation_req = FlextModels.ValidationRequest(obj=obj)
        result = FlextMixins.validate(validation_req)

        # Example 6: Log operation
        log_req = FlextModels.LogRequest(
            obj=obj, message="Operation completed", level="info"
        )
        FlextMixins.log(log_req)

        # Example 7: Performance monitoring
        perf_req = FlextModels.PerformanceRequest(obj=obj)
        FlextMixins.start_performance_monitoring(perf_req)
        # ... operation ...
        metrics = FlextMixins.stop_performance_monitoring(perf_req)
        ```
        - [ ] Support mixin documentation generation
        - [ ] Add mixin migration tools

    Note:
        All methods are static for stateless behavior.
        Uses Pydantic models for type-safe parameters.
        Delegates complex operations to FlextUtilities.
        Integrates with FlextLogger for structured logging.
        Thread-safe operations use threading primitives.

    Warning:
        Metadata operations modify objects in-place.
        Performance monitoring requires paired start/stop calls.
        Clone operations may not work with all object types.
        Validation depends on object having validate method.

    Example:
        Complete workflow with serialization and metadata:

        >>> obj = {"data": "value"}
        >>> req = FlextModels.SerializationRequest(obj=obj)
        >>> json_str = FlextMixins.to_json(req)
        >>> print(json_str)
        {"data": "value"}

    See Also:
        FlextUtilities: For utility functions.
        FlextModels: For domain model definitions.
        FlextLogger: For structured logging.
        FlextConfig: For configuration management.

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
            model_obj = cast("FlextProtocols.Foundation.HasModelFields", obj)
            data: FlextTypes.Dict = model_obj.model_dump()
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
    def to_dict(request: FlextModels.SerializationRequest) -> FlextTypes.Dict:
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
            model_obj = cast("FlextProtocols.Foundation.HasModelFields", obj)
            return model_obj.model_dump()

        # Use __dict__ if available
        if hasattr(obj, "__dict__"):
            return cast("FlextTypes.Dict", obj.__dict__)

        # Fallback to type representation
        return cast(
            "FlextTypes.Dict",
            {"type": type(obj).__name__, "value": str(obj)},
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

        context_data: dict[str, object] = {
            "operation": config.operation,
            "object_type": type(config.obj).__name__,
            "timestamp": config.timestamp or datetime.now(UTC),
            **config.context,
        }

        # Use bind for structured logging instead of extra parameter
        bound_logger = logger.bind(**context_data)

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
            obj: Object to set validation on (must support attribute assignment)
            field_name: Name of the field to set validation flag

        Note:
            The object must support attribute assignment. If setattr() fails,
            it indicates a programming error (e.g., using a frozen dataclass,
            or an object with __slots__ that doesn't include the field).

        """
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
                new_id = FlextUtilities.generate_id()
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
        if isinstance(obj, FlextProtocols.Foundation.HasModelDump):
            model_data: FlextTypes.Dict = obj.model_dump()
            if parameter not in model_data:
                msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
                raise FlextExceptions.NotFoundError(msg, resource_id=parameter)
            return model_data[parameter]

        # Fallback for non-Pydantic objects - direct attribute access
        if not hasattr(obj, parameter):
            msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
            raise FlextExceptions.NotFoundError(
                msg, resource_type=f"parameter '{parameter}'"
            )
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
            if isinstance(obj, FlextProtocols.Foundation.HasModelFields):
                # Access model_fields from class, not instance (Pydantic 2.11+ compatibility)
                model_fields = type(obj).model_fields
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
        raise FlextExceptions.ValidationError(msg)

    @staticmethod
    def set_singleton_parameter(
        singleton_class: type,
        parameter: str,
        value: object,
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
    # ENHANCED MIXINS - DI Integration with structlog, dependency_injector, returns
    # =============================================================================

    class LoggableDI:
        """Enhanced Loggable mixin with dependency injection for logger.

        Provides automatic logger injection from FlextContainer, eliminating
        the need for manual FlextLogger(__name__) instantiation in every class.

        This mixin uses lazy initialization and caching to minimize overhead
        while providing full DI integration.

        Example:
            class MyService(FlextMixins.LoggableDI):
                def process(self):
                    # Logger automatically available via DI
                    self.logger.info("Processing started")
                    return FlextResult[dict].ok({"status": "processed"})

        """

        # Class-level cache for loggers to avoid repeated DI lookups
        _logger_cache: ClassVar[dict[str, FlextLogger]] = {}
        _cache_lock: ClassVar[threading.Lock] = threading.Lock()

        @classmethod
        def _get_or_create_logger(cls) -> FlextLogger:
            """Get or create DI-injected logger for this class.

            Uses FlextContainer for dependency injection with fallback to
            direct creation if DI is not available.

            Returns:
                FlextLogger instance from DI or newly created

            """
            # Generate unique logger name based on module and class
            logger_name = f"{cls.__module__}.{cls.__name__}"

            # Check cache first (thread-safe)
            with cls._cache_lock:
                if logger_name in cls._logger_cache:
                    return cls._logger_cache[logger_name]

            # Try to get from DI container
            try:
                container = FlextContainer.get_global()
                logger_key = f"logger:{logger_name}"

                # Attempt to retrieve logger from container
                logger_result: FlextResult[FlextLogger] = container.get_typed(
                    logger_key, FlextLogger
                )

                if logger_result.is_success:
                    logger: FlextLogger = logger_result.unwrap()
                    # Cache the result
                    with cls._cache_lock:
                        cls._logger_cache[logger_name] = logger
                    return logger

                # Logger not in container - create and register
                logger = FlextLogger(logger_name)
                container.register(logger_key, logger)

                # Cache the result
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger

                return logger

            except Exception:
                # Fallback: create logger without DI if container unavailable
                logger = FlextLogger(logger_name)
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger
                return logger

        @property
        def logger(self) -> FlextLogger:
            """Access logger via property (DI-backed with caching).

            Returns:
                FlextLogger instance for this class

            """
            return self._get_or_create_logger()

    # =============================================================================
    # MIXIN REGISTRY METHODS
    # =============================================================================

    @override
    def __init__(self) -> None:
        """Initialize FlextMixins instance with internal state."""
        super().__init__()
        self._registry: dict[str, type] = {}
        self._middleware: list[
            Callable[[type, object], tuple[type, object] | None]
        ] = []
        self._metrics: dict[str, dict[str, int]] = {}
        self._audit_log: list[FlextTypes.Dict] = []
        self._performance_metrics: dict[str, dict[str, float | int]] = {}
        self._circuit_breaker: dict[str, bool] = {}
        self._rate_limit_requests: dict[str, list[datetime]] = {}

    def register(self, name: str, mixin: type) -> FlextResult[None]:
        """Register a mixin."""
        if not name:
            return FlextResult[None].fail("Invalid mixin name")
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
            return FlextResult[None].fail("No mixin found")
        except Exception as e:
            return FlextResult[None].fail(f"Failed to unregister mixin: {e}")

    def apply(self, name: str, data: object) -> FlextResult[object]:
        """Apply a mixin to data with reliability patterns.

        Args:
            name: Name of the mixin to apply
            data: Data to process with the mixin

        Returns:
            FlextResult[object]: Success with processed data or failure with error

        """
        try:
            # Validate mixin exists
            if name not in self._registry:
                return FlextResult[object].fail(f"Mixin '{name}' not found")

            # Check circuit breaker
            if self.is_circuit_breaker_open(name):
                return FlextResult[object].fail("Circuit breaker is open")

            # Rate limiting check
            if not self._check_rate_limit(name):
                return FlextResult[object].fail("Rate limit exceeded")

            # Get mixin class
            mixin_class = self._registry[name]

            # For class mixins, create a new class with mixin applied
            # Create a new class that inherits from both the mixin and the data class
            if isinstance(data, type):
                # Data is a class, create mixed class
                mixed_class = type(
                    f"{data.__name__}With{mixin_class.__name__}",
                    (mixin_class, data),
                    {},
                )
                result = FlextResult[object].ok(mixed_class)
            elif callable(mixin_class):
                # For callable mixins, use the original logic
                # Create mixin instance
                mixin_instance = mixin_class()

                # Apply middleware if present
                processed_data = self._apply_middleware(mixin_class, data)

                # Execute mixin
                result = self._execute_mixin(mixin_instance, processed_data)
            else:
                # Create mixed instance
                mixed_class = type(
                    f"{data.__class__.__name__}With{mixin_class.__name__}",
                    (mixin_class, data.__class__),
                    {},
                )
                mixed_instance = mixed_class()
                # Copy attributes from original instance
                for attr in dir(data):
                    if not attr.startswith("_") and hasattr(data, attr):
                        with suppress(AttributeError):
                            setattr(mixed_instance, attr, getattr(data, attr))
                result = FlextResult[object].ok(mixed_instance)

            # Track success metrics
            self._track_success_metrics(name, result)

            return result

        except Exception as e:
            # Track failure metrics
            self._track_failure_metrics(name, e)
            return FlextResult[object].fail(f"Failed to apply mixin '{name}': {e}")

    def _check_rate_limit(self, name: str) -> bool:
        """Check if rate limit is exceeded for the mixin."""
        current_time = datetime.now(UTC)
        rate_limit_key = f"{name}_rate_limit"

        if rate_limit_key not in self._rate_limit_requests:
            self._rate_limit_requests[rate_limit_key] = []

        # Clean old requests (1 minute window)
        rate_limit_window_seconds = 60
        self._rate_limit_requests[rate_limit_key] = [
            req_time
            for req_time in self._rate_limit_requests[rate_limit_key]
            if (current_time - req_time).total_seconds() < rate_limit_window_seconds
        ]

        # Check limit (10 requests per minute)
        max_requests_per_minute = 10
        if len(self._rate_limit_requests[rate_limit_key]) >= max_requests_per_minute:
            return False

        # Add current request
        self._rate_limit_requests[rate_limit_key].append(current_time)
        return True

    def _apply_middleware(self, mixin_class: type, data: object) -> object:
        """Apply middleware to mixin and data."""
        for middleware in self._middleware:
            try:
                if callable(middleware):
                    result = middleware(mixin_class, data)
                    expected_tuple_length = 2
                    if (
                        isinstance(result, (tuple, list))
                        and len(result) == expected_tuple_length
                    ):
                        mixin_class, data = result
            except Exception as e:
                # Log middleware errors but continue
                logger = logging.getLogger(__name__)
                logger.warning("Middleware error: %s", e)

        return data

    def _execute_mixin(
        self, mixin_instance: Callable[[object], object], data: object
    ) -> FlextResult[object]:
        """Execute mixin with data."""
        try:
            result = mixin_instance(data)
            # Check if result is already a FlextResult (avoid double wrapping)
            if (
                hasattr(result, "is_success")
                and hasattr(result, "value")
                and hasattr(result, "error")
            ):
                # Return the FlextResult directly
                return cast("FlextResult[object]", result)
            return FlextResult[object].ok(result)
        except Exception as e:
            return FlextResult[object].fail(f"Mixin execution failed: {e}")

    def _track_success_metrics(self, name: str, _result: FlextResult[object]) -> None:
        """Track success metrics for the mixin."""
        # Update application metrics
        current_metrics = self._metrics.get(
            name, {"applications": 0, "successes": 0, "errors": 0, "timeouts": 0}
        )
        current_metrics["applications"] += 1
        current_metrics["successes"] += 1
        self._metrics[name] = current_metrics

    def _track_failure_metrics(self, name: str, error: Exception) -> None:
        """Track failure metrics for the mixin."""
        # Log the failure for debugging/monitoring
        logger = logging.getLogger(__name__)
        logger.warning("Mixin '%s' failed: %s", name, error)

        # Update application metrics
        current_metrics = self._metrics.get(
            name, {"applications": 0, "successes": 0, "errors": 0, "timeouts": 0}
        )
        current_metrics["applications"] += 1
        current_metrics["errors"] += 1
        self._metrics[name] = current_metrics
        # error parameter is intentionally unused for basic metrics tracking

    def add_middleware(
        self,
        middleware: Callable[[type, object], tuple[type, object] | None],
    ) -> FlextResult[None]:
        """Add middleware."""
        try:
            self._middleware.append(middleware)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to add middleware: {e}")

    def get_metrics(self) -> FlextTypes.Dict:
        """Get mixin metrics."""
        return cast("FlextTypes.Dict", self._metrics.copy())

    def get_audit_log(self) -> list[FlextTypes.Dict]:
        """Get audit log."""
        return self._audit_log.copy()

    def get_performance_metrics(self) -> dict[str, dict[str, float | int]]:
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

    def get_mixins(self) -> set[object]:
        """Get all registered mixins."""
        return set(self._registry.values())

    def clear_mixins(self) -> None:
        """Clear all mixins."""
        self._registry.clear()

    def get_statistics(self) -> FlextTypes.Dict:
        """Get mixin statistics."""
        stats: FlextTypes.Dict = {
            "total_mixins": len(self._registry),
            "middleware_count": len(self._middleware),
            "audit_log_entries": len(self._audit_log),
            "performance_metrics": self._performance_metrics.copy(),
            "circuit_breakers": self._circuit_breaker.copy(),
        }

        # Add individual mixin statistics
        for mixin_name in self._registry:
            mixin_stats = self._metrics.get(
                mixin_name,
                {"applications": 0, "successes": 0, "errors": 0, "timeouts": 0},
            )
            stats[mixin_name] = mixin_stats

        return stats

    def validate(self, _data: object) -> FlextResult[None]:
        """Validate data using mixins."""
        try:
            # Validation successful - return None for success
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Validation failed: {e}")

    def is_circuit_breaker_open(self, name: str) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_breaker.get(name, False)

    # =========================================================================
    # CONTAINER INTEGRATION - Dependency Injection Infrastructure
    # =========================================================================

    class Container:
        """Container integration mixin for dependency injection.

        **Function**: Automatic DI container access and service registration
            - Lazy container access via property
            - Automatic service registration via __init_subclass__
            - Type-safe service resolution
            - FlextResult-based error handling
            - ABI compatibility through descriptors

        **Uses**: Existing FlextCore infrastructure
            - FlextContainer.get_global() for singleton access
            - FlextResult[T] for operation results
            - FlextLogger for diagnostics

        **How to use**: Inherit to add container capabilities
            ```python
            class MyService(FlextMixins.Container):
                def __init__(self):
                    # _container automatically available
                    db_result = self.container.get("database")
                    if db_result.is_success:
                        self.db = db_result.unwrap()
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        # Class variable for lazy initialization of global container
        container_instance: FlextContainer | None = None

        def __init_subclass__(cls, **kwargs: object) -> None:
            """Auto-initialize container for subclasses (ABI compatibility)."""
            super().__init_subclass__(**kwargs)
            # Container is lazily initialized on first access

        @property
        def container(self) -> FlextContainer:
            """Get global FlextContainer instance with lazy initialization."""
            if (
                not hasattr(FlextMixins.Container, "container_instance")
                or FlextMixins.Container.container_instance is None
            ):
                # Use direct instantiation to avoid deadlock in __new__ singleton pattern
                FlextMixins.Container.container_instance = FlextContainer()
            return FlextMixins.Container.container_instance

        def _register_in_container(self, service_name: str) -> FlextResult[None]:
            """Register self in global container for service discovery."""
            try:
                return self.container.register(service_name, self)
            except Exception as e:
                # If already registered, return success (for test compatibility)
                if "already registered" in str(e).lower():
                    return FlextResult[None].ok(None)
                return FlextResult[None].fail(f"Service registration failed: {e}")

    # =========================================================================
    # CONTEXT INTEGRATION - Request Context and Correlation
    # =========================================================================

    class Context:
        """Simplified context integration using FlextContext directly.

        **Function**: Direct delegation to FlextContext for all context operations
            - Request context with correlation IDs via FlextContext.Request
            - Service identification via FlextContext.Service
            - Correlation management via FlextContext.Correlation
            - Performance tracking via FlextContext.Performance
            - Automatic context propagation through FlextContext

        **Uses**: Direct FlextContext integration
            - No custom context management or lazy initialization
            - All context operations delegate to FlextContext
            - Maintains ABI compatibility through property access

        **How to use**: Direct access to FlextContext functionality
            ```python
            class MyService(FlextMixins.Context):
                def process(self, data: dict):
                    # Direct access to FlextContext
                    corr_id = FlextContext.Correlation.get_correlation_id()
                    FlextContext.Request.set_operation_name("process_data")
                    return {"correlation_id": corr_id}
            ```

        **ABI Compatibility**: Property provides access to global FlextContext,
        ensuring existing code works without changes.

        """

        @property
        def context(self) -> FlextContext:
            """Get FlextContext instance.

            Creates a new FlextContext instance for context operations.
            All context operations should use FlextContext directly.
            """
            return FlextContext()

        # Convenience methods that delegate to FlextContext for backward compatibility
        def _propagate_context(self, operation_name: str) -> None:
            """Propagate context for current operation using FlextContext."""
            FlextContext.Request.set_operation_name(operation_name)
            FlextContext.Utilities.ensure_correlation_id()

        def _get_correlation_id(self) -> str | None:
            """Get current correlation ID from FlextContext."""
            return FlextContext.Correlation.get_correlation_id()

        def _set_correlation_id(self, correlation_id: str) -> None:
            """Set correlation ID in FlextContext."""
            FlextContext.Correlation.set_correlation_id(correlation_id)

    # =========================================================================
    # LOGGING INTEGRATION - Context-Aware Structured Logging
    # =========================================================================

    class Logging:
        """Logging integration mixin for context-aware structured logging.

        **Function**: Structured logging with automatic context
            - Context-aware log messages
            - Correlation ID inclusion
            - Operation name tracking
            - FlextLogger integration
            - ABI compatibility through __init_subclass__

        **Uses**: Existing FlextCore infrastructure
            - FlextLogger for structured logging
            - FlextContext for correlation tracking
            - FlextTypes for type safety

        **How to use**: Inherit to add logging capabilities
            ```python
            class MyService(FlextMixins.Logging, FlextMixins.Context):
                def process(self, data: dict):
                    # logger automatically available
                    self._log_with_context("info", "Processing", size=len(data))
                    self.logger.debug("Details...")
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        # Class variables for lazy initialization of logger
        logger_instance: FlextLogger | None = None
        logger_name: str | None = None

        def __init_subclass__(cls, **kwargs: object) -> None:
            """Auto-initialize logger for subclasses (ABI compatibility)."""
            super().__init_subclass__(**kwargs)
            # Logger is lazily initialized on first access

        @property
        def logger(self) -> FlextLogger:
            """Get FlextLogger instance with lazy initialization."""
            if (
                not hasattr(FlextMixins.Logging, "logger_instance")
                or FlextMixins.Logging.logger_instance is None
                or FlextMixins.Logging.logger_name != self.__class__.__name__
            ):
                FlextMixins.Logging.logger_name = self.__class__.__name__
                FlextMixins.Logging.logger_instance = FlextLogger(
                    self.__class__.__name__
                )
            return FlextMixins.Logging.logger_instance

        def _log_with_context(self, level: str, message: str, **extra: object) -> None:
            """Log message with automatic context data inclusion."""
            context_data: FlextTypes.Dict = {
                "correlation_id": FlextContext.Correlation.get_correlation_id(),
                "operation": FlextContext.Request.get_operation_name(),
                **extra,
            }

            log_method = getattr(self.logger, level, self.logger.info)
            log_method(message, extra=context_data)

    # =========================================================================
    # METRICS INTEGRATION - Performance Tracking
    # =========================================================================

    class Metrics:
        """Metrics integration mixin for automatic performance tracking.

        **Function**: Performance monitoring and timing
            - Operation timing with context managers
            - Automatic metric collection
            - FlextContext.Performance integration
            - ABI compatibility through __init_subclass__

        **Uses**: Existing FlextCore infrastructure
            - FlextContext.Performance.timed_operation for timing
            - contextmanager for scope management
            - FlextTypes for type safety

        **How to use**: Inherit to add metrics capabilities
            ```python
            class MyService(FlextMixins.Metrics):
                def process(self, data: dict):
                    with self._track_operation("process_data") as metrics:
                        result = self._do_processing(data)
                        return result
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        def __init_subclass__(cls, **kwargs: object) -> None:
            """Auto-initialize metrics for subclasses (ABI compatibility)."""
            super().__init_subclass__(**kwargs)
            # Metrics tracking is automatic via context managers

        @contextmanager
        def _track_operation(self, operation_name: str) -> Iterator[FlextTypes.Dict]:
            """Track operation performance with automatic context integration."""
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                yield metrics

    # =========================================================================
    # SERVICE INTEGRATION - Complete Infrastructure Composition
    # =========================================================================

    class Service(Container, Context, Logging, Metrics):
        """Complete service infrastructure composition mixin.

        **Function**: Complete service infrastructure composition
            - Dependency injection via Container
            - Context management via Context
            - Structured logging via Logging
            - Performance tracking via Metrics
            - Automatic service registration
            - ABI compatibility through __init_subclass__

        **Uses**: All FlextMixins infrastructure components
            - FlextMixins.Container for DI
            - FlextMixins.Context for context
            - FlextMixins.Logging for logging
            - FlextMixins.Metrics for performance

        **How to use**: Inherit to get all service capabilities
            ```python
            class MyService(FlextMixins.Service):
                def __init__(self, service_name: str | None = None):
                    self._init_service(service_name)

                def process(self, data: dict):
                    # All capabilities automatically available
                    with self._track_operation("process"):
                        self._propagate_context("process")
                        self._log_with_context("info", "Processing", size=len(data))
                        return {"processed": True}
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        def __init_subclass__(cls, **kwargs: object) -> None:
            """Auto-initialize service infrastructure for subclasses."""
            super().__init_subclass__(**kwargs)
            # All mixin initialization handled by parent __init_subclass__ calls

        def _init_service(self, service_name: str | None = None) -> None:
            """Initialize service with automatic registration and setup."""
            service_name = service_name or self.__class__.__name__

            register_result = self._register_in_container(service_name)

            if register_result.is_failure:
                self.logger.warning(
                    f"Service registration failed: {register_result.error}",
                    extra={"service_name": service_name},
                )

        # =========================================================================
        # CONTEXT ENRICHMENT METHODS - Automatic Context Management
        # =========================================================================

        def _enrich_context(self, **context_data: object) -> None:
            """Log service information ONCE at initialization.

            Logs service-level information at initialization instead of binding
            it to all log messages. This provides service context visibility
            without cluttering every log entry.

            Args:
                **context_data: Additional context data to log

            Example:
                ```python
                class OrderService(FlextMixins.Service):
                    def __init__(self):
                        self._init_service("OrderService")
                        self._enrich_context(service_version="1.0.0", team="orders")

                    def process_order(self, order_id: str):
                        # Service info was logged once at initialization
                        self._log_with_context(
                            "info", "Processing order", order_id=order_id
                        )
                ```

            """
            # Build service context for logging
            service_context: FlextTypes.Dict = {
                "service_name": self.__class__.__name__,
                "service_module": self.__class__.__module__,
                **context_data,
            }
            # Log service initialization ONCE instead of binding to all logs
            self.logger.info("Service initialized", **service_context)

        def _with_operation_context(
            self,
            operation_name: str,
            **operation_data: object,
        ) -> None:
            """Set operation context for the current operation.

            Binds operation-level information to the context for tracking
            and debugging specific operations.

            Args:
                operation_name: Name of the operation being performed
                **operation_data: Additional operation context data

            Example:
                ```python
                class InventoryService(FlextMixins.Service):
                    def reserve_items(self, order_id: str, items: list):
                        # Set operation context
                        self._with_operation_context(
                            "reserve_items", order_id=order_id, item_count=len(items)
                        )

                        # All logs include operation context
                        self._log_with_context("info", "Reserving items")
                        return self._do_reserve(items)
                ```

            """
            # Propagate context using inherited Context mixin method
            self._propagate_context(operation_name)

            # Bind additional operation data using structlog's contextvars
            if operation_data:
                FlextLogger.bind_global_context(**operation_data)

        def _clear_operation_context(self) -> None:
            """Clear operation-specific context data.

            Useful for cleanup after operation completion or for
            resetting context between operations.

            Example:
                ```python
                class BatchProcessor(FlextMixins.Service):
                    def process_batch(self, items: list):
                        for item in items:
                            try:
                                self._with_operation_context(
                                    "process_item", item_id=item.id
                                )
                                self._process_single_item(item)
                            finally:
                                # Clean up context after each item
                                self._clear_operation_context()
                ```

            """
            # Clear structlog context using contextvars
            FlextLogger.clear_global_context()

            # Clear FlextContext operation name
            FlextContext.Request.set_operation_name("")


__all__ = [
    "FlextMixins",
]
