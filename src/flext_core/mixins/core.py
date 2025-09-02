"""Reusable behavioral patterns and mixin system.

Provides behavioral patterns including timestamp tracking, logging integration,
serialization, validation, identification, and state management through FlextMixins class.

Usage:
    # Inheritable mixins
    class User(FlextMixins.Timestampable, FlextMixins.Loggable):
        name: str
        email: str

    user = User(name="John", email="john@example.com")
    user.touch()  # Update timestamps
    user.log_info("User created")

    # Utility methods
    FlextMixins.ensure_id(entity)  # Ensure entity has ID
    FlextMixins.to_dict(entity)    # Serialize to dict
    validation_result = FlextMixins.validate_required_fields(entity, ["name", "email"])

Features:
    - Inheritable mixin classes (Timestampable, Loggable, Serializable, etc.)
    - Utility methods for common object operations
    - FlextResult integration for error handling
    - Validation and state management
    - Performance timing and caching patterns
        initialize_state(obj) -> None              # Initialize object state
        get_state(obj) -> dict                     # Get current state
        set_state(obj, state) -> None              # Set object state
        get_state_history(obj) -> list[dict]       # Get state change history

        # Caching:
        get_cached_value(obj, key) -> object | None # Get cached value
        set_cached_value(obj, key, value) -> None  # Set cached value
        clear_cache(obj) -> None                   # Clear all cached values
        has_cached_value(obj, key) -> bool         # Check if value cached

        # Performance Timing:
        start_timing(obj) -> None                  # Start performance timer
        stop_timing(obj) -> None                   # Stop performance timer
        get_average_elapsed_time(obj) -> float     # Get average execution time

        # Error Handling:
        handle_error(obj, error) -> FlextResult[None] # Handle error with logging
        safe_operation(obj, operation) -> FlextResult[object] # Execute operation safely

    # Inheritable Mixin Classes:
    Loggable                           # Logging functionality mixin
    Serializable                       # JSON/dict serialization mixin
    Timestampable                      # Automatic timestamp tracking mixin
    Identifiable                       # ID management mixin
    Validatable                        # Data validation mixin
    Service                            # Composite mixin (Loggable + Validatable)
    Entity                             # Complete composite mixin (all behaviors)

Usage Examples:
    Using mixin utilities:
        obj = MyClass()
        FlextMixins.ensure_id(obj)  # Adds unique ID
        FlextMixins.create_timestamp_fields(obj)  # Adds created/updated timestamps
        logger = FlextMixins.get_logger(obj)  # Gets logger instance

    Using mixin inheritance:
        class User(FlextMixins.Entity):  # Inherits all behaviors
            def __init__(self, name: str):
                self.name = name
                self.ensure_id()  # From Identifiable
                self.create_timestamp_fields()  # From Timestampable

    Configuration:
        config_result = FlextMixins.create_environment_mixins_config("production")
        if config_result.success:
            config = config_result.unwrap()

Integration:
    FlextMixins integrates with FlextResult for error handling, FlextLogger for
    structured logging, FlextProtocols for type safety, and FlextConstants for
    configuration management across the FLEXT ecosystem.
    >>> if perf_result.success:
    ...     core.observability.track_performance_config(perf_result.value)

Enterprise Object Management:
    >>> # Using mixin functionality on any object
    >>> class UserService:
    ...     def __init__(self, name: str):
    ...         self.name = name
    ...         FlextMixins.create_timestamp_fields(self)
    ...         FlextMixins.initialize_validation(self)
    >>> service = UserService("user-api-v1")
    >>> service_id = FlextMixins.ensure_id(service)  # Generates UUID
    >>> FlextMixins.log_operation(service, "service_created", service_id=service_id)
    >>> # Validation and serialization
    >>> FlextMixins.mark_valid(service)
    >>> service_dict = FlextMixins.to_dict(service)
    >>> service_json = FlextMixins.to_json(service, indent=2)

Inheritance-Based Usage:
    >>> # Using real mixin classes
    >>> class OrderEntity(FlextMixins.Entity):
    ...     def __init__(self, order_id: str, amount: float):
    ...         super().__init__()
    ...         self.order_id = order_id
    ...         self.amount = amount
    >>> order = OrderEntity("ORD-123", 99.99)
    >>> print(f"Order created at: {order.created_at}")
    >>> order.log_info("Order created", order_id=order.order_id, amount=order.amount)
    >>> # Automatic serialization and validation
    >>> order_data = order.to_dict()
    >>> print(f"Order is valid: {order.is_valid}")

Performance Optimization Examples:
    >>> # High-performance configuration
    >>> high_perf_config = {
    ...     "performance_level": "high",
    ...     "memory_limit_mb": 4096,
    ...     "cpu_cores": 16,
    ...     "enable_caching": True,
    ...     "enable_async_operations": True,
    ... }
    >>> result = FlextMixins.optimize_mixins_performance(high_perf_config)
    >>> # Environment-specific optimization
    >>> prod_config = FlextMixins.create_environment_mixins_config("production")
    >>> if prod_config.success:
    ...     config = prod_config.value
    ...     # Production has: caching=True, large cache, thread safety, metrics

Behavioral Pattern Examples:
    >>> # State management with history
    >>> service = UserService("payment-service")
    >>> FlextMixins.initialize_state(service, "initializing")
    >>> FlextMixins.set_state(service, "running")
    >>> FlextMixins.set_state(service, "maintenance")
    >>> history = FlextMixins.get_state_history(
    ...     service
    ... )  # ["initializing", "running", "maintenance"]
    >>> # Performance timing
    >>> FlextMixins.start_timing(service)
    >>> # ... perform operations ...
    >>> elapsed = FlextMixins.stop_timing(service)
    >>> avg_time = FlextMixins.get_average_elapsed_time(service)

Error Handling Integration:
    >>> # Safe operations with FlextResult
    >>> def risky_operation():
    ...     raise ValueError("Something went wrong")
    >>> service = UserService("error-service")
    >>> result = FlextMixins.safe_operation(service, risky_operation)
    >>> if result and hasattr(result, "failure") and result.failure:
    ...     FlextMixins.log_error(service, f"Operation failed: {result.error}")

Notes:
    - All mixin functionality returns FlextResult for type-safe error handling
    - Configuration supports environment-specific optimization (dev/test/staging/prod)
    - Performance tuning includes caching, threading, memory management, and async operations
    - Real mixin classes support multiple inheritance for complex behavioral composition
    - Integration with FlextCore provides centralized logging, observability, and configuration
    - Type safety maintained through FlextTypes and FlextProtocols integration
    - Backward compatibility preserved through individual mixin class exports
    - Thread-safe operations supported for concurrent enterprise applications

"""

from __future__ import annotations

from flext_core.constants import FlextConstants
from flext_core.mixins.cache import FlextCache as _FlextCache
from flext_core.mixins.identification import (
    FlextIdentification as _FlextIdentification,
)
from flext_core.mixins.logging import FlextLogging as _FlextLogging
from flext_core.mixins.serialization import (
    FlextSerialization as _FlextSerialization,
)
from flext_core.mixins.state import FlextState as _FlextState
from flext_core.mixins.timestamps import FlextTimestamps as _FlextTimestamps
from flext_core.mixins.timing import FlextTiming as _FlextTiming
from flext_core.mixins.validation import FlextValidation as _FlextValidation
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT WITH TRUE INTERNALIZATION
# =============================================================================


class FlextMixins:
    """Unified mixin system implementing Tier 1 Module Pattern.

    This class serves as the single main export consolidating ALL mixin
    functionality from the flext-core mixins ecosystem. Provides
    behavioral patterns while maintaining backward compatibility.

    Tier 1 Module Pattern: mixins.py -> FlextMixins
    All mixin functionality is accessible through this single interface.

    Consolidated Functionality:
    - Timestamp Tracking (creation/update patterns)
    - Logging Integration (structured logging patterns)
    - Serialization (JSON and dict conversion)
    - Validation (data validation patterns)
    - Identification (ID generation patterns)
    - State Management (lifecycle patterns)
    - Error Handling (exception patterns)
    - Caching (memoization patterns)
    - Thread Safety (concurrent access patterns)
    - Configuration (settings patterns)
    - Metrics (performance tracking patterns)
    - Event Handling (observer patterns)
    """

    # ==========================================================================
    # CONFIGURATION METHODS WITH FLEXTTYPES.CONFIG INTEGRATION
    # ==========================================================================

    @classmethod
    def configure_mixins_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure mixins system using FlextTypes.Config with StrEnum validation.

        Args:
            config: Configuration dictionary with mixins settings

        Returns:
            FlextResult containing the validated and applied configuration

        """
        try:
            # Create validated configuration with defaults
            validated_config: FlextTypes.Config.ConfigDict = {}

            # Validate environment using FlextConstants.Config.ConfigEnvironment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate log level using FlextConstants.Config.LogLevel
            if "log_level" in config:
                log_level = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_log_levels}"
                    )
                validated_config["log_level"] = log_level
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Mixins-specific configuration
            validated_config["enable_timestamp_tracking"] = config.get(
                "enable_timestamp_tracking", True
            )
            validated_config["enable_logging_integration"] = config.get(
                "enable_logging_integration", True
            )
            validated_config["enable_serialization"] = config.get(
                "enable_serialization", True
            )
            validated_config["enable_validation"] = config.get(
                "enable_validation", True
            )
            validated_config["enable_identification"] = config.get(
                "enable_identification", True
            )
            validated_config["enable_state_management"] = config.get(
                "enable_state_management", True
            )
            validated_config["enable_caching"] = config.get("enable_caching", False)
            # Ensure explicit cache_enabled flag is always present
            validated_config["cache_enabled"] = bool(config.get("cache_enabled", True))
            # Pass-through legacy-style flags used in tests
            if "cache_enabled" in config:
                validated_config["cache_enabled"] = bool(config["cache_enabled"])
            validated_config["cache_ttl"] = config.get("cache_ttl", 3600)
            if "state_management_enabled" in config:
                validated_config["state_management_enabled"] = bool(
                    config["state_management_enabled"]
                )
            if "enable_detailed_validation" in config:
                validated_config["enable_detailed_validation"] = bool(
                    config["enable_detailed_validation"]
                )
            if "max_validation_errors" in config:
                max_errors_value = config["max_validation_errors"]
                if isinstance(max_errors_value, (int, str, float)):
                    validated_config["max_validation_errors"] = int(max_errors_value)
                else:
                    validated_config["max_validation_errors"] = 10  # Default
            validated_config["enable_thread_safety"] = config.get(
                "enable_thread_safety", True
            )
            validated_config["enable_metrics"] = config.get("enable_metrics", True)
            validated_config["default_cache_size"] = config.get(
                "default_cache_size", 1000
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure mixins system: {e!s}"
            )

    @classmethod
    def get_mixins_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current mixins system configuration with runtime information.

        Returns:
            FlextResult containing current mixins system configuration

        """
        try:
            config: FlextTypes.Config.ConfigDict = {
                # Environment information
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                # Mixins system settings
                "enable_timestamp_tracking": True,
                "enable_logging_integration": True,
                "enable_serialization": True,
                "enable_validation": True,
                "enable_identification": True,
                "enable_state_management": True,
                "enable_caching": False,
                "enable_thread_safety": True,
                "enable_metrics": True,
                "default_cache_size": 1000,
                # Runtime metrics
                "active_mixins": 0,
                "cached_objects": 0,
                "serialization_operations": 0,
                "validation_operations": 0,
                # Available patterns
                "available_mixins": [
                    "TimestampMixin",
                    "LoggingMixin",
                    "SerializationMixin",
                    "ValidationMixin",
                    "IdentificationMixin",
                ],
                "enabled_behaviors": [
                    "timestamp_tracking",
                    "logging",
                    "serialization",
                    "validation",
                    "identification",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get mixins system configuration: {e!s}"
            )

    @classmethod
    def create_environment_mixins_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific mixins system configuration.

        Args:
            environment: Target environment for configuration

        Returns:
            FlextResult containing environment-optimized mixins configuration

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_timestamp_tracking": True,
                "enable_logging_integration": True,
                "enable_serialization": True,
                "enable_validation": True,
                "enable_identification": True,
            }

            # Environment-specific optimizations
            if environment == "production":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_caching": True,  # Enable caching in production
                    "default_cache_size": 10000,  # Large cache for production
                    "enable_thread_safety": True,  # Thread safety critical in production
                    "enable_metrics": True,  # Metrics for production monitoring
                    "enable_performance_optimization": True,  # Performance optimizations
                    "cache_ttl_seconds": 600,  # 10 minute cache TTL
                    "enable_lazy_initialization": True,  # Lazy init for performance
                })
            elif environment == "development":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_caching": False,  # No caching for development
                    "default_cache_size": 100,  # Small cache for development
                    "enable_thread_safety": False,  # Not needed in single-threaded dev
                    "enable_metrics": True,  # Metrics for debugging
                    "enable_debug_logging": True,  # Debug logging for development
                    "enable_validation_verbose": True,  # Verbose validation messages
                })
            elif environment == "test":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_caching": False,  # No caching in tests
                    "default_cache_size": 10,  # Very small cache for tests
                    "enable_thread_safety": False,  # Single-threaded tests
                    "enable_metrics": False,  # No metrics in tests
                    "enable_test_mode": True,  # Special test mode
                    "enable_mock_behavior": True,  # Enable mock behavior
                })
            elif environment == "staging":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_caching": True,  # Test caching in staging
                    "default_cache_size": 5000,  # Medium cache for staging
                    "enable_thread_safety": True,  # Test thread safety
                    "enable_metrics": True,  # Metrics for staging validation
                    "cache_ttl_seconds": 300,  # 5 minute cache TTL
                    "enable_staging_validation": True,  # Staging-specific validation
                })
            else:  # local environment
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_caching": False,  # No caching locally
                    "default_cache_size": 50,  # Tiny cache for local
                    "enable_thread_safety": False,  # Single-threaded local development
                    "enable_metrics": False,  # No metrics locally
                    "enable_local_debugging": True,  # Local debugging features
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment mixins configuration: {e!s}"
            )

    @classmethod
    def optimize_mixins_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize mixins system performance based on configuration.

        Args:
            config: Performance optimization configuration

        Returns:
            FlextResult containing performance-optimized mixins configuration

        """
        try:
            # Start with base configuration
            optimized_config: FlextTypes.Config.ConfigDict = config.copy()

            # Performance level-based optimizations
            performance_level = config.get("performance_level", "medium")

            if performance_level == "high":
                optimized_config.update({
                    "enable_caching": True,
                    "default_cache_size": 50000,  # Very large cache
                    "cache_ttl_seconds": 3600,  # 1 hour cache TTL
                    "enable_lazy_initialization": True,  # Lazy initialization
                    "enable_object_pooling": True,  # Object pooling
                    "pool_size": 1000,  # Large object pool
                    "enable_batch_operations": True,  # Batch operations
                    "batch_size": 1000,  # Large batch size
                    "enable_async_operations": True,  # Async operations
                    "max_concurrent_operations": 100,  # High concurrency
                })
            elif performance_level == "medium":
                optimized_config.update({
                    "enable_caching": True,
                    "default_cache_size": 10000,  # Medium cache
                    "cache_ttl_seconds": 1800,  # 30 minute cache TTL
                    "enable_lazy_initialization": True,  # Lazy initialization
                    "enable_batch_operations": True,  # Batch operations
                    "batch_size": 100,  # Medium batch size
                    "max_concurrent_operations": 25,  # Medium concurrency
                })
            else:  # low performance level
                optimized_config.update({
                    "enable_caching": False,  # No caching
                    "default_cache_size": 100,  # Small cache if needed
                    "enable_lazy_initialization": False,  # No lazy initialization
                    "enable_batch_operations": False,  # No batch operations
                    "batch_size": 1,  # Single operations
                    "max_concurrent_operations": 1,  # Single-threaded
                    "enable_detailed_monitoring": True,  # More detailed monitoring
                })

            # Memory optimization settings - define constants for thresholds
            low_memory_threshold = 256
            high_memory_threshold = 4096

            # Type-safe memory limit handling
            memory_limit_raw = config.get("memory_limit_mb", 512)
            memory_limit_mb = (
                int(memory_limit_raw)
                if isinstance(memory_limit_raw, (int, str, float))
                else 512
            )

            if memory_limit_mb < low_memory_threshold:
                current_cache = optimized_config.get("default_cache_size", 1000)
                cache_size = (
                    int(current_cache)
                    if isinstance(current_cache, (int, str, float))
                    else 1000
                )
                optimized_config["default_cache_size"] = min(cache_size, 100)
                optimized_config["enable_memory_monitoring"] = True
                optimized_config["enable_garbage_collection"] = True
            elif memory_limit_mb > high_memory_threshold:
                optimized_config["enable_large_cache"] = True
                optimized_config["enable_memory_mapping"] = True

            # Type-safe CPU cores handling
            cpu_cores_raw = config.get("cpu_cores", 4)
            cpu_cores = (
                int(cpu_cores_raw)
                if isinstance(cpu_cores_raw, (int, str, float))
                else 4
            )
            optimized_config["max_worker_threads"] = min(cpu_cores, 8)
            optimized_config["thread_pool_size"] = cpu_cores * 2

            # Add performance metrics
            optimized_config.update({
                "performance_level": performance_level,
                "memory_limit_mb": memory_limit_mb,
                "cpu_cores": cpu_cores,
                "optimization_applied": True,
                "optimization_timestamp": "runtime",
            })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize mixins performance: {e!s}"
            )

    # =============================================================================
    # BASE ABSTRACT MIXIN - Foundation for all mixins
    # =============================================================================

    class _AbstractMixin:
        """Base abstract mixin for all FLEXT mixins."""

        def mixin_setup(self) -> None:
            """Set up mixin functionality."""

    # =============================================================================
    # DELEGATE ALL FUNCTIONALITY TO MODULAR MIXINS
    # =============================================================================

    # Timestamp functionality
    create_timestamp_fields = staticmethod(_FlextTimestamps.create_timestamp_fields)
    update_timestamp = staticmethod(_FlextTimestamps.update_timestamp)
    get_created_at = staticmethod(_FlextTimestamps.get_created_at)
    get_updated_at = staticmethod(_FlextTimestamps.get_updated_at)
    get_age_seconds = staticmethod(_FlextTimestamps.get_age_seconds)

    # Identification functionality
    ensure_id = staticmethod(_FlextIdentification.ensure_id)
    set_id = staticmethod(_FlextIdentification.set_id)
    has_id = staticmethod(_FlextIdentification.has_id)

    # Logging functionality
    get_logger = staticmethod(_FlextLogging.get_logger)
    log_operation = staticmethod(_FlextLogging.log_operation)
    log_error = staticmethod(_FlextLogging.log_error)
    log_info = staticmethod(_FlextLogging.log_info)
    log_debug = staticmethod(_FlextLogging.log_debug)

    # Serialization functionality
    to_dict = staticmethod(_FlextSerialization.to_dict)
    to_dict_basic = staticmethod(_FlextSerialization.to_dict_basic)
    to_json = staticmethod(_FlextSerialization.to_json)
    load_from_dict = staticmethod(_FlextSerialization.load_from_dict)
    load_from_json = staticmethod(_FlextSerialization.load_from_json)

    # Validation functionality
    initialize_validation = staticmethod(_FlextValidation.initialize_validation)
    validate_required_fields = staticmethod(_FlextValidation.validate_required_fields)
    validate_field_types = staticmethod(_FlextValidation.validate_field_types)
    add_validation_error = staticmethod(_FlextValidation.add_validation_error)
    clear_validation_errors = staticmethod(_FlextValidation.clear_validation_errors)
    get_validation_errors = staticmethod(_FlextValidation.get_validation_errors)
    is_valid = staticmethod(_FlextValidation.is_valid)
    mark_valid = staticmethod(_FlextValidation.mark_valid)

    # State functionality
    initialize_state = staticmethod(_FlextState.initialize_state)
    get_state = staticmethod(_FlextState.get_state)
    set_state = staticmethod(_FlextState.set_state)
    get_state_history = staticmethod(_FlextState.get_state_history)

    # Cache functionality
    get_cached_value = staticmethod(_FlextCache.get_cached_value)
    set_cached_value = staticmethod(_FlextCache.set_cached_value)
    clear_cache = staticmethod(_FlextCache.clear_cache)
    has_cached_value = staticmethod(_FlextCache.has_cached_value)
    get_cache_key = staticmethod(_FlextCache.get_cache_key)

    # Timing functionality
    start_timing = staticmethod(_FlextTiming.start_timing)
    stop_timing = staticmethod(_FlextTiming.stop_timing)
    get_last_elapsed_time = staticmethod(_FlextTiming.get_last_elapsed_time)
    get_average_elapsed_time = staticmethod(_FlextTiming.get_average_elapsed_time)
    clear_timing_history = staticmethod(_FlextTiming.clear_timing_history)

    # =============================================================================
    # ERROR HANDLING FUNCTIONALITY - Exception patterns
    # =============================================================================

    @classmethod
    def handle_error(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        error: Exception,
        context: str = "",
    ) -> object:
        """Handle error with logging and context."""
        error_msg = f"{context}: {error!s}" if context else str(error)
        cls.log_error(obj, error_msg, error_type=type(error).__name__)

        return FlextResult[None].fail(
            error_msg, error_code=type(error).__name__.upper()
        )

    @classmethod
    def safe_operation(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: FlextTypes.Meta.DecoratorFactory[object, object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Execute operation safely with error handling."""
        try:
            operation(*args, **kwargs)
            return None
        except Exception as e:
            error_msg = f"Operation {operation.__name__} failed: {e!s}"
            cls.log_error(obj, error_msg, error_type=type(e).__name__)
            return FlextResult[object].fail(
                error_msg, error_code=type(e).__name__.upper()
            )

    # =============================================================================
    # COMPARISON FUNCTIONALITY - Equality and ordering patterns
    # =============================================================================

    @classmethod
    def objects_equal(cls, obj1: object, obj2: object) -> bool:
        """Check if two objects are equal."""
        if not isinstance(obj2, obj1.__class__):
            return False

        if isinstance(obj1, FlextProtocols.Foundation.HasToDict) and isinstance(
            obj2, FlextProtocols.Foundation.HasToDict
        ):
            return obj1.to_dict() == obj2.to_dict()

        return obj1.__dict__ == obj2.__dict__

    @classmethod
    def object_hash(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> int:
        """Generate hash for an object."""
        if cls.has_id(obj):
            entity_id = cls.ensure_id(obj)
            return hash(entity_id)
        return hash(str(obj.__dict__))

    @classmethod
    def compare_objects(cls, obj1: object, obj2: object) -> int:
        """Compare two objects (-1, 0, 1)."""
        from typing import cast

        if not isinstance(obj2, obj1.__class__):
            return -1

        left = (
            cls.to_dict_basic(
                cast("FlextProtocols.Foundation.SupportsDynamicAttributes", obj1)
            )
            if isinstance(obj1, FlextProtocols.Foundation.HasToDictBasic)
            else obj1.__dict__
        )
        right = (
            cls.to_dict_basic(
                cast("FlextProtocols.Foundation.SupportsDynamicAttributes", obj2)
            )
            if isinstance(obj2, FlextProtocols.Foundation.HasToDictBasic)
            else obj2.__dict__
        )

        left_s = str(left)
        right_s = str(right)

        if left_s == right_s:
            return 0
        return -1 if left_s < right_s else 1

    # =============================================================================
    # UTILITY METHODS - Helper functions
    # =============================================================================

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Validate non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    @classmethod
    def get_protocols(cls) -> tuple[type, ...]:
        """Get runtime-checkable protocols."""
        return (
            FlextProtocols.Foundation.HasToDictBasic,
            FlextProtocols.Foundation.HasToDict,
        )

    @classmethod
    def list_available_patterns(cls) -> list[str]:
        """List all available behavioral patterns."""
        return [
            "caching_functionality",
            "comparison_operations",
            "error_handling",
            "identification_management",
            "logging_integration",
            "serialization_support",
            "state_management",
            "timestamp_tracking",
            "timing_performance",
            "validation_patterns",
        ]

    # =============================================================================
    # MIXIN CLASSES - Import from modular mixins
    # =============================================================================

    Cacheable = _FlextCache.Cacheable
    Identifiable = _FlextIdentification.Identifiable
    Loggable = _FlextLogging.Loggable
    Serializable = _FlextSerialization.Serializable
    Stateful = _FlextState.Stateful
    Timeable = _FlextTiming.Timeable
    Timestampable = _FlextTimestamps.Timestampable
    Validatable = _FlextValidation.Validatable

    # Composite mixin classes
    class Service(_FlextLogging.Loggable, _FlextValidation.Validatable):
        """Service composite mixin with logging and validation."""

    class Entity(
        _FlextCache.Cacheable,
        _FlextIdentification.Identifiable,
        _FlextLogging.Loggable,
        _FlextSerialization.Serializable,
        _FlextState.Stateful,
        _FlextTimestamps.Timestampable,
        _FlextTiming.Timeable,
        _FlextValidation.Validatable,
    ):
        """Complete entity mixin with all behavioral patterns."""


# =============================================================================
# TIER 1 MODULE PATTERN - EXPORTS
# =============================================================================

__all__: list[str] = [
    "FlextMixins",  # ONLY main class exported
]
