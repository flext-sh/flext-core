"""Reusable behavioral patterns for enterprise applications."""

from __future__ import annotations

import json
import time

# uuid removed - using FlextUtilities.Generators instead
from collections.abc import Iterable
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT WITH TRUE INTERNALIZATION
# =============================================================================


class FlextMixins:
    """Unified mixin system implementing Tier 1 Module Pattern.

    This class serves as the single main export consolidating ALL mixin
    functionality from the flext-core mixins ecosystem. Provides comprehensive
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
    # TIMESTAMP FUNCTIONALITY - Creation and update tracking
    # =============================================================================

    @classmethod
    def create_timestamp_fields(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Initialize timestamp fields on an object."""
        current_time = time.time()
        obj._created_at = current_time
        obj._updated_at = current_time
        obj._timestamp_initialized = True

    @classmethod
    def update_timestamp(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Update the timestamp on an object."""
        if not hasattr(obj, "_timestamp_initialized"):
            cls.create_timestamp_fields(obj)
        obj._updated_at = time.time()

    @classmethod
    def get_created_at(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Get creation timestamp."""
        if not hasattr(obj, "_timestamp_initialized"):
            cls.create_timestamp_fields(obj)
        return getattr(obj, "_created_at", time.time())

    @classmethod
    def get_updated_at(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Get last update timestamp."""
        if not hasattr(obj, "_timestamp_initialized"):
            cls.create_timestamp_fields(obj)
        return getattr(obj, "_updated_at", time.time())

    @classmethod
    def get_age_seconds(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Get age in seconds since creation."""
        created_at = cls.get_created_at(obj)
        return time.time() - created_at

    # =============================================================================
    # IDENTIFICATION FUNCTIONALITY - Entity ID management
    # =============================================================================

    @classmethod
    def ensure_id(cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> str:
        """Ensure object has an ID, generating if needed."""
        id_value = getattr(obj, "_id", None)
        if isinstance(id_value, str) and len(id_value.strip()) > 0:
            return id_value

        # Try to get from object dict
        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            id_value = obj_dict.get("id")
            if isinstance(id_value, str) and len(id_value.strip()) > 0:
                obj._id = id_value
                return id_value
        except Exception as e:
            msg = f"Failed to get ID from object: {e}"
            raise ValueError(msg) from e

        # Generate new ID using simple UUID approach
        # Use centralized UUID generation
        generated_id = FlextUtilities.Generators.generate_uuid()
        obj._id = generated_id
        return generated_id

    @classmethod
    def set_id(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, entity_id: str
    ) -> FlextResult[None]:
        """Set entity ID with validation."""
        if not entity_id or len(entity_id.strip()) == 0:
            return FlextResult[None].fail(f"Invalid entity ID: {entity_id}")

        obj._id = entity_id.strip()
        return FlextResult[None].ok(None)

    @classmethod
    def has_id(cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> bool:
        """Check if object has a valid ID."""
        id_value = getattr(obj, "_id", None)
        if isinstance(id_value, str) and len(id_value.strip()) > 0:
            return True

        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            id_value = obj_dict.get("id")
            return isinstance(id_value, str) and len(id_value.strip()) > 0
        except Exception:
            return False

    # =============================================================================
    # LOGGING FUNCTIONALITY - Structured logging support
    # =============================================================================

    @classmethod
    def get_logger(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> FlextLogger:
        """Get FlextLogger instance for an object."""
        if not hasattr(obj, "_logger"):
            logger_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            obj._logger = FlextLogger(logger_name)
        return obj._logger  # type: ignore[return-value]

    @classmethod
    def log_operation(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: str,
        **kwargs: object,
    ) -> None:
        """Log an operation with context."""
        logger = cls.get_logger(obj)
        logger.info(
            f"Operation: {operation}",
            operation=operation,
            object_type=obj.__class__.__name__,
            **kwargs,
        )

    @classmethod
    def log_error(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        error: str,
        **kwargs: object,
    ) -> None:
        """Log an error with context."""
        logger = cls.get_logger(obj)
        logger.error(
            f"Error: {error}",
            error=error if isinstance(error, Exception) else None,
            error_details=str(error),
            object_type=obj.__class__.__name__,
            **kwargs,
        )

    @classmethod
    def log_info(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log an info message with context."""
        logger = cls.get_logger(obj)
        logger.info(message, **kwargs)

    @classmethod
    def log_debug(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log a debug message with context."""
        logger = cls.get_logger(obj)
        logger.debug(message, **kwargs)

    # =============================================================================
    # SERIALIZATION FUNCTIONALITY - JSON and dict conversion
    # =============================================================================

    @classmethod
    def to_dict_basic(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> FlextTypes.Core.Dict:
        """Convert object to basic dictionary representation."""
        result = {}

        # Get object attributes
        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            for key, value in obj_dict.items():
                if not key.startswith("_"):
                    result[key] = cls._serialize_value(value)
        except Exception as e:
            msg = f"Failed to get object attributes: {e}"
            raise ValueError(msg) from e

        # Add timestamp info if available
        if hasattr(obj, "_timestamp_initialized"):
            result["created_at"] = cls.get_created_at(obj)
            result["updated_at"] = cls.get_updated_at(obj)

        # Add ID if available
        if cls.has_id(obj):
            result["id"] = cls.ensure_id(obj)

        return cast("FlextTypes.Core.Dict", result)

    @classmethod
    def to_dict(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> FlextTypes.Core.Dict:
        """Convert object to dictionary with advanced serialization."""
        result: FlextTypes.Core.Dict = {}

        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            for key, value in obj_dict.items():
                if key.startswith("_"):
                    continue

                # Try to_dict_basic first
                if isinstance(value, FlextProtocols.Foundation.HasToDictBasic):
                    try:
                        result[key] = value.to_dict_basic()
                        continue
                    except Exception as e:
                        msg = f"Failed to get object attributes: {e}"
                        raise ValueError(
                            msg,
                            # validation_details={"object": obj},
                        ) from e

                # Try to_dict
                if isinstance(value, FlextProtocols.Foundation.HasToDict):
                    try:
                        result[key] = value.to_dict()
                        continue
                    except Exception as e:
                        msg = f"Failed to get object attributes: {e}"
                        raise ValueError(
                            msg,
                            # validation_details={"object": obj},
                        ) from e

                # Handle lists
                if isinstance(value, list):
                    serialized_list: FlextTypes.Core.List = []
                    item_list: list[object] = cast("list[object]", value)
                    for item in item_list:
                        if isinstance(item, FlextProtocols.Foundation.HasToDictBasic):
                            try:
                                item_dict = item.to_dict_basic()
                                serialized_list.append(item_dict)
                                continue
                            except Exception as e:
                                msg = f"Failed to get object attributes: {e}"
                                raise ValueError(
                                    msg,
                                    # validation_details={"object": obj},
                                ) from e
                        serialized_list.append(item)
                    result[key] = serialized_list
                    continue

                # Skip None values
                if value is None:
                    continue

                result[key] = value
        except Exception as e:
            msg = f"Failed to get object attributes: {e}"
            raise ValueError(msg) from e

        return result

    @classmethod
    def to_json(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        indent: int | None = None,
    ) -> str:
        """Convert object to JSON string."""
        data = cls.to_dict_basic(obj)
        return json.dumps(data, indent=indent, default=str)

    @classmethod
    def load_from_dict(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        data: FlextTypes.Core.Dict,
    ) -> None:
        """Load object attributes from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    @classmethod
    def load_from_json(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, json_str: str
    ) -> FlextResult[None]:
        """Load object attributes from JSON string."""
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return FlextResult[None].fail("JSON data must be a dictionary")
            cls.load_from_dict(obj, cast("FlextTypes.Core.Dict", data))
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to load from JSON: {e}")

    @classmethod
    def _serialize_value(cls, value: object) -> object:
        """Serialize a value for JSON compatibility."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            typed_value: Iterable[object] = cast("Iterable[object]", value)
            return [cls._serialize_value(item) for item in typed_value]
        if isinstance(value, dict):
            typed_dict: dict[object, object] = cast("dict[object, object]", value)
            return {str(k): cls._serialize_value(v) for k, v in typed_dict.items()}
        # For complex objects, use string representation
        return str(value)

    # =============================================================================
    # VALIDATION FUNCTIONALITY - Data validation patterns
    # =============================================================================

    @classmethod
    def initialize_validation(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Initialize validation state on an object."""
        obj._validation_errors = []
        obj._is_valid = False
        obj._validation_initialized = True

    @classmethod
    def validate_required_fields(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, fields: list[str]
    ) -> object:
        """Validate that required fields are present and not empty."""
        missing_fields: list[str] = []

        for field in fields:
            value = getattr(obj, field, None)
            if value is None or (isinstance(value, str) and len(value.strip()) == 0):
                field_name: str = str(field)
                missing_fields.append(field_name)

        if missing_fields:
            return FlextResult[None].fail(
                f"Missing required fields: {', '.join(missing_fields)}",
                error_code="MISSING_REQUIRED_FIELDS",
            )

        return None

    @classmethod
    def validate_field_types(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        field_types: dict[str, type],
    ) -> object:
        """Validate that fields match expected types."""
        type_errors: list[str] = []

        for field, expected_type in field_types.items():
            value = getattr(obj, field, None)
            if value is not None and not isinstance(value, expected_type):
                error_msg: str = f"{field!s}: expected {expected_type.__name__}, got {type(value).__name__}"
                type_errors.append(error_msg)

        if type_errors:
            return FlextResult[None].fail(
                f"Type validation errors: {'; '.join(type_errors)}",
                error_code="TYPE_VALIDATION_FAILED",
            )

        return None

    @classmethod
    def add_validation_error(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, error: str
    ) -> None:
        """Add a validation error to an object."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        errors = getattr(obj, "_validation_errors", [])
        errors.append(str(error))
        obj._validation_errors = errors
        obj._is_valid = False

    @classmethod
    def clear_validation_errors(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Clear all validation errors."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        obj._validation_errors = []
        obj._is_valid = False

    @classmethod
    def get_validation_errors(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> list[str]:
        """Get all validation errors."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        return list(getattr(obj, "_validation_errors", []))

    @classmethod
    def is_valid(cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> bool:
        """Check if object is valid (no validation errors)."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        errors = getattr(obj, "_validation_errors", [])
        return len(errors) == 0 and getattr(obj, "_is_valid", False)

    @classmethod
    def mark_valid(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Mark object as valid."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        obj._is_valid = True

    # =============================================================================
    # STATE MANAGEMENT FUNCTIONALITY - Object lifecycle patterns
    # =============================================================================

    @classmethod
    def initialize_state(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        initial_state: str = "created",
    ) -> None:
        """Initialize object state."""
        obj._state = initial_state
        obj._state_history = [initial_state]
        obj._state_initialized = True

    @classmethod
    def get_state(cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> str:
        """Get current state."""
        if not hasattr(obj, "_state_initialized"):
            cls.initialize_state(obj)
        return getattr(obj, "_state", "created")

    @classmethod
    def set_state(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, new_state: str
    ) -> object:
        """Set new state with validation."""
        if not new_state or len(new_state.strip()) == 0:
            return FlextResult[None].fail(
                f"Invalid state: {new_state}", error_code="INVALID_STATE"
            )

        if not hasattr(obj, "_state_initialized"):
            cls.initialize_state(obj)

        old_state = cls.get_state(obj)
        obj._state = new_state.strip()

        # Update state history
        history = getattr(obj, "_state_history", [])
        history.append(new_state.strip())
        obj._state_history = history

        cls.log_operation(obj, "state_change", old_state=old_state, new_state=new_state)
        return None

    @classmethod
    def get_state_history(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> list[str]:
        """Get state change history."""
        if not hasattr(obj, "_state_initialized"):
            cls.initialize_state(obj)
        return list(getattr(obj, "_state_history", ["created"]))

    # =============================================================================
    # CACHING FUNCTIONALITY - Memoization patterns
    # =============================================================================

    @classmethod
    def get_cached_value(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, key: str
    ) -> object:
        """Get cached value by key."""
        if not hasattr(obj, "_cache"):
            obj._cache = {}

        cache = getattr(obj, "_cache", {})
        if key in cache:
            return None

        return None

    @classmethod
    def set_cached_value(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
        value: object,
    ) -> None:
        """Set cached value by key."""
        if not hasattr(obj, "_cache"):
            obj._cache = {}

        cache = getattr(obj, "_cache", {})
        cache[key] = value
        obj._cache = cache

    @classmethod
    def clear_cache(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Clear all cached values."""
        obj._cache = {}

    @classmethod
    def has_cached_value(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes, key: str
    ) -> bool:
        """Check if value is cached."""
        if not hasattr(obj, "_cache"):
            return False

        cache = getattr(obj, "_cache", {})
        return key in cache

    @classmethod
    def get_cache_key(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> str:
        """Generate cache key for an object."""
        if cls.has_id(obj):
            entity_id = cls.ensure_id(obj)
            return f"{obj.__class__.__name__}:{entity_id}"
        return f"{obj.__class__.__name__}:{hash(str(obj.__dict__))}"

    # =============================================================================
    # TIMING FUNCTIONALITY - Performance tracking patterns
    # =============================================================================

    @classmethod
    def start_timing(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Start timing operation and return start time."""
        start_time = time.time()
        obj._start_time = start_time
        return start_time

    @classmethod
    def stop_timing(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Stop timing and return elapsed seconds."""
        start_time = getattr(obj, "_start_time", None)
        if start_time is None:
            return 0.0

        elapsed: float = time.time() - start_time

        # Store in timing history
        if not hasattr(obj, "_elapsed_times"):
            obj._elapsed_times = []

        elapsed_times = getattr(obj, "_elapsed_times", [])
        elapsed_times.append(elapsed)
        obj._elapsed_times = elapsed_times
        obj._start_time = None

        return elapsed

    @classmethod
    def get_last_elapsed_time(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Get last elapsed time."""
        elapsed_times = getattr(obj, "_elapsed_times", [])
        return elapsed_times[-1] if elapsed_times else 0.0

    @classmethod
    def get_average_elapsed_time(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> float:
        """Get average elapsed time."""
        elapsed_times = getattr(obj, "_elapsed_times", [])
        return sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    @classmethod
    def clear_timing_history(
        cls, obj: FlextProtocols.Foundation.SupportsDynamicAttributes
    ) -> None:
        """Clear timing history."""
        obj._elapsed_times = []

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
            "timestamp_tracking",
            "identification_management",
            "logging_integration",
            "serialization_support",
            "validation_patterns",
            "state_management",
            "caching_functionality",
            "timing_performance",
            "error_handling",
            "comparison_operations",
        ]

    # =============================================================================
    # TIER 1 MODULE PATTERN - EXPORTS
    # =============================================================================

    # =============================================================================
    # MIXIN CLASSES - Real Python mixins as nested classes
    # =============================================================================

    class Loggable:
        """Real mixin class for logging functionality."""

        @property
        def logger(self) -> FlextLogger:
            """Get logger instance for this object."""
            return FlextMixins.get_logger(self)

        def log_operation(self, operation: str, **kwargs: object) -> None:
            """Log an operation with context."""
            FlextMixins.log_operation(self, operation, **kwargs)

        def log_error(self, error: str, **kwargs: object) -> None:
            """Log an error with context."""
            FlextMixins.log_error(self, error, **kwargs)

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log an info message with context."""
            FlextMixins.log_info(self, message, **kwargs)

        def log_debug(self, message: str, **kwargs: object) -> None:
            """Log a debug message with context."""
            FlextMixins.log_debug(self, message, **kwargs)

    class Serializable:
        """Real mixin class for serialization functionality."""

        def to_dict_basic(self) -> FlextTypes.Core.Dict:
            """Convert object to basic dictionary representation."""
            return FlextMixins.to_dict_basic(self)

        def to_dict(self) -> FlextTypes.Core.Dict:
            """Convert object to dictionary with advanced serialization."""
            return FlextMixins.to_dict(self)

        def load_from_dict(self, data: FlextTypes.Core.Dict) -> None:
            """Load object attributes from dictionary."""
            FlextMixins.load_from_dict(self, data)

        def load_from_json(self, json_str: str) -> None:
            """Load object attributes from JSON string."""
            result = FlextMixins.load_from_json(self, json_str)
            if result.is_failure:
                raise ValueError(result.error)

    class Timestampable:
        """Real mixin class for timestamp functionality."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize timestamp fields."""
            super().__init__(*args, **kwargs)
            FlextMixins.create_timestamp_fields(self)

        def update_timestamp(self) -> None:
            """Update the timestamp on this object."""
            FlextMixins.update_timestamp(self)

        @property
        def created_at(self) -> float:
            """Get creation timestamp."""
            return FlextMixins.get_created_at(self)

        @property
        def updated_at(self) -> float:
            """Get last update timestamp."""
            return FlextMixins.get_updated_at(self)

        def get_age_seconds(self) -> float:
            """Get age in seconds since creation."""
            return FlextMixins.get_age_seconds(self)

    class Identifiable:
        """Real mixin class for ID management functionality."""

        @property
        def id(self) -> str:
            """Get ID, generating if needed."""
            return FlextMixins.ensure_id(self)

        @id.setter
        def id(self, value: str) -> None:
            """Set entity ID with validation."""
            result = FlextMixins.set_id(self, value)
            if result.is_failure:
                raise ValueError(result.error or "Invalid entity ID")

        def has_id(self) -> bool:
            """Check if object has a valid ID."""
            return FlextMixins.has_id(self)

    class Validatable:
        """Real mixin class for validation functionality."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize validation state."""
            super().__init__(*args, **kwargs)
            FlextMixins.initialize_validation(self)

        def add_validation_error(self, error: str) -> None:
            """Add a validation error."""
            FlextMixins.add_validation_error(self, error)

        def clear_validation_errors(self) -> None:
            """Clear all validation errors."""
            FlextMixins.clear_validation_errors(self)

        @property
        def validation_errors(self) -> list[str]:
            """Get all validation errors."""
            return FlextMixins.get_validation_errors(self)

        @property
        def is_valid(self) -> bool:
            """Check if object is valid (no validation errors)."""
            return FlextMixins.is_valid(self)

        def mark_valid(self) -> None:
            """Mark object as valid."""
            FlextMixins.mark_valid(self)

    # Composite mixins
    class Service(Loggable, Validatable):
        """Composite mixin for service classes."""

    class Entity(Timestampable, Identifiable, Loggable, Validatable, Serializable):
        """Composite mixin for entity classes."""


# =============================================================================
# TIER 1 MODULE PATTERN - EXPORTS
# =============================================================================

__all__: list[str] = [
    # Backward compatibility - individual mixin classes
    "FlextLoggableMixin",
    "FlextMixins",  # ONLY main class exported
    "FlextSerializableMixin",
    "FlextTimestampMixin",
    "FlextValidatableMixin",
]

# Backward compatibility - export nested classes with Flext prefix
FlextLoggableMixin = FlextMixins.Loggable
FlextSerializableMixin = FlextMixins.Serializable
FlextTimestampMixin = FlextMixins.Timestampable
FlextValidatableMixin = FlextMixins.Validatable
