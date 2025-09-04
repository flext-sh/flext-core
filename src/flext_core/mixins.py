"""FLEXT Mixins - Unified behavioral patterns."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextMixins:
    """Unified mixin system with direct implementation."""

    # ==========================================================================
    # TIMESTAMP FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def create_timestamp_fields(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Create timestamp fields directly on object."""
        now = datetime.now(UTC)
        obj.created_at = now
        obj.updated_at = now
        obj._timestamp_initialized = True
        obj._created_at = now
        obj._updated_at = now

    @staticmethod
    def update_timestamp(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Update timestamp field directly."""
        now = datetime.now(UTC)
        obj.updated_at = now
        obj._updated_at = now

    @staticmethod
    def get_created_at(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> datetime | None:
        """Get created timestamp, initialize if not present."""
        created_at = getattr(obj, "created_at", None)
        if created_at is None:
            now = datetime.now(UTC)
            obj.created_at = now
            return now
        return created_at

    @staticmethod
    def get_updated_at(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> datetime | None:
        """Get updated timestamp, initialize if not present."""
        updated_at = getattr(obj, "updated_at", None)
        if updated_at is None:
            now = datetime.now(UTC)
            obj.updated_at = now
            return now
        return updated_at

    @staticmethod
    def get_age_seconds(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Get object age in seconds."""
        created_at = getattr(obj, "created_at", None)
        if not created_at:
            return 0.0
        return float((datetime.now(UTC) - created_at).total_seconds())

    # ==========================================================================
    # IDENTIFICATION FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def ensure_id(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> str:
        """Ensure object has unique ID."""
        if not hasattr(obj, "id") or not obj.id:
            entity_id = str(uuid.uuid4())
            obj.id = entity_id
            return entity_id
        return str(obj.id)

    @staticmethod
    def set_id(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, entity_id: str,
    ) -> None:
        """Set object ID directly."""
        obj.id = entity_id

    @staticmethod
    def has_id(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> bool:
        """Check if object has ID."""
        return hasattr(obj, "id") and obj.id is not None

    @staticmethod
    def object_hash(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> str:
        """Generate hash for object."""
        if hasattr(obj, "id") and obj.id is not None:
            return f"hash_{obj.id}"
        return f"hash_{id(obj)}"

    # ==========================================================================
    # LOGGING FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def flext_logger(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> FlextLogger:
        """Get logger for object."""
        return FlextLogger(obj.__class__.__name__)

    @staticmethod
    def log_operation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: str,
        **kwargs: object,
    ) -> None:
        """Log operation with context."""
        logger = FlextMixins.flext_logger(obj)
        if hasattr(logger, "info"):
            logger.info(f"Operation: {operation}", extra=kwargs)

    @staticmethod
    def log_error(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log error with context."""
        logger = FlextMixins.flext_logger(obj)
        if hasattr(logger, "error"):
            logger.error(message, extra=kwargs)

    @staticmethod
    def log_info(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log info message."""
        logger = FlextMixins.flext_logger(obj)
        if hasattr(logger, "info"):
            logger.info(message, extra=kwargs)

    @staticmethod
    def log_debug(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log debug message."""
        logger = FlextMixins.flext_logger(obj)
        if hasattr(logger, "debug"):
            logger.debug(message, extra=kwargs)

    # ==========================================================================
    # SERIALIZATION FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def to_dict(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> dict[str, object]:
        """Convert object to dictionary."""
        result: dict[str, object] = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            else:
                result[key] = str(value)
        return result

    @staticmethod
    def to_dict_basic(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> dict[str, object]:
        """Convert object to basic dictionary."""
        return {
            k: v
            for k, v in obj.__dict__.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }

    @staticmethod
    def to_json(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        indent: int | None = None,
    ) -> str:
        """Convert object to JSON string."""
        return json.dumps(FlextMixins.to_dict(obj), indent=indent)

    @staticmethod
    def load_from_dict(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        data: dict[str, object],
    ) -> None:
        """Load object from dictionary."""
        for key, value in data.items():
            setattr(obj, key, value)

    @staticmethod
    def load_from_json(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, json_str: str,
    ) -> None:
        """Load object from JSON string."""
        data = json.loads(json_str)
        FlextMixins.load_from_dict(obj, data)

    # ==========================================================================
    # VALIDATION FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def initialize_validation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Initialize validation state."""
        obj._validation_errors = []
        obj._is_valid = True

    @staticmethod
    def validate_required_fields(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        required_fields: list[str],
    ) -> FlextResult[None]:
        """Validate required fields."""
        errors = [
            f"Required field '{field}' is missing"
            for field in required_fields
            if not hasattr(obj, field) or getattr(obj, field) is None
        ]

        if errors:
            return FlextResult[None].fail("; ".join(errors))
        return FlextResult[None].ok(None)

    @staticmethod
    def add_validation_error(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        error: str,
    ) -> None:
        """Add validation error."""
        if not hasattr(obj, "_validation_errors"):
            FlextMixins.initialize_validation(obj)
        errors = getattr(obj, "_validation_errors", [])
        errors.append(error)
        obj._validation_errors = errors
        obj._is_valid = False

    @staticmethod
    def clear_validation_errors(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear all validation errors."""
        obj._validation_errors = []
        obj._is_valid = True

    @staticmethod
    def get_validation_errors(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> list[str]:
        """Get validation errors."""
        return getattr(obj, "_validation_errors", [])

    @staticmethod
    def is_valid(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> bool:
        """Check if object is valid."""
        return getattr(obj, "_is_valid", True)

    @staticmethod
    def mark_valid(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> None:
        """Mark object as valid."""
        obj._is_valid = True
        obj._validation_errors = []

    @staticmethod
    def validate_email(email: str) -> FlextResult[bool]:
        """Validate email address."""
        try:
            if not email or not isinstance(email, str):
                return FlextResult[bool].fail("Invalid email: empty or not string")

            if "@" not in email:
                return FlextResult[bool].fail("Invalid email: missing @ symbol")

            domain_part = email.split("@")[-1]
            if "." not in domain_part:
                return FlextResult[bool].fail("Invalid email: invalid domain")

            return FlextResult[bool].ok(data=True)
        except Exception as e:
            return FlextResult[bool].fail(f"Email validation error: {e}")

    # ==========================================================================
    # STATE FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def initialize_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        initial_state: str = "initialized",
    ) -> None:
        """Initialize state management."""
        obj._current_state = initial_state
        obj._state_history = [initial_state]

    @staticmethod
    def get_state(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> str:
        """Get current state."""
        return getattr(obj, "_current_state", "unknown")

    @staticmethod
    def set_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, state: str,
    ) -> None:
        """Set current state."""
        obj._current_state = state
        history = getattr(obj, "_state_history", [])
        history.append(state)
        obj._state_history = history

    @staticmethod
    def get_state_history(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> list[str]:
        """Get state history."""
        return getattr(obj, "_state_history", [])

    # ==========================================================================
    # CACHE FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def get_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, key: str,
    ) -> object:
        """Get cached value."""
        cache = getattr(obj, "_cache", {})
        return cache.get(key)

    @staticmethod
    def set_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
        value: object,
    ) -> None:
        """Set cached value."""
        if not hasattr(obj, "_cache"):
            obj._cache = {}
        cache = obj._cache
        cache[key] = value

    @staticmethod
    def clear_cache(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> None:
        """Clear all cached values."""
        obj._cache = {}

    @staticmethod
    def has_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, key: str,
    ) -> bool:
        """Check if value is cached."""
        cache = getattr(obj, "_cache", {})
        return key in cache

    @staticmethod
    def get_cache_key(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, *args: object,
    ) -> str:
        """Generate cache key."""
        obj_id = FlextMixins.ensure_id(obj)
        return f"{obj_id}:{':'.join(str(arg) for arg in args)}"

    # ==========================================================================
    # TIMING FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def start_timing(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> None:
        """Start performance timer."""
        obj._timing_start = time.perf_counter()

    @staticmethod
    def stop_timing(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> float:
        """Stop performance timer and return elapsed time."""
        start_time = getattr(obj, "_timing_start", None)
        if start_time is None:
            return 0.0

        elapsed = time.perf_counter() - start_time

        # Update timing history
        history = getattr(obj, "_timing_history", [])
        history.append(elapsed)
        obj._timing_history = history

        return float(elapsed)

    @staticmethod
    def get_last_elapsed_time(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Get last elapsed time."""
        history = getattr(obj, "_timing_history", [])
        return history[-1] if history else 0.0

    @staticmethod
    def get_average_elapsed_time(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Get average elapsed time."""
        history = getattr(obj, "_timing_history", [])
        return sum(history) / len(history) if history else 0.0

    @staticmethod
    def clear_timing_history(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear timing history."""
        obj._timing_history = []

    # ==========================================================================
    # ERROR HANDLING FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @classmethod
    def handle_error(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        error: Exception,
        context: str = "",
    ) -> FlextResult[None]:
        """Handle error with logging and context."""
        error_msg = f"{context}: {error!s}" if context else str(error)
        cls.log_error(obj, error_msg, error_type=type(error).__name__)
        return FlextResult[None].fail(
            error_msg, error_code=type(error).__name__.upper(),
        )

    @classmethod
    def safe_operation(
        cls,
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: Callable[[], object],
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation safely with error handling."""
        try:
            result = operation(*args, **kwargs)
            return FlextResult[object].ok(result)
        except Exception as e:
            operation_name = getattr(operation, "__name__", "unknown")
            error_msg = f"Operation {operation_name} failed: {e!s}"
            cls.log_error(obj, error_msg, error_type=type(e).__name__)
            return FlextResult[object].fail(
                error_msg, error_code=type(e).__name__.upper(),
            )

    # ==========================================================================
    # CONFIGURATION FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @classmethod
    def configure_mixins_system(
        cls, config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure mixins system with validation."""
        try:
            validated_config: FlextTypes.Config.ConfigDict = {}

            # Environment validation
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid: {valid_environments}",
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Apply defaults
            validated_config.update({
                "log_level": config.get(
                    "log_level", FlextConstants.Config.LogLevel.DEBUG.value,
                ),
                "enable_timestamp_tracking": config.get(
                    "enable_timestamp_tracking", True,
                ),
                "enable_logging_integration": config.get(
                    "enable_logging_integration", True,
                ),
                "enable_serialization": config.get("enable_serialization", True),
                "enable_validation": config.get("enable_validation", True),
                "enable_identification": config.get("enable_identification", True),
                "enable_state_management": config.get("enable_state_management", True),
                "enable_caching": config.get("enable_caching", False),
                "enable_thread_safety": config.get("enable_thread_safety", True),
                "enable_metrics": config.get("enable_metrics", True),
                "default_cache_size": config.get("default_cache_size", 1000),
            })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Configuration failed: {e!s}",
            )

    # ==========================================================================
    # MIXIN CLASSES - Direct implementation without wrappers
    # ==========================================================================

    class Timestampable:
        """Timestampable mixin class."""

        def __init__(self) -> None:
            FlextMixins.create_timestamp_fields(self)

        def touch(self) -> None:
            """Update timestamp."""
            FlextMixins.update_timestamp(self)

    class Identifiable:
        """Identifiable mixin class."""

        def __init__(self) -> None:
            FlextMixins.ensure_id(self)

        def ensure_id(self) -> str:
            """Ensure this object has a unique ID."""
            return FlextMixins.ensure_id(self)

        def set_id(self, entity_id: str) -> None:
            """Set the ID for this object."""
            FlextMixins.set_id(self, entity_id)

        def get_id(self) -> str | None:
            """Get the ID for this object."""
            return getattr(self, "id", None)

    class Loggable:
        """Loggable mixin class."""

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log info message."""
            FlextMixins.log_info(self, message, **kwargs)

        def log_error(self, message: str, **kwargs: object) -> None:
            """Log error message."""
            FlextMixins.log_error(self, message, **kwargs)

        def log_debug(self, message: str, **kwargs: object) -> None:
            """Log debug message."""
            FlextMixins.log_debug(self, message, **kwargs)

    class Serializable:
        """Serializable mixin class."""

        def to_dict(self) -> dict[str, object]:
            """Convert to dictionary."""
            return FlextMixins.to_dict(self)

        def to_json(self, indent: int | None = None) -> str:
            """Convert to JSON."""
            return FlextMixins.to_json(self, indent)

    class Validatable:
        """Validatable mixin class."""

        def __init__(self) -> None:
            FlextMixins.initialize_validation(self)

        def is_valid(self) -> bool:
            """Check if valid."""
            return FlextMixins.is_valid(self)

    class Stateful:
        """Stateful mixin class."""

        def __init__(self) -> None:
            FlextMixins.initialize_state(self)

        def set_state(self, state: str) -> None:
            """Set state."""
            FlextMixins.set_state(self, state)

        def get_state(self) -> str:
            """Get state."""
            return FlextMixins.get_state(self)

    class Cacheable:
        """Cacheable mixin class."""

        def get_cached(self, key: str) -> object:
            """Get cached value."""
            return FlextMixins.get_cached_value(self, key)

        def set_cached(self, key: str, value: object) -> None:
            """Set cached value."""
            FlextMixins.set_cached_value(self, key, value)

        def get_cached_value(self, key: str) -> object:
            """Get cached value (alias for compatibility)."""
            return FlextMixins.get_cached_value(self, key)

        def set_cached_value(self, key: str, value: object) -> None:
            """Set cached value (alias for compatibility)."""
            FlextMixins.set_cached_value(self, key, value)

        def has_cached_value(self, key: str) -> bool:
            """Check if value is cached."""
            return FlextMixins.has_cached_value(self, key)

        def clear_cache(self) -> None:
            """Clear all cached values."""
            FlextMixins.clear_cache(self)

    class Timeable:
        """Timeable mixin class."""

        def start_timing(self) -> None:
            """Start timing."""
            FlextMixins.start_timing(self)

        def stop_timing(self) -> float:
            """Stop timing."""
            return FlextMixins.stop_timing(self)

    # Composite mixin classes
    class Service(Loggable, Validatable):
        """Service composite mixin."""

        def __init__(self) -> None:
            super().__init__()

    class Entity(
        Timestampable,
        Identifiable,
        Loggable,
        Serializable,
        Validatable,
        Stateful,
        Cacheable,
        Timeable,
    ):
        """Complete entity mixin with all behaviors."""

        def __init__(self) -> None:
            super().__init__()

    # Configuration methods required by core.py
    @classmethod
    def get_mixins_system_config(cls) -> dict[str, object]:
        """Get mixins system configuration."""
        return {
            "mixin_types": ["Loggable", "Serializable", "Validatable", "Cacheable"],
            "auto_initialization": True,
            "validation_enabled": True,
            "logging_enabled": True,
        }

    @classmethod
    def optimize_mixins_performance(
        cls, config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Optimize mixins performance based on configuration using FlextResult pattern."""
        # Get memory limit to determine optimization level
        memory_limit_mb = config.get("memory_limit_mb", 512)
        default_cache_size = config.get("default_cache_size", 1000)

        # Optimize based on memory constraints
        low_memory_threshold_mb = 100  # Memory limit considered low
        if (
            isinstance(memory_limit_mb, (int, float))
            and memory_limit_mb <= low_memory_threshold_mb
        ):
            # Low memory optimization
            optimized_cache_size = min(
                default_cache_size if isinstance(default_cache_size, int) else 1000,
                low_memory_threshold_mb,
            )
            optimized_config = {
                "cache_enabled": True,
                "lazy_logging": True,
                "batch_validation": True,
                "default_cache_size": optimized_cache_size,
                "enable_memory_monitoring": True,
                "enable_caching": True,
                "enable_detailed_monitoring": False,
                "enable_batch_operations": False,  # Limited for low memory
            }
        else:
            # High memory optimization
            optimized_config = {
                "cache_enabled": True,
                "lazy_logging": False,
                "batch_validation": False,
                "default_cache_size": default_cache_size
                if isinstance(default_cache_size, int)
                else 1000,
                "enable_memory_monitoring": False,
                "enable_caching": True,
                "enable_detailed_monitoring": True,
                "enable_batch_operations": True,  # Enabled for high memory
            }

        # Convert dict[str, int] to dict[str, object] for type compatibility
        optimized_config_obj: dict[str, object] = dict(optimized_config)
        return FlextResult[dict[str, object]].ok(optimized_config_obj)

    @staticmethod
    def _normalize_context(**kwargs: object) -> dict[str, object]:
        """Normalize context data for logging and serialization."""
        normalized: dict[str, object] = {}

        for key, value in kwargs.items():
            if isinstance(value, list):
                # Normalize list items (handle BaseModel instances)
                normalized_list: list[object] = []
                for item in value:
                    if hasattr(item, "model_dump"):  # Pydantic BaseModel
                        # Type narrowing for PyRight
                        model_dump_method = item.model_dump
                        if callable(model_dump_method):
                            normalized_list.append(model_dump_method())
                        else:
                            normalized_list.append(item)
                    elif hasattr(item, "dict"):  # Legacy Pydantic v1
                        # Type narrowing for PyRight
                        dict_method = item.dict
                        if callable(dict_method):
                            normalized_list.append(dict_method())
                        else:
                            normalized_list.append(item)
                    else:
                        normalized_list.append(item)
                normalized[key] = normalized_list
            elif hasattr(value, "model_dump"):  # Single BaseModel
                # Type narrowing for PyRight
                model_dump_method = value.model_dump
                if callable(model_dump_method):
                    normalized[key] = model_dump_method()
                else:
                    normalized[key] = value
            elif hasattr(value, "dict"):  # Legacy Pydantic v1
                # Type narrowing for PyRight
                dict_method = value.dict
                if callable(dict_method):
                    normalized[key] = dict_method()
                else:
                    normalized[key] = value
            else:
                normalized[key] = value

        return normalized

    @classmethod
    def validate_field_types(
        cls, obj: object, field_mapping: dict[str, type],
    ) -> dict[str, object]:
        """Validate field types."""
        result: dict[str, object] = {"success": True, "errors": []}

        for field_name, expected_type in field_mapping.items():
            if hasattr(obj, field_name):
                value = getattr(obj, field_name)
                if value is not None and not isinstance(value, expected_type):
                    result["success"] = False
                    result["errors"].append(
                        f"Field '{field_name}' expected {expected_type.__name__}, got {type(value).__name__}",
                    )

        return result


__all__: list[str] = [
    "FlextMixins",
]
