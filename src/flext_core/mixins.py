"""FLEXT Mixins - Unified behavioral patterns without wrappers.

Consolidated mixin system implementing FLEXT patterns with zero tolerance for:
- Wrappers and redeclarations
- Multiple files for single functionality
- Helper functions outside classes
- Unnecessary delegation layers

All behavioral patterns are implemented directly in FlextMixins class.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextMixins:
    """Unified mixin system with direct implementation - NO WRAPPERS.

    Consolidates ALL mixin functionality without delegation or redeclarations.
    Following FLEXT zero tolerance policy for wrappers and unnecessary abstractions.

    Behavioral Patterns:
    - Timestamp tracking (creation/update)
    - Logging integration (structured logging)
    - Serialization (JSON/dict conversion)
    - Validation (data validation)
    - Identification (ID generation)
    - State management (lifecycle)
    - Caching (memoization)
    - Timing (performance tracking)
    - Error handling (exception patterns)
    """

    # ==========================================================================
    # TIMESTAMP FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def create_timestamp_fields(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Create timestamp fields directly on object."""
        now = datetime.now(UTC)
        setattr(obj, "created_at", now)
        setattr(obj, "updated_at", now)

    @staticmethod
    def update_timestamp(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Update timestamp field directly."""
        setattr(obj, "updated_at", datetime.now(UTC))

    @staticmethod
    def get_created_at(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> datetime | None:
        """Get created timestamp."""
        return getattr(obj, "created_at", None)

    @staticmethod
    def get_updated_at(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> datetime | None:
        """Get updated timestamp."""
        return getattr(obj, "updated_at", None)

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
        if not hasattr(obj, "id") or not getattr(obj, "id"):
            entity_id = str(uuid.uuid4())
            setattr(obj, "id", entity_id)
            return entity_id
        return str(getattr(obj, "id"))

    @staticmethod
    def set_id(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, entity_id: str
    ) -> None:
        """Set object ID directly."""
        setattr(obj, "id", entity_id)

    @staticmethod
    def has_id(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> bool:
        """Check if object has ID."""
        return hasattr(obj, "id") and getattr(obj, "id") is not None

    # ==========================================================================
    # LOGGING FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def get_logger(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> FlextLogger:
        """Get logger for object."""
        # Import here to avoid circular dependency
        from flext_core.loggings import FlextLogger

        return FlextLogger(obj.__class__.__name__)

    @staticmethod
    def log_operation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: str,
        **kwargs: object,
    ) -> None:
        """Log operation with context."""
        logger = FlextMixins.get_logger(obj)
        if hasattr(logger, "info"):
            logger.info(f"Operation: {operation}", extra=kwargs)  # type: ignore[attr-defined]

    @staticmethod
    def log_error(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log error with context."""
        logger = FlextMixins.get_logger(obj)
        if hasattr(logger, "error"):
            logger.error(message, extra=kwargs)  # type: ignore[attr-defined]

    @staticmethod
    def log_info(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log info message."""
        logger = FlextMixins.get_logger(obj)
        if hasattr(logger, "info"):
            logger.info(message, extra=kwargs)  # type: ignore[attr-defined]

    @staticmethod
    def log_debug(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log debug message."""
        logger = FlextMixins.get_logger(obj)
        if hasattr(logger, "debug"):
            logger.debug(message, extra=kwargs)  # type: ignore[attr-defined]

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
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, json_str: str
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
        setattr(obj, "_validation_errors", [])
        setattr(obj, "_is_valid", True)

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
        setattr(obj, "_validation_errors", errors)
        setattr(obj, "_is_valid", False)

    @staticmethod
    def clear_validation_errors(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear all validation errors."""
        setattr(obj, "_validation_errors", [])
        setattr(obj, "_is_valid", True)

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
        setattr(obj, "_is_valid", True)
        setattr(obj, "_validation_errors", [])

    # ==========================================================================
    # STATE FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @staticmethod
    def initialize_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        initial_state: str = "initialized",
    ) -> None:
        """Initialize state management."""
        setattr(obj, "_current_state", initial_state)
        setattr(obj, "_state_history", [initial_state])

    @staticmethod
    def get_state(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> str:
        """Get current state."""
        return getattr(obj, "_current_state", "unknown")

    @staticmethod
    def set_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, state: str
    ) -> None:
        """Set current state."""
        setattr(obj, "_current_state", state)
        history = getattr(obj, "_state_history", [])
        history.append(state)
        setattr(obj, "_state_history", history)

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
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, key: str
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
            setattr(obj, "_cache", {})
        cache = getattr(obj, "_cache")
        cache[key] = value

    @staticmethod
    def clear_cache(obj: FlextProtocols.Foundation.SupportsDynamicAttributes) -> None:
        """Clear all cached values."""
        setattr(obj, "_cache", {})

    @staticmethod
    def has_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, key: str
    ) -> bool:
        """Check if value is cached."""
        cache = getattr(obj, "_cache", {})
        return key in cache

    @staticmethod
    def get_cache_key(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, *args: object
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
        setattr(obj, "_timing_start", time.perf_counter())

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
        setattr(obj, "_timing_history", history)

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
        setattr(obj, "_timing_history", [])

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
            error_msg, error_code=type(error).__name__.upper()
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
                error_msg, error_code=type(e).__name__.upper()
            )

    # ==========================================================================
    # CONFIGURATION FUNCTIONALITY - Direct implementation
    # ==========================================================================

    @classmethod
    def configure_mixins_system(
        cls, config: FlextTypes.Config.ConfigDict
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
                        f"Invalid environment '{env_value}'. Valid: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Apply defaults
            validated_config.update(
                {
                    "log_level": config.get(
                        "log_level", FlextConstants.Config.LogLevel.DEBUG.value
                    ),
                    "enable_timestamp_tracking": config.get(
                        "enable_timestamp_tracking", True
                    ),
                    "enable_logging_integration": config.get(
                        "enable_logging_integration", True
                    ),
                    "enable_serialization": config.get("enable_serialization", True),
                    "enable_validation": config.get("enable_validation", True),
                    "enable_identification": config.get("enable_identification", True),
                    "enable_state_management": config.get(
                        "enable_state_management", True
                    ),
                    "enable_caching": config.get("enable_caching", False),
                    "enable_thread_safety": config.get("enable_thread_safety", True),
                    "enable_metrics": config.get("enable_metrics", True),
                    "default_cache_size": config.get("default_cache_size", 1000),
                }
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Configuration failed: {e!s}"
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
    def optimize_mixins_performance(cls, level: str = "standard") -> dict[str, object]:
        """Optimize mixins performance."""
        if level == "high":
            return {
                "cache_enabled": True,
                "lazy_logging": True,
                "batch_validation": True,
            }
        return {
            "cache_enabled": False,
            "lazy_logging": False,
            "batch_validation": False,
        }

    @classmethod
    def validate_field_types(cls, obj: object, field_mapping: dict[str, type]) -> bool:
        """Validate field types."""
        for field_name, expected_type in field_mapping.items():
            if hasattr(obj, field_name):
                value = getattr(obj, field_name)
                if value is not None and not isinstance(value, expected_type):
                    return False
        return True


__all__: list[str] = [
    "FlextMixins",
]
