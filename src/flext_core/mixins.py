"""Reusable behavioral patterns for enterprise applications."""

from __future__ import annotations

import json
import time
from typing import Protocol, cast, runtime_checkable

from flext_core.exceptions import FlextOperationError, FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult
from flext_core.typings import FlextDecoratedFunction, FlextLoggerProtocol, FlextTypes
from flext_core.utilities import FlextUtilities

# =============================================================================
# PROTOCOLS - Runtime-checkable interfaces
# =============================================================================


@runtime_checkable
class HasToDictBasic(Protocol):
    """Runtime-checkable protocol for objects exposing to_dict_basic."""

    def to_dict_basic(self) -> FlextTypes.Core.Dict:  # pragma: no cover - typing helper
        ...


@runtime_checkable
class HasToDict(Protocol):
    """Runtime-checkable protocol for objects exposing to_dict."""

    def to_dict(self) -> FlextTypes.Core.Dict:  # pragma: no cover - typing helper
        ...


@runtime_checkable
class SupportsDynamicAttributes(Protocol):
    """Protocol for objects that support dynamic attribute setting.

    This protocol allows mixins to set arbitrary attributes on objects
    without triggering MyPy errors for missing attributes.
    """

    def __setattr__(self, name: str, value: object, /) -> None: ...  # noqa: ANN401
    def __getattribute__(self, name: str, /) -> object: ...  # noqa: ANN401


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
    def create_timestamp_fields(cls, obj: SupportsDynamicAttributes) -> None:
        """Initialize timestamp fields on an object."""
        current_time = time.time()
        obj._created_at = current_time
        obj._updated_at = current_time
        obj._timestamp_initialized = True

    @classmethod
    def update_timestamp(cls, obj: SupportsDynamicAttributes) -> None:
        """Update the timestamp on an object."""
        if not hasattr(obj, "_timestamp_initialized"):
            cls.create_timestamp_fields(obj)
        obj._updated_at = time.time()

    @classmethod
    def get_created_at(cls, obj: SupportsDynamicAttributes) -> float:
        """Get creation timestamp."""
        if not hasattr(obj, "_timestamp_initialized"):
            cls.create_timestamp_fields(obj)
        return getattr(obj, "_created_at", time.time())

    @classmethod
    def get_updated_at(cls, obj: SupportsDynamicAttributes) -> float:
        """Get last update timestamp."""
        if not hasattr(obj, "_timestamp_initialized"):
            cls.create_timestamp_fields(obj)
        return getattr(obj, "_updated_at", time.time())

    @classmethod
    def get_age_seconds(cls, obj: SupportsDynamicAttributes) -> float:
        """Get age in seconds since creation."""
        created_at = cls.get_created_at(obj)
        return time.time() - created_at

    # =============================================================================
    # IDENTIFICATION FUNCTIONALITY - Entity ID management
    # =============================================================================

    @classmethod
    def ensure_id(cls, obj: SupportsDynamicAttributes) -> str:
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
            raise FlextOperationError(
                msg,
                validation_details={"object": obj},
            ) from e

        # Generate new ID
        generated_id = FlextUtilities.generate_id()
        obj._id = generated_id
        return generated_id

    @classmethod
    def set_id(
        cls, obj: SupportsDynamicAttributes, entity_id: str
    ) -> FlextResult[None]:
        """Set entity ID with validation."""
        if not isinstance(entity_id, str) or len(entity_id.strip()) == 0:
            return FlextResult[None].fail(
                f"Invalid entity ID: {entity_id}", error_code="INVALID_ENTITY_ID"
            )

        obj._id = entity_id.strip()
        return FlextResult[None].ok(None)

    @classmethod
    def has_id(cls, obj: SupportsDynamicAttributes) -> bool:
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
    def get_logger(cls, obj: SupportsDynamicAttributes) -> FlextLoggerProtocol:
        """Get logger instance for an object."""
        if not hasattr(obj, "_logger"):
            logger_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            obj._logger = FlextLoggerFactory.get_logger(logger_name)
        return cast("FlextLoggerProtocol", obj._logger)

    @classmethod
    def log_operation(
        cls, obj: SupportsDynamicAttributes, operation: str, **kwargs: object
    ) -> None:
        """Log an operation with context."""
        logger = cls.get_logger(obj)
        context = {
            "operation": operation,
            "object_type": obj.__class__.__name__,
            **kwargs,
        }
        logger.info(f"Operation: {operation}", extra=context)

    @classmethod
    def log_error(
        cls, obj: SupportsDynamicAttributes, error: str, **kwargs: object
    ) -> None:
        """Log an error with context."""
        logger = cls.get_logger(obj)
        context = {"error": error, "object_type": obj.__class__.__name__, **kwargs}
        logger.error(f"Error: {error}", extra=context)

    @classmethod
    def log_info(
        cls, obj: SupportsDynamicAttributes, message: str, **kwargs: object
    ) -> None:
        """Log an info message with context."""
        logger = cls.get_logger(obj)
        logger.info(message, **kwargs)

    @classmethod
    def log_debug(
        cls, obj: SupportsDynamicAttributes, message: str, **kwargs: object
    ) -> None:
        """Log a debug message with context."""
        logger = cls.get_logger(obj)
        logger.debug(message, **kwargs)

    # =============================================================================
    # SERIALIZATION FUNCTIONALITY - JSON and dict conversion
    # =============================================================================

    @classmethod
    def to_dict_basic(cls, obj: SupportsDynamicAttributes) -> dict[str, object]:
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
            raise FlextOperationError(
                msg,
                validation_details={"object": obj},
            ) from e

        # Add timestamp info if available
        if hasattr(obj, "_timestamp_initialized"):
            result["created_at"] = cls.get_created_at(obj)
            result["updated_at"] = cls.get_updated_at(obj)

        # Add ID if available
        if cls.has_id(obj):
            result["id"] = cls.ensure_id(obj)

        return result

    @classmethod
    def to_dict(cls, obj: SupportsDynamicAttributes) -> dict[str, object]:
        """Convert object to dictionary with advanced serialization."""
        result: dict[str, object] = {}

        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            for key, value in obj_dict.items():
                if key.startswith("_"):
                    continue

                # Try to_dict_basic first
                if isinstance(value, HasToDictBasic):
                    try:
                        result[key] = value.to_dict_basic()
                        continue
                    except Exception as e:
                        msg = f"Failed to get object attributes: {e}"
                        raise FlextOperationError(
                            msg,
                            validation_details={"object": obj},
                        ) from e

                # Try to_dict
                if isinstance(value, HasToDict):
                    try:
                        result[key] = value.to_dict()
                        continue
                    except Exception as e:
                        msg = f"Failed to get object attributes: {e}"
                        raise FlextOperationError(
                            msg,
                            validation_details={"object": obj},
                        ) from e

                # Handle lists
                if isinstance(value, list):
                    serialized_list: list[object] = []
                    for item in value:
                        if isinstance(item, HasToDictBasic):
                            try:
                                serialized_list.append(item.to_dict_basic())
                                continue
                            except Exception as e:
                                msg = f"Failed to get object attributes: {e}"
                                raise FlextOperationError(
                                    msg,
                                    validation_details={"object": obj},
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
            raise FlextOperationError(
                msg,
                validation_details={"object": obj},
            ) from e

        return result

    @classmethod
    def to_json(cls, obj: SupportsDynamicAttributes, indent: int | None = None) -> str:
        """Convert object to JSON string."""
        data = cls.to_dict_basic(obj)
        return json.dumps(data, indent=indent, default=str)

    @classmethod
    def load_from_dict(
        cls, obj: SupportsDynamicAttributes, data: dict[str, object]
    ) -> None:
        """Load object attributes from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    @classmethod
    def load_from_json(
        cls, obj: SupportsDynamicAttributes, json_str: str
    ) -> FlextResult[None]:
        """Load object attributes from JSON string."""
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return FlextResult[None].fail("JSON must be an object")
            cls.load_from_dict(obj, data)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"JSON parsing failed: {e}")

    @classmethod
    def _serialize_value(cls, value: object) -> object:
        """Serialize a value for JSON compatibility."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [cls._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: cls._serialize_value(v) for k, v in value.items()}
        # For complex objects, use string representation
        return str(value)

    # =============================================================================
    # VALIDATION FUNCTIONALITY - Data validation patterns
    # =============================================================================

    @classmethod
    def initialize_validation(cls, obj: SupportsDynamicAttributes) -> None:
        """Initialize validation state on an object."""
        obj._validation_errors = []
        obj._is_valid = False
        obj._validation_initialized = True

    @classmethod
    def validate_required_fields(
        cls, obj: SupportsDynamicAttributes, fields: list[str]
    ) -> FlextResult[None]:
        """Validate that required fields are present and not empty."""
        missing_fields = []

        for field in fields:
            value = getattr(obj, field, None)
            if value is None or (isinstance(value, str) and len(value.strip()) == 0):
                missing_fields.append(field)

        if missing_fields:
            return FlextResult[None].fail(
                f"Missing required fields: {', '.join(missing_fields)}",
                error_code="MISSING_REQUIRED_FIELDS",
            )

        return FlextResult[None].ok(None)

    @classmethod
    def validate_field_types(
        cls, obj: SupportsDynamicAttributes, field_types: dict[str, type]
    ) -> FlextResult[None]:
        """Validate that fields match expected types."""
        type_errors = []

        for field, expected_type in field_types.items():
            value = getattr(obj, field, None)
            if value is not None and not isinstance(value, expected_type):
                type_errors.append(
                    f"{field}: expected {expected_type.__name__}, got {type(value).__name__}"
                )

        if type_errors:
            return FlextResult[None].fail(
                f"Type validation errors: {'; '.join(type_errors)}",
                error_code="TYPE_VALIDATION_FAILED",
            )

        return FlextResult[None].ok(None)

    @classmethod
    def add_validation_error(cls, obj: SupportsDynamicAttributes, error: str) -> None:
        """Add a validation error to an object."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        errors = getattr(obj, "_validation_errors", [])
        errors.append(str(error))
        obj._validation_errors = errors
        obj._is_valid = False

    @classmethod
    def clear_validation_errors(cls, obj: SupportsDynamicAttributes) -> None:
        """Clear all validation errors."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        obj._validation_errors = []
        obj._is_valid = False

    @classmethod
    def get_validation_errors(cls, obj: SupportsDynamicAttributes) -> list[str]:
        """Get all validation errors."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        return list(getattr(obj, "_validation_errors", []))

    @classmethod
    def is_valid(cls, obj: SupportsDynamicAttributes) -> bool:
        """Check if object is valid (no validation errors)."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        errors = getattr(obj, "_validation_errors", [])
        return len(errors) == 0 and getattr(obj, "_is_valid", False)

    @classmethod
    def mark_valid(cls, obj: SupportsDynamicAttributes) -> None:
        """Mark object as valid."""
        if not hasattr(obj, "_validation_initialized"):
            cls.initialize_validation(obj)

        obj._is_valid = True

    # =============================================================================
    # STATE MANAGEMENT FUNCTIONALITY - Object lifecycle patterns
    # =============================================================================

    @classmethod
    def initialize_state(
        cls, obj: SupportsDynamicAttributes, initial_state: str = "created"
    ) -> None:
        """Initialize object state."""
        obj._state = initial_state
        obj._state_history = [initial_state]
        obj._state_initialized = True

    @classmethod
    def get_state(cls, obj: SupportsDynamicAttributes) -> str:
        """Get current state."""
        if not hasattr(obj, "_state_initialized"):
            cls.initialize_state(obj)
        return getattr(obj, "_state", "created")

    @classmethod
    def set_state(
        cls, obj: SupportsDynamicAttributes, new_state: str
    ) -> FlextResult[None]:
        """Set new state with validation."""
        if not isinstance(new_state, str) or len(new_state.strip()) == 0:
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
        return FlextResult[None].ok(None)

    @classmethod
    def get_state_history(cls, obj: SupportsDynamicAttributes) -> list[str]:
        """Get state change history."""
        if not hasattr(obj, "_state_initialized"):
            cls.initialize_state(obj)
        return list(getattr(obj, "_state_history", ["created"]))

    # =============================================================================
    # CACHING FUNCTIONALITY - Memoization patterns
    # =============================================================================

    @classmethod
    def get_cached_value(
        cls, obj: SupportsDynamicAttributes, key: str
    ) -> FlextResult[object]:
        """Get cached value by key."""
        if not hasattr(obj, "_cache"):
            obj._cache = {}

        cache = getattr(obj, "_cache", {})
        if key in cache:
            return FlextResult[object].ok(cache[key])

        return FlextResult[object].fail(f"Cache key not found: {key}")

    @classmethod
    def set_cached_value(
        cls, obj: SupportsDynamicAttributes, key: str, value: object
    ) -> None:
        """Set cached value by key."""
        if not hasattr(obj, "_cache"):
            obj._cache = {}

        cache = getattr(obj, "_cache", {})
        cache[key] = value
        obj._cache = cache

    @classmethod
    def clear_cache(cls, obj: SupportsDynamicAttributes) -> None:
        """Clear all cached values."""
        obj._cache = {}

    @classmethod
    def has_cached_value(cls, obj: SupportsDynamicAttributes, key: str) -> bool:
        """Check if value is cached."""
        if not hasattr(obj, "_cache"):
            return False

        cache = getattr(obj, "_cache", {})
        return key in cache

    @classmethod
    def get_cache_key(cls, obj: SupportsDynamicAttributes) -> str:
        """Generate cache key for an object."""
        if cls.has_id(obj):
            entity_id = cls.ensure_id(obj)
            return f"{obj.__class__.__name__}:{entity_id}"
        return f"{obj.__class__.__name__}:{hash(str(obj.__dict__))}"

    # =============================================================================
    # TIMING FUNCTIONALITY - Performance tracking patterns
    # =============================================================================

    @classmethod
    def start_timing(cls, obj: SupportsDynamicAttributes) -> float:
        """Start timing operation and return start time."""
        start_time = time.time()
        obj._start_time = start_time
        return start_time

    @classmethod
    def stop_timing(cls, obj: SupportsDynamicAttributes) -> float:
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
    def get_last_elapsed_time(cls, obj: SupportsDynamicAttributes) -> float:
        """Get last elapsed time."""
        elapsed_times = getattr(obj, "_elapsed_times", [])
        return elapsed_times[-1] if elapsed_times else 0.0

    @classmethod
    def get_average_elapsed_time(cls, obj: SupportsDynamicAttributes) -> float:
        """Get average elapsed time."""
        elapsed_times = getattr(obj, "_elapsed_times", [])
        return sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    @classmethod
    def clear_timing_history(cls, obj: SupportsDynamicAttributes) -> None:
        """Clear timing history."""
        obj._elapsed_times = []

    # =============================================================================
    # ERROR HANDLING FUNCTIONALITY - Exception patterns
    # =============================================================================

    @classmethod
    def handle_error(
        cls, obj: SupportsDynamicAttributes, error: Exception, context: str = ""
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
        obj: SupportsDynamicAttributes,
        operation: FlextDecoratedFunction[object],
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation safely with error handling."""
        try:
            result = operation(*args, **kwargs)
            return FlextResult[object].ok(result)
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

        if isinstance(obj1, HasToDict) and isinstance(obj2, HasToDict):
            return obj1.to_dict() == obj2.to_dict()

        return obj1.__dict__ == obj2.__dict__

    @classmethod
    def object_hash(cls, obj: SupportsDynamicAttributes) -> int:
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
            cls.to_dict_basic(cast("SupportsDynamicAttributes", obj1))
            if isinstance(obj1, HasToDictBasic)
            else obj1.__dict__
        )
        right = (
            cls.to_dict_basic(cast("SupportsDynamicAttributes", obj2))
            if isinstance(obj2, HasToDictBasic)
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
        return (HasToDictBasic, HasToDict)

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
# LEGACY COMPATIBILITY LAYER - Maintain existing imports
# =============================================================================


# Base compatibility class that enables mixin inheritance for legacy code
class _CompatibilityMixin:
    """Base compatibility class that delegates to FlextMixins methods."""

    def mixin_setup(self) -> None:
        """Setup mixin functionality."""


# All original mixin classes as compatibility facades
class FlextLoggableMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins.log_* methods."""

    @property
    def logger(self) -> FlextLoggerProtocol:
        """Get logger via FlextMixins."""
        return FlextMixins.get_logger(self)

    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation via FlextMixins."""
        FlextMixins.log_operation(self, operation, **kwargs)

    def log_info(self, message: str, **kwargs: object) -> None:
        """Log info via FlextMixins."""
        FlextMixins.log_info(self, message, **kwargs)

    def log_error(self, message: str, **kwargs: object) -> None:
        """Log error via FlextMixins."""
        FlextMixins.log_error(self, message, **kwargs)

    def log_debug(self, message: str, **kwargs: object) -> None:
        """Log debug via FlextMixins."""
        FlextMixins.log_debug(self, message, **kwargs)


class FlextTimestampMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins timestamp methods."""

    def update_timestamp(self) -> None:
        """Update timestamp via FlextMixins."""
        FlextMixins.update_timestamp(self)

    @property
    def created_at(self) -> float:
        """Get created timestamp via FlextMixins."""
        return FlextMixins.get_created_at(self)

    @property
    def updated_at(self) -> float:
        """Get updated timestamp via FlextMixins."""
        return FlextMixins.get_updated_at(self)

    def get_age_seconds(self) -> float:
        """Get age via FlextMixins."""
        return FlextMixins.get_age_seconds(self)


class FlextIdentifiableMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins ID methods."""

    @property
    def id(self) -> str:
        """Get ID via FlextMixins."""
        return FlextMixins.ensure_id(self)

    @id.setter
    def id(self, value: str) -> None:
        """Set ID via FlextMixins."""
        result = FlextMixins.set_id(self, value)
        if result.is_failure:
            raise FlextValidationError(result.error or "Invalid entity ID")

    def get_id(self) -> str:
        """Get ID via FlextMixins."""
        return FlextMixins.ensure_id(self)

    def has_id(self) -> bool:
        """Check ID via FlextMixins."""
        return FlextMixins.has_id(self)


class FlextValidatableMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins validation methods."""

    def validate(self) -> FlextResult[None]:
        """Validate via FlextMixins."""
        if FlextMixins.is_valid(self):
            return FlextResult[None].ok(None)
        errors = FlextMixins.get_validation_errors(self)
        return FlextResult[None].fail(f"Validation failed: {'; '.join(errors)}")

    @property
    def is_valid(self) -> bool:
        """Check validity via FlextMixins."""
        return FlextMixins.is_valid(self)

    def add_validation_error(self, error: str) -> None:
        """Add validation error via FlextMixins."""
        FlextMixins.add_validation_error(self, error)

    def clear_validation_errors(self) -> None:
        """Clear validation errors via FlextMixins."""
        FlextMixins.clear_validation_errors(self)

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors via FlextMixins."""
        return FlextMixins.get_validation_errors(self)

    def has_validation_errors(self) -> bool:
        """Check if has validation errors via FlextMixins."""
        return len(FlextMixins.get_validation_errors(self)) > 0


class FlextSerializableMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins serialization methods."""

    def to_dict(self) -> dict[str, object]:
        """Convert to dict via FlextMixins."""
        return FlextMixins.to_dict(self)

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dict via FlextMixins."""
        return FlextMixins.to_dict_basic(self)

    def to_json(self) -> str:
        """Convert to JSON via FlextMixins."""
        return FlextMixins.to_json(self)

    def load_from_dict(self, data: dict[str, object]) -> None:
        """Load from dict via FlextMixins."""
        FlextMixins.load_from_dict(self, data)

    def load_from_json(self, json_str: str) -> None:
        """Load from JSON via FlextMixins."""
        result = FlextMixins.load_from_json(self, json_str)
        if result.is_failure:
            raise ValueError(result.error)


# Additional compatibility classes for complete legacy support
class FlextTimingMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins timing methods."""

    def start_timing(self) -> float:
        """Start timing via FlextMixins."""
        return FlextMixins.start_timing(self)

    def stop_timing(self) -> float:
        """Stop timing via FlextMixins."""
        return FlextMixins.stop_timing(self)


# Composite mixins for legacy compatibility
class FlextEntityMixin(
    FlextTimestampMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Legacy compatibility - composite entity mixin."""


class FlextValueObjectMixin(
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Legacy compatibility - composite value object mixin."""


class FlextServiceMixin(
    FlextLoggableMixin,
    FlextValidatableMixin,
):
    """Legacy compatibility - composite service mixin."""


# Abstract base classes for compatibility
class FlextAbstractMixin(_CompatibilityMixin):
    """Abstract base for compatibility."""


FlextAbstractTimestampMixin = FlextAbstractMixin
FlextAbstractIdentifiableMixin = FlextAbstractMixin
FlextAbstractLoggableMixin = FlextAbstractMixin
FlextAbstractValidatableMixin = FlextAbstractMixin
FlextAbstractSerializableMixin = FlextAbstractMixin
FlextAbstractEntityMixin = FlextAbstractMixin
FlextAbstractServiceMixin = FlextAbstractMixin


# Additional compatibility classes needed by flext-cli
class FlextComparableMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins comparison methods."""

    def __eq__(self, other: object) -> bool:
        """Check equality via FlextMixins."""
        return FlextMixins.objects_equal(self, other)

    def __hash__(self) -> int:
        """Generate hash via FlextMixins."""
        return FlextMixins.object_hash(self)

    def __lt__(self, other: object) -> bool:
        """Compare via FlextMixins."""
        return FlextMixins.compare_objects(self, other) < 0

    def compare_to(self, other: object) -> int:
        """Compare objects via FlextMixins."""
        return FlextMixins.compare_objects(self, other)


FlextTimestampableMixin = FlextTimestampMixin  # Alias for naming consistency
FlextStateableMixin = FlextAbstractMixin  # State management compatibility
FlextCacheableMixin = FlextAbstractMixin  # Caching compatibility
FlextObservableMixin = FlextAbstractMixin  # Observer pattern compatibility
FlextConfigurableMixin = FlextAbstractMixin  # Configuration compatibility

# =============================================================================
# TIER 1 MODULE PATTERN - EXPORTS
# =============================================================================

__all__: list[str] = [
    "FlextAbstractEntityMixin",
    "FlextAbstractIdentifiableMixin",
    "FlextAbstractLoggableMixin",
    "FlextAbstractMixin",
    "FlextAbstractSerializableMixin",
    "FlextAbstractServiceMixin",
    "FlextAbstractTimestampMixin",
    "FlextAbstractValidatableMixin",
    "FlextCacheableMixin",
    "FlextComparableMixin",
    "FlextConfigurableMixin",
    "FlextEntityMixin",
    "FlextIdentifiableMixin",
    # =======================================================================
    # LEGACY COMPATIBILITY LAYER - All original class names
    # =======================================================================
    "FlextLoggableMixin",
    "FlextMixins",  # 🎯 SINGLE EXPORT: All mixin functionality consolidated
    "FlextObservableMixin",
    "FlextSerializableMixin",
    "FlextServiceMixin",
    "FlextStateableMixin",
    "FlextTimestampMixin",
    "FlextTimestampableMixin",
    "FlextTimingMixin",
    "FlextValidatableMixin",
    "FlextValueObjectMixin",
    "HasToDict",  # Protocol for to_dict support
    # =======================================================================
    # PROTOCOLS - Runtime-checkable interfaces (kept for type checking)
    # =======================================================================
    "HasToDictBasic",  # Protocol for to_dict_basic support
    # =======================================================================
    # NOTE: Legacy classes delegate to FlextMixins methods for TRUE consolidation
    # Modern usage: FlextMixins.method(obj) - direct method calls
    # Legacy usage: class MyClass(FlextLoggableMixin) - inheritance still works
    # =======================================================================
]
