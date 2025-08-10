"""Complete Mixin Compatibility Layer for Tests.

⚠️  IMPORTANT: This module provides COMPATIBILITY classes for the hybrid approach.
    These classes bridge between the new abstract mixin architecture and old
    test expectations.

Purpose:
    Instead of modifying the core mixins, we provide complete compatibility
    implementations
    in this separate module. Tests can use these legacy-compatible classes while the
    core architecture remains clean and follows SOLID/DDD principles.

Architecture:
    - NEW: Abstract mixins in base_mixins.py with proper SOLID design
    - COMPATIBILITY: Complete implementations here for backward compatibility
    - TESTS: Use compatibility classes for seamless transition
    - FUTURE: Tests can be gradually migrated to use new abstract mixins

Usage in Tests:
    from flext_core.mixin_compat import (
        LegacyCompatibleTimestampMixin,
        LegacyCompatibleEntityMixin,
        # ... other compatibility classes
    )
"""

from __future__ import annotations

import json
import logging
import time

from flext_core.exceptions import FlextValidationError
from flext_core.result import FlextResult


class LegacyCompatibleTimestampMixin:
    """Backward-compatible TimestampMixin for tests."""

    def __init__(self) -> None:
        super().__init__()
        self._created_at = time.time()
        self._updated_at = time.time()

    def get_timestamp(self) -> float:
        """Default timestamp implementation for compatibility."""
        return time.time()

    def update_timestamp(self) -> None:
        """Default timestamp update for compatibility."""
        self._update_timestamp()

    def mixin_setup(self) -> None:
        """Setup mixin (no-op for compatibility)."""

    @property
    def created_at(self) -> float:
        """Get creation timestamp."""
        return getattr(self, "_created_at", self.get_timestamp())

    @property
    def updated_at(self) -> float:
        """Get update timestamp."""
        return getattr(self, "_updated_at", self.get_timestamp())

    def _update_timestamp(self) -> None:
        """Update timestamp."""
        self._updated_at = time.time()

    def get_age_seconds(self) -> float:
        """Get age in seconds since creation."""
        return time.time() - self.created_at


class LegacyCompatibleIdentifiableMixin:
    """Backward-compatible IdentifiableMixin for tests."""

    def __init__(self) -> None:
        super().__init__()
        self._id = "default-id"

    def get_id(self) -> str:
        """Default ID implementation for compatibility."""
        return getattr(self, "_id", "default-id")

    def mixin_setup(self) -> None:
        """Setup mixin (no-op for compatibility)."""

    @property
    def id(self) -> str:
        """Get ID property."""
        return getattr(self, "_id", "default-id")

    @id.setter
    def id(self, value: str) -> None:
        """Set ID property."""
        self._id = value

    def set_id(self, entity_id: str) -> None:
        """Set entity ID with validation."""
        if entity_id and entity_id.strip():
            self._id = entity_id
        else:
            error_msg = "Invalid entity ID: empty string"
            raise FlextValidationError(
                error_msg, validation_details={"field": "entity_id", "value": entity_id}
            )

    def has_id(self) -> bool:
        """Check if entity has ID set."""
        return hasattr(self, "_id") and self._id is not None


class LegacyCompatibleValidatableMixin:
    """Backward-compatible ValidatableMixin for tests."""

    def __init__(self) -> None:
        super().__init__()
        self._validation_errors: list[str] = []
        self._is_valid: bool = False

    def mixin_setup(self) -> None:
        """Setup mixin (no-op for compatibility)."""

    @property
    def is_valid(self) -> bool:
        """Get validation status."""
        return getattr(self, "_is_valid", False)

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors list."""
        return getattr(self, "_validation_errors", [])

    def add_validation_error(self, message: str) -> None:
        """Add a validation error message."""
        if not hasattr(self, "_validation_errors"):
            self._validation_errors = []
        if message:  # Only add non-empty messages
            self._validation_errors.append(message)
        self._is_valid = False

    def clear_validation_errors(self) -> None:
        """Clear validation errors."""
        if hasattr(self, "_validation_errors"):
            self._validation_errors.clear()
        self._is_valid = False

    def validate_data(self) -> bool:
        """Validate data integrity."""
        return self.is_valid

    def validate(self) -> FlextResult[None]:
        """Validate using FlextResult pattern."""
        if self.validate_data():
            return FlextResult.ok(None)
        return FlextResult.fail("Validation failed")


class LegacyCompatibleSerializableMixin:
    """Backward-compatible SerializableMixin for tests."""

    def mixin_setup(self) -> None:
        """Setup mixin (no-op for compatibility)."""

    def to_dict_basic(self) -> dict[str, object]:  # noqa: PLR0912,C901
        """Basic serialization for test compatibility."""
        result: dict[str, object] = {}
        for attr_name in dir(self):
            if not attr_name.startswith("_") and not callable(getattr(self, attr_name)):
                try:
                    value = getattr(self, attr_name)
                    # Skip None values for complex data test compatibility
                    if value is None:
                        continue
                    if isinstance(value, (str, int, float, bool)):
                        result[attr_name] = value
                    elif isinstance(value, (list, dict)):
                        # For collections, try to serialize contents
                        if isinstance(value, list):
                            serialized_list: list[object] = []
                            for item in value:
                                if hasattr(item, "to_dict_basic"):
                                    serialized_list.append(item.to_dict_basic())
                                else:
                                    serialized_list.append(item)
                            result[attr_name] = serialized_list
                        else:
                            # Ensure dict has JSON-serializable primitives only
                            safe_dict: dict[str, object] = {}
                            for k, v in value.items():
                                if isinstance(v, (str, int, float, bool)):
                                    safe_dict[str(k)] = v
                            result[attr_name] = safe_dict
                    elif hasattr(value, "to_dict_basic"):
                        result[attr_name] = value.to_dict_basic()
                except (TypeError, AttributeError):
                    # Skip problematic attributes
                    continue
        return result

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict_basic())

    def load_from_dict(self, data: dict[str, object]) -> None:
        """Load from dictionary (basic implementation)."""
        for key, value in data.items():
            if not key.startswith("_") and hasattr(self, key):
                setattr(self, key, value)


class LegacyCompatibleLoggableMixin:
    """Backward-compatible LoggableMixin for tests."""

    def __init__(self) -> None:
        super().__init__()
        self._logger: logging.Logger | None = None

    def mixin_setup(self) -> None:
        """Setup mixin (no-op for compatibility)."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        if self._logger is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)
        return self._logger

    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation with context."""
        logger = self.logger
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(f"Operation: {operation} ({extra_info})")
        else:
            logger.info(f"Operation: {operation}")

    def log_with_context(self, level: str, message: str, **context: object) -> None:
        """Log message with additional context."""
        logger = self.logger
        log_method = getattr(logger, level.lower(), logger.info)
        if context:
            context_info = ", ".join(f"{k}={v}" for k, v in context.items())
            log_method(f"{message} ({context_info})")
        else:
            log_method(message)


class LegacyCompatibleTimingMixin:
    """Backward-compatible TimingMixin for tests."""

    def _start_timing(self) -> float:
        """Start timing measurement."""
        return time.perf_counter()

    def _get_execution_time_ms(self, start_time: float) -> float:
        """Get execution time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000.0

    def _get_execution_time_ms_rounded(self, start_time: float) -> float:
        """Get execution time in milliseconds (rounded)."""
        return round(self._get_execution_time_ms(start_time), 2)


class LegacyCompatibleComparableMixin:
    """Backward-compatible ComparableMixin for tests."""

    def _comparison_key(self) -> object:
        """Get key for comparison (override in subclasses)."""
        return str(self)

    def _get_comparison_key(self) -> tuple[object, ...]:
        """Get key for comparison (backward compatibility)."""
        return (self._comparison_key(),)

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, self.__class__):
            return False
        return self._get_comparison_key() == other._get_comparison_key()

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        if not isinstance(other, self.__class__):
            return False
        return self._get_comparison_key() < other._get_comparison_key()

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        """Greater than comparison."""
        if not isinstance(other, self.__class__):
            return False
        return self._get_comparison_key() > other._get_comparison_key()

    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison."""
        return self > other or self == other

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self._get_comparison_key())


class LegacyCompatibleCacheableMixin:
    """Backward-compatible CacheableMixin for tests."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[str, object] = {}

    def cache_set(self, key: str, value: object) -> None:
        """Set cache value."""
        self._cache[key] = value

    def cache_get(self, key: str) -> object:
        """Get cache value."""
        return self._cache.get(key)

    def cache_clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get cache size."""
        return len(self._cache)

    def cache_remove(self, key: str) -> bool:
        """Remove cache entry."""
        return self._cache.pop(key, None) is not None

    def get_cache_key(self) -> str:
        """Get cache key."""
        return f"{self.__class__.__name__}:{id(self)}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return 300  # 5 minutes default

    def should_cache(self) -> bool:
        """Determine if object should be cached."""
        return True

    def invalidate_cache(self) -> None:
        """Invalidate cache entry."""
        self._cache.clear()


# =============================================================================
# COMPOSITE MIXINS - Combining multiple capabilities
# =============================================================================


class LegacyCompatibleEntityMixin(
    LegacyCompatibleTimestampMixin,
    LegacyCompatibleIdentifiableMixin,
    LegacyCompatibleValidatableMixin,
):
    """Backward-compatible EntityMixin combining all functionality for tests."""

    def __init__(self, entity_id: str | None = None) -> None:
        LegacyCompatibleTimestampMixin.__init__(self)
        LegacyCompatibleIdentifiableMixin.__init__(self)
        LegacyCompatibleValidatableMixin.__init__(self)
        if entity_id:
            self.set_id(entity_id)

    def get_domain_events(self) -> list[object]:
        """Default domain events implementation for compatibility."""
        return getattr(self, "_domain_events", [])

    def clear_domain_events(self) -> None:
        """Default clear domain events implementation for compatibility."""
        if hasattr(self, "_domain_events"):
            self._domain_events.clear()

    def mixin_setup(self) -> None:
        """Default mixin setup for compatibility."""

    def _compare_basic(self, other: object) -> int:
        """Default comparison implementation for compatibility."""
        return 0 if str(self) == str(other) else 1


class LegacyCompatibleValueObjectMixin(
    LegacyCompatibleComparableMixin,
    LegacyCompatibleValidatableMixin,
):
    """Backward-compatible ValueObjectMixin for tests."""

    def __init__(self) -> None:
        LegacyCompatibleComparableMixin.__init__(self)
        LegacyCompatibleValidatableMixin.__init__(self)


class LegacyCompatibleServiceMixin(
    LegacyCompatibleLoggableMixin,
    LegacyCompatibleValidatableMixin,
):
    """Backward-compatible ServiceMixin for tests."""

    def __init__(self, service_name: str = "default-service") -> None:
        LegacyCompatibleLoggableMixin.__init__(self)
        LegacyCompatibleValidatableMixin.__init__(self)
        self.service_name = service_name
        self.id = service_name
        self._service_initialized = True

    def get_service_name(self) -> str:
        """Get service name."""
        return self.service_name

    def initialize_service(self) -> FlextResult[None]:
        """Initialize service."""
        return FlextResult.ok(None)

    def get_service_info(self) -> dict[str, object]:
        """Get service info."""
        return {
            "service_name": self.service_name,
            "service_type": self.__class__.__name__,
            "is_initialized": True,
            "is_valid": False,  # Default validation state
        }


class LegacyCompatibleCommandMixin(
    LegacyCompatibleTimestampMixin,
    LegacyCompatibleValidatableMixin,
    LegacyCompatibleSerializableMixin,
):
    """Backward-compatible CommandMixin for tests."""

    def __init__(self) -> None:
        LegacyCompatibleTimestampMixin.__init__(self)
        LegacyCompatibleValidatableMixin.__init__(self)
        LegacyCompatibleSerializableMixin.__init__(self)

    def validate_and_set(self, **kwargs: object) -> None:
        """Validate and set command attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Update timestamp after setting values
        self._update_timestamp()


class LegacyCompatibleDataMixin(
    LegacyCompatibleValidatableMixin,
    LegacyCompatibleSerializableMixin,
):
    """Backward-compatible DataMixin for tests."""

    def __init__(self) -> None:
        LegacyCompatibleValidatableMixin.__init__(self)
        LegacyCompatibleSerializableMixin.__init__(self)

    def _compare_basic(self, other: object) -> int:
        """Compare data objects."""
        if not isinstance(other, self.__class__):
            return 1  # Different classes are considered unequal
        # Use string comparison by default
        self_str = str(self)
        other_str = str(other)
        return (self_str > other_str) - (self_str < other_str)


class LegacyCompatibleFullMixin(
    LegacyCompatibleEntityMixin,
    LegacyCompatibleLoggableMixin,
    LegacyCompatibleCacheableMixin,
    LegacyCompatibleSerializableMixin,
):
    """Backward-compatible FullMixin combining ALL capabilities for tests."""

    def __init__(self) -> None:
        LegacyCompatibleEntityMixin.__init__(self)
        LegacyCompatibleLoggableMixin.__init__(self)
        LegacyCompatibleCacheableMixin.__init__(self)
        LegacyCompatibleSerializableMixin.__init__(self)

    def _compare_basic(self, other: object) -> int:
        """Compare full objects using ID if available."""
        if not isinstance(other, self.__class__):
            return 1  # Different classes are considered unequal
        # Use ID comparison if both have IDs
        if hasattr(self, "entity_id") and hasattr(other, "entity_id"):
            self_id = str(self.entity_id)
            other_id = str(other.entity_id)
            return (self_id > other_id) - (self_id < other_id)
        # Fall back to string comparison
        self_str = str(self)
        other_str = str(other)
        return (self_str > other_str) - (self_str < other_str)


# =============================================================================
# EXPORTS - Compatibility mixins for test usage
# =============================================================================

__all__ = [
    # Core compatibility mixins
    "LegacyCompatibleCacheableMixin",
    # Composite compatibility mixins
    "LegacyCompatibleCommandMixin",
    "LegacyCompatibleComparableMixin",
    "LegacyCompatibleDataMixin",
    "LegacyCompatibleEntityMixin",
    "LegacyCompatibleFullMixin",
    "LegacyCompatibleIdentifiableMixin",
    "LegacyCompatibleLoggableMixin",
    "LegacyCompatibleSerializableMixin",
    "LegacyCompatibleServiceMixin",
    "LegacyCompatibleTimestampMixin",
    "LegacyCompatibleTimingMixin",
    "LegacyCompatibleValidatableMixin",
    "LegacyCompatibleValueObjectMixin",
]
