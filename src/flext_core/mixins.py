"""Reusable mixin classes for common functionality.

Provides mixin classes for timestamps, logging, validation, serialization,
and other common patterns. Designed for multiple inheritance without conflicts.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from flext_core.base_mixins import (
    FlextAbstractEntityMixin,
    FlextAbstractIdentifiableMixin,
    FlextAbstractLoggableMixin,
    FlextAbstractSerializableMixin,
    FlextAbstractServiceMixin,
    FlextAbstractTimestampMixin,
    FlextAbstractValidatableMixin,
)
from flext_core.exceptions import FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from flext_core.protocols import FlextLoggerProtocol
    from flext_core.typings import TEntityId

# =============================================================================
# UTILITY CLASSES - Helper functionality for mixins
# =============================================================================

# REFACTORED: FlextGenerators moved to utilities.py (single source of truth)
# Now imported from canonical location above


class FlextValidators:
    """Validation functions for mixin functionality.

    Provides common validation patterns used by mixins.
    """

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Validate non-empty string.

        Args:
            value: Value to validate.

        Returns:
            True if value is non-empty string.

        """
        return isinstance(value, str) and len(value.strip()) > 0


# =============================================================================
# TIMESTAMP MIXIN - Creation and update timestamp tracking
# =============================================================================


class FlextTimestampMixin(FlextAbstractTimestampMixin):
    """Concrete mixin for creation and update timestamp tracking.

    Provides automatic timestamp management using base abstractions
    following SOLID principles.
    """

    def mixin_setup(self) -> None:
        """Initialize timestamp state lazily."""
        # No-op; state created lazily in accessors
        return

    def update_timestamp(self) -> None:
        """Update timestamp - implements abstract method."""
        self._update_timestamp()

    def get_timestamp(self) -> float:
        """Get timestamp - implements abstract method."""
        self.__ensure_timestamp_state()
        return self._updated_at

    def _update_timestamp(self) -> None:
        """Update timestamp - implements abstract method."""
        # Ensure internal state exists
        _ = self.updated_at
        self._updated_at = FlextGenerators.generate_timestamp()

    def get_age_seconds(self) -> float:
        """Get age in seconds - implements abstract method."""
        current_time = FlextGenerators.generate_timestamp()
        return current_time - self.created_at

    def __ensure_timestamp_state(self) -> None:
        """Ensure timestamp state is initialized (lazy initialization)."""
        if not hasattr(self, "_timestamp_initialized"):
            current_time = FlextGenerators.generate_timestamp()
            self._created_at = current_time
            self._updated_at = current_time
            self._timestamp_initialized = True

    @property
    def created_at(self) -> float:
        """Get creation timestamp.

        Returns:
            Unix timestamp of creation.

        """
        self.__ensure_timestamp_state()
        return self._created_at

    @property
    def updated_at(self) -> float:
        """Get last update timestamp.

        Returns:
            Unix timestamp of last update.

        """
        self.__ensure_timestamp_state()
        return self._updated_at

    def get_age_in_seconds(self) -> float:
        """Get age in seconds since creation."""
        self.__ensure_timestamp_state()
        return FlextGenerators.generate_timestamp() - self._created_at


# =============================================================================
# IDENTIFIABLE MIXIN - Unique ID management
# =============================================================================


class FlextIdentifiableMixin(FlextAbstractIdentifiableMixin):
    """Concrete identifiable mixin using base abstractions.

    Foundation identification pattern with ID management following
    SOLID principles.
    """

    def get_id(self) -> TEntityId:
        """Get entity ID - implements abstract method."""
        # Try _id first (private attribute)
        id_value = getattr(self, "_id", None)
        if isinstance(id_value, str):
            return id_value

        # Try public id attribute
        id_value = getattr(self, "id", None)
        if isinstance(id_value, str):
            return id_value
        # Generate default ID if none exists
        generated_id = self._generate_default_id()
        self._id: TEntityId = generated_id
        return generated_id

    def _generate_default_id(self) -> TEntityId:
        """Generate default ID - implements abstract method."""
        return FlextGenerators.generate_id()

    def get_identity(self) -> TEntityId:
        """Get entity identity - implements abstract method."""
        return self.get_id()

    @property
    def id(self) -> TEntityId:
        """Get ID property."""
        return self.get_id()

    @id.setter
    def id(self, value: TEntityId) -> None:
        """Set ID property."""
        self._id = value

    def set_id(self, entity_id: TEntityId) -> None:
        """Set entity ID with validation."""
        if FlextValidators.is_non_empty_string(entity_id):
            self._id = entity_id
        else:
            msg = f"Invalid entity ID: {entity_id}"
            raise FlextValidationError(
                msg,
                validation_details={"field": "entity_id", "value": entity_id},
            )

    def generate_id(self) -> None:
        """Generate new unique ID."""
        self.id = FlextGenerators.generate_id()

    @property
    def entity_id(self) -> TEntityId:
        """Get entity ID (generates if not set)."""
        return self.id

    def has_id(self) -> bool:
        """Check if entity has ID set."""
        return self.id is not None


# =============================================================================
# LOGGABLE MIXIN - Structured logging with correlation IDs
# =============================================================================


class FlextLoggableMixin(FlextAbstractLoggableMixin):
    """Concrete loggable mixin using base abstractions.

    Structured logging capabilities following SOLID principles.
    """

    def mixin_setup(self) -> None:
        """Set up loggable mixin - implements abstract method."""
        # Initialize logger if not already present
        if not hasattr(self, "_logger"):
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = FlextLoggerFactory.get_logger(logger_name)

    @property
    def logger(self) -> FlextLoggerProtocol:
        """Get logger instance - implements abstract property."""
        if not hasattr(self, "_logger"):
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = FlextLoggerFactory.get_logger(logger_name)
        return self._logger

    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation with context - implements abstract method."""
        self.logger.info(f"Operation: {operation}", **kwargs)

    def log_with_context(self, level: str, message: str, **context: object) -> None:
        """Log message with additional context."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, **context)


# =============================================================================
# TIMING MIXIN - Performance timing and measurement
# =============================================================================


class FlextTimingMixin:
    """Centralized timing mixin.

    Performance timing and execution measurement patterns
    with high-precision measurements and monitoring support.
    """

    def _start_timing(self) -> float:
        """Start timing measurement."""
        return time.perf_counter()

    def _get_execution_time_ms(self, start_time: float) -> float:
        """Get execution time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000.0

    def _get_execution_time_ms_rounded(self, start_time: float) -> float:
        """Get execution time in milliseconds (rounded)."""
        return round(self._get_execution_time_ms(start_time), 2)


# =============================================================================
# VALIDATABLE MIXIN - Domain validation patterns
# =============================================================================


class FlextValidatableMixin(FlextAbstractValidatableMixin):
    """Concrete validatable mixin using base abstractions.

    Domain validation patterns following SOLID principles.
    """

    def __init__(self) -> None:
        """Initialize validation state."""
        super().__init__()
        # Deferred init: properties below are accessed lazily as needed
        self._validation_errors: list[str] = []
        self._is_valid: bool | None = None

    def mixin_setup(self) -> None:
        """No-op setup to satisfy abstract base contract."""
        return

    @property
    def validation_errors(self) -> list[str]:
        """Get collected validation errors (compat API)."""
        errors = getattr(self, "_validation_errors", None)
        if errors is None:
            self._validation_errors = []
            return []
        return list(errors)

    @property
    def is_valid(self) -> bool:
        """Return validation status; None treated as False for tests."""
        is_valid_val = getattr(self, "_is_valid", False)
        return bool(is_valid_val)

    def add_validation_error(self, message: str) -> None:
        """Add a validation error message and mark invalid."""
        if message:
            self._validation_errors.append(message)
        self._is_valid = False

    def clear_validation_errors(self) -> None:
        """Clear errors and reset status to False (unknown treated as False)."""
        self._validation_errors.clear()
        self._is_valid = False

    def mark_valid(self) -> None:
        """Explicitly mark object as valid."""
        self._is_valid = True

    def has_validation_errors(self) -> bool:
        """Check if there are collected validation errors."""
        return len(self._validation_errors) > 0

    def validate(self) -> FlextResult[None]:
        """Validate domain rules - implements abstract method."""
        result = self.validate_business_rules()
        if result.is_failure and result.error:
            self.add_validation_error(result.error)
        elif not self._validation_errors:
            self._is_valid = True
        return result

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules (override in subclasses)."""
        return FlextResult.ok(None)

    # is_valid property implemented above (compat semantics)


# =============================================================================
# SERIALIZABLE MIXIN - JSON/dict serialization patterns
# =============================================================================


class FlextSerializableMixin(FlextAbstractSerializableMixin):
    """Concrete serializable mixin using base abstractions.

    JSON and dictionary serialization following SOLID principles.
    """

    def mixin_setup(self) -> None:
        """No-op setup to satisfy abstract base contract."""
        return

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary - implements abstract method."""
        if hasattr(self, "model_dump"):
            # Type cast to satisfy MyPy - model_dump returns dict[str, object]
            return dict(self.model_dump())

        # Default implementation using __dict__
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_json(self) -> str:
        """Convert to JSON string - implements abstract method."""
        return json.dumps(self.to_dict())

    def load_from_dict(self, data: dict[str, object]) -> None:
        """Load from dictionary (override in subclasses)."""
        for key, value in data.items():
            if not key.startswith("_") and hasattr(self, key):
                setattr(self, key, value)

    # Back-compat helper used by various tests and payload serialization
    def to_dict_basic(self) -> dict[str, object]:
        """Convert to dict removing private/internal attributes."""
        data = self.to_dict()
        return {k: v for k, v in data.items() if not k.startswith("_")}


# =============================================================================
# COMPARABLE MIXIN - Comparison operations for value objects
# =============================================================================


class FlextComparableMixin:
    """Centralized comparable mixin.

    Comparison operations for value objects and entities
    with rich comparison operators and sorting support.
    """

    def _get_comparison_key(self) -> tuple[object, ...]:
        """Get key for comparison (override in subclasses)."""
        return (str(self),)

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, self.__class__):
            return False
        return self._get_comparison_key() == other._get_comparison_key()

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        if not isinstance(other, self.__class__):
            return False  # Different classes are considered unequal
        return self._get_comparison_key() < other._get_comparison_key()

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        """Greater than comparison."""
        if not isinstance(other, self.__class__):
            return False  # Different classes are considered unequal
        return self._get_comparison_key() > other._get_comparison_key()

    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison."""
        return self > other or self == other

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self._get_comparison_key())


# =============================================================================
# CACHEABLE MIXIN - Caching patterns with TTL support
# =============================================================================


class FlextCacheableMixin:
    """Centralized cacheable mixin.

    Caching patterns with TTL support, cache invalidation,
    and key generation methods.
    """

    def get_cache_key(self) -> str:
        """Get cache key (override in subclasses)."""
        return f"{self.__class__.__name__}:{id(self)}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds (override in subclasses)."""
        return 300  # 5 minutes default

    def should_cache(self) -> bool:
        """Determine if object should be cached."""
        return True

    def invalidate_cache(self) -> None:
        """Invalidate cache entry."""
        # Implementation would depend on cache backend


# =============================================================================
# ENTITY MIXIN - Domain entity patterns with lifecycle
# =============================================================================


class FlextEntityMixin(
    FlextAbstractEntityMixin,
    FlextTimestampMixin,
    FlextIdentifiableMixin,
):
    """Concrete entity mixin using base abstractions.

    Domain entity patterns following DDD and SOLID principles.
    """

    def __init__(self, entity_id: TEntityId | None = None) -> None:  # noqa: ARG002
        """Initialize entity mixin with proper MRO."""
        FlextAbstractEntityMixin.__init__(self)
        FlextTimestampMixin.__init__(self)
        FlextIdentifiableMixin.__init__(self)


# =============================================================================
# VALUE OBJECT MIXIN - Immutable value object patterns
# =============================================================================


class FlextValueObjectMixin(FlextComparableMixin, FlextValidatableMixin):
    """Centralized value object mixin.

    Immutable value object patterns with comparison, validation,
    and structural equality support.
    """

    # Implementation comes from composition of comparable and validatable mixins


# =============================================================================
# COMMAND MIXIN - Command pattern integration
# =============================================================================


class FlextCommandMixin(
    FlextTimestampMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Centralized command mixin for command pattern implementations.

    Command pattern combining timestamp, validation, and
    serialization functionality.
    """

    def validate_and_set(self, **kwargs: object) -> None:
        """Validate and set command attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Update timestamp after setting values
        self._update_timestamp()

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dictionary representation for commands."""
        data = self.to_dict()
        # Remove internal attributes
        return {k: v for k, v in data.items() if not k.startswith("_")}


# =============================================================================
# SERVICE MIXIN - Service pattern integration
# =============================================================================


class FlextServiceMixin(
    FlextAbstractServiceMixin,
    FlextLoggableMixin,
    FlextValidatableMixin,
):
    """Concrete service mixin using base abstractions.

    Service pattern following SOLID principles.
    """

    def __init__(self, service_name: str) -> None:
        """Initialize service with proper MRO."""
        FlextAbstractServiceMixin.__init__(self)
        FlextLoggableMixin.__init__(self)
        FlextValidatableMixin.__init__(self)
        self.service_name = service_name
        self.id = service_name
        self._service_initialized = True

    def get_service_info(self) -> dict[str, object]:
        """Get service info - implements abstract method."""
        return {
            "service_name": self.service_name,
            "service_type": self.__class__.__name__,
            "is_initialized": getattr(self, "_service_initialized", False),
            "is_valid": bool(self.is_valid),
        }


# =============================================================================
# DATA MIXIN - Data pattern integration
# =============================================================================


class FlextDataMixin(FlextValidatableMixin, FlextSerializableMixin):
    """Centralized data mixin for data pattern implementations.

    Data pattern combining validation, serialization,
    and comparison functionality.
    """

    def _compare_basic(self, other: object) -> int:
        """Compare data objects."""
        if not isinstance(other, self.__class__):
            return 1  # Different classes are considered unequal
        # Use string comparison by default
        self_str = str(self)
        other_str = str(other)
        return (self_str > other_str) - (self_str < other_str)

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dictionary representation for data."""
        data = self.to_dict()
        # Remove internal attributes
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def validate_data(self) -> bool:
        """Validate data integrity."""
        return self.validate().is_success


# =============================================================================
# FULL MIXIN - Complete functionality integration
# =============================================================================


class FlextFullMixin(
    FlextEntityMixin,
    FlextLoggableMixin,
    FlextCacheableMixin,
    FlextSerializableMixin,
    FlextValidatableMixin,
):
    """Centralized full mixin combining ALL capabilities.

    Comprehensive mixin combining entity, logging, caching,
    serialization, and validation functionality.
    """

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

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dictionary representation for full objects."""
        data = self.to_dict()
        # Remove internal attributes
        return {k: v for k, v in data.items() if not k.startswith("_")}

    # Simple cache implementation for testing
    def cache_set(self, key: str, value: object) -> None:
        """Set cache value."""
        if not hasattr(self, "_cache"):
            self._cache: dict[str, object] = {}
        self._cache[key] = value

    def cache_get(self, key: str) -> object:
        """Get cache value."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache.get(key)


# =============================================================================
# EXPORTS - Centralized mixin implementations
# =============================================================================

__all__: list[str] = [
    "FlextCacheableMixin",
    "FlextCommandMixin",
    "FlextComparableMixin",
    "FlextDataMixin",
    # Composite mixins
    "FlextEntityMixin",
    "FlextFullMixin",
    # Utility classes
    "FlextGenerators",
    "FlextIdentifiableMixin",
    "FlextLoggableMixin",
    "FlextSerializableMixin",
    "FlextServiceMixin",
    # Core mixins
    "FlextTimestampMixin",
    "FlextTimingMixin",
    "FlextValidatableMixin",
    "FlextValidators",
    "FlextValueObjectMixin",
]

# Total exports: 12 items - centralized mixin implementations
# These are the SINGLE SOURCE OF TRUTH for all mixin patterns in FLEXT
