"""FLEXT Core Mixins - Reusable behavioral patterns for enterprise applications.

Consolidates all mixin patterns following PEP8 naming conventions.
Provides abstract base classes and concrete implementations for common
behaviors like timestamps, logging, validation, and serialization.

Architecture:
    - Abstract Base Classes: Foundation mixin patterns
    - Concrete Implementations: Production-ready mixin classes
    - Composite Mixins: Higher-level combinations
    - Legacy Compatibility: Backward compatibility aliases

Usage:
    from flext_core import FlextTimestampMixin, FlextValidatableMixin

    class User(FlextTimestampMixin, FlextValidatableMixin):
        name: str
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flext_core.exceptions import FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from flext_core.protocols import FlextLoggerProtocol
    from flext_core.typings import TEntityId

# =============================================================================
# ABSTRACT BASE CLASSES - Foundation mixin patterns
# =============================================================================


class FlextAbstractMixin(ABC):
    """Abstract base class for all FLEXT mixins following SOLID principles.

    Provides foundation for implementing mixins with proper separation
    of concerns and dependency inversion.
    """

    @abstractmethod
    def mixin_setup(self) -> None:
        """Set up mixin - must be implemented by concrete mixins."""
        ...

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize abstract mixin and tolerate extra args for MRO chains."""
        del args, kwargs
        self._mixin_initialized = True


class FlextAbstractTimestampMixin(FlextAbstractMixin):
    """Abstract timestamp mixin for entity time tracking."""

    @abstractmethod
    def update_timestamp(self) -> None:
        """Update timestamp - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get timestamp - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up timestamp mixin."""
        self.update_timestamp()


class FlextAbstractIdentifiableMixin(FlextAbstractMixin):
    """Abstract identifiable mixin for entity identification."""

    @abstractmethod
    def get_id(self) -> TEntityId:
        """Get entity ID - must be implemented by subclasses."""
        ...

    @abstractmethod
    def set_id(self, entity_id: TEntityId) -> None:
        """Set entity ID - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up identifiable mixin."""


class FlextAbstractLoggableMixin(FlextAbstractMixin):
    """Abstract loggable mixin for entity logging."""

    @property
    @abstractmethod
    def logger(self) -> FlextLoggerProtocol:
        """Get logger instance - must be implemented by subclasses."""
        ...

    @abstractmethod
    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up loggable mixin."""


class FlextAbstractValidatableMixin(FlextAbstractMixin):
    """Abstract validatable mixin for entity validation."""

    @abstractmethod
    def validate(self) -> FlextResult[None]:
        """Validate entity - must be implemented by subclasses."""
        ...

    @property
    @abstractmethod
    def is_valid(self) -> bool:  # pragma: no cover - abstract property declaration
        """Check if entity is valid - must be implemented by subclasses."""
        raise NotImplementedError

    def mixin_setup(self) -> None:
        """Set up validatable mixin."""

    # ------------------------------------------------------------------
    # Default validation state management (available to all subclasses)
    # These methods are used heavily by examples and keep state locally
    # ------------------------------------------------------------------

    def clear_validation_errors(self) -> None:
        """Clear collected validation errors and reset valid flag."""
        self._ensure_validation_state()
        self._validation_errors.clear()
        self._is_valid = True

    def add_validation_error(self, message: str) -> None:
        """Add a validation error message and mark entity as invalid."""
        self._ensure_validation_state()
        self._validation_errors.append(str(message))
        self._is_valid = False

    @property
    def validation_errors(self) -> list[str]:
        """Return current validation errors (read-only list)."""
        self._ensure_validation_state()
        return list(self._validation_errors)

    def mark_valid(self) -> None:
        """Explicitly mark entity as valid (clears errors)."""
        self._ensure_validation_state()
        self._is_valid = True

    # Internal lazy state initializer for validation
    def _ensure_validation_state(self) -> None:
        if not hasattr(self, "_validation_initialized"):
            self._validation_errors: list[str] = []
            self._is_valid = True
            self._validation_initialized = True


class FlextAbstractSerializableMixin(FlextAbstractMixin):
    """Abstract serializable mixin for entity serialization."""

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary - must be implemented by subclasses."""
        ...

    @abstractmethod
    def load_from_dict(self, data: dict[str, object]) -> None:
        """Load from dictionary - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up serializable mixin."""


class FlextAbstractEntityMixin(FlextAbstractMixin):
    """Abstract entity mixin for domain entities."""

    @abstractmethod
    def get_domain_events(self) -> list[object]:
        """Get domain events - must be implemented by subclasses."""
        ...

    @abstractmethod
    def clear_domain_events(self) -> None:
        """Clear domain events - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up entity mixin."""


class FlextAbstractServiceMixin(FlextAbstractMixin):
    """Abstract service mixin for service classes."""

    @abstractmethod
    def get_service_name(self) -> str:
        """Get service name - must be implemented by subclasses."""
        ...

    @abstractmethod
    def initialize_service(self) -> FlextResult[None]:
        """Initialize service - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up service mixin."""


# =============================================================================
# UTILITY CLASSES - Helper functionality for mixins
# =============================================================================


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
# CONCRETE IMPLEMENTATIONS - Production-ready mixin classes
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
                field="entity_id",
                validation_details={"field": "entity_id", "value": entity_id},
            )

    def generate_id(self) -> None:
        """Generate new unique ID."""
        self.id = self._generate_default_id()

    @property
    def entity_id(self) -> TEntityId:
        """Get entity ID (generates if not set)."""
        return self.id

    def has_id(self) -> bool:
        """Check if entity has ID set."""
        return self.id is not None


class FlextLoggableMixin(FlextAbstractLoggableMixin):
    """Concrete loggable mixin using base abstractions.

    Provides automatic logging capability with structured logging
    following SOLID principles.
    """

    @property
    def logger(self) -> FlextLoggerProtocol:
        """Get logger instance - implements abstract method."""
        if not hasattr(self, "_logger"):
            self._logger = FlextLoggerFactory.get_logger(
                self.__class__.__module__ + "." + self.__class__.__name__,
            )
        return self._logger

    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation - implements abstract method."""
        self.logger.info(f"Operation: {operation}", **kwargs)

    def mixin_setup(self) -> None:
        """Set up loggable mixin."""
        # Initialize logger lazily
        _ = self.logger

    def log_info(self, message: str, **kwargs: object) -> None:
        """Log info message with context."""
        self.logger.info(message, **kwargs)

    def log_error(self, message: str, **kwargs: object) -> None:
        """Log error message with context."""
        self.logger.error(message, **kwargs)

    def log_debug(self, message: str, **kwargs: object) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)


class FlextTimingMixin:
    """Timing mixin for performance tracking.

    Provides start/stop timing functionality for performance measurement.
    """

    def __init__(self) -> None:
        """Initialize timing state."""
        super().__init__()
        self._start_time: float | None = None
        self._elapsed_times: list[float] = []

    def start_timing(self) -> None:
        """Start timing operation."""
        self._start_time = time.time()

    def stop_timing(self) -> float:
        """Stop timing and return elapsed seconds."""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        self._elapsed_times.append(elapsed)
        self._start_time = None
        return elapsed

    def _start_timing(self) -> float:
        """Compatibility helper used in examples: return start timestamp.

        Returns:
            The timestamp (seconds) when timing started.

        """
        start = time.time()
        self._start_time = start
        return start

    def _get_execution_time_ms(self, start_time: float) -> float:
        """Compatibility helper used in examples: return elapsed time in ms.

        Args:
            start_time: The start timestamp (seconds) previously returned by _start_timing.

        Returns:
            Elapsed time in milliseconds.

        """
        return (time.time() - start_time) * 1000.0

    def get_last_elapsed_time(self) -> float:
        """Get last elapsed time."""
        return self._elapsed_times[-1] if self._elapsed_times else 0.0

    def get_average_elapsed_time(self) -> float:
        """Get average elapsed time."""
        if not self._elapsed_times:
            return 0.0
        return sum(self._elapsed_times) / len(self._elapsed_times)

    def clear_timing_history(self) -> None:
        """Clear timing history."""
        self._elapsed_times.clear()


class FlextValidatableMixin(FlextAbstractValidatableMixin):
    """Concrete validatable mixin using base abstractions.

    Provides validation capability with structured error reporting
    following SOLID principles.
    """

    def validate(self) -> FlextResult[None]:
        """Validate entity - implements abstract method."""
        # Basic validation - can be overridden
        if not self.is_valid:
            return FlextResult.fail("Entity validation failed")
        return FlextResult.ok(None)

    @property
    def is_valid(self) -> bool:
        """Check if entity is valid - implements abstract method."""
        # Ensure validation state exists
        self._ensure_validation_state()
        return self._is_valid and len(self._validation_errors) == 0

    def clear_validation_errors(self) -> None:
        """Clear collected validation errors and reset valid flag to False (override base behavior)."""
        self._ensure_validation_state()
        self._validation_errors.clear()
        self._is_valid = (
            False  # Override: concrete mixin treats cleared as False rather than True
        )

    def _ensure_validation_state(self) -> None:
        """Override base initialization to start with False (concrete mixin behavior)."""
        if not hasattr(self, "_validation_initialized"):
            self._validation_errors: list[str] = []
            self._is_valid = False  # Concrete implementation starts invalid until explicitly validated
            self._validation_initialized = True

    def mixin_setup(self) -> None:
        """Set up validatable mixin."""
        # No initialization needed for basic validation


class FlextSerializableMixin(FlextAbstractSerializableMixin):
    """Concrete serializable mixin using base abstractions.

    Provides JSON serialization capability following SOLID principles.
    """

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary - implements abstract method."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if hasattr(value, "to_dict"):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_dict_basic(self) -> dict[str, object]:
        """Alias for to_dict to maintain backward compatibility."""
        return self.to_dict()

    def load_from_dict(self, data: dict[str, object]) -> None:
        """Load from dictionary - implements abstract method."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def load_from_json(self, json_str: str) -> None:
        """Load from JSON string."""
        data = json.loads(json_str)
        self.load_from_dict(data)

    def mixin_setup(self) -> None:
        """Set up serializable mixin."""
        # No initialization needed for serialization


class FlextComparableMixin:
    """Comparable mixin for ordering and equality."""

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, self.__class__):
            return False
        if hasattr(self, "to_dict") and hasattr(other, "to_dict"):
            return bool(self.to_dict() == other.to_dict())
        return bool(self.__dict__ == other.__dict__)

    def __hash__(self) -> int:
        """Generate hash."""
        if hasattr(self, "id"):
            return hash(self.id)
        return hash(str(self.__dict__))

    # Rich comparison operators for ordering in tests
    def __lt__(self, other: object) -> bool:
        """Less than comparison based on string representation of dict."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        left = self.to_dict_basic() if hasattr(self, "to_dict_basic") else self.__dict__
        right = (
            other.to_dict_basic() if hasattr(other, "to_dict_basic") else other.__dict__
        )
        return str(left) < str(right)

    def __le__(self, other: object) -> bool:  # pragma: no cover - trivial glue
        """Less than or equal comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:  # pragma: no cover - trivial glue
        """Greater than comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:  # pragma: no cover - trivial glue
        """Greater than or equal comparison."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return not self < other


class FlextCacheableMixin:
    """Cacheable mixin for cache key generation."""

    def get_cache_key(self) -> str:
        """Generate cache key."""
        if hasattr(self, "id"):
            return f"{self.__class__.__name__}:{self.id}"
        return f"{self.__class__.__name__}:{hash(str(self.__dict__))}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return 3600  # Default 1 hour


# =============================================================================
# COMPOSITE MIXINS - Higher-level combinations
# =============================================================================


class FlextEntityMixin(
    FlextTimestampMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Composite entity mixin combining common entity behaviors."""

    def mixin_setup(self) -> None:
        """Set up all component mixins."""
        super().mixin_setup()


class FlextValueObjectMixin(
    FlextValidatableMixin,
    FlextSerializableMixin,
    FlextComparableMixin,
):
    """Composite value object mixin for immutable values."""

    def mixin_setup(self) -> None:
        """Set up all component mixins."""
        super().mixin_setup()


class FlextCommandMixin(
    FlextIdentifiableMixin,
    FlextTimestampMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Composite command mixin for CQRS commands."""

    def mixin_setup(self) -> None:
        """Set up all component mixins."""
        super().mixin_setup()


class FlextServiceMixin(
    FlextLoggableMixin,
    FlextValidatableMixin,
):
    """Composite service mixin for service classes."""

    def mixin_setup(self) -> None:
        """Set up all component mixins."""
        super().mixin_setup()


class FlextDataMixin(
    FlextTimestampMixin,
    FlextSerializableMixin,
    FlextValidatableMixin,
):
    """Composite data mixin for data transfer objects."""

    def mixin_setup(self) -> None:
        """Set up all component mixins."""
        super().mixin_setup()


class FlextFullMixin(
    FlextTimestampMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
    FlextComparableMixin,
    FlextCacheableMixin,
):
    """Full-featured mixin with all common behaviors."""

    def mixin_setup(self) -> None:
        """Set up all component mixins."""
        super().mixin_setup()


# =============================================================================
# LEGACY COMPATIBILITY ALIASES - Maintain backward compatibility
# =============================================================================

# Legacy compatibility mixins (map to modern implementations)
LegacyCompatibleTimestampMixin = FlextTimestampMixin
LegacyCompatibleIdentifiableMixin = FlextIdentifiableMixin
LegacyCompatibleValidatableMixin = FlextValidatableMixin
LegacyCompatibleSerializableMixin = FlextSerializableMixin
LegacyCompatibleLoggableMixin = FlextLoggableMixin
LegacyCompatibleTimingMixin = FlextTimingMixin
LegacyCompatibleComparableMixin = FlextComparableMixin
LegacyCompatibleCacheableMixin = FlextCacheableMixin
LegacyCompatibleEntityMixin = FlextEntityMixin
LegacyCompatibleCommandMixin = FlextCommandMixin
LegacyCompatibleDataMixin = FlextDataMixin
LegacyCompatibleFullMixin = FlextFullMixin
LegacyCompatibleServiceMixin = FlextServiceMixin
LegacyCompatibleValueObjectMixin = FlextValueObjectMixin

# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [  # noqa: RUF022
    "FlextAbstractEntityMixin",
    "FlextAbstractIdentifiableMixin",
    "FlextAbstractLoggableMixin",
    # Abstract Base Classes
    "FlextAbstractMixin",
    "FlextAbstractSerializableMixin",
    "FlextAbstractServiceMixin",
    "FlextAbstractTimestampMixin",
    "FlextAbstractValidatableMixin",
    "FlextCacheableMixin",
    "FlextCommandMixin",
    "FlextComparableMixin",
    "FlextDataMixin",
    # Composite Mixins
    "FlextEntityMixin",
    "FlextFullMixin",
    "FlextIdentifiableMixin",
    "FlextLoggableMixin",
    "FlextSerializableMixin",
    "FlextServiceMixin",
    # Concrete Mixins
    "FlextTimestampMixin",
    "FlextTimingMixin",
    "FlextValidatableMixin",
    # Utilities
    "FlextValidators",
    "FlextValueObjectMixin",
    "LegacyCompatibleCacheableMixin",
    "LegacyCompatibleCommandMixin",
    "LegacyCompatibleComparableMixin",
    "LegacyCompatibleDataMixin",
    "LegacyCompatibleEntityMixin",
    "LegacyCompatibleFullMixin",
    "LegacyCompatibleIdentifiableMixin",
    "LegacyCompatibleLoggableMixin",
    "LegacyCompatibleSerializableMixin",
    # Legacy Compatibility
    "LegacyCompatibleTimestampMixin",
    "LegacyCompatibleTimingMixin",
    "LegacyCompatibleValidatableMixin",
    "LegacyCompatibleServiceMixin",
    "LegacyCompatibleValueObjectMixin",
]
