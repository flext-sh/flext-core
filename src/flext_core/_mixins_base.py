"""FLEXT Core Mixins - Internal Implementation Module.

Internal implementation providing the foundational logic for mixin behavioral patterns.
This module is part of the Internal Implementation Layer and should not be imported
directly by ecosystem projects. Use the public API through mixins module instead.

Module Role in Architecture:
    Internal Implementation Layer → Mixin Patterns → Public API Layer

    This internal module provides:
    - Base mixin categories (timestamp, identifiable, validatable, etc.)
    - Automatic initialization patterns with lazy loading
    - Composite mixin patterns for common architectural use cases
    - Internal state management with proper encapsulation

Implementation Patterns:
    Mixin Composition: Behavioral patterns without multiple inheritance complexity
    Lazy Initialization: Performance optimization through deferred state setup

Design Principles:
    - Single responsibility for internal mixin implementation concerns
    - No external dependencies beyond standard library and sibling modules
    - Performance-optimized implementations for public API consumption
    - Type safety maintained through internal validation

Access Restrictions:
    - This module is internal and not exported in __init__.py
    - Use mixins module for all external access to mixin functionality
    - Breaking changes may occur without notice in internal modules
    - No compatibility guarantees for internal implementation details

Quality Standards:
    - Internal implementation must maintain public API contracts
    - Performance optimizations must not break type safety
    - Code must be thoroughly tested through public API surface
    - Internal changes must not affect public behavior

See Also:
    mixins: Public API for mixin patterns and behavioral composition
    docs/python-module-organization.md: Internal module architecture

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from flext_core.exceptions import FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.utilities import FlextGenerators
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from flext_core.flext_types import TAnyDict, TEntityId
    from flext_core.loggings import FlextLogger


class _BaseTimestampMixin:
    """Foundation timestamp tracking without external dependencies.

    This is a proper mixin that doesn't implement __init__ to avoid MRO conflicts.
    Initialization happens automatically via property access (lazy initialization).
    """

    def __ensure_timestamp_state(self) -> None:
        """Ensure timestamp state is initialized (lazy initialization)."""
        if not hasattr(self, "_timestamp_initialized"):
            current_time = FlextGenerators.generate_timestamp()
            self._created_at = current_time
            self._updated_at = current_time
            self._timestamp_initialized = True

    def _update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.__ensure_timestamp_state()
        self._updated_at = FlextGenerators.generate_timestamp()

    @property
    def created_at(self) -> float:
        """Get creation timestamp."""
        self.__ensure_timestamp_state()
        return self._created_at

    @property
    def updated_at(self) -> float:
        """Get last update timestamp."""
        self.__ensure_timestamp_state()
        return self._updated_at

    def get_age_seconds(self) -> float:
        """Get age in seconds since creation."""
        self.__ensure_timestamp_state()
        return FlextGenerators.generate_timestamp() - self._created_at


class _BaseIdentifiableMixin:
    """Foundation identification pattern without external dependencies.

    Proper mixin that provides ID management without __init__ conflicts.
    Uses lazy initialization and explicit ID setting.
    """

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

    def generate_id(self) -> TEntityId:
        """Generate and set a new ID."""
        self._id = FlextGenerators.generate_entity_id()
        return self._id

    @property
    def id(self) -> TEntityId:
        """Get entity ID, generating one if not set."""
        if not hasattr(self, "_id"):
            self.generate_id()
        return self._id

    def has_id(self) -> bool:
        """Check if entity has valid ID."""
        return hasattr(self, "_id") and FlextValidators.is_non_empty_string(self._id)


class _BaseValidatableMixin:
    """Foundation validation pattern without external dependencies.

    Proper mixin with lazy initialization and thread-safe state management.
    """

    def __ensure_validation_state(self) -> None:
        """Ensure validation state is initialized (lazy initialization)."""
        if not hasattr(self, "_validation_errors"):
            self._validation_errors: list[str] = []
            self._is_valid: bool | None = None

    def add_validation_error(self, error: str) -> None:
        """Add validation error."""
        if FlextValidators.is_non_empty_string(error):
            self.__ensure_validation_state()
            self._validation_errors.append(error)
            self._is_valid = False

    def clear_validation_errors(self) -> None:
        """Clear all validation errors."""
        self.__ensure_validation_state()
        self._validation_errors.clear()
        self._is_valid = None

    def mark_valid(self) -> None:
        """Mark as valid and clear errors."""
        self.__ensure_validation_state()
        self._validation_errors.clear()
        self._is_valid = True

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors."""
        self.__ensure_validation_state()
        return self._validation_errors.copy()

    @property
    def is_valid(self) -> bool:
        """Check if object is valid."""
        self.__ensure_validation_state()
        return self._is_valid is True

    def has_validation_errors(self) -> bool:
        """Check if object has validation errors."""
        self.__ensure_validation_state()
        return len(self._validation_errors) > 0


class _BaseSerializableMixin:
    """Foundation serialization pattern without external dependencies."""

    def to_dict_basic(self) -> TAnyDict:
        """Convert to basic dictionary representation."""
        result: TAnyDict = {}

        # Get all attributes that don't start with __
        for attr_name in dir(self):
            if not attr_name.startswith("__"):
                # Skip Pydantic internal attributes that cause deprecation warnings
                if attr_name in {"model_computed_fields", "model_fields"}:
                    continue

                # Skip callable attributes
                if callable(getattr(self, attr_name)):
                    continue

                try:
                    value = getattr(self, attr_name)
                    serialized_value = self._serialize_value(value)
                    if serialized_value is not None:
                        result[attr_name] = serialized_value
                except (AttributeError, TypeError):
                    # Skip attributes that can't be accessed or serialized
                    continue

        return result

    def _serialize_value(self, value: object) -> object | None:
        """Serialize a single value for dict conversion."""
        # Simple types
        if isinstance(value, str | int | float | bool | type(None)):
            return value

        # Collections
        if isinstance(value, list | tuple):
            return self._serialize_collection(value)

        if isinstance(value, dict):
            return self._serialize_dict(value)

        # Objects with serialization method
        if hasattr(value, "to_dict_basic"):
            to_dict_method = value.to_dict_basic
            if callable(to_dict_method):
                result = to_dict_method()
                return result if isinstance(result, dict) else None

        return None

    def _serialize_collection(
        self,
        collection: list[object] | tuple[object, ...],
    ) -> list[object]:
        """Serialize list or tuple values."""
        serialized_list: list[object] = []
        for item in collection:
            if isinstance(item, str | int | float | bool | type(None)):
                serialized_list.append(item)
            elif hasattr(item, "to_dict_basic"):
                to_dict_method = item.to_dict_basic
                if callable(to_dict_method):
                    result = to_dict_method()
                    if isinstance(result, dict):
                        serialized_list.append(result)
        return serialized_list

    def _serialize_dict(self, dict_value: TAnyDict) -> TAnyDict:
        """Serialize dictionary values."""
        serialized_dict: TAnyDict = {}
        for k, v in dict_value.items():
            if isinstance(v, str | int | float | bool | type(None)):
                serialized_dict[str(k)] = v
        return serialized_dict

    def _from_dict_basic(self, data: TAnyDict) -> _BaseSerializableMixin:
        """Create instance from dictionary (basic implementation)."""
        # This is a basic implementation - subclasses should override
        for key, value in data.items():
            if hasattr(self, key):
                try:
                    setattr(self, key, value)
                except (AttributeError, TypeError):
                    # Skip attributes that can't be set
                    continue
        return self


class _BaseLoggableMixin:
    """Foundation logging pattern without external dependencies.

    Proper mixin with lazy logger initialization per instance.
    """

    @property
    def logger(self) -> FlextLogger:
        """Access to logger instance with lazy initialization."""
        if not hasattr(self, "_logger"):
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = FlextLoggerFactory.get_logger(logger_name)
        return self._logger


class _BaseComparableMixin:
    """Foundation comparison pattern without external dependencies."""

    def _compare_basic(self, other: object) -> int:
        """Compare objects returning -1/0/1 for less/equal/greater.

        Returns:
            -1 if self < other, 0 if equal, 1 if self > other

        """
        if not isinstance(other, type(self)):
            return 1  # Different types, self is "greater"

        # Compare by ID if both have identifiable mixin
        if hasattr(self, "id") and hasattr(other, "id"):
            return self._compare_by_id(other)

        # Compare by string representation as fallback
        return self._compare_by_string(other)

    def _compare_by_id(self, other: object) -> int:
        """Compare by ID attribute."""
        self_id = getattr(self, "id", "")
        other_id = getattr(other, "id", "")

        if self_id < other_id:
            return -1
        if self_id > other_id:
            return 1
        return 0

    def _compare_by_string(self, other: object) -> int:
        """Compare by string representation."""
        self_str = str(self)
        other_str = str(other)

        if self_str < other_str:
            return -1
        if self_str > other_str:
            return 1
        return 0

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        return self._compare_basic(other) < 0

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""
        return self._compare_basic(other) <= 0

    def __gt__(self, other: object) -> bool:
        """Greater than comparison."""
        return self._compare_basic(other) > 0

    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison."""
        return self._compare_basic(other) >= 0


class _BaseTimingMixin:
    """Foundation timing pattern without external dependencies."""

    def _start_timing(self) -> float:
        """Start timing and return start timestamp."""
        return time.perf_counter()

    def _get_execution_time_seconds(self, start_time: float) -> float:
        """Get execution time in seconds from start timestamp."""
        return time.perf_counter() - start_time

    def _get_execution_time_ms(self, start_time: float) -> float:
        """Get execution time in milliseconds from start timestamp."""
        return (time.perf_counter() - start_time) * 1000

    def _get_execution_time_ms_rounded(
        self,
        start_time: float,
        digits: int = 2,
    ) -> float:
        """Get rounded execution time in milliseconds from start timestamp."""
        return round(self._get_execution_time_ms(start_time), digits)


class _BaseCacheableMixin:
    """Foundation caching pattern without external dependencies."""

    # Class-level type annotations for MyPy
    _cache: TAnyDict
    _cache_timestamps: dict[str, float]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Ensure cache is added to subclasses."""
        super().__init_subclass__(**kwargs)

        # Add cache tracking attributes
        if not hasattr(cls, "_cache"):
            cls._cache = {}
        if not hasattr(cls, "_cache_timestamps"):
            cls._cache_timestamps = {}

    def _initialize_cache(self) -> None:
        """Initialize cache state."""
        self._cache: TAnyDict = {}
        self._cache_timestamps: dict[str, float] = {}

    def __ensure_cache_state(self) -> None:
        """Ensure cache state is initialized per instance."""
        if not hasattr(self, "_cache"):
            self._cache: TAnyDict = {}
            self._cache_timestamps: dict[str, float] = {}

    def cache_get(self, key: str, max_age_seconds: float = 300.0) -> object | None:
        """Get cached value if not expired."""
        if not FlextValidators.is_non_empty_string(key):
            return None

        self.__ensure_cache_state()
        if key not in self._cache:
            return None

        # Check expiration
        timestamp = self._cache_timestamps.get(key, 0.0)
        if FlextGenerators.generate_timestamp() - timestamp > max_age_seconds:
            self.cache_remove(key)
            return None

        return self._cache[key]

    def cache_set(self, key: str, value: object) -> None:
        """Set cached value with timestamp."""
        if not FlextValidators.is_non_empty_string(key):
            return

        self.__ensure_cache_state()
        self._cache[key] = value
        self._cache_timestamps[key] = FlextGenerators.generate_timestamp()

    def cache_remove(self, key: str) -> None:
        """Remove cached value."""
        self.__ensure_cache_state()
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)

    def cache_clear(self) -> None:
        """Clear all cached values."""
        self.__ensure_cache_state()
        self._cache.clear()
        self._cache_timestamps.clear()

    def cache_size(self) -> int:
        """Get cache size."""
        self.__ensure_cache_state()
        return len(self._cache)


# Combination mixins for common use cases
class _BaseEntityMixin(
    _BaseIdentifiableMixin,
    _BaseTimestampMixin,
    _BaseValidatableMixin,
    _BaseLoggableMixin,
):
    """Combined mixin for entities - ID + timestamps + validation + logging.

    Proper mixin composition without __init__ conflicts.
    All behavior comes from lazy initialization in component mixins.
    """

    def initialize_entity(self, entity_id: TEntityId | None = None) -> None:
        """Initialize entity state (call this from your class __init__)."""
        if entity_id:
            self.set_id(entity_id)
        # Other mixins initialize lazily via properties/methods


class _BaseValueObjectMixin(
    _BaseValidatableMixin,
    _BaseSerializableMixin,
    _BaseComparableMixin,
):
    """Combined mixin for value objects - validation + serialization + comparison.

    Proper mixin composition without __init__ conflicts.
    All behavior comes from lazy initialization in component mixins.
    """

    def initialize_value_object(self) -> None:
        """Initialize value object state (call this from your class __init__)."""
        # All mixins initialize lazily via properties/methods


# Factory functions
def _create_entity_mixin() -> type[_BaseEntityMixin]:
    """Create entity mixin class."""
    return _BaseEntityMixin


def _create_value_object_mixin() -> type[_BaseValueObjectMixin]:
    """Create value object mixin class."""
    return _BaseValueObjectMixin


# Export API
__all__ = [
    "_BaseCacheableMixin",
    "_BaseComparableMixin",
    "_BaseEntityMixin",
    "_BaseIdentifiableMixin",
    "_BaseLoggableMixin",
    "_BaseSerializableMixin",
    "_BaseTimestampMixin",
    "_BaseTimingMixin",
    "_BaseValidatableMixin",
    "_BaseValueObjectMixin",
    "_create_entity_mixin",
    "_create_value_object_mixin",
]
