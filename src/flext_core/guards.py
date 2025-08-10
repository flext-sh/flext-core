"""Comprehensive type guards and validation system for FLEXT.

Provides enterprise-grade safety patterns, runtime type checking,
validation decorators, and safe object construction for distributed
data integration systems.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ValidationError

from flext_core.constants import FlextConstants
from flext_core.decorators import FlextDecorators
from flext_core.exceptions import FlextValidationError
from flext_core.mixins import FlextSerializableMixin, FlextValidatableMixin
from flext_core.utilities import FlextGenericFactory, FlextTypeGuards, FlextUtilities
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.result import FlextResult
    from flext_core.typings import T

from collections.abc import Callable  # noqa: TC003

Platform = FlextConstants.Platform


# =============================================================================
# TYPE GUARDS - Re-exported from FlextUtilities for backward compatibility
# =============================================================================

# All type guards are now in FlextTypeGuards class
is_not_none = FlextUtilities.is_not_none_guard
is_list_of = FlextTypeGuards.is_list_of
is_instance_of = FlextTypeGuards.is_instance_of


# =============================================================================
# FLEXT GUARDS - Static Class for Guard Functions
# =============================================================================


class FlextGuards:
    """Static class for type guards, validation, and factory methods.

    Provides centralized organization of safety functions and object
    construction patterns.
    """

    @staticmethod
    def is_dict_of(obj: object, value_type: type) -> bool:
        """Check if object is a dict with values of specific type."""
        if not isinstance(obj, dict):
            return False
        return all(isinstance(value, value_type) for value in obj.values())

    @staticmethod
    def immutable[T](target_class: type[T]) -> type[T]:
        """Make class immutable using decorator pattern.

        Args:
            target_class: The class to make immutable

        Returns:
            Immutable version of the class

        """

        class ImmutableWrapper(target_class):  # type: ignore[misc,valid-type]
            """Immutable wrapper class."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Initialize and mark as immutable."""
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "_initialized", True)

            def __setattr__(self, name: str, value: object) -> None:
                """Prevent attribute modification after initialization."""
                if hasattr(self, "_initialized"):
                    error_msg = f"Cannot modify immutable object attribute '{name}'"
                    raise AttributeError(error_msg)
                super().__setattr__(name, value)

            def __hash__(self) -> int:
                """Make object hashable based on all attributes."""
                try:
                    attrs = tuple(
                        getattr(self, attr)
                        for attr in dir(self)
                        if not attr.startswith("_")
                        and not callable(getattr(self, attr))
                    )
                    return hash((self.__class__.__name__, attrs))
                except TypeError:
                    # If any attribute is unhashable, use object id
                    return hash(id(self))

        # Preserve class metadata
        ImmutableWrapper.__name__ = target_class.__name__
        ImmutableWrapper.__qualname__ = getattr(
            target_class,
            "__qualname__",
            target_class.__name__,
        )
        ImmutableWrapper.__module__ = getattr(target_class, "__module__", __name__)

        return ImmutableWrapper

    @staticmethod
    def pure[T](func: T) -> T:
        """Mark function as pure with memoization caching.

        Args:
            func: Function to make pure

        Returns:
            Pure version of the function with caching

        """
        if not callable(func):
            return func  # Not a function, return as-is

        # Cache for memoization
        cache: dict[tuple[object, ...], object] = {}

        def pure_wrapper(*args: object, **kwargs: object) -> object:
            """Pure function wrapper with memoization."""
            # Create cache key from args and kwargs
            try:
                cache_key = (args, tuple(sorted(kwargs.items())))

                # Return cached result if available
                if cache_key in cache:
                    return cache[cache_key]

                # Compute and cache result
                result = func(*args, **kwargs)
                cache[cache_key] = result
            except TypeError:
                # Arguments not hashable, can't cache - just call function
                return func(*args, **kwargs)
            else:
                return result

        # Use functools.wraps to properly preserve function metadata
        pure_wrapper = wraps(func)(pure_wrapper)

        # Mark function as pure for introspection
        # using type ignore for custom attributes
        pure_wrapper.__pure__ = True  # type: ignore[attr-defined]
        pure_wrapper.__cache_size__ = lambda: len(cache)  # type: ignore[attr-defined]
        pure_wrapper.__clear_cache__ = cache.clear  # type: ignore[attr-defined]

        return pure_wrapper  # type: ignore[return-value]

    @staticmethod
    def make_factory(target_class: type) -> Callable[[], object]:
        """Create simple factory function for safe object construction."""

        def factory(*args: object, **kwargs: object) -> object:
            return target_class(*args, **kwargs)

        return factory

    @staticmethod
    def make_builder(target_class: type) -> Callable[[], object]:
        """Create simple builder function for fluent object construction."""

        def builder(*args: object, **kwargs: object) -> object:
            return target_class(*args, **kwargs)

        return builder


# =============================================================================
# AUTOMATIC VALIDATION - Custom validated model for backward compatibility
# =============================================================================


class FlextValidatedModel(  # type: ignore[misc]
    BaseModel,
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Automatic validation model with Pydantic and FLEXT integration.

    Provides validated object construction with enhanced error handling
    and backward compatibility for existing FLEXT patterns.
    """

    def __init__(self, **data: object) -> None:
        """Initialize with proper mixin inheritance and enhanced error handling."""
        try:
            super().__init__(**data)
            # Mixins now initialize themselves through proper inheritance
        except ValidationError as e:
            # Convert Pydantic errors to user-friendly format
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
            error_msg: str = f"Invalid data: {'; '.join(errors)}"
            raise FlextValidationError(
                error_msg,
                validation_details={"errors": errors},
            ) from e

    # Mixin functionality is now inherited properly:
    # - Validation methods from FlextValidatableMixin
    # - Serialization methods from FlextSerializableMixin

    @classmethod
    def create(cls, **data: object) -> FlextResult[Self]:
        """Create instance using centralized factory."""
        # ARCHITECTURAL DECISION: Use centralized factory to eliminate duplication
        factory = FlextGenericFactory(cls)
        # Type cast to correct return type since factory returns FlextResult[object]
        return factory.create(**data)  # type: ignore[return-value]


# =============================================================================
# VALIDATION UTILITIES - Simple implementations using FlextUtilities
# =============================================================================


class FlextValidationUtils:
    """Validation utility functions."""

    @staticmethod
    def require_not_none(
        value: object,
        message: str = "Value cannot be None",
    ) -> object:
        """Require value is not None with assertion-style validation."""
        if value is None:
            raise FlextValidationError(
                message,
                validation_details={"field": "required_value", "value": value},
            )
        return value

    @staticmethod
    def require_positive(
        value: object,
        message: str = "Value must be positive",
    ) -> object:
        """Require value is a positive integer with comprehensive validation."""
        if not (isinstance(value, int) and value > 0):
            raise FlextValidationError(
                message,
                validation_details={"field": "positive_value", "value": value},
            )
        return value

    @staticmethod
    def require_in_range(
        value: object,
        min_val: int,
        max_val: int,
        message: str | None = None,
    ) -> object:
        """Require value is within specified range with bounds validation."""
        if not (isinstance(value, (int, float)) and min_val <= value <= max_val):
            if not message:
                message = f"Value must be between {min_val} and {max_val}"
            raise FlextValidationError(
                message,
                validation_details={
                    "field": "range_value",
                    "value": value,
                    "min_val": min_val,
                    "max_val": max_val,
                },
            )
        return value

    @staticmethod
    def require_non_empty(
        value: object,
        message: str = "Value cannot be empty",
    ) -> object:
        """Require value is a non-empty string with comprehensive validation."""
        if not FlextValidators.is_non_empty_string(value):
            raise FlextValidationError(
                message,
                validation_details={"field": "non_empty_string", "value": value},
            )
        return value


# Duplicate function removed - methods already exist in FlextGuards class

# =============================================================================
# COMPATIBILITY ALIASES - For backward compatibility
# =============================================================================

# Re-export FlextValidationDecorators methods as module-level functions
validated = FlextDecorators.validated_with_result
safe = FlextDecorators.safe_result

# Compatibility aliases for loose functions now in FlextGuards
is_dict_of = FlextGuards.is_dict_of
immutable = FlextGuards.immutable
pure = FlextGuards.pure
make_factory = FlextGuards.make_factory
make_builder = FlextGuards.make_builder
require_not_none = FlextValidationUtils.require_not_none
require_positive = FlextValidationUtils.require_positive
require_in_range = FlextValidationUtils.require_in_range
require_non_empty = FlextValidationUtils.require_non_empty

# Backward compatibility alias for validated model
ValidatedModel = FlextValidatedModel


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [  # noqa: RUF022
    # Main static class
    "FlextGuards",
    # Utility classes
    "FlextValidationUtils",
    # Base classes
    "FlextValidatedModel",
    "ValidatedModel",
    # Compatibility aliases for functions
    "immutable",
    "is_dict_of",
    "is_instance_of",
    "is_list_of",
    "is_not_none",
    "make_builder",
    "make_factory",
    "pure",
    "require_in_range",
    "require_non_empty",
    "require_not_none",
    "require_positive",
    "safe",
    "validated",
]
