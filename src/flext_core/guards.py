"""Guards and validation utilities for FLEXT validation system."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

from pydantic import BaseModel, ValidationError

from flext_core.constants import FlextConstants
from flext_core.decorators import FlextDecorators
from flext_core.exceptions import FlextValidationError
from flext_core.mixins import FlextSerializableMixin
from flext_core.result import FlextResult
from flext_core.typings import (
    TFactory,
)
from flext_core.utilities import FlextTypeGuards, FlextUtilities
from flext_core.validation import FlextValidators

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


class _PureWrapper[R]:
    """Wrapper class for pure functions with memoization."""

    def __init__(self, func: Callable[[object], R] | Callable[[], R]) -> None:
        self.func = func
        self.cache: dict[object, R] = {}
        self.__pure__ = True
        # Copy function metadata safely
        if hasattr(func, "__name__"):
            self.__name__ = func.__name__
        if hasattr(func, "__doc__"):
            self.__doc__ = func.__doc__

    def __call__(self, *args: object, **kwargs: object) -> R:
        """Invoke the wrapped function with memoization."""
        try:
            cache_key = (args, tuple(sorted(kwargs.items())))
            if cache_key in self.cache:
                return self.cache[cache_key]
            result = self.func(*args, **kwargs)
            self.cache[cache_key] = result
            return result
        except TypeError:
            return self.func(*args, **kwargs)

    def __cache_size__(self) -> int:
        """Return the size of the cache."""
        return len(self.cache)


class FlextGuards:
    """Static class for type guards, validation, and factory methods.

    Provides centralized organization of safety functions and object
    construction patterns.
    """

    @staticmethod
    def is_dict_of(obj: object, value_type: type) -> bool:
        """Check if an object is a dict with values of a specific type."""
        if not isinstance(obj, dict):
            return False
        return all(
            isinstance(value, value_type)
            for value in cast("dict[object, object]", obj).values()
        )

    @staticmethod
    def immutable(target_class: type) -> type:
        """Make class immutable using a decorator pattern.

        Args:
            target_class: The class to make immutable

        Returns:
            Immutable version of the class

        """

        def _init(self: object, *args: object, **kwargs: object) -> None:
            # Call the original class initializer directly to avoid super() pitfalls
            try:
                target_class.__init__(self, *args, **kwargs)  # type: ignore[misc]
            except Exception:
                # If original init fails, attempt base object init as fallback
                object.__init__(self)
            object.__setattr__(self, "_initialized", True)

        def _setattr(self: object, name: str, value: object) -> None:
            if hasattr(self, "_initialized"):
                msg = "Cannot modify immutable object attribute '" + name + "'"
                raise AttributeError(msg)
            object.__setattr__(self, name, value)

        def _hash(self: object) -> int:
            try:
                attrs = tuple(
                    getattr(self, attr)
                    for attr in dir(self)
                    if not attr.startswith("_") and not callable(getattr(self, attr))
                )
                return hash((self.__class__.__name__, attrs))
            except TypeError:
                return hash(id(self))

        # Create wrapper class
        return type(
            target_class.__name__,
            (target_class,),
            {
                "__init__": _init,
                "__setattr__": _setattr,
                "__hash__": _hash,
                "__module__": getattr(target_class, "__module__", __name__),
                "__qualname__": getattr(
                    target_class,
                    "__qualname__",
                    target_class.__name__,
                ),
            },
        )

    @staticmethod
    def pure[R](
        func: Callable[[object], R] | Callable[[], R],
    ) -> Callable[[object], R] | Callable[[], R]:
        """Mark function as pure with memoization caching.

        Args:
            func: Function to make pure

        Returns:
            Pure version of the function with caching

        """
        return _PureWrapper(func)

    @staticmethod
    def make_factory(target_class: type) -> TFactory:
        """Create a simple factory function for safe object construction."""

        def factory(*args: object, **kwargs: object) -> object:
            return target_class(*args, **kwargs)

        return factory

    @staticmethod
    def make_builder(target_class: type) -> TFactory:
        """Create a simple builder function for fluent object construction."""

        def builder(*args: object, **kwargs: object) -> object:
            return target_class(*args, **kwargs)

        return builder


# =============================================================================
# AUTOMATIC VALIDATION - Custom validated model for backward compatibility
# =============================================================================


class FlextValidatedModel(BaseModel, FlextSerializableMixin):
    """Automatic validation model with Pydantic and FLEXT integration.

    Provides validated object construction with enhanced error handling
    and backward compatibility for existing FLEXT patterns.
    """

    def __init__(self, **data: object) -> None:
        """Initialize with proper mixin inheritance and enhanced error handling."""
        try:
            super().__init__(**data)
        except ValidationError as e:
            # Convert Pydantic errors to user-friendly format
            errors: list[str] = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"]) if error.get("loc") else ""
                msg = error.get("msg", "Validation error")
                # Some tests expect messages without 'Input should be' prefix
                normalized = (
                    msg.replace("Input should be ", "")
                    .replace("Input should be a ", "a ")
                    .strip()
                )
                errors.append(f"{loc}: {normalized}" if loc else normalized)
            # Join messages using '; ' consistent with expectations
            error_msg: str = f"Invalid data: {'; '.join(errors)}"
            raise FlextValidationError(
                error_msg,
                validation_details={"errors": errors},
            ) from e

    # Add FlextValidatable methods without conflicts
    def validate_flext(self) -> FlextResult[None]:
        """Validate the model using Pydantic validation (renamed to avoid conflicts)."""
        try:
            # Use Pydantic's validation
            self.model_validate(self.model_dump())
            return FlextResult[None].ok(None)
        except ValidationError as e:
            errors = [error["msg"] for error in e.errors()]
            return FlextResult[None].fail(f"Validation failed: {'; '.join(errors)}")

    @property
    def is_valid(self) -> bool:
        """Check if the model is valid."""
        try:
            self.model_validate(self.model_dump())
            return True
        except ValidationError:
            return False

    @classmethod
    def create(cls, **data: object) -> FlextResult[Self]:
        """Create instance using centralized factory.

        On failure, return FlextResult with normalized 'Invalid data' message
        instead of raising, to align with tests expecting failure results.
        """
        try:
            instance = cls(**data)
            return FlextResult[Self].ok(instance)
        except (ValidationError, FlextValidationError) as e:
            errors: list[str] = []
            if isinstance(e, ValidationError):
                for error in e.errors():
                    loc = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    normalized = (
                        msg.replace("Input should be ", "")
                        .replace("Input should be a ", "a ")
                        .strip()
                    )
                    errors.append(f"{loc}: {normalized}" if loc else normalized)
                return FlextResult[Self].fail(f"Invalid data: {'; '.join(errors)}")
            # FlextValidationError already has normalized message
            return FlextResult[Self].fail(str(e))


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
        """Require value is within a specified range with bounds validation."""
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
        if not isinstance(value, str) or not FlextValidators.is_non_empty_string(value):
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

# Backward compatibility alias for a validated model
ValidatedModel = FlextValidatedModel


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    # Main static class
    "FlextGuards",
    # Base classes
    "FlextValidatedModel",
    # Utility classes
    "FlextValidationUtils",
    # Compatibility aliases for functions (alphabetically sorted)
    "ValidatedModel",
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
