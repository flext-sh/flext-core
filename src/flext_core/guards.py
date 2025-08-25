"""Guards and validation utilities for FLEXT validation system.

SINGLE CONSOLIDATED MODULE following FLEXT architectural patterns.
All guard, validation, and pure function functionality consolidated into FlextGuards.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import cast

from flext_core.decorators import FlextDecorators
from flext_core.exceptions import FlextValidationError
from flext_core.result import FlextResult
from flext_core.validation import FlextValidators


class FlextGuards:
    """SINGLE CONSOLIDATED CLASS for all guard and validation functionality.

    Following FLEXT architectural patterns - consolidates ALL guard functionality
    including pure function wrapping, validation utilities, and factory methods
    into one main class with nested classes for organization.

    CONSOLIDATED CLASSES: _PureWrapper + FlextGuards + FlextValidationUtils
    """

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class PureWrapper[R]:
        """Nested wrapper class for pure functions with memoization."""

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

        def __get__(self, instance: object, owner: type | None = None) -> object:
            """Descriptor protocol to handle method binding."""
            if instance is None:
                return self

            # Create a bound method-like callable that preserves __pure__ attribute
            def bound_method(*args: object, **kwargs: object) -> R:
                return self(instance, *args, **kwargs)

            # Safely add the __pure__ attribute to the function using setattr
            with contextlib.suppress(AttributeError, TypeError):
                bound_method.__pure__ = True  # type: ignore[attr-defined]
            return bound_method

    class ValidationUtils:
        """Nested validation utility functions."""

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

    # ==========================================================================
    # MAIN GUARD FUNCTIONALITY
    # ==========================================================================

    @staticmethod
    def is_dict_of(obj: object, value_type: type) -> bool:
        """Check if an object is a dict with values of a specific type."""
        if not isinstance(obj, dict):
            return False
        # After isinstance check, obj is narrowed to dict type
        dict_obj = cast("dict[object, object]", obj)
        return all(isinstance(value, value_type) for value in dict_obj.values())

    @staticmethod
    def is_list_of(obj: object, item_type: type) -> bool:
        """Check if an object is a list with items of a specific type."""
        if not isinstance(obj, list):
            return False
        # After isinstance check, obj is narrowed to list type
        list_obj = cast("list[object]", obj)
        return all(isinstance(item, item_type) for item in list_obj)

    @staticmethod
    def immutable(target_class: type) -> type:
        """Make class immutable using a decorator pattern.

        Args:
            target_class: The class to make immutable

        Returns:
            Immutable version of the class

        """

        def _init(self: object, *args: object, **kwargs: object) -> None:
            # Call original class initialization first
            try:
                # Call the target class's __init__ method directly from the class
                # Use getattr to safely access __init__ method from the class
                init_method = getattr(target_class, "__init__", None)
                if init_method is not None:
                    init_method(self, *args, **kwargs)
                else:
                    object.__init__(self)
            except Exception:
                # Fallback to basic initialization if original fails
                object.__init__(self)
            # Mark as initialized to prevent further modifications
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
        return FlextGuards.PureWrapper(func)

    @staticmethod
    def make_factory(target_class: type) -> object:
        """Create a simple factory class for safe object construction."""

        class _Factory:
            def create(self, **kwargs: object) -> FlextResult[object]:
                try:
                    instance = target_class(**kwargs)
                    return FlextResult[object].ok(instance)
                except Exception as e:
                    return FlextResult[object].fail(f"Factory failed: {e}")

        return _Factory()

    @staticmethod
    def make_builder(target_class: type) -> object:
        """Create a simple builder class for fluent object construction."""

        class _Builder:
            def create(self, **kwargs: object) -> FlextResult[object]:
                try:
                    instance = target_class(**kwargs)
                    return FlextResult[object].ok(instance)
                except Exception as e:
                    return FlextResult[object].fail(f"Builder failed: {e}")

        return _Builder()

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Consolidated approach
# =============================================================================

# Export nested classes for external access (backward compatibility)
FlextValidationUtils = FlextGuards.ValidationUtils
_PureWrapper = FlextGuards.PureWrapper

# Re-export FlextValidationDecorators methods as module-level functions
validated = FlextDecorators.validated_with_result
safe = FlextDecorators.safe_result

# Compatibility aliases pointing to consolidated FlextGuards
immutable = FlextGuards.immutable
pure = FlextGuards.pure
make_factory = FlextGuards.make_factory
make_builder = FlextGuards.make_builder
require_not_none = FlextGuards.ValidationUtils.require_not_none
require_positive = FlextGuards.ValidationUtils.require_positive
require_non_empty = FlextGuards.ValidationUtils.require_non_empty


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    # Main static class
    "FlextGuards",
    # Utility classes
    "FlextValidationUtils",
    # Compatibility aliases for functions (alphabetically sorted)
    "immutable",
    "make_builder",
    "make_factory",
    "pure",
    "require_non_empty",
    "require_not_none",
    "require_positive",
    "safe",
    "validated",
]
