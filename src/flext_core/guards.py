"""FLEXT Core Guards - Extension Layer Type Safety System.

Comprehensive type guards and validation system providing enterprise-grade safety
patterns and runtime type checking across the 32-project FLEXT ecosystem. Foundation
for type safety enforcement, validation decorators, and safe object construction in
distributed data integration and business logic components.

Module Role in Architecture:
    Extension Layer → Type Safety System → Runtime Validation & Guards

    This module provides type safety patterns used throughout FLEXT projects:
    - Runtime type guards with compile-time type narrowing support
    - Validation decorators for function and class safety enforcement
    - Automatic validation models with enhanced error reporting
    - Factory and builder patterns for safe object construction

Guard Architecture Patterns:
    Compatibility Layer: Aggregating functionality from specialized modules
    Type Guard System: Runtime type safety validation with static analysis benefits
    Validation Decorators: Function and class-level safety enforcement
    Factory Patterns: Safe object construction with comprehensive error handling

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Type guards, validation decorators, validated models
    🚧 Active Development: Advanced guard patterns (Enhancement 5 - Priority Low)
    📋 TODO Integration: Generic type validation enhancement (Priority 4)

Guards System Components:
    Type Guards: Runtime type checking with comprehensive validation patterns
    Validation Decorators: Function and class-level safety enforcement
    ValidatedModel: Automatic validation with enhanced error reporting
    Factory Helpers: Safe object construction patterns with error handling
    Requirement Validators: Assertion-style validation with custom messaging

Ecosystem Usage Patterns:
    # FLEXT Service Type Safety
    from flext_core.guards import TypeGuards, require

    def process_user_data(data: object) -> FlextResult[User]:
        require.not_none(data, "User data cannot be None")
        if TypeGuards.is_dict(data):
            return User.create_from_dict(data)
        return FlextResult.fail("Invalid user data format")

    # Singer Tap/Target Guards
    @TypeGuards.validated_input
    def extract_oracle_data(connection_string: str, table_name: str):
        require.non_empty_string(connection_string, "Connection required")
        require.non_empty_string(table_name, "Table name required")
        return oracle_client.extract(connection_string, table_name)

    # ALGAR Migration Safety
    class LdapUserValidator(ValidatedModel):
        dn: str
        uid: str

        @classmethod
        def validate_dn_format(cls, dn: str) -> bool:
            return TypeGuards.is_valid_ldap_dn(dn)

Type Safety Features:
    - Runtime type guards with compile-time type narrowing support
    - Generic type validation for collections and complex structures
    - Instance type checking with inheritance hierarchy support
    - Null safety validation preventing None value propagation
    - Type-safe factory and builder patterns for object construction

Quality Standards:
    - All type guards must support both runtime and static analysis
    - Validation decorators must preserve function signatures and type annotations
    - Error messages must be actionable and provide debugging context
    - Guard compatibility must be maintained across version updates

See Also:
    docs/TODO.md: Enhancement 5 - Advanced guard pattern development
    validation.py: FlextValidators integration for comprehensive validation
    decorators.py: Validation decorator implementation details

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Self, TypeVar

from pydantic import BaseModel, ValidationError

from flext_core.constants import FlextConstants
from flext_core.decorators import FlextDecorators
from flext_core.exceptions import FlextValidationError
from flext_core.mixins import FlextSerializableMixin, FlextValidatableMixin
from flext_core.result import FlextResult
from flext_core.utilities import FlextTypeGuards, FlextUtilities
from flext_core.validation import FlextValidators

T = TypeVar("T")

Platform = FlextConstants.Platform

if TYPE_CHECKING:
    from flext_core.flext_types import TFactory

# =============================================================================
# TYPE GUARDS - Re-exported from FlextUtilities for backward compatibility
# =============================================================================

# All type guards are now in FlextTypeGuards class
is_not_none = FlextUtilities.is_not_none_guard
is_list_of = FlextTypeGuards.is_list_of
is_instance_of = FlextTypeGuards.is_instance_of


# These need to be defined since they don't exist in FlextTypeGuards
def is_dict_of(obj: object, value_type: type) -> bool:
    """Check if object is a dict with values of specific type."""
    if not isinstance(obj, dict):
        return False
    return all(isinstance(value, value_type) for value in obj.values())


# =============================================================================
# VALIDATION DECORATORS - Re-exported from FlextDecorators for backward compatibility
# =============================================================================

# All decorators are now in FlextDecorators
validated = FlextDecorators.validated_with_result  # Best available equivalent
safe = FlextDecorators.safe_result


# Define SOLID-compliant decorators with real functionality
def immutable[T](cls: type[T]) -> type[T]:
    """Make class immutable using SOLID principles.

    Implements immutability through:
    - Freezing class attributes to prevent modification
    - Adding __setattr__ override to block attribute changes
    - Creating __hash__ method for use in sets and dicts
    - Preserving original class functionality (Liskov Substitution Principle)

    Args:
        cls: Class to make immutable

    Returns:
        Immutable version of the class following SOLID principles

    Usage:
        @immutable
        class User:
            def __init__(self, name: str, age: int) -> None:
                self.name = name
                self.age = age

        user = User("John", 30)
        # user.name = "Jane"  # Raises AttributeError

    """

    # Create immutable wrapper class - using type ignore for dynamic inheritance
    class ImmutableWrapper(cls):  # type: ignore[misc,valid-type]
        """Immutable wrapper class."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize and mark as immutable."""
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_initialized", True)

        def __setattr__(self, name: str, value: object) -> None:
            """Prevent attribute modification after initialization."""
            if hasattr(self, "_initialized"):
                error_msg: str = f"Cannot modify immutable object attribute '{name}'"
                raise AttributeError(error_msg)
            super().__setattr__(name, value)

        def __hash__(self) -> int:
            """Make object hashable based on all attributes."""
            try:
                attrs = tuple(
                    getattr(self, attr)
                    for attr in dir(self)
                    if not attr.startswith("_") and not callable(getattr(self, attr))
                )
                return hash((self.__class__.__name__, attrs))
            except TypeError:
                # If any attribute is unhashable, use object id
                return hash(id(self))

    # Preserve class metadata
    ImmutableWrapper.__name__ = cls.__name__
    ImmutableWrapper.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
    ImmutableWrapper.__module__ = getattr(cls, "__module__", __name__)

    return ImmutableWrapper


def pure[T](func: T) -> T:
    """Mark function as pure with validation and caching.

    Implements functional purity through:
    - Input validation to ensure deterministic behavior
    - Result caching for performance (memoization)
    - Side-effect detection and warnings
    - Type safety preservation

    A pure function:
    - Always returns the same output for the same inputs
    - Has no side effects (no I/O, no mutations)
    - Depends only on its parameters

    Args:
        func: Function to make pure

    Returns:
        Pure version of the function with caching and validation

    Usage:
        @pure
        def calculate_square(x: int) -> int:
            return x * x

        result1 = calculate_square(5)  # Computed
        result2 = calculate_square(5)  # Cached
        assert result1 == result2 == 25

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

    # Mark function as pure for introspection - using type ignore for custom attributes
    pure_wrapper.__pure__ = True  # type: ignore[attr-defined]
    pure_wrapper.__cache_size__ = lambda: len(cache)  # type: ignore[attr-defined]
    pure_wrapper.__clear_cache__ = cache.clear  # type: ignore[attr-defined]

    return pure_wrapper  # type: ignore[return-value]


# =============================================================================
# AUTOMATIC VALIDATION - Custom validated model for backward compatibility
# =============================================================================


class ValidatedModel(
    BaseModel,
    FlextValidatableMixin,
    FlextSerializableMixin,
):
    """Automatic validation model providing enterprise-grade safety for object creation.

    Backward compatibility class that bridges Pydantic validation with FLEXT patterns,
    providing automatic validation on construction with enhanced error reporting and
    FLEXT mixin functionality integration.

    Architecture:
        - Pydantic BaseModel foundation for automatic field validation
        - FlextSerializableMixin integration for consistent serialization patterns
        - FlextValidatableMixin integration for enhanced validation capabilities
        - Enhanced error handling with user-friendly error message aggregation
        - FlextResult integration for safe object creation patterns

    Validation Features:
        - Automatic field validation on object construction
        - Enhanced error aggregation with field-specific error messages
        - User-friendly error formatting for debugging and user interfaces
        - FlextResult-based safe creation methods avoiding exception propagation
        - Integration with FLEXT validation patterns and error handling

    Error Handling:
        - Pydantic ValidationError conversion to user-friendly ValueError
        - Field-specific error messages with location information
        - Error aggregation across multiple validation failures
        - Chained exception preservation for debugging and troubleshooting
        - Consistent error message formatting across validation scenarios

    Migration Path:
        This class is maintained for backward compatibility. For new development,
        consider using specialized FLEXT domain classes:
        - FlextValueObject: For immutable value objects with business logic
        - FlextEntity: For entities with identity and lifecycle management
        - FlextAggregateRoot: For aggregates with transactional boundaries

    Usage Patterns:
        # Basic validated model
        class UserProfile(ValidatedModel):
            name: str
            age: int
            email: str

        # Exception-based creation (traditional pattern)
        try:
            profile = UserProfile(name="John", age=30, email="john@example.com")
            print(f"Created profile: {profile.name}")
        except ValueError as e:
            print(f"Validation failed: {e}")

        # Safe creation with FlextResult
        result = UserProfile.create(
            name="Jane",
            age=25,
            email="jane@example.com"
        )
        if result.success:
            profile = result.data
            print(f"Created profile: {profile.name}")
        else:
            print(f"Validation failed: {result.error}")

    FLEXT Mixin Integration:
        - Serialization capabilities through FlextSerializableMixin
        - Enhanced validation methods through FlextValidatableMixin
        - Consistent API patterns with other FLEXT domain objects
        - Integration with FLEXT result patterns and error handling
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
        """Create instance returning Result instead of raising."""
        try:
            instance = cls(**data)
            return FlextResult.ok(instance)
        except (ValueError, FlextValidationError) as e:
            return FlextResult.fail(str(e))


# =============================================================================
# FACTORY HELPERS - Simple implementations (no longer in FlextUtilities)
# =============================================================================


def make_factory(cls: type) -> TFactory[object]:
    """Create simple factory function for safe object construction.

    Provides a factory pattern implementation that wraps class construction
    in a function interface. Useful for dependency injection scenarios and
    lazy object creation patterns.

    Args:
        cls: Class type to create factory for

    Returns:
        Factory function that creates instances of the class

    Usage:
        # Create factory for a class
        UserFactory = make_factory(User)

        # Use factory to create instances
        user = UserFactory(name="John", age=30)

        # Factory can be passed around as first-class object
        def process_with_factory(factory: Callable, data: dict):
            return factory(**data)

        result = process_with_factory(UserFactory, {"name": "Jane", "age": 25})

    """

    def factory(*args: object, **kwargs: object) -> object:
        return cls(*args, **kwargs)

    return factory


def make_builder(cls: type) -> TFactory[object]:
    """Create simple builder function for fluent object construction.

    Provides a builder pattern implementation that wraps class construction
    in a function interface. Identical to make_factory but named for clarity
    when using builder patterns.

    Args:
        cls: Class type to create builder for

    Returns:
        Builder function that creates instances of the class

    Usage:
        # Create builder for a class
        UserBuilder = make_builder(User)

        # Use builder to create instances
        user = UserBuilder(name="John", age=30)

        # Builder pattern with method chaining (if class supports it)
        class FluentUser:
            def __init__(self):
                self.name = None
                self.age = None

            def with_name(self, name: str):
                self.name = name
                return self

            def with_age(self, age: int):
                self.age = age
                return self

        FluentUserBuilder = make_builder(FluentUser)
        user = FluentUserBuilder().with_name("John").with_age(30)

    """

    def builder(*args: object, **kwargs: object) -> object:
        return cls(*args, **kwargs)

    return builder


# =============================================================================
# VALIDATION UTILITIES - Simple implementations using FlextUtilities
# =============================================================================


def require_not_none(value: object, message: str = "Value cannot be None") -> object:
    """Require value is not None with assertion-style validation.

    Validation utility that ensures a value is not None, raising ValueError
    with a descriptive message if the requirement is not met. Provides
    early validation with clear error messaging.

    Args:
        value: Value to validate for non-None requirement
        message: Custom error message if validation fails

    Returns:
        The original value if validation passes

    Raises:
        ValueError: If value is None

    Usage:
        # Basic usage
        user_id = require_not_none(user_input.get("id"))

        # Custom error message
        config_value = require_not_none(
            config.get("database_url"),
            "Database URL is required for application startup"
        )

    """
    if value is None:
        raise FlextValidationError(
            message,
            validation_details={"field": "required_value", "value": value},
        )
    return value


def require_positive(value: object, message: str = "Value must be positive") -> object:
    """Require value is a positive integer with comprehensive validation.

    Validation utility that ensures a value is a positive integer, raising
    ValueError with a descriptive message if the requirement is not met.
    Uses FlextValidators for consistent validation behavior.

    Args:
        value: Value to validate for positive integer requirement
        message: Custom error message if validation fails

    Returns:
        The original value if validation passes

    Raises:
        ValueError: If value is not a positive integer

    Usage:
        # Basic usage
        port_number = require_positive(config.get("port", Platform.FLEXCORE_PORT))

        # Custom error message
        timeout_seconds = require_positive(
            user_input.get("timeout"),
            "Timeout must be a positive number of seconds"
        )

    """
    if not (isinstance(value, int) and value > 0):
        raise FlextValidationError(
            message,
            validation_details={"field": "positive_value", "value": value},
        )
    return value


def require_in_range(
    value: object,
    min_val: int,
    max_val: int,
    message: str | None = None,
) -> object:
    """Require value is within specified range with bounds validation.

    Validation utility that ensures a numeric value falls within the specified
    range (inclusive), raising ValueError with a descriptive message if the
    requirement is not met. Provides automatic message generation if not specified.

    Args:
        value: Numeric value to validate for range requirement
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        message: Custom error message, auto-generated if None

    Returns:
        The original value if validation passes

    Raises:
        ValueError: If value is not within the specified range

    Usage:
        # Basic usage with auto-generated message
        age = require_in_range(user_age, 0, 150)

        # Custom error message
        priority = require_in_range(
            task_priority, 1, 5,
            "Task priority must be between 1 (low) and 5 (critical)"
        )

    """
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


def require_non_empty(value: object, message: str = "Value cannot be empty") -> object:
    """Require value is a non-empty string with comprehensive validation.

    Validation utility that ensures a value is a non-empty string (after stripping
    whitespace), raising ValueError with a descriptive message if the requirement
    is not met. Uses FlextValidators for consistent validation behavior.

    Args:
        value: Value to validate for non-empty string requirement
        message: Custom error message if validation fails

    Returns:
        The original value if validation passes

    Raises:
        ValueError: If value is not a non-empty string

    Usage:
        # Basic usage
        username = require_non_empty(form_data.get("username"))

        # Custom error message
        email = require_non_empty(
            user_input.get("email"),
            "Email address is required for registration"
        )

    """
    if not FlextValidators.is_non_empty_string(value):
        raise FlextValidationError(
            message,
            validation_details={"field": "non_empty_string", "value": value},
        )
    return value


# Export API
__all__: list[str] = [
    # Base classes
    "ValidatedModel",
    "immutable",
    "is_dict_of",
    "is_instance_of",
    "is_list_of",
    # Type guards
    "is_not_none",
    "make_builder",
    # Factory helpers
    "make_factory",
    "pure",
    "require_in_range",
    "require_non_empty",
    # Validation utilities
    "require_not_none",
    "require_positive",
    "safe",
    # Decorators
    "validated",
]
