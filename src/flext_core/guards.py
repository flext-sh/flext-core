"""FLEXT Core Guards Module.

Comprehensive type guards and validation system providing enterprise-grade safety
patterns for the FLEXT Core library. Implements compatibility layer with consolidated
functionality from specialized modules while maintaining backward compatibility.

Architecture:
    - Compatibility layer aggregating functionality from specialized modules
    - Type guard system providing runtime type safety validation
    - Validation decorators for function and class safety enforcement
    - Automatic validation models with comprehensive error handling
    - Factory and builder patterns for safe object construction
    - Requirement validators with descriptive error messaging

Guards System Components:
    - Type Guards: Runtime type checking with comprehensive validation patterns
    - Validation Decorators: Function and class-level safety enforcement
    - ValidatedModel: Automatic validation with enhanced error reporting
    - Factory Helpers: Safe object construction patterns with error handling
    - Requirement Validators: Assertion-style validation with custom messaging
    - Compatibility Exports: Re-exports from specialized modules for API stability

Maintenance Guidelines:
    - This module serves as a compatibility layer, avoid adding new functionality
    - Direct users to specialized modules (FlextValidators, FlextDecorators, etc.)
    - Maintain backward compatibility for existing public API surface
    - Update re-exports when underlying modules change their interfaces
    - Keep validation utilities simple and focused on common use cases
    - Preserve error message consistency across validation patterns

Design Decisions:
    - Compatibility layer pattern aggregating specialized module functionality
    - Re-export strategy preserving existing API contracts for stability
    - ValidatedModel as bridge between Pydantic and FLEXT patterns
    - Factory helpers providing simple construction patterns
    - Requirement validators with assertion-style API for common validations
    - FlextResult integration for safe validation operations

Type Safety Features:
    - Runtime type guards with compile-time type narrowing support
    - Generic type validation for collections and complex structures
    - Instance type checking with inheritance hierarchy support
    - Null safety validation preventing None value propagation
    - Type-safe factory and builder patterns for object construction

Validation Patterns:
    - Decorator-based validation for functions and classes
    - Model-based validation with automatic error aggregation
    - Requirement-style validation with assertion semantics
    - Safe object construction through factory patterns
    - Result-based validation avoiding exception propagation

Backward Compatibility:
    - All previously exported functions remain available
    - Error message formats preserved for existing integrations
    - Function signatures maintained for API stability
    - Decorator interfaces unchanged for existing code
    - Type guard behavior consistent with previous versions

Dependencies:
    - validation: FlextValidators for core validation functionality
    - decorators: FlextDecorators for validation decorator patterns
    - mixins: FlextSerializableMixin and FlextValidatableMixin for model enhancement
    - result: FlextResult for safe validation operations
    - utilities: FlextUtilities for type guard functionality

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from flext_core._mixins_base import _BaseSerializableMixin, _BaseValidatableMixin
from flext_core.decorators import FlextDecorators
from flext_core.result import FlextResult
from flext_core.utilities import FlextTypeGuards, FlextUtilities
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from collections.abc import Callable


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


# Define simple decorators that don't exist in the consolidated version
def immutable(cls: type) -> type:
    """Make class immutable (placeholder)."""
    return cls


def pure(func: object) -> object:
    """Mark function as pure (placeholder)."""
    return func


# =============================================================================
# AUTOMATIC VALIDATION - Custom validated model for backward compatibility
# =============================================================================


class ValidatedModel(BaseModel):
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
        if result.is_success:
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
        """Initialize with validation and mixin functionality through composition."""
        try:
            super().__init__(**data)
            # Initialize mixin functionality through composition
            self._validation_errors: list[str] = []
            self._is_valid: bool | None = None
        except ValidationError as e:
            # Convert Pydantic errors to user-friendly format
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
            error_msg = f"Invalid data: {'; '.join(errors)}"
            raise ValueError(error_msg) from e

    # =========================================================================
    # VALIDATION FUNCTIONALITY - Composition-based delegation
    # =========================================================================

    def _add_validation_error(self, error: str) -> None:
        """Add validation error (delegates to base)."""
        return _BaseValidatableMixin._add_validation_error(self, error)

    def _clear_validation_errors(self) -> None:
        """Clear all validation errors (delegates to base)."""
        return _BaseValidatableMixin._clear_validation_errors(self)

    def _mark_valid(self) -> None:
        """Mark as valid and clear errors (delegates to base)."""
        return _BaseValidatableMixin._mark_valid(self)

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors (delegates to base)."""
        return _BaseValidatableMixin.validation_errors.fget(self)  # type: ignore[misc]

    @property
    def is_valid(self) -> bool:
        """Check if object is valid (delegates to base)."""
        return _BaseValidatableMixin.is_valid.fget(self)  # type: ignore[misc]

    def has_validation_errors(self) -> bool:
        """Check if object has validation errors (delegates to base)."""
        return _BaseValidatableMixin.has_validation_errors(self)

    # =========================================================================
    # SERIALIZATION FUNCTIONALITY - Composition-based delegation
    # =========================================================================

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dictionary representation (delegates to base)."""
        return _BaseSerializableMixin.to_dict_basic(self)

    def _serialize_value(self, value: object) -> object | None:
        """Serialize a single value for dict conversion (delegates to base)."""
        return _BaseSerializableMixin._serialize_value(self, value)

    def _serialize_collection(
        self,
        collection: list[object] | tuple[object, ...],
    ) -> list[object]:
        """Serialize list or tuple values (delegates to base)."""
        return _BaseSerializableMixin._serialize_collection(self, collection)

    def _serialize_dict(self, dict_value: dict[str, object]) -> dict[str, object]:
        """Serialize dictionary values (delegates to base)."""
        return _BaseSerializableMixin._serialize_dict(self, dict_value)

    @classmethod
    def create(cls, **data: object) -> FlextResult[ValidatedModel]:
        """Create instance returning Result instead of raising."""
        try:
            instance = cls(**data)
            return FlextResult.ok(instance)
        except ValueError as e:
            return FlextResult.fail(str(e))


# =============================================================================
# FACTORY HELPERS - Simple implementations (no longer in FlextUtilities)
# =============================================================================


def make_factory(cls: type) -> Callable[[object], object]:
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


def make_builder(cls: type) -> Callable[[object], object]:
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
        raise ValueError(message)
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
        port_number = require_positive(config.get("port", 8080))

        # Custom error message
        timeout_seconds = require_positive(
            user_input.get("timeout"),
            "Timeout must be a positive number of seconds"
        )

    """
    if not FlextValidators.is_positive_int(value):
        raise ValueError(message)
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
    if not FlextValidators.is_in_range(value, min_val, max_val):
        if not message:
            message = f"Value must be between {min_val} and {max_val}"
        raise ValueError(message)
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
        raise ValueError(message)
    return value


# Export API
__all__ = [
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
