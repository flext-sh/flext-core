"""FLEXT Core Validation Module.

Comprehensive validation system for the FLEXT Core library providing consolidated
functionality through multiple inheritance patterns and specialized validation classes.

Architecture:
    - Multiple inheritance combining specialized validation base classes
    - Pydantic-based validation models with extensive configuration support
    - Functional programming patterns with predicates and validators
    - Railway-oriented validation with structured FlextResult integration
    - No underscore prefixes on public objects

Validation System Components:
    - FlextValidation: Main validation class with comprehensive method inheritance
    - FlextPredicates: Functional predicates for filtering and validation patterns
    - FlextValidationResult: Structured validation results with success/failure handling
    - FlextValidationConfig: Pydantic-based configuration for validation parameters
    - Convenience functions: High-level validation helpers with FlextResult integration
    - Validator composition: Chain and combine validators with complex orchestration

Maintenance Guidelines:
    - Add new validator types to appropriate specialized classes first
    - Use multiple inheritance for validator combination patterns
    - Maintain functional programming patterns with pure functions
    - Integrate FlextResult pattern for all operations that can fail
    - Keep validator composition patterns for complex validation scenarios

Design Decisions:
    - Multiple inheritance pattern for maximum validation functionality reuse
    - Pydantic extensive usage for validation configuration and results
    - Functional patterns with validator chaining and composition
    - FlextResult integration for consistent error handling across the system
    - Clean separation between predicates (bool) and validators (FlextResult)
    - Backward compatibility through function aliases and legacy patterns

Validation Patterns:
    - Simple validation: FlextValidation.is_string(value)
    - Structured validation: validate_string(value, "email", min_length=5)
    - Composed validation: FlextValidation.chain(validator1, validator2)
    - Predicate filtering: FlextPredicates.non_empty_string()(value)
    - Configuration-based: FlextValidationConfig with Pydantic models
    - Result-based: All validators return FlextResult for safe error handling

Dependencies:
    - pydantic: Validation models and configuration management
    - result: FlextResult pattern for consistent error handling
    - constants: Core regex patterns and validation constants
    - Standard library: re, collections.abc for pattern matching

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

# Regex patterns from constants - sem underscore conforme diretrizes
EMAIL_PATTERN = FlextConstants.EMAIL_PATTERN
UUID_PATTERN = FlextConstants.UUID_PATTERN
URL_PATTERN = FlextConstants.URL_PATTERN


# =============================================================================
# PYDANTIC-BASED VALIDATION MODELS - sem underscore conforme diretrizes
# =============================================================================


class FlextValidationConfig(BaseModel):
    """Pydantic config for validation parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_name: str = Field(..., min_length=1)
    min_length: int = Field(default=0, ge=0)
    max_length: int | None = Field(default=None, ge=1)
    min_val: float | None = Field(default=None)
    max_val: float | None = Field(default=None)
    pattern: str | None = Field(default=None)

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int | None, info: object) -> int | None:
        """Validate max_length is greater than min_length."""
        if v is not None and hasattr(info, "data") and "min_length" in info.data:
            min_length = info.data["min_length"]
            if v <= min_length:
                error_msg = "max_length must be greater than min_length"
                raise ValueError(error_msg)
        return v


class FlextValidationResult(BaseModel):
    """Pydantic model for validation results."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    error_message: str | None = None
    field_name: str | None = None

    @classmethod
    def success(cls) -> FlextValidationResult:
        """Create successful validation result."""
        return cls(is_valid=True)

    @classmethod
    def failure(
        cls,
        message: str,
        field_name: str | None = None,
    ) -> FlextValidationResult:
        """Create failed validation result."""
        return cls(is_valid=False, error_message=message, field_name=field_name)


# =============================================================================
# CONSOLIDATED VALIDATORS - sem underscore conforme diretrizes
# =============================================================================


class FlextValidators:
    """Consolidated validation functions using Pydantic extensively.

    Base class designed for extension by FlextValidation.
    Single source of truth for all validation - no duplication.
    """

    # Basic type checks
    @staticmethod
    def is_callable(obj: object) -> bool:
        """Check if object is callable."""
        return callable(obj)

    @staticmethod
    def is_not_none(value: object) -> bool:
        """Check if value is not None."""
        return value is not None

    @staticmethod
    def is_string(value: object) -> bool:
        """Check if value is a string."""
        return isinstance(value, str)

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Check if value is a non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    @staticmethod
    def is_int(value: object) -> bool:
        """Check if value is an integer."""
        return isinstance(value, int)

    @staticmethod
    def is_positive_int(value: object) -> bool:
        """Check if value is a positive integer."""
        return isinstance(value, int) and value > 0

    @staticmethod
    def is_list(value: object) -> bool:
        """Check if value is a list."""
        return isinstance(value, list)

    @staticmethod
    def is_non_empty_list(value: object) -> bool:
        """Check if value is a non-empty list."""
        return isinstance(value, list) and len(value) > 0

    @staticmethod
    def is_dict(value: object) -> bool:
        """Check if value is a dictionary."""
        return isinstance(value, dict)

    @staticmethod
    def is_non_empty_dict(value: object) -> bool:
        """Check if value is a non-empty dictionary."""
        return isinstance(value, dict) and len(value) > 0

    # Advanced pattern matching
    @staticmethod
    def is_email(value: object) -> bool:
        """Check if value is a valid email using regex."""
        if not isinstance(value, str):
            return False
        return bool(re.match(EMAIL_PATTERN, value))

    @staticmethod
    def is_uuid(value: object) -> bool:
        """Check if value is a valid UUID using regex."""
        if not isinstance(value, str):
            return False
        return bool(re.match(UUID_PATTERN, value.lower()))

    @staticmethod
    def is_url(value: object) -> bool:
        """Check if value is a valid URL using regex."""
        if not isinstance(value, str):
            return False
        return bool(re.match(URL_PATTERN, value))

    @staticmethod
    def matches_pattern(value: object, pattern: str) -> bool:
        """Check if string value matches regex pattern."""
        if not isinstance(value, str):
            return False
        try:
            return bool(re.match(pattern, value))
        except re.error:
            return False

    # Numeric and range validation
    @staticmethod
    def is_in_range(value: object, min_val: float, max_val: float) -> bool:
        """Check if numeric value is in range."""
        if not isinstance(value, (int, float)):
            return False
        return min_val <= value <= max_val

    @staticmethod
    def has_min_length(value: object, min_length: int) -> bool:
        """Check if value has minimum length."""
        if not hasattr(value, "__len__"):
            return False
        return len(value) >= min_length

    @staticmethod
    def has_max_length(value: object, max_length: int) -> bool:
        """Check if value has maximum length."""
        if not hasattr(value, "__len__"):
            return False
        return len(value) <= max_length

    @staticmethod
    def is_instance_of(value: object, type_class: type[object]) -> bool:
        """Check if value is instance of given type."""
        return isinstance(value, type_class)

    # From utilities_base - useful methods to preserve
    @staticmethod
    def is_valid_identifier(name: str) -> bool:
        """Check if string is a valid Python identifier."""
        return isinstance(name, str) and name.isidentifier()


# =============================================================================
# FUNCTIONAL PREDICATES - sem underscore conforme diretrizes
# =============================================================================


class FlextPredicates:
    """Functional predicates with Pydantic-based configuration.

    Base class designed for extension by FlextPredicates.
    """

    @staticmethod
    def not_none() -> Callable[[object], bool]:
        """Predicate that checks if value is not None."""
        return lambda x: x is not None

    @staticmethod
    def non_empty_string() -> Callable[[object], bool]:
        """Predicate that checks if value is non-empty string."""
        return lambda x: isinstance(x, str) and len(x.strip()) > 0

    @staticmethod
    def positive_number() -> Callable[[object], bool]:
        """Predicate that checks if value is positive number."""
        return lambda x: isinstance(x, (int, float)) and x > 0

    @staticmethod
    def min_length(length: int) -> Callable[[object], bool]:
        """Predicate that checks minimum length."""
        return lambda x: hasattr(x, "__len__") and len(x) >= length

    @staticmethod
    def max_length(length: int) -> Callable[[object], bool]:
        """Predicate that checks maximum length."""
        return lambda x: hasattr(x, "__len__") and len(x) <= length

    @staticmethod
    def matches_regex(pattern: str) -> Callable[[object], bool]:
        """Predicate that checks regex match."""
        compiled_pattern = re.compile(pattern)
        return lambda x: isinstance(x, str) and bool(compiled_pattern.match(x))

    @staticmethod
    def is_email() -> Callable[[object], bool]:
        """Predicate that checks if value is valid email."""
        return FlextPredicates.matches_regex(EMAIL_PATTERN)

    @staticmethod
    def is_uuid() -> Callable[[object], bool]:
        """Predicate that checks if value is valid UUID."""
        return FlextPredicates.matches_regex(UUID_PATTERN)

    @staticmethod
    def is_url() -> Callable[[object], bool]:
        """Predicate that checks if value is valid URL."""
        return FlextPredicates.matches_regex(URL_PATTERN)

    @staticmethod
    def in_range(min_val: float, max_val: float) -> Callable[[object], bool]:
        """Predicate that checks if value is in numeric range."""
        return lambda x: isinstance(x, (int, float)) and min_val <= x <= max_val

    @staticmethod
    def contains(item: object) -> Callable[[object], bool]:
        """Predicate that checks if container contains item."""
        return lambda x: hasattr(x, "__contains__") and item in x

    @staticmethod
    def starts_with(prefix: str) -> Callable[[object], bool]:
        """Predicate that checks if string starts with prefix."""
        return lambda x: isinstance(x, str) and x.startswith(prefix)

    @staticmethod
    def ends_with(suffix: str) -> Callable[[object], bool]:
        """Predicate that checks if string ends with suffix."""
        return lambda x: isinstance(x, str) and x.endswith(suffix)


# =============================================================================
# PYDANTIC-ENHANCED FIELD VALIDATORS - sem underscore conforme diretrizes
# =============================================================================


def validate_required_field(
    value: object,
    field_name: str,
) -> FlextValidationResult:
    """Validate required field using Pydantic result model."""
    if value is None:
        return FlextValidationResult.failure(
            f"Field '{field_name}' is required but was None",
            field_name,
        )

    if isinstance(value, str) and not value.strip():
        return FlextValidationResult.failure(
            f"Field '{field_name}' is required but was empty",
            field_name,
        )

    return FlextValidationResult.success()


def validate_string_field(
    value: object,
    field_name: str,
    min_length: int = 0,
    max_length: int | None = None,
) -> FlextValidationResult:
    """Validate string field with Pydantic configuration."""
    try:
        # Use Pydantic to validate configuration
        config = FlextValidationConfig(
            field_name=field_name,
            min_length=min_length,
            max_length=max_length,
        )
    except (TypeError, ValueError, AttributeError) as e:
        return FlextValidationResult.failure(f"Invalid validation config: {e}")

    if not isinstance(value, str):
        return FlextValidationResult.failure(
            f"Field '{config.field_name}' must be a string, got {type(value).__name__}",
            config.field_name,
        )

    if len(value) < config.min_length:
        return FlextValidationResult.failure(
            f"Field '{config.field_name}' must be at least "
            f"{config.min_length} characters",
            config.field_name,
        )

    if config.max_length is not None and len(value) > config.max_length:
        return FlextValidationResult.failure(
            f"Field '{config.field_name}' must be at most "
            f"{config.max_length} characters",
            config.field_name,
        )

    return FlextValidationResult.success()


def validate_numeric_field(
    value: object,
    field_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> FlextValidationResult:
    """Validate numeric field with Pydantic configuration."""
    try:
        config = FlextValidationConfig(
            field_name=field_name,
            min_val=min_val,
            max_val=max_val,
        )
    except (TypeError, ValueError, AttributeError) as e:
        return FlextValidationResult.failure(f"Invalid validation config: {e}")

    if not isinstance(value, (int, float)):
        return FlextValidationResult.failure(
            f"Field '{config.field_name}' must be a number, got {type(value).__name__}",
            config.field_name,
        )

    if config.min_val is not None and value < config.min_val:
        return FlextValidationResult.failure(
            f"Field '{config.field_name}' must be at least {config.min_val}",
            config.field_name,
        )

    if config.max_val is not None and value > config.max_val:
        return FlextValidationResult.failure(
            f"Field '{config.field_name}' must be at most {config.max_val}",
            config.field_name,
        )

    return FlextValidationResult.success()


def validate_email_field(
    value: object,
    field_name: str,
) -> FlextValidationResult:
    """Validate email field."""
    if not isinstance(value, str):
        return FlextValidationResult.failure(
            f"Field '{field_name}' must be a string",
            field_name,
        )

    if not FlextValidators.is_email(value):
        return FlextValidationResult.failure(
            f"Field '{field_name}' must be a valid email address",
            field_name,
        )

    return FlextValidationResult.success()


# =============================================================================
# ENTITY VALIDATION FUNCTIONS - sem underscore conforme diretrizes
# =============================================================================


def validate_entity_id(entity_id: object) -> bool:
    """Validate entity ID is valid."""
    return FlextValidators.is_non_empty_string(entity_id)


def validate_non_empty_string(value: object) -> bool:
    """Validate value is non-empty string."""
    return FlextValidators.is_non_empty_string(value)


# =============================================================================
# FLEXT VALIDATION - Consolidados com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextValidation(FlextValidators):
    """Main validation interface providing comprehensive validation capabilities.

    Serves as the primary external API for all validation operations, inheriting
    complete functionality from consolidated base implementation while adding
    composition and high-level validation patterns.

    Architecture:
        - Inherits all basic validation methods from FlextValidators
        - Adds validator composition methods for complex scenarios
        - Provides validation configuration creation
        - Maintains clean separation between predicates and validators

    Inherited Validation Methods:
        - is_not_none, is_string, is_non_empty_string: Basic type checks
        - is_email, is_numeric, is_boolean: Type-specific validations
        - has_min_length, has_max_length: String length validations
        - matches_pattern: Regular expression pattern matching

    Composition Features:
        - chain: AND logic for multiple validators
        - any_of: OR logic for alternative validators
        - create_validation_config: Pydantic-based configuration

    Usage:
        # Basic validation
        if FlextValidation.is_email("user@example.com"):
            process_email()

        # Composed validation
        email_validator = FlextValidation.chain(
            FlextValidation.is_string,
            FlextValidation.is_email
        )

        # Configuration-based validation
        config = FlextValidation.create_validation_config(
            field_name="password",
            min_length=8,
            max_length=128
        )
    """

    # Inherited from FlextValidators automatically via class inheritance
    # All basic validation methods are now available without explicit delegation

    # Entity validation functions
    validate_entity_id = validate_entity_id
    validate_non_empty_string = validate_non_empty_string

    @staticmethod
    def chain(*validators: Callable[[object], bool]) -> Callable[[object], bool]:
        """Chain multiple validators together with AND logic.

        Args:
            *validators: Validator functions to chain

        Returns:
            Chained validator function that returns True only if all validators pass

        """

        def chained_validator(value: object) -> bool:
            return all(validator(value) for validator in validators)

        return chained_validator

    @staticmethod
    def any_of(*validators: Callable[[object], bool]) -> Callable[[object], bool]:
        """Chain multiple validators together with OR logic.

        Args:
            *validators: Validator functions to chain

        Returns:
            Chained validator function that returns True if any validator passes

        """

        def any_validator(value: object) -> bool:
            return any(validator(value) for validator in validators)

        return any_validator

    # Field validation functions
    validate_required_field = validate_required_field
    validate_string_field = validate_string_field
    validate_numeric_field = validate_numeric_field
    validate_email_field = validate_email_field

    @staticmethod
    def create_validation_config(
        field_name: str,
        min_length: int = 0,
        max_length: int | None = None,
    ) -> FlextValidationConfig:
        """Create validation configuration using Pydantic.

        Args:
            field_name: Name of the field being validated
            min_length: Minimum length for string fields
            max_length: Maximum length for string fields

        Returns:
            Validated configuration object

        """
        return FlextValidationConfig(
            field_name=field_name,
            min_length=min_length,
            max_length=max_length,
        )

    @classmethod
    def safe_validate(
        cls,
        value: object,
        validator: Callable[[object], bool],
    ) -> FlextResult[object]:
        """Safely validate value with FlextResult error handling.

        Complex orchestration pattern combining inherited validation capabilities
        with FlextResult patterns for comprehensive safe validation. Automatically
        captures validation errors and converts them to FlextResult failures.

        Args:
            value: Value to validate
            validator: Validator function to apply

        Returns:
            FlextResult with validation outcome

        """
        try:
            if validator(value):
                return FlextResult.ok(value)
            return FlextResult.fail(f"Validation failed for value: {value}")
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"Validation error: {e}")


# Re-expose FlextPredicates class (already defined above)
# This maintains the original API while using the consolidated implementation


# FlextValidationResult class already defined above with consolidated implementation
# This maintains backward compatibility through class re-exposure

# Create aliases for backward compatibility
ValidationResultFactory = FlextValidationResult


# =============================================================================
# CONVENIENCE FUNCTIONS - High-level validation helpers
# =============================================================================


def validate_required(
    value: object,
    field_name: str = "field",
) -> FlextValidationResult:
    """Validate that a field has a required value with comprehensive checking.

    Performs null/None validation ensuring the field contains a meaningful value.
    Supports various data types and provides clear error messaging.

    Args:
        value: Value to validate (any type accepted)
        field_name: Name of field for error messages and debugging

    Returns:
        FlextValidationResult with is_valid boolean and error details

    Usage:
        result = validate_required(user_input, "username")
        if not result.is_valid:
            return error_response(result.error_message)

    """
    return validate_required_field(value, field_name)


def validate_string(
    value: object,
    field_name: str = "field",
    min_length: int = 0,
    max_length: int | None = None,
) -> FlextValidationResult:
    """Validate string field with comprehensive length and type constraints.

    Performs type checking, length validation, and content verification for
    string fields. Supports configurable length constraints and clear error reporting.

    Args:
        value: Value to validate (expected to be string)
        field_name: Name of field for error messages and debugging
        min_length: Minimum required length (inclusive, default 0)
        max_length: Maximum allowed length (inclusive, None for unlimited)

    Returns:
        FlextValidationResult with is_valid boolean and detailed error information

    Usage:
        result = validate_string(password, "password", min_length=8, max_length=128)
        if result.is_valid:
            store_password(password)

    """
    return validate_string_field(value, field_name, min_length, max_length)


def validate_numeric(
    value: object,
    field_name: str = "field",
    min_val: float | None = None,
    max_val: float | None = None,
) -> FlextValidationResult:
    """Validate numeric field with comprehensive range and type constraints.

    Performs type checking and range validation for numeric values (int, float).
    Supports configurable range constraints and detailed error reporting.

    Args:
        value: Value to validate (expected to be numeric)
        field_name: Name of field for error messages and debugging
        min_val: Minimum allowed value (inclusive, None for no minimum)
        max_val: Maximum allowed value (inclusive, None for no maximum)

    Returns:
        FlextValidationResult with is_valid boolean and range error details

    Usage:
        result = validate_numeric(age, "age", min_val=0, max_val=150)
        if result.is_valid:
            process_age(age)

    """
    return validate_numeric_field(value, field_name, min_val, max_val)


def validate_email(value: object, field_name: str = "field") -> FlextValidationResult:
    """Validate email field with format and structure checking.

    Performs comprehensive email validation including format checking,
    domain validation, and structural requirements verification.

    Args:
        value: Value to validate (expected to be email string)
        field_name: Name of field for error messages and debugging

    Returns:
        FlextValidationResult with is_valid boolean and format error details

    Usage:
        result = validate_email(email_input, "user_email")
        if result.is_valid:
            send_confirmation_email(email_input)
        else:
            show_error("Invalid email format")

    """
    return validate_email_field(value, field_name)


# =============================================================================
# ESSENTIAL ALIASES - apenas os necessários para funcionamento interno
# =============================================================================

# Aliases essenciais para funcionamento de outros módulos
_BaseValidators = FlextValidators
_BasePredicates = FlextPredicates


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Core consolidated classes - objetos sem underscore
    "FlextPredicates",
    "FlextValidation",
    "FlextValidationConfig",
    "FlextValidationResult",
    "FlextValidators",
    # Essential internal aliases
    "_BasePredicates",
    "_BaseValidators",
    # Essential legacy aliases
    "ValidationResultFactory",
    # Convenience functions
    "validate_email",
    "validate_email_field",
    # Entity validation functions
    "validate_entity_id",
    "validate_non_empty_string",
    "validate_numeric",
    "validate_numeric_field",
    "validate_required",
    # Field validation functions - objetos sem underscore
    "validate_required_field",
    "validate_string",
    "validate_string_field",
]
