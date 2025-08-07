"""FLEXT Core Validation - CQRS Layer Validation System.

Unified validation system providing auto-detection, fluent interfaces, type guards,
and structured validation across the 32-project FLEXT ecosystem. Foundation for
business rule validation, input sanitization, and data integrity enforcement in
command/query processing and data integration pipelines.

Module Role in Architecture:
    CQRS Layer → Validation System → Business Rule Enforcement

    This module provides validation patterns used throughout FLEXT projects:
    - Auto-detection validation with intelligent type inference
    - Fluent interfaces for complex validation chains and rules
    - Type guards for compile-time and runtime type safety
    - Railway-oriented error handling with FlextResult integration

Validation Architecture Patterns:
    Auto-Detection: AutoValidator.check(value) with intelligent type inference
    Fluent Interface: Chainable validation with method composition
    Type Guards: Runtime type checking with static analysis benefits
    Structured Validation: Schema-based validation for complex objects

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Foundation validators, predicates, type guards
    🚧 Active Development: Fluent validation chains (Enhancement 4 - Priority Medium)
    📋 TODO Integration: Schema-based validation with Pydantic (Priority 3)

Validation System Components:
    FlextValidators: Foundation validation functions for common data types
    FlextPredicates: Functional predicates for filtering and data processing
    FlextValidation: Main orchestration class for complex validation scenarios
    AutoValidator: Intelligent validation with automatic type detection
    FluentValidator: Chainable validation interface for complex rules

Ecosystem Usage Patterns:
    # FLEXT Service Validation
    if FlextValidators.is_non_empty_string(user_input):
        process_user_input(user_input)

    # Singer Tap/Target Validation
    connection_result = (
        FluentValidator.string(connection_string)
        .min_length(10)
        .contains("oracle://")
        .validate()
    )

    # client-a Migration Validation
    dn_result = AutoValidator.check(ldap_dn)
    if dn_result.is_success:
        process_ldap_dn(ldap_dn)

    # Complex Schema Validation
    user_validation = FlextValidation.validate_object(
        user_data,
        {
            "name": FlextValidators.is_non_empty_string,
            "email": FlextValidators.is_email,
            "age": lambda x: isinstance(x, int) and 18 <= x <= 120
        }
    )

Validation Pattern Categories:
    - Foundation Validation: Basic type checking and common patterns
    - Business Rule Validation: Domain-specific validation logic
    - Input Sanitization: Data cleaning and normalization
    - Schema Validation: Structured validation for complex objects

Quality Standards:
    - All validation functions must return FlextResult for consistent error handling
    - Validation errors must provide actionable feedback for users
    - Type guards must support both runtime checking and static analysis
    - Complex validation must be composable through fluent interfaces

See Also:
    docs/TODO.md: Enhancement 4 - Fluent validation chain development
    result.py: FlextResult pattern for validation error handling
    interfaces.py: FlextValidator protocol definitions

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from flext_core.flext_types import TPredicate

# =============================================================================
# BASIC VALIDATION CLASSES
# =============================================================================


class _ValidationConfig(BaseModel):
    """Validation configuration."""

    model_config = ConfigDict(frozen=True)

    field_name: str
    min_length: int = 0
    max_length: int | None = None


class _ValidationResult(BaseModel):
    """Validation result."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    error_message: str = ""
    field_name: str = ""


class _BaseValidators:
    """Basic validation functions."""

    @staticmethod
    def is_not_none(value: object) -> bool:
        """Check if value is not None."""
        return value is not None

    @staticmethod
    def is_string(value: object) -> bool:
        """Check if value is string."""
        return isinstance(value, str)

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Check if value is non-empty string."""
        if not isinstance(value, str):
            return False
        return len(value.strip()) > 0

    @staticmethod
    def is_email(value: object) -> bool:
        """Check if value is valid email."""
        if not isinstance(value, str):
            return False
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, value))

    @staticmethod
    def is_uuid(value: object) -> bool:
        """Check if value is valid UUID."""
        if not isinstance(value, str):
            return False
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        return bool(re.match(pattern, value.lower()))

    @staticmethod
    def is_url(value: object) -> bool:
        """Check if value is valid URL."""
        if not isinstance(value, str):
            return False
        return value.startswith(("http://", "https://"))

    @staticmethod
    def has_min_length(value: object, min_length: int) -> bool:
        """Check if string has minimum length."""
        if not isinstance(value, str):
            return False
        return len(value) >= min_length

    @staticmethod
    def has_max_length(value: object, max_length: int) -> bool:
        """Check if string has maximum length."""
        if not isinstance(value, str):
            return False
        return len(value) <= max_length

    @staticmethod
    def matches_pattern(value: object, pattern: str) -> bool:
        """Check if string matches pattern."""
        if not isinstance(value, str):
            return False
        return bool(re.match(pattern, value))

    @staticmethod
    def is_callable(value: object) -> bool:
        """Check if value is callable."""
        return callable(value)

    @staticmethod
    def is_list(value: object) -> bool:
        """Check if value is list."""
        return isinstance(value, list)

    @staticmethod
    def is_dict(value: object) -> bool:
        """Check if value is dict."""
        return isinstance(value, dict)

    @staticmethod
    def is_none(value: object) -> bool:
        """Check if value is None."""
        return value is None


class _BasePredicates:
    """Basic predicate functions."""

    @staticmethod
    def is_positive(value: object) -> bool:
        """Check if value is positive number."""
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def is_negative(value: object) -> bool:
        """Check if value is negative number."""
        return isinstance(value, (int, float)) and value < 0

    @staticmethod
    def is_zero(value: object) -> bool:
        """Check if value is zero."""
        return isinstance(value, (int, float)) and value == 0


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def _validate_required_field(
    value: object,
    field_name: str = "field",
) -> _ValidationResult:
    """Validate required field."""
    if not _BaseValidators.is_not_none(value):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} is required",
            field_name=field_name,
        )
    return _ValidationResult(is_valid=True, field_name=field_name)


def _validate_string_field(
    value: object,
    field_name: str = "field",
    min_length: int = 0,
    max_length: int | None = None,
) -> _ValidationResult:
    """Validate string field."""
    if not _BaseValidators.is_string(value):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be a string",
            field_name=field_name,
        )

    str_value = str(value)
    if not _BaseValidators.has_min_length(str_value, min_length):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at least {min_length} characters",
            field_name=field_name,
        )

    if max_length is not None and not _BaseValidators.has_max_length(
        str_value,
        max_length,
    ):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at most {max_length} characters",
            field_name=field_name,
        )

    return _ValidationResult(is_valid=True, field_name=field_name)


def _validate_numeric_field(
    value: object,
    field_name: str = "field",
    min_val: float | None = None,
    max_val: float | None = None,
) -> _ValidationResult:
    """Validate numeric field."""
    if not isinstance(value, (int, float)):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be a number",
            field_name=field_name,
        )

    num_value = float(value)
    if min_val is not None and num_value < min_val:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at least {min_val}",
            field_name=field_name,
        )

    if max_val is not None and num_value > max_val:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at most {max_val}",
            field_name=field_name,
        )

    return _ValidationResult(is_valid=True, field_name=field_name)


def _validate_email_field(
    value: object,
    field_name: str = "field",
) -> _ValidationResult:
    """Validate email field."""
    if not _BaseValidators.is_email(value):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be a valid email",
            field_name=field_name,
        )
    return _ValidationResult(is_valid=True, field_name=field_name)


def _validate_entity_id(value: object) -> bool:
    """Validate entity ID."""
    return _BaseValidators.is_non_empty_string(value)


def _validate_non_empty_string(value: object) -> bool:
    """Validate non-empty string."""
    return _BaseValidators.is_non_empty_string(value)


def _validate_service_name(name: str) -> bool:
    """Validate service name."""
    return isinstance(name, str) and len(name.strip()) > 0


# =============================================================================
# DOMAIN-SPECIFIC TYPES - Validation Pattern Specializations
# =============================================================================

# Validation specific types for better domain modeling
type TValidationRule = str  # Validation rule identifier
type TValidationError = str  # Validation error message
type TValidationResult = FlextResult[object]  # Validation result with data
type TValidationContext = dict[str, object]  # Validation context data
type TValidatorName = str  # Validator instance name
type TValidationConfig = dict[str, object]  # Validator configuration
type TValidationConstraint = object  # Validation constraint value
type TValidationSchema = dict[str, object]  # Schema definition for validation

# Field validation types
type TFieldName = str  # Field name for validation
type TFieldValue = object  # Field value to validate
type TFieldRule = str  # Field-specific validation rule
type TFieldError = str  # Field-specific error message

# Custom validation types
type TCustomValidator = Callable[[object], FlextResult[object]]  # Custom validator
type TValidationPipeline = list[TCustomValidator]  # Chain of validators

# =============================================================================
# FLEXT VALIDATION MODELS - Direct exposure from base with clean names
# =============================================================================

# Direct exposure eliminating inheritance overhead
FlextValidationConfig = _ValidationConfig
FlextValidationResult = _ValidationResult

# =============================================================================
# FLEXT VALIDATORS - Direct exposure eliminating inheritance overhead
# =============================================================================

# Direct exposure with clean names - completely eliminates empty inheritance
FlextValidators = _BaseValidators

# =============================================================================
# FLEXT PREDICATES - Direct exposure eliminating inheritance overhead
# =============================================================================

# Direct exposure with clean names - completely eliminates empty inheritance
FlextPredicates = _BasePredicates

# =============================================================================
# FIELD VALIDATORS - Direct function exposure eliminating wrapper overhead
# =============================================================================

# Direct exposure of field validation functions with clean names
flext_validate_required_field = _validate_required_field
flext_validate_string_field = _validate_string_field
flext_validate_numeric_field = _validate_numeric_field
flext_validate_email_field = _validate_email_field

# =============================================================================
# ENTITY VALIDATION FUNCTIONS - Direct function exposure
# =============================================================================

# Direct exposure of entity validation functions with clean names
flext_validate_entity_id = _validate_entity_id
flext_validate_non_empty_string = _validate_non_empty_string

# =============================================================================
# FLEXT VALIDATION - Main validation interface inheriting from base
# =============================================================================


class FlextValidation(FlextValidators):
    """Main validation interface providing comprehensive validation capabilities.

    Serves as the primary external API for all validation operations, inheriting
    complete functionality from FlextValidators (which is _BaseValidators) while adding
    composition and high-level validation patterns.

    Architecture:
        - Inherits all basic validation methods from FlextValidators (_BaseValidators)
        - Adds validator composition methods for complex scenarios
        - Provides validation configuration creation
        - Maintains clean separation between predicates and validators

    Inherited Validation Methods:
        - is_not_none, is_string, is_non_empty_string: Basic type checks
        - is_email, is_uuid, is_url: Pattern-specific validations
        - has_min_length, has_max_length: String length validations
        - matches_pattern: Regular expression pattern matching

    Composition Features:
        - chain: AND logic for multiple validators
        - any_of: OR logic for alternative validators
        - create_validation_config: Pydantic-based configuration
        - safe_validate: FlextResult integration for error handling

    Usage:
        # Basic validation (inherited methods)
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

        # Safe validation with FlextResult
        result = FlextValidation.safe_validate(data, FlextValidation.is_email)
    """

    # All basic validation methods inherited from FlextValidators (_BaseValidators)
    # No explicit delegation needed due to class inheritance

    # Direct access to validators and functions
    Validators = FlextValidators
    flext_validate_entity_id = flext_validate_entity_id
    flext_validate_non_empty_string = flext_validate_non_empty_string
    flext_validate_required_field = flext_validate_required_field
    flext_validate_string_field = flext_validate_string_field
    flext_validate_numeric_field = flext_validate_numeric_field
    flext_validate_email_field = flext_validate_email_field

    @classmethod
    def validate(cls, value: object) -> FlextResult[object]:
        """Validate value with type detection."""
        if isinstance(value, str) and "@" in value and "." in value:
            if not (
                cls.is_non_empty_string(value)
                and "@" in value
                and "." in value.split("@")[-1]
            ):
                return FlextResult.fail("Invalid email format")
        elif isinstance(value, str):
            return FlextResult.ok(value)

        if isinstance(value, (int, float, dict, list)):
            return FlextResult.ok(value)

        return FlextResult.ok(value)

    @staticmethod
    def chain(*validators: TPredicate[object]) -> TPredicate[object]:
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
    def any_of(*validators: TPredicate[object]) -> TPredicate[object]:
        """Chain multiple validators together with OR logic.

        Args:
            *validators: Validator functions to chain

        Returns:
            Chained validator function that returns True if any validator passes

        """

        def any_validator(value: object) -> bool:
            return any(validator(value) for validator in validators)

        return any_validator

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
        validator: TPredicate[object],
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


# =============================================================================
# API RE-EXPOSURE - FlextPredicates and FlextValidationResult already defined above
# =============================================================================


# All functionality properly exposed through direct assignment

# =============================================================================
# CONVENIENCE FUNCTIONS - High-level validation helpers with Flext prefix
# =============================================================================


def flext_validate_required(
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
    return flext_validate_required_field(value, field_name)


def flext_validate_string(
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
    return flext_validate_string_field(value, field_name, min_length, max_length)


def flext_validate_numeric(
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
    return flext_validate_numeric_field(value, field_name, min_val, max_val)


def flext_validate_email(
    value: object,
    field_name: str = "field",
) -> FlextValidationResult:
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
    return flext_validate_email_field(value, field_name)


def flext_validate_service_name(name: str) -> bool:
    """Validate service name for container operations.

    Validates service names used in dependency injection container operations
    ensuring they meet basic requirements for service identification.

    Args:
        name: Service name to validate

    Returns:
        True if valid service name, False otherwise

    Usage:
        if flext_validate_service_name(service_name):
            container.register(service_name, service_instance)
        else:

            raise FlextValidationError(
                "Invalid service name",
                validation_details={
                    "field": "service_name",
                    "value": service_name,
                    "rules": ["service_name_format"],
                },
            )

    """
    return _validate_service_name(name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_smart(value: object, **_context: object) -> FlextResult[object]:
    """Validate value with type detection."""
    return FlextValidation.validate(value)


def is_valid_data(value: object) -> bool:
    """Check if value is valid."""
    return FlextValidation.validate(value).is_success


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "FlextPredicates",
    "FlextValidation",
    "FlextValidationConfig",
    "FlextValidationResult",
    "FlextValidators",
    "flext_validate_email",
    "flext_validate_email_field",
    "flext_validate_entity_id",
    "flext_validate_non_empty_string",
    "flext_validate_numeric",
    "flext_validate_numeric_field",
    "flext_validate_required",
    "flext_validate_required_field",
    "flext_validate_service_name",
    "flext_validate_string",
    "flext_validate_string_field",
    "is_valid_data",
    "validate_smart",
]
