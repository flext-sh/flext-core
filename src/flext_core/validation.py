"""FLEXT Core Validation Module.

Comprehensive validation system for the FLEXT Core library providing consolidated
functionality through inheritance from specialized validation base classes.

Architecture:
    - Inheritance from specialized validation base classes (_BaseValidators,
    _BasePredicates)
    - Direct base exposure eliminating nested class overhead
    - Single source of truth pattern with _validation_base.py as internal definitions
    - FlextResult integration for consistent error handling across the system
    - No underscore prefixes on public objects for clean API

Validation System Components:
    - FlextValidators: Core validation functions inherited from _BaseValidators
    - FlextPredicates: Functional predicates inherited from _BasePredicates
    - FlextValidation: Main validation class with composition and orchestration
    - FlextValidationResult: Structured validation results with success/failure handling
    - FlextValidationConfig: Pydantic-based configuration for validation parameters
    - Convenience functions: High-level validation helpers with FlextResult integration

Maintenance Guidelines:
    - Add new validator types to _validation_base.py first
    - Use inheritance from base classes for consistent functionality
    - Maintain functional programming patterns with pure functions
    - Integrate FlextResult pattern for all operations that can fail
    - Keep validator composition patterns for complex validation scenarios

Design Decisions:
    - Single source of truth with _validation_base.py for internal definitions
    - Direct inheritance from base classes eliminating code duplication
    - Clean public API with Flext* prefixed classes
    - FlextResult integration for consistent error handling
    - Backward compatibility through function aliases

Validation Patterns:
    - Simple validation: FlextValidators.is_string(value)
    - Structured validation: validate_string(value, "email", min_length=5)
    - Composed validation: FlextValidation.chain(validator1, validator2)
    - Predicate filtering: FlextPredicates.non_empty_string()(value)
    - Configuration-based: FlextValidationConfig with Pydantic models
    - Result-based: All validators return FlextResult for safe error handling

Dependencies:
    - _validation_base: Foundation validation implementations
    - result: FlextResult pattern for consistent error handling
    - constants: Core regex patterns and validation constants

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._validation_base import (
    _BasePredicates,
    _BaseValidators,
    _validate_email_field,
    _validate_entity_id,
    _validate_non_empty_string,
    _validate_numeric_field,
    _validate_required_field,
    _validate_string_field,
    _ValidationConfig,
    _ValidationResult,
)
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

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


# =============================================================================
# API RE-EXPOSURE - FlextPredicates and FlextValidationResult already defined above
# =============================================================================

# FlextPredicates = _BasePredicates (already defined above)
# FlextValidationResult = _ValidationResult (already defined above)
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
    # Convenience functions with flext_ prefix
    "flext_validate_email",
    "flext_validate_email_field",
    "flext_validate_entity_id",
    "flext_validate_non_empty_string",
    "flext_validate_numeric",
    "flext_validate_numeric_field",
    "flext_validate_required",
    "flext_validate_required_field",
    "flext_validate_string",
    "flext_validate_string_field",
]
