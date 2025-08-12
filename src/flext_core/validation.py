"""Domain validation implementations for FLEXT ecosystem.

Concrete validation implementations using abstract validator patterns.
Provides business-specific validators and domain validation logic.

Classes:
    FlextValidators: Foundation validation functions.
    FlextValidation: Main validation orchestrator.

"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from flext_core.protocols import FlextValidationRule as _FlextValidationRuleProtocol
from flext_core.result import FlextResult

T = TypeVar("T")


class FlextAbstractValidator(ABC, Generic[T]):  # noqa: UP046
    """Abstract validator for validation patterns."""

    @abstractmethod
    def validate(self, value: T) -> FlextResult[T]:
        """Validate value."""
        ...


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
        """Check if the value is not None."""
        return value is not None

    @staticmethod
    def is_string(value: object) -> bool:
        """Check if the value is string."""
        return isinstance(value, str)

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Check if the value is non-empty string."""
        if not isinstance(value, str):
            return False
        return len(value.strip()) > 0

    @staticmethod
    def is_email(value: object) -> bool:
        """Check if the value is valid email."""
        if not isinstance(value, str):
            return False
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, value))

    @staticmethod
    def is_uuid(value: object) -> bool:
        """Check if the value is valid UUID."""
        if not isinstance(value, str):
            return False
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        return bool(re.match(pattern, value.lower()))

    @staticmethod
    def is_url(value: object) -> bool:
        """Check if the value is valid URL."""
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
        """Check if the value is callable."""
        return callable(value)

    @staticmethod
    def is_list(value: object) -> bool:
        """Check if the value is a list."""
        return isinstance(value, list)

    @staticmethod
    def is_dict(value: object) -> bool:
        """Check if the value is dict."""
        return isinstance(value, dict)

    @staticmethod
    def is_none(value: object) -> bool:
        """Check if the value is None."""
        return value is None


class _BasePredicates:
    """Basic predicate functions."""

    @staticmethod
    def is_positive(value: object) -> bool:
        """Check if value is a positive number."""
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def is_negative(value: object) -> bool:
        """Check if value is negative number."""
        return isinstance(value, (int, float)) and value < 0

    @staticmethod
    def is_zero(value: object) -> bool:
        """Check if the value is zero."""
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

# Field validation types

# Custom validation types

# =============================================================================
# FLEXT VALIDATION MODELS - Direct exposure from base with clean names
# =============================================================================

# Direct exposure removing inheritance overhead
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
    complete functionality from FlextValidators while adding composition and
    high-level validation patterns for complex validation scenarios.
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
            FlextResult with a validation outcome

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
        FlextValidationResult with is_valid boolean and error details.

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
        FlextValidationResult with is_valid boolean and detailed error information.

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
        FlextValidationResult with is_valid boolean and range error details.

    """
    return flext_validate_numeric_field(value, field_name, min_val, max_val)


def flext_validate_email(
    value: object,
    field_name: str = "field",
) -> FlextValidationResult:
    """Validate email field with format and structure checking.

    Performs comprehensive email validation including format checking,
    domain validation, and structural requirement verification.

    Args:
        value: Value to validate (expected to be email string)
        field_name: Name of field for error messages and debugging

    Returns:
        FlextValidationResult with is_valid boolean and format error details.

    """
    return flext_validate_email_field(value, field_name)


def flext_validate_service_name(name: str) -> bool:
    """Validate service name for container operations.

    Validates service names used in dependency injection container operations
    ensuring they meet basic requirements for service identification.

    Args:
        name: Service name to validate

    Returns:
        True if valid service name, False otherwise.

    """
    return _validate_service_name(name)


# =============================================================================
# MIGRATION NOTICE - Legacy functions moved to legacy.py
# =============================================================================

# IMPORTANT: Legacy functions validate_smart and is_valid_data have been
# completely moved to legacy.py. Import them from there if needed:
#   from flext_core.legacy import validate_smart, is_valid_data
#
# These functions are NOT re-exported here to avoid circular imports.


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

# =============================================================================
# BASE VALIDATORS - Core validation patterns
# =============================================================================


class FlextDomainValidator(FlextAbstractValidator[object]):
    """Domain-specific validation with business rules.

    SOLID compliance: Single responsibility for domain validation.
    """

    def __init__(self, business_rules: list[object] | None = None) -> None:
        """Initialize domain validator with business rules."""
        self.business_rules = business_rules or []

    def validate_value(self, value: object) -> FlextResult[object]:
        """Validate value against business rules (Strategy pattern)."""
        for rule in self.business_rules:
            if callable(rule) and not rule(value):
                return FlextResult.fail("Business rule validation failed")
        return FlextResult.ok(value)

    # Backward-compatible method name maintained during migration
    def validate(self, value: object) -> FlextResult[object]:
        """Alias to validate_value for compatibility with legacy code."""
        return self.validate_value(value)


__all__: list[str] = [
    # Base validators
    "FlextBaseValidator",
    "FlextDomainValidator",
    "FlextPredicates",
    "FlextValidation",
    "FlextValidationConfig",
    "FlextValidationResult",
    "FlextValidators",
    # Validation functions
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
    # Note: Legacy functions (is_valid_data, validate_smart) moved to legacy.py
    # Import from flext_core.legacy if needed for backward compatibility
]

# Backward-compatibility alias: prefer FlextAbstractValidator
FlextBaseValidator = FlextAbstractValidator

# Re-export protocol name for convenience
FlextValidationRule = _FlextValidationRuleProtocol


# =============================================================================
# LEGACY VALIDATION FUNCTIONS - Backward compatibility
# Note: These functions are already defined above. This section is for
# backward compatibility imports only.
# =============================================================================
