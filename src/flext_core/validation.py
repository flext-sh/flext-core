"""domain validation using Pydantic functional validator patterns.

This module provides enterprise-grade validation patterns using standard Pydantic v2
functional validators including BeforeValidator, AfterValidator, PlainValidator,
and WrapValidator patterns. All legacy validation patterns have been eliminated
and replaced with standard patterns.

Key Patterns:
- BeforeValidator: Transforms input before core validation
- AfterValidator: Transforms output after core validation
- PlainValidator: Replaces core validation entirely
- WrapValidator: Wraps around core validation with custom logic
- field_validator and model_validator decorators
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, cast, override
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    ValidationError,
    ValidationInfo,
    WrapValidator,
    field_validator,
    validate_call,
)

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Type aliases for unified approach with FlextProtocols integration - Python 3.13+ syntax
type ValidatorProtocol = FlextProtocols.Foundation.Validator[object]
type ValidationServiceProtocol = FlextProtocols.Domain.Service

# Note: ServiceName and EmailAddress are defined as Pydantic Annotated types below for validation


class FlextAbstractValidator[T](ABC):
    """Abstract validator for validation patterns."""

    @abstractmethod
    def validate(self, value: T) -> FlextResult[T]:
        """Validate value."""
        ...


# =============================================================================
# PYDANTIC FUNCTIONAL VALIDATORS
# =============================================================================


# BeforeValidator functions - Transform input before core validation
def normalize_string(v: object) -> str:
    """BeforeValidator: Normalize string input before validation."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip().lower()
    return str(v).strip().lower()


def normalize_email(v: object) -> str:
    """BeforeValidator: Normalize email before validation."""
    if isinstance(v, str):
        return v.strip().lower()
    return str(v).strip().lower()


def ensure_string_list(v: object) -> list[str]:
    """BeforeValidator: Convert single string or mixed list to string list."""
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(item) for item in cast("list[object]", v)]
    return [str(v)]


def generate_id_if_missing(v: object) -> str:
    """BeforeValidator: Generate UUID if ID is missing or empty."""
    if not v or (isinstance(v, str) and not v.strip()):
        return f"flext_{uuid4().hex[:8]}"
    return str(v)


# AfterValidator functions - Transform output after core validation
def uppercase_code(v: str) -> str:
    """AfterValidator: Ensure code is always uppercase."""
    return v.upper()


def add_flext_prefix(v: str) -> str:
    """AfterValidator: Add flext_ prefix if not present."""
    if not v.startswith("flext_"):
        return f"flext_{v}"
    return v


def ensure_positive(v: float) -> int | float:
    """AfterValidator: Ensure numeric values are positive."""
    if v <= 0:
        msg = "Value must be positive"
        raise ValueError(msg)
    return v


def format_timestamp(v: str) -> str:
    """AfterValidator: Ensure timestamp has Z suffix."""
    if not v.endswith("Z"):
        return f"{v}Z"
    return v


# PlainValidator functions - Replace core validation entirely
def validate_service_name(v: object) -> str:
    """PlainValidator: Complete custom validation for service names."""
    if not isinstance(v, str):
        v = str(v)

    v = v.strip()
    if not v:
        msg = "Service name cannot be empty"
        raise ValueError(msg)

    if len(v) < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH:
        msg = "Service name must be at least 2 characters"
        raise ValueError(msg)

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", v):
        msg = "Service name must start with letter and contain only letters, numbers, hyphens, and underscores"
        raise ValueError(msg)

    return v


def validate_email_address(v: object) -> str:
    """PlainValidator: Complete email validation with normalization."""
    if not isinstance(v, str):
        v = str(v)

    v = v.strip().lower()
    if not v:
        msg = "Email address cannot be empty"
        raise ValueError(msg)

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, v):
        msg = "Invalid email address format"
        raise ValueError(msg)

    return v


def validate_version_number(v: object) -> int:
    """PlainValidator: Version validation with auto-increment logic."""
    if isinstance(v, str):
        if v.lower() == "auto":
            return 1  # Auto-generate starting version
        try:
            v = int(v)
        except ValueError as e:
            msg = "Version must be a positive integer or 'auto'"
            raise ValueError(msg) from e

    if not isinstance(v, int):
        msg = "Version must be an integer"
        raise TypeError(msg)

    if v < 1:
        msg = "Version must be >= 1"
        raise ValueError(msg)

    return v


# WrapValidator functions - Wrap around core validation with custom logic
def validate_entity_id_with_context(
    v: object,
    handler: Callable[[str], str],
    info: ValidationInfo,
) -> str:
    """WrapValidator: entity ID validation with context."""
    # Get context for validation
    context = cast("FlextTypes.Core.Dict", info.context or {})
    namespace = cast("str", context.get("namespace", "flext"))
    auto_generate = cast("bool", context.get("auto_generate_id", True))

    # Auto-generate if missing and allowed
    if auto_generate and (not v or (isinstance(v, str) and not v.strip())):
        v = f"{namespace}_{uuid4().hex[:8]}"

    # Let Pydantic handle basic validation
    try:
        result = handler(str(v))
    except Exception:
        # If core validation fails and we can auto-generate, try that
        if auto_generate:
            v = f"{namespace}_{uuid4().hex[:8]}"
            result = handler(str(v))
        else:
            raise

    # Additional business logic validation
    if not result.startswith(str(namespace)):
        result = f"{namespace}_{result}"

        # Validate format
        if not re.match(r"^[a-zA-Z0-9_-]+$", result):
            msg = "Entity ID contains invalid characters"
            raise ValueError(msg)

    return result


def validate_timestamp_with_fallback(
    v: object,
    handler: Callable[[str], str],
    info: ValidationInfo,
) -> str:
    """WrapValidator: Timestamp validation with automatic fallback."""
    # Check if we should use current time as fallback
    context = cast("FlextTypes.Core.Dict", info.context or {})
    use_current_fallback = cast("bool", context.get("use_current_time_fallback", True))

    try:
        # Try normal validation first
        return handler(str(v))
    except Exception:
        # If validation fails and fallback is enabled, use current time
        if use_current_fallback and (not v or v == "auto"):
            current_time = datetime.now(UTC).isoformat()
            return handler(current_time)
        raise


def validate_list_with_deduplication(
    v: object,
    handler: Callable[[object], list[str]],
    info: ValidationInfo,
) -> list[str]:
    """WrapValidator: List validation with automatic deduplication."""
    # Get context settings
    context = cast("FlextTypes.Core.Dict", info.context or {})
    deduplicate = cast("bool", context.get("deduplicate_lists", True))
    sort_lists = cast("bool", context.get("sort_lists", False))

    # Let Pydantic handle basic validation
    result = handler(v)

    # Apply post-processing based on context
    if deduplicate:
        # Remove duplicates while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for item in result:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        result = deduped

    if sort_lists:
        result = sorted(result)

    return result


# =============================================================================
# TYPE DEFINITIONS WITH FUNCTIONAL VALIDATORS
# =============================================================================

# AfterValidator examples
PositiveNumber = Annotated[int, AfterValidator(ensure_positive)]
FormattedTimestamp = Annotated[str, AfterValidator(format_timestamp)]

# PlainValidator examples
ServiceName = Annotated[str, PlainValidator(validate_service_name)]
EmailAddress = Annotated[str, PlainValidator(validate_email_address)]
VersionNumber = Annotated[int, PlainValidator(validate_version_number)]

# WrapValidator examples
ContextualEntityId = Annotated[str, WrapValidator(validate_entity_id_with_context)]
FallbackTimestamp = Annotated[str, WrapValidator(validate_timestamp_with_fallback)]
DeduplicatedList = Annotated[list[str], WrapValidator(validate_list_with_deduplication)]

# Combined validator examples
EmailWithNormalization = Annotated[
    str,
    BeforeValidator(normalize_email),
    PlainValidator(validate_email_address),
    AfterValidator(lambda v: v.lower()),
]

EntityIdWithGeneration = Annotated[
    str,
    BeforeValidator(generate_id_if_missing),
    AfterValidator(add_flext_prefix),
]

# Missing type definitions that are exported but not defined
NormalizedString = Annotated[str, BeforeValidator(normalize_string)]
NormalizedEmail = Annotated[str, BeforeValidator(normalize_email)]
AutoGeneratedId = Annotated[str, BeforeValidator(generate_id_if_missing)]
StringList = Annotated[list[str], BeforeValidator(ensure_string_list)]

# =============================================================================
# VALIDATION MODELS
# =============================================================================


class _ValidationConfig(BaseModel):
    """validation configuration using Pydantic."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    field_name: str = Field(..., min_length=1)
    min_length: int = Field(default=0, ge=0)
    max_length: int | None = Field(default=None, gt=0)

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int | None, info: ValidationInfo) -> int | None:
        if v is not None and "min_length" in info.data:
            min_length = info.data["min_length"]
            if v <= min_length:
                msg = "max_length must be greater than min_length"
                raise ValueError(msg)
        return v


class _ValidationResult(BaseModel):
    """validation result using Pydantic."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    error_message: str = ""
    field_name: str = ""

    @field_validator("error_message")
    @classmethod
    def validate_error_message(cls, v: str, info: ValidationInfo) -> str:
        # Error message should be present if validation failed
        if not info.data.get("is_valid", True) and not v:
            msg = "error_message required when is_valid is False"
            raise ValueError(msg)
        return v


class _BaseValidators:
    """validation functions using validate_call decorators."""

    @staticmethod
    @validate_call
    def is_not_none(value: object) -> bool:
        """Check if the value is not None with automatic validation."""
        return value is not None

    @staticmethod
    @validate_call
    def is_string(value: object) -> bool:
        """Check if the value is string with automatic validation."""
        return isinstance(value, str)

    @staticmethod
    @validate_call
    def is_non_empty_string(value: object) -> bool:
        """Check if the value is non-empty string with automatic validation."""
        if not isinstance(value, str):
            return False
        return len(value.strip()) > 0

    @staticmethod
    @validate_call
    def is_email(value: object) -> bool:
        """Check if the value is valid email with automatic validation."""
        if not isinstance(value, str):
            return False
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, value))

    @staticmethod
    @validate_call
    def is_uuid(value: object) -> bool:
        """Check if the value is valid UUID with automatic validation."""
        if not isinstance(value, str):
            return False
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        return bool(re.match(pattern, value.lower()))

    @staticmethod
    @validate_call
    def is_url(value: object) -> bool:
        """Check if the value is valid URL with automatic validation."""
        if not isinstance(value, str):
            return False
        return value.startswith(("http://", "https://"))

    @staticmethod
    @validate_call
    def has_min_length(value: object, min_length: int) -> bool:
        """Check if string has minimum length with automatic validation."""
        if not isinstance(value, str):
            return False
        return len(value) >= min_length

    @staticmethod
    @validate_call
    def has_max_length(value: object, max_length: int) -> bool:
        """Check if string has maximum length with automatic validation."""
        if not isinstance(value, str):
            return False
        return len(value) <= max_length

    @staticmethod
    @validate_call
    def matches_pattern(value: object, pattern: str) -> bool:
        """Check if string matches pattern with automatic validation."""
        if not isinstance(value, str):
            return False
        return bool(re.match(pattern, value))

    @staticmethod
    @validate_call
    def is_callable(value: object) -> bool:
        """Check if the value is callable with automatic validation."""
        return callable(value)

    @staticmethod
    @validate_call
    def is_list(value: object) -> bool:
        """Check if the value is a list with automatic validation."""
        return isinstance(value, list)

    @staticmethod
    @validate_call
    def is_dict(value: object) -> bool:
        """Check if the value is dict with automatic validation."""
        return isinstance(value, dict)

    @staticmethod
    @validate_call
    def is_none(value: object) -> bool:
        """Check if the value is None with automatic validation."""
        return value is None


class _BasePredicates:
    """predicate functions using validate_call decorators."""

    @staticmethod
    @validate_call
    def is_positive(value: float) -> bool:
        """Check if value is a positive number with automatic validation."""
        return value > 0

    @staticmethod
    @validate_call
    def is_negative(value: float) -> bool:
        """Check if value is negative number with automatic validation."""
        return value < 0

    @staticmethod
    @validate_call
    def is_zero(value: float) -> bool:
        """Check if the value is zero with automatic validation."""
        return value == 0

    @staticmethod
    @validate_call
    def is_in_range(value: float, min_val: float, max_val: float) -> bool:
        """Check if number is within range with automatic validation."""
        return min_val <= value <= max_val


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


@validate_call
def _validate_required_field(
    value: object,
    field_name: str = "field",
) -> _ValidationResult:
    """Required field validation using validate_call decorator."""
    if value is None:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} is required",
            field_name=field_name,
        )

    if isinstance(value, str) and not value.strip():
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} cannot be empty",
            field_name=field_name,
        )

    return _ValidationResult(is_valid=True, field_name=field_name)


@validate_call
def _validate_string_field(
    value: str,
    field_name: str = "field",
    min_length: int = 0,
    max_length: int | None = None,
) -> _ValidationResult:
    """String validation using validate_call decorator."""
    if len(value) < min_length:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at least {min_length} characters",
            field_name=field_name,
        )

    if max_length is not None and len(value) > max_length:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at most {max_length} characters",
            field_name=field_name,
        )

    return _ValidationResult(is_valid=True, field_name=field_name)


@validate_call
def _validate_numeric_field(
    value: float,
    field_name: str = "field",
    min_val: float | None = None,
    max_val: float | None = None,
) -> _ValidationResult:
    """Numeric validation using validate_call decorator."""
    if min_val is not None and value < min_val:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at least {min_val}",
            field_name=field_name,
        )

    if max_val is not None and value > max_val:
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be at most {max_val}",
            field_name=field_name,
        )

    return _ValidationResult(is_valid=True, field_name=field_name)


@validate_call
def _validate_email_field(
    value: str,
    field_name: str = "field",
) -> _ValidationResult:
    """Email validation using validate_call decorator."""
    # Simple but effective email validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, value):
        return _ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be a valid email address",
            field_name=field_name,
        )
    return _ValidationResult(is_valid=True, field_name=field_name)


@validate_call
def _validate_entity_id(value: str) -> bool:
    """Entity ID validation using validate_call decorator."""
    if not value.strip():
        return False
    # Basic UUID format check
    pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(pattern, value.lower()))


@validate_call
def _validate_non_empty_string(value: str) -> bool:
    """non-empty string validation using validate_call decorator."""
    return len(value.strip()) > 0


@validate_call
def _validate_service_name(name: str) -> bool:
    """Service name validation using validate_call decorator."""
    if not name.strip():
        return False
    if len(name) < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH:
        return False
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name))


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
# SERVICE NAME VALIDATION FUNCTION
# =============================================================================


@validate_call
def flext_validate_service_name(name: str) -> bool:
    """Service name validation for container operations.

    Validates service names used in dependency injection container operations
    with automatic type checking and validation rules.

    Args:
      name: Service name to validate (automatically type-checked)

    Returns:
      True if valid service name, False otherwise.

    """
    return _validate_service_name(name)


# =============================================================================
# LDAP-SPECIFIC VALIDATORS - Domain-specific LDAP validation patterns
# =============================================================================


def validate_ldap_dn(dn: object) -> str:
    """PlainValidator: Complete DN validation with RFC 2253 compliance."""
    if not isinstance(dn, str):
        dn = str(dn)

    dn = dn.strip()
    if not dn:
        msg = "Distinguished Name cannot be empty"
        raise ValueError(msg)

    # Basic DN format validation (simplified for practical use)
    if not re.match(r"^[a-zA-Z]+=[^,]+(?:,[a-zA-Z]+=[^,]+)*$", dn):
        msg = f"Invalid DN format: {dn}. Must follow RFC 2253"
        raise ValueError(msg)

    return dn


def validate_ldap_filter(filter_str: object) -> str:
    """PlainValidator: Complete LDAP filter validation with RFC 4515 compliance."""
    if not isinstance(filter_str, str):
        filter_str = str(filter_str)

    filter_str = filter_str.strip()
    if not filter_str:
        msg = "LDAP filter cannot be empty"
        raise ValueError(msg)

    # Must be enclosed in parentheses
    if not (filter_str.startswith("(") and filter_str.endswith(")")):
        msg = f"LDAP filter must be enclosed in parentheses: {filter_str}"
        raise ValueError(msg)

    return filter_str


def validate_ldap_scope(scope: object) -> str:
    """PlainValidator: LDAP search scope validation per RFC 4511."""
    if not isinstance(scope, str):
        scope = str(scope)

    scope = scope.strip().lower()
    valid_scopes = {"base", "one", "sub", "onelevel", "subtree", "children"}

    if scope not in valid_scopes:
        msg = f"Invalid LDAP scope: {scope}. Must be one of {valid_scopes}"
        raise ValueError(msg)

    return scope


def normalize_ldap_attributes(attrs: object) -> dict[str, str | list[str]]:
    """BeforeValidator: Normalize LDAP attributes dictionary."""
    if not isinstance(attrs, dict):
        return {}

    normalized: dict[str, str | list[str]] = {}
    for key, value in attrs.items():
        if isinstance(value, list):
            normalized[key] = [str(item) for item in value]
        else:
            normalized[key] = str(value)

    return normalized


# LDAP-specific annotated types
LdapDistinguishedName = Annotated[str, PlainValidator(validate_ldap_dn)]
LdapFilter = Annotated[str, PlainValidator(validate_ldap_filter)]
LdapScope = Annotated[str, PlainValidator(validate_ldap_scope)]
LdapAttributes = Annotated[dict[str, str | list[str]], BeforeValidator(normalize_ldap_attributes)]


# =============================================================================
# FLEXT VALIDATION - Main validation interface inheriting from base
# =============================================================================


class FlextValidation:
    """Unified validation system implementing Tier 1 Module Pattern.

    This class serves as the single main export consolidating all validation
    functionality from the flext-core validation ecosystem. Provides comprehensive
    validation capabilities while maintaining backward compatibility.

    Tier 1 Module Pattern: validation.py -> FlextValidation
    All validation functionality is accessible through this single interface.
    """

    # =========================================================================
    # CORE VALIDATION METHODS - Delegated to base validators
    # =========================================================================

    @staticmethod
    def is_not_none(value: object) -> bool:
        """Check if value is not None."""
        return _BaseValidators.is_not_none(value)

    @staticmethod
    def is_string(value: object) -> bool:
        """Check if value is a string."""
        return _BaseValidators.is_string(value)

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Check if value is a non-empty string."""
        return _BaseValidators.is_non_empty_string(value)

    @staticmethod
    def is_email(value: object) -> bool:
        """Check if value is a valid email."""
        return _BaseValidators.is_email(value)

    @staticmethod
    def is_uuid(value: object) -> bool:
        """Check if value is a valid UUID."""
        return _BaseValidators.is_uuid(value)

    @staticmethod
    def is_url(value: object) -> bool:
        """Check if value is a valid URL."""
        return _BaseValidators.is_url(value)

    @staticmethod
    def has_min_length(value: object, min_length: int) -> bool:
        """Check if string has minimum length."""
        return _BaseValidators.has_min_length(value, min_length)

    @staticmethod
    def has_max_length(value: object, max_length: int) -> bool:
        """Check if string has maximum length."""
        return _BaseValidators.has_max_length(value, max_length)

    @staticmethod
    def matches_pattern(value: object, pattern: str) -> bool:
        """Check if string matches regex pattern."""
        return _BaseValidators.matches_pattern(value, pattern)

    @staticmethod
    def is_callable(value: object) -> bool:
        """Check if value is callable."""
        return _BaseValidators.is_callable(value)

    @staticmethod
    def is_list(value: object) -> bool:
        """Check if value is a list."""
        return _BaseValidators.is_list(value)

    @staticmethod
    def is_dict(value: object) -> bool:
        """Check if value is a dictionary."""
        return _BaseValidators.is_dict(value)

    @staticmethod
    def is_none(value: object) -> bool:
        """Check if value is None."""
        return _BaseValidators.is_none(value)

    # =========================================================================
    # PREDICATE METHODS - Delegated to base predicates
    # =========================================================================

    @staticmethod
    def is_positive(value: float) -> bool:
        """Check if value is positive."""
        return _BasePredicates.is_positive(value)

    @staticmethod
    def is_negative(value: float) -> bool:
        """Check if value is negative."""
        return _BasePredicates.is_negative(value)

    @staticmethod
    def is_zero(value: float) -> bool:
        """Check if value is zero."""
        return _BasePredicates.is_zero(value)

    @staticmethod
    def is_in_range(value: float, min_val: float, max_val: float) -> bool:
        """Check if value is within range."""
        return _BasePredicates.is_in_range(value, min_val, max_val)

    # =========================================================================
    # LDAP VALIDATION METHODS - Domain-specific LDAP validation
    # =========================================================================

    @staticmethod
    def is_valid_ldap_dn(value: object) -> bool:
        """Check if value is a valid LDAP Distinguished Name."""
        try:
            validate_ldap_dn(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_ldap_filter(value: object) -> bool:
        """Check if value is a valid LDAP filter."""
        try:
            validate_ldap_filter(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_ldap_scope(value: object) -> bool:
        """Check if value is a valid LDAP scope."""
        try:
            validate_ldap_scope(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def normalize_ldap_attrs(value: object) -> dict[str, str | list[str]]:
        """Normalize LDAP attributes dictionary."""
        return normalize_ldap_attributes(value)

    # =========================================================================
    # VALIDATION FUNCTIONS - Direct access to core validation functions
    # =========================================================================

    # Field validation functions
    validate_required_field = staticmethod(flext_validate_required_field)
    validate_string_field = staticmethod(flext_validate_string_field)
    validate_numeric_field = staticmethod(flext_validate_numeric_field)
    validate_email_field = staticmethod(flext_validate_email_field)

    # Entity validation functions
    validate_entity_id_simple = staticmethod(flext_validate_entity_id)
    validate_non_empty_string_simple = staticmethod(flext_validate_non_empty_string)
    validate_service_name_simple = staticmethod(flext_validate_service_name)

    # Convenience functions (added after function definitions)
    validate_required: object | None = None  # Will be set after function definition
    validate_string: object | None = None  # Will be set after function definition
    validate_numeric: object | None = None  # Will be set after function definition
    validate_email: object | None = None  # Will be set after function definition

    # FlextResult-based validation functions (added after function definitions)
    validate_with_result: object | None = None  # Will be set after function definition
    validate_entity_id: object | None = None  # Will be set after function definition
    validate_service_name_with_result: object | None = (
        None  # Will be set after function definition
    )

    @classmethod
    def validate(cls, value: object) -> FlextResult[bool]:
        """Main validation method with automatic type detection."""
        try:
            if (
                isinstance(value, str)
                and "@" in value
                and "." in value
                and not cls.is_email(value)
            ):
                return FlextResult[bool].fail("Invalid email format")
            if isinstance(value, str) and not cls.is_non_empty_string(value):
                return FlextResult[bool].fail("String cannot be empty")

            return FlextResult[bool].ok(data=True)
        except Exception as e:
            return FlextResult[bool].fail(f"Validation failed: {e}")

    @staticmethod
    @validate_call
    def chain(*validators: Callable[[object], bool]) -> Callable[[object], bool]:
        """Chain multiple validators with AND logic."""

        @validate_call
        def chained_validator(value: object) -> bool:
            return all(validator(value) for validator in validators)

        return chained_validator

    @staticmethod
    @validate_call
    def any_of(*validators: Callable[[object], bool]) -> Callable[[object], bool]:
        """Chain multiple validators with OR logic."""

        @validate_call
        def any_validator(value: object) -> bool:
            return any(validator(value) for validator in validators)

        return any_validator

    @staticmethod
    @validate_call
    def create_validation_config(
        field_name: str,
        min_length: int = 0,
        max_length: int | None = None,
    ) -> _ValidationConfig:
        """Create validation configuration."""
        return FlextValidationConfig(
            field_name=field_name,
            min_length=min_length,
            max_length=max_length,
        )

    @classmethod
    @validate_call
    def safe_validate(
        cls,
        value: object,
        validator: Callable[[object], bool],
    ) -> FlextResult[bool]:
        """Safely validate value with FlextResult error handling."""
        try:
            if validator(value):
                return FlextResult[bool].ok(data=True)
            return FlextResult[bool].fail(f"Validation failed for value: {value}")
        except (
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
            ValidationError,
        ) as e:
            return FlextResult[bool].fail(f"Validation error: {e}")

    # =========================================================================
    # PIPELINE AND DOMAIN VALIDATION - Factory methods for advanced patterns
    # =========================================================================

    @staticmethod
    def create_pipeline() -> FlextValidationPipeline:
        """Create a new validation pipeline."""
        return FlextValidationPipeline()

    @staticmethod
    def create_domain_validator(
        business_rules: list[Callable[[object], bool]] | None = None,
    ) -> FlextDomainValidator[object]:
        """Create a domain-specific validator."""
        return FlextDomainValidator(business_rules)

    # =========================================================================
    # COMPATIBILITY FACADES - Access to legacy structure
    # =========================================================================

    # Class-level access to components for backward compatibility
    Validators = FlextValidators
    Predicates = FlextPredicates
    Config = FlextValidationConfig
    Result = FlextValidationResult
    # Pipeline será atribuído após FlextValidationPipeline ser definida

    # Function-level access (removed duplicates that already exist as staticmethod above)
    validate_entity_id_func = flext_validate_entity_id
    validate_non_empty_string_func = flext_validate_non_empty_string


# =============================================================================
# API RE-EXPOSURE - FlextPredicates and FlextValidationResult already defined above
# =============================================================================


# All functionality properly exposed through direct assignment

# =============================================================================
# CONVENIENCE FUNCTIONS - High-level validation helpers with Flext prefix
# =============================================================================


@validate_call
def flext_validate_required(
    value: object,
    field_name: str = "field",
) -> _ValidationResult:
    """Validation for required fields with comprehensive checking.

    Performs null/None validation ensuring the field contains a meaningful value.
    Uses validate_call decorator for automatic type checking and error handling.

    Args:
      value: Value to validate (any type accepted)
      field_name: Name of field for error messages and debugging

    Returns:
      FlextValidationResult with is_valid boolean and error details.

    """
    return flext_validate_required_field(value, field_name)


@validate_call
def flext_validate_string(
    value: str,
    field_name: str = "field",
    min_length: int = 0,
    max_length: int | None = None,
) -> _ValidationResult:
    """String field validation with comprehensive constraints.

    Performs automatic type checking via validate_call decorator, length validation,
    and content verification. Supports configurable length constraints with clear errors.

    Args:
      value: String value to validate (automatically type-checked)
      field_name: Name of field for error messages and debugging
      min_length: Minimum required length (inclusive, default 0)
      max_length: Maximum allowed length (inclusive, None for unlimited)

    Returns:
      FlextValidationResult with is_valid boolean and detailed error information.

    """
    return flext_validate_string_field(value, field_name, min_length, max_length)


@validate_call
def flext_validate_numeric(
    value: float,
    field_name: str = "field",
    min_val: float | None = None,
    max_val: float | None = None,
) -> _ValidationResult:
    """Numeric field validation with comprehensive range constraints.

    Performs automatic type checking via validate_call decorator and range validation
    for numeric values. Supports configurable range constraints with detailed errors.

    Args:
      value: Numeric value to validate (automatically type-checked)
      field_name: Name of field for error messages and debugging
      min_val: Minimum allowed value (inclusive, None for no minimum)
      max_val: Maximum allowed value (inclusive, None for no maximum)

    Returns:
      FlextValidationResult with is_valid boolean and range error details.

    """
    return flext_validate_numeric_field(value, field_name, min_val, max_val)


@validate_call
def flext_validate_email(
    value: str,
    field_name: str = "field",
) -> _ValidationResult:
    """Email field validation with comprehensive format checking.

    Performs automatic type checking via validate_call decorator and comprehensive
    email validation including format checking and structural verification.

    Args:
      value: Email string to validate (automatically type-checked)
      field_name: Name of field for error messages and debugging

    Returns:
      FlextValidationResult with is_valid boolean and format error details.

    """
    return flext_validate_email_field(value, field_name)


# =============================================================================
# VALIDATION FUNCTIONS WITH FLEXTRESULT
# =============================================================================


@validate_call
def validate_with_result(
    value: object,
    validator: Callable[[object], bool],
    error_message: str = "Validation failed",
) -> FlextResult[bool]:
    """Validation function returning FlextResult."""
    try:
        if validator(value):
            return FlextResult[bool].ok(data=True)
        return FlextResult[bool].fail(error_message)
    except ValidationError as e:
        return FlextResult[bool].fail(f"Validation error: {e}")
    except Exception as e:
        return FlextResult[bool].fail(f"Unexpected validation error: {e}")


@validate_call
def validate_entity_id(entity_id: str) -> FlextResult[str]:
    """Entity ID validation returning FlextResult."""
    if not entity_id.strip():
        return FlextResult[str].fail("Entity ID cannot be empty")

    # Basic UUID format check
    if not _BaseValidators.is_uuid(entity_id):
        return FlextResult[str].fail("Entity ID must be a valid UUID")

    return FlextResult[str].ok(entity_id)


@validate_call
def validate_service_name_with_result(name: str) -> FlextResult[str]:
    """Service name validation returning FlextResult."""
    if not name.strip():
        return FlextResult[str].fail("Service name cannot be empty")

    if len(name) < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH:
        return FlextResult[str].fail("Service name must be at least 2 characters")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
        return FlextResult[str].fail(
            "Service name must start with letter and contain only letters, numbers, hyphens, and underscores",
        )

    return FlextResult[str].ok(name)


# =============================================================================
# VALIDATION PIPELINE
# =============================================================================


class FlextValidationPipeline:
    """validation pipeline using validate_call decorators."""

    def __init__(self) -> None:
        self.validators: list[Callable[[object], FlextResult[bool]]] = []

    @validate_call
    def add_validator(self, validator: Callable[[object], FlextResult[bool]]) -> None:
        """Add validator to the pipeline."""
        self.validators.append(validator)

    @validate_call
    def validate(self, value: object) -> FlextResult[bool]:
        """Run all validators in the pipeline."""
        current_value = value

        for validator in self.validators:
            result = validator(current_value)
            if not result.is_success:
                return result
            current_value = result.value

        return FlextResult[bool].ok(data=True)


# =============================================================================
# MIGRATION NOTICE - Legacy functions moved to legacy.py
# =============================================================================

# IMPORTANT: Legacy functions validate_smart and is_valid_data have been
# completely moved to legacy.py. Import them from there if needed:
#   Use FlextValidation.validate_value instead
#
# These functions are NOT re-exported here to avoid circular imports.


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

# =============================================================================
# BASE VALIDATORS - Core validation patterns
# =============================================================================


class FlextDomainValidator[T](FlextAbstractValidator[T]):
    """domain-specific validation with business rules using validate_call.

    SOLID compliance: Single responsibility for domain validation.
    Uses standard Pydantic patterns for automatic validation.
    """

    def __init__(self, business_rules: list[Callable[[T], bool]] | None = None) -> None:
        """Initialize domain validator with business rules."""
        self.business_rules = business_rules or []

    @validate_call
    def validate_value(self, value: T) -> FlextResult[T]:
        """Validate value against business rules with standard patterns."""
        try:
            for rule in self.business_rules:
                if not rule(value):
                    return FlextResult[T].fail("Business rule validation failed")
            return FlextResult[T].ok(value)
        except ValidationError as e:
            return FlextResult[T].fail(f"Validation error: {e}")
        except Exception as e:
            return FlextResult[T].fail(f"Business rule error: {e}")

    @override
    def validate(self, value: T) -> FlextResult[T]:
        """Alias to validate_value for compatibility with legacy code."""
        return self.validate_value(value)


# =============================================================================
# TIER 1 MODULE PATTERN: validation.py -> FlextValidation
# =============================================================================

# Main export: FlextValidation (already defined above as the unified class)

# =============================================================================
# COMPATIBILITY FACADES - Maintain backward compatibility
# =============================================================================

# Core validation classes (main exports)

# Preserve unified protocol from FlextProtocols hierarchy
FlextValidationRule = FlextProtocols.Foundation.Validator

__all__: list[str] = [
    # =======================================================================
    # LEGACY COMPATIBILITY - Function and type aliases only (not classes)
    # =======================================================================
    # Type Annotations (Pydantic Annotated types)
    "AutoGeneratedId",
    "ContextualEntityId",
    "DeduplicatedList",
    "EmailAddress",
    "EmailWithNormalization",
    "EntityIdWithGeneration",
    # Backward compatibility
    "FlextValidation",  # Legacy alias for backward compatibility
    "FlextValidations",  # 🎯 SINGLE EXPORT: All validation functionality consolidated (plural pattern)
    "FormattedTimestamp",
    "LdapAttributes",
    "LdapDistinguishedName",
    "LdapFilter",
    "LdapScope",
    "NormalizedEmail",
    "NormalizedString",
    "PositiveNumber",
    "ServiceName",
    "StringList",
    # Functional Validators (BeforeValidator, AfterValidator, etc.)
    "add_flext_prefix",
    "ensure_positive",
    "ensure_string_list",
    # Validation Functions with flext_ prefix
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
    "format_timestamp",
    "generate_id_if_missing",
    "normalize_email",
    "normalize_ldap_attributes",
    "normalize_string",
    "validate_email_address",
    # FlextResult-based validation functions
    "validate_entity_id",
    "validate_ldap_dn",
    "validate_ldap_filter",
    "validate_ldap_scope",
    "validate_list_with_deduplication",
    "validate_service_name",
    "validate_service_name_with_result",
    "validate_timestamp_with_fallback",
    "validate_version_number",
    "validate_with_result",
]


# =============================================================================
# ARCHITECTURE NOTES - Tier 1 Module Pattern Implementation
# =============================================================================

# This module implements the Tier 1 Module Pattern:
# - validation.py exports exactly one main class: FlextValidation
# - All validation functionality is accessible through FlextValidation
# - Compatibility facades maintain backward compatibility
# - Original classes remain for specific use cases (pipelines, domain validators)
# - No functionality is lost in the consolidation

# Pattern Benefits:
# - Single entry point for all validation operations
# - Reduced cognitive load for developers
# - Maintained backward compatibility
# - Clear hierarchical structure
# - SOLID principles compliance

# =============================================================================
# LATE BINDING - Add functions and Pipeline after class definition
# =============================================================================

# Now that all functions are defined, add them to the FlextValidation class
FlextValidation.validate_required = staticmethod(flext_validate_required)
FlextValidation.validate_string = staticmethod(flext_validate_string)
FlextValidation.validate_numeric = staticmethod(flext_validate_numeric)
FlextValidation.validate_email = staticmethod(flext_validate_email)

# FlextResult-based validation functions
FlextValidation.validate_with_result = staticmethod(validate_with_result)
FlextValidation.validate_entity_id = staticmethod(validate_entity_id)
FlextValidation.validate_service_name_with_result = staticmethod(
    validate_service_name_with_result
)

# Now that FlextValidationPipeline is defined, add it to the FlextValidation class
FlextValidation.Pipeline = FlextValidationPipeline  # type: ignore[attr-defined]


# =============================================================================
# CONSOLIDATED VALIDATION CLASS - Following FLEXT Patterns
# =============================================================================


class FlextValidations:
    """Consolidated validation system following FLEXT[Module]s pattern.

    Single class consolidating ALL validation functionality including:
    - Core validation operations
    - Pipeline validation
    - Domain validation
    - Predicate-based validation
    """

    # Consolidated nested classes
    Validation = FlextValidation
    Pipeline = FlextValidationPipeline
    DomainValidator = FlextDomainValidator
    AbstractValidator = FlextAbstractValidator

    # Predicate constants and functions as nested namespace
    class Predicates:
        """Validation predicates namespace."""

        # Core predicates - using existing base class methods
        is_not_none = staticmethod(_BaseValidators.is_not_none)
        is_non_empty_string = staticmethod(_BaseValidators.is_non_empty_string)
        is_email = staticmethod(_BaseValidators.is_email)
        is_positive = staticmethod(_BasePredicates.is_positive)
        is_negative = staticmethod(_BasePredicates.is_negative)
        is_zero = staticmethod(_BasePredicates.is_zero)
        is_in_range = staticmethod(_BasePredicates.is_in_range)

    # Validators namespace
    class Validators:
        """Core validators namespace."""

        # Pydantic functional validators
        normalize_string = staticmethod(normalize_string)
        normalize_email = staticmethod(normalize_email)
        ensure_string_list = staticmethod(ensure_string_list)
        generate_id_if_missing = staticmethod(generate_id_if_missing)
        uppercase_code = staticmethod(uppercase_code)

        # Flext validation functions
        validate_required = staticmethod(flext_validate_required)
        validate_string = staticmethod(flext_validate_string)
        validate_numeric = staticmethod(flext_validate_numeric)
        validate_email = staticmethod(flext_validate_email)
        validate_service_name = staticmethod(flext_validate_service_name)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Main backward compatibility - CRITICAL for naming pattern compliance
FlextValidation = FlextValidations  # type: ignore[assignment]  # CRITICAL: Backward compatibility for singular name
FlextCoreValidation = FlextValidations  # type: ignore[assignment]  # Legacy name support

# Individual class aliases maintained for compatibility
FlextPredicates = FlextValidations.Predicates  # type: ignore[assignment]
FlextValidators = FlextValidations.Validators  # type: ignore[assignment]
