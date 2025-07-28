"""FLEXT Validation Base - Consolidated validation using maximum Pydantic.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Single source of truth for validation - uses Pydantic extensively.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core.constants import FlextConstants

if TYPE_CHECKING:
    from collections.abc import Callable


# Regex patterns from constants - sem underscore conforme diretrizes
EMAIL_PATTERN = FlextConstants.EMAIL_PATTERN
UUID_PATTERN = FlextConstants.UUID_PATTERN
URL_PATTERN = FlextConstants.URL_PATTERN


# =============================================================================
# PYDANTIC-BASED VALIDATION MODELS
# =============================================================================


class _ValidationConfig(BaseModel):
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


class _ValidationResult(BaseModel):
    """Pydantic model for validation results."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    error_message: str | None = None
    field_name: str | None = None

    @classmethod
    def success(cls) -> _ValidationResult:
        """Create successful validation result."""
        return cls(is_valid=True)

    @classmethod
    def failure(cls, message: str, field_name: str | None = None) -> _ValidationResult:
        """Create failed validation result."""
        return cls(is_valid=False, error_message=message, field_name=field_name)


# =============================================================================
# CONSOLIDATED VALIDATORS - Single source of truth
# =============================================================================


class _BaseValidators:
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
    def is_none(value: object) -> bool:
        """Check if value is None."""
        return value is None

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
# FUNCTIONAL PREDICATES - Using Pydantic for configuration
# =============================================================================


class _BasePredicates:
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
        return _BasePredicates.matches_regex(EMAIL_PATTERN)

    @staticmethod
    def is_uuid() -> Callable[[object], bool]:
        """Predicate that checks if value is valid UUID."""
        return _BasePredicates.matches_regex(UUID_PATTERN)

    @staticmethod
    def is_url() -> Callable[[object], bool]:
        """Predicate that checks if value is valid URL."""
        return _BasePredicates.matches_regex(URL_PATTERN)

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
# PYDANTIC-ENHANCED FIELD VALIDATORS
# =============================================================================


def _validate_required_field(
    value: object,
    field_name: str,
) -> _ValidationResult:
    """Validate required field using Pydantic result model."""
    if value is None:
        return _ValidationResult.failure(
            f"Field '{field_name}' is required but was None",
            field_name,
        )

    if isinstance(value, str) and not value.strip():
        return _ValidationResult.failure(
            f"Field '{field_name}' is required but was empty",
            field_name,
        )

    return _ValidationResult.success()


def _validate_string_field(
    value: object,
    field_name: str,
    min_length: int = 0,
    max_length: int | None = None,
) -> _ValidationResult:
    """Validate string field with Pydantic configuration."""
    try:
        # Use Pydantic to validate configuration
        config = _ValidationConfig(
            field_name=field_name,
            min_length=min_length,
            max_length=max_length,
        )
    except (TypeError, ValueError, AttributeError) as e:
        return _ValidationResult.failure(f"Invalid validation config: {e}")

    if not isinstance(value, str):
        return _ValidationResult.failure(
            f"Field '{config.field_name}' must be a string, got {type(value).__name__}",
            config.field_name,
        )

    if len(value) < config.min_length:
        return _ValidationResult.failure(
            f"Field '{config.field_name}' must be at least "
            f"{config.min_length} characters",
            config.field_name,
        )

    if config.max_length is not None and len(value) > config.max_length:
        return _ValidationResult.failure(
            f"Field '{config.field_name}' must be at most "
            f"{config.max_length} characters",
            config.field_name,
        )

    return _ValidationResult.success()


def _validate_numeric_field(
    value: object,
    field_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> _ValidationResult:
    """Validate numeric field with Pydantic configuration."""
    try:
        config = _ValidationConfig(
            field_name=field_name,
            min_val=min_val,
            max_val=max_val,
        )
    except (TypeError, ValueError, AttributeError) as e:
        return _ValidationResult.failure(f"Invalid validation config: {e}")

    if not isinstance(value, (int, float)):
        return _ValidationResult.failure(
            f"Field '{config.field_name}' must be a number, got {type(value).__name__}",
            config.field_name,
        )

    if config.min_val is not None and value < config.min_val:
        return _ValidationResult.failure(
            f"Field '{config.field_name}' must be at least {config.min_val}",
            config.field_name,
        )

    if config.max_val is not None and value > config.max_val:
        return _ValidationResult.failure(
            f"Field '{config.field_name}' must be at most {config.max_val}",
            config.field_name,
        )

    return _ValidationResult.success()


def _validate_email_field(
    value: object,
    field_name: str,
) -> _ValidationResult:
    """Validate email field."""
    if not isinstance(value, str):
        return _ValidationResult.failure(
            f"Field '{field_name}' must be a string",
            field_name,
        )

    if not _BaseValidators.is_email(value):
        return _ValidationResult.failure(
            f"Field '{field_name}' must be a valid email address",
            field_name,
        )

    return _ValidationResult.success()


# =============================================================================
# ENTITY VALIDATION FUNCTIONS
# =============================================================================


def _validate_entity_id(entity_id: object) -> bool:
    """Validate entity ID is valid."""
    return _BaseValidators.is_non_empty_string(entity_id)


def _validate_non_empty_string(value: object) -> bool:
    """Validate value is non-empty string."""
    return _BaseValidators.is_non_empty_string(value)


# =============================================================================
# EXPORTS - Clean API surface
# =============================================================================

__all__ = [
    "_BasePredicates",
    "_BaseValidators",
    "_ValidationConfig",
    "_ValidationResult",
    "_validate_email_field",
    "_validate_entity_id",
    "_validate_non_empty_string",
    "_validate_numeric_field",
    "_validate_required_field",
    "_validate_string_field",
]
