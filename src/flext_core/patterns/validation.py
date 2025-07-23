"""FLEXT Core Validation System - Unified Validation Pattern.

Enterprise-grade validation system with standardized rules,
field validation, and comprehensive error reporting.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import TypeVar

from flext_core.patterns.typedefs import FlextFieldPath
from flext_core.patterns.typedefs import FlextRuleName
from flext_core.patterns.typedefs import FlextValidatorId

# =============================================================================
# TYPE VARIABLES - Generic validation typing
# =============================================================================

TData = TypeVar("TData")
TValue = TypeVar("TValue")

# =============================================================================
# VALIDATION RESULT - Comprehensive validation results
# =============================================================================


class FlextValidationResult:
    """Comprehensive validation result with detailed error information."""

    def __init__(
        self,
        is_valid: bool,  # noqa: FBT001
        errors: list[str] | None = None,
        field_errors: dict[str, list[str]] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Initialize validation result.

        Args:
            is_valid: Whether validation passed
            errors: List of general validation errors
            field_errors: Field-specific validation errors
            warnings: Non-critical validation warnings

        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.field_errors = field_errors or {}
        self.warnings = warnings or []

    @classmethod
    def success(
        cls,
        warnings: list[str] | None = None,
    ) -> FlextValidationResult:
        """Create successful validation result."""
        return cls(is_valid=True, warnings=warnings)

    @classmethod
    def failure(
        cls,
        errors: list[str] | None = None,
        field_errors: dict[str, list[str]] | None = None,
    ) -> FlextValidationResult:
        """Create failed validation result."""
        return cls(is_valid=False, errors=errors, field_errors=field_errors)

    def add_error(self, error: str) -> None:
        """Add general validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_field_error(self, field: str, error: str) -> None:
        """Add field-specific validation error."""
        if field not in self.field_errors:
            self.field_errors[field] = []
        self.field_errors[field].append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)

    def get_all_errors(self) -> list[str]:
        """Get all errors (general + field-specific)."""
        all_errors = list(self.errors)
        for field, field_errs in self.field_errors.items():
            all_errors.extend(f"{field}: {error}" for error in field_errs)
        return all_errors

    def has_errors(self) -> bool:
        """Check if any errors exist."""
        return len(self.errors) > 0 or len(self.field_errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings exist."""
        return len(self.warnings) > 0

    def merge(self, other: FlextValidationResult) -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False

        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

        for field, errors in other.field_errors.items():
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].extend(errors)


# =============================================================================
# VALIDATION RULE - Individual validation rules
# =============================================================================


class FlextValidationRule[TValue](ABC):
    """Base class for validation rules."""

    def __init__(
        self,
        rule_name: FlextRuleName,
        error_message: str | None = None,
    ) -> None:
        """Initialize validation rule.

        Args:
            rule_name: Name of the validation rule
            error_message: Custom error message for failures

        """
        self.rule_name = rule_name
        self.error_message = error_message or self._get_default_error_message()

    @abstractmethod
    def validate(self, value: TValue) -> bool:
        """Validate a value against this rule.

        Args:
            value: Value to validate

        Returns:
            True if validation passes

        """

    @abstractmethod
    def _get_default_error_message(self) -> str:
        """Get default error message for this rule."""

    def get_error_message(self, value: TValue) -> str:  # noqa: ARG002
        """Get error message for failed validation."""
        return self.error_message


# =============================================================================
# FIELD VALIDATOR - Field-specific validation
# =============================================================================


class FlextFieldValidator:
    """Validator for individual fields with multiple rules."""

    def __init__(
        self,
        field_path: FlextFieldPath,
        rules: list[FlextValidationRule[Any]] | None = None,
        *,
        required: bool = False,
    ) -> None:
        """Initialize field validator.

        Args:
            field_path: Path to the field being validated
            rules: List of validation rules
            required: Whether field is required

        """
        self.field_path = field_path
        self.rules = rules or []
        self.required = required

    def add_rule(self, rule: FlextValidationRule[Any]) -> None:
        """Add validation rule."""
        self.rules.append(rule)

    def validate(self, value: object) -> FlextValidationResult:
        """Validate field value against all rules.

        Args:
            value: Value to validate

        Returns:
            FlextValidationResult with validation results

        """
        result = FlextValidationResult.success()

        # Check required field
        if self.required and (value is None or value == ""):
            result.add_field_error(
                self.field_path,
                "Field is required",
            )
            return result

        # Skip validation if value is None/empty and not required
        if not self.required and (value is None or value == ""):
            return result

        # Apply all rules
        for rule in self.rules:
            try:
                if not rule.validate(value):
                    result.add_field_error(
                        self.field_path,
                        rule.get_error_message(value),
                    )
            except Exception as e:  # noqa: BLE001
                result.add_field_error(
                    self.field_path,
                    f"Validation rule '{rule.rule_name}' failed: {e!s}",
                )

        return result


# =============================================================================
# MAIN VALIDATOR - Comprehensive data validation
# =============================================================================


class FlextValidator[TData](ABC):
    """Base class for comprehensive data validators."""

    def __init__(
        self,
        validator_id: FlextValidatorId | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            validator_id: Optional validator ID

        """
        self.validator_id = validator_id or FlextValidatorId(
            f"{self.__class__.__name__}_{id(self)}",
        )
        self._field_validators: dict[str, FlextFieldValidator] = {}

    def add_field_validator(
        self,
        field_path: str,
        validator: FlextFieldValidator,
    ) -> None:
        """Add field validator.

        Args:
            field_path: Path to field
            validator: Field validator

        """
        self._field_validators[field_path] = validator

    def validate_fields(self, data: TData) -> FlextValidationResult:
        """Validate all configured fields.

        Args:
            data: Data to validate

        Returns:
            FlextValidationResult with field validation results

        """
        result = FlextValidationResult.success()

        for field_path, field_validator in self._field_validators.items():
            # Extract field value from data
            field_value = self._extract_field_value(data, field_path)

            # Validate field
            field_result = field_validator.validate(field_value)
            result.merge(field_result)

        return result

    @abstractmethod
    def validate_business_rules(self, data: TData) -> FlextValidationResult:
        """Validate business-specific rules.

        Args:
            data: Data to validate

        Returns:
            FlextValidationResult with business rule validation

        """

    def validate(self, data: TData) -> FlextValidationResult:
        """Comprehensive validation of data.

        Args:
            data: Data to validate

        Returns:
            FlextValidationResult with complete validation results

        """
        result = FlextValidationResult.success()

        # Validate fields
        field_result = self.validate_fields(data)
        result.merge(field_result)

        # Validate business rules (only if field validation passes)
        if field_result.is_valid:
            business_result = self.validate_business_rules(data)
            result.merge(business_result)

        return result

    def _extract_field_value(self, data: TData, field_path: str) -> object:
        """Extract field value from data using field path.

        Args:
            data: Data object
            field_path: Path to field (supports dot notation)

        Returns:
            Field value or None if not found

        """
        try:
            value = data
            for part in field_path.split("."):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
        except (AttributeError, KeyError, TypeError):
            return None
        return value


# =============================================================================
# COMMON VALIDATION RULES - Reusable validation rules
# =============================================================================


class NotEmptyRule(FlextValidationRule[str]):
    """Rule to check string is not empty."""

    def __init__(self) -> None:
        """Initialize NotEmptyRule."""
        super().__init__(FlextRuleName("not_empty"))

    def validate(self, value: str) -> bool:
        """Validate that string is not empty."""
        return value is not None and len(value.strip()) > 0

    def _get_default_error_message(self) -> str:
        """Get default error message for empty values."""
        return "Value cannot be empty"


class MinLengthRule(FlextValidationRule[str]):
    """Rule to check minimum string length."""

    def __init__(self, min_length: int) -> None:
        """Initialize MinLengthRule.

        Args:
            min_length: Minimum required length

        """
        self.min_length = min_length
        super().__init__(FlextRuleName("min_length"))

    def validate(self, value: str) -> bool:
        """Validate string meets minimum length."""
        return value is not None and len(value) >= self.min_length

    def _get_default_error_message(self) -> str:
        """Get default error message for length violations."""
        return f"Value must be at least {self.min_length} characters"


class MaxLengthRule(FlextValidationRule[str]):
    """Rule to check maximum string length."""

    def __init__(self, max_length: int) -> None:
        """Initialize MaxLengthRule.

        Args:
            max_length: Maximum allowed length

        """
        self.max_length = max_length
        super().__init__(FlextRuleName("max_length"))

    def validate(self, value: str) -> bool:
        """Validate string does not exceed maximum length."""
        return value is None or len(value) <= self.max_length

    def _get_default_error_message(self) -> str:
        """Get default error message for length violations."""
        return f"Value must be no more than {self.max_length} characters"


class RangeRule(FlextValidationRule[int | float]):
    """Rule to check numeric value is within range."""

    def __init__(self, min_value: float, max_value: float) -> None:
        """Initialize RangeRule with minimum and maximum value limits."""
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(FlextRuleName("range"))

    def validate(self, value: float) -> bool:
        """Validate that numeric value is within the specified range."""
        return value is not None and self.min_value <= value <= self.max_value

    def _get_default_error_message(self) -> str:
        return f"Value must be between {self.min_value} and {self.max_value}"


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "FlextFieldValidator",
    "FlextValidationResult",
    "FlextValidationRule",
    "FlextValidator",
    "MaxLengthRule",
    "MinLengthRule",
    "NotEmptyRule",
    "RangeRule",
    "TData",
    "TValue",
]
