"""FlextValidators - Intuitive Validation Helpers.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Enterprise validation utilities that make validation simple and powerful.
Provides fluent validation chains, common validation rules, and easy
error aggregation with minimal boilerplate.
"""

from __future__ import annotations

import re
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import final

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class FlextValidator[T](ABC):
    """Abstract base class for type-safe validators."""

    @abstractmethod
    def validate(self, value: T) -> FlextResult[T]:
        """Validate value and return result."""

    def __call__(self, value: T) -> FlextResult[T]:
        """Allow validator to be called as function."""
        return self.validate(value)


class FlextValidationChain[T]:
    """Fluent validation chain for complex validation scenarios."""

    def __init__(self, value: T) -> None:
        """Initialize validation chain with value."""
        self._value = value
        self._errors: list[str] = []
        self._valid = True

    def validate_with(
        self,
        validator: FlextValidator[T],
    ) -> FlextValidationChain[T]:
        """Add validator to chain."""
        if self._valid:
            result = validator.validate(self._value)
            if not result.success:
                self._errors.append(result.error or "Validation failed")
                self._valid = False
        return self

    def validate_if(
        self,
        condition: bool,  # noqa: FBT001
        validator: FlextValidator[T],
    ) -> FlextValidationChain[T]:
        """Conditionally validate with validator."""
        if condition:
            return self.validate_with(validator)
        return self

    def custom(
        self,
        validator_func: Callable[[T], bool],
        error: str,
    ) -> FlextValidationChain[T]:
        """Add custom validation function."""
        if self._valid:
            try:
                if not validator_func(self._value):
                    self._errors.append(error)
                    self._valid = False
            except Exception as e:  # noqa: BLE001
                self._errors.append(f"Validation error: {e}")
                self._valid = False
        return self

    def result(self) -> FlextResult[T]:
        """Get final validation result."""
        if self._valid:
            return FlextResult.ok(self._value)
        return FlextResult.fail("; ".join(self._errors))

    def unwrap(self) -> T:
        """Unwrap validated value or raise exception."""
        return self.result().unwrap()


# Common Validators


@final
class NotNoneValidator[T](FlextValidator[T]):
    """Validates that value is not None."""

    def validate(self, value: T) -> FlextResult[T]:
        """Validate value is not None."""
        if value is None:
            return FlextResult.fail("Value cannot be None")
        return FlextResult.ok(value)


@final
class NotEmptyValidator(FlextValidator[str]):
    """Validates that string is not empty."""

    def __init__(self, *, strip_whitespace: bool = True) -> None:
        """Initialize validator."""
        self.strip_whitespace = strip_whitespace

    def validate(self, value: str | None) -> FlextResult[str]:
        """Validate string is not empty."""
        if value is None:
            return FlextResult.fail("String cannot be None")

        check_value = value.strip() if self.strip_whitespace else value
        if not check_value:
            return FlextResult.fail("String cannot be empty")

        return FlextResult.ok(value)


@final
class LengthValidator(FlextValidator[str]):
    """Validates string length constraints."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: int | None = None,
    ) -> None:
        """Initialize with length constraints."""
        if min_length < 0:
            msg = "min_length cannot be negative"
            raise ValueError(msg)
        if max_length is not None and max_length < min_length:
            msg = "max_length cannot be less than min_length"
            raise ValueError(msg)

        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: str | None) -> FlextResult[str]:
        """Validate string length."""
        if value is None:
            return FlextResult.fail("String cannot be None")

        length = len(value)

        if length < self.min_length:
            msg = f"String too short: {length} < {self.min_length}"
            return FlextResult.fail(msg)

        if self.max_length is not None and length > self.max_length:
            msg = f"String too long: {length} > {self.max_length}"
            return FlextResult.fail(msg)

        return FlextResult.ok(value)


@final
class EmailValidator(FlextValidator[str]):
    """Validates email format."""

    def __init__(self) -> None:
        """Initialize email validator with regex pattern."""
        self.pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

    def validate(self, value: str | None) -> FlextResult[str]:
        """Validate email format."""
        if value is None:
            return FlextResult.fail("Email cannot be None")

        if not self.pattern.match(value.strip()):
            return FlextResult.fail("Invalid email format")

        return FlextResult.ok(value)


@final
class RangeValidator[T](FlextValidator[T]):
    """Validates numeric range constraints."""

    def __init__(
        self,
        min_value: T | None = None,
        max_value: T | None = None,
    ) -> None:
        """Initialize with range constraints."""
        if (
            min_value is not None
            and max_value is not None
            and hasattr(min_value, "__gt__")
            and min_value > max_value
        ):
            msg = "min_value cannot be greater than max_value"
            raise ValueError(msg)

        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: T | None) -> FlextResult[T]:
        """Validate numeric range."""
        if value is None:
            return FlextResult.fail("Value cannot be None")

        if (
            self.min_value is not None
            and hasattr(value, "__lt__")
            and value < self.min_value
        ):
            msg = f"Value too small: {value} < {self.min_value}"
            return FlextResult.fail(msg)

        if (
            self.max_value is not None
            and hasattr(value, "__gt__")
            and value > self.max_value
        ):
            msg = f"Value too large: {value} > {self.max_value}"
            return FlextResult.fail(msg)

        return FlextResult.ok(value)


@final
class RegexValidator(FlextValidator[str]):
    """Validates string against regex pattern."""

    def __init__(self, pattern: str, error_message: str | None = None) -> None:
        """Initialize with regex pattern."""
        try:
            self.regex = re.compile(pattern)
        except re.error as e:
            msg = f"Invalid regex pattern: {e}"
            raise ValueError(msg) from e

        self.error_message = (
            error_message or f"String does not match pattern: {pattern}"
        )

    def validate(self, value: object) -> FlextResult[str]:
        """Validate string against regex."""
        if value is None:
            return FlextResult.fail("String cannot be None")

        if not isinstance(value, str):
            return FlextResult.fail("Value must be a string")

        if not self.regex.match(value):
            return FlextResult.fail(self.error_message)

        return FlextResult.ok(value)


@final
class ChoiceValidator[T](FlextValidator[T]):
    """Validates value is in allowed choices."""

    def __init__(self, choices: list[T]) -> None:
        """Initialize with allowed choices."""
        if not choices:
            msg = "Choices cannot be empty"
            raise ValueError(msg)
        self.choices = set(choices)

    def validate(self, value: T) -> FlextResult[T]:
        """Validate value is in choices."""
        if value not in self.choices:
            return FlextResult.fail(
                f"Value must be one of: {', '.join(map(str, self.choices))}",
            )

        return FlextResult.ok(value)


@final
class TypeValidator[T](FlextValidator[Any]):
    """Validates value is of expected type."""

    def __init__(self, expected_type: type[T]) -> None:
        """Initialize with expected type."""
        self.expected_type = expected_type

    def validate(self, value: object) -> FlextResult[T]:
        """Validate value type."""
        if not isinstance(value, self.expected_type):
            msg = f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
            return FlextResult.fail(msg)

        return FlextResult.ok(value)


# Utility Functions for Quick Validation


def validate[T](value: T) -> FlextValidationChain[T]:
    """Start a validation chain for a value."""
    return FlextValidationChain(value)


def validate_string(
    value: str,
    *,
    not_empty: bool = True,
    min_length: int = 0,
    max_length: int | None = None,
    pattern: str | None = None,
) -> FlextResult[str]:
    """Quick string validation with common rules."""
    chain = validate(value)

    if not_empty:
        chain = chain.validate_with(NotEmptyValidator())

    if min_length > 0 or max_length is not None:
        chain = chain.validate_with(LengthValidator(min_length, max_length))

    if pattern:
        chain = chain.validate_with(RegexValidator(pattern))

    return chain.result()


def validate_email(value: str) -> FlextResult[str]:
    """Quick email validation."""
    return (
        validate(value)
        .validate_with(NotEmptyValidator())
        .validate_with(EmailValidator())
        .result()
    )


def validate_number(
    value: T,
    *,
    min_value: T | None = None,
    max_value: T | None = None,
) -> FlextResult[T]:
    """Quick numeric validation."""
    return (
        validate(value)
        .validate_with(NotNoneValidator())
        .validate_with(RangeValidator(min_value, max_value))
        .result()
    )


def validate_choice(value: T, choices: list[T]) -> FlextResult[T]:
    """Quick choice validation."""
    return validate(value).validate_with(ChoiceValidator(choices)).result()


def validate_type[T](value: object, expected_type: type[T]) -> FlextResult[T]:
    """Quick type validation."""
    return TypeValidator(expected_type).validate(value)


def validate_all(*validations: FlextResult[Any]) -> FlextResult[list[Any]]:
    """Validate all results are successful."""
    values = []
    errors = []

    for validation in validations:
        if validation.success:
            values.append(validation.data)
        else:
            errors.append(validation.error or "Validation failed")

    if errors:
        return FlextResult.fail("; ".join(errors))

    return FlextResult.ok(values)


def validate_any[T](*validations: FlextResult[T]) -> FlextResult[T]:
    """Return first successful validation."""
    errors = []

    for validation in validations:
        if validation.success:
            return validation
        errors.append(validation.error or "Validation failed")

    return FlextResult.fail(f"All validations failed: {'; '.join(errors)}")


__all__ = [
    "ChoiceValidator",
    "EmailValidator",
    "FlextValidationChain",
    "FlextValidator",
    "LengthValidator",
    "NotEmptyValidator",
    "NotNoneValidator",
    "RangeValidator",
    "RegexValidator",
    "TypeValidator",
    "validate",
    "validate_all",
    "validate_any",
    "validate_choice",
    "validate_email",
    "validate_number",
    "validate_string",
    "validate_type",
]
