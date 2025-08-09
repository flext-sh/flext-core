"""Base validation abstractions following SOLID principles.

This module provides abstract validation patterns used across
the FLEXT ecosystem. Concrete implementations should be in their
respective domain modules.

Classes:
    FlextAbstractValidator: Base class for validators.
    FlextCompositeValidator: Chains multiple validators.
    FlextValidationRule: Abstract validation rule pattern.
    FlextValidationContext: Context for validation operations.

Functions:
    create_flext_validator: Create validator from predicate function.
    chain_flext_validators: Chain multiple validators.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextAbstractValidator[T](ABC):
    """Abstract base class for validators following SOLID principles."""

    @abstractmethod
    def validate_value(self, value: T) -> FlextResult[T]:
        """Validate a single value."""
        ...

    def __call__(self, value: T) -> FlextResult[T]:
        """Make validator callable as a function."""
        return self.validate_value(value)


class FlextCompositeValidator[T](FlextAbstractValidator[T]):
    """Composite validator that chains multiple validators."""

    def __init__(self, validators: list[FlextAbstractValidator[T]]) -> None:
        """Initialize composite validator."""
        self._validators = validators

    def validate_value(self, value: T) -> FlextResult[T]:
        """Validate value through all validators in sequence."""
        for validator in self._validators:
            result = validator.validate_value(value)
            if result.is_failure:
                return result
        return FlextResult.ok(value)


class FlextAbstractValidationRule(ABC):
    """Abstract validation rule following Strategy pattern.

    Note: Prefer the runtime-checkable protocol `FlextValidationRule`
    from `flext_core.protocols` for type annotations. This abstract
    class remains for concrete rule implementations that need shared
    behavior and to support composition without duplication.
    """

    @abstractmethod
    def apply(self, value: object) -> FlextResult[None]:
        """Apply the validation rule."""
        ...

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the rule."""
        ...


class FlextValidationContext:
    """Context for validation operations."""

    def __init__(self) -> None:
        """Initialize validation context."""
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._metadata: dict[str, object] = {}

    def add_error(self, error: str) -> None:
        """Add validation error."""
        self._errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self._warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if context has errors."""
        return len(self._errors) > 0

    def get_result(self) -> FlextResult[None]:
        """Get validation result from context."""
        if self.has_errors():
            return FlextResult.fail("; ".join(self._errors))
        return FlextResult.ok(None)

    def clear(self) -> None:
        """Clear all errors and warnings."""
        self._errors.clear()
        self._warnings.clear()
        self._metadata.clear()


def create_flext_validator[T](
    validate_func: Callable[[T], bool],
    error_message: str = "Validation failed",
) -> FlextAbstractValidator[T]:
    """Create a validator from a simple predicate function."""

    class SimpleValidator(FlextAbstractValidator[T]):
        def validate_value(self, value: T) -> FlextResult[T]:
            if validate_func(value):
                return FlextResult.ok(value)
            return FlextResult.fail(error_message)

    return SimpleValidator()


def chain_flext_validators[T](
    *validators: FlextAbstractValidator[T],
) -> FlextAbstractValidator[T]:
    """Chain multiple validators into a composite."""
    return FlextCompositeValidator(list(validators))


# Export API
__all__ = [
    "FlextAbstractValidationRule",
    "FlextAbstractValidator",
    "FlextCompositeValidator",
    "FlextValidationContext",
    "chain_flext_validators",
    "create_flext_validator",
]
