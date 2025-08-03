"""FLEXT Core Builder - Internal Implementation Module.

Internal implementation providing the foundational logic for builder patterns.
This module is part of the Internal Implementation Layer and should not be imported
directly by ecosystem projects. Use the public API through guards module instead.

Module Role in Architecture:
    Internal Implementation Layer → Builder Patterns → Public API Layer

    This internal module provides:
    - Base builder class with property management
    - Fluent builder interface with method chaining
    - Validation infrastructure for builder state
    - Factory functions for builder creation

Implementation Patterns:
    Builder Pattern: Progressive configuration with validation
    Fluent Interface: Method chaining with conditional building

Design Principles:
    - Single responsibility for internal builder implementation concerns
    - No external dependencies beyond standard library and sibling modules
    - Performance-optimized implementations for public API consumption
    - Type safety maintained through internal validation

Access Restrictions:
    - This module is internal and not exported in __init__.py
    - Use guards module for all external access to builder functionality
    - Breaking changes may occur without notice in internal modules
    - No compatibility guarantees for internal implementation details

Quality Standards:
    - Internal implementation must maintain public API contracts
    - Performance optimizations must not break type safety
    - Code must be thoroughly tested through public API surface
    - Internal changes must not affect public behavior

See Also:
    guards: Public API for builder patterns and validation
    docs/python-module-organization.md: Internal module architecture

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from flext_core.flext_types import TAnyDict


class _BaseBuilder:
    """Foundation builder pattern without external dependencies."""

    def __init__(self, builder_name: str = "unnamed") -> None:
        """Initialize base builder.

        Args:
            builder_name: Name for this builder instance

        """
        self._builder_name = builder_name
        self._properties: TAnyDict = {}
        self._validation_errors: list[str] = []
        self._is_built = False

    def _set_property(self, key: str, value: object) -> _BaseBuilder:
        """Set a property value.

        Args:
            key: Property key
            value: Property value

        Returns:
            Self for chaining

        """
        if self._is_built:
            self._validation_errors.append("Cannot modify built object")
            return self

        if not FlextValidators.is_non_empty_string(key):
            self._validation_errors.append(f"Invalid property key: {key}")
            return self

        self._properties[key] = value
        return self

    def _get_property(self, key: str, default: object = None) -> object:
        """Get property value.

        Args:
            key: Property key
            default: Default value if not found

        Returns:
            Property value or default

        """
        if not FlextValidators.is_non_empty_string(key):
            return default

        return self._properties.get(key, default)

    def _has_property(self, key: str) -> bool:
        """Check if property exists.

        Args:
            key: Property key to check

        Returns:
            True if property exists

        """
        if not FlextValidators.is_non_empty_string(key):
            return False

        return key in self._properties

    def _validate_required(self, key: str, error_message: str | None = None) -> bool:
        """Validate that a required property exists.

        Args:
            key: Property key to validate
            error_message: Custom error message

        Returns:
            True if property exists and is valid

        """
        if not self._has_property(key):
            message = error_message or f"Required property '{key}' is missing"
            self._validation_errors.append(message)
            return False

        value = self._get_property(key)
        if value is None:
            message = error_message or f"Required property '{key}' cannot be None"
            self._validation_errors.append(message)
            return False

        return True

    def _validate_type(
        self,
        key: str,
        expected_type: type[object],
        error_message: str | None = None,
    ) -> bool:
        """Validate property type.

        Args:
            key: Property key to validate
            expected_type: Expected type
            error_message: Custom error message

        Returns:
            True if property has correct type

        """
        if not self._has_property(key):
            return False

        value = self._get_property(key)
        if value is not None and not isinstance(value, expected_type):
            message = (
                error_message
                or f"Property '{key}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            self._validation_errors.append(message)
            return False

        return True

    def _validate_string_length(
        self,
        key: str,
        min_length: int = 0,
        max_length: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Validate string property length.

        Args:
            key: Property key to validate
            min_length: Minimum length
            max_length: Maximum length
            error_message: Custom error message

        Returns:
            True if string length is valid

        """
        if not self._has_property(key):
            return False

        value = self._get_property(key)
        if not isinstance(value, str):
            return False

        if len(value) < min_length:
            message = (
                error_message
                or f"Property '{key}' must be at least {min_length} characters"
            )
            self._validation_errors.append(message)
            return False

        if max_length is not None and len(value) > max_length:
            message = (
                error_message
                or f"Property '{key}' must be at most {max_length} characters"
            )
            self._validation_errors.append(message)
            return False

        return True

    def _clear_errors(self) -> None:
        """Clear all validation errors."""
        self._validation_errors.clear()

    def _add_error(self, error: str) -> None:
        """Add validation error.

        Args:
            error: Error message to add

        """
        if FlextValidators.is_non_empty_string(error):
            self._validation_errors.append(error)

    def _is_valid(self) -> bool:
        """Check if builder state is valid.

        Returns:
            True if no validation errors

        """
        return len(self._validation_errors) == 0

    def _get_errors(self) -> list[str]:
        """Get all validation errors.

        Returns:
            List of validation error messages

        """
        return self._validation_errors.copy()

    def _reset(self) -> _BaseBuilder:
        """Reset builder to initial state.

        Returns:
            Self for chaining

        """
        self._properties.clear()
        self._validation_errors.clear()
        self._is_built = False
        return self

    def _mark_built(self) -> None:
        """Mark builder as built (immutable)."""
        self._is_built = True

    @property
    def builder_name(self) -> str:
        """Get builder name."""
        return self._builder_name

    @property
    def property_count(self) -> int:
        """Get number of properties."""
        return len(self._properties)

    @property
    def error_count(self) -> int:
        """Get number of validation errors."""
        return len(self._validation_errors)

    @property
    def is_built(self) -> bool:
        """Check if builder has been built."""
        return self._is_built

    @property
    def property_keys(self) -> list[str]:
        """Get list of all property keys."""
        return list(self._properties.keys())


class _BaseFluentBuilder(_BaseBuilder):
    """Fluent builder with method chaining support."""

    def with_property(self, key: str, value: object) -> _BaseFluentBuilder:
        """Set property with fluent interface.

        Args:
            key: Property key
            value: Property value

        Returns:
            Self for chaining

        """
        self._set_property(key, value)
        return self

    def when(self, *, condition: bool) -> _BaseFluentBuilder:
        """Conditional builder for fluent interface.

        Args:
            condition: Condition to check

        Returns:
            Self for chaining

        """
        # Store condition for use in subsequent calls
        self._set_property("_last_condition", condition)
        return self

    def then_set(self, key: str, value: object) -> _BaseFluentBuilder:
        """Set property if last condition was true.

        Args:
            key: Property key
            value: Property value

        Returns:
            Self for chaining

        """
        last_condition = self._get_property("_last_condition", default=True)
        if isinstance(last_condition, bool) and last_condition:
            self._set_property(key, value)
        return self

    def validate(self) -> _BaseFluentBuilder:
        """Trigger validation manually.

        Returns:
            Self for chaining

        """
        # Subclasses can override this
        return self

    def clear_errors(self) -> _BaseFluentBuilder:
        """Clear validation errors with fluent interface.

        Returns:
            Self for chaining

        """
        self._clear_errors()
        return self

    def reset(self) -> _BaseFluentBuilder:
        """Reset with fluent interface.

        Returns:
            Self for chaining

        """
        self._reset()
        return self


# Factory functions
def _create_builder(builder_name: str = "unnamed") -> _BaseBuilder:
    """Create a basic builder instance."""
    return _BaseBuilder(builder_name)


def _create_fluent_builder(builder_name: str = "unnamed") -> _BaseFluentBuilder:
    """Create a fluent builder instance."""
    return _BaseFluentBuilder(builder_name)


# Export API
__all__ = [
    "_BaseBuilder",
    "_BaseFluentBuilder",
    "_create_builder",
    "_create_fluent_builder",
]
