"""FLEXT Validation Mixin - Data validation using centralized components.

This module provides validation mixins that leverage the centralized FLEXT
ecosystem components for type-safe, consistent validation patterns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.type_adapters import FlextTypeAdapters
from flext_core.utilities import FlextUtilities
from flext_core.validations import FlextValidations


class FlextValidation:
    """Unified validation system using centralized FLEXT components."""

    @staticmethod
    def initialize_validation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Initialize validation state on an object."""
        validation_id = FlextUtilities.Generators.generate_entity_id()
        obj._validation_errors = []
        obj._is_valid = True
        obj._validation_initialized = True
        obj._validation_id = validation_id

    @staticmethod
    def validate_required_fields(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        fields: list[str],
    ) -> FlextResult[None]:
        """Validate required fields using FlextValidations."""
        field_values: dict[str, object] = {}
        for field in fields:
            field_values[field] = getattr(obj, field, None)

        for field, value in field_values.items():
            # Check if value is None or empty string
            if value is None or (isinstance(value, str) and not value.strip()):
                error_msg = f"Required field '{field}' is missing or empty"
                return FlextResult[None].fail(
                    error_msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        return FlextResult[None].ok(None)

    @staticmethod
    def validate_field_types(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        field_types: Mapping[str, type],
    ) -> FlextResult[None]:
        """Validate field types using FlextTypeAdapters."""
        validation_errors: list[str] = []

        for field, expected_type in field_types.items():
            value = getattr(obj, field, None)

            if value is None:
                continue

            adapter = FlextTypeAdapters.Foundation.create_basic_adapter(expected_type)
            validation_result = FlextTypeAdapters.Foundation.validate_with_adapter(
                adapter, value
            )

            if validation_result.is_failure:
                actual_type = type(value).__name__
                expected_name = expected_type.__name__
                error_msg = (
                    f"Field '{field}': expected {expected_name}, got {actual_type}"
                )
                validation_errors.append(error_msg)

        if validation_errors:
            return FlextResult[None].fail(
                f"Type validation failed: {'; '.join(validation_errors)}",
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

        return FlextResult[None].ok(None)

    @staticmethod
    def validate_email(email: str) -> FlextResult[str]:
        """Validate email using FlextValidations."""
        return FlextValidations.validate_email(email)

    @staticmethod
    def validate_url(url: str) -> FlextResult[str]:
        """Validate URL using FlextConstants patterns."""
        import re

        url_pattern = FlextConstants.Patterns.URL_PATTERN
        if re.match(url_pattern, url):
            return FlextResult[str].ok(url)
        return FlextResult[str].fail(
            f"Invalid URL format: {url}",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

    @staticmethod
    def validate_phone(phone: str) -> FlextResult[str]:
        """Validate phone number using basic pattern."""
        import re

        # Basic international phone pattern
        phone_pattern = r"^\+?[1-9]\d{1,14}$"
        clean_phone = re.sub(r"[\s\-\(\)]", "", phone)
        if re.match(phone_pattern, clean_phone):
            return FlextResult[str].ok(phone)
        return FlextResult[str].fail(
            f"Invalid phone format: {phone}",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

    @staticmethod
    def add_validation_error(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        error: str | Exception,
    ) -> None:
        """Add a validation error to an object."""
        if not hasattr(obj, "_validation_initialized"):
            FlextValidation.initialize_validation(obj)

        error_msg = str(error) if isinstance(error, Exception) else error
        errors = cast("list[str]", getattr(obj, "_validation_errors", []))
        errors.append(error_msg)
        obj._validation_errors = errors
        obj._is_valid = False

    @staticmethod
    def clear_validation_errors(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear all validation errors."""
        if not hasattr(obj, "_validation_initialized"):
            FlextValidation.initialize_validation(obj)

        obj._validation_errors = []
        obj._is_valid = True

    @staticmethod
    def get_validation_errors(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> list[str]:
        """Get all validation errors."""
        if not hasattr(obj, "_validation_initialized"):
            FlextValidation.initialize_validation(obj)

        return cast("list[str]", getattr(obj, "_validation_errors", []))

    @staticmethod
    def is_valid(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> bool:
        """Check if object is valid."""
        if not hasattr(obj, "_validation_initialized"):
            FlextValidation.initialize_validation(obj)

        errors = cast("list[str]", getattr(obj, "_validation_errors", []))
        return len(errors) == 0 and cast("bool", getattr(obj, "_is_valid", True))

    @staticmethod
    def mark_valid(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Mark object as valid."""
        if not hasattr(obj, "_validation_initialized"):
            FlextValidation.initialize_validation(obj)

        obj._is_valid = True

    class Validatable:
        """Mixin class providing validation capabilities."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize validation state."""
            super().__init__(*args, **kwargs)
            FlextValidation.initialize_validation(self)

        def validate_required_fields(self, fields: list[str]) -> FlextResult[None]:
            """Validate required fields are present."""
            return FlextValidation.validate_required_fields(self, fields)

        def validate_field_types(
            self, field_types: Mapping[str, type]
        ) -> FlextResult[None]:
            """Validate field types match expectations."""
            return FlextValidation.validate_field_types(self, field_types)

        def add_validation_error(self, error: str | Exception) -> None:
            """Add a validation error."""
            FlextValidation.add_validation_error(self, error)

        def clear_validation_errors(self) -> None:
            """Clear all validation errors."""
            FlextValidation.clear_validation_errors(self)

        def get_validation_errors(self) -> list[str]:
            """Get all validation errors."""
            return FlextValidation.get_validation_errors(self)

        def is_valid(self) -> bool:
            """Check if object is valid."""
            return FlextValidation.is_valid(self)

        def mark_valid(self) -> None:
            """Mark object as valid."""
            FlextValidation.mark_valid(self)
