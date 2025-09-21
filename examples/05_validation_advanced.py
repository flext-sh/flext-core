#!/usr/bin/env python3
"""05 - Advanced Validation using FLEXT Core EXISTING functionality.

Demonstrates direct use of FlextModels validation methods
and direct validation patterns.
ZERO code duplication - uses only what already exists.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

from flext_core import (
    FlextDomainService,
    FlextModels,
    FlextResult,
)


class ValidationResult(FlextModels.Value):
    """Structured validation result using FlextModels for type safety."""

    validation_type: str
    value: str
    is_valid: bool
    error_message: str | None = None


class AdvancedValidationService(FlextDomainService[ValidationResult]):
    """Advanced validation service using FLEXT Core patterns."""

    def __init__(self) -> None:
        """Initialize with FLEXT Core components."""
        super().__init__()

    def _validate_user_data(self, user_data: dict[str, object]) -> FlextResult[None]:
        """Validate user data using direct validation.

        Args:
            user_data: Dictionary containing user data to validate

        Returns:
            FlextResult[None]: Success or failure result

        """
        if not user_data:
            return FlextResult[None].fail("User data cannot be empty")
        name = user_data.get("name", "")
        email = user_data.get("email", "")
        if not isinstance(name, str) or len(name) < 2:
            return FlextResult[None].fail("Name must be at least 2 characters")
        if not isinstance(email, str):
            return FlextResult[None].fail("Email must be a string")
        email_result = FlextModels.create_validated_email(email)
        if email_result.is_failure:
            return FlextResult[None].fail(email_result.error or "Invalid email")
        return FlextResult[None].ok(None)

    def validate_input(
        self,
        value: object,
        validation_type: str,
    ) -> FlextResult[ValidationResult]:
        """Unified validation method handling all validation types.

        Args:
            value: The value to validate
            validation_type: Type of validation to perform

        Returns:
            FlextResult[ValidationResult]: Validation result wrapped in FlextResult

        """
        str_value = str(value) if value is not None else ""

        if validation_type == "user" and isinstance(value, dict):
            user_result = self._validate_user_data(value)
            return FlextResult[ValidationResult].ok(
                ValidationResult(
                    validation_type=validation_type,
                    value=str(value),
                    is_valid=user_result.is_success,
                    error_message=user_result.error if user_result.is_failure else None,
                ),
            )
        if validation_type == "email":
            email_result = FlextModels.create_validated_email(str_value)
            return FlextResult[ValidationResult].ok(
                ValidationResult(
                    validation_type=validation_type,
                    value=str_value,
                    is_valid=email_result.is_success,
                    error_message=email_result.error
                    if email_result.is_failure
                    else None,
                ),
            )
        if validation_type == "string":
            string_valid = isinstance(value, str) and bool(value and value.strip())
            return FlextResult[ValidationResult].ok(
                ValidationResult(
                    validation_type=validation_type,
                    value=str_value,
                    is_valid=string_valid,
                    error_message=None if string_valid else "String must be non-empty",
                ),
            )
        if validation_type == "numeric":
            numeric_valid = isinstance(value, (int, float)) and not isinstance(
                value,
                bool,
            )
            return FlextResult[ValidationResult].ok(
                ValidationResult(
                    validation_type=validation_type,
                    value=str_value,
                    is_valid=numeric_valid,
                    error_message=None if numeric_valid else "Value must be a number",
                ),
            )
        if validation_type == "api_request" and isinstance(value, dict):
            # Basic API request validation
            has_action = "action" in value and isinstance(value["action"], str)
            has_version = "version" in value and isinstance(value["version"], str)
            api_valid = has_action and has_version
            error_msg = (
                None
                if api_valid
                else "API request must have 'action' and 'version' fields"
            )
            return FlextResult[ValidationResult].ok(
                ValidationResult(
                    validation_type=validation_type,
                    value=str(value),
                    is_valid=api_valid,
                    error_message=error_msg,
                ),
            )

        return FlextResult[ValidationResult].fail(
            f"Unknown validation type: {validation_type}",
        )

    def execute(self) -> FlextResult[ValidationResult]:
        """Execute demo functionality - required by FlextDomainService.

        Returns:
            FlextResult[ValidationResult]: Demo validation result wrapped in FlextResult

        """
        # Demo execution with default user validation
        demo_user = {"name": "Demo User", "email": "demo@example.com"}
        return self.validate_input(demo_user, "user")


def main() -> None:
    """Advanced FLEXT Core validation with data-driven patterns."""
    print("🚀 Advanced FLEXT Core Validation Showcase")
    print("=" * 50)
    print("Architecture: FlextDomainService • FlextModels • Data-Driven Validation")
    print()

    service = AdvancedValidationService()

    # Data-driven validation scenarios
    validation_scenarios = [
        # User validations
        ("Valid User", {"name": "Alice Johnson", "email": "alice@example.com"}, "user"),
        ("Invalid User", {"name": "Bob", "email": "invalid-email"}, "user"),
        # Email validations
        ("Valid Email", "valid@example.com", "email"),
        ("Invalid Email", "invalid-email", "email"),
        ("Complex Email", "another.valid+email@domain.co.uk", "email"),
        # String validations
        ("Valid String", "Valid String", "string"),
        ("Empty String", "", "string"),
        ("Non-String", 123, "string"),
        # Numeric validations
        ("Valid Number", 42, "numeric"),
        ("Pi Number", math.pi, "numeric"),
        ("Invalid Number", "not a number", "numeric"),
        # API request validations
        (
            "Valid API Request",
            {"action": "create_user", "version": "1.0", "data": {}},
            "api_request",
        ),
        ("Invalid API Request", {"incomplete": "request"}, "api_request"),
    ]

    print("Comprehensive Validation Results:")
    print("-" * 35)

    for description, value, validation_type in validation_scenarios:
        result = service.validate_input(value, validation_type)
        if result.is_success:
            validation_result = result.unwrap()
            status = "✅" if validation_result.is_valid else "❌"
            error_info = (
                f" - {validation_result.error_message}"
                if validation_result.error_message
                else ""
            )
            print(f"{status} {description} ({validation_type}){error_info}")
        else:
            print(
                f"❌ {description} ({validation_type}) - Service Error: {result.error}",
            )

    print("\n✅ Advanced validation patterns demonstrated!")


if __name__ == "__main__":
    main()
