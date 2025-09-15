#!/usr/bin/env python3
"""05 - Advanced Validation using FlextCore EXISTING functionality.

Demonstrates direct use of FlextValidations.Domain.UserValidator,
FlextValidations.Service, and existing FlextCore validation patterns.
ZERO code duplication - uses only what already exists.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

from flext_core import (
    FlextResult,
    FlextTypes,
    FlextValidations,
)


class ValidationShowcaseService:
    """Showcase usando APENAS funcionalidades existentes do FlextCore."""

    def __init__(self) -> None:
        """Initialize usando apenas funcionalidades existentes."""
        # Create user validator using available methods
        self._user_validator = FlextValidations.create_user_validator()

    def demonstrate_user_validation(
        self, user_data: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Demonstra validação de usuário usando UserValidator existente."""
        # Call the validator function with user data
        return self._user_validator(user_data)

    def demonstrate_email_validation(self, email: str) -> FlextResult[str]:
        """Demonstra validação de email usando funcionalidade existente."""
        return FlextValidations.validate_email(email)

    def demonstrate_string_validation(self, value: object) -> FlextResult[bool]:
        """Demonstra validação de string usando funcionalidade existente."""
        if isinstance(value, str):
            return FlextResult[bool].ok(data=True)
        return FlextResult[bool].fail("Value must be a string")

    def demonstrate_numeric_validation(self, value: object) -> FlextResult[bool]:
        """Demonstra validação numérica usando funcionalidade existente."""
        return FlextValidations.validate_numeric_field(value)

    def demonstrate_api_request_validation(
        self, request_data: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Demonstra validação de request usando funcionalidade existente."""
        # Use the available API request validation method
        return FlextValidations.validate_api_request(request_data)


def main() -> None:
    """Main demonstration usando APENAS funcionalidades existentes do FlextCore."""
    validator = ValidationShowcaseService()

    print("🚀 FlextCore Advanced Validation Showcase - ZERO Duplication")
    print("=" * 50)
    print(
        "Features: FlextValidations.Domain.UserValidator • FlextValidations.Service • Direct API"
    )
    print()

    # Test user validation usando UserValidator existente
    print("1. User Validation usando FlextValidations.Domain.UserValidator:")
    test_users = [
        {"name": "Alice Johnson", "email": "alice@example.com"},
        {"name": "Bob", "email": "invalid-email"},  # Invalid
    ]

    for user_data in test_users:
        result = validator.demonstrate_user_validation(dict(user_data))
        if result.success:
            validated = result.data
            if validated is not None:
                print(f"✅ Valid: {validated.get('name')} ({validated.get('email')})")
            else:
                print("❌ Invalid: No data returned")
        else:
            print(f"❌ Invalid: {result.error}")

    # Test email validation usando FlextValidations.validate_email
    print("\n2. Email Validation usando FlextValidations.validate_email:")
    emails = ["valid@example.com", "invalid-email", "another.valid+email@domain.co.uk"]

    for email in emails:
        # Use FlextValidations directly instead of method calls
        email_result = FlextValidations.validate_email(str(email))
        status = "✅ Valid" if email_result.success else f"❌ {email_result.error}"
        print(f"Email {email}: {status}")

    # Test string validation usando FlextValidations.validate_string_field
    print("\n3. String Field Validation usando FlextValidations.validate_string_field:")
    strings = ["Valid String", "", 123]  # Mixed types

    for string_val in strings:
        # Use FlextValidations directly
        string_result = FlextValidations.validate_string_field(str(string_val))
        status = "✅ Valid" if string_result.success else f"❌ {string_result.error}"
        print(f"String '{string_val}': {status}")

    # Test numeric validation usando FlextValidations.validate_numeric_field
    print(
        "\n4. Numeric Field Validation usando FlextValidations.validate_numeric_field:"
    )
    numbers = [42, "not a number", math.pi]

    for number in numbers:
        # Use FlextValidations directly
        numeric_result = FlextValidations.validate_numeric_field(number)
        status = "✅ Valid" if numeric_result.success else f"❌ {numeric_result.error}"
        print(f"Number '{number}': {status}")

    # Test API request validation usando ApiRequestValidator herdado
    print(
        "\n5. API Request Validation usando FlextValidations.Service.ApiRequestValidator:"
    )
    api_requests: list[FlextTypes.Core.Dict] = [
        {"action": "create_user", "version": "1.0", "data": {}},
        {"incomplete": "request"},  # Missing required fields
    ]

    for request_data in api_requests:
        # Use FlextValidations directly for API request validation
        api_result = FlextValidations.validate_api_request(request_data)
        status = "✅ Valid" if api_result.success else f"❌ {api_result.error}"
        print(f"API Request: {status}")

    print("\n✅ FlextCore Advanced Validation Demo Completed Successfully!")
    print(
        "💪 ZERO code duplication - using only existing FlextValidations functionality!"
    )


if __name__ == "__main__":
    main()
