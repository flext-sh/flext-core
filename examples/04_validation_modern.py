#!/usr/bin/env python3
"""04 - Modern Validation using FLEXT Core components DIRECTLY.

Demonstrates DIRECT usage of FlextModels and built-in validation:
- FlextModels.create_validated_* methods for field validation
- Direct isinstance checks for type validation
- FlextDomainService for service architecture

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

from flext_core import (
    FlextDomainService,
    FlextModels,
    FlextResult,
    FlextTypes,
)


class ValidationReport(FlextModels.Value):
    """Validation report using FlextModels for type safety."""

    total_validations: int
    successful_validations: int
    failed_validations: int
    success_rate: float
    validation_errors: FlextTypes.Core.StringList


class ProfessionalValidationService(FlextDomainService[ValidationReport]):
    """Advanced validation service using FlextUtilities.Validation and FlextModels."""

    def __init__(self) -> None:
        """Initialize with FLEXT Core components."""
        super().__init__()

    def _validate_user_data(self, user_data: dict[str, object]) -> FlextResult[None]:
        """Validate user data using direct validation."""
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

    def _validate_data_batch(
        self, validation_scenarios: list[tuple[str, object, str]],
    ) -> ValidationReport:
        """Process validation batch using FlextUtilities.Validation patterns."""
        successful = 0
        failed = 0
        errors: FlextTypes.Core.StringList = []

        for _description, value, validation_type in validation_scenarios:
            # Use direct validation methods
            if validation_type == "email":
                email_result = FlextModels.create_validated_email(str(value))
                result = email_result
            elif validation_type == "string":
                if isinstance(value, str) and bool(value and value.strip()):
                    result = FlextResult[str].ok(value)
                else:
                    result = FlextResult[str].fail("String must be non-empty")
            elif validation_type == "number":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result = FlextResult[str].ok(str(value))
                else:
                    result = FlextResult[str].fail("Value must be a number")
            else:
                result = FlextResult[str].fail("Unknown validation type")

            if result.is_success:
                successful += 1
            else:
                failed += 1
                errors.append(f"{validation_type}: {result.error or 'Unknown error'}")

        total = len(validation_scenarios)
        success_rate = (successful / total) if total > 0 else 0.0

        return ValidationReport(
            total_validations=total,
            successful_validations=successful,
            failed_validations=failed,
            success_rate=success_rate,
            validation_errors=errors,
        )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate using FlextDomainService pattern."""
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[ValidationReport]:
        """Execute comprehensive validation - required by FlextDomainService."""
        # Comprehensive validation scenarios using data-driven approach
        validation_scenarios = [
            # User validation scenarios
            ("Alice Johnson", "alice@example.com", "email"),
            ("Bob Smith", "bob@company.com", "email"),
            ("", "invalid-email", "email"),
            # String validation scenarios
            ("Valid String", "Valid String", "string"),
            ("Empty string", "", "string"),
            ("Non-string", 123, "string"),
            # Number validation scenarios
            ("Valid number", 42, "number"),
            ("Pi number", math.pi, "number"),
            ("Invalid number", "not-a-number", "number"),
        ]

        return FlextResult[ValidationReport].ok(
            self._validate_data_batch(validation_scenarios),
        )

    def demonstrate_validation_patterns(self) -> None:
        """Demonstrate validation patterns with consolidated output."""
        print("🔍 FLEXT Core Validation Patterns:")
        print("-" * 35)

        # User validation using direct validation
        test_users = [
            {"name": "Alice Johnson", "email": "alice@example.com"},
            {"name": "", "email": "invalid-email"},
        ]

        for user_data in test_users:
            user_data_obj = dict[str, object](user_data)
            result = self._validate_user_data(user_data_obj)
            status = "✅" if result.is_success else "❌"
            name = user_data.get("name", "(empty)")
            email = user_data.get("email", "")
            print(f"{status} User: {name} | {email}")

        # Direct validation examples
        validation_examples = [
            ("Email", "valid@example.com"),
            ("Email", "invalid-email"),
            ("String", "Hello World"),
            ("Number", 42),
        ]

        for val_type, value in validation_examples:
            # Handle each validation type with direct validation
            if val_type == "Email":
                email_result = FlextModels.create_validated_email(str(value))
                status = "✅" if email_result.is_success else "❌"
            elif val_type == "String":
                string_valid = isinstance(value, str) and bool(value and value.strip())
                status = "✅" if string_valid else "❌"
            elif val_type == "Number":
                number_valid = isinstance(value, (int, float)) and not isinstance(
                    value, bool,
                )
                status = "✅" if number_valid else "❌"
            else:
                status = "❌"

            print(f"{status} {val_type}: {value}")


def main() -> None:
    """Advanced FLEXT Core validation with data-driven patterns."""
    print("🚀 Advanced FLEXT Core Validation Demo")
    print("=" * 40)
    print("Architecture: FlextDomainService • FlextModels • Direct Validation")
    print()

    service = ProfessionalValidationService()

    # Demonstrate validation patterns
    service.demonstrate_validation_patterns()

    # Execute comprehensive validation
    result = service.execute()
    if result.is_success:
        report = result.unwrap()
        print("\n📊 Validation Report:")
        print(f"   Total: {report.total_validations}")
        print(f"   Success: {report.successful_validations}")
        print(f"   Failed: {report.failed_validations}")
        print(f"   Success Rate: {report.success_rate:.1%}")

        if report.validation_errors:
            print(f"   Errors: {len(report.validation_errors)} found")
    else:
        print(f"❌ Validation failed: {result.error}")

    print("\n✅ Advanced validation patterns demonstrated!")


# Setup function for test compatibility
def create_validation_service() -> FlextResult[ProfessionalValidationService]:
    """Create validation service for test compatibility."""
    service = ProfessionalValidationService()
    return FlextResult[ProfessionalValidationService].ok(service)


if __name__ == "__main__":
    main()
