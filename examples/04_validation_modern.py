#!/usr/bin/env python3
"""04 - Modern Validation using FlextCore DIRECTLY.

Demonstrates DIRECT usage of FlextValidations eliminating ALL duplication:
- FlextValidations.create_user_validator() for business rules
- FlextValidations.Rules for field validation
- FlextDomainService for service architecture

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

from flext_core import (
    FlextDomainService,
    FlextResult,
    FlextTypes,
    FlextValidations,
)


class ProfessionalValidationService(FlextDomainService[object]):
    """UNIFIED validation service using FlextCore DIRECTLY.

    Eliminates ALL duplication by using FlextValidations components directly:
    - FlextValidations.create_user_validator() for user business rules
    - FlextValidations.Rules.StringRules for string validation
    - FlextValidations.validate_* convenience methods
    """

    def __init__(self) -> None:
        """Initialize with FlextCore components."""
        super().__init__()
        self._user_validator = FlextValidations.create_user_validator()

    class ValidationReport:
        """Validation report using simple data class pattern."""

        def __init__(
            self,
            total_validations: int,
            successful_validations: int,
            failed_validations: int,
            success_rate: float,
            validation_errors: FlextTypes.Core.StringList,
        ) -> None:
            """Initialize validation report with metrics."""
            self.total_validations = total_validations
            self.successful_validations = successful_validations
            self.failed_validations = failed_validations
            self.success_rate = success_rate
            self.validation_errors = validation_errors

    class _ValidationHelper:
        """Nested helper for validation operations."""

        @staticmethod
        def process_validation_batch(
            validation_data: list[tuple[str, object, str]], validator_func: object
        ) -> tuple[int, int, FlextTypes.Core.StringList]:
            """Process batch of validations with unified error handling."""
            successful = 0
            failed = 0
            errors: FlextTypes.Core.StringList = []

            for description, value, expected_type in validation_data:
                if callable(validator_func):
                    try:
                        result = validator_func(value)
                        if hasattr(result, "is_success") and getattr(
                            result, "is_success", False
                        ):
                            successful += 1
                        else:
                            failed += 1
                            error_msg = getattr(
                                result, "error", f"Validation failed for {description}"
                            )
                            errors.append(f"{expected_type}: {error_msg}")
                    except Exception as e:
                        failed += 1
                        errors.append(f"{expected_type}: Exception - {e}")
                else:
                    failed += 1
                    errors.append(f"{expected_type}: Invalid validator function")

            return successful, failed, errors

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate using FlextDomainService pattern."""
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[object]:
        """Execute comprehensive validation - required by FlextDomainService."""
        return self.demonstrate_comprehensive_validation()

    def demonstrate_user_validation(self) -> FlextResult[None]:
        """Demonstrate user validation using FlextValidations.create_user_validator() DIRECTLY."""
        print("1. User Validation usando FlextValidations.create_user_validator():")

        test_users = [
            {"name": "Alice Johnson", "email": "alice@example.com"},
            {"name": "Bob Smith", "email": "bob@company.com"},
            {"name": "", "email": "invalid-email"},  # Should fail
        ]

        for user_data in test_users:
            # Convert to dict[str, object] for validation API
            user_data_obj = dict[str, object](user_data)
            result = self._user_validator.validate_business_rules(user_data_obj)
            if result.is_success:
                validated = result.unwrap()
                print(f"✅ Valid: {validated.get('name')} ({validated.get('email')})")
            else:
                print(f"❌ Invalid: {result.error}")

        return FlextResult[None].ok(None)

    def demonstrate_email_validation(self) -> FlextResult[None]:
        """Demonstrate email validation using FlextValidations.Rules.StringRules DIRECTLY."""
        print("\n2. Email Validation usando FlextValidations.Rules.StringRules:")

        emails = [
            "valid@example.com",
            "another.valid+email@domain.co.uk",
            "invalid-email",
            "@invalid.com",
            "invalid@",
        ]

        for email in emails:
            result = FlextValidations.Rules.StringRules.validate_email(email)
            status = "✅ Valid" if result.is_success else f"❌ {result.error}"
            print(f"Email {email}: {status}")

        return FlextResult[None].ok(None)

    def demonstrate_field_validation(self) -> FlextResult[None]:
        """Demonstrate field validation using FlextValidations convenience methods DIRECTLY."""
        print("\n3. Field Validation usando FlextValidations convenience methods:")

        # String field validation using FlextValidations DIRECTLY
        strings = ["Valid String", "", 123]  # Mixed types
        for string_val in strings:
            result = FlextValidations.validate_string_field(string_val)
            status = "✅ Valid" if result.is_success else f"❌ {result.error}"
            print(f"String '{string_val}': {status}")

        # Numeric field validation using FlextValidations DIRECTLY
        numbers = [42, "not a number", math.pi]
        for number in numbers:
            result = FlextValidations.validate_numeric_field(number)
            status = "✅ Valid" if result.is_success else f"❌ {result.error}"
            print(f"Number '{number}': {status}")

        return FlextResult[None].ok(None)

    def demonstrate_comprehensive_validation(self) -> FlextResult[object]:
        """Demonstrate comprehensive validation using FlextValidations DIRECTLY."""
        print("\n4. Comprehensive Validation Report using FlextValidations:")

        # Test data for comprehensive validation
        validation_test_data = [
            ("Email format", "test@example.com", "Email"),
            ("Invalid email", "invalid-email", "Email"),
            ("Valid string", "Hello World", "String"),
            ("Empty string", "", "String"),
            ("Valid number", 42, "Number"),
            ("Invalid number", "not-a-number", "Number"),
        ]

        successful, failed, errors = self._ValidationHelper.process_validation_batch(
            validation_test_data, FlextValidations.validate_string_field
        )

        total = len(validation_test_data)
        success_rate = (successful / total) if total > 0 else 0.0

        report = self.ValidationReport(
            total_validations=total,
            successful_validations=successful,
            failed_validations=failed,
            success_rate=success_rate,
            validation_errors=errors,
        )

        print(
            f"✅ Validation Report: {successful}/{total} successful ({success_rate:.1%})"
        )
        if errors:
            print(f"⚠️  Errors found: {len(errors)}")

        return FlextResult[object].ok(report)


def main() -> None:
    """Main demonstration using FlextCore DIRECTLY - ZERO duplication."""
    service = ProfessionalValidationService()

    print("🚀 FlextCore Professional Validation Showcase - ZERO Duplication")
    print("=" * 50)
    print(
        "Features: FlextDomainService • FlextValidations.create_user_validator() • FlextValidations.Rules"
    )
    print()

    # Demonstrate user validation using FlextValidations DIRECTLY
    user_result = service.demonstrate_user_validation()
    if user_result.is_failure:
        print(f"❌ User validation failed: {user_result.error}")
        return

    # Demonstrate email validation using FlextValidations.Rules DIRECTLY
    email_result = service.demonstrate_email_validation()
    if email_result.is_failure:
        print(f"❌ Email validation failed: {email_result.error}")
        return

    # Demonstrate field validation using FlextValidations convenience methods DIRECTLY
    field_result = service.demonstrate_field_validation()
    if field_result.is_failure:
        print(f"❌ Field validation failed: {field_result.error}")
        return

    # Execute comprehensive validation using FlextDomainService pattern
    execution_result = service.execute()
    if execution_result.is_failure:
        print(f"❌ Comprehensive validation failed: {execution_result.error}")
        return
    report_obj = execution_result.unwrap()
    if hasattr(report_obj, "success_rate"):
        print(
            f"\n📊 Comprehensive validation completed with {getattr(report_obj, 'success_rate', 0.0):.1%} success rate"
        )

    print("\n✅ FlextCore Professional Validation Demo Completed Successfully!")
    print(
        "💪 Professional architecture using FlextDomainService with ZERO code duplication!"
    )


# Setup function for test compatibility
def create_validation_service() -> FlextResult[ProfessionalValidationService]:
    """Create validation service for test compatibility."""
    service = ProfessionalValidationService()
    return FlextResult[ProfessionalValidationService].ok(service)


if __name__ == "__main__":
    main()
