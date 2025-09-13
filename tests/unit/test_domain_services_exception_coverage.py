"""Targeted domain services exception coverage tests for missing lines 50-52.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextDomainService, FlextResult


class TestDomainServiceExceptionCoverage:
    """Targeted tests for FlextDomainService exception handling coverage."""

    def test_is_valid_exception_handling_comprehensive(self) -> None:
        """Test is_valid method exception handling - covers lines 50-52."""

        class ExceptionThrowingService(FlextDomainService[str]):
            """Service that throws exceptions in validate_business_rules."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

            def validate_business_rules(self) -> FlextResult[None]:
                # This will raise an exception during is_valid check
                msg = "Business rules validation failed unexpectedly"
                raise RuntimeError(msg)

        service = ExceptionThrowingService()

        # This should catch the exception and return False (lines 50-52)
        result = service.is_valid()
        assert result is False

    def test_is_valid_exception_handling_various_error_types(self) -> None:
        """Test is_valid method handles different exception types."""

        class ValueErrorService(FlextDomainService[str]):
            """Service that throws ValueError in validate_business_rules."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

            def validate_business_rules(self) -> FlextResult[None]:
                msg = "Invalid business rule configuration"
                raise ValueError(msg)

        service = ValueErrorService()
        result = service.is_valid()
        assert result is False

        class TypeErrorService(FlextDomainService[str]):
            """Service that throws TypeError in validate_business_rules."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

            def validate_business_rules(self) -> FlextResult[None]:
                msg = "Type mismatch in business rules"
                raise TypeError(msg)

        service = TypeErrorService()
        result = service.is_valid()
        assert result is False

    def test_is_valid_exception_handling_with_system_errors(self) -> None:
        """Test is_valid exception handling with system-level errors."""

        class SystemErrorService(FlextDomainService[str]):
            """Service that throws system-level errors."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

            def validate_business_rules(self) -> FlextResult[None]:
                msg = "System error during validation"
                raise OSError(msg)

        service = SystemErrorService()
        result = service.is_valid()
        assert result is False

    def test_is_valid_success_path_still_works(self) -> None:
        """Verify is_valid success path still works after exception testing."""

        class SuccessfulService(FlextDomainService[str]):
            """Service with successful validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("success")

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        service = SuccessfulService()
        result = service.is_valid()
        assert result is True

    def test_is_valid_business_rule_failure_vs_exception(self) -> None:
        """Test difference between business rule failure and exception."""

        # Business rule failure (returns False from result)
        class BusinessFailureService(FlextDomainService[str]):
            """Service with business rule failure."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].fail("Business rule failed")

        business_service = BusinessFailureService()
        result = business_service.is_valid()
        assert result is False

        # Exception during validation (caught by exception handler)
        class ExceptionService(FlextDomainService[str]):
            """Service with exception during validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

            def validate_business_rules(self) -> FlextResult[None]:
                msg = "Validation threw exception"
                raise ValueError(msg)

        exception_service = ExceptionService()
        result = exception_service.is_valid()
        assert result is False
