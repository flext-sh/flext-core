"""Service case generator helpers for flext-core tests."""

from __future__ import annotations


from tests._utilities.case_service_factories import (
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
)
from tests import c
from tests import m

from tests import p, t


class TestsFlextUtilitiesCaseGeneratorsMixin(
    TestsFlextUtilitiesCaseServiceFactoriesMixin
):
    """Service case generator helpers."""

    class TestDataGenerators:
        """Advanced test data generators using comprehensions and patterns."""

        @staticmethod
        def generate_user_success_cases(
            num_cases: int = 3,
        ) -> t.SequenceOf[p.Tests.ServiceTestCase]:
            """Generate successful user service test cases."""
            return [
                m.Tests.ServiceTestCase(
                    service_type=c.Tests.SERVICE_TEST_TYPE_GET_USER,
                    input_value=str(i * 100 + 1),
                    description=f"Valid user ID {i}",
                )
                for i in range(1, num_cases + 1)
            ]

        @staticmethod
        def generate_validation_success_cases(
            num_cases: int = 2,
        ) -> t.SequenceOf[p.Tests.ServiceTestCase]:
            """Generate successful validation test cases."""
            return [
                m.Tests.ServiceTestCase(
                    service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                    input_value=f"value_{i}",
                    description=f"Valid input {i}",
                )
                for i in range(1, num_cases + 1)
            ] + [
                m.Tests.ServiceTestCase(
                    service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                    input_value="test",
                    extra_param=2,
                    description="Custom min length",
                )
            ]

        @staticmethod
        def generate_validation_failure_cases() -> t.SequenceOf[
            m.Tests.ServiceTestCase
        ]:
            """Generate validation failure test cases."""
            return [
                m.Tests.ServiceTestCase(
                    service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                    input_value="ab",
                    expected_success=False,
                    expected_error="must be at least 3 characters",
                    description="Too short input",
                ),
                m.Tests.ServiceTestCase(
                    service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                    input_value="x",
                    expected_success=False,
                    expected_error="must be at least 5 characters",
                    extra_param=5,
                    description="Custom length requirement",
                ),
            ]

    class ServiceTestCases:
        """Unified factory for all test cases using advanced patterns."""

        @staticmethod
        def user_success() -> t.SequenceOf[p.Tests.ServiceTestCase]:
            """Generate cached-style success cases on demand."""
            return TestsFlextUtilitiesCaseGeneratorsMixin.TestDataGenerators.generate_user_success_cases()

        @staticmethod
        def validate_success() -> t.SequenceOf[p.Tests.ServiceTestCase]:
            """Generate cached-style validation success cases on demand."""
            return TestsFlextUtilitiesCaseGeneratorsMixin.TestDataGenerators.generate_validation_success_cases()

        @staticmethod
        def validate_failure() -> t.SequenceOf[p.Tests.ServiceTestCase]:
            """Generate cached-style validation failure cases on demand."""
            return TestsFlextUtilitiesCaseGeneratorsMixin.TestDataGenerators.generate_validation_failure_cases()

        @staticmethod
        def create_service(
            case: p.Tests.ServiceTestCase,
        ) -> (
            TestsFlextUtilitiesCaseGeneratorsMixin.GetUserService
            | TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingService
            | TestsFlextUtilitiesCaseGeneratorsMixin.FailingService
        ):
            """Create appropriate service based on case type."""
            return TestsFlextUtilitiesCaseGeneratorsMixin.ServiceFactoryRegistry.create_service(
                case
            )

    class GenericModelFactory:
        """Factories for generic reusable models (Value, Snapshot, Progress)."""

        @staticmethod
        def operation_progress(
            success: int = 0, failure: int = 0, skipped: int = 0
        ) -> p.Tests.Operation:
            """Create OperationProgress."""
            return m.Tests.Operation(
                success_count=success,
                failure_count=failure,
                skipped_count=skipped,
                metadata={},
            )

        @staticmethod
        def conversion_progress() -> p.Tests.Conversion:
            """Create ConversionProgress."""
            return m.Tests.Conversion(
                converted=[], errors=[], warnings=[], skipped=[], metadata={}
            )

    @staticmethod
    def reset_all_factories() -> None:
        """Reset all factory states for test isolation."""
        TestsFlextUtilitiesCaseGeneratorsMixin.UserFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.GetUserServiceFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingServiceFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.GetUserServiceAutoFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingServiceAutoFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ServiceTestCaseFactory.reset()


__all__: list[str] = ["TestsFlextUtilitiesCaseGeneratorsMixin"]
