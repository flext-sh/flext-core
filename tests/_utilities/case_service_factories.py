"""Service case construction helpers for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests._utilities.service_factories import TestsFlextUtilitiesServiceFactoriesMixin
from tests.constants import c
from tests.models import m

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.typings import p, t


class TestsFlextUtilitiesCaseServiceFactoriesMixin(
    TestsFlextUtilitiesServiceFactoriesMixin,
):
    """Service case construction helpers."""

    class ServiceTestCaseFactory:
        """Factory for m.Tests.ServiceTestCase."""

        _service_types: ClassVar[Sequence[c.Tests.ServiceType]] = [
            c.Tests.SERVICE_TEST_TYPE_GET_USER,
            c.Tests.SERVICE_TEST_TYPE_VALIDATE,
            c.Tests.SERVICE_TEST_TYPE_FAIL,
        ]
        _type_index: ClassVar[int] = 0
        _words: ClassVar[Sequence[str]] = [
            "test",
            "sample",
            "example",
            "demo",
            "data",
        ]
        _word_index: ClassVar[int] = 0

        @classmethod
        def _next_type(cls) -> c.Tests.ServiceType:
            """Get next service type from rotation."""
            service_type = cls._service_types[cls._type_index % len(cls._service_types)]
            cls._type_index += 1
            return service_type

        @classmethod
        def _next_word(cls) -> str:
            """Get next word from rotation."""
            word = cls._words[cls._word_index % len(cls._words)]
            cls._word_index += 1
            return word

        @classmethod
        def build(
            cls,
            *,
            service_type: c.Tests.ServiceType | None = None,
            input_value: str | None = None,
            expected_success: bool = True,
            expected_error: str | None = None,
            extra_param: int = c.Tests.MIN_LENGTH_DEFAULT,
            description: str | None = None,
        ) -> p.Tests.ServiceTestCase:
            """Build a m.Tests.ServiceTestCase instance."""
            actual_type = service_type if service_type is not None else cls._next_type()
            actual_input = input_value if input_value is not None else cls._next_word()
            actual_description = (
                description
                if description is not None
                else f"Test case for {actual_type} with {actual_input}"
            )
            return m.Tests.ServiceTestCase(
                service_type=actual_type,
                input_value=actual_input,
                expected_success=expected_success,
                expected_error=expected_error,
                extra_param=extra_param,
                description=actual_description,
            )

        @classmethod
        def build_batch(cls, size: int) -> t.SequenceOf[p.Tests.ServiceTestCase]:
            """Build multiple m.Tests.ServiceTestCase instances with auto-generated values."""
            return [cls.build() for _ in range(size)]

        @classmethod
        def reset(cls) -> None:
            """Reset factory state."""
            cls._type_index = 0
            cls._word_index = 0

    class ServiceFactoryRegistry:
        """Registry for service factories using pattern matching."""

        @classmethod
        def create_service(
            cls,
            case: p.Tests.ServiceTestCase,
        ) -> (
            TestsFlextUtilitiesCaseServiceFactoriesMixin.GetUserService
            | TestsFlextUtilitiesCaseServiceFactoriesMixin.ValidatingService
            | TestsFlextUtilitiesCaseServiceFactoriesMixin.FailingService
        ):
            """Create appropriate service based on case type using pattern matching."""
            service: (
                TestsFlextUtilitiesCaseServiceFactoriesMixin.GetUserService
                | TestsFlextUtilitiesCaseServiceFactoriesMixin.ValidatingService
                | TestsFlextUtilitiesCaseServiceFactoriesMixin.FailingService
            )
            match case.service_type:
                case c.Tests.SERVICE_TEST_TYPE_GET_USER:
                    service = TestsFlextUtilitiesCaseServiceFactoriesMixin.GetUserServiceFactory.build(
                        user_id=case.input_value,
                    )
                case c.Tests.SERVICE_TEST_TYPE_VALIDATE:
                    service = TestsFlextUtilitiesCaseServiceFactoriesMixin.ValidatingServiceFactory.build(
                        value_input=case.input_value,
                        min_length=case.extra_param,
                    )
                case c.Tests.SERVICE_TEST_TYPE_FAIL:
                    service = TestsFlextUtilitiesCaseServiceFactoriesMixin.FailingServiceFactory.build(
                        error_message=case.input_value or c.Tests.DEFAULT_ERROR_MESSAGE,
                    )
                case _:
                    msg = f"Unsupported service type: {case.service_type}"
                    raise ValueError(msg)
            return service


__all__: list[str] = ["TestsFlextUtilitiesCaseServiceFactoriesMixin"]
