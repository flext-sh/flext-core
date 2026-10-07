"""Service case construction helpers for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from tests._utilities.service_factories import TestsFlextUtilitiesServiceFactoriesMixin
from tests.constants import c
from tests.models import m

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.typings import t


class TestsFlextUtilitiesCaseServiceFactoriesMixin(
    TestsFlextUtilitiesServiceFactoriesMixin,
):
    """Service case construction helpers."""

    class ServiceTestCaseFactory(TestsFlextUtilitiesServiceFactoriesMixin.WordRotation):
        """Factory for m.Tests.ServiceTestCase."""

        _service_types: ClassVar[Sequence[c.Tests.ServiceType]] = [
            c.Tests.SERVICE_TEST_TYPE_GET_USER,
            c.Tests.SERVICE_TEST_TYPE_VALIDATE,
            c.Tests.SERVICE_TEST_TYPE_FAIL,
        ]
        _type_index: ClassVar[int] = 0
        _words: ClassVar[Sequence[str]] = ["test", "sample", "example", "demo", "data"]
        _word_index: ClassVar[int] = 0

        @classmethod
        def _next_type(cls) -> c.Tests.ServiceType:
            """Get next service type from rotation.

            Returns:
                The resulting ``c.Tests.ServiceType``.

            """
            service_type = cls._service_types[cls._type_index % len(cls._service_types)]
            cls._type_index += 1
            return service_type

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
        ) -> m.Tests.ServiceTestCase:
            """Build a m.Tests.ServiceTestCase instance.

            Returns:
                The resulting ``m.Tests.ServiceTestCase``.

            """
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
        def build_batch(cls, size: int) -> t.SequenceOf[m.Tests.ServiceTestCase]:
            """Build multiple ServiceTestCase instances with auto-generated values.

            Returns:
                The resulting ``t.SequenceOf[m.Tests.ServiceTestCase]``.

            """
            return [cls.build() for _ in range(size)]

        @classmethod
        @override
        def reset(cls) -> None:
            """Reset factory state."""
            cls._type_index = 0
            super().reset()


__all__: list[str] = ["TestsFlextUtilitiesCaseServiceFactoriesMixin"]
