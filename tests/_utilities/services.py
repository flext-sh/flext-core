"""Service helper classes for flext-core tests."""

from __future__ import annotations

from typing import Annotated, ClassVar, override

from flext_tests import r, u

from tests.base import s
from tests.constants import c
from tests.protocols import p
from tests.typings import t

from .railway_services import TestsFlextUtilitiesRailwayServicesMixin


class TestsFlextUtilitiesServicesMixin:
    """Service helper classes."""

    class MemoryCounter(p.Tests.Counter):
        """Real in-memory adapter of the ``p.Tests.Counter`` port."""

        def __init__(self) -> None:
            """Start the counter at zero."""
            self._value = 0

        @override
        def next_value(self) -> int:
            """Advance the counter and return its new value."""
            self._value += 1
            return self._value

    class CountingService(s[int]):
        """Service whose only collaborator is the ``p.Tests.Counter`` port."""

        counter: t.Port[p.Tests.Counter] = u.Field(
            exclude=True, description="Counter port the service advances."
        )

        @override
        def execute(self) -> p.Result[int]:
            """Advance the counter once and return the new value."""
            return r[int].ok(self.counter.next_value())

    class ValidatingService(s[str]):
        """Service with validation."""

        value_input: Annotated[
            str, u.Field(description="String input validated by business rules.")
        ]
        min_length: Annotated[
            int, u.Field(description="Minimum accepted input length.")
        ] = c.Tests.MIN_LENGTH_DEFAULT

        @override
        def execute(self) -> p.Result[str]:
            """Validate and return value."""
            if len(self.value_input) < self.min_length:
                return r[str].fail(
                    f"Value must be at least {self.min_length} characters"
                )
            return r[str].ok(self.value_input.upper())

    class GetUserServiceAuto(TestsFlextUtilitiesRailwayServicesMixin.GetUserService):
        """Auto-executing `GetUserService`."""

        auto_execute: ClassVar[bool] = True

    class ValidatingServiceAuto(ValidatingService):
        """Auto-executing `ValidatingService`."""

        auto_execute: ClassVar[bool] = True


__all__: list[str] = ["TestsFlextUtilitiesServicesMixin"]
