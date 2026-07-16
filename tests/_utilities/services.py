"""Service helper classes for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, override

from flext_tests import r, u

from tests._utilities.railway_services import TestsFlextUtilitiesRailwayServicesMixin
from tests.base import s
from tests.constants import c

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextUtilitiesServicesMixin:
    """Service helper classes."""

    class ValidatingService(s[str]):
        """Service with validation."""

        value_input: Annotated[
            str,
            u.Field(description="String input validated by business rules."),
        ]
        min_length: Annotated[
            int,
            u.Field(description="Minimum accepted input length."),
        ] = c.Tests.MIN_LENGTH_DEFAULT

        @override
        def execute(self) -> p.Result[str]:
            """Validate and return value."""
            if len(self.value_input) < self.min_length:
                return r[str].fail(
                    f"Value must be at least {self.min_length} characters",
                )
            return r[str].ok(self.value_input.upper())

    class FailingService(s[str]):
        """Service that always fails."""

        error_message: Annotated[
            str,
            u.Field(description="Failure message emitted by execute()."),
        ] = c.Tests.DEFAULT_ERROR_MESSAGE

        @override
        def execute(self) -> p.Result[str]:
            """Fail unconditionally with the configured message."""
            return r[str].fail(self.error_message)

    class GetUserServiceAuto(TestsFlextUtilitiesRailwayServicesMixin.GetUserService):
        """Auto-executing `GetUserService`."""

        auto_execute: ClassVar[bool] = True

    class ValidatingServiceAuto(ValidatingService):
        """Auto-executing `ValidatingService`."""

        auto_execute: ClassVar[bool] = True

    class FailingServiceAuto(FailingService):
        """Auto-executing FailingService."""

        auto_execute: ClassVar[bool] = True


__all__: list[str] = ["TestsFlextUtilitiesServicesMixin"]
