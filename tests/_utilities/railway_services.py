"""Railway service helpers for flext-core tests."""

from __future__ import annotations

from typing import Annotated, override

from flext_tests import m as tm, r, u
from tests import c, m, p, t
from tests._models.mixins import TestsFlextModelsMixins
from tests.base import s


class TestsFlextUtilitiesRailwayServicesMixin:
    """Railway service helpers."""

    class GetUserService(s[t.JsonMapping]):
        """Service to get user."""

        user_id: Annotated[
            str, u.Field(description="Identifier of the user to fetch.")
        ] = ""

        @override
        def execute(self) -> p.Result[tm.Tests.User]:
            if self.user_id in c.Tests.USER_IDS_INVALID:
                return r[tm.Tests.User].fail(c.Tests.USER_NOT_FOUND)
            return r[tm.Tests.User].ok(
                tm.Tests.User(
                    id=self.user_id,
                    unique_id=self.user_id,
                    name=f"{c.Tests.DEFAULT_USER_NAME_PREFIX}{self.user_id}",
                    email=f"user{self.user_id}{c.Tests.DEFAULT_EMAIL_DOMAIN}",
                )
            )

    class SendEmailService(s[t.JsonMapping]):
        """Service to send email."""

        to: Annotated[str, u.Field(description="Destination email address.")] = ""
        subject: Annotated[str, u.Field(description="Email subject line.")] = ""

        @override
        def execute(self) -> p.Result[p.Tests.EmailResponse]:
            if "@" not in self.to:
                return r[p.Tests.EmailResponse].fail(c.Tests.INVALID_EMAIL)
            return r[p.Tests.EmailResponse].ok(
                m.Tests.EmailResponse(status="sent", message_id=f"msg-{self.to}")
            )

    class ValidationService(s[t.JsonMapping]):
        """Service to validate values."""

        value: Annotated[int, u.Field(description="Integer value to validate.")] = 0

        @override
        def execute(self) -> p.Result[t.JsonMapping]:
            if self.value < 0:
                return r[t.JsonMapping].fail(c.Tests.VALUE_TOO_LOW)
            if self.value > c.Tests.MAX_VALUE:
                return r[t.JsonMapping].fail(c.Tests.VALUE_TOO_HIGH)
            return r[t.JsonMapping].ok({"valid": True, "value": self.value})

    class MultiOperationService(s[t.JsonMapping]):
        """Service for multiple operations."""

        operation: Annotated[
            str, u.Field(description="Operation name (double / square / ...).")
        ] = ""
        value: Annotated[
            int, u.Field(description="Numeric operand for the operation.")
        ] = 0

        @override
        def execute(self) -> p.Result[t.JsonMapping]:
            match self.operation:
                case c.Tests.RAILWAY_OPERATION_DOUBLE:
                    return r[t.JsonMapping].ok({
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_DOUBLE,
                        c.Tests.OPERATION_RESULT_KEY: self.value
                        * c.Tests.OPERATION_FACTORS[c.Tests.RAILWAY_OPERATION_DOUBLE],
                    })
                case c.Tests.RAILWAY_OPERATION_SQUARE:
                    return r[t.JsonMapping].ok({
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_SQUARE,
                        c.Tests.OPERATION_RESULT_KEY: self.value**2,
                    })
                case c.Tests.RAILWAY_OPERATION_NEGATE:
                    return r[t.JsonMapping].ok({
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_NEGATE,
                        c.Tests.OPERATION_RESULT_KEY: self.value
                        * c.Tests.OPERATION_FACTORS[c.Tests.RAILWAY_OPERATION_NEGATE],
                    })
                case _:
                    return r[t.JsonMapping].fail(
                        f"{c.Tests.UNKNOWN_OPERATION_PREFIX} {self.operation}"
                    )

    @staticmethod
    def value_lt_100(data: t.JsonMapping) -> bool:
        target: TestsFlextModelsMixins._TargetModel = (
            TestsFlextModelsMixins._TargetModel.model_validate(data)
        )
        upper_bound = 100
        return target.value < upper_bound

    @staticmethod
    def make[T](service_type: type[T], **kwargs: t.Scalar) -> T:
        return service_type(**kwargs)

    @staticmethod
    def create_user_service(
        case: p.Tests.ServiceTestCase,
    ) -> TestsFlextUtilitiesRailwayServicesMixin.GetUserService:
        """Create a user service from a documented service case."""
        return TestsFlextUtilitiesRailwayServicesMixin.make(
            TestsFlextUtilitiesRailwayServicesMixin.GetUserService,
            user_id=case.user_id or case.input_value or "",
        )


__all__: list[str] = ["TestsFlextUtilitiesRailwayServicesMixin"]
