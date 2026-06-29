"""Railway pipeline helpers for flext-core tests."""

from __future__ import annotations

from flext_tests import e, m as tm, r

from tests._utilities.railway_services import TestsFlextUtilitiesRailwayServicesMixin
from tests.constants import c
from tests.models import m
from tests.protocols import p


class TestsFlextUtilitiesRailwayPipelinesMixin(TestsFlextUtilitiesRailwayServicesMixin):
    """Railway pipeline helpers."""

    @staticmethod
    def execute_v1_pipeline(
        case: m.Tests.RailwayTestCase,
    ) -> p.Result[str | tm.Tests.User | m.Tests.EmailResponse]:
        """Execute the documented V1 railway pipeline."""
        if not case.user_ids:
            return r[str | tm.Tests.User | m.Tests.EmailResponse].fail(
                c.Tests.NO_USER_IDS_PROVIDED,
            )
        user_result: p.Result[tm.Tests.User] = (
            TestsFlextUtilitiesRailwayPipelinesMixin.make(
                TestsFlextUtilitiesRailwayPipelinesMixin.GetUserService,
                user_id=case.user_ids[0],
            ).execute()
        )
        result: p.Result[str | tm.Tests.User | m.Tests.EmailResponse] = user_result.map(
            lambda user: user
        )
        for operation in case.operations:
            if operation == "get_email":
                result = result.map(
                    lambda user: (
                        user.email if isinstance(user, tm.Tests.User) else str(user)
                    ),
                )
            elif operation == "send_email":
                email_result: p.Result[m.Tests.EmailResponse] = result.flat_map(
                    lambda email: TestsFlextUtilitiesRailwayPipelinesMixin.make(
                        TestsFlextUtilitiesRailwayPipelinesMixin.SendEmailService,
                        to=str(email),
                        subject="Test",
                    ).execute(),
                )
                result = email_result.map(lambda response: response)
            elif operation == "get_status":
                result = result.map(
                    lambda response: (
                        response.status
                        if isinstance(response, m.Tests.EmailResponse)
                        else str(response)
                    ),
                )
        return result

    @staticmethod
    def execute_v2_pipeline(
        case: m.Tests.RailwayTestCase,
    ) -> tm.Tests.User | str:
        """Execute the documented V2 railway pipeline."""
        if not case.user_ids:
            msg = c.Tests.NO_USER_IDS_PROVIDED
            raise e.BaseError(msg)
        raw_user_result = TestsFlextUtilitiesRailwayPipelinesMixin.make(
            TestsFlextUtilitiesRailwayPipelinesMixin.GetUserService,
            user_id=case.user_ids[0],
        ).execute()
        if raw_user_result.failure:
            msg = raw_user_result.error or c.Tests.USER_NOT_FOUND
            raise e.BaseError(msg)
        raw_user = raw_user_result.unwrap_or(None)
        if not isinstance(raw_user, tm.Tests.User):
            msg = c.Tests.USER_NOT_FOUND
            raise e.BaseError(msg)
        user: tm.Tests.User | str = raw_user
        for operation in case.operations:
            if operation == "get_email":
                user = user.email if isinstance(user, tm.Tests.User) else user
            elif operation == "send_email":
                email_to = user if isinstance(user, str) else str(user)
                raw_response_result = TestsFlextUtilitiesRailwayPipelinesMixin.make(
                    TestsFlextUtilitiesRailwayPipelinesMixin.SendEmailService,
                    to=email_to,
                    subject="Test",
                ).execute()
                if raw_response_result.failure:
                    msg = raw_response_result.error or c.Tests.INVALID_EMAIL
                    raise e.BaseError(msg)
                raw_response = raw_response_result.unwrap_or(None)
                if not isinstance(raw_response, m.Tests.EmailResponse):
                    msg = c.Tests.INVALID_EMAIL
                    raise e.BaseError(msg)
                response_obj: m.Tests.EmailResponse = raw_response
                user = response_obj.status
        return user


__all__: list[str] = ["TestsFlextUtilitiesRailwayPipelinesMixin"]
