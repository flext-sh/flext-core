"""Railway pipeline helpers for flext-core tests."""

from __future__ import annotations

from typing import cast

from flext_tests import e, m as tm, r

from tests.constants import c
from tests.models import m
from tests.protocols import p

from .railway_services import TestsFlextUtilitiesRailwayServicesMixin


class TestsFlextUtilitiesRailwayPipelinesMixin(TestsFlextUtilitiesRailwayServicesMixin):
    """Railway pipeline helpers."""

    @staticmethod
    def execute_v1_pipeline(
        case: m.Tests.RailwayTestCase,
    ) -> p.ResultView[str | tm.Tests.User | m.Tests.EmailResponse]:
        """Execute the documented V1 railway pipeline."""
        if not case.user_ids:
            return cast(
                "p.ResultView[str | tm.Tests.User | m.Tests.EmailResponse]",
                r[str | tm.Tests.User | m.Tests.EmailResponse].fail(
                    c.Tests.NO_USER_IDS_PROVIDED
                ),
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

                def _get_email(
                    user: str | tm.Tests.User | m.Tests.EmailResponse,
                ) -> str:
                    return user.email if isinstance(user, tm.Tests.User) else str(user)

                result = result.map(_get_email)
            elif operation == "send_email":

                def _send(
                    email: str | tm.Tests.User | m.Tests.EmailResponse,
                ) -> p.Result[m.Tests.EmailResponse]:
                    to = email if isinstance(email, str) else str(email)
                    return TestsFlextUtilitiesRailwayPipelinesMixin.make(
                        TestsFlextUtilitiesRailwayPipelinesMixin.SendEmailService,
                        to=to,
                        subject="Test",
                    ).execute()

                email_result: p.Result[m.Tests.EmailResponse] = result.flat_map(_send)

                def _identity(response: m.Tests.EmailResponse) -> m.Tests.EmailResponse:
                    return response

                result = email_result.map(_identity)
            elif operation == "get_status":

                def _get_status(
                    response: str | tm.Tests.User | m.Tests.EmailResponse,
                ) -> str:
                    return (
                        response.status
                        if isinstance(response, m.Tests.EmailResponse)
                        else str(response)
                    )

                result = result.map(_get_status)
        return cast("p.ResultView[str | tm.Tests.User | m.Tests.EmailResponse]", result)

    @staticmethod
    def execute_v2_pipeline(case: m.Tests.RailwayTestCase) -> tm.Tests.User | str:
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
        user: tm.Tests.User | str = raw_user_result.value
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
                # The service's result payload is EmailResponse by contract;
                # pyright proves the isinstance guard redundant here.
                response_obj: m.Tests.EmailResponse = raw_response_result.value
                user = response_obj.status
        return user


__all__: list[str] = ["TestsFlextUtilitiesRailwayPipelinesMixin"]
