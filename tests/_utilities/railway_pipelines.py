"""Railway pipeline helpers for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_tests import e, r

from tests import c, m, p
from tests._utilities import TestsFlextUtilitiesRailwayServices

type _V1Payload = str | p.Tests.User | p.Tests.EmailResponse


class TestsFlextUtilitiesRailwayPipelines(TestsFlextUtilitiesRailwayServices):
    """Railway pipeline helpers."""

    @staticmethod
    def _v1_email_view(value: _V1Payload) -> str:
        """Project one V1 pipeline payload onto its email view.

        Returns:
            The resulting ``str``.

        """
        return value.email if isinstance(value, p.Tests.User) else str(value)

    @staticmethod
    def _v1_send_email(value: _V1Payload) -> p.Result[p.Tests.EmailResponse]:
        """Send one V1 pipeline email for the current payload.

        Returns:
            The resulting ``p.Result[p.Tests.EmailResponse]``.

        """
        to = value if isinstance(value, str) else str(value)
        return TestsFlextUtilitiesRailwayPipelines.make(
            TestsFlextUtilitiesRailwayPipelines.SendEmailService,
            to=to,
            subject="Test",
        ).execute()

    @staticmethod
    def _v1_identity_response(
        response: p.Tests.EmailResponse,
    ) -> p.Tests.EmailResponse:
        """Return one V1 pipeline email response unchanged."""
        return response

    @staticmethod
    def _v1_status_view(value: _V1Payload) -> str:
        """Project one V1 pipeline payload onto its status view.

        Returns:
            The resulting ``str``.

        """
        return value.status if isinstance(value, p.Tests.EmailResponse) else str(value)

    @staticmethod
    def execute_v1_pipeline(
        case: p.Tests.RailwayTestCase,
    ) -> p.ResultView[str | p.Tests.User | p.Tests.EmailResponse]:
        """Execute the documented V1 railway pipeline.

        Returns:
            The resulting ``p.ResultView[str | p.Tests.User | p.Tests.EmailResponse]``.

        """
        if not case.user_ids:
            return cast(
                "p.ResultView[str | p.Tests.User | p.Tests.EmailResponse]",
                r[str | p.Tests.User | p.Tests.EmailResponse].fail(
                    c.Tests.NO_USER_IDS_PROVIDED,
                ),
            )
        user_result: p.Result[p.Tests.User] = TestsFlextUtilitiesRailwayPipelines.make(
            TestsFlextUtilitiesRailwayPipelines.GetUserService,
            user_id=case.user_ids[0],
        ).execute()
        result: p.Result[str | p.Tests.User | p.Tests.EmailResponse] = user_result.map(
            lambda user: user,
        )
        for operation in case.operations:
            if operation == "get_email":
                result = result.map(TestsFlextUtilitiesRailwayPipelines._v1_email_view)
            elif operation == "send_email":
                result = result.flat_map(
                    TestsFlextUtilitiesRailwayPipelines._v1_send_email
                ).map(TestsFlextUtilitiesRailwayPipelines._v1_identity_response)
            elif operation == "get_status":
                result = result.map(TestsFlextUtilitiesRailwayPipelines._v1_status_view)
        return cast("p.ResultView[str | p.Tests.User | p.Tests.EmailResponse]", result)

    @staticmethod
    def execute_v2_pipeline(case: p.Tests.RailwayTestCase) -> p.Tests.User | str:
        """Execute the documented V2 railway pipeline.

        Returns:
            The resulting ``p.Tests.User | str``.

        Raises:
            BaseError: If ``not case.user_ids``; or if ``raw_user_result.failure``; or
                if ``raw_response_result.failure``.

        """
        if not case.user_ids:
            msg = c.Tests.NO_USER_IDS_PROVIDED
            raise e.BaseError(msg)
        raw_user_result = TestsFlextUtilitiesRailwayPipelines.make(
            TestsFlextUtilitiesRailwayPipelines.GetUserService,
            user_id=case.user_ids[0],
        ).execute()
        if raw_user_result.failure:
            msg = raw_user_result.error or c.Tests.USER_NOT_FOUND
            raise e.BaseError(msg)
        user: p.Tests.User | str = raw_user_result.value
        for operation in case.operations:
            if operation == "get_email":
                user = user.email if isinstance(user, p.Tests.User) else user
            elif operation == "send_email":
                email_to = user if isinstance(user, str) else str(user)
                raw_response_result = TestsFlextUtilitiesRailwayPipelines.make(
                    TestsFlextUtilitiesRailwayPipelines.SendEmailService,
                    to=email_to,
                    subject="Test",
                ).execute()
                if raw_response_result.failure:
                    msg = raw_response_result.error or c.Tests.INVALID_EMAIL
                    raise e.BaseError(msg)
                # The service's result payload is EmailResponse by contract;
                # pyright proves the isinstance guard redundant here.
                response_obj: p.Tests.EmailResponse = raw_response_result.value
                user = response_obj.status
        return user


__all__: list[str] = ["TestsFlextUtilitiesRailwayPipelines"]
