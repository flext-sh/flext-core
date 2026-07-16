"""Railway case helpers for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.constants import c

if TYPE_CHECKING:
    from tests.typings import p, t


class TestsFlextUtilitiesRailwayCasesMixin:
    """Railway case helpers."""

    @staticmethod
    def success_cases() -> t.SequenceOf[tuple[str, str]]:
        return [
            (
                c.Tests.USER_IDS_SUCCESS[0],
                "Valid user ID",
            ),
            (
                c.Tests.USER_IDS_SUCCESS[1],
                "Another valid user ID",
            ),
            (
                c.Tests.USER_IDS_SUCCESS[2],
                "Third valid user ID",
            ),
        ]

    @staticmethod
    def failure_cases() -> t.SequenceOf[tuple[str, str, str]]:
        return [
            (
                "invalid",
                "not found",
                "Invalid user ID",
            ),
            (
                "",
                "not found",
                "Empty user ID",
            ),
        ]

    @staticmethod
    def railway_success_cases() -> t.SequenceOf[
        tuple[t.StrSequence, t.StrSequence, int, str]
    ]:
        return [
            (
                [c.Tests.USER_IDS_SUCCESS[0]],
                (),
                1,
                "Simple user retrieval",
            ),
            (
                [c.Tests.USER_IDS_SUCCESS[1]],
                [c.Tests.RAILWAY_OPERATION_GET_EMAIL],
                2,
                "User to email transformation",
            ),
            (
                [c.Tests.USER_IDS_SUCCESS[2]],
                [
                    c.Tests.RAILWAY_OPERATION_GET_EMAIL,
                    c.Tests.RAILWAY_OPERATION_SEND_EMAIL,
                    c.Tests.RAILWAY_OPERATION_GET_STATUS,
                ],
                4,
                "Full pipeline: user -> email -> send -> status",
            ),
        ]

    @staticmethod
    def multi_operation_cases() -> t.SequenceOf[tuple[str, int, t.JsonMapping]]:
        return [
            (
                c.Tests.RAILWAY_OPERATION_DOUBLE,
                5,
                {
                    c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_DOUBLE,
                    c.Tests.OPERATION_RESULT_KEY: 10,
                },
            ),
            (
                c.Tests.RAILWAY_OPERATION_SQUARE,
                4,
                {
                    c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_SQUARE,
                    c.Tests.OPERATION_RESULT_KEY: 16,
                },
            ),
            (
                c.Tests.RAILWAY_OPERATION_NEGATE,
                7,
                {
                    c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_NEGATE,
                    c.Tests.OPERATION_RESULT_KEY: -7,
                },
            ),
            (
                c.Tests.RAILWAY_OPERATION_DOUBLE,
                0,
                {
                    c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_DOUBLE,
                    c.Tests.OPERATION_RESULT_KEY: 0,
                },
            ),
            (
                c.Tests.RAILWAY_OPERATION_SQUARE,
                1,
                {
                    c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_SQUARE,
                    c.Tests.OPERATION_RESULT_KEY: 1,
                },
            ),
        ]


__all__: list[str] = ["TestsFlextUtilitiesRailwayCasesMixin"]
