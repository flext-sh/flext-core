"""Text and external-service contract helpers for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_tests import tm, u

from tests.constants import c

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextUtilitiesContractsMixin:
    """Text and external-service contract helpers."""

    class Contract:
        """Shared contract for text utility behavior."""

        SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = (
            c.Tests.CORE_SAFE_STRING_VALID_CASES
        )
        SAFE_STRING_INVALID_CASES: ClassVar[Sequence[tuple[str | None, str]]] = (
            c.Tests.CORE_SAFE_STRING_INVALID_CASES
        )
        FORMAT_APP_ID_CASES: ClassVar[Sequence[tuple[str, str]]] = (
            c.Tests.CORE_FORMAT_APP_ID_CASES
        )

        @staticmethod
        def assert_safe_string_valid(raw: str, expected: str) -> None:
            """Assert safe string normalization for valid input."""
            tm.that(u.safe_string(raw), eq=expected)

        @staticmethod
        def assert_format_app_id(raw: str, expected: str) -> None:
            """Assert app id formatting behavior."""
            tm.that(u.format_app_id(raw), eq=expected)

    @staticmethod
    def assert_safe_string_valid(raw: str, expected: str) -> None:
        """Assert safe string normalization for valid input."""
        TestsFlextUtilitiesContractsMixin.Contract.assert_safe_string_valid(
            raw, expected
        )

    @staticmethod
    def assert_format_app_id(raw: str, expected: str) -> None:
        """Assert app id formatting behavior."""
        tm.that(u.format_app_id(raw), eq=expected)


__all__: list[str] = ["TestsFlextUtilitiesContractsMixin"]
