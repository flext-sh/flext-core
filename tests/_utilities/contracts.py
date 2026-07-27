"""Text and external-service contract helpers for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_tests import r, tm, u
from tests.constants import c

if TYPE_CHECKING:
    from collections.abc import MutableSequence, Sequence

    from tests.protocols import p


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
        TestsFlextUtilitiesContractsMixin.Contract.assert_format_app_id(raw, expected)

    class FunctionalExternalService:
        """Mock external service for integration testing.

        Provides real functionality for testing service integration patterns
        with dependency injection and result handling.
        """

        def __init__(self) -> None:
            """Initialize external service with empty state."""
            self.processed_items: MutableSequence[str] = []
            self.call_count = 0

        def process(self, input_data: str) -> p.Result[str]:
            """Process input data by prefixing with 'processed_'.

            Args:
                input_data: String to process

            Returns:
                r[str]: Processed result or failure

            """
            try:
                self.call_count += 1
                processed = f"processed_{input_data}"
                self.processed_items.append(processed)
                return r[str].ok(processed)
            except (ValueError, TypeError, RuntimeError) as e:
                return r[str].fail(f"Processing failed: {e}")

        def get_call_count(self) -> int:
            """Get number of times process() was called."""
            return self.call_count


__all__: list[str] = ["TestsFlextUtilitiesContractsMixin"]
