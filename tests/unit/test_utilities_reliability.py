"""FLEXT Core Reliability Utilities Tests - Comprehensive Coverage.

Tests for flext_core._utilities.reliability.FlextUtilitiesReliability covering:
- Retry mechanisms (success after failure, parameter validation)

Modules tested: flext_core._utilities.reliability.FlextUtilitiesReliability
Scope: Reliability utility methods with 100% coverage including edge cases

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from typing import Final

from flext_core import r
from tests import u


class TestFlextUtilitiesReliability:
    """Tests for FlextUtilitiesReliability retry methods."""

    class Constants:
        """Test constants for reliability utilities."""

        MAX_ATTEMPTS_VALID: Final[int] = 3
        MAX_ATTEMPTS_INVALID: Final[int] = 0
        SUCCESS_VALUE: Final[int] = 42

    class Factories:
        """Factories for creating test operations."""

        @staticmethod
        def create_retry_operation(
            success_after: int,
            success_value: int = 42,
        ) -> tuple[Callable[[], r[int]], Sequence[int]]:
            """Create retry operation that succeeds after N attempts."""
            attempts: MutableSequence[int] = []

            def op() -> r[int]:
                attempts.append(len(attempts))
                if len(attempts) >= success_after:
                    return r[int].ok(success_value)
                return r[int].fail("transient")

            return (op, attempts)

    def test_retry_succeeds_after_failure(self) -> None:
        """Test retry succeeds after initial failure."""
        op, attempts = self.Factories.create_retry_operation(success_after=2)
        result: r[int] = u.retry(
            op,
            max_attempts=self.Constants.MAX_ATTEMPTS_VALID,
            delay_seconds=0.0,
        )
        u.Tests.Result.assert_success_with_value(result, self.Constants.SUCCESS_VALUE)
        assert len(attempts) == 2

    def test_retry_validation_error(self) -> None:
        """Test retry parameter validation."""
        result: r[int] = u.retry(
            lambda: r[int].fail("fail"),
            max_attempts=self.Constants.MAX_ATTEMPTS_INVALID,
        )
        _ = u.Tests.Result.assert_failure(result)
        assert "Max attempts must be at least" in (result.error or "")
