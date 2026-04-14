"""FLEXT Core Reliability Utilities Tests - Comprehensive Coverage.

Tests for flext_core.FlextUtilitiesReliability covering:
- Retry mechanisms (success after failure, parameter validation)

Modules tested: flext_core.FlextUtilitiesReliability
Scope: Reliability utility methods with 100% coverage including edge cases

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from typing import Final

from tests import p, r, u


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
        ) -> tuple[Callable[[], p.Result[int]], Sequence[int]]:
            """Create retry operation that succeeds after N attempts."""
            attempts: MutableSequence[int] = []

            def op() -> p.Result[int]:
                attempts.append(len(attempts))
                if len(attempts) >= success_after:
                    return r[int].ok(success_value)
                return r[int].fail("transient")

            return (op, attempts)

    def test_retry_succeeds_after_failure(self) -> None:
        """Test retry succeeds after initial failure."""
        op, attempts = self.Factories.create_retry_operation(success_after=2)
        result: p.Result[int] = u.retry(
            op,
            max_attempts=self.Constants.MAX_ATTEMPTS_VALID,
            delay_seconds=0.0,
        )
        assert result.success
        assert result.value == self.Constants.SUCCESS_VALUE
        assert len(attempts) == 2

    def test_retry_validation_error(self) -> None:
        """Test retry parameter validation."""
        result: p.Result[int] = u.retry(
            lambda: p.Result[int].fail("fail"),
            max_attempts=self.Constants.MAX_ATTEMPTS_INVALID,
        )
        assert result.failure
        assert "Max attempts must be at least" in (result.error or "")
