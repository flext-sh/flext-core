"""Behavioral contract tests for the ``r`` / FlextResult public surface.

Every assertion here targets observable public behavior of a result value:
its success/failure state, the wrapped value, the error payload, and the
monadic combinators a caller composes. No private attribute is touched and
no internal collaborator is spied on.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import r
from tests import p


def _double(value: int) -> int:
    return value * 2


def _increment(value: int) -> p.Result[int]:
    return r[int].ok(value + 1)


def _shout(error: str) -> str:
    return error.upper()


def _forty_two(_error: str) -> int:
    return 42


def _zero(_error: str) -> int:
    return 0


class TestsFlextCoreDeprecationWarnings:
    """Public contract of ``r[T]`` success and failure results."""

    @staticmethod
    def test_ok_reports_success_state() -> None:
        # Arrange / Act
        """Test ok reports success state."""
        result: p.Result[str] = r[str].ok("value")

        # Assert
        assert result.success is True
        assert result.failure is False

    @staticmethod
    def test_fail_reports_failure_state() -> None:
        # Arrange / Act
        """Test fail reports failure state."""
        result: p.Result[str] = r[str].fail("deprecated")

        # Assert
        assert result.failure is True
        assert result.success is False

    @staticmethod
    def test_ok_exposes_wrapped_value() -> None:
        """Test ok exposes wrapped value."""
        result: p.Result[int] = r[int].ok(42)

        assert result.value == 42
        assert result.unwrap() == 42

    @staticmethod
    def test_fail_exposes_error_message() -> None:
        """Test fail exposes error message."""
        result: p.Result[int] = r[int].fail("boom")

        assert result.error == "boom"

    @staticmethod
    def test_unwrap_on_failure_raises() -> None:
        """Test unwrap on failure raises."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError):
            result.unwrap()

    @staticmethod
    def test_map_transforms_success_value() -> None:
        """Test map transforms success value."""
        result: p.Result[int] = r[int].ok(5).map(_double)

        assert result.success is True
        assert result.unwrap() == 10

    @staticmethod
    def test_map_is_skipped_on_failure() -> None:
        """Test map is skipped on failure."""
        result: p.Result[int] = r[int].fail("boom").map(_double)

        assert result.failure is True
        assert result.error == "boom"

    @staticmethod
    def test_flat_map_chains_fallible_success() -> None:
        """Test flat map chains fallible success."""
        result: p.Result[int] = r[int].ok(5).flat_map(_increment)

        assert result.unwrap() == 6

    @staticmethod
    def test_flat_map_short_circuits_on_failure() -> None:
        """Test flat map short circuits on failure."""
        result: p.Result[int] = r[int].fail("boom").flat_map(_increment)

        assert result.failure is True
        assert result.error == "boom"

    @staticmethod
    def test_map_error_transforms_failure_only() -> None:
        """Test map error transforms failure only."""
        failed: p.Result[int] = r[int].fail("boom").map_error(_shout)
        succeeded: p.Result[int] = r[int].ok(1).map_error(_shout)

        assert failed.error == "BOOM"
        assert succeeded.unwrap() == 1

    @staticmethod
    def test_recover_supplies_value_on_failure() -> None:
        """Test recover supplies value on failure."""
        result: p.Result[int] = r[int].fail("boom").recover(_forty_two)

        assert result.success is True
        assert result.unwrap() == 42

    @staticmethod
    def test_recover_leaves_success_untouched() -> None:
        """Test recover leaves success untouched."""
        result: p.Result[int] = r[int].ok(5).recover(_zero)

        assert result.unwrap() == 5

    @staticmethod
    def test_tap_observes_success_without_changing_value() -> None:
        """Test tap observes success without changing value."""
        seen: list[int] = []

        result: p.Result[int] = r[int].ok(5).tap(seen.append)

        assert seen == [5]
        assert result.unwrap() == 5

    @staticmethod
    def test_tap_error_runs_only_on_failure() -> None:
        """Test tap error runs only on failure."""
        failure_seen: list[str] = []
        success_seen: list[str] = []

        r[int].fail("boom").tap_error(failure_seen.append)
        r[int].ok(1).tap_error(success_seen.append)

        assert failure_seen == ["boom"]
        assert success_seen == []
