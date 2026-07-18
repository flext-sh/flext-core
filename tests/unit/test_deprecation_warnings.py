"""Behavioral contract tests for the ``r`` / FlextResult public surface.

Every assertion here targets observable public behavior of a result value:
its success/failure state, the wrapped value, the error payload, and the
monadic combinators a caller composes. No private attribute is touched and
no internal collaborator is spied on.
"""

from __future__ import annotations

import pytest
from flext_tests import r


class TestsFlextCoreDeprecationWarnings:
    """Public contract of ``r[T]`` success and failure results."""

    def test_ok_reports_success_state(self) -> None:
        # Arrange / Act
        result = r[str].ok("value")

        # Assert
        assert result.success is True
        assert result.failure is False

    def test_fail_reports_failure_state(self) -> None:
        # Arrange / Act
        result = r[str].fail("deprecated")

        # Assert
        assert result.failure is True
        assert result.success is False

    def test_ok_exposes_wrapped_value(self) -> None:
        result = r[int].ok(42)

        assert result.value == 42
        assert result.unwrap() == 42

    def test_fail_exposes_error_message(self) -> None:
        result = r[int].fail("boom")

        assert result.error == "boom"

    def test_unwrap_on_failure_raises(self) -> None:
        result = r[int].fail("boom")

        with pytest.raises(RuntimeError):
            result.unwrap()

    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [(r[int].ok(7), 99, 7), (r[int].fail("nope"), 99, 99)],
    )
    def test_unwrap_or_returns_value_or_default(
        self, result: r[int], default: int, expected: int
    ) -> None:
        assert result.unwrap_or(default) == expected

    def test_map_transforms_success_value(self) -> None:
        result = r[int].ok(5).map(lambda x: x * 2)

        assert result.success is True
        assert result.unwrap() == 10

    def test_map_is_skipped_on_failure(self) -> None:
        result = r[int].fail("boom").map(lambda x: x * 2)

        assert result.failure is True
        assert result.error == "boom"

    def test_flat_map_chains_fallible_success(self) -> None:
        result = r[int].ok(5).flat_map(lambda x: r[int].ok(x + 1))

        assert result.unwrap() == 6

    def test_flat_map_short_circuits_on_failure(self) -> None:
        result = r[int].fail("boom").flat_map(lambda x: r[int].ok(x + 1))

        assert result.failure is True
        assert result.error == "boom"

    def test_map_error_transforms_failure_only(self) -> None:
        failed = r[int].fail("boom").map_error(lambda e: e.upper())
        succeeded = r[int].ok(1).map_error(lambda e: e.upper())

        assert failed.error == "BOOM"
        assert succeeded.unwrap() == 1

    def test_recover_supplies_value_on_failure(self) -> None:
        result = r[int].fail("boom").recover(lambda _e: 42)

        assert result.success is True
        assert result.unwrap() == 42

    def test_recover_leaves_success_untouched(self) -> None:
        result = r[int].ok(5).recover(lambda _e: 0)

        assert result.unwrap() == 5

    def test_tap_observes_success_without_changing_value(self) -> None:
        seen: list[int] = []

        result = r[int].ok(5).tap(seen.append)

        assert seen == [5]
        assert result.unwrap() == 5

    def test_tap_error_runs_only_on_failure(self) -> None:
        failure_seen: list[str] = []
        success_seen: list[str] = []

        r[int].fail("boom").tap_error(failure_seen.append)
        r[int].ok(1).tap_error(success_seen.append)

        assert failure_seen == ["boom"]
        assert success_seen == []
