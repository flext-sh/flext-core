"""Behavioral contract tests for the ``r`` / FlextResult public surface.

Every assertion here targets observable public behavior of a result value:
its success/failure state, the wrapped value, the error payload, and the
monadic combinators a caller composes. No private attribute is touched and
no internal collaborator is spied on.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import pytest

from tests import p, r, tm, u


class TestsFlextCoreDeprecationWarnings:
    """Public contract of ``r[T]`` success and failure results."""

    @staticmethod
    def test_ok_reports_success_state() -> None:
        """A successful result reports success and never failure."""
        result: p.Result[str] = r[str].ok("value")

        tm.ok(result.success)
        tm.fail(result.failure)

    @staticmethod
    def test_fail_reports_failure_state() -> None:
        """A failed result reports failure and never success."""
        result: p.Result[str] = r[str].fail("deprecated")

        tm.ok(result.failure)
        tm.fail(result.success)

    @staticmethod
    def test_ok_exposes_wrapped_value() -> None:
        """A successful result exposes its wrapped value."""
        result: p.Result[int] = r[int].ok(42)

        tm.that(result.value, eq=42)
        tm.that(result.unwrap(), eq=42)

    @staticmethod
    def test_fail_exposes_error_message() -> None:
        """A failed result exposes its error message."""
        result: p.Result[int] = r[int].fail("boom")

        tm.that(result.error, eq="boom")

    @staticmethod
    def test_unwrap_on_failure_raises() -> None:
        """Unwrapping a failed result raises instead of returning a value."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError, match="boom"):
            result.unwrap()

    @staticmethod
    def test_map_transforms_success_value() -> None:
        """Map applies the callable to a successful value."""
        result: p.Result[int] = r[int].ok(5).map(u.Tests.double)

        tm.ok(result.success)
        tm.that(result.unwrap(), eq=10)

    @staticmethod
    def test_map_is_skipped_on_failure() -> None:
        """Map keeps the failure untouched when the result failed."""
        result: p.Result[int] = r[int].fail("boom").map(u.Tests.double)

        tm.ok(result.failure)
        tm.that(result.error, eq="boom")

    @staticmethod
    def test_flat_map_chains_fallible_success() -> None:
        """flat_map chains the fallible callable on success."""
        result: p.Result[int] = r[int].ok(5).flat_map(u.Tests.increment)

        tm.that(result.unwrap(), eq=6)

    @staticmethod
    def test_flat_map_short_circuits_on_failure() -> None:
        """flat_map keeps the failure untouched when the result failed."""
        result: p.Result[int] = r[int].fail("boom").flat_map(u.Tests.increment)

        tm.ok(result.failure)
        tm.that(result.error, eq="boom")

    @staticmethod
    def test_map_error_transforms_failure_only() -> None:
        """map_error transforms only the failure payload."""
        failed: p.Result[int] = r[int].fail("boom").map_error(u.Tests.shout)
        succeeded: p.Result[int] = r[int].ok(1).map_error(u.Tests.shout)

        tm.that(failed.error, eq="BOOM")
        tm.that(succeeded.unwrap(), eq=1)

    @staticmethod
    def test_recover_supplies_value_on_failure() -> None:
        """Recover supplies the callable's value on failure."""
        result: p.Result[int] = r[int].fail("boom").recover(u.Tests.forty_two)

        tm.ok(result.success)
        tm.that(result.unwrap(), eq=42)

    @staticmethod
    def test_recover_leaves_success_untouched() -> None:
        """Recover keeps the successful value untouched."""
        result: p.Result[int] = r[int].ok(5).recover(u.Tests.zero)

        tm.that(result.unwrap(), eq=5)

    @staticmethod
    def test_tap_observes_success_without_changing_value() -> None:
        """Tap observes a successful value without changing it."""
        seen: list[int] = []

        result: p.Result[int] = r[int].ok(5).tap(seen.append)

        tm.that(seen, eq=[5])
        tm.that(result.unwrap(), eq=5)

    @staticmethod
    def test_tap_error_runs_only_on_failure() -> None:
        """tap_error observes only failures."""
        failure_seen: list[str] = []
        success_seen: list[str] = []

        r[int].fail("boom").tap_error(failure_seen.append)
        r[int].ok(1).tap_error(success_seen.append)

        tm.that(failure_seen, eq=["boom"])
        tm.that(success_seen, eq=[])
