"""Behavioral tests for recently added ``r[T]`` railway behaviors.

Every test asserts observable public contract of ``r[T]`` — success/failure
state, carried value/error/error_code, and the combinator surface
(map, map_error, flat_map, flow_through, create_from_callable, with_resource).
No private attributes, internal collaborators, or implementation details are
inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import m, r, tm

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreResultRecentBehaviors:
    """Public-contract behavior of the ``r[T]`` railway type."""

    @staticmethod
    def test_ok_carries_value_and_reports_success() -> None:
        """A success result reports success and exposes its value."""
        result = r[bool].ok(value=True)

        tm.that(result.success is True, truth=True)
        tm.that(result.value is True, truth=True)

    @staticmethod
    def test_map_error_transforms_message_and_preserves_error_code() -> None:
        """``map_error`` rewrites the failure message but keeps the error code."""
        failure: p.Result[int] = r[int].fail(
            "bad",
            error_code="E1",
            error_data=m.ConfigMap(root={"k": "v"}),
        )

        transformed = failure.map_error(lambda msg: f"{msg}_mapped")

        tm.that(transformed.failure is True, truth=True)
        error = tm.not_none(transformed.error)
        tm.that("bad_mapped" in error, truth=True)
        tm.that(transformed.error_code, eq="E1")

    @staticmethod
    def test_map_error_leaves_success_unchanged() -> None:
        """``map_error`` never runs its handler on a success result."""
        success = r[int].ok(1)

        transformed = success.map_error(lambda msg: f"{msg}_x")

        tm.that(transformed.success is True, truth=True)
        tm.that(transformed.value, eq=1)

    @staticmethod
    def test_flow_through_stops_at_first_failure() -> None:
        """``flow_through`` runs steps in order and halts at the first failure."""
        visited: list[int] = []

        def step1(value: int) -> p.Result[int]:
            visited.append(value)
            return r[int].ok(value + 1)

        def fail_step(_: int) -> p.Result[int]:
            return r[int].fail("stop")

        def unreachable(_: int) -> p.Result[int]:
            visited.append(999)
            return r[int].ok(0)

        result = r[int].ok(1).flow_through(step1, fail_step, unreachable)

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, none=False)
        tm.that("stop" in str(result.error), truth=True)
        tm.that(visited, eq=[1])

    @staticmethod
    def test_flow_through_threads_value_through_all_steps() -> None:
        """When every step succeeds, ``flow_through`` returns the final value."""

        def increment(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        def multiply(value: int) -> p.Result[int]:
            return r[int].ok(value * 10)

        result = r[int].ok(1).flow_through(increment, multiply)

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq=20)

    @staticmethod
    def test_create_from_callable_fails_when_result_is_none() -> None:
        """``create_from_callable`` fails when the callable returns ``None``."""
        result: p.Result[int] = r[int].create_from_callable(lambda: None)

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, none=False)
        tm.that("Callable returned None" in str(result.error), truth=True)

    @staticmethod
    def test_create_from_callable_captures_raised_exception() -> None:
        """A raised exception is captured as a failure carrying its message."""

        def raises_boom() -> int:
            msg = "boom"
            raise ValueError(msg)

        result = r[int].create_from_callable(raises_boom)

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, none=False)
        tm.that("boom" in str(result.error), truth=True)

    @staticmethod
    def test_create_from_callable_wraps_value_on_success() -> None:
        """A non-``None`` return becomes a success carrying that value."""
        result = r[int].create_from_callable(lambda: 42)

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq=42)

    @staticmethod
    def test_flat_map_into_success_value_reports_success() -> None:
        """Flat map can continue with non-`None` success payload."""
        result: p.Result[int] = r[str].ok("x").flat_map(lambda _: r[int].ok(1))

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq=1)

    @staticmethod
    def test_map_returning_none_reports_failure() -> None:
        """``map`` producing ``None`` fails closed instead of empty success."""
        result = r[str].ok("x").map(lambda _: None)

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, none=False)
        tm.that("cannot be None" in str(result.error), truth=True)

    @staticmethod
    def test_with_resource_returns_value_and_runs_cleanup() -> None:
        """``with_resource`` returns the op value and always runs cleanup."""
        cleanup_calls: list[str] = []

        def factory() -> list[int]:
            return []

        def op(resource: list[int]) -> p.Result[str]:
            resource.append(1)
            return r[str].ok("done")

        def cleanup(resource: list[int]) -> None:
            resource.clear()
            cleanup_calls.append("ran")

        result = r[str].with_resource(factory, op, cleanup)

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq="done")
        tm.that(cleanup_calls, eq=["ran"])
