"""Behavioral tests for recently added ``r[T]`` railway behaviors.

Every test asserts observable public contract of ``r[T]`` — success/failure
state, carried value/error/error_code, and the combinator surface
(map, map_error, flat_map, flow_through, create_from_callable, with_resource).
No private attributes, internal collaborators, or implementation details are
inspected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import r

from tests import m

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreResultRecentBehaviors:
    """Public-contract behavior of the ``r[T]`` railway type."""

    def test_ok_carries_value_and_reports_success(self) -> None:
        """A success result reports success and exposes its value."""
        result = r[bool].ok(True)

        assert result.success is True
        assert result.value is True

    def test_map_error_transforms_message_and_preserves_error_code(self) -> None:
        """``map_error`` rewrites the failure message but keeps the error code."""
        failure: p.Result[int] = r[int].fail(
            "bad",
            error_code="E1",
            error_data=m.ConfigMap(root={"k": "v"}),
        )

        transformed = failure.map_error(lambda msg: f"{msg}_mapped")

        assert transformed.failure is True
        assert transformed.error is not None
        assert "bad_mapped" in transformed.error
        assert transformed.error_code == "E1"

    def test_map_error_leaves_success_unchanged(self) -> None:
        """``map_error`` never runs its handler on a success result."""
        success = r[int].ok(1)

        transformed = success.map_error(lambda msg: f"{msg}_x")

        assert transformed.success is True
        assert transformed.value == 1

    def test_flow_through_stops_at_first_failure(self) -> None:
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

        assert result.failure is True
        assert result.error is not None
        assert "stop" in result.error
        assert visited == [1]

    def test_flow_through_threads_value_through_all_steps(self) -> None:
        """When every step succeeds, ``flow_through`` returns the final value."""
        result = (
            r[int]
            .ok(1)
            .flow_through(
                lambda v: r[int].ok(v + 1),
                lambda v: r[int].ok(v * 10),
            )
        )

        assert result.success is True
        assert result.value == 20

    def test_create_from_callable_fails_when_result_is_none(self) -> None:
        """``create_from_callable`` fails when the callable returns ``None``."""
        result: p.Result[int] = r[int].create_from_callable(lambda: None)

        assert result.failure is True
        assert result.error is not None
        assert "Callable returned None" in result.error

    def test_create_from_callable_captures_raised_exception(self) -> None:
        """A raised exception is captured as a failure carrying its message."""

        def raises_boom() -> int:
            msg = "boom"
            raise ValueError(msg)

        result = r[int].create_from_callable(raises_boom)

        assert result.failure is True
        assert result.error is not None
        assert "boom" in result.error

    def test_create_from_callable_wraps_value_on_success(self) -> None:
        """A non-``None`` return becomes a success carrying that value."""
        result = r[int].create_from_callable(lambda: 42)

        assert result.success is True
        assert result.value == 42

    def test_flat_map_into_none_success_reports_success(self) -> None:
        """Chaining ``flat_map`` into ``r[None].ok(None)`` yields a success."""
        result: p.Result[None] = r[str].ok("x").flat_map(lambda _: r[None].ok(None))

        assert result.success is True

    def test_map_returning_none_reports_success(self) -> None:
        """``map`` producing ``None`` yields a success with a ``None`` payload."""
        result: p.Result[None] = r[str].ok("x").map(lambda _: None)

        assert result.success is True

    def test_with_resource_returns_value_and_runs_cleanup(self) -> None:
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

        assert result.success is True
        assert result.value == "done"
        assert cleanup_calls == ["ran"]
