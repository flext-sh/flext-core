"""Recent result behavior tests."""

from __future__ import annotations

from collections.abc import MutableSequence

from flext_tests import tm

from tests import m, p, r
from tests.unit._result_scenarios import (
    ResultOperationType,
)


class TestsFlextResultRecentBehaviors:
    ResultOperationType = ResultOperationType

    def test_ok_accepts_valid_value(self) -> None:
        result = r[bool].ok(True)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)

    def test_map_error_transforms_failure_and_preserves_code(self) -> None:
        failure: p.Result[int] = r[int].fail(
            "bad",
            error_code="E1",
            error_data=m.ConfigMap(root={"k": "v"}),
        )
        transformed = failure.map_error(lambda msg: f"{msg}_mapped")
        tm.fail(transformed)
        tm.that(transformed.error, contains="bad_mapped")
        tm.that(transformed.error_code, eq="E1")

    def test_map_error_short_circuits_on_success(self) -> None:
        success = r[int].ok(1)
        tm.that(success.map_error(lambda msg: msg + "_x") is success, eq=True)

    def test_flow_through_stops_at_first_failure(self) -> None:
        visited: MutableSequence[int] = []

        def step1(v: int) -> p.Result[int]:
            visited.append(v)
            return r[int].ok(v + 1)

        def fail_step(_: int) -> p.Result[int]:
            return r[int].fail("stop")

        def unreachable(_: int) -> p.Result[int]:
            visited.append(999)
            return r[int].ok(0)

        result = r[int].ok(1).flow_through(step1, fail_step, unreachable)
        tm.fail(result)
        tm.that(result.error, contains="stop")
        tm.that(visited, eq=[1])

    def test_create_from_callable_handles_none_and_exception(self) -> None:
        def none_callable() -> int | None:
            return None

        none_result = r[int].create_from_callable(none_callable)
        tm.fail(none_result)
        tm.that(none_result.error, contains="Callable returned None")

        def error_callable() -> int | None:
            msg = "boom"
            raise ValueError(msg)

        error_result = r[int].create_from_callable(error_callable)
        tm.fail(error_result)
        tm.that(error_result.error, contains="boom")

    def test_with_resource_invokes_cleanup_after_success(self) -> None:
        cleanup_calls: MutableSequence[str] = []

        def factory() -> MutableSequence[int]:
            return []

        def op(resource: MutableSequence[int]) -> p.Result[str]:
            resource.append(1)
            return r[str].ok("done")

        def cleanup(resource: MutableSequence[int]) -> None:
            resource.clear()
            cleanup_calls.append("ran")

        result = r[str].with_resource(factory, op, cleanup)
        tm.ok(result)
        tm.that(result.value, eq="done")
        tm.that(cleanup_calls, eq=["ran"])
