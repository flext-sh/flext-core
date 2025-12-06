"""Additional coverage for flext_core.result using real flows (no mocks).

Exercises edge/error paths not covered in the base suite:
- ok(None) guard
- IOResult conversions (invalid/exception paths)
- map_error identity/transform
- flow_through short-circuit on failure
- parallel_map fail-fast vs accumulate
- create_from_callable None/exception handling
- to_io_result failure path and __repr__
"""

from __future__ import annotations

from typing import cast

import pytest
from returns.io import IOFailure, IOResult, IOSuccess

from flext_core import r, t
from flext_tests import u


class ExplodingGetattr(IOSuccess[int]):
    """IOSuccess that raises on getattr to exercise error guard."""

    def __getattribute__(self, name: str) -> object:
        """Raise on attribute access except for class/dict."""
        # Allow __class__ and isinstance checks
        if name in {"__class__", "__dict__"}:
            return super().__getattribute__(name)
        # Raise on any other attribute access (like _inner_value)
        msg = "boom"
        raise RuntimeError(msg)


def test_ok_raises_on_none() -> None:
    """Ensure None successes are rejected."""
    with pytest.raises(ValueError):
        r[object].ok(None)


def test_from_io_result_invalid_type() -> None:
    """Cover invalid IOResult type path."""
    invalid: IOResult[t.GeneralValueType, str] = cast(
        "IOResult[t.GeneralValueType, str]",
        "not_io_result",
    )
    invalid_result = r.from_io_result(invalid)
    u.Tests.Result.assert_result_failure(invalid_result)
    assert "Invalid IO result type" in (invalid_result.error or "")


def test_map_error_identity_and_transform() -> None:
    """map_error should short-circuit on success and transform failures."""
    success = r[int].ok(1)
    assert success.map_error(lambda msg: msg + "_x") is success

    failure = r[int].fail("bad", error_code="E1", error_data={"k": "v"})
    transformed = failure.map_error(lambda msg: f"{msg}_mapped")
    u.Tests.Result.assert_failure_with_error(
        transformed,
        "bad_mapped",
    )
    assert transformed.error_code == "E1"
    assert transformed.error_data == {"k": "v"}


def test_flow_through_short_circuits_on_failure() -> None:
    """flow_through must stop when a step fails."""
    visited: list[int] = []

    def step1(value: int) -> r[int]:
        visited.append(value)
        return r[int].ok(value + 1)

    def fail_step(_: int) -> r[int]:
        return r[int].fail("stop")

    def unreachable(_: int) -> r[int]:
        visited.append(999)
        return r[int].ok(0)

    result = r[int].ok(1).flow_through(step1, fail_step, unreachable)
    u.Tests.Result.assert_failure_with_error(result, "stop")
    assert visited == [1]


def test_parallel_map_fail_fast_and_accumulate() -> None:
    """Cover both fail-fast and accumulate_errors branches."""
    items = [1, 2]

    def fn(value: int) -> r[int]:
        if value == 2:
            return r[int].fail("boom")
        return r[int].ok(value * 2)

    fast = r.parallel_map(items, fn, fail_fast=True)
    u.Tests.Result.assert_result_failure(fast)
    assert "boom" in (fast.error or "")

    accumulated = r.parallel_map(items, fn, fail_fast=False)
    u.Tests.Result.assert_result_failure(accumulated)
    assert "boom" in (accumulated.error or "")

    all_success = r.parallel_map([3, 4], lambda v: r[int].ok(v + 1))
    u.Tests.Result.assert_success_with_value(
        all_success,
        [4, 5],
    )


def test_create_from_callable_and_repr() -> None:
    """Exercise callable None/exception branches and repr formatting."""
    none_result = r[int].create_from_callable(lambda: None)
    u.Tests.Result.assert_result_failure(none_result)
    assert "Callable returned None" in (none_result.error or "")

    error_result = r[int].create_from_callable(lambda: 1 / 0)
    u.Tests.Result.assert_result_failure(error_result)
    assert "division" in (error_result.error or "")

    success_result = r[int].create_from_callable(lambda: 7)
    assert repr(success_result) == "r.ok(7)"

    failure_repr = r[int].fail("oops")
    assert repr(failure_repr) == "r.fail('oops')"


def test_to_io_result_failure_path() -> None:
    """Ensure failures produce IOFailure with propagated message."""
    failure = r[str].fail("io_fail")
    io_result = failure.to_io_result()
    assert isinstance(io_result, IOFailure)
    # IOFailure.__str__ includes the IO wrapper formatting
    assert "io_fail" in str(io_result)


def test_with_resource_cleanup_runs() -> None:
    """with_resource should call cleanup even on success."""
    cleanup_calls: list[str] = []

    def factory() -> list[int]:
        return []

    def op(resource: list[int]) -> r[str]:
        resource.append(1)
        return r[str].ok("done")

    def cleanup(resource: list[int]) -> None:
        resource.clear()
        cleanup_calls.append("ran")

    result = r[str].with_resource(factory, op, cleanup)
    u.Tests.Result.assert_success_with_value(result, "done")
    assert cleanup_calls == ["ran"]


def test_data_alias_matches_value() -> None:
    """Confirm data alias returns same value property."""
    success = u.Tests.Result.create_success_result("v")
    assert success.data == success.value == "v"
