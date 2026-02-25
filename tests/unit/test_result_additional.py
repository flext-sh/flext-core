"""Additional coverage for flext_core.result using real flows (no mocks).

Exercises edge/error paths not covered in the base suite:
- ok(None) guard
- map_error identity/transform
- flow_through short-circuit on failure
- create_from_callable None/exception handling
- to_io_result failure path and __repr__
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from returns.io import IOFailure

from flext_core import r, t
from flext_tests import u


class ExplodingGetattr:
    """Mock IOSuccess that raises on getattr to exercise error guard.

    Note: Cannot inherit from IOSuccess as it's a final class from returns library.
    Instead, we create a mock that behaves similarly for testing purposes.
    """

    def __init__(self, value: int) -> None:
        """Initialize with a value."""
        self._value = value

    def __getattribute__(self, name: str) -> object:
        """Raise on attribute access except for class/dict/value."""
        # Allow __class__ and __dict__ for isinstance checks
        if name in {"__class__", "__dict__", "_value"}:
            return super().__getattribute__(name)
        # Raise on any other attribute access (like _inner_value)
        msg = "boom"
        raise RuntimeError(msg)


def test_ok_raises_on_none() -> None:
    """Ensure None successes are rejected."""
    with pytest.raises(ValueError):
        r[object].ok(None)


def test_map_error_identity_and_transform() -> None:
    """map_error should short-circuit on success and transform failures."""
    success = r[int].ok(1)
    assert success.map_error(lambda msg: msg + "_x") is success

    failure: r[int] = r[int].fail(
        "bad",
        error_code="E1",
        error_data=t.ConfigMap(root={"k": "v"}),
    )
    transformed = failure.map_error(lambda msg: f"{msg}_mapped")
    assert transformed.is_failure
    assert transformed.error is not None and "bad_mapped" in transformed.error
    assert transformed.error_code == "E1"
    assert transformed.error_data == {"k": "v"}


def test_flow_through_short_circuits_on_failure() -> None:
    """flow_through must stop when a step fails."""
    visited: list[int] = []

    def step1(v: int) -> r[int]:
        visited.append(v)
        return r[int].ok(v + 1)

    def fail_step(_: int) -> r[int]:
        return r[int].fail("stop")

    def unreachable(_: int) -> r[int]:
        visited.append(999)
        return r[int].ok(0)

    result = r[int].ok(1).flow_through(step1, fail_step, unreachable)
    assert result.is_failure
    assert result.error is not None and "stop" in result.error
    assert visited == [1]


def test_create_from_callable_and_repr() -> None:
    """Exercise callable None/exception branches and repr formatting."""
    none_callable: Callable[[], int] = cast("Callable[[], int]", lambda: None)
    none_result = r[int].create_from_callable(none_callable)
    u.Tests.Result.assert_result_failure(none_result)
    assert "Callable returned None" in (none_result.error or "")

    error_callable: Callable[[], int] = cast("Callable[[], int]", lambda: 1 / 0)
    error_result = r[int].create_from_callable(error_callable)
    u.Tests.Result.assert_result_failure(error_result)
    assert "division" in (error_result.error or "")

    success_result = r[int].create_from_callable(lambda: 7)
    assert repr(success_result) == "r.ok(7)"

    failure_repr: r[int] = r[int].fail("oops")
    assert repr(failure_repr) == "r.fail('oops')"


def test_to_io_result_failure_path() -> None:
    """Ensure failures produce IOFailure with propagated message."""
    failure: r[str] = r[str].fail("io_fail")
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
