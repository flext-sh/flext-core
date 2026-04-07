"""Additional coverage for flext_core.result using real flows (no mocks).

Exercises edge/error paths not covered in the base suite:
- ok(None) guard
- map_error identity/transform
- flow_through short-circuit on failure
- create_from_callable None/exception handling
- __repr__ formatting
"""

from __future__ import annotations

from collections.abc import MutableSequence

from tests import r, t, u


def test_ok_accepts_none() -> None:
    """None is a valid success value when T includes None."""
    result = r[str | None].ok(None)
    assert result.is_success
    assert result.value is None


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
    assert transformed.error_data == t.ConfigMap(root={"k": "v"})


def test_flow_through_short_circuits_on_failure() -> None:
    """flow_through must stop when a step fails."""
    visited: MutableSequence[int] = []

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

    def none_callable() -> int | None:
        return None

    none_result = r[int].create_from_callable(none_callable)
    _ = u.Tests.assert_failure(none_result)
    assert "Callable returned None" in (none_result.error or "")
    error_message = "test error"

    def error_callable() -> int | None:
        raise ValueError(error_message)

    error_result = r[int].create_from_callable(error_callable)
    _ = u.Tests.assert_failure(error_result)
    assert error_message in (error_result.error or "")
    success_result = r[int].create_from_callable(lambda: 7)
    success_repr = repr(success_result)
    assert success_repr.startswith("r[T].ok(")
    assert "7" in success_repr
    failure_result: r[int] = r[int].fail("oops")
    failure_repr = repr(failure_result)
    assert failure_repr.startswith("r[T].fail(")
    assert "oops" in failure_repr


def test_with_resource_cleanup_runs() -> None:
    """with_resource should call cleanup even on success."""
    cleanup_calls: MutableSequence[str] = []

    def factory() -> MutableSequence[int]:
        return []

    def op(resource: MutableSequence[int]) -> r[str]:
        resource.append(1)
        return r[str].ok("done")

    def cleanup(resource: MutableSequence[int]) -> None:
        resource.clear()
        cleanup_calls.append("ran")

    result = r[str].with_resource(factory, op, cleanup)
    u.Tests.assert_success_with_value(result, "done")
    assert cleanup_calls == ["ran"]
