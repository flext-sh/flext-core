"""Result traversal and resource tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import r, tm

from tests.models import m
from tests.unit._result_scenarios import (
    ResultOperationType,
)
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests.protocols import p
    from tests.typings import t


class TestsFlextResultTraverseResource:
    ResultOperationType = ResultOperationType

    def test_accumulate_errors_all_success(self) -> None:
        """Test accumulate_errors with all successes."""
        results = [r[int].ok(1), r[int].ok(2), r[int].ok(3)]
        accumulated = r.accumulate_errors(*results)
        tm.ok(accumulated)
        tm.that(accumulated.value, eq=[1, 2, 3])

    def test_accumulate_errors_with_failures(self) -> None:
        """Test accumulate_errors collects all errors."""
        results = [r[int].ok(1), r[int].fail("error1"), r[int].fail("error2")]
        accumulated = r.accumulate_errors(*results)
        tm.fail(accumulated)
        tm.that(accumulated.error, none=False)
        tm.that(str(accumulated.error), has="error1")
        tm.that(str(accumulated.error), has="error2")

    def test_traverse_fail_fast_true(self) -> None:
        """Test traverse with fail_fast=True (default) stops on first failure."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
            fail_fast=True,
        )
        _ = u.Tests.assert_failure(result)
        tm.that(result.error, eq="error")

    def test_traverse_fail_fast_false(self) -> None:
        """Test traverse with fail_fast=False collects all errors."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail(f"error_{x}") if x in {2, 3} else r[int].ok(x),
            fail_fast=False,
        )
        _ = u.Tests.assert_failure(result)
        tm.that(result.error, none=False)
        tm.that(str(result.error), has="error_2")
        tm.that(str(result.error), has="error_3")

    def test_with_resource(self) -> None:
        """Test with_resource manages resource lifecycle."""
        resource_created: MutableSequence[str] = []
        resource_cleaned: MutableSequence[str] = []

        def factory() -> MutableSequence[str]:
            resource_created.append("created")
            return ["resource"]

        def op(resource: MutableSequence[str]) -> p.Result[str]:
            resource.append("used")
            return r[str].ok("success")

        def cleanup(resource: MutableSequence[str]) -> None:
            resource_cleaned.append("cleaned")
            resource.clear()

        result: p.Result[str] = r[str].with_resource(factory, op, cleanup)
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="success")
        tm.that(len(resource_created), eq=1)
        tm.that(len(resource_cleaned), eq=1)

    def test_with_resource_factory_exception_returns_failure(self) -> None:
        """Test with_resource converts factory exceptions into failed results."""
        cleanup_calls: MutableSequence[str] = []

        def factory() -> MutableSequence[str]:
            msg = "factory failed"
            raise RuntimeError(msg)

        def op(resource: MutableSequence[str]) -> p.Result[str]:
            resource.append("used")
            return r[str].ok("success")

        def cleanup(_resource: MutableSequence[str]) -> None:
            cleanup_calls.append("cleaned")

        result: p.Result[str] = r[str].with_resource(factory, op, cleanup)

        tm.fail(result)
        tm.that(result.error, eq="factory failed")
        tm.that(tuple(cleanup_calls), eq=())

    def test_with_resource_operation_exception_returns_failure_and_cleans(self) -> None:
        """Test with_resource converts operation exceptions and still cleans."""
        cleanup_calls: MutableSequence[str] = []

        def factory() -> MutableSequence[str]:
            return ["resource"]

        def op(_resource: MutableSequence[str]) -> p.Result[str]:
            msg = "operation failed"
            raise RuntimeError(msg)

        def cleanup(resource: MutableSequence[str]) -> None:
            resource.clear()
            cleanup_calls.append("cleaned")

        result: p.Result[str] = r[str].with_resource(factory, op, cleanup)

        tm.fail(result)
        tm.that(result.error, eq="operation failed")
        tm.that(tuple(cleanup_calls), eq=("cleaned",))

    def test_with_resource_cleanup_exception_returns_failure(self) -> None:
        """Test with_resource converts cleanup exceptions into failed results."""

        def factory() -> MutableSequence[str]:
            return ["resource"]

        def op(_resource: MutableSequence[str]) -> p.Result[str]:
            return r[str].ok("success")

        def cleanup(_resource: MutableSequence[str]) -> None:
            msg = "cleanup failed"
            raise RuntimeError(msg)

        result: p.Result[str] = r[str].with_resource(factory, op, cleanup)

        tm.fail(result)
        tm.that(result.error, eq="cleanup failed")

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        result = r[str].ok("value")
        with result as ctx_result:
            tm.that(ctx_result is result, eq=True)
            tm.that(ctx_result.value, eq="value")

    def test_repr_success(self) -> None:
        """Test __repr__ for success result."""
        result = r[str].ok("test")
        repr_str = repr(result)
        tm.that(repr_str, has="r[T].ok")
        tm.that(repr_str, has="test")

    def test_repr_failure(self) -> None:
        """Test __repr__ for failure result."""
        result: p.Result[str] = r[str].fail("error")
        repr_str = repr(result)
        tm.that(repr_str, has="r[T].fail")
        tm.that(repr_str, has="error")

    def test_value_property_failure(self) -> None:
        """Test value property raises RuntimeError on failure."""
        result: p.Result[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            _ = result.value

    def test_error_property_success(self) -> None:
        """Test error property returns None for success."""
        result = r[str].ok("test")
        tm.that(result.error, none=True)

    def test_error_code_property(self) -> None:
        """Test error_code property."""
        result: p.Result[str] = r[str].fail("error", error_code="TEST_ERROR")
        tm.that(result.error_code, eq="TEST_ERROR")
        success = r[str].ok("test")
        tm.that(success.error_code, none=True)

    def test_error_data_property(self) -> None:
        """Test error_data property."""
        error_payload: dict[str, t.JsonPayload] = {"key": "value"}
        error_data = m.ConfigMap(root=error_payload)
        result: p.Result[str] = r[str].fail("error", error_data=error_data)
        tm.that(result.error_data, eq=error_payload)
        success = r[str].ok("test")
        tm.that(success.error_data, none=True)

    def test_error_data_property_accepts_model_dump_carrier(self) -> None:
        """Result error_data accepts protocol carriers through runtime normalization."""

        class ModelDumpCarrier:
            def model_dump(
                self,
                *,
                mode: str = "python",
            ) -> t.MappingKV[str, t.JsonPayload | None]:
                _ = mode
                return {"alpha": 1, "beta": "two"}

        result: p.Result[str] = r[str].fail(
            "error",
            error_data=ModelDumpCarrier(),
        )

        tm.that(result.error_data, eq={"alpha": 1, "beta": "two"})

    def test_unwrap_failure(self) -> None:
        """Test unwrap raises RuntimeError on failure."""
        result: p.Result[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            result.value

    def test_flat_map_inner_failure(self) -> None:
        """Test flat_map inner function returns Failure."""
        result = r[int].ok(5)

        def failing_func(value: int) -> p.Result[str]:
            return r[str].fail("flat_map failed")

        bound = result.flat_map(failing_func)
        tm.fail(bound)

    def test_flow_through_empty(self) -> None:
        """Test flow_through with no functions."""
        result = r[int].ok(5)
        tm.that(result.flow_through() is result, eq=True)
        tm.that(result.value, eq=5)
