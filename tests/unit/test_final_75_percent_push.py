"""Final push to 75% coverage - simple, focused tests.

Module: flext_core (coverage tests)
Scope: r, FlextContainer, FlextExceptions, u

Simple tests targeting uncovered lines.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContainer, FlextExceptions, r, u


class ResultOperationScenario(BaseModel):
    """r operation test scenario."""

    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(description="Result operation scenario name")]
    initial_value: Annotated[int | None, Field(description="Initial result value")]
    operations: Annotated[list[str], Field(description="Operation chain to apply")]
    expected_success: Annotated[bool, Field(description="Expected success state")]
    expected_value: Annotated[
        int | None, Field(default=None, description="Expected resulting value")
    ] = None


class ExceptionTypeScenario(BaseModel):
    """Exception type test scenario."""

    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(description="Exception scenario name")]
    exception_type: Annotated[
        type[FlextExceptions.BaseError],
        Field(
            description="Exception class under test",
        ),
    ]
    message: Annotated[str, Field(description="Exception message")]
    expected_in_str: Annotated[str, Field(description="Expected string marker")]


class CoverageScenarios:
    """Centralized coverage test scenarios using FlextConstants."""

    RESULT_OPERATIONS: ClassVar[list[ResultOperationScenario]] = [
        ResultOperationScenario(
            name="map",
            initial_value=5,
            operations=["map"],
            expected_success=True,
            expected_value=10,
        ),
        ResultOperationScenario(
            name="flat_map",
            initial_value=5,
            operations=["flat_map"],
            expected_success=True,
            expected_value=10,
        ),
        ResultOperationScenario(
            name="flat_map_fail",
            initial_value=5,
            operations=["flat_map_fail"],
            expected_success=False,
            expected_value=None,
        ),
        ResultOperationScenario(
            name="lash_success",
            initial_value=42,
            operations=["lash"],
            expected_success=True,
            expected_value=42,
        ),
        ResultOperationScenario(
            name="lash_failure",
            initial_value=None,
            operations=["lash"],
            expected_success=True,
            expected_value=99,
        ),
        ResultOperationScenario(
            name="chaining",
            initial_value=10,
            operations=["map", "map"],
            expected_success=True,
            expected_value=40,
        ),
        ResultOperationScenario(
            name="failure_propagation",
            initial_value=None,
            operations=["map", "map"],
            expected_success=False,
            expected_value=None,
        ),
    ]
    EXCEPTION_TYPES: ClassVar[list[ExceptionTypeScenario]] = [
        ExceptionTypeScenario(
            name="base",
            exception_type=FlextExceptions.BaseError,
            message="test",
            expected_in_str="test",
        ),
        ExceptionTypeScenario(
            name="validation",
            exception_type=FlextExceptions.ValidationError,
            message="invalid",
            expected_in_str="VALIDATION_ERROR",
        ),
        ExceptionTypeScenario(
            name="type_error",
            exception_type=FlextExceptions.TypeError,
            message="wrong type",
            expected_in_str="TYPE_ERROR",
        ),
        ExceptionTypeScenario(
            name="operation",
            exception_type=FlextExceptions.OperationError,
            message="failed",
            expected_in_str="OPERATION_ERROR",
        ),
        ExceptionTypeScenario(
            name="auth",
            exception_type=FlextExceptions.AuthenticationError,
            message="auth issue",
            expected_in_str="AUTH",
        ),
        ExceptionTypeScenario(
            name="config",
            exception_type=FlextExceptions.ConfigurationError,
            message="config issue",
            expected_in_str="CONFIG",
        ),
        ExceptionTypeScenario(
            name="connection",
            exception_type=FlextExceptions.ConnectionError,
            message="connection issue",
            expected_in_str="CONNECTION",
        ),
        ExceptionTypeScenario(
            name="timeout",
            exception_type=FlextExceptions.TimeoutError,
            message="timeout issue",
            expected_in_str="TIMEOUT",
        ),
    ]


class TestCoveragePush75Percent:
    """Simple tests targeting uncovered lines using FlextTestsUtilities."""

    def test_result_basic_ok(self) -> None:
        """Test basic r ok."""
        result = r[int].ok(42)
        assert result.is_success
        assert result.value == 42

    def test_result_basic_fail(self) -> None:
        """Test basic r fail."""
        result: r[int] = r[int].fail("error")
        assert result.is_failure
        assert result.error == "error"

    @pytest.mark.parametrize(
        "scenario",
        CoverageScenarios.RESULT_OPERATIONS,
        ids=lambda s: s.name,
    )
    def test_result_operations(self, scenario: ResultOperationScenario) -> None:
        """Test r operations with various scenarios."""
        if scenario.initial_value is not None:
            initial_result = r[int].ok(scenario.initial_value)
            result: r[int] = r[int](initial_result._result)
        else:
            initial_result = r[int].fail("error")
            result = r[int](initial_result._result)
        for op in scenario.operations:
            if op == "map":
                result = result.map(lambda x: x * 2 if isinstance(x, int) else x)
            elif op == "flat_map":

                def flat_map_func(x) -> r[int]:
                    if isinstance(x, int):
                        return r[int].ok(x * 2)
                    return r[int].fail("error")

                result = result.flat_map(flat_map_func)
            elif op == "flat_map_fail":
                result = result.flat_map(lambda _: r[int].fail("error"))
            elif op == "lash":
                result = result.lash(lambda _: r[int].ok(99))
        if scenario.expected_success:
            assert result.is_success
            if scenario.expected_value is not None:
                assert result.value == scenario.expected_value
        else:
            assert result.is_failure

    def test_container_basic(self) -> None:
        """Test basic container operations."""
        c = FlextContainer()
        result = c.register("test", "value")
        assert result is c
        r2 = c.get("test")
        assert r2.is_success
        assert r2.value == "value"

    def test_container_not_found(self) -> None:
        """Test container get not found."""
        c = FlextContainer()
        result = c.get("nonexistent")
        assert result.is_failure

    def test_container_clear_all(self) -> None:
        """Test container clear_all."""
        c = FlextContainer()
        c.register("test", "value")
        c.clear_all()
        result = c.get("test")
        assert result.is_failure

    def test_container_unregister(self) -> None:
        """Test container unregister."""
        c = FlextContainer()
        c.register("test", "value")
        c.unregister("test")
        result = c.get("test")
        assert result.is_failure

    def test_container_register_multiple(self) -> None:
        """Test registering multiple services."""
        c = FlextContainer()
        c.register("svc1", "val1")
        c.register("svc2", "val2")
        assert c.get("svc1").value == "val1"
        assert c.get("svc2").value == "val2"

    @pytest.mark.parametrize(
        "scenario",
        CoverageScenarios.EXCEPTION_TYPES,
        ids=lambda s: s.name,
    )
    def test_exception_types(self, scenario: ExceptionTypeScenario) -> None:
        """Test exception types."""
        exc = scenario.exception_type(scenario.message)
        assert isinstance(exc, Exception)
        assert scenario.expected_in_str.upper() in str(exc).upper()

    def test_utilities_id(self) -> None:
        """Test ID generation."""
        id1 = u.generate()
        id2 = u.generate()
        assert id1 != id2
        assert len(id1) == 36

    def test_utilities_timestamp(self) -> None:
        """Test timestamp generation."""
        ts = u.generate_iso_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_result_value_property(self) -> None:
        """Test result .value property."""
        result = r[str].ok("value")
        assert result.value == "value"

    def test_result_unwrap_or(self) -> None:
        """Test unwrap_or with default."""
        result: r[int] = r[int].fail("error")
        assert result.unwrap_or(42) == 42
        r2 = r[int].ok(10)
        assert r2.unwrap_or(42) == 10

    def test_result_bool(self) -> None:
        """Test result as boolean."""
        success = r[int].ok(42)
        assert bool(success) is True
        failure: r[int] = r[int].fail("error")
        assert bool(failure) is False

    def test_result_or_operator(self) -> None:
        """Test | operator for default."""
        result: r[int] = r[int].fail("error")
        defaulted = result | 42
        assert defaulted == 42

    def test_result_repr(self) -> None:
        """Test result repr."""
        result = r[int].ok(42)
        repr_str = repr(result)
        assert "r[T].ok" in repr_str

    def test_result_filter(self) -> None:
        """Test filter method."""
        result = r[int].ok(42)
        r2 = result.filter(lambda x: x > 0)
        assert r2.is_success
        assert r2.value == 42
        r3 = result.filter(lambda x: x > 100)
        assert r3.is_failure

    def test_result_safe_factory(self) -> None:
        """Test safe factory method."""

        def divide(a: int, b: int) -> int:
            return a // b

        divide_wrapped = r.safe(divide)
        result: r[int] = divide_wrapped(10, 2)
        assert result.is_success
        assert result.value == 5
        r2: r[int] = divide_wrapped(10, 0)
        assert r2.is_failure


__all__ = ["TestCoveragePush75Percent"]
