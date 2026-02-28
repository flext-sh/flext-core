"""Final push to 75% coverage - simple, focused tests.

Module: flext_core (coverage tests)
Scope: FlextResult, FlextContainer, FlextExceptions, u

Simple tests targeting uncovered lines.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from flext_core import (
    FlextContainer,
    FlextExceptions,
    FlextResult,
    p,
    u,
)


@dataclass(frozen=True, slots=True)
class ResultOperationScenario:
    """FlextResult operation test scenario."""

    name: str
    initial_value: int | None
    operations: list[str]
    expected_success: bool
    expected_value: int | None = None


@dataclass(frozen=True, slots=True)
class ExceptionTypeScenario:
    """Exception type test scenario."""

    name: str
    exception_type: type[FlextExceptions.BaseError]
    message: str
    expected_in_str: str


class CoverageScenarios:
    """Centralized coverage test scenarios using FlextConstants."""

    RESULT_OPERATIONS: ClassVar[list[ResultOperationScenario]] = [
        ResultOperationScenario("map", 5, ["map"], True, 10),
        ResultOperationScenario("flat_map", 5, ["flat_map"], True, 10),
        ResultOperationScenario("flat_map_fail", 5, ["flat_map_fail"], False, None),
        ResultOperationScenario("lash_success", 42, ["lash"], True, 42),
        ResultOperationScenario("lash_failure", None, ["lash"], True, 99),
        ResultOperationScenario("chaining", 10, ["map", "map"], True, 40),
        ResultOperationScenario(
            "failure_propagation",
            None,
            ["map", "map"],
            False,
            None,
        ),
    ]

    EXCEPTION_TYPES: ClassVar[list[ExceptionTypeScenario]] = [
        ExceptionTypeScenario("base", FlextExceptions.BaseError, "test", "test"),
        ExceptionTypeScenario(
            "validation",
            FlextExceptions.ValidationError,
            "invalid",
            "VALIDATION_ERROR",
        ),
        ExceptionTypeScenario(
            "type_error",
            FlextExceptions.TypeError,
            "wrong type",
            "TYPE_ERROR",
        ),
        ExceptionTypeScenario(
            "operation",
            FlextExceptions.OperationError,
            "failed",
            "OPERATION_ERROR",
        ),
        ExceptionTypeScenario(
            "auth",
            FlextExceptions.AuthenticationError,
            "auth issue",
            "AUTH",
        ),
        ExceptionTypeScenario(
            "config",
            FlextExceptions.ConfigurationError,
            "config issue",
            "CONFIG",
        ),
        ExceptionTypeScenario(
            "connection",
            FlextExceptions.ConnectionError,
            "connection issue",
            "CONNECTION",
        ),
        ExceptionTypeScenario(
            "timeout",
            FlextExceptions.TimeoutError,
            "timeout issue",
            "TIMEOUT",
        ),
    ]


class TestCoveragePush75Percent:
    """Simple tests targeting uncovered lines using FlextTestsUtilities."""

    def test_result_basic_ok(self) -> None:
        """Test basic FlextResult ok."""
        r = FlextResult[int].ok(42)
        assert r.is_success
        assert r.value == 42

    def test_result_basic_fail(self) -> None:
        """Test basic FlextResult fail."""
        r: FlextResult[int] = FlextResult[int].fail("error")
        assert r.is_failure
        assert r.error == "error"

    @pytest.mark.parametrize(
        "scenario",
        CoverageScenarios.RESULT_OPERATIONS,
        ids=lambda s: s.name,
    )
    def test_result_operations(self, scenario: ResultOperationScenario) -> None:
        """Test FlextResult operations with various scenarios."""
        if scenario.initial_value is not None:
            initial_result = FlextResult[int].ok(scenario.initial_value)
            r: FlextResult[object] = FlextResult[object](initial_result._result)
        else:
            initial_result = FlextResult[int].fail("error")
            r = FlextResult[object](initial_result._result)
        for op in scenario.operations:
            if op == "map":
                r = r.map(lambda x: x * 2 if isinstance(x, int) else x)
            elif op == "flat_map":

                def flat_map_func(x: object) -> FlextResult[object]:
                    if isinstance(x, int):
                        return FlextResult[object].ok(x * 2)
                    return FlextResult[object].ok(x)

                r = r.flat_map(flat_map_func)
            elif op == "flat_map_fail":
                r = r.flat_map(lambda _: FlextResult[object].fail("error"))
            elif op == "lash":
                r = r.lash(lambda _: FlextResult[object].ok(99))
        if scenario.expected_success:
            assert r.is_success
            if scenario.expected_value is not None:
                assert r.value == scenario.expected_value
        else:
            assert r.is_failure

    def test_container_basic(self) -> None:
        """Test basic container operations."""
        c = FlextContainer()
        r = c.with_service("test", "value")
        assert r is c
        r2 = c.get("test")
        assert r2.is_success
        assert r2.value == "value"

    def test_container_not_found(self) -> None:
        """Test container get not found."""
        c = FlextContainer()
        r = c.get("nonexistent")
        assert r.is_failure

    def test_container_clear_all(self) -> None:
        """Test container clear_all."""
        c = FlextContainer()
        c.with_service("test", "value")
        c.clear_all()
        r = c.get("test")
        assert r.is_failure

    def test_container_unregister(self) -> None:
        """Test container unregister."""
        c = FlextContainer()
        c.with_service("test", "value")
        c.unregister("test")
        r = c.get("test")
        assert r.is_failure

    def test_container_register_multiple(self) -> None:
        """Test registering multiple services."""
        c = FlextContainer()
        c.with_service("svc1", "val1")
        c.with_service("svc2", "val2")
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
        ts = u.Generators.generate_iso_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_result_value_property(self) -> None:
        """Test result .value property."""
        r = FlextResult[str].ok("value")
        assert r.value == "value"

    def test_result_unwrap_or(self) -> None:
        """Test unwrap_or with default."""
        r: FlextResult[int] = FlextResult[int].fail("error")
        assert r.unwrap_or(42) == 42
        r2 = FlextResult[int].ok(10)
        assert r2.unwrap_or(42) == 10

    def test_result_bool(self) -> None:
        """Test result as boolean."""
        success = FlextResult[int].ok(42)
        assert bool(success) is True
        failure: FlextResult[int] = FlextResult[int].fail("error")
        assert bool(failure) is False

    def test_result_or_operator(self) -> None:
        """Test | operator for default."""
        r: FlextResult[int] = FlextResult[int].fail("error")
        result = r | 42
        assert result == 42

    def test_result_repr(self) -> None:
        """Test result repr."""
        result = FlextResult[int].ok(42)
        repr_str = repr(result)
        assert "r.ok" in repr_str

    def test_result_filter(self) -> None:
        """Test filter method."""
        r = FlextResult[int].ok(42)
        r2 = r.filter(lambda x: x > 0)
        assert r2.is_success
        assert r2.value == 42
        r3 = r.filter(lambda x: x > 100)
        assert r3.is_failure

    def test_result_safe_factory(self) -> None:
        """Test safe factory method."""

        # Use cast to match VariadicCallable signature
        def divide(a: int, b: int) -> int:
            return a // b

        # Cast function to match protocol signature before applying decorator
        divide_func = cast("p.VariadicCallable[int]", divide)
        divide_wrapped: p.VariadicCallable[FlextResult[int]] = FlextResult.safe(
            divide_func,
        )

        r: FlextResult[int] = divide_wrapped(10, 2)
        assert r.is_success
        assert r.value == 5
        r2: FlextResult[int] = divide_wrapped(10, 0)
        assert r2.is_failure


__all__ = ["TestCoveragePush75Percent"]
