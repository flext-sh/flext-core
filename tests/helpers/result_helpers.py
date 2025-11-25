"""Generic test helpers for FlextResult testing.

Provides reusable test utilities, factories, and test case data structures
for FlextResult testing using Python 3.13 advanced features. Designed to be
generic and reusable across test modules with advanced patterns for code reduction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from returns.io import IOFailure, IOSuccess
from returns.maybe import Nothing, Some

from flext_core import FlextResult

TResult = TypeVar("TResult")
TNewResult = TypeVar("TNewResult")


class ResultOperation(StrEnum):
    """Result operation types for testing."""

    MAP = "map"
    FLAT_MAP = "flat_map"
    TAP = "tap"
    RECOVER = "recover"
    LASH = "lash"
    FLOW_THROUGH = "flow_through"
    UNWRAP_OR = "unwrap_or"
    EXPECT = "expect"


@dataclass(frozen=True, slots=True)
class ResultTestCase:
    """Test case for FlextResult operations."""

    initial_value: object
    operation: ResultOperation
    expected_success: bool = True
    expected_value: object | None = None
    expected_error: str | None = None
    error_code: str | None = None
    transform_fn: Callable[[object], object] | None = None
    transform_result_fn: Callable[[object], FlextResult[object]] | None = None
    default_value: object | None = None
    description: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class ResultCreationCase:
    """Test case for Result creation."""

    value: object
    is_success: bool = True
    error: str | None = None
    error_code: str | None = None
    expected_value: object | None = None
    expected_error: str | None = None
    description: str = field(default="", compare=False)


class ResultTestHelpers:
    """Generic test helpers for FlextResult testing."""

    @staticmethod
    def create_success(value: object) -> FlextResult[object]:
        """Create successful result."""
        return FlextResult[object].ok(value)

    @staticmethod
    def create_failure(
        error: str,
        error_code: str | None = None,
    ) -> FlextResult[object]:
        """Create failed result."""
        return FlextResult[object].fail(error, error_code=error_code)

    @staticmethod
    def assert_success(
        result: FlextResult[object],
        expected_value: object | None = None,
    ) -> None:
        """Assert result is success."""
        assert result.is_success, f"Expected success, got failure: {result.error}"
        if expected_value is not None:
            assert result.value == expected_value

    @staticmethod
    def assert_failure(
        result: FlextResult[object],
        expected_error: str | None = None,
    ) -> None:
        """Assert result is failure."""
        assert result.is_failure, f"Expected failure, got success: {result.value}"
        if expected_error:
            assert result.error is not None
            assert expected_error in result.error

    @staticmethod
    def create_map_cases() -> list[ResultTestCase]:
        """Create test cases for map operation."""

        def double(x: object) -> object:
            if isinstance(x, int):
                return x * 2
            return x

        def upper(x: object) -> object:
            if isinstance(x, str):
                return x.upper()
            return x

        return [
            ResultTestCase(
                initial_value=5,
                operation=ResultOperation.MAP,
                transform_fn=double,
                expected_value=10,
                description="Map success: double",
            ),
            ResultTestCase(
                initial_value="test",
                operation=ResultOperation.MAP,
                transform_fn=upper,
                expected_value="TEST",
                description="Map success: uppercase",
            ),
        ]

    @staticmethod
    def create_flat_map_cases() -> list[ResultTestCase]:
        """Create test cases for flat_map operation."""
        return [
            ResultTestCase(
                initial_value=5,
                operation=ResultOperation.FLAT_MAP,
                transform_result_fn=lambda x: FlextResult[object].ok(f"value_{x}"),
                expected_value="value_5",
                description="Flat map success: to string",
            ),
        ]

    @staticmethod
    def create_railway_cases() -> list[
        tuple[
            object,
            list[Callable[[object], FlextResult[object]]],
            bool,
            object | None,
            str | None,
        ]
    ]:
        """Create test cases for railway composition."""

        def validate(data: object) -> FlextResult[object]:
            if isinstance(data, dict) and data.get("value"):
                return FlextResult[object].ok(data)
            return FlextResult[object].fail("Missing value")

        def process(data: object) -> FlextResult[object]:
            if isinstance(data, dict):
                return FlextResult[object].ok(int(data.get("value", 0)) * 2)
            return FlextResult[object].fail("Invalid data")

        def format_result(value: object) -> FlextResult[object]:
            return FlextResult[object].ok(f"Result: {value}")

        return [
            (
                {"value": 5},
                [validate, process, format_result],
                True,
                "Result: 10",
                None,
            ),
            (
                {},
                [validate, process, format_result],
                False,
                None,
                "Missing value",
            ),
        ]

    @staticmethod
    def create_creation_cases() -> list[ResultCreationCase]:
        """Create test cases for result creation."""
        return [
            ResultCreationCase(
                value="test_value",
                is_success=True,
                expected_value="test_value",
                description="Success creation: string",
            ),
            ResultCreationCase(
                value=42,
                is_success=True,
                expected_value=42,
                description="Success creation: int",
            ),
            ResultCreationCase(
                value=None,
                is_success=False,
                error="test_error",
                expected_error="test_error",
                description="Failure creation: basic",
            ),
            ResultCreationCase(
                value=None,
                is_success=False,
                error="test_error",
                error_code="TEST_ERROR",
                expected_error="test_error",
                description="Failure creation: with code",
            ),
        ]

    @staticmethod
    def create_interop_cases() -> list[tuple[FlextResult[object], str, object]]:
        """Create test cases for monad interoperability."""
        return [
            (FlextResult[object].ok("test"), "to_maybe", Some("test")),
            (FlextResult[object].fail("error"), "to_maybe", Nothing),
            (FlextResult[object].ok("test"), "to_io", IOSuccess("test")),
            (FlextResult[object].fail("error"), "to_io", IOFailure("error")),
        ]
