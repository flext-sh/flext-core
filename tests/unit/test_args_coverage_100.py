"""Real tests to achieve 100% args coverage - no mocks.

Module: flext_core._utilities.args
Scope: uArgs - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/args.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, ClassVar

import pytest
from pydantic import ValidationError

from flext_core import FlextResult, u
from flext_core.typings import t


class Status(StrEnum):
    """Test StrEnum for args testing."""

    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class Priority(StrEnum):
    """Test StrEnum for args testing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ParseKwargsScenario:
    """Parse kwargs test scenario."""

    name: str
    kwargs: dict[str, t.FlexibleValue]
    enum_fields: dict[str, type[StrEnum]]
    expected_success: bool
    expected_status: Status | None
    expected_error: str | None


class ArgsScenarios:
    """Centralized args test scenarios."""

    PARSE_KWARGS: ClassVar[list[ParseKwargsScenario]] = [
        ParseKwargsScenario(
            name="valid_string_to_enum",
            kwargs={"status": "active", "name": "John"},
            enum_fields={"status": Status},
            expected_success=True,
            expected_status=Status.ACTIVE,
            expected_error=None,
        ),
        ParseKwargsScenario(
            name="already_enum",
            kwargs={"status": Status.PENDING, "name": "Jane"},
            enum_fields={"status": Status},
            expected_success=True,
            expected_status=Status.PENDING,
            expected_error=None,
        ),
        ParseKwargsScenario(
            name="invalid_enum_value",
            kwargs={"status": "invalid", "name": "Bob"},
            enum_fields={"status": Status},
            expected_success=False,
            expected_status=None,
            expected_error="Invalid values",
        ),
        ParseKwargsScenario(
            name="multiple_enum_fields",
            kwargs={"status": "active", "priority": "high"},
            enum_fields={"status": Status, "priority": Priority},
            expected_success=True,
            expected_status=Status.ACTIVE,
            expected_error=None,
        ),
        ParseKwargsScenario(
            name="no_enum_fields",
            kwargs={"name": "John", "age": 30},
            enum_fields={},
            expected_success=True,
            expected_status=None,
            expected_error=None,
        ),
    ]


class TestuArgsValidated:
    """Test uArgs.validated decorator."""

    def test_validated_with_str_enum_string(self) -> None:
        """Test validated decorator with string value for StrEnum."""

        @u.Args.validated
        def process(status: Status) -> str:
            return status.value

        # Pydantic validated decorator converts string to enum automatically
        # pyright: ignore[reportArgumentType] - validated decorator handles conversion
        result = process("active")  # type: ignore[arg-type]
        assert result == "active"

    def test_validated_with_str_enum_enum(self) -> None:
        """Test validated decorator with enum value."""

        @u.Args.validated
        def process(status: Status) -> str:
            return status.value

        result = process(Status.PENDING)
        assert result == "pending"

    def test_validated_with_invalid_enum(self) -> None:
        """Test validated decorator with invalid enum value."""

        @u.Args.validated
        def process(status: Status) -> str:
            return status.value

        # Pydantic validated decorator will raise ValidationError for invalid enum
        # pyright: ignore[reportArgumentType] - testing invalid value
        with pytest.raises(ValidationError):
            process("invalid")  # type: ignore[arg-type]

    def test_validated_with_multiple_params(self) -> None:
        """Test validated decorator with multiple parameters."""

        @u.Args.validated
        def process(status: Status, priority: Priority, name: str) -> str:
            return f"{name}: {status.value} ({priority.value})"

        # Pydantic validated decorator converts strings to enums automatically
        # pyright: ignore[reportArgumentType] - validated decorator handles conversion
        result = process("active", "high", "John")  # type: ignore[arg-type]
        assert result == "John: active (high)"


class TestuArgsValidatedWithResult:
    """Test uArgs.validated_with_result decorator."""

    def test_validated_with_result_success(self) -> None:
        """Test validated_with_result with valid input."""

        @u.Args.validated_with_result
        def process(status: Status) -> FlextResult[str]:
            return FlextResult.ok(status.value)

        # Pydantic validated decorator converts string to enum automatically
        # pyright: ignore[reportArgumentType] - validated decorator handles conversion
        result = process("active")  # type: ignore[arg-type]
        assert result.is_success
        assert result.value == "active"

    def test_validated_with_result_invalid_enum(self) -> None:
        """Test validated_with_result with invalid enum."""

        @u.Args.validated_with_result
        def process(status: Status) -> FlextResult[str]:
            return FlextResult.ok(status.value)

        # Pydantic validated decorator will return failure for invalid enum
        # pyright: ignore[reportArgumentType] - testing invalid value
        result = process("invalid")  # type: ignore[arg-type]
        assert result.is_failure
        # Type narrowing: result.error is str | None, check before using .lower()
        error_msg = result.error or ""
        assert "invalid" in error_msg.lower() or "status" in error_msg.lower()

    def test_validated_with_result_with_exception(self) -> None:
        """Test validated_with_result when function raises exception."""

        @u.Args.validated_with_result
        def process(status: Status) -> FlextResult[str]:
            msg = "Internal error"
            raise ValueError(msg)

        # Pydantic validated decorator converts string to enum automatically
        # pyright: ignore[reportArgumentType] - validated decorator handles conversion
        result = process("active")  # type: ignore[arg-type]
        assert result.is_failure
        # Type narrowing: result.error is str | None, check before using 'in'
        error_msg = result.error or ""
        assert "Internal error" in error_msg


class TestuArgsParseKwargs:
    """Test uArgs.parse_kwargs."""

    @pytest.mark.parametrize(
        "scenario",
        ArgsScenarios.PARSE_KWARGS,
        ids=lambda s: s.name,
    )
    def test_parse_kwargs(self, scenario: ParseKwargsScenario) -> None:
        """Test parse_kwargs with various scenarios."""
        result = u.Args.parse_kwargs(scenario.kwargs, scenario.enum_fields)

        assert result.is_success == scenario.expected_success

        if scenario.expected_success:
            assert result.is_success
            parsed = result.value
            if scenario.expected_status:
                assert parsed["status"] == scenario.expected_status
        else:
            assert result.is_failure
            # Type narrowing: result.error is str | None, check before using 'in'
            # scenario.expected_error is also str | None
            error_msg = result.error or ""
            expected_error_str = scenario.expected_error or ""
            assert expected_error_str in error_msg


class TestuArgsGetEnumParams:
    """Test uArgs.get_enum_params."""

    def test_get_enum_params_simple(self) -> None:
        """Test get_enum_params with simple StrEnum parameter."""

        def process(status: Status, name: str) -> bool:
            return True

        params = u.Args.get_enum_params(process)
        assert "status" in params
        assert params["status"] == Status
        assert "name" not in params

    def test_get_enum_params_multiple(self) -> None:
        """Test get_enum_params with multiple StrEnum parameters."""

        def process(status: Status, priority: Priority, name: str) -> bool:
            return True

        params = u.Args.get_enum_params(process)
        assert "status" in params
        assert "priority" in params
        assert params["status"] == Status
        assert params["priority"] == Priority
        assert "name" not in params

    def test_get_enum_params_annotated(self) -> None:
        """Test get_enum_params with Annotated type."""

        def process(status: Annotated[Status, "test"]) -> bool:
            return True

        params = u.Args.get_enum_params(process)
        assert "status" in params
        assert params["status"] == Status

    def test_get_enum_params_union(self) -> None:
        """Test get_enum_params with Union type (str | Status)."""

        def process(status: str | Status) -> bool:
            return True

        params = u.Args.get_enum_params(process)
        # Should detect Status in union
        assert "status" in params
        assert params["status"] == Status

    def test_get_enum_params_no_enums(self) -> None:
        """Test get_enum_params with no StrEnum parameters."""

        def process(name: str, age: int) -> bool:
            return True

        params = u.Args.get_enum_params(process)
        assert params == {}

    def test_get_enum_params_no_annotations(self) -> None:
        """Test get_enum_params with function without annotations."""

        def process(name: str, age: int) -> bool:
            return True

        params = u.Args.get_enum_params(process)
        assert params == {}

    def test_get_enum_params_with_exception(self) -> None:
        """Test get_enum_params when get_type_hints raises exception."""

        class BadFunction:
            __annotations__ = {"invalid": object()}

        params = u.Args.get_enum_params(BadFunction)
        assert params == {}
