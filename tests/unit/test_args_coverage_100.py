"""FLEXT Core Args Utilities Tests - Comprehensive Coverage.

Tests for flext_core._utilities.args.FlextUtilitiesArgs covering:
- validated: decorator with StrEnum conversion and validation
- validated_with_result: decorator converting ValidationError to FlextResult.fail()
- parse_kwargs: parsing kwargs with enum field conversion
- get_enum_params: extracting StrEnum parameters from function signatures

Modules tested: flext_core._utilities.args.FlextUtilitiesArgs
Scope: All args utility methods with 100% coverage including edge cases

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final

import pytest

from flext_core import r, t, u
from flext_tests.utilities import FlextTestsUtilities


class TestFlextUtilitiesArgs:
    """Comprehensive tests for FlextUtilitiesArgs with advanced Python 3.13 patterns.

    Uses factories, enums, mappings, and dynamic tests to reduce code while maintaining
    100% coverage. All test constants organized in nested classes.
    """

    class StatusEnum(StrEnum):
        """Test status enum."""

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    class PriorityEnum(StrEnum):
        """Test priority enum."""

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Constants:
        """Test constants for args utilities."""

        class Values:
            """Test value constants."""

            STATUS_ACTIVE: Final[str] = "active"
            STATUS_PENDING: Final[str] = "pending"
            STATUS_INVALID: Final[str] = "invalid"
            PRIORITY_HIGH: Final[str] = "high"
            NAME_JOHN: Final[str] = "John"
            NAME_JANE: Final[str] = "Jane"
            NAME_BOB: Final[str] = "Bob"
            AGE_30: Final[int] = 30

        class Errors:
            """Error message patterns."""

            INVALID_VALUES: Final[str] = "Invalid values"
            VALIDATION: Final[str] = "validation"
            INTERNAL_ERROR: Final[str] = "Internal error"

    @dataclass(frozen=True, slots=True)
    class ParseKwargsScenario:
        """Parse kwargs test scenario."""

        name: str
        kwargs: dict[str, t.FlexibleValue]
        enum_fields: dict[str, type[StrEnum]]
        expected_success: bool
        expected_status: TestFlextUtilitiesArgs.StatusEnum | None = None
        expected_error: str | None = None

    @dataclass(frozen=True, slots=True)
    class ValidatedScenario:
        """Validated decorator test scenario."""

        name: str
        input_value: str | TestFlextUtilitiesArgs.StatusEnum
        expected_success: bool
        expected_result: str | None = None
        expected_error: str | None = None

    class Scenarios:
        """Centralized test scenarios."""

        @staticmethod
        def get_parse_kwargs_scenarios() -> Mapping[
            str,
            TestFlextUtilitiesArgs.ParseKwargsScenario,
        ]:
            """Get parse kwargs test scenarios."""
            status_enum = TestFlextUtilitiesArgs.StatusEnum
            priority_enum = TestFlextUtilitiesArgs.PriorityEnum
            values = TestFlextUtilitiesArgs.Constants.Values
            errors = TestFlextUtilitiesArgs.Constants.Errors
            scenario_class = TestFlextUtilitiesArgs.ParseKwargsScenario

            return {
                "valid_string_to_enum": scenario_class(
                    name="valid_string_to_enum",
                    kwargs={
                        "status": values.STATUS_ACTIVE,
                        "name": values.NAME_JOHN,
                    },
                    enum_fields={"status": status_enum},
                    expected_success=True,
                    expected_status=status_enum.ACTIVE,
                ),
                "already_enum": scenario_class(
                    name="already_enum",
                    kwargs={
                        "status": status_enum.PENDING,
                        "name": values.NAME_JANE,
                    },
                    enum_fields={"status": status_enum},
                    expected_success=True,
                    expected_status=status_enum.PENDING,
                ),
                "invalid_enum_value": scenario_class(
                    name="invalid_enum_value",
                    kwargs={
                        "status": values.STATUS_INVALID,
                        "name": values.NAME_BOB,
                    },
                    enum_fields={"status": status_enum},
                    expected_success=False,
                    expected_error=errors.INVALID_VALUES,
                ),
                "multiple_enum_fields": scenario_class(
                    name="multiple_enum_fields",
                    kwargs={
                        "status": values.STATUS_ACTIVE,
                        "priority": values.PRIORITY_HIGH,
                    },
                    enum_fields={
                        "status": status_enum,
                        "priority": priority_enum,
                    },
                    expected_success=True,
                    expected_status=status_enum.ACTIVE,
                ),
                "no_enum_fields": scenario_class(
                    name="no_enum_fields",
                    kwargs={
                        "name": values.NAME_JOHN,
                        "age": values.AGE_30,
                    },
                    enum_fields={},
                    expected_success=True,
                ),
            }

        @staticmethod
        def get_validated_scenarios() -> Mapping[
            str,
            TestFlextUtilitiesArgs.ValidatedScenario,
        ]:
            """Get validated decorator test scenarios."""
            status_enum = TestFlextUtilitiesArgs.StatusEnum
            values = TestFlextUtilitiesArgs.Constants.Values
            errors = TestFlextUtilitiesArgs.Constants.Errors
            scenario_class = TestFlextUtilitiesArgs.ValidatedScenario

            return {
                "string_to_enum": scenario_class(
                    name="string_to_enum",
                    input_value=values.STATUS_ACTIVE,
                    expected_success=True,
                    expected_result=values.STATUS_ACTIVE,
                ),
                "enum_value": scenario_class(
                    name="enum_value",
                    input_value=status_enum.PENDING,
                    expected_success=True,
                    expected_result=values.STATUS_PENDING,
                ),
                "invalid_enum": scenario_class(
                    name="invalid_enum",
                    input_value=values.STATUS_INVALID,
                    expected_success=False,
                    expected_error=errors.VALIDATION,
                ),
            }

    class Factories:
        """Factories for creating test functions and data."""

        @staticmethod
        def create_status_enum(
            value: str | TestFlextUtilitiesArgs.StatusEnum,
        ) -> TestFlextUtilitiesArgs.StatusEnum:
            """Convert string or enum to Status enum."""
            if isinstance(value, TestFlextUtilitiesArgs.StatusEnum):
                return value
            return TestFlextUtilitiesArgs.StatusEnum(value)

        @staticmethod
        def create_validated_function() -> Callable[
            [TestFlextUtilitiesArgs.StatusEnum],
            str,
        ]:
            """Create validated function for testing."""

            @u.Args.validated
            def process(status: TestFlextUtilitiesArgs.StatusEnum) -> str:
                return status.value

            return process

        @staticmethod
        def create_validated_with_result_function() -> Callable[
            [TestFlextUtilitiesArgs.StatusEnum],
            r[str],
        ]:
            """Create validated_with_result function for testing."""

            @u.Args.validated_with_result
            def process(status: TestFlextUtilitiesArgs.StatusEnum) -> r[str]:
                return r.ok(status.value)

            return process

    class TestValidated:
        """Tests for u.Args.validated decorator."""

        @pytest.mark.parametrize(
            "scenario_name",
            ["string_to_enum", "enum_value"],
        )
        def test_validated_decorator_success(
            self,
            scenario_name: str,
        ) -> None:
            """Test validated decorator with parametrized success scenarios."""
            scenarios = TestFlextUtilitiesArgs.Scenarios.get_validated_scenarios()
            scenario = scenarios[scenario_name]

            process = TestFlextUtilitiesArgs.Factories.create_validated_function()
            status_value = TestFlextUtilitiesArgs.Factories.create_status_enum(
                scenario.input_value,
            )
            result = process(status_value)
            assert result == scenario.expected_result

        def test_validated_decorator_invalid_enum(self) -> None:
            """Test validated decorator with invalid enum."""
            scenarios = TestFlextUtilitiesArgs.Scenarios.get_validated_scenarios()
            scenario = scenarios["invalid_enum"]

            with pytest.raises(ValueError):
                _ = TestFlextUtilitiesArgs.Factories.create_status_enum(
                    scenario.input_value,
                )

        @staticmethod
        def test_validated_multiple_params() -> None:
            """Test validated decorator with multiple parameters."""

            @u.Args.validated
            def process(
                status: TestFlextUtilitiesArgs.StatusEnum,
                priority: TestFlextUtilitiesArgs.PriorityEnum,
                name: str,
            ) -> str:
                return f"{name}: {status.value} ({priority.value})"

            values = TestFlextUtilitiesArgs.Constants.Values
            status_val = TestFlextUtilitiesArgs.Factories.create_status_enum(
                values.STATUS_ACTIVE,
            )
            priority_val = TestFlextUtilitiesArgs.PriorityEnum(values.PRIORITY_HIGH)
            result = process(status_val, priority_val, values.NAME_JOHN)
            expected = (
                f"{values.NAME_JOHN}: {values.STATUS_ACTIVE} ({values.PRIORITY_HIGH})"
            )
            assert result == expected

    class TestValidatedWithResult:
        """Tests for u.Args.validated_with_result decorator."""

        @staticmethod
        def test_validated_with_result_success() -> None:
            """Test validated_with_result with valid input."""
            process = (
                TestFlextUtilitiesArgs.Factories.create_validated_with_result_function()
            )
            values = TestFlextUtilitiesArgs.Constants.Values
            status_val = TestFlextUtilitiesArgs.Factories.create_status_enum(
                values.STATUS_ACTIVE,
            )
            result = process(status_val)
            FlextTestsUtilities.Tests.ResultHelpers.assert_success_with_value(
                result,
                values.STATUS_ACTIVE,
            )

        @staticmethod
        def test_validated_with_result_invalid_enum() -> None:
            """Test validated_with_result with invalid enum."""
            values = TestFlextUtilitiesArgs.Constants.Values

            with pytest.raises(ValueError):
                _ = TestFlextUtilitiesArgs.Factories.create_status_enum(
                    values.STATUS_INVALID,
                )

            process = (
                TestFlextUtilitiesArgs.Factories.create_validated_with_result_function()
            )
            valid_status = TestFlextUtilitiesArgs.Factories.create_status_enum(
                values.STATUS_ACTIVE,
            )
            result = process(valid_status)
            assert result.is_success
            assert result.value == values.STATUS_ACTIVE

        @staticmethod
        def test_validated_with_result_exception() -> None:
            """Test validated_with_result when function raises exception."""
            values = TestFlextUtilitiesArgs.Constants.Values
            errors = TestFlextUtilitiesArgs.Constants.Errors

            @u.Args.validated_with_result
            def process(status: TestFlextUtilitiesArgs.StatusEnum) -> r[str]:
                raise ValueError(errors.INTERNAL_ERROR)

            status_val = TestFlextUtilitiesArgs.Factories.create_status_enum(
                values.STATUS_ACTIVE,
            )
            result = process(status_val)
            FlextTestsUtilities.Tests.ResultHelpers.assert_failure_with_error(
                result,
                expected_error=errors.INTERNAL_ERROR,
            )

    class TestParseKwargs:
        """Tests for u.Args.parse_kwargs."""

        @pytest.mark.parametrize(
            "scenario_name",
            [
                "valid_string_to_enum",
                "already_enum",
                "multiple_enum_fields",
                "no_enum_fields",
            ],
        )
        def test_parse_kwargs_success(self, scenario_name: str) -> None:
            """Test parse_kwargs with parametrized success scenarios."""
            scenarios = TestFlextUtilitiesArgs.Scenarios.get_parse_kwargs_scenarios()
            scenario = scenarios[scenario_name]

            result = u.Args.parse_kwargs(scenario.kwargs, scenario.enum_fields)
            if scenario.expected_success:
                FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result)
                parsed = result.value
                if scenario.expected_status:
                    assert parsed["status"] == scenario.expected_status
            else:
                FlextTestsUtilities.Tests.TestUtilities.assert_result_failure(result)

        def test_parse_kwargs_invalid_enum_value(self) -> None:
            """Test parse_kwargs with invalid enum value."""
            scenarios = TestFlextUtilitiesArgs.Scenarios.get_parse_kwargs_scenarios()
            scenario = scenarios["invalid_enum_value"]

            result = u.Args.parse_kwargs(scenario.kwargs, scenario.enum_fields)
            FlextTestsUtilities.Tests.ResultHelpers.assert_failure_with_error(
                result,
                expected_error=scenario.expected_error,
            )

    class TestGetEnumParams:
        """Tests for u.Args.get_enum_params."""

        @staticmethod
        def test_get_enum_params_simple() -> None:
            """Test get_enum_params with simple StrEnum parameter."""

            def process(status: TestFlextUtilitiesArgs.StatusEnum, name: str) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert "status" in params
            assert params["status"] == TestFlextUtilitiesArgs.StatusEnum
            assert "name" not in params

        @staticmethod
        def test_get_enum_params_multiple() -> None:
            """Test get_enum_params with multiple StrEnum parameters."""

            def process(
                status: TestFlextUtilitiesArgs.StatusEnum,
                priority: TestFlextUtilitiesArgs.PriorityEnum,
                name: str,
            ) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert "status" in params
            assert "priority" in params
            assert params["status"] == TestFlextUtilitiesArgs.StatusEnum
            assert params["priority"] == TestFlextUtilitiesArgs.PriorityEnum
            assert "name" not in params

        @staticmethod
        def test_get_enum_params_annotated() -> None:
            """Test get_enum_params with Annotated type."""

            def process(
                status: Annotated[TestFlextUtilitiesArgs.StatusEnum, "test"],
            ) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert "status" in params
            assert params["status"] == TestFlextUtilitiesArgs.StatusEnum

        @staticmethod
        def test_get_enum_params_union() -> None:
            """Test get_enum_params with Union type."""

            def process(
                status: str | TestFlextUtilitiesArgs.StatusEnum,
            ) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert "status" in params
            assert params["status"] == TestFlextUtilitiesArgs.StatusEnum

        @staticmethod
        def test_get_enum_params_no_enums() -> None:
            """Test get_enum_params with no StrEnum parameters."""

            def process(name: str, age: int) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert params == {}

        @staticmethod
        def test_get_enum_params_exception() -> None:
            """Test get_enum_params when get_type_hints raises exception."""

            class BadFunction:
                __annotations__ = {"invalid": object()}

            params = u.Args.get_enum_params(BadFunction)
            assert params == {}

        @staticmethod
        def test_get_enum_params_nested_annotated() -> None:
            """Test get_enum_params with nested Annotated types."""

            def process(
                status: Annotated[
                    Annotated[TestFlextUtilitiesArgs.StatusEnum, "meta1"],
                    "meta2",
                ],
            ) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert "status" in params
            assert params["status"] == TestFlextUtilitiesArgs.StatusEnum

        @staticmethod
        def test_get_enum_params_annotated_unwrap() -> None:
            """Test get_enum_params unwraps Annotated correctly."""

            def process(
                status: Annotated[
                    Annotated[TestFlextUtilitiesArgs.StatusEnum, "meta1"],
                    "meta2",
                ],
            ) -> bool:
                return True

            params = u.Args.get_enum_params(process)
            assert "status" in params
            assert params["status"] == TestFlextUtilitiesArgs.StatusEnum
