"""FLEXT Core Args Utilities Tests - Comprehensive Coverage.

Tests for flext_core._utilities.args.FlextUtilitiesArgs covering:
- validated: decorator with StrEnum conversion and validation
- validated_with_result: decorator converting ValidationError to r.fail()
- parse_kwargs: parsing kwargs with enum field conversion
- get_enum_params: extracting StrEnum parameters from function signatures

Modules tested: flext_core._utilities.args.FlextUtilitiesArgs
Scope: All args utility methods with 100% coverage including edge cases

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Final

import pytest
from pydantic import BaseModel, ConfigDict, Field

from flext_tests import tm
from tests import t, u


class TestFlextUtilitiesArgs:
    """Comprehensive tests for FlextUtilitiesArgs with advanced Python 3.13 patterns.

    Uses factories, enums, mappings, and dynamic tests to reduce code while maintaining
    100% coverage. All test constants organized in nested classes.
    """

    @unique
    class StatusEnum(StrEnum):
        """Test status enum."""

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    @unique
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

    class ParseKwargsScenario(BaseModel):
        """Parse kwargs test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse kwargs scenario name")]
        kwargs: Annotated[
            t.ConfigMap,
            Field(description="Keyword arguments input payload"),
        ]
        enum_fields: Annotated[
            Mapping[str, type[StrEnum]],
            Field(description="Enum fields map for conversion"),
        ]
        expected_success: Annotated[
            bool,
            Field(description="Whether parsing should succeed"),
        ]
        expected_status: Annotated[
            TestFlextUtilitiesArgs.StatusEnum | None,
            Field(default=None, description="Expected parsed status enum"),
        ] = None
        expected_error: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

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
            scenarios = [
                scenario_class.model_validate({
                    "name": "valid_string_to_enum",
                    "kwargs": {
                        "status": values.STATUS_ACTIVE,
                        "name": values.NAME_JOHN,
                    },
                    "enum_fields": {"status": status_enum},
                    "expected_success": True,
                    "expected_status": status_enum.ACTIVE,
                }),
                scenario_class.model_validate({
                    "name": "already_enum",
                    "kwargs": {"status": status_enum.PENDING, "name": values.NAME_JANE},
                    "enum_fields": {"status": status_enum},
                    "expected_success": True,
                    "expected_status": status_enum.PENDING,
                }),
                scenario_class.model_validate({
                    "name": "invalid_enum_value",
                    "kwargs": {
                        "status": values.STATUS_INVALID,
                        "name": values.NAME_BOB,
                    },
                    "enum_fields": {"status": status_enum},
                    "expected_success": False,
                    "expected_error": errors.INVALID_VALUES,
                }),
                scenario_class.model_validate({
                    "name": "multiple_enum_fields",
                    "kwargs": {
                        "status": values.STATUS_ACTIVE,
                        "priority": values.PRIORITY_HIGH,
                    },
                    "enum_fields": {"status": status_enum, "priority": priority_enum},
                    "expected_success": True,
                    "expected_status": status_enum.ACTIVE,
                }),
                scenario_class.model_validate({
                    "name": "no_enum_fields",
                    "kwargs": {"name": values.NAME_JOHN, "age": values.AGE_30},
                    "enum_fields": {},
                    "expected_success": True,
                }),
            ]
            return {s.name: s for s in scenarios}

    class TestParseKwargs:
        """Tests for u.parse_kwargs."""

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
            result = u.parse_kwargs(scenario.kwargs.root, scenario.enum_fields)
            if scenario.expected_success:
                _ = u.Tests.Result.assert_success(result)
                parsed = result.value
                if scenario.expected_status:
                    tm.that(parsed["status"], eq=scenario.expected_status)
            else:
                _ = u.Tests.Result.assert_failure(result)

        def test_parse_kwargs_invalid_enum_value(self) -> None:
            """Test parse_kwargs with invalid enum value."""
            scenarios = TestFlextUtilitiesArgs.Scenarios.get_parse_kwargs_scenarios()
            scenario = scenarios["invalid_enum_value"]
            result = u.parse_kwargs(scenario.kwargs.root, scenario.enum_fields)
            u.Tests.Result.assert_failure_with_error(
                result,
                expected_error=scenario.expected_error,
            )
