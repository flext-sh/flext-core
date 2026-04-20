"""Real tests to achieve 100% enum utilities coverage - no mocks.

Module: flext_core
Scope: FlextUtilitiesEnum - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/enum.py.

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from flext_tests import tm

from tests import m, u


class TestEnumUtilitiesCoverage:
    @unique
    class Status(StrEnum):
        """Test StrEnum for enum testing."""

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    @unique
    class Priority(StrEnum):
        """Test StrEnum for enum testing."""

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class ParseScenario(m.BaseModel):
        """Parse test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Parse scenario name")]
        value: Annotated[str | StrEnum, m.Field(description="Input value to parse")]
        expected_success: Annotated[
            bool,
            m.Field(description="Whether parse should succeed"),
        ]
        expected_status: Annotated[
            StrEnum | None, m.Field(description="Expected parsed enum status")
        ] = None
        expected_error: Annotated[
            str | None, m.Field(description="Expected error message fragment")
        ] = None

    class ParseOrDefaultScenario(m.BaseModel):
        """Parse or default test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Parse or default scenario name")]
        value: Annotated[
            str | StrEnum | None,
            m.Field(description="Input value to parse"),
        ]
        default: Annotated[StrEnum, m.Field(description="Default enum value")]
        expected: Annotated[StrEnum, m.Field(description="Expected output enum value")]

    PARSE: ClassVar[Sequence[ParseScenario]] = [
        ParseScenario(
            name="valid_string",
            value="active",
            expected_success=True,
            expected_status=Status.ACTIVE,
            expected_error=None,
        ),
        ParseScenario(
            name="valid_enum",
            value=Status.PENDING,
            expected_success=True,
            expected_status=Status.PENDING,
            expected_error=None,
        ),
        ParseScenario(
            name="invalid_string",
            value="invalid",
            expected_success=False,
            expected_status=None,
            expected_error="Cannot parse",
        ),
    ]
    PARSE_OR_DEFAULT: ClassVar[Sequence[ParseOrDefaultScenario]] = [
        ParseOrDefaultScenario(
            name="valid_string",
            value="active",
            default=Status.PENDING,
            expected=Status.ACTIVE,
        ),
        ParseOrDefaultScenario(
            name="valid_enum",
            value=Status.INACTIVE,
            default=Status.PENDING,
            expected=Status.INACTIVE,
        ),
        ParseOrDefaultScenario(
            name="none",
            value=None,
            default=Status.PENDING,
            expected=Status.PENDING,
        ),
        ParseOrDefaultScenario(
            name="invalid_string",
            value="invalid",
            default=Status.PENDING,
            expected=Status.PENDING,
        ),
    ]

    @pytest.mark.parametrize("scenario", PARSE, ids=lambda s: s.name)
    def test_parse(self, scenario: ParseScenario) -> None:
        """Test parse with various scenarios."""
        result = u.parse(scenario.value, self.Status)
        if scenario.expected_success:
            tm.ok(result)
            tm.that(result.value, eq=scenario.expected_status)
        else:
            assert result.failure
            assert (
                result.error is not None
                and scenario.expected_error is not None
                and (scenario.expected_error in result.error)
            )

    @pytest.mark.parametrize(
        "scenario",
        PARSE_OR_DEFAULT,
        ids=lambda s: s.name,
    )
    def test_parse_or_default(self, scenario: ParseOrDefaultScenario) -> None:
        """Test parse_or_default with various scenarios."""
        if isinstance(scenario.default, TestEnumUtilitiesCoverage.Status):
            default: TestEnumUtilitiesCoverage.Status = scenario.default
        else:
            default = self.Status.PENDING
        result = u.parse_or_default(self.Status, scenario.value, default)
        tm.that(result, eq=scenario.expected)

    def test_values(self) -> None:
        """Test values method."""
        values = u.enum_values(self.Status)
        tm.that(values.__class__, eq=frozenset)
        assert "active" in values
        assert "pending" in values
        assert "inactive" in values

    def test_metadata_caching(self) -> None:
        """Test that metadata methods are cached."""
        values1 = u.enum_values(self.Status)
        values2 = u.enum_values(self.Status)
        tm.that(values1 is values2, eq=True)
