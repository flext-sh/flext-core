"""Real tests to achieve 100% enum utilities coverage - no mocks.

Module: flext_core._utilities.enum
Scope: FlextUtilitiesEnum - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/enum.py.

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Annotated, Any, ClassVar, cast

import pytest
from flext_tests import tm, u
from pydantic import BaseModel, ConfigDict, Field


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

    class IsMemberScenario(BaseModel):
        """Is member test scenario."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Is member scenario name")]
        value: Annotated[object, Field(description="Input value to validate")]
        expected: Annotated[bool, Field(description="Expected membership result")]

    class IsSubsetScenario(BaseModel):
        """Is subset test scenario."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Is subset scenario name")]
        valid_members: Annotated[
            frozenset[StrEnum], Field(description="Allowed enum members")
        ]
        value: Annotated[object, Field(description="Input value to validate")]
        expected: Annotated[
            bool, Field(description="Expected subset membership result")
        ]

    class ParseScenario(BaseModel):
        """Parse test scenario."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse scenario name")]
        value: Annotated[str | StrEnum, Field(description="Input value to parse")]
        expected_success: Annotated[
            bool, Field(description="Whether parse should succeed")
        ]
        expected_status: Annotated[
            StrEnum | None,
            Field(default=None, description="Expected parsed enum status"),
        ] = None
        expected_error: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class ParseOrDefaultScenario(BaseModel):
        """Parse or default test scenario."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse or default scenario name")]
        value: Annotated[
            str | StrEnum | None, Field(description="Input value to parse")
        ]
        default: Annotated[StrEnum, Field(description="Default enum value")]
        expected: Annotated[StrEnum, Field(description="Expected output enum value")]

    class CoerceValidatorScenario(BaseModel):
        """Coerce validator test scenario."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Coerce validator scenario name")]
        value: Annotated[
            object | StrEnum | None, Field(description="Input value for coercion")
        ]
        expected_success: Annotated[
            bool, Field(description="Whether coercion should succeed")
        ]
        expected_status: Annotated[
            StrEnum | None, Field(default=None, description="Expected coerced status")
        ] = None
        expected_error: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    IS_MEMBER: ClassVar[list[IsMemberScenario]] = [
        IsMemberScenario(name="valid_enum", value=Status.ACTIVE, expected=True),
        IsMemberScenario(name="valid_string", value="active", expected=True),
        IsMemberScenario(name="invalid_string", value="invalid", expected=False),
        IsMemberScenario(name="invalid_type", value=123, expected=False),
        IsMemberScenario(name="none", value=None, expected=False),
    ]
    IS_SUBSET: ClassVar[list[IsSubsetScenario]] = [
        IsSubsetScenario(
            name="valid_enum_in_subset",
            valid_members=frozenset({Status.ACTIVE, Status.PENDING}),
            value=Status.ACTIVE,
            expected=True,
        ),
        IsSubsetScenario(
            name="valid_string_in_subset",
            valid_members=frozenset({Status.ACTIVE, Status.PENDING}),
            value="active",
            expected=True,
        ),
        IsSubsetScenario(
            name="valid_enum_not_in_subset",
            valid_members=frozenset({Status.ACTIVE, Status.PENDING}),
            value=Status.INACTIVE,
            expected=False,
        ),
        IsSubsetScenario(
            name="invalid_string",
            valid_members=frozenset({Status.ACTIVE, Status.PENDING}),
            value="invalid",
            expected=False,
        ),
        IsSubsetScenario(
            name="invalid_type",
            valid_members=frozenset({Status.ACTIVE, Status.PENDING}),
            value=123,
            expected=False,
        ),
    ]
    PARSE: ClassVar[list[ParseScenario]] = [
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
    PARSE_OR_DEFAULT: ClassVar[list[ParseOrDefaultScenario]] = [
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
    COERCE_VALIDATOR: ClassVar[list[CoerceValidatorScenario]] = [
        CoerceValidatorScenario(
            name="valid_string",
            value="active",
            expected_success=True,
            expected_status=Status.ACTIVE,
            expected_error=None,
        ),
        CoerceValidatorScenario(
            name="valid_enum",
            value=Status.PENDING,
            expected_success=True,
            expected_status=Status.PENDING,
            expected_error=None,
        ),
        CoerceValidatorScenario(
            name="invalid_string",
            value="invalid",
            expected_success=False,
            expected_status=None,
            expected_error="Invalid Status",
        ),
        CoerceValidatorScenario(
            name="invalid_type",
            value=123,
            expected_success=False,
            expected_status=None,
            expected_error="Invalid Status",
        ),
    ]

    @pytest.mark.parametrize("scenario", IS_MEMBER, ids=lambda s: s.name)
    def test_is_member(self, scenario: IsMemberScenario) -> None:
        """Test is_member with various scenarios."""
        value_typed: bool | float | int | str | StrEnum = (
            scenario.value
            if isinstance(scenario.value, (str, int, float, bool, StrEnum))
            else str(scenario.value)
        )
        result = u.is_member(self.Status, value_typed)
        tm.that(result, eq=scenario.expected)

    @pytest.mark.parametrize("scenario", IS_SUBSET, ids=lambda s: s.name)
    def test_is_subset(self, scenario: IsSubsetScenario) -> None:
        """Test is_subset with various scenarios."""
        if isinstance(scenario.value, (str, int, float, bool, self.Status)):
            value_typed: bool | float | int | str | self.Status = scenario.value
        else:
            value_typed = str(scenario.value)
        valid_members: frozenset[self.Status] = scenario.valid_members  # type: ignore[assignment]
        result = u.is_subset(
            self.Status,
            valid_members,
            value_typed,
        )
        tm.that(result, eq=scenario.expected)

    @pytest.mark.parametrize("scenario", PARSE, ids=lambda s: s.name)
    def test_parse(self, scenario: ParseScenario) -> None:
        """Test parse with various scenarios."""
        result = u.parse(scenario.value, self.Status)
        if scenario.expected_success:
            tm.ok(result)
            tm.that(result.value, eq=scenario.expected_status)
        else:
            _ = u.Tests.Result.assert_failure(result)
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
        # Narrow the type from StrEnum to Status
        if isinstance(scenario.default, self.Status):
            default: self.Status = scenario.default
        else:
            default = self.Status.PENDING
        result = u.parse_or_default(self.Status, scenario.value, cast("Any", default))
        tm.that(result, eq=scenario.expected)

    @pytest.mark.parametrize(
        "scenario",
        COERCE_VALIDATOR,
        ids=lambda s: s.name,
    )
    def test_coerce_validator(self, scenario: CoerceValidatorScenario) -> None:
        """Test coerce_validator with various scenarios."""
        validator = u.coerce_validator(self.Status)
        value: bool | float | int | str | StrEnum = (
            scenario.value
            if isinstance(scenario.value, (str, int, float, bool, StrEnum))
            else str(scenario.value)
        )
        if scenario.expected_success:
            result = validator(value)
            tm.that(result, eq=scenario.expected_status)
        else:
            with pytest.raises(ValueError) as exc_info:
                _ = validator(value)
            error_str = str(exc_info.value) if exc_info.value else ""
            assert (
                scenario.expected_error is not None
                and scenario.expected_error in error_str
            )

    def test_coerce_by_name_validator_by_name(self) -> None:
        """Test coerce_by_name_validator with member name."""
        validator = u.coerce_by_name_validator(self.Status)
        result = validator("ACTIVE")
        tm.that(result, eq=self.Status.ACTIVE)

    def test_coerce_by_name_validator_by_value(self) -> None:
        """Test coerce_by_name_validator with member value."""
        validator = u.coerce_by_name_validator(self.Status)
        result = validator("active")
        tm.that(result, eq=self.Status.ACTIVE)

    def test_coerce_by_name_validator_direct_enum(self) -> None:
        """Test coerce_by_name_validator with direct enum."""
        validator = u.coerce_by_name_validator(self.Status)
        result = validator(self.Status.PENDING)
        tm.that(result, eq=self.Status.PENDING)

    def test_coerce_by_name_validator_invalid(self) -> None:
        """Test coerce_by_name_validator with invalid value."""
        validator = u.coerce_by_name_validator(self.Status)
        with pytest.raises(ValueError) as exc_info:
            _ = validator("invalid")
        assert "Invalid Status" in str(exc_info.value)

    def test_values(self) -> None:
        """Test values method."""
        values = u.values(self.Status)
        tm.that(values.__class__, eq=frozenset)
        assert "active" in values
        assert "pending" in values
        assert "inactive" in values

    def test_names(self) -> None:
        """Test names method."""
        names = u.names(self.Status)
        tm.that(names.__class__, eq=frozenset)
        assert "ACTIVE" in names
        assert "PENDING" in names
        assert "INACTIVE" in names

    def test_members(self) -> None:
        """Test members method."""
        members = u.members(self.Status)
        tm.that(members.__class__, eq=frozenset)
        assert self.Status.ACTIVE in members
        assert self.Status.PENDING in members
        assert self.Status.INACTIVE in members

    def test_metadata_caching(self) -> None:
        """Test that metadata methods are cached."""
        values1 = u.values(self.Status)
        values2 = u.values(self.Status)
        tm.that(values1 is values2, eq=True)
