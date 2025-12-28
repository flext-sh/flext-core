"""Real tests to achieve 100% enum utilities coverage - no mocks.

Module: flext_core._utilities.enum
Scope: FlextUtilitiesEnum - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/enum.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations
from flext_core.typings import t

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import pytest

from flext_core import t
from flext_core.result import r
from flext_tests import u


class Status(StrEnum):
    """Test StrEnum for enum testing."""

    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class Priority(StrEnum):
    """Test StrEnum for enum testing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class IsMemberScenario:
    """Is member test scenario."""

    name: str
    value: object
    expected: bool


@dataclass(frozen=True, slots=True)
class IsSubsetScenario:
    """Is subset test scenario."""

    name: str
    valid_members: frozenset[Status]
    value: object
    expected: bool


@dataclass(frozen=True, slots=True)
class ParseScenario:
    """Parse test scenario."""

    name: str
    value: str | Status
    expected_success: bool
    expected_status: Status | None
    expected_error: str | None


@dataclass(frozen=True, slots=True)
class ParseOrDefaultScenario:
    """Parse or default test scenario."""

    name: str
    value: str | Status | None
    default: Status
    expected: Status


@dataclass(frozen=True, slots=True)
class CoerceValidatorScenario:
    """Coerce validator test scenario."""

    name: str
    value: t.FlexibleValue
    expected_success: bool
    expected_status: Status | None
    expected_error: str | None


class EnumScenarios:
    """Centralized enum test scenarios."""

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
            expected_error="Invalid Status",
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


class TestuEnumIsMember:
    """Test FlextUtilitiesEnum.is_member."""

    @pytest.mark.parametrize("scenario", EnumScenarios.IS_MEMBER, ids=lambda s: s.name)
    def test_is_member(self, scenario: IsMemberScenario) -> None:
        """Test is_member with various scenarios."""
        # Convert object to t.GeneralValueType for type compatibility
        value_typed: t.GeneralValueType = (
            scenario.value
            if isinstance(scenario.value, (str, int, float, bool, type(None)))
            else str(scenario.value)
        )
        result = u.Enum.is_member(Status, value_typed)
        assert result == scenario.expected


class TestuEnumIsSubset:
    """Test FlextUtilitiesEnum.is_subset."""

    @pytest.mark.parametrize("scenario", EnumScenarios.IS_SUBSET, ids=lambda s: s.name)
    def test_is_subset(self, scenario: IsSubsetScenario) -> None:
        """Test is_subset with various scenarios."""
        # Convert object to t.GeneralValueType for type compatibility
        value_typed: t.GeneralValueType = (
            scenario.value
            if isinstance(scenario.value, (str, int, float, bool, type(None)))
            else str(scenario.value)
        )
        result = u.Enum.is_subset(
            Status,
            scenario.valid_members,
            value_typed,
        )
        assert result == scenario.expected


class TestuEnumParse:
    """Test FlextUtilitiesEnum.parse."""

    @pytest.mark.parametrize("scenario", EnumScenarios.PARSE, ids=lambda s: s.name)
    def test_parse(self, scenario: ParseScenario) -> None:
        """Test parse with various scenarios."""
        result = u.Enum.parse(Status, scenario.value)

        if scenario.expected_success:
            # Type annotation: Status is StrEnum, compatible with t.GeneralValueType
            # Use explicit type annotation to help mypy infer TValue
            expected_status_cast: t.GeneralValueType = cast(
                "t.GeneralValueType",
                scenario.expected_status,
            )
            # Type annotation: mypy cannot infer TValue from StrEnum, specify explicitly
            # Cast result to r[t.GeneralValueType] and expected_value to t.GeneralValueType
            result_typed: r[t.GeneralValueType] = cast(
                "r[t.GeneralValueType]",
                result,
            )
            expected_typed: t.GeneralValueType = expected_status_cast
            u.Tests.Result.assert_success_with_value(
                result_typed,
                expected_typed,
            )
        else:
            u.Tests.Result.assert_result_failure(result)
            assert (
                result.error is not None
                and scenario.expected_error is not None
                and scenario.expected_error in result.error
            )


class TestuEnumParseOrDefault:
    """Test FlextUtilitiesEnum.parse_or_default."""

    @pytest.mark.parametrize(
        "scenario",
        EnumScenarios.PARSE_OR_DEFAULT,
        ids=lambda s: s.name,
    )
    def test_parse_or_default(self, scenario: ParseOrDefaultScenario) -> None:
        """Test parse_or_default with various scenarios."""
        result = u.Enum.parse_or_default(
            Status,
            scenario.value,
            scenario.default,
        )
        assert result == scenario.expected


class TestuEnumCoerceValidator:
    """Test FlextUtilitiesEnum.coerce_validator."""

    @pytest.mark.parametrize(
        "scenario",
        EnumScenarios.COERCE_VALIDATOR,
        ids=lambda s: s.name,
    )
    def test_coerce_validator(self, scenario: CoerceValidatorScenario) -> None:
        """Test coerce_validator with various scenarios."""
        validator = u.Enum.coerce_validator(Status)

        if scenario.expected_success:
            result = validator(scenario.value)
            assert result == scenario.expected_status
        else:
            with pytest.raises(ValueError) as exc_info:
                validator(scenario.value)
            error_str = str(exc_info.value) if exc_info.value else ""
            assert (
                scenario.expected_error is not None
                and scenario.expected_error in error_str
            )


class TestuEnumCoerceByNameValidator:
    """Test FlextUtilitiesEnum.coerce_by_name_validator."""

    def test_coerce_by_name_validator_by_name(self) -> None:
        """Test coerce_by_name_validator with member name."""
        validator = u.Enum.coerce_by_name_validator(Status)
        result = validator("ACTIVE")
        assert result == Status.ACTIVE

    def test_coerce_by_name_validator_by_value(self) -> None:
        """Test coerce_by_name_validator with member value."""
        validator = u.Enum.coerce_by_name_validator(Status)
        result = validator("active")
        assert result == Status.ACTIVE

    def test_coerce_by_name_validator_direct_enum(self) -> None:
        """Test coerce_by_name_validator with direct enum."""
        validator = u.Enum.coerce_by_name_validator(Status)
        result = validator(Status.PENDING)
        assert result == Status.PENDING

    def test_coerce_by_name_validator_invalid(self) -> None:
        """Test coerce_by_name_validator with invalid value."""
        validator = u.Enum.coerce_by_name_validator(Status)

        with pytest.raises(ValueError) as exc_info:
            validator("invalid")
        assert "Invalid Status" in str(exc_info.value)


class TestuEnumMetadata:
    """Test FlextUtilitiesEnum metadata methods."""

    def test_values(self) -> None:
        """Test values method."""
        values = u.Enum.values(Status)
        assert isinstance(values, frozenset)
        assert "active" in values
        assert "pending" in values
        assert "inactive" in values

    def test_names(self) -> None:
        """Test names method."""
        names = u.Enum.names(Status)
        assert isinstance(names, frozenset)
        assert "ACTIVE" in names
        assert "PENDING" in names
        assert "INACTIVE" in names

    def test_members(self) -> None:
        """Test members method."""
        members = u.Enum.members(Status)
        assert isinstance(members, frozenset)
        assert Status.ACTIVE in members
        assert Status.PENDING in members
        assert Status.INACTIVE in members

    def test_metadata_caching(self) -> None:
        """Test that metadata methods are cached."""
        values1 = u.Enum.values(Status)
        values2 = u.Enum.values(Status)
        # Should return same object due to caching
        assert values1 is values2
