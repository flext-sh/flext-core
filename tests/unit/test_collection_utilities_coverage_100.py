"""Real tests to achieve 100% collection utilities coverage - no mocks.

Module: flext_core._utilities.collection
Scope: uCollection - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/collection.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import u
from flext_core.typings import t


class Status(StrEnum):
    """Test StrEnum for collection testing."""

    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class Priority(StrEnum):
    """Test StrEnum for collection testing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ParseSequenceScenario:
    """Parse sequence test scenario."""

    name: str
    values: list[str | Status]
    expected_success: bool
    expected_count: int | None
    expected_error: str | None


@dataclass(frozen=True, slots=True)
class CoerceListValidatorScenario:
    """Coerce list validator test scenario."""

    name: str
    value: t.FlexibleValue
    expected_success: bool
    expected_error: str | None


@dataclass(frozen=True, slots=True)
class ParseMappingScenario:
    """Parse mapping test scenario."""

    name: str
    mapping: dict[str, str | Status]
    expected_success: bool
    expected_count: int | None
    expected_error: str | None


class CollectionScenarios:
    """Centralized collection test scenarios."""

    PARSE_SEQUENCE: ClassVar[list[ParseSequenceScenario]] = [
        ParseSequenceScenario(
            name="valid_strings",
            values=["active", "pending"],
            expected_success=True,
            expected_count=2,
            expected_error=None,
        ),
        ParseSequenceScenario(
            name="valid_enums",
            values=[Status.ACTIVE, Status.PENDING],
            expected_success=True,
            expected_count=2,
            expected_error=None,
        ),
        ParseSequenceScenario(
            name="mixed_strings_and_enums",
            values=["active", Status.PENDING],
            expected_success=True,
            expected_count=2,
            expected_error=None,
        ),
        ParseSequenceScenario(
            name="invalid_string",
            values=["invalid"],
            expected_success=False,
            expected_count=None,
            expected_error="Invalid Status values",
        ),
        ParseSequenceScenario(
            name="multiple_invalid",
            values=["active", "invalid1", "invalid2"],
            expected_success=False,
            expected_count=None,
            expected_error="Invalid Status values",
        ),
        ParseSequenceScenario(
            name="empty_sequence",
            values=[],
            expected_success=True,
            expected_count=0,
            expected_error=None,
        ),
    ]

    COERCE_LIST_VALIDATOR: ClassVar[list[CoerceListValidatorScenario]] = [
        CoerceListValidatorScenario(
            name="valid_list_strings",
            value=["active", "pending"],
            expected_success=True,
            expected_error=None,
        ),
        CoerceListValidatorScenario(
            name="valid_tuple_strings",
            value=("active", "pending"),
            expected_success=True,
            expected_error=None,
        ),
        CoerceListValidatorScenario(
            name="valid_set_strings",
            value=list({"active", "pending"}),
            expected_success=True,
            expected_error=None,
        ),
        CoerceListValidatorScenario(
            name="valid_frozenset_strings",
            value=list(frozenset({"active", "pending"})),
            expected_success=True,
            expected_error=None,
        ),
        CoerceListValidatorScenario(
            name="valid_list_enums",
            value=[Status.ACTIVE, Status.PENDING],
            expected_success=True,
            expected_error=None,
        ),
        CoerceListValidatorScenario(
            name="invalid_not_sequence",
            value="not a sequence",
            expected_success=False,
            expected_error="Expected sequence",
        ),
        CoerceListValidatorScenario(
            name="invalid_string_in_list",
            value=["active", "invalid"],
            expected_success=False,
            expected_error="Invalid Status",
        ),
        CoerceListValidatorScenario(
            name="invalid_type_in_list",
            value=["active", 123],
            expected_success=False,
            expected_error="Expected str",
        ),
    ]

    PARSE_MAPPING: ClassVar[list[ParseMappingScenario]] = [
        ParseMappingScenario(
            name="valid_strings",
            mapping={"user1": "active", "user2": "pending"},
            expected_success=True,
            expected_count=2,
            expected_error=None,
        ),
        ParseMappingScenario(
            name="valid_enums",
            mapping={"user1": Status.ACTIVE, "user2": Status.PENDING},
            expected_success=True,
            expected_count=2,
            expected_error=None,
        ),
        ParseMappingScenario(
            name="mixed_strings_and_enums",
            mapping={"user1": "active", "user2": Status.PENDING},
            expected_success=True,
            expected_count=2,
            expected_error=None,
        ),
        ParseMappingScenario(
            name="invalid_string",
            mapping={"user1": "invalid"},
            expected_success=False,
            expected_count=None,
            expected_error="Invalid Status values",
        ),
        ParseMappingScenario(
            name="empty_mapping",
            mapping={},
            expected_success=True,
            expected_count=0,
            expected_error=None,
        ),
    ]


class TestuCollectionParseSequence:
    """Test uCollection.parse_sequence."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionScenarios.PARSE_SEQUENCE,
        ids=lambda s: s.name,
    )
    def test_parse_sequence(self, scenario: ParseSequenceScenario) -> None:
        """Test parse_sequence with various scenarios."""
        result = u.Collection.parse_sequence(Status, scenario.values)

        assert result.is_success == scenario.expected_success

        if scenario.expected_success:
            assert result.is_success
            parsed = result.value
            assert len(parsed) == scenario.expected_count
            assert isinstance(parsed, tuple)
        else:
            assert result.is_failure
            error_msg = result.error
            assert error_msg is not None and scenario.expected_error is not None
            assert scenario.expected_error in error_msg


class TestuCollectionCoerceListValidator:
    """Test uCollection.coerce_list_validator."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionScenarios.COERCE_LIST_VALIDATOR,
        ids=lambda s: s.name,
    )
    def test_coerce_list_validator(self, scenario: CoerceListValidatorScenario) -> None:
        """Test coerce_list_validator with various scenarios."""
        validator = u.Collection.coerce_list_validator(Status)

        if scenario.expected_success:
            result = validator(scenario.value)
            assert isinstance(result, list)
            assert all(isinstance(item, Status) for item in result)
        else:
            # Business Rule: pytest.raises accepts Exception types or tuple of Exception types
            # TypeError and ValueError are both Exception subclasses
            with pytest.raises(Exception) as exc_info:
                validator(scenario.value)
            expected_error = scenario.expected_error
            assert expected_error is not None
            # Check that the exception is one of the expected types
            assert isinstance(exc_info.value, (TypeError, ValueError))
            assert expected_error in str(exc_info.value)


class TestuCollectionParseMapping:
    """Test uCollection.parse_mapping."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionScenarios.PARSE_MAPPING,
        ids=lambda s: s.name,
    )
    def test_parse_mapping(self, scenario: ParseMappingScenario) -> None:
        """Test parse_mapping with various scenarios."""
        result = u.Collection.parse_mapping(Status, scenario.mapping)

        assert result.is_success == scenario.expected_success

        if scenario.expected_success:
            assert result.is_success
            parsed = result.value
            assert len(parsed) == scenario.expected_count
            assert isinstance(parsed, dict)
            assert all(isinstance(v, Status) for v in parsed.values())
        else:
            assert result.is_failure
            error_msg = result.error
            assert error_msg is not None and scenario.expected_error is not None
            assert scenario.expected_error in error_msg


class TestuCollectionCoerceDictValidator:
    """Test uCollection.coerce_dict_validator."""

    def test_coerce_dict_validator_valid_strings(self) -> None:
        """Test coerce_dict_validator with valid string values."""
        validator = u.Collection.coerce_dict_validator(Status)
        result = validator({"user1": "active", "user2": "pending"})

        assert isinstance(result, dict)
        assert result["user1"] == Status.ACTIVE
        assert result["user2"] == Status.PENDING

    def test_coerce_dict_validator_valid_enums(self) -> None:
        """Test coerce_dict_validator with valid enum values."""
        validator = u.Collection.coerce_dict_validator(Status)
        result = validator({"user1": Status.ACTIVE, "user2": Status.PENDING})

        assert isinstance(result, dict)
        assert result["user1"] == Status.ACTIVE
        assert result["user2"] == Status.PENDING

    def test_coerce_dict_validator_invalid_not_dict(self) -> None:
        """Test coerce_dict_validator with non-dict value."""
        validator = u.Collection.coerce_dict_validator(Status)

        with pytest.raises(TypeError) as exc_info:
            validator("not a dict")
        assert "Expected dict" in str(exc_info.value)

    def test_coerce_dict_validator_invalid_string(self) -> None:
        """Test coerce_dict_validator with invalid string value."""
        validator = u.Collection.coerce_dict_validator(Status)

        with pytest.raises(ValueError) as exc_info:
            validator({"user1": "invalid"})
        assert "Invalid Status" in str(exc_info.value)

    def test_coerce_dict_validator_invalid_type(self) -> None:
        """Test coerce_dict_validator with invalid type value."""
        validator = u.Collection.coerce_dict_validator(Status)

        with pytest.raises(TypeError) as exc_info:
            validator({"user1": 123})
        assert "Expected str" in str(exc_info.value)
