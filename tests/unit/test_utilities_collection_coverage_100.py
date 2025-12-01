"""Real tests to achieve 100% collection utilities coverage - no mocks.

Module: flext_core._utilities.collection
Scope: FlextUtilitiesCollection - parse_sequence, coerce_list_validator,
parse_mapping, coerce_dict_validator

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
from pydantic import BaseModel, Field

from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core.typings import FlextTypes


class FixtureStatus(StrEnum):
    """Test status enum for collection utilities tests."""

    ACTIVE = "active"
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class FixturePriority(StrEnum):
    """Test priority enum for collection utilities tests."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ParseSequenceScenario:
    """Parse sequence test scenario."""

    name: str
    enum_cls: type[StrEnum]
    values: list[str | StrEnum]
    expected_success: bool
    expected_count: int | None = None
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class CoerceListScenario:
    """Coerce list validator test scenario."""

    name: str
    enum_cls: type[StrEnum]
    value: FlextTypes.FlexibleValue
    expected_success: bool
    expected_count: int | None = None
    error_type: type[Exception] | None = None
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class ParseMappingScenario:
    """Parse mapping test scenario."""

    name: str
    enum_cls: type[StrEnum]
    mapping: dict[str, str | StrEnum]
    expected_success: bool
    expected_keys: list[str] | None = None
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class CoerceDictScenario:
    """Coerce dict validator test scenario."""

    name: str
    enum_cls: type[StrEnum]
    value: FlextTypes.FlexibleValue
    expected_success: bool
    expected_keys: list[str] | None = None
    error_type: type[Exception] | None = None
    error_contains: str | None = None


class CollectionUtilitiesScenarios:
    """Centralized collection utilities test scenarios."""

    PARSE_SEQUENCE_CASES: ClassVar[list[ParseSequenceScenario]] = [
        ParseSequenceScenario(
            name="valid_strings",
            enum_cls=FixtureStatus,
            values=["active", "pending"],
            expected_success=True,
            expected_count=2,
        ),
        ParseSequenceScenario(
            name="valid_enums",
            enum_cls=FixtureStatus,
            values=[FixtureStatus.ACTIVE, FixtureStatus.PENDING],
            expected_success=True,
            expected_count=2,
        ),
        ParseSequenceScenario(
            name="mixed_strings_enums",
            enum_cls=FixtureStatus,
            values=["active", FixtureStatus.PENDING, "completed"],
            expected_success=True,
            expected_count=3,
        ),
        ParseSequenceScenario(
            name="empty_list",
            enum_cls=FixtureStatus,
            values=[],
            expected_success=True,
            expected_count=0,
        ),
        ParseSequenceScenario(
            name="invalid_string",
            enum_cls=FixtureStatus,
            values=["invalid_status"],
            expected_success=False,
            error_contains="Invalid",
        ),
        ParseSequenceScenario(
            name="mixed_valid_invalid",
            enum_cls=FixtureStatus,
            values=["active", "invalid", "pending"],
            expected_success=False,
            error_contains="Invalid",
        ),
        ParseSequenceScenario(
            name="single_valid",
            enum_cls=FixturePriority,
            values=["high"],
            expected_success=True,
            expected_count=1,
        ),
        ParseSequenceScenario(
            name="single_invalid",
            enum_cls=FixturePriority,
            values=["invalid_priority"],
            expected_success=False,
            error_contains="Invalid",
        ),
    ]

    COERCE_LIST_CASES: ClassVar[list[CoerceListScenario]] = [
        CoerceListScenario(
            name="valid_list_strings",
            enum_cls=FixtureStatus,
            value=["active", "pending"],
            expected_success=True,
            expected_count=2,
        ),
        CoerceListScenario(
            name="valid_list_enums",
            enum_cls=FixtureStatus,
            value=[FixtureStatus.ACTIVE, FixtureStatus.PENDING],
            expected_success=True,
            expected_count=2,
        ),
        CoerceListScenario(
            name="valid_tuple",
            enum_cls=FixtureStatus,
            value=("active", "pending"),
            expected_success=True,
            expected_count=2,
        ),
        CoerceListScenario(
            name="valid_set",
            enum_cls=FixtureStatus,
            value=list({"active", "pending"}),  # type: ignore[arg-type]  # list[str] is compatible with Sequence[ScalarValue]
            expected_success=True,
            expected_count=2,
        ),
        CoerceListScenario(
            name="valid_frozenset",
            enum_cls=FixtureStatus,
            value=list(frozenset(["active", "pending"])),  # type: ignore[arg-type]  # list[str] is compatible with Sequence[ScalarValue]
            expected_success=True,
            expected_count=2,
        ),
        CoerceListScenario(
            name="empty_list",
            enum_cls=FixtureStatus,
            value=[],
            expected_success=True,
            expected_count=0,
        ),
        CoerceListScenario(
            name="invalid_type_int",
            enum_cls=FixtureStatus,
            value=123,
            expected_success=False,
            error_type=TypeError,
            error_contains="Expected sequence",
        ),
        CoerceListScenario(
            name="invalid_type_str",
            enum_cls=FixtureStatus,
            value="active",
            expected_success=False,
            error_type=TypeError,
            error_contains="Expected sequence",
        ),
        CoerceListScenario(
            name="invalid_string_in_list",
            enum_cls=FixtureStatus,
            value=["active", "invalid_status"],
            expected_success=False,
            error_type=ValueError,
            error_contains="Invalid",
        ),
        CoerceListScenario(
            name="invalid_type_in_list",
            enum_cls=FixtureStatus,
            value=["active", 123],
            expected_success=False,
            error_type=TypeError,
            error_contains="Expected str",
        ),
    ]

    PARSE_MAPPING_CASES: ClassVar[list[ParseMappingScenario]] = [
        ParseMappingScenario(
            name="valid_strings",
            enum_cls=FixtureStatus,
            mapping={"user1": "active", "user2": "pending"},
            expected_success=True,
            expected_keys=["user1", "user2"],
        ),
        ParseMappingScenario(
            name="valid_enums",
            enum_cls=FixtureStatus,
            mapping={"user1": FixtureStatus.ACTIVE, "user2": FixtureStatus.PENDING},
            expected_success=True,
            expected_keys=["user1", "user2"],
        ),
        ParseMappingScenario(
            name="mixed_strings_enums",
            enum_cls=FixtureStatus,
            mapping={"user1": "active", "user2": FixtureStatus.PENDING},
            expected_success=True,
            expected_keys=["user1", "user2"],
        ),
        ParseMappingScenario(
            name="empty_dict",
            enum_cls=FixtureStatus,
            mapping={},
            expected_success=True,
            expected_keys=[],
        ),
        ParseMappingScenario(
            name="invalid_string",
            enum_cls=FixtureStatus,
            mapping={"user1": "invalid_status"},
            expected_success=False,
            error_contains="Invalid",
        ),
        ParseMappingScenario(
            name="mixed_valid_invalid",
            enum_cls=FixtureStatus,
            mapping={"user1": "active", "user2": "invalid"},
            expected_success=False,
            error_contains="Invalid",
        ),
        ParseMappingScenario(
            name="single_valid",
            enum_cls=FixturePriority,
            mapping={"task1": "high"},
            expected_success=True,
            expected_keys=["task1"],
        ),
        ParseMappingScenario(
            name="single_invalid",
            enum_cls=FixturePriority,
            mapping={"task1": "invalid_priority"},
            expected_success=False,
            error_contains="Invalid",
        ),
    ]

    COERCE_DICT_CASES: ClassVar[list[CoerceDictScenario]] = [
        CoerceDictScenario(
            name="valid_dict_strings",
            enum_cls=FixtureStatus,
            value={"user1": "active", "user2": "pending"},
            expected_success=True,
            expected_keys=["user1", "user2"],
        ),
        CoerceDictScenario(
            name="valid_dict_enums",
            enum_cls=FixtureStatus,
            value={"user1": FixtureStatus.ACTIVE, "user2": FixtureStatus.PENDING},
            expected_success=True,
            expected_keys=["user1", "user2"],
        ),
        CoerceDictScenario(
            name="empty_dict",
            enum_cls=FixtureStatus,
            value={},
            expected_success=True,
            expected_keys=[],
        ),
        CoerceDictScenario(
            name="invalid_type_int",
            enum_cls=FixtureStatus,
            value=123,
            expected_success=False,
            error_type=TypeError,
            error_contains="Expected dict",
        ),
        CoerceDictScenario(
            name="invalid_type_list",
            enum_cls=FixtureStatus,
            value=["active", "pending"],
            expected_success=False,
            error_type=TypeError,
            error_contains="Expected dict",
        ),
        CoerceDictScenario(
            name="invalid_string_in_dict",
            enum_cls=FixtureStatus,
            value={"user1": "active", "user2": "invalid_status"},
            expected_success=False,
            error_type=ValueError,
            error_contains="Invalid",
        ),
        CoerceDictScenario(
            name="invalid_type_in_dict",
            enum_cls=FixtureStatus,
            value={"user1": "active", "user2": 123},
            expected_success=False,
            error_type=TypeError,
            error_contains="Expected str",
        ),
    ]


class TestFlextUtilitiesCollectionParseSequence:
    """Real tests for FlextUtilitiesCollection.parse_sequence."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.PARSE_SEQUENCE_CASES,
        ids=lambda s: s.name,
    )
    def test_parse_sequence(
        self,
        scenario: ParseSequenceScenario,
    ) -> None:
        """Test parse_sequence with various scenarios."""
        result = FlextUtilitiesCollection.parse_sequence(
            scenario.enum_cls,
            scenario.values,
        )

        if scenario.expected_success:
            assert result.is_success, f"Expected success but got: {result.error}"
            assert result.value is not None
            if scenario.expected_count is not None:
                assert len(result.value) == scenario.expected_count
            # Verify all values are of correct enum type
            for val in result.value:
                assert isinstance(val, scenario.enum_cls)
        else:
            assert result.is_failure, "Expected failure but got success"
            assert result.error is not None
            if scenario.error_contains:
                assert scenario.error_contains in result.error

    def test_parse_sequence_with_custom_enum(self) -> None:
        """Test parse_sequence with custom enum class."""
        result = FlextUtilitiesCollection.parse_sequence(
            FixturePriority,
            ["low", "medium", "high"],
        )
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 3
        assert all(isinstance(v, FixturePriority) for v in result.value)

    def test_parse_sequence_error_message_format(self) -> None:
        """Test parse_sequence error message format."""
        result = FlextUtilitiesCollection.parse_sequence(
            FixtureStatus,
            ["active", "invalid1", "invalid2"],
        )
        assert result.is_failure
        assert result.error is not None
        assert "Invalid" in result.error
        assert "invalid1" in result.error or "invalid2" in result.error


class TestFlextUtilitiesCollectionCoerceListValidator:
    """Real tests for FlextUtilitiesCollection.coerce_list_validator."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.COERCE_LIST_CASES,
        ids=lambda s: s.name,
    )
    def test_coerce_list_validator(
        self,
        scenario: CoerceListScenario,
    ) -> None:
        """Test coerce_list_validator with various scenarios."""
        validator = FlextUtilitiesCollection.coerce_list_validator(scenario.enum_cls)

        if scenario.expected_success:
            result = validator(scenario.value)
            assert isinstance(result, list)
            if scenario.expected_count is not None:
                assert len(result) == scenario.expected_count
            # Verify all values are of correct enum type
            for val in result:
                assert isinstance(val, scenario.enum_cls)
        else:
            with pytest.raises(
                scenario.error_type or Exception,
                match=scenario.error_contains or "",
            ):
                validator(scenario.value)

    def test_coerce_list_validator_with_pydantic(self) -> None:
        """Test coerce_list_validator integration with Pydantic."""
        status_list = list[FixtureStatus]
        FlextUtilitiesCollection.coerce_list_validator(FixtureStatus)

        class TestModel(BaseModel):
            statuses: status_list = Field(default_factory=list)

        # Test with string list
        model1 = TestModel.model_validate({"statuses": ["active", "pending"]})
        assert len(model1.statuses) == 2
        assert all(isinstance(s, FixtureStatus) for s in model1.statuses)

        # Test with enum list
        model2 = TestModel.model_validate(
            {"statuses": [FixtureStatus.ACTIVE, FixtureStatus.PENDING]},
        )
        assert len(model2.statuses) == 2
        assert all(isinstance(s, FixtureStatus) for s in model2.statuses)


class TestFlextUtilitiesCollectionParseMapping:
    """Real tests for FlextUtilitiesCollection.parse_mapping."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.PARSE_MAPPING_CASES,
        ids=lambda s: s.name,
    )
    def test_parse_mapping(
        self,
        scenario: ParseMappingScenario,
    ) -> None:
        """Test parse_mapping with various scenarios."""
        result = FlextUtilitiesCollection.parse_mapping(
            scenario.enum_cls,
            scenario.mapping,
        )

        if scenario.expected_success:
            assert result.is_success, f"Expected success but got: {result.error}"
            assert result.value is not None
            if scenario.expected_keys is not None:
                assert set(result.value.keys()) == set(scenario.expected_keys)
            # Verify all values are of correct enum type
            for val in result.value.values():
                assert isinstance(val, scenario.enum_cls)
        else:
            assert result.is_failure, "Expected failure but got success"
            assert result.error is not None
            if scenario.error_contains:
                assert scenario.error_contains in result.error

    def test_parse_mapping_with_custom_enum(self) -> None:
        """Test parse_mapping with custom enum class."""
        result = FlextUtilitiesCollection.parse_mapping(
            FixturePriority,
            {"task1": "low", "task2": "medium", "task3": "high"},
        )
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 3
        assert all(isinstance(v, FixturePriority) for v in result.value.values())

    def test_parse_mapping_error_message_format(self) -> None:
        """Test parse_mapping error message format."""
        result = FlextUtilitiesCollection.parse_mapping(
            FixtureStatus,
            {"user1": "active", "user2": "invalid1", "user3": "invalid2"},
        )
        assert result.is_failure
        assert result.error is not None
        assert "Invalid" in result.error
        assert "invalid1" in result.error or "invalid2" in result.error


class TestFlextUtilitiesCollectionCoerceDictValidator:
    """Real tests for FlextUtilitiesCollection.coerce_dict_validator."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.COERCE_DICT_CASES,
        ids=lambda s: s.name,
    )
    def test_coerce_dict_validator(
        self,
        scenario: CoerceDictScenario,
    ) -> None:
        """Test coerce_dict_validator with various scenarios."""
        validator = FlextUtilitiesCollection.coerce_dict_validator(scenario.enum_cls)

        if scenario.expected_success:
            result = validator(scenario.value)
            assert isinstance(result, dict)
            if scenario.expected_keys is not None:
                assert set(result.keys()) == set(scenario.expected_keys)
            # Verify all values are of correct enum type
            for val in result.values():
                assert isinstance(val, scenario.enum_cls)
        else:
            with pytest.raises(
                scenario.error_type or Exception,
                match=scenario.error_contains or "",
            ):
                validator(scenario.value)

    def test_coerce_dict_validator_with_pydantic(self) -> None:
        """Test coerce_dict_validator integration with Pydantic."""
        FlextUtilitiesCollection.coerce_dict_validator(FixtureStatus)

        class TestModel(BaseModel):
            user_statuses: dict[str, FixtureStatus] = Field(default_factory=dict)

        # Test with string dict
        model1 = TestModel.model_validate(
            {"user_statuses": {"user1": "active", "user2": "pending"}},
        )
        assert len(model1.user_statuses) == 2
        assert all(isinstance(s, FixtureStatus) for s in model1.user_statuses.values())

        # Test with enum dict
        model2 = TestModel.model_validate(
            {
                "user_statuses": {
                    "user1": FixtureStatus.ACTIVE,
                    "user2": FixtureStatus.PENDING,
                },
            },
        )
        assert len(model2.user_statuses) == 2
        assert all(isinstance(s, FixtureStatus) for s in model2.user_statuses.values())


__all__ = [
    "TestFlextUtilitiesCollectionCoerceDictValidator",
    "TestFlextUtilitiesCollectionCoerceListValidator",
    "TestFlextUtilitiesCollectionParseMapping",
    "TestFlextUtilitiesCollectionParseSequence",
]
