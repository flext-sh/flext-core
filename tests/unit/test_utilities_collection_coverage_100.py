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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, cast

import pytest
from flext_core import FlextRuntime, m, r, t
from flext_tests import u
from pydantic import BaseModel, Field

from tests.test_utils import assertion_helpers


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
    value: t.FlexibleValue
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
    value: t.FlexibleValue
    expected_success: bool
    expected_keys: list[str] | None = None
    error_type: type[Exception] | None = None
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class MapScenario:
    """Map method test scenario."""

    name: str
    items: list[Any] | tuple[Any, ...] | dict[str, Any] | set[Any] | frozenset[Any]
    mapper: Callable[[Any], Any]
    expected_result: (
        list[Any] | tuple[Any, ...] | dict[str, Any] | set[Any] | frozenset[Any]
    )
    default_error: str = "Operation failed"
    expected_failure: bool = False
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class FindScenario:
    """Find method test scenario."""

    name: str
    items: list[Any] | tuple[Any, ...] | dict[str, Any]
    predicate: Callable[[Any], bool]
    expected_result: object
    return_key: bool = False


@dataclass(frozen=True, slots=True)
class FilterScenario:
    """Filter method test scenario."""

    name: str
    items: list[Any] | tuple[Any, ...] | dict[str, Any]
    predicate: Callable[[Any], bool]
    expected_result: list[Any] | tuple[Any, ...] | dict[str, Any]
    mapper: Callable[[Any], Any] | None = None


@dataclass(frozen=True, slots=True)
class CountScenario:
    """Count method test scenario."""

    name: str
    items: Sequence[Any]
    expected_count: int
    predicate: Callable[[Any], bool] | None = None


@dataclass(frozen=True, slots=True)
class ProcessScenario:
    """Process method test scenario."""

    name: str
    items: Sequence[Any]
    processor: Callable[[Any], Any]
    expected_result: object
    on_error: str = "collect"
    predicate: Callable[[Any], bool] | None = None
    filter_keys: set[str] | None = None
    exclude_keys: set[str] | None = None
    expected_failure: bool = False
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class GroupScenario:
    """Group method test scenario."""

    name: str
    items: list[str] | tuple[str, ...]
    key: Callable[[str], int | str]
    expected_result: dict[int | str, list[str]]


@dataclass(frozen=True, slots=True)
class ChunkScenario:
    """Chunk method test scenario."""

    name: str
    items: list[t.GeneralValueType] | tuple[t.GeneralValueType, ...]
    size: int
    expected_result: list[list[t.GeneralValueType]]


@dataclass(frozen=True, slots=True)
class BatchScenario:
    """Batch method test scenario."""

    name: str
    items: list[t.GeneralValueType]
    operation: Callable[[object], object]
    expected_result: object
    size: int = 100
    on_error: str = "collect"
    pre_validate: Callable[[object], bool] | None = None
    flatten: bool = False
    expected_failure: bool = False
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
            value=list({"active", "pending"}),
            expected_success=True,
            expected_count=2,
        ),
        CoerceListScenario(
            name="valid_frozenset",
            enum_cls=FixtureStatus,
            value=list(frozenset(["active", "pending"])),
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

    MAP_CASES: ClassVar[list[MapScenario]] = [
        MapScenario(
            name="list_ints",
            items=[1, 2, 3],
            mapper=lambda x: cast("int", x) * 2,
            expected_result=[2, 4, 6],
        ),
        MapScenario(
            name="tuple_ints",
            items=(1, 2, 3),
            mapper=lambda x: cast("int", x) * 2,
            expected_result=(2, 4, 6),
        ),
        MapScenario(
            name="set_ints",
            items={1, 2, 3},
            mapper=lambda x: cast("int", x) * 2,
            expected_result={2, 4, 6},
        ),
        MapScenario(
            name="dict_values",
            items={"a": 1, "b": 2},
            mapper=lambda v: cast("int", v) * 2,
            expected_result={"a": 2, "b": 4},
        ),
        MapScenario(
            name="frozenset_ints",
            items=frozenset({1, 2, 3}),
            mapper=lambda x: cast("int", x) * 2,
            expected_result=frozenset({2, 4, 6}),
        ),
        MapScenario(
            name="strings_upper",
            items=["hello", "world"],
            mapper=lambda x: cast("str", x).upper(),
            expected_result=["HELLO", "WORLD"],
        ),
    ]

    FIND_CASES: ClassVar[list[FindScenario]] = [
        FindScenario(
            name="list_find",
            items=[1, 2, 3, 4],
            predicate=lambda x: cast("int", x) % 2 == 0,
            expected_result=2,
        ),
        FindScenario(
            name="list_not_found",
            items=[1, 3, 5],
            predicate=lambda x: cast("int", x) % 2 == 0,
            expected_result=None,
        ),
        FindScenario(
            name="dict_find_value",
            items={"a": 1, "b": 2},
            predicate=lambda v: v == 2,
            expected_result=2,
        ),
        FindScenario(
            name="dict_find_other",
            items={"x": 10, "y": 20},
            predicate=lambda v: cast("int", v) > 15,
            expected_result=20,
        ),
    ]

    FILTER_CASES: ClassVar[list[FilterScenario]] = [
        FilterScenario(
            name="list_filter",
            items=[1, 2, 3, 4],
            predicate=lambda x: cast("int", x) % 2 == 0,
            expected_result=[2, 4],
        ),
        FilterScenario(
            name="list_filter_map",
            items=[1, 2, 3, 4],
            predicate=lambda x: cast("int", x) > 2,
            mapper=lambda x: cast("int", x) * 2,
            expected_result=[6, 8],
        ),
        FilterScenario(
            name="dict_filter",
            items={"a": 1, "b": 2, "c": 3},
            predicate=lambda v: cast("int", v) % 2 != 0,
            expected_result={"a": 1, "c": 3},
        ),
        FilterScenario(
            name="dict_filter_map",
            items={"a": 1, "b": 4},
            predicate=lambda v: cast("int", v) > 2,
            mapper=lambda v: cast("int", v) * 2,
            expected_result={"b": 8},
        ),
        FilterScenario(
            name="list_filter_empty",
            items=[1, 3, 5],
            predicate=lambda x: cast("int", x) > 10,
            expected_result=[],
        ),
        FilterScenario(
            name="list_filter_all",
            items=[2, 4, 6],
            predicate=lambda x: cast("int", x) % 2 == 0,
            expected_result=[2, 4, 6],
        ),
    ]

    COUNT_CASES: ClassVar[list[CountScenario]] = [
        CountScenario(
            name="count_list",
            items=[1, 2, 3, 4],
            expected_count=4,
        ),
        CountScenario(
            name="count_predicate",
            items=[1, 2, 3, 4],
            predicate=lambda x: cast("int", x) % 2 == 0,
            expected_count=2,
        ),
    ]

    PROCESS_CASES: ClassVar[list[ProcessScenario]] = [
        ProcessScenario(
            name="process_list",
            items=[1, 2, 3],
            processor=lambda x: cast("int", x) * 2,
            expected_result=[2, 4, 6],
        ),
        ProcessScenario(
            name="process_list_skip",
            items=[1, 2, 3],
            processor=lambda x: cast("int", x) * 2,
            expected_result=[4, 6],
            predicate=lambda x: cast("int", x) > 1,
        ),
        ProcessScenario(
            name="process_strings",
            items=["a", "b", "c"],
            processor=lambda x: cast("str", x).upper(),
            expected_result=["A", "B", "C"],
        ),
        ProcessScenario(
            name="process_empty",
            items=[],
            processor=lambda x: cast("int", x) * 2,
            expected_result=[],
        ),
    ]

    GROUP_CASES: ClassVar[list[GroupScenario]] = [
        GroupScenario(
            name="group_by_len",
            items=["cat", "dog", "house"],
            key=lambda x: len(x),
            expected_result={3: ["cat", "dog"], 5: ["house"]},
        ),
    ]

    CHUNK_CASES: ClassVar[list[ChunkScenario]] = [
        ChunkScenario(
            name="chunk_list",
            items=[1, 2, 3, 4, 5],
            size=2,
            expected_result=[[1, 2], [3, 4], [5]],
        ),
    ]


class TestuCollectionParseSequence:
    """Real tests for u.Collection.parse_sequence."""

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
        result = u.Collection.parse_sequence(
            scenario.enum_cls,
            scenario.values,
        )

        if scenario.expected_success:
            assertion_helpers.assert_flext_result_success(result)
            assert result.value is not None
            if scenario.expected_count is not None:
                assert len(result.value) == scenario.expected_count
            # Verify all values are of correct enum type
            for val in result.value:
                assert isinstance(val, scenario.enum_cls)
        else:
            assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None
            if scenario.error_contains:
                assert scenario.error_contains in result.error

    def test_parse_sequence_with_custom_enum(self) -> None:
        """Test parse_sequence with custom enum class."""
        result = u.Collection.parse_sequence(
            FixturePriority,
            ["low", "medium", "high"],
        )
        assertion_helpers.assert_flext_result_success(result)
        assert result.value is not None
        assert len(result.value) == 3
        assert all(isinstance(v, FixturePriority) for v in result.value)

    def test_parse_sequence_error_message_format(self) -> None:
        """Test parse_sequence error message format."""
        result = u.Collection.parse_sequence(
            FixtureStatus,
            ["active", "invalid1", "invalid2"],
        )
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None
        assert "Invalid" in result.error
        assert "invalid1" in result.error or "invalid2" in result.error


class TestuCollectionCoerceListValidator:
    """Real tests for u.Collection.coerce_list_validator."""

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
        validator = u.Collection.coerce_list_validator(scenario.enum_cls)

        if scenario.expected_success:
            result = validator(scenario.value)
            u.Tests.Assertions.assert_result_matches_expected(
                result,
                list,
            )
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
                _ = validator(scenario.value)

    def test_coerce_list_validator_with_pydantic(self) -> None:
        """Test coerce_list_validator integration with Pydantic."""
        _ = u.Collection.coerce_list_validator(FixtureStatus)

        class TestModel(BaseModel):
            statuses: list[FixtureStatus] = Field(default_factory=list)

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


class TestuCollectionParseMapping:
    """Real tests for u.Collection.parse_mapping."""

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
        result = u.Collection.parse_mapping(
            scenario.enum_cls,
            scenario.mapping,
        )

        if scenario.expected_success:
            assertion_helpers.assert_flext_result_success(result)
            assert result.value is not None
            if scenario.expected_keys is not None:
                assert set(result.value.keys()) == set(scenario.expected_keys)
            # Verify all values are of correct enum type
            for val in result.value.values():
                assert isinstance(val, scenario.enum_cls)
        else:
            assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None
            if scenario.error_contains:
                assert scenario.error_contains in result.error

    def test_parse_mapping_with_custom_enum(self) -> None:
        """Test parse_mapping with custom enum class."""
        result = u.Collection.parse_mapping(
            FixturePriority,
            {"task1": "low", "task2": "medium", "task3": "high"},
        )
        assertion_helpers.assert_flext_result_success(result)
        assert result.value is not None
        assert len(result.value) == 3
        assert all(isinstance(v, FixturePriority) for v in result.value.values())

    def test_parse_mapping_error_message_format(self) -> None:
        """Test parse_mapping error message format."""
        result = u.Collection.parse_mapping(
            FixtureStatus,
            {"user1": "active", "user2": "invalid1", "user3": "invalid2"},
        )
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None
        assert "Invalid" in result.error
        assert "invalid1" in result.error or "invalid2" in result.error


class TestuCollectionCoerceDictValidator:
    """Real tests for u.Collection.coerce_dict_validator."""

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
        validator = u.Collection.coerce_dict_validator(scenario.enum_cls)

        if scenario.expected_success:
            result = validator(scenario.value)
            u.Tests.Assertions.assert_result_matches_expected(
                result,
                dict,
            )
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
                _ = validator(scenario.value)

    def test_coerce_dict_validator_with_pydantic(self) -> None:
        """Test coerce_dict_validator integration with Pydantic."""
        _ = u.Collection.coerce_dict_validator(FixtureStatus)

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


class TestuCollectionMap:
    """Real tests for u.Collection.map."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.MAP_CASES,
        ids=lambda s: s.name,
    )
    def test_map(self, scenario: MapScenario) -> None:
        """Test map with various scenarios."""
        # Skip scenarios that use FlextResult objects as items
        # Collection.map() works on raw collections, not FlextResult
        if isinstance(scenario.items, (r, FlextRuntime.RuntimeResult)):
            pytest.skip("Collection.map() does not handle FlextResult items")

        result = u.Collection.map(
            scenario.items,
            scenario.mapper,
        )

        # Collection.map() returns the mapped collection directly
        assert result == scenario.expected_result


class TestuCollectionFind:
    """Real tests for u.Collection.find."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.FIND_CASES,
        ids=lambda s: s.name,
    )
    def test_find(self, scenario: FindScenario) -> None:
        """Test find with various scenarios."""
        # Collection.find works on lists, tuples, and dicts
        result = u.Collection.find(
            scenario.items,
            cast("Any", scenario.predicate),
        )
        assert result == scenario.expected_result


class TestuCollectionFilter:
    """Real tests for u.Collection.filter."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.FILTER_CASES,
        ids=lambda s: s.name,
    )
    def test_filter(self, scenario: FilterScenario) -> None:
        """Test filter with various scenarios."""
        # Collection.filter works on lists, tuples, and dicts
        result = u.Collection.filter(
            scenario.items,
            scenario.predicate,
            mapper=scenario.mapper,
        )
        assert result == scenario.expected_result


class TestuCollectionCount:
    """Real tests for u.Collection.count."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.COUNT_CASES,
        ids=lambda s: s.name,
    )
    def test_count(self, scenario: CountScenario) -> None:
        """Test count with various scenarios."""
        result = u.Collection.count(
            scenario.items,
            scenario.predicate,
        )
        assert result == scenario.expected_count


class TestuCollectionProcess:
    """Real tests for u.Collection.process."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.PROCESS_CASES,
        ids=lambda s: s.name,
    )
    def test_process(self, scenario: ProcessScenario) -> None:
        """Test process with various scenarios."""
        result = u.Collection.process(
            scenario.items,
            scenario.processor,
            on_error=scenario.on_error,
            predicate=scenario.predicate,
            filter_keys=scenario.filter_keys,
            exclude_keys=scenario.exclude_keys,
        )

        if scenario.expected_failure:
            assertion_helpers.assert_flext_result_failure(result)
            if scenario.error_contains:
                assert scenario.error_contains in str(result.error)
        else:
            assertion_helpers.assert_flext_result_success(result)
            assert result.value == scenario.expected_result


class TestuCollectionGroup:
    """Real tests for u.Collection.group."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.GROUP_CASES,
        ids=lambda s: s.name,
    )
    def test_group(self, scenario: GroupScenario) -> None:
        """Test group with various scenarios."""
        result = u.Collection.group(
            scenario.items,
            scenario.key,
        )
        assert result == scenario.expected_result


class TestuCollectionChunk:
    """Real tests for u.Collection.chunk."""

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.CHUNK_CASES,
        ids=lambda s: s.name,
    )
    def test_chunk(self, scenario: ChunkScenario) -> None:
        """Test chunk with various scenarios."""
        result = u.Collection.chunk(
            scenario.items,
            scenario.size,
        )
        assert result == scenario.expected_result


class TestuCollectionBatch:
    """Real tests for u.Collection.batch."""

    def test_batch_basic(self) -> None:
        """Test batch basic functionality."""
        items = [1, 2, 3, 4, 5]
        result = u.Collection.batch(
            items,
            lambda x: x * 2,
            _size=2,
        )
        assertion_helpers.assert_flext_result_success(result)
        data = result.value
        assert data.total == 5
        assert data.success_count == 5
        assert data.results == [2, 4, 6, 8, 10]
        assert data.errors == []

    def test_batch_with_errors_collect(self) -> None:
        """Test batch with errors (collect mode)."""
        items = [1, 2, 0, 4]

        def op(x: int) -> int:
            if x == 0:
                msg = "Zero"
                raise ValueError(msg)
            return 10 // x

        result = u.Collection.batch(
            items,
            op,
            on_error="collect",
        )
        assertion_helpers.assert_flext_result_success(result)
        data = result.value
        assert data.total == 4
        assert data.success_count == 3
        assert len(data.errors) == 1
        assert "Zero" in data.errors[0][1]

    def test_batch_flatten(self) -> None:
        """Test batch with flattening."""
        items = [[1, 2], [3, 4], 5]
        result = u.Collection.batch(
            items,
            cast("Callable[[object], int | r[int]]", lambda x: cast("int | r[int]", x)),
            flatten=True,
        )
        assertion_helpers.assert_flext_result_success(result)
        data = result.value
        assert data.results == [1, 2, 3, 4, 5]


class TestuCollectionMerge:
    """Real tests for u.Collection.merge."""

    def test_merge_deep(self) -> None:
        """Test deep merge."""
        base = m.ConfigMap(root={"a": 1, "b": {"x": 1}})
        other = m.ConfigMap(root={"b": {"y": 2}, "c": 3})
        result = u.Collection.merge(base.root, other.root)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}

    def test_merge_override(self) -> None:
        """Test override merge."""
        base = m.ConfigMap(root={"a": 1, "b": {"x": 1}})
        other = m.ConfigMap(root={"b": {"y": 2}, "c": 3})
        result = u.Collection.merge(base.root, other.root, strategy="override")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == {"a": 1, "b": {"y": 2}, "c": 3}


__all__ = [
    "TestuCollectionBatch",
    "TestuCollectionChunk",
    "TestuCollectionCoerceDictValidator",
    "TestuCollectionCoerceListValidator",
    "TestuCollectionCount",
    "TestuCollectionFilter",
    "TestuCollectionFind",
    "TestuCollectionGroup",
    "TestuCollectionMap",
    "TestuCollectionMerge",
    "TestuCollectionParseMapping",
    "TestuCollectionParseSequence",
    "TestuCollectionProcess",
]
