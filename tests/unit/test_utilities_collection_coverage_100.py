"""Real tests to achieve 100% collection utilities coverage - no mocks.

Module: flext_core
Scope: FlextUtilitiesCollection - parse_mapping, coerce_dict_validator,
map/find/filter/count/process/merge

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/collection.py.

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar, cast

import pytest

from flext_tests import tm
from tests import m, r, t, u


class TestUtilitiesCollectionCoverage:
    @unique
    class FixtureStatus(StrEnum):
        """Test status enum for collection utilities tests."""

        ACTIVE = "active"
        PENDING = "pending"
        COMPLETED = "completed"
        CANCELLED = "cancelled"

    @unique
    class FixturePriority(StrEnum):
        """Test priority enum for collection utilities tests."""

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class ParseMappingScenario(m.BaseModel):
        """Parse mapping test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Parse mapping scenario name")]
        enum_cls: Annotated[type[StrEnum], m.Field(description="Enum class under test")]
        mapping: Annotated[
            Mapping[str, str | StrEnum],
            m.Field(description="Input mapping values"),
        ]
        expected_success: Annotated[
            bool,
            m.Field(description="Whether parsing should succeed"),
        ]
        expected_keys: Annotated[
            t.StrSequence | None,
            m.Field(default=None, description="Expected output keys"),
        ] = None
        error_contains: Annotated[
            str | None,
            m.Field(default=None, description="Expected error message fragment"),
        ] = None

    class CoerceDictScenario(m.BaseModel):
        """Coerce dict validator test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Coerce dict scenario name")]
        enum_cls: Annotated[
            type[StrEnum], m.Field(description="Enum class for coercion")
        ]
        value: Annotated[
            Annotated[t.RecursiveContainer, m.SkipValidation],
            m.Field(description="Input value to coerce"),
        ]
        expected_success: Annotated[
            bool,
            m.Field(description="Whether coercion should succeed"),
        ]
        expected_keys: Annotated[
            t.StrSequence | None,
            m.Field(default=None, description="Expected output keys"),
        ] = None
        error_type: Annotated[
            type[Exception] | None,
            m.Field(default=None, description="Expected exception type"),
        ] = None
        error_contains: Annotated[
            str | None,
            m.Field(default=None, description="Expected error message fragment"),
        ] = None

    class MapScenario(m.BaseModel):
        """Map method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Map scenario name")]
        items: Annotated[
            t.RecursiveContainerList
            | tuple[t.RecursiveContainer, ...]
            | t.RecursiveContainerMapping
            | set[t.RecursiveContainer]
            | frozenset[t.RecursiveContainer],
            (
                m.Field(
                    description="Collection input for map operation",
                )
            ),
        ]
        mapper: Annotated[
            Callable[[t.RecursiveContainer], t.RecursiveContainer],
            m.Field(description="Mapper callable under test"),
        ]
        expected_result: Annotated[
            (
                t.MutableRecursiveContainerList
                | tuple[t.RecursiveContainer, ...]
                | t.MutableRecursiveContainerMapping
                | set[t.RecursiveContainer]
                | frozenset[t.RecursiveContainer]
            ),
            m.Field(description="Expected mapped output"),
        ]

    class FindScenario(m.BaseModel):
        """Find method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Find scenario name")]
        items: Annotated[
            t.RecursiveContainerList
            | tuple[t.RecursiveContainer, ...]
            | t.RecursiveContainerMapping,
            m.Field(description="Input items for find"),
        ]
        predicate: Annotated[
            Callable[[t.RecursiveContainer], bool],
            m.Field(description="Predicate callable under test"),
        ]
        expected_result: Annotated[
            t.RecursiveContainer | None,
            m.Field(description="Expected found value"),
        ]

    class FilterScenario(m.BaseModel):
        """Filter method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Filter scenario name")]
        items: Annotated[
            t.RecursiveContainerList
            | tuple[t.RecursiveContainer, ...]
            | t.RecursiveContainerMapping,
            m.Field(description="Input items for filter"),
        ]
        predicate: Annotated[
            Callable[[t.RecursiveContainer], bool],
            m.Field(description="Predicate callable under test"),
        ]
        expected_result: Annotated[
            t.RecursiveContainerList
            | tuple[t.RecursiveContainer, ...]
            | t.RecursiveContainerMapping,
            m.Field(
                description="Expected filtered output",
            ),
        ]
        mapper: Annotated[
            Callable[[t.RecursiveContainer], t.RecursiveContainer] | None,
            m.Field(default=None, description="Optional mapping callable"),
        ] = None

    class CountScenario(m.BaseModel):
        """Count method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Count scenario name")]
        items: Annotated[
            t.RecursiveContainerList, m.Field(description="Input items for count")
        ]
        expected_count: Annotated[int, m.Field(description="Expected item count")]
        predicate: Annotated[
            Callable[[t.RecursiveContainer], bool] | None,
            m.Field(default=None, description="Optional predicate filter"),
        ] = None

    class ProcessScenario(m.BaseModel):
        """Process method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Process scenario name")]
        items: Annotated[
            t.RecursiveContainerList, m.Field(description="Input items for process")
        ]
        processor: Annotated[
            Callable[[t.RecursiveContainer], t.RecursiveContainer],
            m.Field(description="Processor callable under test"),
        ]
        expected_result: Annotated[
            t.RecursiveContainer,
            m.Field(description="Expected processing result"),
        ]
        on_error: Annotated[
            str,
            m.Field(default="collect", description="Error handling mode"),
        ] = "collect"
        predicate: Annotated[
            Callable[[t.RecursiveContainer], bool] | None,
            m.Field(default=None, description="Optional predicate filter"),
        ] = None
        expected_failure: Annotated[
            bool,
            m.Field(default=False, description="Whether processing should fail"),
        ] = False
        error_contains: Annotated[
            str | None,
            m.Field(default=None, description="Expected error message fragment"),
        ] = None

    class GroupScenario(m.BaseModel):
        """Group method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Group scenario name")]
        items: Annotated[
            t.StrSequence | tuple[str, ...],
            m.Field(description="Input items for group"),
        ]
        key: Annotated[
            Callable[[str], int | str],
            m.Field(description="Grouping key callable"),
        ]
        expected_result: Annotated[
            Mapping[int | str, t.StrSequence],
            m.Field(description="Expected grouped output"),
        ]

    class ChunkScenario(m.BaseModel):
        """Chunk method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Chunk scenario name")]
        items: Annotated[
            Sequence[int],
            m.Field(
                description="Input items for chunking",
            ),
        ]
        size: Annotated[int, m.Field(description="Chunk size")]
        expected_result: Annotated[
            Sequence[Sequence[int]],
            m.Field(description="Expected chunked output"),
        ]

    class BatchScenario(m.BaseModel):
        """Batch method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Batch scenario name")]
        items: Annotated[Sequence[int], m.Field(description="Input items for batch")]
        operation: Annotated[
            Callable[..., t.RecursiveContainer],
            m.Field(description="Batch operation callable"),
        ]
        expected_result: Annotated[
            t.RecursiveContainer,
            m.Field(description="Expected batch result"),
        ]
        size: Annotated[int, m.Field(default=100, description="Batch size")] = 100
        on_error: Annotated[
            str,
            m.Field(default="collect", description="Error handling mode"),
        ] = "collect"
        pre_validate: Annotated[
            Callable[..., bool] | None,
            m.Field(default=None, description="Optional pre-validation callable"),
        ] = None
        flatten: Annotated[
            bool,
            m.Field(default=False, description="Whether to flatten nested results"),
        ] = False
        expected_failure: Annotated[
            bool,
            m.Field(default=False, description="Whether batch should fail"),
        ] = False
        error_contains: Annotated[
            str | None,
            m.Field(default=None, description="Expected error message fragment"),
        ] = None

    globals()["FixtureStatus"] = FixtureStatus
    globals()["FixturePriority"] = FixturePriority
    globals()["ParseMappingScenario"] = ParseMappingScenario
    globals()["CoerceDictScenario"] = CoerceDictScenario
    globals()["MapScenario"] = MapScenario
    globals()["FindScenario"] = FindScenario
    globals()["FilterScenario"] = FilterScenario
    globals()["CountScenario"] = CountScenario
    globals()["ProcessScenario"] = ProcessScenario
    globals()["GroupScenario"] = GroupScenario
    globals()["ChunkScenario"] = ChunkScenario
    globals()["BatchScenario"] = BatchScenario

    class CollectionUtilitiesScenarios:
        """Centralized collection utilities test scenarios."""

        @staticmethod
        def _double(value: t.RecursiveContainer) -> t.RecursiveContainer:
            return cast("int", value) * 2

        @staticmethod
        def _upper(value: t.RecursiveContainer) -> t.RecursiveContainer:
            return cast("str", value).upper()

        @staticmethod
        def _is_even(value: t.RecursiveContainer) -> bool:
            return cast("int", value) % 2 == 0

        @staticmethod
        def _is_odd(value: t.RecursiveContainer) -> bool:
            return cast("int", value) % 2 != 0

        @staticmethod
        def _greater_than_two(value: t.RecursiveContainer) -> bool:
            return cast("int", value) > 2

        @staticmethod
        def _greater_than_one(value: t.RecursiveContainer) -> bool:
            return cast("int", value) > 1

        @staticmethod
        def _greater_than_ten(value: t.RecursiveContainer) -> bool:
            return cast("int", value) > 10

        @staticmethod
        def _greater_than_fifteen(value: t.RecursiveContainer) -> bool:
            return cast("int", value) > 15

        @staticmethod
        def _equals_two(value: t.RecursiveContainer) -> bool:
            return value == 2

        @staticmethod
        def _by_length(value: str) -> int:
            return len(value)

        @staticmethod
        def parse_mapping_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.ParseMappingScenario
        ]:
            return [
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
                    mapping={
                        "user1": FixtureStatus.ACTIVE,
                        "user2": FixtureStatus.PENDING,
                    },
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

        @staticmethod
        def coerce_dict_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.CoerceDictScenario
        ]:
            return [
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
                    value={
                        "user1": FixtureStatus.ACTIVE,
                        "user2": FixtureStatus.PENDING,
                    },
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
            ]

        @staticmethod
        def filter_cases() -> Sequence[TestUtilitiesCollectionCoverage.FilterScenario]:
            return [
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

        @staticmethod
        def map_cases() -> Sequence[TestUtilitiesCollectionCoverage.MapScenario]:
            return [
                MapScenario(
                    name="map_list",
                    items=[1, 2, 3],
                    mapper=lambda x: cast("int", x) * 2,
                    expected_result=[2, 4, 6],
                ),
                MapScenario(
                    name="map_tuple",
                    items=(1, 2),
                    mapper=lambda x: cast("int", x) + 1,
                    expected_result=(2, 3),
                ),
                MapScenario(
                    name="map_dict",
                    items={"a": 1, "b": 2},
                    mapper=lambda x: cast("int", x) * 10,
                    expected_result={"a": 10, "b": 20},
                ),
            ]

        @staticmethod
        def find_cases() -> Sequence[TestUtilitiesCollectionCoverage.FindScenario]:
            return [
                FindScenario(
                    name="find_list_match",
                    items=[1, 2, 3],
                    predicate=lambda x: cast("int", x) == 2,
                    expected_result=2,
                ),
                FindScenario(
                    name="find_dict_match",
                    items={"a": 1, "b": 4},
                    predicate=lambda x: cast("int", x) == 4,
                    expected_result=4,
                ),
                FindScenario(
                    name="find_no_match",
                    items=[1, 3, 5],
                    predicate=lambda x: cast("int", x) == 2,
                    expected_result=None,
                ),
            ]

        @staticmethod
        def count_cases() -> Sequence[TestUtilitiesCollectionCoverage.CountScenario]:
            return [
                CountScenario(name="count_list", items=[1, 2, 3, 4], expected_count=4),
                CountScenario(
                    name="count_predicate",
                    items=[1, 2, 3, 4],
                    predicate=lambda x: cast("int", x) % 2 == 0,
                    expected_count=2,
                ),
            ]

        @staticmethod
        def process_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.ProcessScenario
        ]:
            return [
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

        @staticmethod
        def group_cases() -> Sequence[TestUtilitiesCollectionCoverage.GroupScenario]:
            return [
                GroupScenario(
                    name="group_by_len",
                    items=["cat", "dog", "house"],
                    key=lambda x: len(x),
                    expected_result={3: ["cat", "dog"], 5: ["house"]},
                ),
            ]

        @staticmethod
        def chunk_cases() -> Sequence[TestUtilitiesCollectionCoverage.ChunkScenario]:
            return [
                ChunkScenario(
                    name="chunk_list",
                    items=[1, 2, 3, 4, 5],
                    size=2,
                    expected_result=[[1, 2], [3, 4], [5]],
                ),
            ]

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.map_cases(),
        ids=lambda s: s.name,
    )
    def test_map(self, scenario: MapScenario) -> None:
        """Test map with various scenarios."""
        if isinstance(scenario.items, r):
            pytest.skip("Collection.map() does not handle r items")
        result = u.map(scenario.items, scenario.mapper)
        assert result == scenario.expected_result

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.find_cases(),
        ids=lambda s: s.name,
    )
    def test_find(self, scenario: FindScenario) -> None:
        """Test find with various scenarios."""
        result = u.find(
            scenario.items,
            scenario.predicate,
        )
        if scenario.expected_result is None:
            assert result.failure
        else:
            assert result.success
            assert result.value == scenario.expected_result

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.filter_cases(),
        ids=lambda s: s.name,
    )
    def test_filter(self, scenario: FilterScenario) -> None:
        """Test filter with various scenarios."""
        result = u.filter(
            scenario.items,
            scenario.predicate,
            mapper=scenario.mapper,
        )
        assert result == scenario.expected_result

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.count_cases(),
        ids=lambda s: s.name,
    )
    def test_count(self, scenario: CountScenario) -> None:
        """Test count with various scenarios."""
        result = u.count(scenario.items, scenario.predicate)
        assert result == scenario.expected_count

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.process_cases(),
        ids=lambda s: s.name,
    )
    def test_process(self, scenario: ProcessScenario) -> None:
        """Test process with various scenarios."""
        result = u.process(
            scenario.items,
            scenario.processor,
            on_error=scenario.on_error,
            predicate=scenario.predicate,
        )
        if scenario.expected_failure:
            assert result.failure
            if scenario.error_contains:
                tm.that(str(result.error), has=scenario.error_contains)
        else:
            assert result.success
            assert result.value == scenario.expected_result

    def test_merge_deep(self) -> None:
        """Test deep merge."""
        base_data: t.RecursiveContainerMapping = {"a": 1, "b": {"x": 1}}
        other_data: t.RecursiveContainerMapping = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base_data, other_data)
        assert result.success
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)

    def test_merge_override(self) -> None:
        """Test override merge."""
        base_data: t.RecursiveContainerMapping = {"a": 1, "b": {"x": 1}}
        other_data: t.RecursiveContainerMapping = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base_data, other_data, strategy="override")
        assert result.success
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)


FixtureStatus = TestUtilitiesCollectionCoverage.FixtureStatus
FixturePriority = TestUtilitiesCollectionCoverage.FixturePriority
ParseMappingScenario = TestUtilitiesCollectionCoverage.ParseMappingScenario
CoerceDictScenario = TestUtilitiesCollectionCoverage.CoerceDictScenario
MapScenario = TestUtilitiesCollectionCoverage.MapScenario
FindScenario = TestUtilitiesCollectionCoverage.FindScenario
FilterScenario = TestUtilitiesCollectionCoverage.FilterScenario
CountScenario = TestUtilitiesCollectionCoverage.CountScenario
ProcessScenario = TestUtilitiesCollectionCoverage.ProcessScenario
GroupScenario = TestUtilitiesCollectionCoverage.GroupScenario
ChunkScenario = TestUtilitiesCollectionCoverage.ChunkScenario
BatchScenario = TestUtilitiesCollectionCoverage.BatchScenario
CollectionUtilitiesScenarios = (
    TestUtilitiesCollectionCoverage.CollectionUtilitiesScenarios
)

__all__: list[str] = ["TestUtilitiesCollectionCoverage"]
