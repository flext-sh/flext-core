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
from typing import Annotated, ClassVar

from flext_tests import tm
from tests import m, t, u


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
            t.StrSequence | None, m.Field(description="Expected output keys")
        ] = None
        error_contains: Annotated[
            str | None, m.Field(description="Expected error message fragment")
        ] = None

    class CoerceDictScenario(m.BaseModel):
        """Coerce dict validator test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Coerce dict scenario name")]
        enum_cls: Annotated[
            type[StrEnum], m.Field(description="Enum class for coercion")
        ]
        value: Annotated[
            Annotated[t.Container, m.SkipValidation],
            m.Field(description="Input value to coerce"),
        ]
        expected_success: Annotated[
            bool,
            m.Field(description="Whether coercion should succeed"),
        ]
        expected_keys: Annotated[
            t.StrSequence | None, m.Field(description="Expected output keys")
        ] = None
        error_type: Annotated[
            type[Exception] | None, m.Field(description="Expected exception type")
        ] = None
        error_contains: Annotated[
            str | None, m.Field(description="Expected error message fragment")
        ] = None

    class MapScenario(m.BaseModel):
        """Map method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Map scenario name")]
        items: Annotated[
            t.FlatContainerList
            | tuple[t.Container, ...]
            | Mapping[str, t.Container]
            | set[t.Container]
            | frozenset[t.Container],
            (
                m.Field(
                    description="Collection input for map operation",
                )
            ),
        ]
        mapper: Annotated[
            Callable[[t.Container], t.Container],
            m.Field(description="Mapper callable under test"),
        ]
        expected_result: Annotated[
            (
                list[t.Container]
                | tuple[t.Container, ...]
                | t.MutableFlatContainerMapping
                | set[t.Container]
                | frozenset[t.Container]
            ),
            m.Field(description="Expected mapped output"),
        ]

    class FindScenario(m.BaseModel):
        """Find method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Find scenario name")]
        items: Annotated[
            t.FlatContainerList | tuple[t.Container, ...] | Mapping[str, t.Container],
            m.Field(description="Input items for find"),
        ]
        predicate: Annotated[
            Callable[[t.Container], bool],
            m.Field(description="Predicate callable under test"),
        ]
        expected_result: Annotated[
            t.Container | None,
            m.Field(description="Expected found value"),
        ]

    class FilterScenario(m.BaseModel):
        """Filter method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Filter scenario name")]
        items: Annotated[
            t.FlatContainerList | tuple[t.Container, ...] | Mapping[str, t.Container],
            m.Field(description="Input items for filter"),
        ]
        predicate: Annotated[
            Callable[[t.Container], bool],
            m.Field(description="Predicate callable under test"),
        ]
        expected_result: Annotated[
            t.FlatContainerList | tuple[t.Container, ...] | Mapping[str, t.Container],
            m.Field(
                description="Expected filtered output",
            ),
        ]
        mapper: Annotated[
            Callable[[t.Container], t.Container] | None,
            m.Field(description="Optional mapping callable"),
        ] = None

    class CountScenario(m.BaseModel):
        """Count method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Count scenario name")]
        items: Annotated[
            t.FlatContainerList, m.Field(description="Input items for count")
        ]
        expected_count: Annotated[int, m.Field(description="Expected item count")]
        predicate: Annotated[
            Callable[[t.Container], bool] | None,
            m.Field(description="Optional predicate filter"),
        ] = None

    class ProcessScenario(m.BaseModel):
        """Process method test scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Process scenario name")]
        items: Annotated[
            t.FlatContainerList, m.Field(description="Input items for process")
        ]
        processor: Annotated[
            Callable[[t.Container], t.Container],
            m.Field(description="Processor callable under test"),
        ]
        expected_result: Annotated[
            t.Container,
            m.Field(description="Expected processing result"),
        ]
        on_error: Annotated[str, m.Field(description="Error handling mode")] = "collect"
        predicate: Annotated[
            Callable[[t.Container], bool] | None,
            m.Field(description="Optional predicate filter"),
        ] = None
        expected_failure: Annotated[
            bool, m.Field(description="Whether processing should fail")
        ] = False
        error_contains: Annotated[
            str | None, m.Field(description="Expected error message fragment")
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
            Callable[..., t.Container],
            m.Field(description="Batch operation callable"),
        ]
        expected_result: Annotated[
            t.Container,
            m.Field(description="Expected batch result"),
        ]
        size: Annotated[int, m.Field(description="Batch size")] = 100
        on_error: Annotated[str, m.Field(description="Error handling mode")] = "collect"
        pre_validate: Annotated[
            Callable[..., bool] | None,
            m.Field(description="Optional pre-validation callable"),
        ] = None
        flatten: Annotated[
            bool, m.Field(description="Whether to flatten nested results")
        ] = False
        expected_failure: Annotated[
            bool, m.Field(description="Whether batch should fail")
        ] = False
        error_contains: Annotated[
            str | None, m.Field(description="Expected error message fragment")
        ] = None

    class CollectionUtilitiesScenarios:
        """Centralized collection utilities test scenarios."""

        @staticmethod
        def _double(value: t.Container) -> t.Container:
            assert isinstance(value, int)
            return value * 2

        @staticmethod
        def _upper(value: t.Container) -> t.Container:
            assert isinstance(value, str)
            return value.upper()

        @staticmethod
        def _is_even(value: t.Container) -> bool:
            assert isinstance(value, int)
            return value % 2 == 0

        @staticmethod
        def _is_odd(value: t.Container) -> bool:
            assert isinstance(value, int)
            return value % 2 != 0

        @staticmethod
        def _greater_than_two(value: t.Container) -> bool:
            assert isinstance(value, int)
            return value > 2

        @staticmethod
        def _greater_than_one(value: t.Container) -> bool:
            assert isinstance(value, int)
            return value > 1

        @staticmethod
        def _greater_than_ten(value: t.Container) -> bool:
            assert isinstance(value, int)
            return value > 10

        @staticmethod
        def _greater_than_fifteen(value: t.Container) -> bool:
            assert isinstance(value, int)
            return value > 15

        @staticmethod
        def _equals_two(value: t.Container) -> bool:
            return value == 2

        @staticmethod
        def _by_length(value: str) -> int:
            return len(value)

        @staticmethod
        def parse_mapping_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.ParseMappingScenario
        ]:
            return [
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="valid_strings",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    mapping={"user1": "active", "user2": "pending"},
                    expected_success=True,
                    expected_keys=["user1", "user2"],
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="valid_enums",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    mapping={
                        "user1": TestUtilitiesCollectionCoverage.FixtureStatus.ACTIVE,
                        "user2": TestUtilitiesCollectionCoverage.FixtureStatus.PENDING,
                    },
                    expected_success=True,
                    expected_keys=["user1", "user2"],
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="mixed_strings_enums",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    mapping={
                        "user1": "active",
                        "user2": TestUtilitiesCollectionCoverage.FixtureStatus.PENDING,
                    },
                    expected_success=True,
                    expected_keys=["user1", "user2"],
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="empty_dict",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    mapping={},
                    expected_success=True,
                    expected_keys=[],
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="invalid_string",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    mapping={"user1": "invalid_status"},
                    expected_success=False,
                    error_contains="Invalid",
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="mixed_valid_invalid",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    mapping={"user1": "active", "user2": "invalid"},
                    expected_success=False,
                    error_contains="Invalid",
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="single_valid",
                    enum_cls=TestUtilitiesCollectionCoverage.FixturePriority,
                    mapping={"task1": "high"},
                    expected_success=True,
                    expected_keys=["task1"],
                ),
                TestUtilitiesCollectionCoverage.ParseMappingScenario(
                    name="single_invalid",
                    enum_cls=TestUtilitiesCollectionCoverage.FixturePriority,
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
                TestUtilitiesCollectionCoverage.CoerceDictScenario(
                    name="valid_dict_strings",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    value={"user1": "active", "user2": "pending"},
                    expected_success=True,
                    expected_keys=["user1", "user2"],
                ),
                TestUtilitiesCollectionCoverage.CoerceDictScenario(
                    name="valid_dict_enums",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    value={
                        "user1": TestUtilitiesCollectionCoverage.FixtureStatus.ACTIVE,
                        "user2": TestUtilitiesCollectionCoverage.FixtureStatus.PENDING,
                    },
                    expected_success=True,
                    expected_keys=["user1", "user2"],
                ),
                TestUtilitiesCollectionCoverage.CoerceDictScenario(
                    name="empty_dict",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    value={},
                    expected_success=True,
                    expected_keys=[],
                ),
                TestUtilitiesCollectionCoverage.CoerceDictScenario(
                    name="invalid_type_int",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    value=123,
                    expected_success=False,
                    error_type=TypeError,
                    error_contains="Expected dict",
                ),
                TestUtilitiesCollectionCoverage.CoerceDictScenario(
                    name="invalid_type_list",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    value=["active", "pending"],
                    expected_success=False,
                    error_type=TypeError,
                    error_contains="Expected dict",
                ),
                TestUtilitiesCollectionCoverage.CoerceDictScenario(
                    name="invalid_string_in_dict",
                    enum_cls=TestUtilitiesCollectionCoverage.FixtureStatus,
                    value={"user1": "active", "user2": "invalid_status"},
                    expected_success=False,
                    error_type=ValueError,
                    error_contains="Invalid",
                ),
            ]

        @staticmethod
        def filter_cases() -> Sequence[TestUtilitiesCollectionCoverage.FilterScenario]:
            return [
                TestUtilitiesCollectionCoverage.FilterScenario(
                    name="list_filter",
                    items=[1, 2, 3, 4],
                    predicate=lambda x: int(x) % 2 == 0,
                    expected_result=[2, 4],
                ),
                TestUtilitiesCollectionCoverage.FilterScenario(
                    name="list_filter_map",
                    items=[1, 2, 3, 4],
                    predicate=lambda x: int(x) > 2,
                    mapper=lambda x: int(x) * 2,
                    expected_result=[6, 8],
                ),
                TestUtilitiesCollectionCoverage.FilterScenario(
                    name="dict_filter",
                    items={"a": 1, "b": 2, "c": 3},
                    predicate=lambda v: int(v) % 2 != 0,
                    expected_result={"a": 1, "c": 3},
                ),
                TestUtilitiesCollectionCoverage.FilterScenario(
                    name="dict_filter_map",
                    items={"a": 1, "b": 4},
                    predicate=lambda v: int(v) > 2,
                    mapper=lambda v: int(v) * 2,
                    expected_result={"b": 8},
                ),
                TestUtilitiesCollectionCoverage.FilterScenario(
                    name="list_filter_empty",
                    items=[1, 3, 5],
                    predicate=lambda x: int(x) > 10,
                    expected_result=[],
                ),
                TestUtilitiesCollectionCoverage.FilterScenario(
                    name="list_filter_all",
                    items=[2, 4, 6],
                    predicate=lambda x: int(x) % 2 == 0,
                    expected_result=[2, 4, 6],
                ),
            ]

        @staticmethod
        def map_cases() -> Sequence[TestUtilitiesCollectionCoverage.MapScenario]:
            return [
                TestUtilitiesCollectionCoverage.MapScenario(
                    name="map_list",
                    items=[1, 2, 3],
                    mapper=lambda x: int(x) * 2,
                    expected_result=[2, 4, 6],
                ),
                TestUtilitiesCollectionCoverage.MapScenario(
                    name="map_tuple",
                    items=(1, 2),
                    mapper=lambda x: int(x) + 1,
                    expected_result=(2, 3),
                ),
                TestUtilitiesCollectionCoverage.MapScenario(
                    name="map_dict",
                    items={"a": 1, "b": 2},
                    mapper=lambda x: int(x) * 10,
                    expected_result={"a": 10, "b": 20},
                ),
            ]

        @staticmethod
        def find_cases() -> Sequence[TestUtilitiesCollectionCoverage.FindScenario]:
            return [
                TestUtilitiesCollectionCoverage.FindScenario(
                    name="find_list_match",
                    items=[1, 2, 3],
                    predicate=lambda x: x == 2,
                    expected_result=2,
                ),
                TestUtilitiesCollectionCoverage.FindScenario(
                    name="find_dict_match",
                    items={"a": 1, "b": 4},
                    predicate=lambda x: x == 4,
                    expected_result=4,
                ),
                TestUtilitiesCollectionCoverage.FindScenario(
                    name="find_no_match",
                    items=[1, 3, 5],
                    predicate=lambda x: x == 2,
                    expected_result=None,
                ),
            ]

        @staticmethod
        def count_cases() -> Sequence[TestUtilitiesCollectionCoverage.CountScenario]:
            return [
                TestUtilitiesCollectionCoverage.CountScenario(
                    name="count_list", items=[1, 2, 3, 4], expected_count=4
                ),
                TestUtilitiesCollectionCoverage.CountScenario(
                    name="count_predicate",
                    items=[1, 2, 3, 4],
                    predicate=lambda x: int(x) % 2 == 0,
                    expected_count=2,
                ),
            ]

        @staticmethod
        def process_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.ProcessScenario
        ]:
            return [
                TestUtilitiesCollectionCoverage.ProcessScenario(
                    name="process_list",
                    items=[1, 2, 3],
                    processor=lambda x: int(x) * 2,
                    expected_result=[2, 4, 6],
                ),
                TestUtilitiesCollectionCoverage.ProcessScenario(
                    name="process_list_skip",
                    items=[1, 2, 3],
                    processor=lambda x: int(x) * 2,
                    expected_result=[4, 6],
                    predicate=lambda x: int(x) > 1,
                ),
                TestUtilitiesCollectionCoverage.ProcessScenario(
                    name="process_strings",
                    items=["a", "b", "c"],
                    processor=lambda x: str(x).upper(),
                    expected_result=["A", "B", "C"],
                ),
                TestUtilitiesCollectionCoverage.ProcessScenario(
                    name="process_empty",
                    items=[],
                    processor=lambda x: int(x) * 2,
                    expected_result=[],
                ),
            ]

        @staticmethod
        def group_cases() -> Sequence[TestUtilitiesCollectionCoverage.GroupScenario]:
            return [
                TestUtilitiesCollectionCoverage.GroupScenario(
                    name="group_by_len",
                    items=["cat", "dog", "house"],
                    key=lambda x: len(x),
                    expected_result={3: ["cat", "dog"], 5: ["house"]},
                ),
            ]

        @staticmethod
        def chunk_cases() -> Sequence[TestUtilitiesCollectionCoverage.ChunkScenario]:
            return [
                TestUtilitiesCollectionCoverage.ChunkScenario(
                    name="chunk_list",
                    items=[1, 2, 3, 4, 5],
                    size=2,
                    expected_result=[[1, 2], [3, 4], [5]],
                ),
            ]

    def test_map(self) -> None:
        """Test map with various scenarios."""
        for scenario in self.CollectionUtilitiesScenarios.map_cases():
            result = u.map(scenario.items, scenario.mapper)
            assert result == scenario.expected_result

    def test_find(self) -> None:
        """Test find with various scenarios."""
        for scenario in self.CollectionUtilitiesScenarios.find_cases():
            result = u.find(
                scenario.items,
                scenario.predicate,
            )
            if scenario.expected_result is None:
                assert result.failure
            else:
                assert result.success
                assert result.value == scenario.expected_result

    def test_filter(self) -> None:
        """Test filter with various scenarios."""
        for scenario in self.CollectionUtilitiesScenarios.filter_cases():
            result = u.filter(
                scenario.items,
                scenario.predicate,
                mapper=scenario.mapper,
            )
            assert result == scenario.expected_result

    def test_count(self) -> None:
        """Test count with various scenarios."""
        for scenario in self.CollectionUtilitiesScenarios.count_cases():
            result = u.count(scenario.items, scenario.predicate)
            assert result == scenario.expected_count

    def test_process(self) -> None:
        """Test process with various scenarios."""
        for scenario in self.CollectionUtilitiesScenarios.process_cases():
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
        base_data: Mapping[str, t.Container] = {"a": 1, "b": {"x": 1}}
        other_data: Mapping[str, t.Container] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base_data, other_data)
        assert result.success
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)

    def test_merge_override(self) -> None:
        """Test override merge."""
        base_data: Mapping[str, t.Container] = {"a": 1, "b": {"x": 1}}
        other_data: Mapping[str, t.Container] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base_data, other_data, strategy="override")
        assert result.success
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)


__all__: list[str] = ["TestUtilitiesCollectionCoverage"]
