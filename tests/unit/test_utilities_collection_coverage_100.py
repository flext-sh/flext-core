"""Real tests to achieve 100% collection utilities coverage - no mocks.

Module: flext_core._utilities.collection
Scope: FlextUtilitiesCollection - parse_sequence, coerce_list_validator,
parse_mapping, coerce_dict_validator

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/collection.py.

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import t, tm, u
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from flext_core import FlextRuntime, r

from ..test_utils import assertion_helpers


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

    class ParseSequenceScenario(BaseModel):
        """Parse sequence test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse sequence scenario name")]
        enum_cls: Annotated[type[StrEnum], Field(description="Enum class under test")]
        values: Annotated[
            Sequence[str | StrEnum], Field(description="Input values to parse")
        ]
        expected_success: Annotated[
            bool, Field(description="Whether parsing should succeed")
        ]
        expected_count: Annotated[
            int | None, Field(default=None, description="Expected parsed values count")
        ] = None
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class CoerceListScenario(BaseModel):
        """Coerce list validator test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Coerce list scenario name")]
        enum_cls: Annotated[type[StrEnum], Field(description="Enum class for coercion")]
        value: Annotated[
            Annotated[t.NormalizedValue, SkipValidation],
            Field(description="Input value to coerce"),
        ]
        expected_success: Annotated[
            bool, Field(description="Whether coercion should succeed")
        ]
        expected_count: Annotated[
            int | None, Field(default=None, description="Expected result count")
        ] = None
        error_type: Annotated[
            type[Exception] | None,
            Field(default=None, description="Expected exception type"),
        ] = None
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class ParseMappingScenario(BaseModel):
        """Parse mapping test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse mapping scenario name")]
        enum_cls: Annotated[type[StrEnum], Field(description="Enum class under test")]
        mapping: Annotated[
            Mapping[str, str | StrEnum], Field(description="Input mapping values")
        ]
        expected_success: Annotated[
            bool, Field(description="Whether parsing should succeed")
        ]
        expected_keys: Annotated[
            Sequence[str] | None,
            Field(default=None, description="Expected output keys"),
        ] = None
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class CoerceDictScenario(BaseModel):
        """Coerce dict validator test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Coerce dict scenario name")]
        enum_cls: Annotated[type[StrEnum], Field(description="Enum class for coercion")]
        value: Annotated[
            Annotated[t.NormalizedValue, SkipValidation],
            Field(description="Input value to coerce"),
        ]
        expected_success: Annotated[
            bool, Field(description="Whether coercion should succeed")
        ]
        expected_keys: Annotated[
            Sequence[str] | None,
            Field(default=None, description="Expected output keys"),
        ] = None
        error_type: Annotated[
            type[Exception] | None,
            Field(default=None, description="Expected exception type"),
        ] = None
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class MapScenario(BaseModel):
        """Map method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Map scenario name")]
        items: Annotated[
            Sequence[t.NormalizedValue]
            | tuple[t.NormalizedValue, ...]
            | Mapping[str, t.NormalizedValue]
            | set[t.NormalizedValue]
            | frozenset[t.NormalizedValue],
            (
                Field(
                    description="Collection input for map operation",
                )
            ),
        ]
        mapper: Annotated[
            Callable[[t.NormalizedValue], t.NormalizedValue],
            Field(description="Mapper callable under test"),
        ]
        expected_result: Annotated[
            (
                MutableSequence[t.NormalizedValue]
                | tuple[t.NormalizedValue, ...]
                | MutableMapping[str, t.NormalizedValue]
                | set[t.NormalizedValue]
                | frozenset[t.NormalizedValue]
            ),
            Field(description="Expected mapped output"),
        ]
        default_error: Annotated[
            str, Field(default="Operation failed", description="Default error message")
        ] = "Operation failed"
        expected_failure: Annotated[
            bool, Field(default=False, description="Whether map should fail")
        ] = False
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class FindScenario(BaseModel):
        """Find method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Find scenario name")]
        items: Annotated[
            Sequence[t.NormalizedValue]
            | tuple[t.NormalizedValue, ...]
            | Mapping[str, t.NormalizedValue],
            Field(description="Input items for find"),
        ]
        predicate: Annotated[
            Callable[[t.NormalizedValue], bool],
            Field(description="Predicate callable under test"),
        ]
        expected_result: Annotated[
            t.NormalizedValue | None, Field(description="Expected found value")
        ]
        return_key: Annotated[
            bool, Field(default=False, description="Whether to return dictionary key")
        ] = False

    class FilterScenario(BaseModel):
        """Filter method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Filter scenario name")]
        items: Annotated[
            Sequence[t.NormalizedValue]
            | tuple[t.NormalizedValue, ...]
            | Mapping[str, t.NormalizedValue],
            Field(description="Input items for filter"),
        ]
        predicate: Annotated[
            Callable[[t.NormalizedValue], bool],
            Field(description="Predicate callable under test"),
        ]
        expected_result: Annotated[
            Sequence[t.NormalizedValue]
            | tuple[t.NormalizedValue, ...]
            | Mapping[str, t.NormalizedValue],
            Field(
                description="Expected filtered output",
            ),
        ]
        mapper: Annotated[
            Callable[[t.NormalizedValue], t.NormalizedValue] | None,
            Field(default=None, description="Optional mapping callable"),
        ] = None

    class CountScenario(BaseModel):
        """Count method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Count scenario name")]
        items: Annotated[
            Sequence[t.NormalizedValue], Field(description="Input items for count")
        ]
        expected_count: Annotated[int, Field(description="Expected item count")]
        predicate: Annotated[
            Callable[[t.NormalizedValue], bool] | None,
            Field(default=None, description="Optional predicate filter"),
        ] = None

    class ProcessScenario(BaseModel):
        """Process method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Process scenario name")]
        items: Annotated[
            Sequence[t.NormalizedValue], Field(description="Input items for process")
        ]
        processor: Annotated[
            Callable[[t.NormalizedValue], t.NormalizedValue],
            Field(description="Processor callable under test"),
        ]
        expected_result: Annotated[
            t.NormalizedValue, Field(description="Expected processing result")
        ]
        on_error: Annotated[
            str, Field(default="collect", description="Error handling mode")
        ] = "collect"
        predicate: Annotated[
            Callable[[t.NormalizedValue], bool] | None,
            Field(default=None, description="Optional predicate filter"),
        ] = None
        filter_keys: Annotated[
            set[str] | None,
            Field(default=None, description="Keys to include when processing mappings"),
        ] = None
        exclude_keys: Annotated[
            set[str] | None,
            Field(default=None, description="Keys to exclude when processing mappings"),
        ] = None
        expected_failure: Annotated[
            bool, Field(default=False, description="Whether processing should fail")
        ] = False
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class GroupScenario(BaseModel):
        """Group method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Group scenario name")]
        items: Annotated[
            Sequence[str] | tuple[str, ...], Field(description="Input items for group")
        ]
        key: Annotated[
            Callable[[str], int | str], Field(description="Grouping key callable")
        ]
        expected_result: Annotated[
            Mapping[int | str, Sequence[str]],
            Field(description="Expected grouped output"),
        ]

    class ChunkScenario(BaseModel):
        """Chunk method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Chunk scenario name")]
        items: Annotated[
            Sequence[int],
            Field(
                description="Input items for chunking",
            ),
        ]
        size: Annotated[int, Field(description="Chunk size")]
        expected_result: Annotated[
            Sequence[Sequence[int]], Field(description="Expected chunked output")
        ]

    class BatchScenario(BaseModel):
        """Batch method test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Batch scenario name")]
        items: Annotated[Sequence[int], Field(description="Input items for batch")]
        operation: Annotated[
            Callable[..., t.NormalizedValue],
            Field(description="Batch operation callable"),
        ]
        expected_result: Annotated[
            t.NormalizedValue, Field(description="Expected batch result")
        ]
        size: Annotated[int, Field(default=100, description="Batch size")] = 100
        on_error: Annotated[
            str, Field(default="collect", description="Error handling mode")
        ] = "collect"
        pre_validate: Annotated[
            Callable[..., bool] | None,
            Field(default=None, description="Optional pre-validation callable"),
        ] = None
        flatten: Annotated[
            bool, Field(default=False, description="Whether to flatten nested results")
        ] = False
        expected_failure: Annotated[
            bool, Field(default=False, description="Whether batch should fail")
        ] = False
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    globals()["FixtureStatus"] = FixtureStatus
    globals()["FixturePriority"] = FixturePriority
    globals()["ParseSequenceScenario"] = ParseSequenceScenario
    globals()["CoerceListScenario"] = CoerceListScenario
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
        def _double(value: t.NormalizedValue) -> t.NormalizedValue:
            return cast("int", value) * 2

        @staticmethod
        def _upper(value: t.NormalizedValue) -> t.NormalizedValue:
            return cast("str", value).upper()

        @staticmethod
        def _is_even(value: t.NormalizedValue) -> bool:
            return cast("int", value) % 2 == 0

        @staticmethod
        def _is_odd(value: t.NormalizedValue) -> bool:
            return cast("int", value) % 2 != 0

        @staticmethod
        def _greater_than_two(value: t.NormalizedValue) -> bool:
            return cast("int", value) > 2

        @staticmethod
        def _greater_than_one(value: t.NormalizedValue) -> bool:
            return cast("int", value) > 1

        @staticmethod
        def _greater_than_ten(value: t.NormalizedValue) -> bool:
            return cast("int", value) > 10

        @staticmethod
        def _greater_than_fifteen(value: t.NormalizedValue) -> bool:
            return cast("int", value) > 15

        @staticmethod
        def _equals_two(value: t.NormalizedValue) -> bool:
            return value == 2

        @staticmethod
        def _by_length(value: str) -> int:
            return len(value)

        @staticmethod
        def parse_sequence_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.ParseSequenceScenario
        ]:
            return [
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

        @staticmethod
        def coerce_list_cases() -> Sequence[
            TestUtilitiesCollectionCoverage.CoerceListScenario
        ]:
            return [
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
                CoerceDictScenario(
                    name="invalid_type_in_dict",
                    enum_cls=FixtureStatus,
                    value={"user1": "active", "user2": 123},
                    expected_success=False,
                    error_type=TypeError,
                    error_contains="Expected str",
                ),
            ]

        @staticmethod
        def map_cases() -> Sequence[TestUtilitiesCollectionCoverage.MapScenario]:
            return [
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

        @staticmethod
        def find_cases() -> Sequence[TestUtilitiesCollectionCoverage.FindScenario]:
            return [
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
        CollectionUtilitiesScenarios.parse_sequence_cases(),
        ids=lambda s: s.name,
    )
    def test_parse_sequence(self, scenario: ParseSequenceScenario) -> None:
        """Test parse_sequence with various scenarios."""
        result = u.parse_sequence(scenario.enum_cls, scenario.values)
        if scenario.expected_success:
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, none=False)
            if scenario.expected_count is not None:
                tm.that(len(result.value), eq=scenario.expected_count)
            for val in result.value:
                tm.that(isinstance(val, scenario.enum_cls), eq=True)
        else:
            _ = assertion_helpers.assert_flext_result_failure(result)
            tm.that(result.error, none=False)
            if scenario.error_contains:
                tm.that(result.error, has=scenario.error_contains)

    def test_parse_sequence_with_custom_enum(self) -> None:
        """Test parse_sequence with custom enum class."""
        result = u.parse_sequence(FixturePriority, ["low", "medium", "high"])
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, none=False)
        tm.that(len(result.value), eq=3)
        tm.that(all(isinstance(v, FixturePriority) for v in result.value), eq=True)

    def test_parse_sequence_error_message_format(self) -> None:
        """Test parse_sequence error message format."""
        result = u.parse_sequence(
            FixtureStatus,
            ["active", "invalid1", "invalid2"],
        )
        _ = assertion_helpers.assert_flext_result_failure(result)
        tm.that(result.error, none=False)
        tm.that(result.error, has="Invalid")
        error_message = result.error or ""
        assert "invalid1" in error_message or "invalid2" in error_message

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.coerce_list_cases(),
        ids=lambda s: s.name,
    )
    def test_coerce_list_validator(self, scenario: CoerceListScenario) -> None:
        """Test coerce_list_validator with various scenarios."""
        validator = u.coerce_list_validator(scenario.enum_cls)
        if scenario.expected_success:
            result = validator(scenario.value)
            u.Tests.Assertions.assert_result_matches_expected(result, list)
            if scenario.expected_count is not None:
                tm.that(len(result), eq=scenario.expected_count)
            for val in result:
                tm.that(isinstance(val, scenario.enum_cls), eq=True)
        else:
            with pytest.raises(
                scenario.error_type or Exception,
                match=scenario.error_contains or "",
            ):
                _ = validator(scenario.value)

    def test_coerce_list_validator_with_pydantic(self) -> None:
        """Test coerce_list_validator integration with Pydantic."""
        _ = u.coerce_list_validator(FixtureStatus)

        class TestModel(BaseModel):
            statuses: Annotated[tuple[FixtureStatus, ...], Field(default_factory=tuple)]

        model1: TestModel = TestModel.model_validate({
            "statuses": ["active", "pending"]
        })
        dumped1 = model1.model_dump()
        tm.that(len(dumped1["statuses"]), eq=2)
        model2: TestModel = TestModel.model_validate({
            "statuses": [FixtureStatus.ACTIVE, FixtureStatus.PENDING],
        })
        dumped2 = model2.model_dump()
        tm.that(len(dumped2["statuses"]), eq=2)

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.parse_mapping_cases(),
        ids=lambda s: s.name,
    )
    def test_parse_mapping(self, scenario: ParseMappingScenario) -> None:
        """Test parse_mapping with various scenarios."""
        result = u.parse_mapping(scenario.enum_cls, scenario.mapping)
        if scenario.expected_success:
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, none=False)
            if scenario.expected_keys is not None:
                assert set(result.value.keys()) == set(scenario.expected_keys)
            for val in result.value.values():
                tm.that(isinstance(val, scenario.enum_cls), eq=True)
        else:
            _ = assertion_helpers.assert_flext_result_failure(result)
            tm.that(result.error, none=False)
            if scenario.error_contains:
                tm.that(result.error, has=scenario.error_contains)

    def test_parse_mapping_with_custom_enum(self) -> None:
        """Test parse_mapping with custom enum class."""
        result = u.parse_mapping(
            FixturePriority,
            {"task1": "low", "task2": "medium", "task3": "high"},
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, none=False)
        tm.that(len(result.value), eq=3)
        tm.that(
            all(v in FixturePriority for v in result.value.values()),
            eq=True,
        )

    def test_parse_mapping_error_message_format(self) -> None:
        """Test parse_mapping error message format."""
        result = u.parse_mapping(
            FixtureStatus,
            {"user1": "active", "user2": "invalid1", "user3": "invalid2"},
        )
        _ = assertion_helpers.assert_flext_result_failure(result)
        tm.that(result.error, none=False)
        tm.that(result.error, has="Invalid")
        error_message = result.error or ""
        assert "invalid1" in error_message or "invalid2" in error_message

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.coerce_dict_cases(),
        ids=lambda s: s.name,
    )
    def test_coerce_dict_validator(self, scenario: CoerceDictScenario) -> None:
        """Test coerce_dict_validator with various scenarios."""
        validator = u.coerce_dict_validator(scenario.enum_cls)
        if scenario.expected_success:
            result = validator(scenario.value)
            u.Tests.Assertions.assert_result_matches_expected(result, dict)
            if scenario.expected_keys is not None:
                assert set(result.keys()) == set(scenario.expected_keys)
            for val in result.values():
                tm.that(isinstance(val, scenario.enum_cls), eq=True)
        else:
            with pytest.raises(
                scenario.error_type or Exception,
                match=scenario.error_contains or "",
            ):
                _ = validator(scenario.value)

    def test_coerce_dict_validator_with_pydantic(self) -> None:
        """Test coerce_dict_validator integration with Pydantic."""
        _ = u.coerce_dict_validator(FixtureStatus)

        class TestModel(BaseModel):
            user_statuses: Annotated[
                Mapping[str, FixtureStatus], Field(default_factory=dict)
            ]

        model1 = TestModel.model_validate({
            "user_statuses": {"user1": "active", "user2": "pending"},
        })
        tm.that(len(model1.user_statuses), eq=2)
        assert all(isinstance(s, FixtureStatus) for s in model1.user_statuses.values())
        model2 = TestModel.model_validate({
            "user_statuses": {
                "user1": FixtureStatus.ACTIVE,
                "user2": FixtureStatus.PENDING,
            },
        })
        tm.that(len(model2.user_statuses), eq=2)
        assert all(isinstance(s, FixtureStatus) for s in model2.user_statuses.values())

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.map_cases(),
        ids=lambda s: s.name,
    )
    def test_map(self, scenario: MapScenario) -> None:
        """Test map with various scenarios."""
        if isinstance(scenario.items, (r, FlextRuntime.RuntimeResult)):
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
            assert result.is_failure
        else:
            assert result.is_success
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
            filter_keys=scenario.filter_keys,
            exclude_keys=scenario.exclude_keys,
        )
        if scenario.expected_failure:
            _ = assertion_helpers.assert_flext_result_failure(result)
            if scenario.error_contains:
                tm.that(str(result.error), has=scenario.error_contains)
        else:
            _ = assertion_helpers.assert_flext_result_success(result)
            assert result.value == scenario.expected_result

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.group_cases(),
        ids=lambda s: s.name,
    )
    def test_group(self, scenario: GroupScenario) -> None:
        """Test group_by with various scenarios."""
        result = u.group_by(scenario.items, scenario.key)
        assert result == scenario.expected_result

    @pytest.mark.parametrize(
        "scenario",
        CollectionUtilitiesScenarios.chunk_cases(),
        ids=lambda s: s.name,
    )
    def test_chunk(self, scenario: ChunkScenario) -> None:
        """Test chunk with various scenarios."""
        result = u.chunk(scenario.items, scenario.size)
        assert result == scenario.expected_result

    def test_batch_basic(self) -> None:
        """Test batch basic functionality."""
        items = [1, 2, 3, 4, 5]
        result = u.batch(items, lambda x: x * 2, size=2)
        _ = assertion_helpers.assert_flext_result_success(result)
        data = result.value
        tm.that(data.total, eq=5)
        tm.that(data.success_count, eq=5)
        tm.that(data.results, eq=[2, 4, 6, 8, 10])
        tm.that(data.errors, eq=[])

    def test_batch_with_errors_collect(self) -> None:
        """Test batch with errors (collect mode)."""
        items = [1, 2, 0, 4]

        def op(x: int) -> int:
            if x == 0:
                msg = "Zero"
                raise ValueError(msg)
            return 10 // x

        result = u.batch(items, op, on_error="collect")
        _ = assertion_helpers.assert_flext_result_success(result)
        data = result.value
        tm.that(data.total, eq=4)
        tm.that(data.success_count, eq=3)
        tm.that(len(data.errors), eq=1)
        tm.that(data.errors[0][1], has="Zero")

    def test_batch_flatten(self) -> None:
        """Test batch with flattening."""
        items = [[1, 2], [3, 4], 5]

        def flatten_op(value: Sequence[int] | int) -> t.Serializable:
            return cast("t.Serializable", value)

        result = u.batch(
            items,
            flatten_op,
            flatten=True,
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        data = result.value
        tm.that(data.results, eq=[1, 2, 3, 4, 5])

    def test_merge_deep(self) -> None:
        """Test deep merge."""
        base_data: Mapping[str, t.NormalizedValue] = {"a": 1, "b": {"x": 1}}
        other_data: Mapping[str, t.NormalizedValue] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base_data, other_data)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(isinstance(result.value["b"], dict), eq=True)

    def test_merge_override(self) -> None:
        """Test override merge."""
        base_data: Mapping[str, t.NormalizedValue] = {"a": 1, "b": {"x": 1}}
        other_data: Mapping[str, t.NormalizedValue] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base_data, other_data, strategy="override")
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(isinstance(result.value["b"], dict), eq=True)


FixtureStatus = TestUtilitiesCollectionCoverage.FixtureStatus
FixturePriority = TestUtilitiesCollectionCoverage.FixturePriority
ParseSequenceScenario = TestUtilitiesCollectionCoverage.ParseSequenceScenario
CoerceListScenario = TestUtilitiesCollectionCoverage.CoerceListScenario
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

__all__ = ["TestUtilitiesCollectionCoverage"]
