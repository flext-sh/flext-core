"""DataMapper-specific test helpers for FlextUtilities.DataMapper testing.

Provides reusable classes and methods for testing data mapping utilities,
reducing code duplication and improving maintainability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from flext_core import FlextUtilities


class DataMapperTestType(StrEnum):
    """Enumeration of data mapper test operation types."""

    MAP_DICT_KEYS = "map_dict_keys"
    BUILD_FLAGS_DICT = "build_flags_dict"
    COLLECT_ACTIVE_KEYS = "collect_active_keys"
    TRANSFORM_VALUES = "transform_values"
    FILTER_DICT = "filter_dict"
    INVERT_DICT = "invert_dict"


@dataclass(frozen=True, slots=True)
class DataMapperTestCase:
    """Test case data structure for data mapper utilities."""

    test_type: DataMapperTestType
    description: str
    input_data: dict[str, Any]
    expected_result: Any
    expected_success: bool = True


class DataMapperTestHelpers:
    """Generic helpers for data mapper utility testing."""

    @staticmethod
    def execute_data_mapper_test(test_case: DataMapperTestCase) -> object:
        """Execute a data mapper test case and return result."""
        data_mapper = FlextUtilities.DataMapper

        match test_case.test_type:
            case DataMapperTestType.MAP_DICT_KEYS:
                source = test_case.input_data["source"]
                mapping = test_case.input_data["mapping"]
                keep_unmapped = test_case.input_data.get("keep_unmapped", True)
                return data_mapper.map_dict_keys(
                    source, mapping, keep_unmapped=keep_unmapped,
                )

            case DataMapperTestType.BUILD_FLAGS_DICT:
                flags = test_case.input_data["flags"]
                mapping = test_case.input_data["mapping"]
                default_value = test_case.input_data.get("default_value", False)
                return data_mapper.build_flags_dict(
                    flags, mapping, default_value=default_value,
                )

            case DataMapperTestType.COLLECT_ACTIVE_KEYS:
                source = test_case.input_data["source"]
                mapping = test_case.input_data["mapping"]
                return data_mapper.collect_active_keys(source, mapping)

            case DataMapperTestType.TRANSFORM_VALUES:
                source = test_case.input_data["source"]
                transform_func = test_case.input_data["transform_func"]
                return data_mapper.transform_values(source, transform_func)

            case DataMapperTestType.FILTER_DICT:
                source = test_case.input_data["source"]
                filter_func = test_case.input_data["filter_func"]
                return data_mapper.filter_dict(source, filter_func)

            case DataMapperTestType.INVERT_DICT:
                source = test_case.input_data["source"]
                handle_collisions = test_case.input_data.get(
                    "handle_collisions", "last",
                )
                return data_mapper.invert_dict(
                    source, handle_collisions=handle_collisions,
                )

        msg = f"Unknown test type: {test_case.test_type}"
        raise ValueError(msg)


class BadDict:
    """Dict-like object that raises exception in items()."""

    def items(self) -> list[tuple[str, object]]:
        msg = "Items failed"
        raise RuntimeError(msg)


class BadList:
    """List-like object that raises exception in __iter__."""

    def __iter__(self) -> object:
        """Raise exception during iteration for testing."""
        msg = "Iteration failed"
        raise RuntimeError(msg)


class BadDictGet:
    """Dict-like object that raises exception in get()."""

    def get(self, key: str) -> bool:
        msg = "Get failed"
        raise RuntimeError(msg)
