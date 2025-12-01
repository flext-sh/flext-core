"""DataMapper-specific test helpers for FlextUtilities.DataMapper testing.

Provides reusable classes and methods for testing data mapping utilities,
reducing code duplication and improving maintainability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from flext_core import FlextResult, FlextUtilities
from flext_core.typings import FlextTypes


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
    input_data: FlextTypes.Types.ConfigurationMapping
    expected_result: FlextTypes.GeneralValueType
    expected_success: bool = True


class DataMapperTestHelpers:
    """Generic helpers for data mapper utility testing."""

    @staticmethod
    def execute_data_mapper_test(  # noqa: PLR0914  # Type narrowing requires many local variables
        test_case: DataMapperTestCase,
    ) -> FlextResult[FlextTypes.GeneralValueType]:
        """Execute a data mapper test case and return FlextResult."""
        data_mapper = FlextUtilities.DataMapper

        match test_case.test_type:
            case DataMapperTestType.MAP_DICT_KEYS:
                source_raw = test_case.input_data["source"]
                mapping_raw = test_case.input_data["mapping"]
                keep_unmapped_raw = test_case.input_data.get("keep_unmapped", True)
                # Type narrowing: input_data contains proper types at runtime
                source = cast("dict[str, FlextTypes.GeneralValueType]", source_raw)  # type: ignore[arg-type]
                mapping = cast("dict[str, str]", mapping_raw)  # type: ignore[arg-type]
                keep_unmapped = cast("bool", keep_unmapped_raw)  # type: ignore[arg-type]
                return data_mapper.map_dict_keys(
                    source,
                    mapping,
                    keep_unmapped=keep_unmapped,
                )

            case DataMapperTestType.BUILD_FLAGS_DICT:
                flags_raw = test_case.input_data["flags"]
                mapping_raw = test_case.input_data["mapping"]
                default_value_raw = test_case.input_data.get("default_value", False)
                # Type narrowing: input_data contains proper types at runtime
                flags = cast("list[str]", flags_raw)  # type: ignore[arg-type]
                mapping = cast("dict[str, str]", mapping_raw)  # type: ignore[arg-type]
                default_value = cast("bool", default_value_raw)  # type: ignore[arg-type]
                return data_mapper.build_flags_dict(
                    flags,
                    mapping,
                    default_value=default_value,
                )

            case DataMapperTestType.COLLECT_ACTIVE_KEYS:
                source_raw = test_case.input_data["source"]
                mapping_raw = test_case.input_data["mapping"]
                # Type narrowing: input_data contains proper types at runtime
                source_bool: dict[str, bool] = cast("dict[str, bool]", source_raw)  # type: ignore[arg-type]
                mapping = cast("dict[str, str]", mapping_raw)  # type: ignore[arg-type]
                return data_mapper.collect_active_keys(source_bool, mapping)

            case DataMapperTestType.TRANSFORM_VALUES:
                source_raw = test_case.input_data["source"]
                transform_func_raw = test_case.input_data["transform_func"]
                # Type narrowing: input_data contains proper types at runtime
                source_transform: dict[str, FlextTypes.GeneralValueType] = cast(
                    "dict[str, FlextTypes.GeneralValueType]", source_raw
                )  # type: ignore[arg-type]
                transform_func = cast(
                    "Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]",
                    transform_func_raw,
                )  # type: ignore[arg-type]
                return data_mapper.transform_values(source_transform, transform_func)

            case DataMapperTestType.FILTER_DICT:
                source_raw = test_case.input_data["source"]
                filter_func_raw = test_case.input_data["filter_func"]
                # Type narrowing: input_data contains proper types at runtime
                source_filter: dict[str, FlextTypes.GeneralValueType] = cast(
                    "dict[str, FlextTypes.GeneralValueType]", source_raw
                )  # type: ignore[arg-type]
                filter_func = cast(
                    "Callable[[str, FlextTypes.GeneralValueType], bool]",
                    filter_func_raw,
                )  # type: ignore[arg-type]
                return data_mapper.filter_dict(source_filter, filter_func)

            case DataMapperTestType.INVERT_DICT:
                source_raw = test_case.input_data["source"]
                handle_collisions_raw = test_case.input_data.get(
                    "handle_collisions",
                    "last",
                )
                # Type narrowing: input_data contains proper types at runtime
                source_typed: dict[str, str] = cast("dict[str, str]", source_raw)  # type: ignore[arg-type]
                handle_collisions = cast("str", handle_collisions_raw)  # type: ignore[arg-type]
                return data_mapper.invert_dict(
                    source_typed,
                    handle_collisions=handle_collisions,
                )

        msg = f"Unknown test type: {test_case.test_type}"
        raise ValueError(msg)


class BadDict:
    """Dict-like object that raises exception in items()."""

    def items(self) -> list[tuple[str, FlextTypes.GeneralValueType]]:
        msg = "Items failed"
        raise RuntimeError(msg)


class BadList:
    """List-like object that raises exception in __iter__."""

    def __iter__(self) -> FlextTypes.GeneralValueType:
        """Raise exception during iteration for testing."""
        msg = "Iteration failed"
        raise RuntimeError(msg)


class BadDictGet:
    """Dict-like object that raises exception in get()."""

    def get(self, key: str) -> bool:
        msg = "Get failed"
        raise RuntimeError(msg)
