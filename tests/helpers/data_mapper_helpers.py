"""DataMapper-specific test helpers for u.DataMapper testing.

Provides reusable classes and methods for testing data mapping utilities,
reducing code duplication and improving maintainability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, cast

from flext_core import FlextResult, u
from flext_core.typings import t

# Test-specific type that allows callables and other test fixtures
# This is more permissive than GeneralValueType for testing purposes
type TestInputData = Mapping[str, Any]


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
    """Test case data structure for data mapper utilities.

    Uses TestInputData (Mapping[str, Any]) instead of ConfigurationMapping
    because test data may include callable functions for transform_func
    and filter_func parameters which aren't part of GeneralValueType.
    """

    test_type: DataMapperTestType
    description: str
    input_data: TestInputData
    expected_result: t.GeneralValueType
    expected_success: bool = True


class DataMapperTestHelpers:
    """Generic helpers for data mapper utility testing."""

    # Dispatch table for test type → handler method
    _HANDLERS: ClassVar[
        dict[
            DataMapperTestType,
            Callable[
                [TestInputData],
                FlextResult[t.GeneralValueType],
            ],
        ]
    ] = {}

    @classmethod
    def _init_handlers(cls) -> None:
        """Initialize handlers if not yet done (lazy initialization)."""
        if cls._HANDLERS:
            return
        cls._HANDLERS = {
            DataMapperTestType.MAP_DICT_KEYS: cls._handle_map_dict_keys,
            DataMapperTestType.BUILD_FLAGS_DICT: cls._handle_build_flags_dict,
            DataMapperTestType.COLLECT_ACTIVE_KEYS: cls._handle_collect_active_keys,
            DataMapperTestType.TRANSFORM_VALUES: cls._handle_transform_values,
            DataMapperTestType.FILTER_DICT: cls._handle_filter_dict,
            DataMapperTestType.INVERT_DICT: cls._handle_invert_dict,
        }

    @staticmethod
    def _handle_map_dict_keys(
        input_data: TestInputData,
    ) -> FlextResult[t.GeneralValueType]:
        """Handle MAP_DICT_KEYS test case."""
        source = cast("dict[str, t.GeneralValueType]", input_data["source"])
        mapping = cast("dict[str, str]", input_data["mapping"])
        keep_unmapped = cast("bool", input_data.get("keep_unmapped", True))
        result = u.DataMapper.map_dict_keys(
            source, mapping, keep_unmapped=keep_unmapped
        )
        return cast("FlextResult[t.GeneralValueType]", result)

    @staticmethod
    def _handle_build_flags_dict(
        input_data: TestInputData,
    ) -> FlextResult[t.GeneralValueType]:
        """Handle BUILD_FLAGS_DICT test case."""
        flags = cast("list[str]", input_data["flags"])
        mapping = cast("dict[str, str]", input_data["mapping"])
        default_value = cast("bool", input_data.get("default_value", False))
        result_flags = u.DataMapper.build_flags_dict(
            flags, mapping, default_value=default_value
        )
        if result_flags.is_failure:
            return FlextResult[t.GeneralValueType].fail(
                result_flags.error or "Unknown error"
            )
        return FlextResult[t.GeneralValueType].ok(result_flags.value)

    @staticmethod
    def _handle_collect_active_keys(
        input_data: TestInputData,
    ) -> FlextResult[t.GeneralValueType]:
        """Handle COLLECT_ACTIVE_KEYS test case."""
        source_bool = cast("dict[str, bool]", input_data["source"])
        mapping = cast("dict[str, str]", input_data["mapping"])
        result_keys = u.DataMapper.collect_active_keys(source_bool, mapping)
        if result_keys.is_failure:
            return FlextResult[t.GeneralValueType].fail(
                result_keys.error or "Unknown error"
            )
        return FlextResult[t.GeneralValueType].ok(result_keys.value)

    @staticmethod
    def _handle_transform_values(
        input_data: TestInputData,
    ) -> FlextResult[t.GeneralValueType]:
        """Handle TRANSFORM_VALUES test case."""
        source = cast("dict[str, t.GeneralValueType]", input_data["source"])
        transform_func = cast(
            "Callable[[t.GeneralValueType], t.GeneralValueType]",
            input_data["transform_func"],
        )
        result_dict = u.DataMapper.transform_values(source, transform_func)
        return FlextResult[t.GeneralValueType].ok(result_dict)

    @staticmethod
    def _handle_filter_dict(
        input_data: TestInputData,
    ) -> FlextResult[t.GeneralValueType]:
        """Handle FILTER_DICT test case."""
        source = cast("dict[str, t.GeneralValueType]", input_data["source"])
        filter_func = cast(
            "Callable[[str, t.GeneralValueType], bool]",
            input_data["filter_func"],
        )
        result_filtered = u.filter(source, filter_func)
        return FlextResult[t.GeneralValueType].ok(result_filtered)

    @staticmethod
    def _handle_invert_dict(
        input_data: TestInputData,
    ) -> FlextResult[t.GeneralValueType]:
        """Handle INVERT_DICT test case."""
        source = cast("dict[str, str]", input_data["source"])
        handle_collisions = cast("str", input_data.get("handle_collisions", "last"))
        result_inverted = u.DataMapper.invert_dict(
            source, handle_collisions=handle_collisions
        )
        return FlextResult[t.GeneralValueType].ok(result_inverted)

    @classmethod
    def execute_data_mapper_test(
        cls,
        test_case: DataMapperTestCase,
    ) -> FlextResult[t.GeneralValueType]:
        """Execute a data mapper test case and return FlextResult."""
        cls._init_handlers()
        handler = cls._HANDLERS.get(test_case.test_type)
        if handler is None:
            msg = f"Unknown test type: {test_case.test_type}"
            raise ValueError(msg)
        return handler(test_case.input_data)


class BadDict:
    """Dict-like object that raises exception in items()."""

    def items(self) -> list[tuple[str, t.GeneralValueType]]:
        msg = "Items failed"
        raise RuntimeError(msg)


class BadList:
    """List-like object that raises exception in __iter__."""

    def __iter__(self) -> t.GeneralValueType:
        """Raise exception during iteration for testing."""
        msg = "Iteration failed"
        raise RuntimeError(msg)


class BadDictGet:
    """Dict-like object that raises exception in get()."""

    def get(self, key: str) -> bool:
        msg = "Get failed"
        raise RuntimeError(msg)
