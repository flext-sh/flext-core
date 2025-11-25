"""Comprehensive tests for FlextUtilities.DataMapper - 100% coverage target.

Tested modules: flext_core._utilities.data_mapper
Test scope: Data mapping utilities for dict key mapping, flags building, active keys
collection, value transformation, dict filtering, and inversion with full edge case coverage.

This module provides real tests (no mocks) for all data mapping functions
in FlextUtilities.DataMapper to achieve 100% code coverage using advanced Python
patterns for code reduction and maintainability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import FlextResult
from tests.fixtures.constants import TestConstants
from tests.helpers import (
    BadDict,
    BadDictGet,
    BadList,
    DataMapperTestCase,
    DataMapperTestHelpers,
    DataMapperTestType,
)


class TestFlextUtilitiesDataMapper:
    """Comprehensive tests for FlextUtilities.DataMapper using advanced patterns."""

    # ============================================================================
    # Factory Methods for Test Cases
    # ============================================================================

    @staticmethod
    def create_map_dict_keys_cases() -> list[DataMapperTestCase]:
        """Create test cases for dict key mapping."""
        return [
            DataMapperTestCase(
                test_type=DataMapperTestType.MAP_DICT_KEYS,
                description="basic",
                input_data={
                    "source": {
                        TestConstants.DataMapper.OLD_KEY: TestConstants.DataMapper.VALUE1,
                        TestConstants.DataMapper.FOO: TestConstants.DataMapper.VALUE2,
                    },
                    "mapping": {
                        TestConstants.DataMapper.OLD_KEY: TestConstants.DataMapper.NEW_KEY,
                        TestConstants.DataMapper.FOO: TestConstants.DataMapper.BAR,
                    },
                },
                expected_result={
                    TestConstants.DataMapper.NEW_KEY: TestConstants.DataMapper.VALUE1,
                    TestConstants.DataMapper.BAR: TestConstants.DataMapper.VALUE2,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.MAP_DICT_KEYS,
                description="keep_unmapped",
                input_data={
                    "source": {
                        TestConstants.DataMapper.OLD_KEY: TestConstants.DataMapper.VALUE1,
                        TestConstants.DataMapper.UNMAPPED: TestConstants.DataMapper.VALUE2,
                    },
                    "mapping": {
                        TestConstants.DataMapper.OLD_KEY: TestConstants.DataMapper.NEW_KEY
                    },
                    "keep_unmapped": True,
                },
                expected_result={
                    TestConstants.DataMapper.NEW_KEY: TestConstants.DataMapper.VALUE1,
                    TestConstants.DataMapper.UNMAPPED: TestConstants.DataMapper.VALUE2,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.MAP_DICT_KEYS,
                description="no_keep_unmapped",
                input_data={
                    "source": {
                        TestConstants.DataMapper.OLD_KEY: TestConstants.DataMapper.VALUE1,
                        TestConstants.DataMapper.UNMAPPED: TestConstants.DataMapper.VALUE2,
                    },
                    "mapping": {
                        TestConstants.DataMapper.OLD_KEY: TestConstants.DataMapper.NEW_KEY
                    },
                    "keep_unmapped": False,
                },
                expected_result={
                    TestConstants.DataMapper.NEW_KEY: TestConstants.DataMapper.VALUE1
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.MAP_DICT_KEYS,
                description="exception_handling",
                input_data={
                    "source": cast("dict[str, object]", BadDict()),
                    "mapping": {},
                },
                expected_result=None,
                expected_success=False,
            ),
        ]

    @staticmethod
    def create_build_flags_dict_cases() -> list[DataMapperTestCase]:
        """Create test cases for flags dict building."""
        return [
            DataMapperTestCase(
                test_type=DataMapperTestType.BUILD_FLAGS_DICT,
                description="basic",
                input_data={
                    "flags": [
                        TestConstants.DataMapper.FLAGS_READ,
                        TestConstants.DataMapper.FLAGS_WRITE,
                    ],
                    "mapping": {
                        TestConstants.DataMapper.FLAGS_READ: TestConstants.DataMapper.CAN_READ,
                        TestConstants.DataMapper.FLAGS_WRITE: TestConstants.DataMapper.CAN_WRITE,
                        TestConstants.DataMapper.FLAGS_DELETE: TestConstants.DataMapper.CAN_DELETE,
                    },
                },
                expected_result={
                    TestConstants.DataMapper.CAN_READ: True,
                    TestConstants.DataMapper.CAN_WRITE: True,
                    TestConstants.DataMapper.CAN_DELETE: False,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.BUILD_FLAGS_DICT,
                description="custom_default",
                input_data={
                    "flags": [TestConstants.DataMapper.FLAGS_READ],
                    "mapping": {
                        TestConstants.DataMapper.FLAGS_READ: TestConstants.DataMapper.CAN_READ,
                        TestConstants.DataMapper.FLAGS_WRITE: TestConstants.DataMapper.CAN_WRITE,
                    },
                    "default_value": True,
                },
                expected_result={
                    TestConstants.DataMapper.CAN_READ: True,
                    TestConstants.DataMapper.CAN_WRITE: True,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.BUILD_FLAGS_DICT,
                description="exception_handling",
                input_data={"flags": cast("list[str]", BadList()), "mapping": {}},
                expected_result=None,
                expected_success=False,
            ),
        ]

    @staticmethod
    def create_collect_active_keys_cases() -> list[DataMapperTestCase]:
        """Create test cases for active keys collection."""
        return [
            DataMapperTestCase(
                test_type=DataMapperTestType.COLLECT_ACTIVE_KEYS,
                description="basic",
                input_data={
                    "source": {
                        TestConstants.DataMapper.FLAGS_READ: True,
                        TestConstants.DataMapper.FLAGS_WRITE: True,
                        TestConstants.DataMapper.FLAGS_DELETE: False,
                    },
                    "mapping": {
                        TestConstants.DataMapper.FLAGS_READ: "r",
                        TestConstants.DataMapper.FLAGS_WRITE: "w",
                        TestConstants.DataMapper.FLAGS_DELETE: "d",
                    },
                },
                expected_result=["r", "w"],
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.COLLECT_ACTIVE_KEYS,
                description="none_active",
                input_data={
                    "source": {
                        TestConstants.DataMapper.FLAGS_READ: False,
                        TestConstants.DataMapper.FLAGS_WRITE: False,
                    },
                    "mapping": {
                        TestConstants.DataMapper.FLAGS_READ: "r",
                        TestConstants.DataMapper.FLAGS_WRITE: "w",
                    },
                },
                expected_result=[],
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.COLLECT_ACTIVE_KEYS,
                description="exception_handling",
                input_data={
                    "source": cast("dict[str, bool]", BadDictGet()),
                    "mapping": {"key": "output"},
                },
                expected_result=None,
                expected_success=False,
            ),
        ]

    @staticmethod
    def create_transform_values_cases() -> list[DataMapperTestCase]:
        """Create test cases for value transformation."""
        return [
            DataMapperTestCase(
                test_type=DataMapperTestType.TRANSFORM_VALUES,
                description="basic",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.HELLO,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.WORLD,
                    },
                    "transform_func": lambda v: str(v).upper(),
                },
                expected_result={
                    TestConstants.DataMapper.A: TestConstants.DataMapper.HELLO_UPPER,
                    TestConstants.DataMapper.B: TestConstants.DataMapper.WORLD_UPPER,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.TRANSFORM_VALUES,
                description="numbers",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.NUM_1,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.NUM_2,
                        TestConstants.DataMapper.C: TestConstants.DataMapper.NUM_3,
                    },
                    "transform_func": lambda v: v * 2 if isinstance(v, int) else v,
                },
                expected_result={
                    TestConstants.DataMapper.A: 2,
                    TestConstants.DataMapper.B: 4,
                    TestConstants.DataMapper.C: 6,
                },
            ),
        ]

    @staticmethod
    def create_filter_dict_cases() -> list[DataMapperTestCase]:
        """Create test cases for dict filtering."""
        return [
            DataMapperTestCase(
                test_type=DataMapperTestType.FILTER_DICT,
                description="basic",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.NUM_1,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.NUM_2,
                        TestConstants.DataMapper.C: TestConstants.DataMapper.NUM_3,
                    },
                    "filter_func": lambda k, v: v > 1 if isinstance(v, int) else False,
                },
                expected_result={
                    TestConstants.DataMapper.B: TestConstants.DataMapper.NUM_2,
                    TestConstants.DataMapper.C: TestConstants.DataMapper.NUM_3,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.FILTER_DICT,
                description="by_key",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.NUM_1,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.NUM_2,
                        TestConstants.DataMapper.C: TestConstants.DataMapper.NUM_3,
                    },
                    "filter_func": lambda k, v: k != TestConstants.DataMapper.B,
                },
                expected_result={
                    TestConstants.DataMapper.A: TestConstants.DataMapper.NUM_1,
                    TestConstants.DataMapper.C: TestConstants.DataMapper.NUM_3,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.FILTER_DICT,
                description="all_filtered",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.NUM_1,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.NUM_2,
                    },
                    "filter_func": lambda k, v: False,
                },
                expected_result={},
            ),
        ]

    @staticmethod
    def create_invert_dict_cases() -> list[DataMapperTestCase]:
        """Create test cases for dict inversion."""
        return [
            DataMapperTestCase(
                test_type=DataMapperTestType.INVERT_DICT,
                description="basic",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.X,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.Y,
                    },
                },
                expected_result={
                    TestConstants.DataMapper.X: TestConstants.DataMapper.A,
                    TestConstants.DataMapper.Y: TestConstants.DataMapper.B,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.INVERT_DICT,
                description="collisions_last",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.X,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.Y,
                        TestConstants.DataMapper.C: TestConstants.DataMapper.X,
                    },
                    "handle_collisions": "last",
                },
                expected_result={
                    TestConstants.DataMapper.X: TestConstants.DataMapper.C,
                    TestConstants.DataMapper.Y: TestConstants.DataMapper.B,
                },
            ),
            DataMapperTestCase(
                test_type=DataMapperTestType.INVERT_DICT,
                description="collisions_first",
                input_data={
                    "source": {
                        TestConstants.DataMapper.A: TestConstants.DataMapper.X,
                        TestConstants.DataMapper.B: TestConstants.DataMapper.Y,
                        TestConstants.DataMapper.C: TestConstants.DataMapper.X,
                    },
                    "handle_collisions": "first",
                },
                expected_result={
                    TestConstants.DataMapper.X: TestConstants.DataMapper.A,
                    TestConstants.DataMapper.Y: TestConstants.DataMapper.B,
                },
            ),
        ]

    # ============================================================================
    # Parametrized Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "test_case",
        create_map_dict_keys_cases.__func__(),
        ids=lambda case: f"map_dict_keys_{case.description}",
    )
    def test_map_dict_keys(self, test_case: DataMapperTestCase) -> None:
        """Test map_dict_keys with various scenarios."""
        result = cast(
            "FlextResult[object]",
            DataMapperTestHelpers.execute_data_mapper_test(test_case),
        )
        if test_case.expected_success:
            assert result.is_success
            assert result.unwrap() == test_case.expected_result
        else:
            assert result.is_failure
            assert result.error is not None
            assert "Failed to map" in result.error

    @pytest.mark.parametrize(
        "test_case",
        create_build_flags_dict_cases.__func__(),
        ids=lambda case: f"build_flags_dict_{case.description}",
    )
    def test_build_flags_dict(self, test_case: DataMapperTestCase) -> None:
        """Test build_flags_dict with various scenarios."""
        result = cast(
            "FlextResult[object]",
            DataMapperTestHelpers.execute_data_mapper_test(test_case),
        )
        if test_case.expected_success:
            assert result.is_success
            assert result.unwrap() == test_case.expected_result
        else:
            assert result.is_failure
            assert result.error is not None
            assert "Failed to build" in result.error

    @pytest.mark.parametrize(
        "test_case",
        create_collect_active_keys_cases.__func__(),
        ids=lambda case: f"collect_active_keys_{case.description}",
    )
    def test_collect_active_keys(self, test_case: DataMapperTestCase) -> None:
        """Test collect_active_keys with various scenarios."""
        result = cast(
            "FlextResult[object]",
            DataMapperTestHelpers.execute_data_mapper_test(test_case),
        )
        if test_case.expected_success:
            assert result.is_success
            assert result.unwrap() == test_case.expected_result
        else:
            assert result.is_failure
            assert result.error is not None
            assert "Failed to collect" in result.error

    @pytest.mark.parametrize(
        "test_case",
        create_transform_values_cases.__func__(),
        ids=lambda case: f"transform_values_{case.description}",
    )
    def test_transform_values(self, test_case: DataMapperTestCase) -> None:
        """Test transform_values with various scenarios."""
        result = DataMapperTestHelpers.execute_data_mapper_test(test_case)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        create_filter_dict_cases.__func__(),
        ids=lambda case: f"filter_dict_{case.description}",
    )
    def test_filter_dict(self, test_case: DataMapperTestCase) -> None:
        """Test filter_dict with various scenarios."""
        result = DataMapperTestHelpers.execute_data_mapper_test(test_case)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        create_invert_dict_cases.__func__(),
        ids=lambda case: f"invert_dict_{case.description}",
    )
    def test_invert_dict(self, test_case: DataMapperTestCase) -> None:
        """Test invert_dict with various scenarios."""
        result = DataMapperTestHelpers.execute_data_mapper_test(test_case)
        assert result == test_case.expected_result
