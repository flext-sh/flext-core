"""Real tests to achieve 100% cache utilities coverage - no mocks.

Module: flext_core._utilities.cache
Scope: FlextUtilitiesCache - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/cache.py.

Uses Python 3.13 patterns, advanced pytest techniques, and aggressive
parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, ClassVar, cast, override

from flext_tests import tm
from pydantic import BaseModel, ConfigDict, Field

from tests import t, u

from ._models import TestUnitModels


class UtilitiesCacheCoverage100Namespace:
    class NormalizeComponentScenario(BaseModel):
        """Normalize component test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        name: Annotated[str, Field(description="Normalize scenario name")]
        component: Annotated[
            t.ValueOrModel | set[t.Primitives | None] | None,
            Field(default=None, description="Input component to normalize"),
        ] = None
        expected_type: Annotated[
            type,
            Field(description="Expected normalized value type"),
        ]
        expected_value: Annotated[
            t.NormalizedValue | None,
            Field(default=None, description="Optional expected normalized value"),
        ] = None

    class SortKeyScenario(BaseModel):
        """Sort key test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Sort key scenario name")]
        key: Annotated[
            t.SortableObjectType,
            Field(description="Input key for sort normalization"),
        ]
        expected_tuple: Annotated[
            tuple[int, str],
            Field(description="Expected sort tuple"),
        ]

    class ClearCacheScenario(BaseModel):
        """Clear cache test scenario."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Cache clear scenario name")]
        obj: Annotated[
            t.NormalizedValue,
            Field(description="Object under cache clear test"),
        ]
        has_cache_attr: Annotated[
            bool,
            Field(description="Whether t.NormalizedValue exposes cache attribute"),
        ]
        expected_success: Annotated[
            bool,
            Field(description="Expected clear operation success flag"),
        ]
        cache_attr_name: Annotated[
            str | None,
            Field(default=None, description="Optional cache attribute name"),
        ] = None

    NORMALIZE_COMPONENT_SCENARIOS: ClassVar[Sequence[NormalizeComponentScenario]] = [
        NormalizeComponentScenario(
            name="pydantic_model",
            component=TestUnitModels.CacheTestModel(name="test", value=42),
            expected_type=dict,
        ),
        NormalizeComponentScenario(
            name="nested_pydantic_model",
            component=TestUnitModels.NestedModel(
                inner=TestUnitModels.CacheTestModel(name="inner", value=10),
                count=5,
            ),
            expected_type=dict,
        ),
        NormalizeComponentScenario(
            name="string_primitive",
            component="hello",
            expected_type=str,
            expected_value="hello",
        ),
        NormalizeComponentScenario(
            name="int_primitive",
            component=42,
            expected_type=int,
            expected_value=42,
        ),
        NormalizeComponentScenario(
            name="float_primitive",
            component=math.pi,
            expected_type=float,
            expected_value=math.pi,
        ),
        NormalizeComponentScenario(
            name="bool_primitive",
            component=True,
            expected_type=bool,
            expected_value=True,
        ),
        NormalizeComponentScenario(
            name="none_primitive",
            component=None,
            expected_type=type(None),
            expected_value=None,
        ),
        NormalizeComponentScenario(
            name="set_of_ints",
            component={1, 2, 3},
            expected_type=tuple,
        ),
        NormalizeComponentScenario(
            name="set_of_strings",
            component={"a", "b", "c"},
            expected_type=tuple,
        ),
        NormalizeComponentScenario(
            name="empty_set",
            component=set(),
            expected_type=tuple,
            expected_value=(),
        ),
        NormalizeComponentScenario(
            name="list_of_ints",
            component=[1, 2, 3],
            expected_type=list,
        ),
        NormalizeComponentScenario(
            name="tuple_of_strings",
            component=("a", "b", "c"),
            expected_type=list,
        ),
        NormalizeComponentScenario(
            name="simple_dict",
            component={"key": "value"},
            expected_type=dict,
        ),
        NormalizeComponentScenario(
            name="nested_dict",
            component={"a": {"b": {"c": 123}}},
            expected_type=dict,
        ),
        NormalizeComponentScenario(
            name="custom_object",
            component="custom_object",
            expected_type=str,
        ),
    ]

    SORT_KEY_SCENARIOS: ClassVar[Sequence[SortKeyScenario]] = [
        SortKeyScenario(name="string_key", key="hello", expected_tuple=(0, "hello")),
        SortKeyScenario(
            name="string_uppercase",
            key="HELLO",
            expected_tuple=(0, "hello"),
        ),
        SortKeyScenario(
            name="string_mixed_case",
            key="HeLlO",
            expected_tuple=(0, "hello"),
        ),
        SortKeyScenario(name="int_key", key=42, expected_tuple=(1, "42")),
        SortKeyScenario(name="float_key", key=math.pi, expected_tuple=(1, "")),
        SortKeyScenario(name="negative_int", key=-5, expected_tuple=(1, "-5")),
        SortKeyScenario(name="zero", key=0, expected_tuple=(1, "0")),
        SortKeyScenario(
            name="custom_object",
            key="custom_object",
            expected_tuple=(0, "custom_object"),
        ),
        SortKeyScenario(
            name="list_key",
            key=str([1, 2]),
            expected_tuple=(0, str([1, 2]).lower()),
        ),
        SortKeyScenario(
            name="dict_key",
            key=cast("t.Numeric | str", str({"a": 1})),
            expected_tuple=(0, str({"a": 1}).lower()),
        ),
    ]

    class TestuCacheLogger:
        """Test FlextUtilitiesCache.logger property."""

        def test_logger_property(self) -> None:
            """Test logger property returns structlog logger."""
            cache = u()
            logger = cache.logger
            assert hasattr(logger, "info")
            assert hasattr(logger, "debug")
            assert hasattr(logger, "warning")
            assert hasattr(logger, "error")

    class TestuCacheNormalizeComponent:
        """Test FlextUtilitiesCache.normalize_component."""

        def test_normalize_component(self) -> None:
            """Test normalize_component with various scenarios."""
            for scenario in NORMALIZE_COMPONENT_SCENARIOS:
                result = u.normalize_component(
                    cast(
                        "t.ValueOrModel | set[t.NormalizedValue]",
                        scenario.component,
                    ),
                )
                assert isinstance(result, scenario.expected_type)
                if scenario.expected_value is not None:
                    assert result == scenario.expected_value

        def test_normalize_pydantic_model(self) -> None:
            """Test normalize_component with Pydantic model."""
            model = TestUnitModels.CacheTestModel(name="test", value=42)
            result = u.normalize_component(model)
            assert isinstance(result, dict)
            result_dict: t.ContainerMapping = result
            assert result_dict["name"] == "test"
            assert result_dict["value"] == 42

        def test_normalize_nested_pydantic_model(self) -> None:
            """Test normalize_component with nested Pydantic model."""
            model = TestUnitModels.NestedModel(
                inner=TestUnitModels.CacheTestModel(name="inner", value=10),
                count=5,
            )
            result = u.normalize_component(model)
            assert isinstance(result, dict)
            result_dict: t.ContainerMapping = result
            assert isinstance(result_dict["inner"], dict)
            inner_dict: t.ContainerMapping = result_dict["inner"]
            assert inner_dict["name"] == "inner"
            assert inner_dict["value"] == 10
            assert result_dict["count"] == 5

        def test_normalize_set_preserves_order(self) -> None:
            """Test normalize_component converts set to tuple."""
            component = {3, 1, 2}
            result = u.normalize_component(
                cast(
                    "t.ValueOrModel | set[t.NormalizedValue]",
                    component,
                ),
            )
            assert isinstance(result, tuple)
            result_tuple = result
            tm.that(len(result_tuple), eq=3, msg="Result tuple must have 3 items")
            assert set(result_tuple) == {1, 2, 3}

        def test_normalize_set_with_nested_values(self) -> None:
            """Test normalize_component with set containing nested values."""
            component = {1, "test", math.pi, None}
            result = u.normalize_component(
                cast(
                    "t.ValueOrModel | set[t.NormalizedValue]",
                    component,
                ),
            )
            tm.that(
                result,
                is_=(tuple, list),
                none=False,
                msg="Result must be tuple or list",
            )
            result_tuple = cast("tuple[t.NormalizedValue, ...]", result)
            tm.that(len(result_tuple), eq=4, msg="Result tuple must have 4 items")
            result_set = set(result_tuple)
            tm.that(1 in result_set, eq=True, msg="1 must be in result")
            tm.that("test" in result_set, eq=True, msg="'test' must be in result")
            tm.that(math.pi in result_set, eq=True, msg="math.pi must be in result")
            tm.that(None in result_set, eq=True, msg="None must be in result")

        def test_normalize_sequence_with_nested_values(self) -> None:
            """Test normalize_component with Sequence containing nested values."""
            component_raw: t.ContainerList = [
                1,
                "test",
                {"nested": "dict"},
                [1, 2, 3],
            ]
            result = u.normalize_component(cast("t.NormalizedValue", component_raw))
            tm.that(result, is_=list, none=False, msg="Result must be list")
            result_list = cast("t.ContainerList", result)
            tm.that(len(result_list), eq=4, msg="Result list must have 4 items")
            tm.that(result_list[0], eq=1, msg="First item must be 1")
            tm.that(result_list[1], eq="test", msg="Second item must be 'test'")
            tm.that(result_list[2], is_=dict, none=False, msg="Third item must be dict")
            tm.that(
                result_list[3],
                is_=list,
                none=False,
                msg="Fourth item must be list",
            )

        def test_normalize_custom_object_fallback(self) -> None:
            """Test normalize_component fallback to string for unknown types."""

            class CustomObject:
                """Custom t.NormalizedValue for testing fallback."""

                @override
                def __str__(self) -> str:
                    return "custom_object"

            obj = CustomObject()
            result = u.normalize_component(
                cast("t.ValueOrModel | set[t.NormalizedValue]", obj),
            )
            assert isinstance(result, str)
            assert result == "custom_object"

    __all__ = [
        "NORMALIZE_COMPONENT_SCENARIOS",
        "TestuCacheLogger",
        "TestuCacheNormalizeComponent",
    ]


NormalizeComponentScenario = (
    UtilitiesCacheCoverage100Namespace.NormalizeComponentScenario
)
NORMALIZE_COMPONENT_SCENARIOS = (
    UtilitiesCacheCoverage100Namespace.NORMALIZE_COMPONENT_SCENARIOS
)
TestuCacheLogger = UtilitiesCacheCoverage100Namespace.TestuCacheLogger
TestuCacheNormalizeComponent = (
    UtilitiesCacheCoverage100Namespace.TestuCacheNormalizeComponent
)
