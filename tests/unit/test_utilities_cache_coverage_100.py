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
from collections import UserDict
from collections.abc import Sequence
from typing import Annotated, ClassVar, cast, override

import pytest
from flext_tests import tm
from pydantic import BaseModel, ConfigDict, Field

from tests import t, u

from ..test_utils import assertion_helpers
from ._models import TestUnitModels


class TestUtilitiesCacheCoverage100:
    class NormalizeComponentScenario(BaseModel):
        """Normalize component test scenario."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

        name: Annotated[str, Field(description="Normalize scenario name")]
        component: Annotated[
            t.Tests.object | set[t.Primitives | None] | None,
            Field(default=None, description="Input component to normalize"),
        ] = None
        expected_type: Annotated[
            type, Field(description="Expected normalized value type")
        ]
        expected_value: Annotated[
            t.Tests.object | None,
            Field(default=None, description="Optional expected normalized value"),
        ] = None

    class SortKeyScenario(BaseModel):
        """Sort key test scenario."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Sort key scenario name")]
        key: Annotated[
            t.SortableObjectType, Field(description="Input key for sort normalization")
        ]
        expected_tuple: Annotated[
            tuple[int, str], Field(description="Expected sort tuple")
        ]

    class ClearCacheScenario(BaseModel):
        """Clear cache test scenario."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Cache clear scenario name")]
        obj: Annotated[
            t.Tests.object, Field(description="Object under cache clear test")
        ]
        has_cache_attr: Annotated[
            bool, Field(description="Whether object exposes cache attribute")
        ]
        expected_success: Annotated[
            bool, Field(description="Expected clear operation success flag")
        ]
        cache_attr_name: Annotated[
            str | None, Field(default=None, description="Optional cache attribute name")
        ] = None

    class CacheScenarios:
        """Centralized cache test scenarios."""

        NORMALIZE_COMPONENT: ClassVar[list[NormalizeComponentScenario]] = [
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
                component=str(object()),
                expected_type=str,
            ),
        ]
        SORT_KEY: ClassVar[list[SortKeyScenario]] = [
            SortKeyScenario(
                name="string_key", key="hello", expected_tuple=(0, "hello")
            ),
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
                key=str(object()),
                expected_tuple=(0, str(object()).lower()),
            ),
            SortKeyScenario(
                name="list_key",
                key=str([1, 2]),
                expected_tuple=(0, str([1, 2]).lower()),
            ),
            SortKeyScenario(
                name="dict_key",
                key=cast("str | int | float", str({"a": 1})),
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

        @pytest.mark.parametrize(
            "scenario",
            CacheScenarios.NORMALIZE_COMPONENT,
            ids=lambda s: s.name,
        )
        def test_normalize_component(
            self, scenario: NormalizeComponentScenario
        ) -> None:
            """Test normalize_component with various scenarios."""
            result = u.normalize_component(
                cast(
                    "t.NormalizedValue | BaseModel | set[t.NormalizedValue]",
                    scenario.component,
                )
            )
            assert isinstance(result, scenario.expected_type)
            if scenario.expected_value is not None:
                assert result == scenario.expected_value

        def test_normalize_pydantic_model(self) -> None:
            """Test normalize_component with Pydantic model."""
            model = TestUnitModels.CacheTestModel(name="test", value=42)
            result = u.normalize_component(model)
            assert isinstance(result, dict)
            result_dict: dict[str, t.NormalizedValue] = result
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
            result_dict: dict[str, t.NormalizedValue] = result
            assert isinstance(result_dict["inner"], dict)
            inner_dict: dict[str, t.NormalizedValue] = result_dict["inner"]
            assert inner_dict["name"] == "inner"
            assert inner_dict["value"] == 10
            assert result_dict["count"] == 5

        def test_normalize_set_preserves_order(self) -> None:
            """Test normalize_component converts set to tuple."""
            component = {3, 1, 2}
            result = u.normalize_component(
                cast(
                    "t.NormalizedValue | BaseModel | set[t.NormalizedValue]",
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
                    "t.NormalizedValue | BaseModel | set[t.NormalizedValue]",
                    component,
                ),
            )
            tm.that(
                result,
                is_=(tuple, list),
                none=False,
                msg="Result must be tuple or list",
            )
            result_tuple = cast("tuple[t.Tests.object, ...]", result)
            tm.that(len(result_tuple), eq=4, msg="Result tuple must have 4 items")
            result_set = set(result_tuple)
            tm.that(1 in result_set, eq=True, msg="1 must be in result")
            tm.that("test" in result_set, eq=True, msg="'test' must be in result")
            tm.that(math.pi in result_set, eq=True, msg="math.pi must be in result")
            tm.that(None in result_set, eq=True, msg="None must be in result")

        def test_normalize_sequence_with_nested_values(self) -> None:
            """Test normalize_component with Sequence containing nested values."""
            component_raw: list[t.NormalizedValue] = [
                1,
                "test",
                {"nested": "dict"},
                [1, 2, 3],
            ]
            component: Sequence[t.NormalizedValue] = cast(
                "Sequence[t.NormalizedValue]",
                component_raw,
            )
            result = u.normalize_component(cast("t.NormalizedValue", component))
            tm.that(result, is_=list, none=False, msg="Result must be list")
            result_list = cast("list[t.Tests.object]", result)
            tm.that(len(result_list), eq=4, msg="Result list must have 4 items")
            tm.that(result_list[0], eq=1, msg="First item must be 1")
            tm.that(result_list[1], eq="test", msg="Second item must be 'test'")
            tm.that(result_list[2], is_=dict, none=False, msg="Third item must be dict")
            tm.that(
                result_list[3], is_=list, none=False, msg="Fourth item must be list"
            )

        def test_normalize_custom_object_fallback(self) -> None:
            """Test normalize_component fallback to string for unknown types."""

            class CustomObject:
                """Custom object for testing fallback."""

                @override
                def __str__(self) -> str:
                    return "custom_object"

            obj = CustomObject()
            result = u.normalize_component(
                cast("t.NormalizedValue | BaseModel | set[t.NormalizedValue]", obj)
            )
            assert isinstance(result, str)
            assert result == "custom_object"

    class TestuCacheSortKey:
        """Test FlextUtilitiesCache.sort_key."""

        @pytest.mark.parametrize(
            "scenario", CacheScenarios.SORT_KEY, ids=lambda s: s.name
        )
        def test_sort_key(self, scenario: SortKeyScenario) -> None:
            """Test sort_key with various scenarios."""
            result = u.sort_key(scenario.key)
            if scenario.name in {"custom_object", "float_key"}:
                assert result[0] == scenario.expected_tuple[0]
                assert isinstance(result[1], str)
                if scenario.name == "float_key":
                    assert result[1] == "3.14" or result[1].startswith("3.1")
            else:
                assert result == scenario.expected_tuple

        def test_sort_key_string_case_insensitive(self) -> None:
            """Test sort_key handles string case-insensitively."""
            assert u.sort_key("Hello") == (0, "hello")
            assert u.sort_key("HELLO") == (0, "hello")
            assert u.sort_key("hello") == (0, "hello")

        def test_sort_key_number_types(self) -> None:
            """Test sort_key handles different number types."""
            assert u.sort_key(42) == (1, "42")
            pi_result = u.sort_key(math.pi)
            assert pi_result[0] == 1
            assert isinstance(pi_result[1], str)
            assert u.sort_key(-10) == (1, "-10")

    class TestuCacheSortDictKeys:
        """Test FlextUtilitiesCache.sort_dict_keys."""

        def test_sort_dict_keys_simple(self) -> None:
            """Test sort_dict_keys with simple dict."""
            data = {"c": 3, "a": 1, "b": 2}
            result = u.sort_dict_keys(data)
            assert isinstance(result, dict)
            result_dict: dict[str, t.NormalizedValue] = result
            assert list(result_dict.keys()) == ["a", "b", "c"]

        def test_sort_dict_keys_with_none_values(self) -> None:
            """Test sort_dict_keys converts None values to empty dict."""
            data: dict[str, str | int | None] = {
                "key1": "value",
                "key2": None,
                "key3": 42,
            }
            result = u.sort_dict_keys(data)
            assert isinstance(result, dict)
            result_dict: dict[str, t.NormalizedValue] = result
            assert result_dict["key1"] == "value"
            assert result_dict["key2"] == {}
            assert result_dict["key3"] == 42

        def test_sort_dict_keys_nested(self) -> None:
            """Test sort_dict_keys with nested dicts."""
            data = {"z": {"c": 3, "a": 1, "b": 2}, "a": {"x": 10, "y": 20}}
            result = u.sort_dict_keys(data)
            assert isinstance(result, dict)
            result_dict: dict[str, t.NormalizedValue] = result
            assert list(result_dict.keys()) == ["a", "z"]
            nested = result_dict["z"]
            assert isinstance(nested, dict)
            nested_dict: dict[str, t.NormalizedValue] = nested
            assert list(nested_dict.keys()) == ["a", "b", "c"]

        def test_sort_dict_keys_non_dict(self) -> None:
            """Test sort_dict_keys returns non-dict unchanged."""
            assert u.sort_dict_keys("string") == "string"
            assert u.sort_dict_keys(42) == 42
            assert u.sort_dict_keys([1, 2, 3]) == [1, 2, 3]

    class TestuCacheClearObjectCache:
        """Test FlextUtilitiesCache.clear_object_cache."""

        def test_clear_object_cache_with_dict_cache(self) -> None:
            """Test clear_object_cache with dict-like cache."""

            class TestObject:
                """Test object with dict cache."""

                _cache: dict[str, str] = {}  # Test double; per-test isolation

                def __init__(self) -> None:
                    self._cache = {"key1": "value1", "key2": "value2"}

            obj = TestObject()
            assert len(obj._cache) == 2
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_success(result)
            assert result.value is True
            assert len(obj._cache) == 0

        def test_clear_object_cache_with_simple_cache(self) -> None:
            """Test clear_object_cache with simple cached value."""

            class TestObject:
                """Test object with simple cache."""

                _cached_value: str | None = "cached_value"

            obj = TestObject()
            assert obj._cached_value == "cached_value"
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_success(result)
            assert getattr(obj, "_cached_value", None) is None

        def test_clear_object_cache_with_non_dict_cache(self) -> None:
            """Test clear_object_cache with cache that doesn't have clear() method."""

            class TestObject:
                """Test object with non-dict cache."""

                _cache: str | None = "simple_string_cache"

            obj = TestObject()
            assert obj._cache == "simple_string_cache"
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_success(result)
            assert getattr(obj, "_cache", None) is None

        def test_clear_object_cache_multiple_attributes(self) -> None:
            """Test clear_object_cache clears multiple cache attributes."""

            class TestObject:
                """Test object with multiple cache attributes."""

                _cache: dict[str, int] = {}  # Test double
                _cached_value: str | None = "value"
                _cached_at: dict[str, int] = {}  # Test double

                def __init__(self) -> None:
                    self._cache = {"a": 1}
                    self._cached_value = "value"
                    self._cached_at = {"b": 2}

            obj = TestObject()
            assert len(obj._cache) == 1
            assert obj._cached_value == "value"
            assert len(obj._cached_at) == 1
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_success(result)
            assert len(obj._cache) == 0
            assert getattr(obj, "_cached_value", None) is None
            assert len(obj._cached_at) == 0

        def test_clear_object_cache_no_cache_attributes(self) -> None:
            """Test clear_object_cache with object without cache attributes."""

            class TestObject:
                """Test object without cache attributes."""

                data: str = "value"

            obj = TestObject()
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_success(result)
            assert result.value is True

        def test_clear_object_cache_with_none_cache(self) -> None:
            """Test clear_object_cache with None cache value."""

            class TestObject:
                """Test object with None cache."""

                _cache: dict[str, str] | None = None

            obj = TestObject()
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_success(result)

        def test_clear_object_cache_error_handling(self) -> None:
            """Test clear_object_cache error handling."""
            error_msg = "Cannot access cache"

            class BadObject:
                @property
                def _cache(self) -> dict[str, str]:
                    raise RuntimeError(error_msg)

            obj = BadObject()
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None and "Failed to clear caches" in result.error

        def test_clear_object_cache_type_error(self) -> None:
            """Test clear_object_cache handles TypeError."""
            error_msg = "Cannot set cache to None"

            class BadObject:
                def __init__(self) -> None:
                    self._cache = object()

                @override
                def __setattr__(self, name: str, value: t.Tests.object | None) -> None:
                    if name == "_cache" and value is None:
                        raise TypeError(error_msg)
                    super().__setattr__(name, value)

            obj = BadObject()
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None and "Failed to clear caches" in result.error

        def test_clear_object_cache_value_error(self) -> None:
            """Test clear_object_cache handles ValueError."""
            error_msg = "Invalid cache"

            class BadObject:
                def __init__(self) -> None:
                    self._cache: dict[str, str] = {}

                @override
                def __getattribute__(self, name: str) -> object:
                    if name == "_cache":
                        raise ValueError(error_msg)
                    return super().__getattribute__(name)

            obj = BadObject()
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None and "Failed to clear caches" in result.error

        def test_clear_object_cache_key_error(self) -> None:
            """Test clear_object_cache handles KeyError."""
            error_msg = "Cannot clear"

            class BadCache(UserDict[str, str]):
                @override
                def clear(self) -> None:
                    raise KeyError(error_msg)

            class BadObject:
                def __init__(self) -> None:
                    self._cache = BadCache({"key": "value"})

            obj = BadObject()
            result = u.clear_object_cache(cast("t.NormalizedValue", obj))
            _ = assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None and "Failed to clear caches" in result.error

        def test_clear_object_cache_with_pydantic_model(self) -> None:
            """Test clear_object_cache with Pydantic model."""

            class ModelWithCache:
                model_config: ClassVar[dict[str, str]] = {"extra": "allow"}
                name: str
                _cache: dict[
                    str, str
                ] = {}  # Test double; cleared by clear_object_cache

                def __init__(self, name: str = "") -> None:
                    self.name = name
                    self._cache = {}

            model = ModelWithCache(name="test")
            model._cache = {"computed": "value"}
            cache_before = getattr(model, "_cache", None)
            assert cache_before is not None
            assert len(cache_before) == 1
            result = u.clear_object_cache(cast("t.NormalizedValue", model))
            _ = assertion_helpers.assert_flext_result_success(result)
            cache_after = getattr(model, "_cache", None)
            assert cache_after == {}

    class TestuCacheHasCacheAttributes:
        """Test FlextUtilitiesCache.has_cache_attributes."""

        def test_has_cache_attributes_true(self) -> None:
            """Test has_cache_attributes returns True when cache exists."""

            class TestObject:
                def __init__(self) -> None:
                    self._cache: dict[str, t.Tests.object] = {}  # Test double

            obj = TestObject()
            assert u.has_cache_attributes(cast("t.NormalizedValue", obj)) is True

        def test_has_cache_attributes_false(self) -> None:
            """Test has_cache_attributes returns False when no cache."""

            class TestObject:
                def __init__(self) -> None:
                    self.data = "value"

            obj = TestObject()
            assert u.has_cache_attributes(cast("t.NormalizedValue", obj)) is False

        def test_has_cache_attributes_multiple(self) -> None:
            """Test has_cache_attributes with multiple cache attributes."""

            class TestObject:
                def __init__(self) -> None:
                    self._cache: dict[str, t.Tests.object] = {}
                    self.cache: dict[str, t.Tests.object] = {}

            obj = TestObject()
            assert u.has_cache_attributes(cast("t.NormalizedValue", obj)) is True

    class TestuCacheGenerateCacheKey:
        """Test FlextUtilitiesCache.generate_cache_key."""

        def test_generate_cache_key_args_only(self) -> None:
            """Test generate_cache_key with positional arguments."""
            key1 = u.generate_cache_key(1, 2, 3)
            key2 = u.generate_cache_key(1, 2, 3)
            assert key1 == key2
            assert len(key1) == 64

        def test_generate_cache_key_kwargs_only(self) -> None:
            """Test generate_cache_key with keyword arguments."""
            key1 = u.generate_cache_key(a=1, b=2)
            key2 = u.generate_cache_key(b=2, a=1)
            assert key1 == key2

        def test_generate_cache_key_mixed(self) -> None:
            """Test generate_cache_key with both args and kwargs."""
            key1 = u.generate_cache_key(1, 2, x=3, y=4)
            key2 = u.generate_cache_key(1, 2, y=4, x=3)
            assert key1 == key2

        def test_generate_cache_key_deterministic(self) -> None:
            """Test generate_cache_key produces deterministic keys."""
            key1 = u.generate_cache_key("test", 42, flag=True)
            key2 = u.generate_cache_key("test", 42, flag=True)
            assert key1 == key2

        def test_generate_cache_key_different_inputs(self) -> None:
            """Test generate_cache_key produces different keys for different inputs."""
            key1 = u.generate_cache_key("test1")
            key2 = u.generate_cache_key("test2")
            assert key1 != key2

    __all__ = [
        "CacheScenarios",
        "TestuCacheClearObjectCache",
        "TestuCacheGenerateCacheKey",
        "TestuCacheHasCacheAttributes",
        "TestuCacheLogger",
        "TestuCacheNormalizeComponent",
        "TestuCacheSortDictKeys",
        "TestuCacheSortKey",
    ]



NormalizeComponentScenario = TestUtilitiesCacheCoverage100.NormalizeComponentScenario
SortKeyScenario = TestUtilitiesCacheCoverage100.SortKeyScenario
ClearCacheScenario = TestUtilitiesCacheCoverage100.ClearCacheScenario
CacheScenarios = TestUtilitiesCacheCoverage100.CacheScenarios
TestuCacheLogger = TestUtilitiesCacheCoverage100.TestuCacheLogger
TestuCacheNormalizeComponent = TestUtilitiesCacheCoverage100.TestuCacheNormalizeComponent
TestuCacheSortKey = TestUtilitiesCacheCoverage100.TestuCacheSortKey
TestuCacheSortDictKeys = TestUtilitiesCacheCoverage100.TestuCacheSortDictKeys
TestuCacheClearObjectCache = TestUtilitiesCacheCoverage100.TestuCacheClearObjectCache
TestuCacheHasCacheAttributes = TestUtilitiesCacheCoverage100.TestuCacheHasCacheAttributes
TestuCacheGenerateCacheKey = TestUtilitiesCacheCoverage100.TestuCacheGenerateCacheKey
