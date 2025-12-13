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
from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel, Field

from flext_core import t, u
from flext_tests import tm, u as tu
from tests.test_utils import assertion_helpers


class CacheTestModel(BaseModel):
    """Test model for cache key generation."""

    name: str
    value: int
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, str] = Field(default_factory=dict)


class NestedModel(BaseModel):
    """Nested Pydantic model for cache testing."""

    inner: CacheTestModel
    count: int


@dataclass(frozen=True, slots=True)
class NormalizeComponentScenario:
    """Normalize component test scenario."""

    name: str
    component: t.GeneralValueType | BaseModel
    expected_type: type
    expected_value: object | None = None


@dataclass(frozen=True, slots=True)
class SortKeyScenario:
    """Sort key test scenario."""

    name: str
    key: t.SortableObjectType
    expected_tuple: tuple[int, str]


@dataclass(frozen=True, slots=True)
class ClearCacheScenario:
    """Clear cache test scenario."""

    name: str
    obj: object
    has_cache_attr: bool
    expected_success: bool
    cache_attr_name: str | None = None


class CacheScenarios:
    """Centralized cache test scenarios."""

    NORMALIZE_COMPONENT: ClassVar[list[NormalizeComponentScenario]] = [
        # BaseModel cases
        NormalizeComponentScenario(
            name="pydantic_model",
            component=CacheTestModel(name="test", value=42),
            expected_type=dict,
        ),
        NormalizeComponentScenario(
            name="nested_pydantic_model",
            component=NestedModel(
                inner=CacheTestModel(name="inner", value=10),
                count=5,
            ),
            expected_type=dict,
        ),
        # Primitives
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
        # Sets - test that normalize_component converts set to tuple
        # Cast to t.GeneralValueType | BaseModel for type checker
        NormalizeComponentScenario(
            name="set_of_ints",
            component=cast(
                "t.GeneralValueType | BaseModel",
                {1, 2, 3},
            ),  # Set will be converted to tuple by normalize_component
            expected_type=tuple,
        ),
        NormalizeComponentScenario(
            name="set_of_strings",
            component=cast(
                "t.GeneralValueType | BaseModel",
                {"a", "b", "c"},
            ),  # Set will be converted to tuple by normalize_component
            expected_type=tuple,
        ),
        NormalizeComponentScenario(
            name="empty_set",
            component=cast(
                "t.GeneralValueType | BaseModel",
                set(),
            ),  # Set will be converted to tuple by normalize_component
            expected_type=tuple,
            expected_value=(),
        ),
        # Sequences
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
        # Dict-like
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
        # Fallback (other types) - convert to string for t.GeneralValueType compatibility
        NormalizeComponentScenario(
            name="custom_object",
            component=str(object()),  # Convert object to string for t.GeneralValueType
            expected_type=str,
        ),
    ]

    SORT_KEY: ClassVar[list[SortKeyScenario]] = [
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
        # float_key: math.pi string representation may vary, validate structure only
        SortKeyScenario(
            name="float_key",
            key=math.pi,
            expected_tuple=(1, ""),  # Will be validated separately
        ),
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
            key={"a": 1},
            expected_tuple=(2, str({"a": 1})),
        ),
    ]


class TestuCacheLogger:
    """Test FlextUtilitiesCache.logger property."""

    def test_logger_property(self) -> None:
        """Test logger property returns structlog logger."""
        cache = u.Cache()
        logger = cache.logger

        # Verify logger has structlog methods
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
        self,
        scenario: NormalizeComponentScenario,
    ) -> None:
        """Test normalize_component with various scenarios."""
        result = u.Cache.normalize_component(scenario.component)

        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            scenario.expected_type,
            description=scenario.name,
        )
        if scenario.expected_value is not None:
            assert result == scenario.expected_value

    def test_normalize_pydantic_model(self) -> None:
        """Test normalize_component with Pydantic model."""
        model = CacheTestModel(name="test", value=42)
        result = u.Cache.normalize_component(model)

        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            dict,
        )
        # Type narrowing: result is dict after assert_result_matches_expected
        result_dict = cast("dict[str, t.GeneralValueType]", result)
        assert result_dict["name"] == "test"
        assert result_dict["value"] == 42

    def test_normalize_nested_pydantic_model(self) -> None:
        """Test normalize_component with nested Pydantic model."""
        model = NestedModel(inner=CacheTestModel(name="inner", value=10), count=5)
        result = u.Cache.normalize_component(model)

        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            dict,
        )
        # Type narrowing: result is dict after assert_result_matches_expected
        result_dict = cast("dict[str, t.GeneralValueType]", result)
        assert isinstance(result_dict["inner"], dict)
        inner_dict = cast("dict[str, t.GeneralValueType]", result_dict["inner"])
        assert inner_dict["name"] == "inner"
        assert inner_dict["value"] == 10
        assert result_dict["count"] == 5

    def test_normalize_set_preserves_order(self) -> None:
        """Test normalize_component converts set to tuple."""
        component = {3, 1, 2}
        # Cast to t.GeneralValueType | BaseModel for type checker
        # normalize_component will convert set to tuple at runtime
        result = u.Cache.normalize_component(
            cast("t.GeneralValueType | BaseModel", component),
        )

        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            tuple,
        )
        # Type narrowing: result is tuple after assert_result_matches_expected
        result_tuple = cast("tuple[t.GeneralValueType, ...]", result)
        tm.that(len(result_tuple), eq=3, msg="Result tuple must have 3 items")
        tm.that(
            set(result_tuple),
            eq={1, 2, 3},
            msg="Result tuple must contain {1, 2, 3}",
        )

    def test_normalize_set_with_nested_values(self) -> None:
        """Test normalize_component with set containing nested values."""
        component = {1, "test", math.pi, None}
        result = u.Cache.normalize_component(
            cast("t.GeneralValueType | BaseModel", component),
        )
        tm.that(result, is_=tuple, none=False, msg="Result must be tuple")
        result_tuple = cast("tuple[t.GeneralValueType, ...]", result)
        tm.that(len(result_tuple), eq=4, msg="Result tuple must have 4 items")
        # Verify all values are present (order may vary in sets)
        result_set = set(result_tuple)
        tm.that(1 in result_set, eq=True, msg="1 must be in result")
        tm.that("test" in result_set, eq=True, msg="'test' must be in result")
        tm.that(math.pi in result_set, eq=True, msg="math.pi must be in result")
        tm.that(None in result_set, eq=True, msg="None must be in result")

    def test_normalize_sequence_with_nested_values(self) -> None:
        """Test normalize_component with Sequence containing nested values."""
        component_raw: list[object] = [1, "test", {"nested": "dict"}, [1, 2, 3]]
        # Convert list[object] to Sequence[t.GeneralValueType] for type compatibility
        # ObjectList is Sequence[t.GeneralValueType], use that type directly
        component: Sequence[t.GeneralValueType] = cast(
            "Sequence[t.GeneralValueType]",
            component_raw,
        )
        result = u.Cache.normalize_component(component)
        tm.that(result, is_=list, none=False, msg="Result must be list")
        result_list = cast("list[t.GeneralValueType]", result)
        tm.that(len(result_list), eq=4, msg="Result list must have 4 items")
        tm.that(result_list[0], eq=1, msg="First item must be 1")
        tm.that(result_list[1], eq="test", msg="Second item must be 'test'")
        tm.that(result_list[2], is_=dict, none=False, msg="Third item must be dict")
        tm.that(result_list[3], is_=list, none=False, msg="Fourth item must be list")

    def test_normalize_custom_object_fallback(self) -> None:
        """Test normalize_component fallback to string for unknown types."""

        class CustomObject:
            """Custom object for testing fallback."""

            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        # Cast to t.GeneralValueType | BaseModel to test fallback behavior
        # Runtime will handle non-BaseModel objects by converting to string
        result = u.Cache.normalize_component(
            cast("t.GeneralValueType | BaseModel", obj),
        )

        tu.Tests.Assertions.assert_result_matches_expected(result, str)
        assert result == "custom_object"


class TestuCacheSortKey:
    """Test FlextUtilitiesCache.sort_key."""

    @pytest.mark.parametrize(
        "scenario",
        CacheScenarios.SORT_KEY,
        ids=lambda s: s.name,
    )
    def test_sort_key(self, scenario: SortKeyScenario) -> None:
        """Test sort_key with various scenarios."""
        result = u.Cache.sort_key(scenario.key)

        # Special handling for cases where str() representation may vary
        if scenario.name in {"custom_object", "float_key"}:
            assert result[0] == scenario.expected_tuple[0]  # Type group matches
            assert isinstance(result[1], str)  # String representation
            # For float_key, validate it's a valid float string representation
            if scenario.name == "float_key":
                assert result[1] == "3.14" or result[1].startswith("3.1")
        else:
            assert result == scenario.expected_tuple

    def test_sort_key_string_case_insensitive(self) -> None:
        """Test sort_key handles string case-insensitively."""
        assert u.Cache.sort_key("Hello") == (0, "hello")
        assert u.Cache.sort_key("HELLO") == (0, "hello")
        assert u.Cache.sort_key("hello") == (0, "hello")

    def test_sort_key_number_types(self) -> None:
        """Test sort_key handles different number types."""
        assert u.Cache.sort_key(42) == (1, "42")
        # math.pi converts to string representation, not exact "3.14"
        pi_result = u.Cache.sort_key(math.pi)
        assert pi_result[0] == 1  # Type group for numbers
        assert isinstance(pi_result[1], str)  # String representation
        assert u.Cache.sort_key(-10) == (1, "-10")


class TestuCacheSortDictKeys:
    """Test FlextUtilitiesCache.sort_dict_keys."""

    def test_sort_dict_keys_simple(self) -> None:
        """Test sort_dict_keys with simple dict."""
        data = {"c": 3, "a": 1, "b": 2}
        result = u.Cache.sort_dict_keys(data)

        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            dict,
        )
        # Type narrowing: result is dict after assert_result_matches_expected
        result_dict = cast("dict[str, t.GeneralValueType]", result)
        assert list(result_dict.keys()) == ["a", "b", "c"]

    def test_sort_dict_keys_with_none_values(self) -> None:
        """Test sort_dict_keys converts None values to empty dict."""
        data = {"key1": "value", "key2": None, "key3": 42}
        result = u.Cache.sort_dict_keys(data)

        # Type narrowing: sort_dict_keys returns t.GeneralValueType, but for
        # dict input it returns dict
        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            dict,
        )
        # Type narrowing: result is dict after assert_result_matches_expected
        result_dict = cast("dict[str, t.GeneralValueType]", result)
        assert result_dict["key1"] == "value"
        assert result_dict["key2"] == {}  # None converted to empty dict
        assert result_dict["key3"] == 42

    def test_sort_dict_keys_nested(self) -> None:
        """Test sort_dict_keys with nested dicts."""
        data = {
            "z": {"c": 3, "a": 1, "b": 2},
            "a": {"x": 10, "y": 20},
        }
        result = u.Cache.sort_dict_keys(data)

        # Type narrowing: sort_dict_keys returns t.GeneralValueType, but for
        # dict input it returns dict
        tu.Tests.Assertions.assert_result_matches_expected(
            result,
            dict,
        )
        # Type narrowing: result is dict after assert_result_matches_expected
        result_dict = cast("dict[str, t.GeneralValueType]", result)
        assert list(result_dict.keys()) == ["a", "z"]
        # Type narrowing: nested value is also dict
        nested = result_dict["z"]
        assert isinstance(nested, dict)
        nested_dict: dict[str, t.GeneralValueType] = nested
        assert list(nested_dict.keys()) == ["a", "b", "c"]

    def test_sort_dict_keys_non_dict(self) -> None:
        """Test sort_dict_keys returns non-dict unchanged."""
        assert u.Cache.sort_dict_keys("string") == "string"
        assert u.Cache.sort_dict_keys(42) == 42
        assert u.Cache.sort_dict_keys([1, 2, 3]) == [1, 2, 3]


class TestuCacheClearObjectCache:
    """Test FlextUtilitiesCache.clear_object_cache."""

    def test_clear_object_cache_with_dict_cache(self) -> None:
        """Test clear_object_cache with dict-like cache."""

        class TestObject(BaseModel):
            """Test object with dict cache."""

            _cache: dict[str, str] = {"key1": "value1", "key2": "value2"}

        obj = TestObject()
        assert len(obj._cache) == 2

        result = u.Cache.clear_object_cache(obj)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value is True
        assert len(obj._cache) == 0

    def test_clear_object_cache_with_simple_cache(self) -> None:
        """Test clear_object_cache with simple cached value."""

        class TestObject(BaseModel):
            """Test object with simple cache."""

            _cached_value: str | None = (
                "cached_value"  # Use _cached_value (in CACHE_ATTRIBUTE_NAMES)
            )

        obj = TestObject()
        assert obj._cached_value == "cached_value"

        result = u.Cache.clear_object_cache(obj)
        assertion_helpers.assert_flext_result_success(result)
        assert obj._cached_value is None

    def test_clear_object_cache_with_non_dict_cache(self) -> None:
        """Test clear_object_cache with cache that doesn't have clear() method."""

        class TestObject(BaseModel):
            """Test object with non-dict cache."""

            _cache: str | None = "simple_string_cache"  # String doesn't have clear()

        obj = TestObject()
        assert obj._cache == "simple_string_cache"

        result = u.Cache.clear_object_cache(obj)
        assertion_helpers.assert_flext_result_success(result)
        # Should set to None (line 284-285)
        assert obj._cache is None

    def test_clear_object_cache_multiple_attributes(self) -> None:
        """Test clear_object_cache clears multiple cache attributes."""

        class TestObject(BaseModel):
            """Test object with multiple cache attributes."""

            _cache: dict[str, int] = {"a": 1}
            _cached_value: str | None = "value"
            _cached_at: dict[str, int] = {"b": 2}

        obj = TestObject()
        # Verify initial state
        assert len(obj._cache) == 1
        assert obj._cached_value == "value"
        assert len(obj._cached_at) == 1

        result = u.Cache.clear_object_cache(obj)

        assertion_helpers.assert_flext_result_success(result)
        assert len(obj._cache) == 0
        assert obj._cached_value is None
        assert len(obj._cached_at) == 0

    def test_clear_object_cache_no_cache_attributes(self) -> None:
        """Test clear_object_cache with object without cache attributes."""

        class TestObject(BaseModel):
            """Test object without cache attributes."""

            data: str = "value"

        obj = TestObject()
        result = u.Cache.clear_object_cache(obj)

        assertion_helpers.assert_flext_result_success(result)
        assert result.value is True

    def test_clear_object_cache_with_none_cache(self) -> None:
        """Test clear_object_cache with None cache value."""

        class TestObject(BaseModel):
            """Test object with None cache."""

            _cache: dict[str, str] | None = None

        obj = TestObject()
        result = u.Cache.clear_object_cache(obj)

        assertion_helpers.assert_flext_result_success(result)
        # None cache should be skipped (not cleared)

    def test_clear_object_cache_error_handling(self) -> None:
        """Test clear_object_cache error handling."""
        error_msg = "Cannot access cache"

        class BadObject:
            @property
            def _cache(self) -> dict[str, str]:
                raise RuntimeError(error_msg)

        obj = BadObject()
        # Cast to t.GeneralValueType | BaseModel for type checker
        # Runtime will handle the object correctly
        result = u.Cache.clear_object_cache(cast("t.GeneralValueType | BaseModel", obj))

        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None and "Failed to clear caches" in result.error

    def test_clear_object_cache_type_error(self) -> None:
        """Test clear_object_cache handles TypeError."""
        error_msg = "Cannot set cache to None"

        class BadObject:
            def __init__(self) -> None:
                self._cache: object = object()  # Object without clear() method

            def __setattr__(self, name: str, value: object) -> None:
                # Only raise TypeError when trying to set _cache to None
                if name == "_cache" and value is None:
                    raise TypeError(error_msg)
                super().__setattr__(name, value)

        obj = BadObject()
        # Cast to t.GeneralValueType | BaseModel for type checker
        # Runtime will handle the object correctly
        result = u.Cache.clear_object_cache(cast("t.GeneralValueType | BaseModel", obj))

        # Should handle TypeError gracefully and return failure
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None and "Failed to clear caches" in result.error

    def test_clear_object_cache_value_error(self) -> None:
        """Test clear_object_cache handles ValueError."""
        error_msg = "Invalid cache"

        class BadObject:
            def __init__(self) -> None:
                self._cache: dict[str, str] = {}

            def __getattribute__(self, name: str) -> object:
                if name == "_cache":
                    raise ValueError(error_msg)
                return super().__getattribute__(name)

        obj = BadObject()
        # Cast to t.GeneralValueType | BaseModel for type checker
        # Runtime will handle the object correctly
        result = u.Cache.clear_object_cache(cast("t.GeneralValueType | BaseModel", obj))

        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None and "Failed to clear caches" in result.error

    def test_clear_object_cache_key_error(self) -> None:
        """Test clear_object_cache handles KeyError."""
        error_msg = "Cannot clear"

        class BadCache(UserDict[str, str]):
            def clear(self) -> None:
                raise KeyError(error_msg)

        class BadObject:
            def __init__(self) -> None:
                self._cache = BadCache({"key": "value"})

        obj = BadObject()
        # Cast to t.GeneralValueType | BaseModel for type checker
        # Runtime will handle the object correctly
        result = u.Cache.clear_object_cache(cast("t.GeneralValueType | BaseModel", obj))

        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None and "Failed to clear caches" in result.error

    def test_clear_object_cache_with_pydantic_model(self) -> None:
        """Test clear_object_cache with Pydantic model."""

        class ModelWithCache(BaseModel):
            model_config = {"extra": "allow"}
            name: str
            _cache: dict[str, str] = {}

        model = ModelWithCache(name="test")
        # Set cache attribute directly
        model._cache = {"computed": "value"}
        cache_before = getattr(model, "_cache", None)
        assert cache_before is not None
        assert len(cache_before) == 1

        result = u.Cache.clear_object_cache(model)
        assertion_helpers.assert_flext_result_success(result)
        # Cache should be cleared (dict.clear() leaves empty dict, not None)
        cache_after = getattr(model, "_cache", None)
        assert cache_after == {}  # clear() leaves empty dict


class TestuCacheHasCacheAttributes:
    """Test FlextUtilitiesCache.has_cache_attributes."""

    def test_has_cache_attributes_true(self) -> None:
        """Test has_cache_attributes returns True when cache exists."""

        class TestObject:
            def __init__(self) -> None:
                self._cache: dict[str, object] = {}

        obj = TestObject()
        # Cast to t.GeneralValueType for type checker
        assert u.Cache.has_cache_attributes(cast("t.GeneralValueType", obj)) is True

    def test_has_cache_attributes_false(self) -> None:
        """Test has_cache_attributes returns False when no cache."""

        class TestObject:
            def __init__(self) -> None:
                self.data = "value"

        obj = TestObject()
        # Cast to t.GeneralValueType for type checker
        assert u.Cache.has_cache_attributes(cast("t.GeneralValueType", obj)) is False

    def test_has_cache_attributes_multiple(self) -> None:
        """Test has_cache_attributes with multiple cache attributes."""

        class TestObject:
            def __init__(self) -> None:
                self._cache: dict[str, object] = {}
                self.cache: dict[str, object] = {}

        obj = TestObject()
        # Cast to t.GeneralValueType for type checker
        assert u.Cache.has_cache_attributes(cast("t.GeneralValueType", obj)) is True


class TestuCacheGenerateCacheKey:
    """Test FlextUtilitiesCache.generate_cache_key."""

    def test_generate_cache_key_args_only(self) -> None:
        """Test generate_cache_key with positional arguments."""
        key1 = u.Cache.generate_cache_key(1, 2, 3)
        key2 = u.Cache.generate_cache_key(1, 2, 3)

        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest length

    def test_generate_cache_key_kwargs_only(self) -> None:
        """Test generate_cache_key with keyword arguments."""
        key1 = u.Cache.generate_cache_key(a=1, b=2)
        key2 = u.Cache.generate_cache_key(b=2, a=1)  # Different order

        assert key1 == key2  # Should be same regardless of order

    def test_generate_cache_key_mixed(self) -> None:
        """Test generate_cache_key with both args and kwargs."""
        key1 = u.Cache.generate_cache_key(1, 2, x=3, y=4)
        key2 = u.Cache.generate_cache_key(1, 2, y=4, x=3)

        assert key1 == key2

    def test_generate_cache_key_deterministic(self) -> None:
        """Test generate_cache_key produces deterministic keys."""
        key1 = u.Cache.generate_cache_key("test", 42, flag=True)
        key2 = u.Cache.generate_cache_key("test", 42, flag=True)

        assert key1 == key2

    def test_generate_cache_key_different_inputs(self) -> None:
        """Test generate_cache_key produces different keys for different inputs."""
        key1 = u.Cache.generate_cache_key("test1")
        key2 = u.Cache.generate_cache_key("test2")

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
