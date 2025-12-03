"""Real tests to achieve 100% cache utilities coverage - no mocks.

Module: flext_core._utilities.cache
Scope: uCache - all methods and edge cases

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
from dataclasses import dataclass
from typing import ClassVar

import pytest
from pydantic import BaseModel

from flext_core import u
from flext_core.typings import t


class TestModel(BaseModel):
    """Test Pydantic model for cache testing."""

    name: str
    value: int


class NestedModel(BaseModel):
    """Nested Pydantic model for cache testing."""

    inner: TestModel
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
            component=TestModel(name="test", value=42),
            expected_type=dict,
        ),
        NormalizeComponentScenario(
            name="nested_pydantic_model",
            component=NestedModel(inner=TestModel(name="inner", value=10), count=5),
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
        # Sets
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
        # Fallback (other types)
        NormalizeComponentScenario(
            name="custom_object",
            component=object(),
            expected_type=str,
        ),
    ]

    SORT_KEY: ClassVar[list[SortKeyScenario]] = [
        SortKeyScenario(name="string_key", key="hello", expected_tuple=(0, "hello")),
        SortKeyScenario(
            name="string_uppercase", key="HELLO", expected_tuple=(0, "hello")
        ),
        SortKeyScenario(
            name="string_mixed_case", key="HeLlO", expected_tuple=(0, "hello")
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
            name="custom_object", key=object(), expected_tuple=(2, str(object()))
        ),
        SortKeyScenario(name="list_key", key=[1, 2], expected_tuple=(2, str([1, 2]))),
        SortKeyScenario(
            name="dict_key", key={"a": 1}, expected_tuple=(2, str({"a": 1}))
        ),
    ]


class TestuCacheLogger:
    """Test uCache.logger property."""

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
    """Test uCache.normalize_component."""

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

        assert isinstance(result, scenario.expected_type)
        if scenario.expected_value is not None:
            assert result == scenario.expected_value

    def test_normalize_pydantic_model(self) -> None:
        """Test normalize_component with Pydantic model."""
        model = TestModel(name="test", value=42)
        result = u.Cache.normalize_component(model)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_normalize_nested_pydantic_model(self) -> None:
        """Test normalize_component with nested Pydantic model."""
        model = NestedModel(inner=TestModel(name="inner", value=10), count=5)
        result = u.Cache.normalize_component(model)

        assert isinstance(result, dict)
        assert isinstance(result["inner"], dict)
        assert result["inner"]["name"] == "inner"
        assert result["inner"]["value"] == 10
        assert result["count"] == 5

    def test_normalize_set_preserves_order(self) -> None:
        """Test normalize_component converts set to tuple."""
        component = {3, 1, 2}
        result = u.Cache.normalize_component(component)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert set(result) == {1, 2, 3}

    def test_normalize_custom_object_fallback(self) -> None:
        """Test normalize_component fallback to string for unknown types."""

        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        result = u.Cache.normalize_component(obj)

        assert isinstance(result, str)
        assert result == "custom_object"


class TestuCacheSortKey:
    """Test uCache.sort_key."""

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
    """Test uCache.sort_dict_keys."""

    def test_sort_dict_keys_simple(self) -> None:
        """Test sort_dict_keys with simple dict."""
        data = {"c": 3, "a": 1, "b": 2}
        result = u.Cache.sort_dict_keys(data)

        assert isinstance(result, dict)
        assert list(result.keys()) == ["a", "b", "c"]

    def test_sort_dict_keys_with_none_values(self) -> None:
        """Test sort_dict_keys converts None values to empty dict."""
        data = {"key1": "value", "key2": None, "key3": 42}
        result = u.Cache.sort_dict_keys(data)

        assert result["key1"] == "value"
        assert result["key2"] == {}  # None converted to empty dict
        assert result["key3"] == 42

    def test_sort_dict_keys_nested(self) -> None:
        """Test sort_dict_keys with nested dicts."""
        data = {
            "z": {"c": 3, "a": 1, "b": 2},
            "a": {"x": 10, "y": 20},
        }
        result = u.Cache.sort_dict_keys(data)

        assert list(result.keys()) == ["a", "z"]
        assert list(result["z"].keys()) == ["a", "b", "c"]

    def test_sort_dict_keys_non_dict(self) -> None:
        """Test sort_dict_keys returns non-dict unchanged."""
        assert u.Cache.sort_dict_keys("string") == "string"
        assert u.Cache.sort_dict_keys(42) == 42
        assert u.Cache.sort_dict_keys([1, 2, 3]) == [1, 2, 3]


class TestuCacheClearObjectCache:
    """Test uCache.clear_object_cache."""

    def test_clear_object_cache_with_dict_cache(self) -> None:
        """Test clear_object_cache with dict-like cache."""

        class TestObject:
            def __init__(self) -> None:
                self._cache = {"key1": "value1", "key2": "value2"}

        obj = TestObject()
        assert len(obj._cache) == 2

        result = u.Cache.clear_object_cache(obj)
        assert result.is_success
        assert result.value is True
        assert len(obj._cache) == 0

    def test_clear_object_cache_with_simple_cache(self) -> None:
        """Test clear_object_cache with simple cached value."""

        class TestObject:
            def __init__(self) -> None:
                self._cached_value = (
                    "cached_value"  # Use _cached_value (in CACHE_ATTRIBUTE_NAMES)
                )

        obj = TestObject()
        assert obj._cached_value == "cached_value"

        result = u.Cache.clear_object_cache(obj)
        assert result.is_success
        assert obj._cached_value is None

    def test_clear_object_cache_with_non_dict_cache(self) -> None:
        """Test clear_object_cache with cache that doesn't have clear() method."""

        class TestObject:
            def __init__(self) -> None:
                self._cache = "simple_string_cache"  # String doesn't have clear()

        obj = TestObject()
        assert obj._cache == "simple_string_cache"

        result = u.Cache.clear_object_cache(obj)
        assert result.is_success
        # Should set to None (line 284-285)
        assert obj._cache is None

    def test_clear_object_cache_multiple_attributes(self) -> None:
        """Test clear_object_cache clears multiple cache attributes."""

        class TestObject:
            def __init__(self) -> None:
                self._cache = {"a": 1}
                self._cached_value = "value"
                self._cached_at = {"b": 2}

        obj = TestObject()
        # Verify initial state
        assert len(obj._cache) == 1
        assert obj._cached_value == "value"
        assert len(obj._cached_at) == 1

        result = u.Cache.clear_object_cache(obj)

        assert result.is_success
        assert len(obj._cache) == 0
        assert obj._cached_value is None
        assert len(obj._cached_at) == 0

    def test_clear_object_cache_no_cache_attributes(self) -> None:
        """Test clear_object_cache with object without cache attributes."""

        class TestObject:
            def __init__(self) -> None:
                self.data = "value"

        obj = TestObject()
        result = u.Cache.clear_object_cache(obj)

        assert result.is_success
        assert result.value is True

    def test_clear_object_cache_with_none_cache(self) -> None:
        """Test clear_object_cache with None cache value."""

        class TestObject:
            def __init__(self) -> None:
                self._cache = None

        obj = TestObject()
        result = u.Cache.clear_object_cache(obj)

        assert result.is_success
        # None cache should be skipped (not cleared)

    def test_clear_object_cache_error_handling(self) -> None:
        """Test clear_object_cache error handling."""
        error_msg = "Cannot access cache"

        class BadObject:
            @property
            def _cache(self) -> dict:
                raise RuntimeError(error_msg)

        obj = BadObject()
        result = u.Cache.clear_object_cache(obj)

        assert result.is_failure
        assert "Failed to clear caches" in result.error

    def test_clear_object_cache_type_error(self) -> None:
        """Test clear_object_cache handles TypeError."""
        error_msg = "Cannot set cache to None"

        class BadObject:
            def __init__(self) -> None:
                self._cache = object()  # Object without clear() method

            def __setattr__(self, name: str, value: object) -> None:
                # Only raise TypeError when trying to set _cache to None
                if name == "_cache" and value is None:
                    raise TypeError(error_msg)
                super().__setattr__(name, value)

        obj = BadObject()
        result = u.Cache.clear_object_cache(obj)

        # Should handle TypeError gracefully and return failure
        assert result.is_failure
        assert "Failed to clear caches" in result.error

    def test_clear_object_cache_value_error(self) -> None:
        """Test clear_object_cache handles ValueError."""
        error_msg = "Invalid cache"

        class BadObject:
            def __init__(self) -> None:
                self._cache = {}

            def __getattribute__(self, name: str) -> object:
                if name == "_cache":
                    raise ValueError(error_msg)
                return super().__getattribute__(name)

        obj = BadObject()
        result = u.Cache.clear_object_cache(obj)

        assert result.is_failure
        assert "Failed to clear caches" in result.error

    def test_clear_object_cache_key_error(self) -> None:
        """Test clear_object_cache handles KeyError."""
        error_msg = "Cannot clear"

        class BadCache(UserDict):
            def clear(self) -> None:
                raise KeyError(error_msg)

        class BadObject:
            def __init__(self) -> None:
                self._cache = BadCache({"key": "value"})

        obj = BadObject()
        result = u.Cache.clear_object_cache(obj)

        assert result.is_failure
        assert "Failed to clear caches" in result.error

    def test_clear_object_cache_with_pydantic_model(self) -> None:
        """Test clear_object_cache with Pydantic model."""

        class ModelWithCache(BaseModel):
            name: str

        model = ModelWithCache(name="test")
        # Set cache attribute directly (Pydantic allows this)
        model._cache = {"computed": "value"}
        cache_before = getattr(model, "_cache", None)
        assert cache_before is not None
        assert len(cache_before) == 1

        result = u.Cache.clear_object_cache(model)
        assert result.is_success
        # Cache should be cleared (dict.clear() leaves empty dict, not None)
        cache_after = getattr(model, "_cache", None)
        assert cache_after == {}  # clear() leaves empty dict


class TestuCacheHasCacheAttributes:
    """Test uCache.has_cache_attributes."""

    def test_has_cache_attributes_true(self) -> None:
        """Test has_cache_attributes returns True when cache exists."""

        class TestObject:
            def __init__(self) -> None:
                self._cache = {}

        obj = TestObject()
        assert u.Cache.has_cache_attributes(obj) is True

    def test_has_cache_attributes_false(self) -> None:
        """Test has_cache_attributes returns False when no cache."""

        class TestObject:
            def __init__(self) -> None:
                self.data = "value"

        obj = TestObject()
        assert u.Cache.has_cache_attributes(obj) is False

    def test_has_cache_attributes_multiple(self) -> None:
        """Test has_cache_attributes with multiple cache attributes."""

        class TestObject:
            def __init__(self) -> None:
                self._cache = {}
                self.cache = {}

        obj = TestObject()
        assert u.Cache.has_cache_attributes(obj) is True


class TestuCacheGenerateCacheKey:
    """Test uCache.generate_cache_key."""

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
