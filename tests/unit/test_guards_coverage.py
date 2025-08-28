# ruff: noqa: ARG001, ARG002
"""Tests to increase coverage of guards.py module.

These tests target specific uncovered lines to improve coverage from 65% to higher.
Missing lines: 34-42, 46, 50-60, 73-77, 100-103, 109-110, 114-122, 165-166, 176-180, 227-239
"""

from __future__ import annotations

import contextlib

import pytest

from flext_core.guards import FlextGuards

# Get PureWrapper from FlextGuards if it exists
_PureWrapper = getattr(FlextGuards, "PureWrapper", None) or object

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestPureWrapperCoverage:
    """Tests for _PureWrapper to cover missing lines."""

    def test_pure_wrapper_cache_with_args_and_kwargs(self) -> None:
        """Test _PureWrapper with both args and kwargs (lines 34-42)."""

        def test_function(arg1: str, arg2: str, kwarg1: str = "default") -> str:
            return f"{arg1}-{arg2}-{kwarg1}"

        wrapper = _PureWrapper(test_function)

        # First call should cache the result
        result1 = wrapper("a", "b", kwarg1="c")
        assert result1 == "a-b-c"

        # Second call with same args should return cached result
        result2 = wrapper("a", "b", kwarg1="c")
        assert result2 == "a-b-c"
        assert id(result1) == id(result2)  # Should be exact same object

        # Cache should have one entry
        assert wrapper.__cache_size__() == 1

    def test_pure_wrapper_cache_key_error_fallback(self) -> None:
        """Test _PureWrapper fallback when cache key generation fails (line 41-42)."""

        def function_with_unhashable_args(data: dict[str, str]) -> str:
            return f"processed_{len(data)}"

        wrapper = _PureWrapper(function_with_unhashable_args)

        # This should work despite unhashable dict argument
        unhashable_dict = {"key": "value"}
        result = wrapper(unhashable_dict)
        assert result == "processed_1"

        # Since caching fails due to unhashable args, this should still work
        result2 = wrapper(unhashable_dict)
        assert result2 == "processed_1"

    def test_pure_wrapper_cache_size_method(self) -> None:
        """Test __cache_size__ method (line 46)."""

        def simple_func(x: int) -> int:
            return x * 2

        wrapper = _PureWrapper(simple_func)

        # Initially cache should be empty
        assert wrapper.__cache_size__() == 0

        # After one call, cache should have one entry
        wrapper(5)
        assert wrapper.__cache_size__() == 1

        # After another call with different args, cache should have two entries
        wrapper(10)
        assert wrapper.__cache_size__() == 2

        # Same args should not increase cache size
        wrapper(5)
        assert wrapper.__cache_size__() == 2

    def test_pure_wrapper_descriptor_with_instance(self) -> None:
        """Test __get__ method with instance (lines 50-60)."""

        class TestClass:
            @_PureWrapper
            def method(self, x: int) -> int:
                return x * 2

        instance = TestClass()

        # Get the bound method
        bound_method = instance.method

        # Should be callable and have __pure__ attribute
        assert callable(bound_method)
        assert hasattr(bound_method, "__pure__")
        assert bound_method.__pure__ is True

        # Should work as expected
        result = bound_method(5)
        assert result == 10

    def test_pure_wrapper_descriptor_with_class(self) -> None:
        """Test __get__ method with class (returns self)."""

        def test_func(x: int) -> int:
            return x * 3

        wrapper = _PureWrapper(test_func)

        # When accessed via class, should return the wrapper itself
        result = wrapper.__get__(None, object)
        assert result is wrapper


class TestFlextGuardsCoverage:
    """Tests for FlextGuards to cover missing lines."""

    def test_is_dict_of_not_dict(self) -> None:
        """Test is_dict_of with non-dict input (lines 73-74)."""
        # Test with string
        result = FlextGuards.is_dict_of("not a dict", str)
        assert result is False

        # Test with list
        result = FlextGuards.is_dict_of([1, 2, 3], int)
        assert result is False

        # Test with None
        result = FlextGuards.is_dict_of(None, str)
        assert result is False

    def test_is_dict_of_wrong_value_types(self) -> None:
        """Test is_dict_of with dict but wrong value types (lines 75-77)."""
        # Dict with mixed types, expecting all strings
        mixed_dict = {"a": "string", "b": 123, "c": "another string"}
        result = FlextGuards.is_dict_of(mixed_dict, str)
        assert result is False

        # Dict with all wrong types
        int_dict = {"a": 1, "b": 2, "c": 3}
        result = FlextGuards.is_dict_of(int_dict, str)
        assert result is False

    def test_is_dict_of_correct_types(self) -> None:
        """Test is_dict_of with correct types (should return True)."""
        # Dict with all string values
        str_dict = {"a": "hello", "b": "world"}
        result = FlextGuards.is_dict_of(str_dict, str)
        assert result is True

        # Dict with all int values
        int_dict = {"x": 1, "y": 2, "z": 3}
        result = FlextGuards.is_dict_of(int_dict, int)
        assert result is True

        # Empty dict should return True
        empty_dict = {}
        result = FlextGuards.is_dict_of(empty_dict, str)
        assert result is True

    def test_immutable_decorator_initialization_exception(self) -> None:
        """Test immutable decorator with class that fails initialization (lines 100-103)."""

        class FailingInitClass:
            def __init__(self, value: str) -> None:
                if value == "fail":
                    msg = "Initialization failed"
                    raise ValueError(msg)
                self.value = value

        # Create immutable version
        ImmutableFailingClass = FlextGuards.immutable(FailingInitClass)

        # Test that it falls back to basic initialization when original init fails
        try:
            instance = ImmutableFailingClass("fail")
            # Should not raise error due to fallback
            assert hasattr(instance, "_initialized")
        except ValueError:
            pytest.fail("Should have fallen back to basic initialization")

    def test_immutable_decorator_setattr_after_init(self) -> None:
        """Test immutable decorator prevents attribute modification (lines 109-110)."""

        class SimpleClass:
            def __init__(self, value: str) -> None:
                self.value = value

        ImmutableClass = FlextGuards.immutable(SimpleClass)
        instance = ImmutableClass("test")

        # Should allow access to value
        assert instance.value == "test"

        # Should prevent modification after initialization
        with pytest.raises(AttributeError) as exc_info:
            instance.value = "new value"

        assert "Cannot modify immutable object" in str(exc_info.value)

    def test_immutable_decorator_hash_with_unhashable_attrs(self) -> None:
        """Test immutable decorator hash fallback (lines 114-122)."""

        class UnhashableClass:
            def __init__(self) -> None:
                self.unhashable_attr = {"key": "value"}  # Dict is unhashable
                self.normal_attr = "string"

        ImmutableClass = FlextGuards.immutable(UnhashableClass)
        instance = ImmutableClass()

        # Should fall back to id-based hash when attributes are unhashable
        hash_value = hash(instance)
        assert isinstance(hash_value, int)

        # Hash should be consistent for same object
        hash_value2 = hash(instance)
        assert hash_value == hash_value2

    def test_immutable_decorator_hash_with_hashable_attrs(self) -> None:
        """Test immutable decorator hash with hashable attributes."""

        class HashableClass:
            def __init__(self, value: str, number: int) -> None:
                self.value = value
                self.number = number

        ImmutableClass = FlextGuards.immutable(HashableClass)
        instance1 = ImmutableClass("test", 42)
        instance2 = ImmutableClass("test", 42)

        # Should have same hash for equivalent objects
        hash1 = hash(instance1)
        hash2 = hash(instance2)
        assert hash1 == hash2

    def test_pure_wrapper_metadata_preservation(self) -> None:
        """Test that _PureWrapper preserves function metadata."""

        def documented_function(x: int) -> int:
            """A function has documentation."""
            return x * 2

        documented_function.__name__ = "custom_name"

        wrapper = _PureWrapper(documented_function)

        # Should preserve name and docstring
        assert wrapper.__name__ == "custom_name"
        assert wrapper.__doc__ == "A function has documentation."

    def test_pure_wrapper_without_metadata(self) -> None:
        """Test _PureWrapper with function without metadata."""

        # Create a function without typical metadata
        def func_without_metadata(x: int) -> int:
            return x * 2

        # Remove metadata attributes if they exist
        for attr in ["__name__", "__doc__"]:
            if hasattr(func_without_metadata, attr):
                with contextlib.suppress(AttributeError, TypeError):
                    delattr(func_without_metadata, attr)

        # Should not fail even without metadata
        wrapper = _PureWrapper(func_without_metadata)

        # Should work normally
        result = wrapper(5)
        assert result == 10


class TestEdgeCasesAndBoundaryConditions:
    """Additional edge cases for complete coverage."""

    def test_pure_wrapper_with_no_args_function(self) -> None:
        """Test _PureWrapper with function that takes no arguments."""
        call_count = 0

        def no_args_function() -> str:
            nonlocal call_count
            call_count += 1
            return f"called_{call_count}"

        wrapper = _PureWrapper(no_args_function)

        # First call should execute and cache
        result1 = wrapper()
        assert result1 == "called_1"
        assert call_count == 1

        # Second call should return cached result without executing
        result2 = wrapper()
        assert result2 == "called_1"  # Should be the cached result
        assert call_count == 1  # Should not increment

    def test_immutable_class_without_init(self) -> None:
        """Test immutable decorator with class that has no __init__."""

        class NoInitClass:
            pass

        ImmutableNoInitClass = FlextGuards.immutable(NoInitClass)
        instance = ImmutableNoInitClass()

        # Should be successfully created and marked as initialized
        assert hasattr(instance, "_initialized")

        # Should still prevent attribute setting
        with pytest.raises(AttributeError):
            instance.new_attr = "value"

    def test_complex_cache_key_scenarios(self) -> None:
        """Test complex scenarios for cache key generation."""

        def complex_function(
            a: int, b: str, c: list | None = None, **kwargs: object
        ) -> str:
            c = c or []
            return f"{a}-{b}-{len(c)}-{len(kwargs)}"

        wrapper = _PureWrapper(complex_function)

        # Test with various argument combinations
        result1 = wrapper(1, "test", c=[1, 2], extra="value")
        result2 = wrapper(1, "test", c=[1, 2], extra="value")

        # Should use cached result
        assert result1 == result2

    def test_pure_wrapper_attribute_access_edge_cases(self) -> None:
        """Test edge cases for __pure__ attribute setting."""

        class RestrictedAttributeClass:
            """Class that restricts attribute setting."""

            def __setattr__(self, name: str, value: object) -> None:
                if name == "__pure__":
                    msg = "Cannot set __pure__ attribute"
                    raise AttributeError(msg)
                super().__setattr__(name, value)

        def test_method(self: RestrictedAttributeClass) -> str:
            return "test"

        wrapper = _PureWrapper(test_method)
        instance = RestrictedAttributeClass()

        # Should handle AttributeError when setting __pure__ gracefully
        bound_method = wrapper.__get__(instance, RestrictedAttributeClass)

        # Should still be callable even if __pure__ couldn't be set
        assert callable(bound_method)
