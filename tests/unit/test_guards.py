"""Tests for FlextGuards module."""

from __future__ import annotations

import pytest

from flext_core import FlextExceptions, FlextGuards, FlextUtilities


class TestFlextGuards:
    """Test FlextGuards functionality."""

    def test_type_guards_basic(self) -> None:
        """Test basic type guard functionality."""
        # Test is_not_none from utilities
        assert FlextUtilities.TypeGuards.is_not_none("test") is True
        assert FlextUtilities.TypeGuards.is_not_none(42) is True
        assert FlextUtilities.TypeGuards.is_not_none([]) is True
        assert FlextUtilities.TypeGuards.is_not_none(None) is False

        # Test other type guards
        assert FlextUtilities.TypeGuards.is_string_non_empty("test") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty("   ") is False

        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False

        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False

        assert FlextUtilities.TypeGuards.has_attribute("test", "upper") is True
        assert FlextUtilities.TypeGuards.has_attribute("test", "nonexistent") is False

    def test_is_list_of(self) -> None:
        """Test is_list_of type guard."""
        # Test with valid lists
        assert FlextGuards.is_list_of([1, 2, 3], int) is True
        assert FlextGuards.is_list_of(["a", "b", "c"], str) is True
        assert FlextGuards.is_list_of([], str) is True  # Empty list is valid

        # Test with invalid lists
        assert FlextGuards.is_list_of([1, "2", 3], int) is False
        assert FlextGuards.is_list_of("not a list", str) is False
        assert FlextGuards.is_list_of(None, str) is False

    def test_is_dict_of(self) -> None:
        """Test is_dict_of type guard."""
        # Test with valid dicts (only value type checking available)
        assert FlextGuards.is_dict_of({"a": 1, "b": 2}, int) is True
        assert FlextGuards.is_dict_of({}, int) is True  # Empty dict is valid
        assert FlextGuards.is_dict_of({"a": "1", "b": "2"}, str) is True

        # Test with invalid dicts
        assert FlextGuards.is_dict_of({"a": 1, "b": "2"}, int) is False
        assert FlextGuards.is_dict_of("not a dict", str) is False

    def test_validation_utils(self) -> None:
        """Test validation utilities."""
        # Test require_not_none - success case
        result = FlextGuards.ValidationUtils.require_not_none(
            "test", "should not be none"
        )
        assert result == "test"

        # Test require_not_none - failure case
        with pytest.raises(FlextExceptions.ValidationError, match="should not be none"):
            FlextGuards.ValidationUtils.require_not_none(None, "should not be none")

        # Test require_non_empty - success case
        result_non_empty = FlextGuards.ValidationUtils.require_non_empty(
            "test", "should not be empty"
        )
        assert result_non_empty == "test"

        # Test require_non_empty - failure case
        with pytest.raises(
            FlextExceptions.ValidationError, match="should not be empty"
        ):
            FlextGuards.ValidationUtils.require_non_empty("", "should not be empty")

    def test_require_in_range(self) -> None:
        """Test require_in_range validation."""
        # Test valid range
        result = FlextGuards.ValidationUtils.require_in_range(5, 1, 10)
        assert result == 5

        # Test invalid range
        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_in_range(15, 1, 10)

        # Test boundary values
        result_min = FlextGuards.ValidationUtils.require_in_range(1, 1, 10)
        assert result_min == 1

        result_max = FlextGuards.ValidationUtils.require_in_range(10, 1, 10)
        assert result_max == 10

    def test_require_positive(self) -> None:
        """Test require_positive validation."""
        # Test positive numbers
        result = FlextGuards.ValidationUtils.require_positive(5, "should be positive")
        assert result == 5

        # Test non-positive numbers raise exceptions
        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_positive(0, "should be positive")

        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_positive(-5, "should be positive")

    def test_pure_function_decorator(self) -> None:
        """Test pure function decorator."""
        call_count = 0

        @FlextGuards.pure
        def expensive_calculation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_calculation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument should use cache
        result2 = expensive_calculation(5)
        assert result2 == 10
        assert call_count == 1  # Should still be 1 due to memoization

        # Different argument should call function
        result3 = expensive_calculation(3)
        assert result3 == 6
        assert call_count == 2

    def test_immutable_decorator(self) -> None:
        """Test immutable decorator."""

        @FlextGuards.immutable
        class ImmutableClass:
            def __init__(self, value: int) -> None:
                self.value = value

        obj = ImmutableClass(42)
        assert obj.value == 42

        # Attempting to modify should raise an error
        with pytest.raises(AttributeError):
            obj.value = 100

    def test_make_builder_and_factory(self) -> None:
        """Test builder and factory creation."""

        # Create a simple class to test with
        class User:
            def __init__(self, name: str, age: int, email: str) -> None:
                self.name = name
                self.age = age
                self.email = email

        # Test make_builder (takes name and fields)
        user_fields = {"name": str, "age": int, "email": str}
        builder_result = FlextGuards.make_builder("User", user_fields)
        assert builder_result.success
        builder = builder_result.unwrap()
        assert builder is not None

        # Test make_factory with name and defaults
        factory_result = FlextGuards.make_factory(
            "User",
            {"name": "default_name", "age": 0, "email": "default@example.com"},
        )
        assert factory_result.success
        factory = factory_result.unwrap()
        assert factory is not None

    def test_integration_with_flext_result(self) -> None:
        """Test integration with exception-based validation pattern."""
        # Test successful validation chain
        validated_value = FlextGuards.ValidationUtils.require_not_none(
            "test", "should not be none"
        )
        final_result = FlextGuards.ValidationUtils.require_non_empty(
            validated_value, "should not be empty"
        )

        assert final_result == "test"

        # Test failure case - expect exception
        with pytest.raises(FlextExceptions.ValidationError, match="should not be none"):
            FlextGuards.ValidationUtils.require_not_none(None, "should not be none")


class TestFlextGuardsAdvanced:
    """Test advanced FlextGuards functionality."""

    def test_guards_system_configuration(self) -> None:
        """Test guards system configuration."""
        config = {
            "validation_level": "strict",
            "cache_enabled": True,
            "max_cache_size": 1000,
        }

        result = FlextGuards.configure_guards_system(config)
        assert result.success is True

        # Get configuration
        current_config = FlextGuards.get_guards_system_config()
        assert current_config.success is True

    def test_guards_performance_optimization(self) -> None:
        """Test guards performance optimization."""
        # Test valid performance levels
        for level in ["low", "balanced", "high", "extreme"]:
            config = {"performance_level": level}
            result = FlextGuards.optimize_guards_performance(config)
            assert result.success is True
            if result.success:
                optimized = result.unwrap()
                assert optimized["performance_level"] == level
                assert "optimization_enabled" in optimized

    def test_complex_type_guards(self) -> None:
        """Test complex type guard scenarios."""
        # Test nested structures
        nested_data = {
            "users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}],
            "metadata": {"version": "1.0", "created": "2024-01-01"},
        }

        # Test that we have a dict with different value types
        assert isinstance(nested_data, dict)
        assert FlextUtilities.TypeGuards.is_dict_non_empty(nested_data) is True
        assert FlextGuards.is_list_of(nested_data["users"], dict) is True

    def test_environment_guards_config(self) -> None:
        """Test environment-specific guards configuration."""
        # Test environment configuration (without unexpected kwargs)
        env_config = FlextGuards.create_environment_guards_config("production")
        assert env_config.success is True
        if env_config.success:
            config = env_config.unwrap()
            assert isinstance(config, dict)
            assert "environment" in config

    def test_performance_validation(self) -> None:
        """Test performance aspects of guards."""
        # Create a large dataset
        large_list = list(range(1000))

        # Type checking should be efficient
        result = FlextGuards.is_list_of(large_list, int)
        assert result is True

        # Test with large dict
        large_dict = {f"key_{i}": i for i in range(1000)}
        dict_result = FlextGuards.is_dict_of(large_dict, int)
        assert dict_result is True

    def test_pure_wrapper_functionality(self) -> None:
        """Test PureWrapper functionality."""
        # Test that PureWrapper exists and is accessible
        assert hasattr(FlextGuards, "PureWrapper")
        assert callable(FlextGuards.PureWrapper)

        # Create a pure wrapper instance
        wrapper = FlextGuards.PureWrapper(lambda x: x * 2)
        assert wrapper is not None

    def test_validation_edge_cases(self) -> None:
        """Test validation edge cases."""
        # Test with edge case values - should raise exceptions
        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_non_empty("")

        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_non_empty("   ")

        # Test boundary values for range validation
        result_boundary = FlextGuards.ValidationUtils.require_in_range(0, 0, 100)
        assert result_boundary == 0

        # Test require_positive edge cases - should raise exceptions
        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_positive(0)

        with pytest.raises(FlextExceptions.ValidationError):
            FlextGuards.ValidationUtils.require_positive(-1)
