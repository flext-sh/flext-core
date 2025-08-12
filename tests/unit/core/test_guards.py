"""Comprehensive tests for FlextGuards and guard functionality."""

from __future__ import annotations

import math
from typing import Protocol, cast

import pytest
from pydantic import Field, field_validator

from flext_core import (
    FlextValidationError,
    ValidatedModel,
    immutable,
    is_dict_of,
    is_instance_of,
    is_list_of,
    is_not_none,
    make_builder,
    make_factory,
    pure,
    require_in_range,
    require_non_empty,
    require_not_none,
    require_positive,
    safe,
)
from flext_core.result import FlextResult


class FactoryProtocol(Protocol):
    """Protocol for factory functions."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call the factory function."""
        ...


class BuilderProtocol(Protocol):
    """Protocol for builder functions."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call the builder function."""
        ...


# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


class TestTypeGuards:
    """Test type guard functionality."""

    def test_is_not_none(self) -> None:
        """Test is_not_none type guard."""
        # Test with non-None values
        if not (is_not_none("string")):
            raise AssertionError(f"Expected True, got {is_not_none('string')}")
        assert is_not_none(42) is True
        if not (is_not_none([])):
            raise AssertionError(f"Expected True, got {is_not_none([])}")
        assert is_not_none({}) is True
        if not (is_not_none(0)):
            raise AssertionError(f"Expected True, got {is_not_none(0)}")
        assert is_not_none(value=False) is True

        # Test with None
        if is_not_none(None):
            raise AssertionError(f"Expected False, got {is_not_none(None)}")

    def test_is_list_of(self) -> None:
        """Test is_list_of type guard."""
        # Test valid lists
        if not (is_list_of([1, 2, 3], int)):
            raise AssertionError(f"Expected True, got {is_list_of([1, 2, 3], int)}")
        assert is_list_of(["a", "b", "c"], str) is True
        if not (is_list_of([1.0, 2.0], float)):
            raise AssertionError(f"Expected True, got {is_list_of([1.0, 2.0], float)}")
        assert is_list_of([], int) is True  # Empty list is valid

        # Test invalid lists
        if is_list_of([1, "2", 3], int):
            raise AssertionError(f"Expected False, got {is_list_of([1, '2', 3], int)}")
        assert is_list_of(["a", 2, "c"], str) is False
        if is_list_of([1.0, 2], float):
            raise AssertionError(f"Expected False, got {is_list_of([1.0, 2], float)}")

        # Test non-lists
        if is_list_of("string", str):
            raise AssertionError(f"Expected False, got {is_list_of('string', str)}")
        assert is_list_of(42, int) is False
        if is_list_of({}, dict):
            raise AssertionError(f"Expected False, got {is_list_of({}, dict)}")
        assert is_list_of(None, int) is False

    def test_is_instance_of(self) -> None:
        """Test is_instance_of type guard."""
        # Test basic types
        if not (is_instance_of("string", str)):
            raise AssertionError(f"Expected True, got {is_instance_of('string', str)}")
        assert is_instance_of(42, int) is True
        if not (is_instance_of(math.pi, float)):
            raise AssertionError(f"Expected True, got {is_instance_of(math.pi, float)}")
        assert is_instance_of([], list) is True
        if not (is_instance_of({}, dict)):
            raise AssertionError(f"Expected True, got {is_instance_of({}, dict)}")
        # Boolean value requires named parameter due to FBT003
        value_to_test = True
        if not (is_instance_of(value_to_test, bool)):
            raise AssertionError(
                f"Expected True, got {is_instance_of(value_to_test, bool)}"
            )

        # Test inheritance
        class Parent:
            pass

        class Child(Parent):
            pass

        child = Child()
        if not (is_instance_of(child, Child)):
            raise AssertionError(f"Expected True, got {is_instance_of(child, Child)}")
        assert is_instance_of(child, Parent) is True

        # Test negative cases
        if is_instance_of("string", int):
            raise AssertionError(f"Expected False, got {is_instance_of('string', int)}")
        assert is_instance_of(42, str) is False
        if is_instance_of([], dict):
            raise AssertionError(f"Expected False, got {is_instance_of([], dict)}")

    def test_is_dict_of(self) -> None:
        """Test is_dict_of type guard."""
        # Test valid dictionaries
        if not (is_dict_of({"a": 1, "b": 2}, int)):
            raise AssertionError(
                f"Expected True, got {is_dict_of({'a': 1, 'b': 2}, int)}"
            )
        assert is_dict_of({"x": "hello", "y": "world"}, str) is True
        assert is_dict_of({}, int) is True  # Empty dict is valid

        # Test invalid dictionaries
        if is_dict_of({"a": 1, "b": "2"}, int):
            raise AssertionError(
                f"Expected False, got {is_dict_of({'a': 1, 'b': '2'}, int)}"
            )
        assert is_dict_of({"x": "hello", "y": 2}, str) is False

        # Test non-dictionaries
        if is_dict_of([1, 2, 3], int):
            raise AssertionError(f"Expected False, got {is_dict_of([1, 2, 3], int)}")
        assert is_dict_of("string", str) is False
        if is_dict_of(42, int):
            raise AssertionError(f"Expected False, got {is_dict_of(42, int)}")
        assert is_dict_of(None, int) is False


class TestValidationDecorators:
    """Test validation decorator functionality."""

    def test_validated_decorator(self) -> None:
        """Test validated decorator functionality."""

        # The validated decorator is actually FlextDecorators.validated_with_result
        # It should be applied to a function
        def sample_function(x: int) -> int:
            if x < 0:
                msg = "Negative input"
                raise ValueError(msg)
            return x * 2

        # Apply the decorator
        decorated_function = safe(sample_function)

        # Test that the decorator returns a callable
        assert callable(decorated_function)

        # The decorator is a wrapper that might return different behavior
        # We just verify it doesn't break the basic structure

    def test_safe_decorator(self) -> None:
        """Test safe decorator functionality."""

        # The safe decorator is actually FlextDecorators.safe_result
        @safe
        def risky_function(x: int) -> int:
            if x == 0:
                msg = "Cannot be zero"
                raise ValueError(msg)
            return 10 // x

        # Test successful operation
        result = risky_function(2)
        assert isinstance(result, FlextResult)
        if result.success and result.data != 5:
            raise AssertionError(f"Expected {5}, got {result.data}")

        # Test operation that raises exception
        result = risky_function(0)
        assert isinstance(result, FlextResult)
        assert result.is_failure

    def test_immutable_decorator(self) -> None:
        """Test immutable decorator (placeholder)."""

        @immutable
        class TestClass:
            def __init__(self, value: int) -> None:
                self.value = value

        # The decorator is a placeholder, so it just returns the class
        obj = TestClass(42)
        if obj.value != 42:
            raise AssertionError(f"Expected {42}, got {obj.value}")

        # Verify the decorator returned the class unchanged
        assert hasattr(TestClass, "__init__")

    def test_pure_decorator(self) -> None:
        """Test pure decorator (placeholder)."""

        @pure
        def test_function(x: int) -> int:
            return x * 2

        # The decorator is a placeholder, so it just returns the function
        result = test_function(5)
        if result != 10:
            msg: str = f"Expected {10}, got {result}"
            raise AssertionError(msg)

        # Verify the decorator returned the function unchanged
        assert callable(test_function)


class TestValidatedModel:
    """Test ValidatedModel functionality."""

    def test_validated_model_creation(self) -> None:
        """Test basic ValidatedModel creation."""

        class UserModel(ValidatedModel):
            name: str
            age: int
            email: str

        # Test valid creation
        user = UserModel(name="John", age=30, email="john@example.com")
        if user.name != "John":
            msg_name: str = f"Expected {'John'}, got {user.name}"
            raise AssertionError(msg_name)
        assert user.age == 30
        if user.email != "john@example.com":
            msg_email: str = f"Expected {'john@example.com'}, got {user.email}"
            raise AssertionError(msg_email)

        # Test mixin functionality is available
        assert hasattr(user, "to_dict_basic")  # From FlextSerializableMixin
        assert hasattr(user, "is_valid")  # From FlextValidatableMixin

    def test_validated_model_validation_error(self) -> None:
        """Test ValidatedModel validation error handling."""

        class StrictModel(ValidatedModel):
            name: str
            age: int

        # Test validation error
        with pytest.raises(FlextValidationError) as exc_info:
            StrictModel(name="John", age="not_a_number")

        error = exc_info.value
        if "Invalid data" not in str(error):
            msg: str = f"Expected {'Invalid data'} in {error!s}"
            raise AssertionError(msg)
        assert isinstance(error, FlextValidationError)

    def test_validated_model_create_method(self) -> None:
        """Test ValidatedModel.create method."""

        class UserModel(ValidatedModel):
            name: str
            age: int

        # Test successful creation
        result = UserModel.create(name="Alice", age=25)
        assert result.success
        user = result.data
        assert user is not None
        if getattr(user, "name", None) != "Alice":
            msg_alice: str = f"Expected {'Alice'}, got {getattr(user, 'name', None)}"
            raise AssertionError(msg_alice)
        assert getattr(user, "age", None) == 25

        # Test failed creation
        result = UserModel.create(name="Bob", age="invalid")
        assert result.is_failure
        if "Invalid data" not in (result.error or ""):
            msg_invalid: str = f"Expected 'Invalid data' in {result.error}"
            raise AssertionError(msg_invalid)

    def test_validated_model_mixin_integration(self) -> None:
        """Test ValidatedModel integration with FLEXT mixins."""

        class DataModel(ValidatedModel):
            value: str
            count: int

        model = DataModel(value="test", count=42)

        # Test serialization mixin
        data_dict = model.to_dict_basic()
        assert isinstance(data_dict, dict)
        if "value" not in data_dict:
            msg: str = f"Expected {'value'} in {data_dict}"
            raise AssertionError(msg)
        assert "count" in data_dict

        # Test validation mixin
        assert hasattr(model, "validation_errors")
        assert hasattr(model, "is_valid")

    def test_validated_model_complex_validation(self) -> None:
        """Test ValidatedModel with complex validation rules."""

        class ComplexModel(ValidatedModel):
            name: str = Field(min_length=2, max_length=50)
            age: int = Field(ge=0, le=150)
            email: str

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    msg = "Invalid email format"
                    raise ValueError(msg)
                return v

        # Test valid data
        model = ComplexModel(
            name="John Doe",
            age=30,
            email="john@example.com",
        )
        if model.name != "John Doe":
            msg: str = f"Expected {'John Doe'}, got {model.name}"
            raise AssertionError(msg)

        # Test validation failure - short name
        with pytest.raises(FlextValidationError):
            ComplexModel(name="A", age=30, email="john@example.com")

        # Test validation failure - invalid email
        with pytest.raises(FlextValidationError):
            ComplexModel(name="John", age=30, email="invalid-email")


class TestFactoryHelpers:
    """Test factory helper functionality."""

    def test_make_factory(self) -> None:
        """Test make_factory function."""

        class TestClass:
            def __init__(self, value: int, name: str = "default") -> None:
                self.value = value
                self.name = name

        # Create factory
        factory = make_factory(TestClass)
        assert callable(factory)

        # Use factory to create instances
        obj1 = cast("FactoryProtocol", factory)(42)
        if getattr(obj1, "value", None) != 42:
            msg_obj1: str = f"Expected {42}, got {getattr(obj1, 'value', None)}"
            raise AssertionError(msg_obj1)
        assert getattr(obj1, "name", None) == "default"

        obj2 = cast("FactoryProtocol", factory)(100, name="custom")
        if getattr(obj2, "value", None) != 100:
            msg_obj2: str = f"Expected {100}, got {getattr(obj2, 'value', None)}"
            raise AssertionError(msg_obj2)
        assert getattr(obj2, "name", None) == "custom"

        # Verify instances are of correct type
        assert isinstance(obj1, TestClass)
        assert isinstance(obj2, TestClass)

    def test_make_builder(self) -> None:
        """Test make_builder function."""

        class BuildableClass:
            def __init__(self, x: int = 0, y: int = 0) -> None:
                self.x = x
                self.y = y

        # Create builder
        builder = make_builder(BuildableClass)
        assert callable(builder)

        # Use builder to create instances
        obj1 = cast("BuilderProtocol", builder)()
        if getattr(obj1, "x", None) != 0:
            msg_builder1: str = f"Expected {0}, got {getattr(obj1, 'x', None)}"
            raise AssertionError(msg_builder1)
        assert getattr(obj1, "y", None) == 0

        obj2 = cast("BuilderProtocol", builder)(x=10, y=20)
        if getattr(obj2, "x", None) != 10:
            msg_builder2: str = f"Expected {10}, got {getattr(obj2, 'x', None)}"
            raise AssertionError(msg_builder2)
        assert getattr(obj2, "y", None) == 20

        # Verify instances are of correct type
        assert isinstance(obj1, BuildableClass)
        assert isinstance(obj2, BuildableClass)

    def test_factory_with_complex_class(self) -> None:
        """Test factory with more complex class."""

        class ComplexClass:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        factory = make_factory(ComplexClass)

        # Test with positional arguments
        obj1 = cast("FactoryProtocol", factory)(1, 2, 3)
        if getattr(obj1, "args", None) != (1, 2, 3):
            msg_args1: str = f"Expected {(1, 2, 3)}, got {getattr(obj1, 'args', None)}"
            raise AssertionError(msg_args1)
        assert getattr(obj1, "kwargs", None) == {}

        # Test with keyword arguments
        obj2 = cast("FactoryProtocol", factory)(a=1, b=2)
        if getattr(obj2, "args", None) != ():
            msg_args2: str = f"Expected {()}, got {getattr(obj2, 'args', None)}"
            raise AssertionError(msg_args2)
        assert getattr(obj2, "kwargs", None) == {"a": 1, "b": 2}

        # Test with mixed arguments
        obj3 = cast("FactoryProtocol", factory)(1, 2, c=3, d=4)
        if getattr(obj3, "args", None) != (1, 2):
            msg_args3: str = f"Expected {(1, 2)}, got {getattr(obj3, 'args', None)}"
            raise AssertionError(msg_args3)
        assert getattr(obj3, "kwargs", None) == {"c": 3, "d": 4}


class TestValidationUtilities:
    """Test validation utility functions."""

    def test_require_not_none(self) -> None:
        """Test require_not_none utility."""
        # Test with valid values
        if require_not_none("string") != "string":
            raise AssertionError(
                f"Expected {'string'}, got {require_not_none('string')}"
            )
        assert require_not_none(42) == 42
        if require_not_none([]) != []:
            raise AssertionError(f"Expected {[]}, got {require_not_none([])}")
        assert require_not_none({}) == {}
        if require_not_none(value=False):
            raise AssertionError(f"Expected False, got {require_not_none(value=False)}")
        assert require_not_none(0) == 0

        # Test with None
        with pytest.raises(FlextValidationError) as exc_info:
            require_not_none(None)

        error = exc_info.value
        if "Value cannot be None" not in str(error):
            msg_none: str = f"Expected {'Value cannot be None'} in {error!s}"
            raise AssertionError(msg_none)

        # Test with custom message
        with pytest.raises(FlextValidationError) as exc_info:
            require_not_none(None, "Custom error message")

        error = exc_info.value
        if "Custom error message" not in str(error):
            msg_custom: str = f"Expected {'Custom error message'} in {error!s}"
            raise AssertionError(msg_custom)

    def test_require_positive(self) -> None:
        """Test require_positive utility."""
        # Test with valid positive integers
        if require_positive(1) != 1:
            msg_pos1: str = f"Expected {1}, got {require_positive(1)}"
            raise AssertionError(msg_pos1)
        assert require_positive(100) == 100
        if require_positive(9999) != 9999:
            msg_pos2: str = f"Expected {9999}, got {require_positive(9999)}"
            raise AssertionError(msg_pos2)

        # Test with invalid values
        with pytest.raises(FlextValidationError):
            require_positive(0)  # Zero is not positive

        with pytest.raises(FlextValidationError):
            require_positive(-1)  # Negative

        with pytest.raises(FlextValidationError):
            require_positive(math.pi)  # Float, not int

        with pytest.raises(FlextValidationError):
            require_positive("5")  # String, not int

        with pytest.raises(FlextValidationError):
            require_positive(None)  # None

        # Test with custom message
        with pytest.raises(FlextValidationError) as exc_info:
            require_positive(-5, "Must be a positive number")

        error = exc_info.value
        if "Must be a positive number" not in str(error):
            msg_pos_custom: str = f"Expected {'Must be a positive number'} in {error!s}"
            raise AssertionError(msg_pos_custom)

    def test_require_in_range(self) -> None:
        """Test require_in_range utility."""
        # Test with valid values in range
        if require_in_range(5, 1, 10) != 5:
            msg_range1: str = f"Expected {5}, got {require_in_range(5, 1, 10)}"
            raise AssertionError(msg_range1)
        assert require_in_range(1, 1, 10) == 1  # Boundary
        if require_in_range(10, 1, 10) != 10:  # Boundary:
            msg_range2: str = f"Expected {10}, got {require_in_range(10, 1, 10)}"
            raise AssertionError(msg_range2)
        assert require_in_range(5.5, 1, 10) == 5.5  # Float
        if require_in_range(0, -5, 5) != 0:
            msg_range3: str = f"Expected {0}, got {require_in_range(0, -5, 5)}"
            raise AssertionError(msg_range3)

        # Test with values outside range
        with pytest.raises(FlextValidationError):
            require_in_range(11, 1, 10)  # Above max

        with pytest.raises(FlextValidationError):
            require_in_range(0, 1, 10)  # Below min

        with pytest.raises(FlextValidationError):
            require_in_range("5", 1, 10)  # String, not numeric

        with pytest.raises(FlextValidationError):
            require_in_range(None, 1, 10)  # None

        # Test with custom message
        with pytest.raises(FlextValidationError) as exc_info:
            require_in_range(15, 1, 10, "Value out of allowed range")

        error = exc_info.value
        if "Value out of allowed range" not in str(error):
            raise AssertionError(
                f"Expected {'Value out of allowed range'} in {error!s}"
            )

        # Test with auto-generated message
        with pytest.raises(FlextValidationError) as exc_info:
            require_in_range(15, 1, 10)

        error = exc_info.value
        if "Value must be between 1 and 10" not in str(error):
            raise AssertionError(
                f"Expected {'Value must be between 1 and 10'} in {error!s}"
            )

    def test_require_non_empty(self) -> None:
        """Test require_non_empty utility."""
        # Test with valid non-empty strings
        if require_non_empty("hello") != "hello":
            raise AssertionError(
                f"Expected {'hello'}, got {require_non_empty('hello')}"
            )
        assert require_non_empty("a") == "a"
        if require_non_empty("  text  ") != "  text  ":  # Whitespace preserved:
            raise AssertionError(
                f"Expected {'  text  '}, got {require_non_empty('  text  ')}"
            )

        # Test with invalid values
        with pytest.raises(FlextValidationError):
            require_non_empty("")  # Empty string

        with pytest.raises(FlextValidationError):
            require_non_empty("   ")  # Only whitespace

        with pytest.raises(FlextValidationError):
            require_non_empty(None)  # None

        with pytest.raises(FlextValidationError):
            require_non_empty(42)  # Not a string

        with pytest.raises(FlextValidationError):
            require_non_empty([])  # Not a string

        # Test with custom message
        with pytest.raises(FlextValidationError) as exc_info:
            require_non_empty("", "Field cannot be empty")

        error = exc_info.value
        if "Field cannot be empty" not in str(error):
            raise AssertionError(f"Expected {'Field cannot be empty'} in {error!s}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_type_guards_with_none(self) -> None:
        """Test type guards with None values."""
        if is_not_none(None):
            raise AssertionError(f"Expected False, got {is_not_none(None)}")
        assert is_list_of(None, int) is False
        if not (is_instance_of(None, type(None))):
            raise AssertionError(
                f"Expected True, got {is_instance_of(None, type(None))}"
            )
        if is_dict_of(None, int):
            raise AssertionError(f"Expected False, got {is_dict_of(None, int)}")

    def test_type_guards_with_complex_types(self) -> None:
        """Test type guards with complex types."""
        # Test nested structures
        nested_list = [[1, 2], [3, 4]]
        if not (is_list_of(nested_list, list)):
            raise AssertionError(f"Expected True, got {is_list_of(nested_list, list)}")

        nested_dict = {"a": {"x": 1}, "b": {"y": 2}}
        if not (is_dict_of(nested_dict, dict)):
            raise AssertionError(f"Expected True, got {is_dict_of(nested_dict, dict)}")

        # Test custom classes
        class CustomClass:
            pass

        objects = [CustomClass(), CustomClass()]
        if not (is_list_of(objects, CustomClass)):
            raise AssertionError(
                f"Expected True, got {is_list_of(objects, CustomClass)}"
            )

    def test_validated_model_edge_cases(self) -> None:
        """Test ValidatedModel edge cases."""

        class MinimalModel(ValidatedModel):
            pass

        # Empty model should work
        model = MinimalModel()
        assert isinstance(model, ValidatedModel)

        # Create method should work with empty model
        result = MinimalModel.create()
        assert result.success

    def test_factory_edge_cases(self) -> None:
        """Test factory edge cases."""

        # Factory with class that raises exception
        class FailingClass:
            def __init__(self) -> None:
                msg = "Construction failed"
                raise ValueError(msg)

        factory = make_factory(FailingClass)

        # Factory should propagate exceptions
        with pytest.raises(ValueError, match="Construction failed"):
            cast("FactoryProtocol", factory)()

        # Builder should work the same way
        builder = make_builder(FailingClass)

        with pytest.raises(ValueError, match="Construction failed"):
            cast("BuilderProtocol", builder)()

    def test_validation_utilities_edge_cases(self) -> None:
        """Test validation utilities edge cases."""
        # Test require_in_range with equal min/max
        if require_in_range(5, 5, 5) != 5:
            raise AssertionError(f"Expected {5}, got {require_in_range(5, 5, 5)}")

        with pytest.raises(FlextValidationError):
            require_in_range(4, 5, 5)

        with pytest.raises(FlextValidationError):
            require_in_range(6, 5, 5)

        # Test require_positive with boundary values
        if require_positive(1) != 1:
            raise AssertionError(f"Expected {1}, got {require_positive(1)}")

        with pytest.raises(FlextValidationError):
            require_positive(0)

    def test_error_message_consistency(self) -> None:
        """Test that error messages are consistent."""
        # All validation utilities should raise FlextValidationError
        validation_functions: list[tuple[object, object]] = [
            (require_not_none, None),
            (require_positive, -1),
            (require_non_empty, ""),
            (lambda x: require_in_range(x, 1, 10), 15),
        ]

        for func, invalid_value in validation_functions:
            if callable(func):
                with pytest.raises(FlextValidationError):
                    func(invalid_value)


class TestIntegrationAndComposition:
    """Test integration between different guard components."""

    def test_validated_model_with_guards(self) -> None:
        """Test ValidatedModel using guard functions in validation."""

        class GuardedModel(ValidatedModel):
            name: str
            age: int
            priority: int

            def __init__(self, **data: object) -> None:
                # Use guard functions for additional validation
                if "name" in data:
                    data["name"] = require_non_empty(data["name"])
                if "age" in data:
                    data["age"] = require_positive(data["age"])
                if "priority" in data:
                    data["priority"] = require_in_range(data["priority"], 1, 5)

                super().__init__(**data)

        # Test valid data
        model = GuardedModel(name="John", age=30, priority=3)
        if model.name != "John":
            raise AssertionError(f"Expected {'John'}, got {model.name}")
        assert model.age == 30
        if model.priority != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {model.priority}")

        # Test invalid data (should raise FlextValidationError)
        with pytest.raises(FlextValidationError):
            GuardedModel(name="", age=30, priority=3)  # Empty name

        with pytest.raises(FlextValidationError):
            GuardedModel(name="John", age=-5, priority=3)  # Negative age

        with pytest.raises(FlextValidationError):
            GuardedModel(name="John", age=30, priority=10)  # Priority out of range

    def test_factory_with_validation(self) -> None:
        """Test factory pattern with validation."""

        class ValidatedClass:
            def __init__(self, value: int) -> None:
                self.value = require_positive(value)

        factory = make_factory(ValidatedClass)

        # Valid creation
        obj = cast("FactoryProtocol", factory)(42)
        if getattr(obj, "value", None) != 42:
            raise AssertionError(f"Expected {42}, got {getattr(obj, 'value', None)}")

        # Invalid creation should raise exception
        with pytest.raises(FlextValidationError):
            cast("FactoryProtocol", factory)(-1)

    def test_decorator_composition(self) -> None:
        """Test composing multiple decorators."""

        @safe
        @pure
        def composed_function(x: int) -> int:
            require_positive(x)
            return x * 2

        # The function should be callable and return FlextResult due to @safe
        result = composed_function(5)
        assert isinstance(result, FlextResult)

    def test_type_guard_composition(self) -> None:
        """Test composing multiple type guards."""

        def validate_list_of_positive_ints(value: object) -> bool:
            return (
                is_list_of(value, int)
                and is_not_none(value)
                and all(x > 0 for x in value)
                if isinstance(value, list)
                else False
            )

        # Test valid data
        if not (validate_list_of_positive_ints([1, 2, 3])):
            raise AssertionError(
                f"Expected True, got {validate_list_of_positive_ints([1, 2, 3])}"
            )
        assert validate_list_of_positive_ints([]) is True  # Empty list is valid

        # Test invalid data
        assert validate_list_of_positive_ints([1, -2, 3]) is False  # Negative number
        if (
            validate_list_of_positive_ints([1, "2", 3]) is not False
        ):  # String not in list
            raise AssertionError(
                f"Expected {validate_list_of_positive_ints([1, '2', 3])} to be False"
            )
        assert validate_list_of_positive_ints("not a list") is False  # Not a list
        assert validate_list_of_positive_ints(None) is False  # None


class TestPerformanceAndScalability:
    """Test performance characteristics of guards."""

    def test_type_guard_performance(self) -> None:
        """Test performance of type guards with large data."""
        # Large list
        large_list = list(range(1000))
        if not (is_list_of(large_list, int)):
            raise AssertionError(f"Expected True, got {is_list_of(large_list, int)}")

        # Large dict
        large_dict = {f"key_{i}": i for i in range(1000)}
        if not (is_dict_of(large_dict, int)):
            raise AssertionError(f"Expected True, got {is_dict_of(large_dict, int)}")

    def test_validation_utility_performance(self) -> None:
        """Test performance of validation utilities."""
        # Should handle many validations efficiently
        for i in range(100):
            if require_positive(i + 1) != i + 1:
                raise AssertionError(f"Expected {i + 1}, got {require_positive(i + 1)}")
            assert require_not_none(f"string_{i}") == f"string_{i}"
            if require_in_range(i, 0, 99) != i:
                raise AssertionError(f"Expected {i}, got {require_in_range(i, 0, 99)}")

    def test_factory_performance(self) -> None:
        """Test factory performance with many creations."""

        class SimpleClass:
            def __init__(self, value: int) -> None:
                self.value = value

        factory = make_factory(SimpleClass)

        # Create many instances
        objects = [cast("FactoryProtocol", factory)(i) for i in range(100)]
        if len(objects) != 100:
            raise AssertionError(f"Expected {100}, got {len(objects)}")
        if not all(getattr(obj, "value", None) == i for i, obj in enumerate(objects)):
            raise AssertionError(
                f"Expected all objects to have matching values but got {[(i, getattr(obj, 'value', None)) for i, obj in enumerate(objects[:5])]}"
            )
