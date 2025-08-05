"""Comprehensive tests for _BaseBuilder and _BaseFluentBuilder.

This test suite provides complete coverage of the builder base functionality,
testing all aspects including property management, validation, fluent interfaces,
and error handling to achieve near 100% coverage.
"""

from __future__ import annotations

import pytest

from flext_core._builder_base import (
    _BaseBuilder,
    _BaseFluentBuilder,
    _create_builder,
    _create_fluent_builder,
)
from flext_core.constants import FlextConstants

pytestmark = [pytest.mark.unit, pytest.mark.core]
# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_TOTAL_PAGES = 8
EXPECTED_DATA_COUNT = 3


@pytest.mark.unit
class TestBaseBuilder:
    """Test _BaseBuilder functionality."""

    def test_builder_initialization_default(self) -> None:
        """Test builder initialization with default name."""
        builder = _BaseBuilder()

        if builder.builder_name != "unnamed":
            name_error_msg_1: str = f"Expected {'unnamed'}, got {builder.builder_name}"
            raise AssertionError(name_error_msg_1)
        assert builder.property_count == 0
        if builder.error_count != 0:
            error_count_msg_2: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_count_msg_2)
        if builder.is_built:
            built_state_msg_3: str = f"Expected False, got {builder.is_built}"
            raise AssertionError(built_state_msg_3)
        assert builder.property_keys == []

    def test_builder_initialization_with_name(self) -> None:
        """Test builder initialization with custom name."""
        builder = _BaseBuilder("test_builder")

        if builder.builder_name != "test_builder":
            name_error_msg: str = (
                f"Expected {'test_builder'}, got {builder.builder_name}"
            )
            raise AssertionError(
                name_error_msg,
            )
        assert builder.property_count == 0
        if builder.error_count != 0:
            error_count_msg_4: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_count_msg_4)
        if builder.is_built:
            built_state_msg_5: str = f"Expected False, got {builder.is_built}"
            raise AssertionError(built_state_msg_5)

    def test_set_property_success(self) -> None:
        """Test successful property setting."""
        builder = _BaseBuilder()

        result = builder._set_property("key1", "value1")

        assert result is builder  # Returns self for chaining
        if builder.property_count != 1:
            property_count_msg_6: str = f"Expected {1}, got {builder.property_count}"
            raise AssertionError(property_count_msg_6)
        assert builder._get_property("key1") == "value1"
        if "key1" not in builder.property_keys:
            property_keys_msg_7: str = f"Expected {'key1'} in {builder.property_keys}"
            raise AssertionError(property_keys_msg_7)

    def test_set_property_multiple(self) -> None:
        """Test setting multiple properties."""
        builder = _BaseBuilder()

        builder._set_property("key1", "value1")
        builder._set_property("key2", 42)
        builder._set_property(
            "key3", "list_test_value"
        )  # Use string for type compatibility

        if builder.property_count != EXPECTED_DATA_COUNT:
            property_count_msg_8: str = f"Expected {3}, got {builder.property_count}"
            raise AssertionError(property_count_msg_8)
        assert builder._get_property("key1") == "value1"
        if builder._get_property("key2") != 42:
            key2_value_msg_9: str = (
                f"Expected {42}, got {builder._get_property('key2')}"
            )
            raise AssertionError(key2_value_msg_9)
        assert builder._get_property("key3") == "list_test_value"
        if set(builder.property_keys) != {"key1", "key2", "key3"}:
            property_keys_set_msg = (
                f"Expected {{'key1', 'key2', 'key3'}}, got {set(builder.property_keys)}"
            )
            raise AssertionError(
                property_keys_set_msg,
            )

    def test_set_property_invalid_key_empty(self) -> None:
        """Test setting property with empty key."""
        builder = _BaseBuilder()

        result = builder._set_property("", "value")

        assert result is builder
        if builder.property_count != 0:
            error_msg_10: str = f"Expected {0}, got {builder.property_count}"
            raise AssertionError(error_msg_10)
        assert builder.error_count == 1
        if "Invalid property key: " not in builder._get_errors()[0]:
            invalid_key_msg: str = (
                f"Expected {'Invalid property key: '} in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                invalid_key_msg,
            )

    def test_set_property_invalid_key_none(self) -> None:
        """Test setting property with None key."""
        builder = _BaseBuilder()

        result = builder._set_property(None, "value")  # type: ignore[arg-type] # Intentional type error for testing

        assert result is builder
        if builder.property_count != 0:
            none_key_property_count_msg: str = (
                f"Expected {0}, got {builder.property_count}"
            )
            raise AssertionError(none_key_property_count_msg)
        assert builder.error_count == 1

    def test_set_property_invalid_key_whitespace(self) -> None:
        """Test setting property with whitespace-only key."""
        builder = _BaseBuilder()

        result = builder._set_property("   ", "value")

        assert result is builder
        if builder.property_count != 0:
            whitespace_key_property_count_msg: str = (
                f"Expected {0}, got {builder.property_count}"
            )
            raise AssertionError(whitespace_key_property_count_msg)
        assert builder.error_count == 1

    def test_set_property_after_built(self) -> None:
        """Test setting property after builder is marked as built."""
        builder = _BaseBuilder()
        builder._mark_built()

        result = builder._set_property("key", "value")

        assert result is builder
        if builder.property_count != 0:
            error_msg_11: str = f"Expected {0}, got {builder.property_count}"
            raise AssertionError(error_msg_11)
        assert builder.error_count == 1
        if "Cannot modify built object" not in builder._get_errors()[0]:
            built_object_msg = (
                f"Expected {'Cannot modify built object'} in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                built_object_msg,
            )

    def test_get_property_existing(self) -> None:
        """Test getting existing property."""
        builder = _BaseBuilder()
        builder._set_property("test_key", "test_value")

        value = builder._get_property("test_key")

        if value != "test_value":
            error_msg_12: str = f"Expected {'test_value'}, got {value}"
            raise AssertionError(error_msg_12)

    def test_get_property_non_existing_no_default(self) -> None:
        """Test getting non-existing property without default."""
        builder = _BaseBuilder()

        value = builder._get_property("non_existing")

        assert value is None

    def test_get_property_non_existing_with_default(self) -> None:
        """Test getting non-existing property with default."""
        builder = _BaseBuilder()

        value = builder._get_property("non_existing", "default_value")

        if value != "default_value":
            error_msg_13: str = f"Expected {'default_value'}, got {value}"
            raise AssertionError(error_msg_13)

    def test_get_property_invalid_key(self) -> None:
        """Test getting property with invalid key."""
        builder = _BaseBuilder()

        value = builder._get_property("", "default")

        if value != "default":
            error_msg_14: str = f"Expected {'default'}, got {value}"
            raise AssertionError(error_msg_14)

    def test_has_property_exists(self) -> None:
        """Test checking if property exists when it does."""
        builder = _BaseBuilder()
        builder._set_property("test_key", "test_value")

        if not (builder._has_property("test_key")):
            error_msg_15: str = (
                f"Expected True, got {builder._has_property('test_key')}"
            )
            raise AssertionError(
                error_msg_15,
            )

    def test_has_property_not_exists(self) -> None:
        """Test checking if property exists when it doesn't."""
        builder = _BaseBuilder()

        if builder._has_property("non_existing"):
            error_msg_16: str = (
                f"Expected False, got {builder._has_property('non_existing')}"
            )
            raise AssertionError(
                error_msg_16,
            )

    def test_has_property_invalid_key(self) -> None:
        """Test checking property existence with invalid key."""
        builder = _BaseBuilder()

        if builder._has_property(""):
            error_msg_17: str = f"Expected False, got {builder._has_property('')}"
            raise AssertionError(error_msg_17)
        assert builder._has_property(None) is False  # type: ignore[arg-type] # Intentional type error for testing

    def test_validate_required_exists(self) -> None:
        """Test validating required property that exists."""
        builder = _BaseBuilder()
        builder._set_property("required_key", "value")

        result = builder._validate_required("required_key")

        if not (result):
            error_msg_18: str = f"Expected True, got {result}"
            raise AssertionError(error_msg_18)
        if builder.error_count != 0:
            error_msg_19: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_19)

    def test_validate_required_missing(self) -> None:
        """Test validating required property that is missing."""
        builder = _BaseBuilder()

        result = builder._validate_required("missing_key")

        if result:
            error_msg_20: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_20)
        assert builder.error_count == 1
        if "Required property 'missing_key' is missing" not in builder._get_errors()[0]:
            error_msg_21: str = f"Expected Required property 'missing_key' is missing in {builder._get_errors()[0]}"
            raise AssertionError(
                error_msg_21,
            )

    def test_validate_required_missing_custom_message(self) -> None:
        """Test validating required property with custom error message."""
        builder = _BaseBuilder()

        result = builder._validate_required("missing_key", "Custom error message")

        if result:
            error_msg_22: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_22)
        assert builder.error_count == 1
        if "Custom error message" not in builder._get_errors()[0]:
            custom_error_msg: str = (
                f"Expected Custom error message in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                custom_error_msg,
            )

    def test_validate_required_none_value(self) -> None:
        """Test validating required property that exists but is None."""
        builder = _BaseBuilder()
        builder._set_property("null_key", None)

        result = builder._validate_required("null_key")

        if result:
            error_msg_23: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_23)
        assert builder.error_count == 1
        if (
            "Required property 'null_key' cannot be None"
            not in builder._get_errors()[0]
        ):
            error_msg_24: str = f"Expected Required property 'null_key' cannot be None in {builder._get_errors()[0]}"
            raise AssertionError(
                error_msg_24,
            )

    def test_validate_type_correct(self) -> None:
        """Test type validation with correct type."""
        builder = _BaseBuilder()
        builder._set_property("string_key", "string_value")

        result = builder._validate_type("string_key", str)

        if not (result):
            error_msg_25: str = f"Expected True, got {result}"
            raise AssertionError(error_msg_25)
        if builder.error_count != 0:
            error_msg_26: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_26)

    def test_validate_type_incorrect(self) -> None:
        """Test type validation with incorrect type."""
        builder = _BaseBuilder()
        builder._set_property("string_key", "string_value")

        result = builder._validate_type("string_key", int)

        if result:
            error_msg_27: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_27)
        assert builder.error_count == 1
        if "Property 'string_key' must be int, got str" not in builder._get_errors()[0]:
            error_msg_28: str = f"Expected Property 'string_key' must be int, got str in {builder._get_errors()[0]}"
            raise AssertionError(
                error_msg_28,
            )

    def test_validate_type_none_value(self) -> None:
        """Test type validation with None value (should pass)."""
        builder = _BaseBuilder()
        builder._set_property("none_key", None)

        result = builder._validate_type("none_key", str)

        assert result is True  # None values pass type validation
        if builder.error_count != 0:
            error_msg_29: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_29)

    def test_validate_type_missing_property(self) -> None:
        """Test type validation on missing property."""
        builder = _BaseBuilder()

        result = builder._validate_type("missing_key", str)

        if result:
            missing_property_result_error_msg_30: str = f"Expected False, got {result}"
            raise AssertionError(missing_property_result_error_msg_30)
        assert builder.error_count == 0  # No error added for missing property

    def test_validate_type_custom_message(self) -> None:
        """Test type validation with custom error message."""
        builder = _BaseBuilder()
        builder._set_property("int_key", "not_an_int")

        result = builder._validate_type("int_key", int, "Custom type error")

        if result:
            error_msg_31: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_31)
        assert builder.error_count == 1
        if "Custom type error" not in builder._get_errors()[0]:
            error_msg_32: str = (
                f"Expected Custom type error in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                error_msg_32,
            )

    def test_validate_string_length_valid(self) -> None:
        """Test string length validation with valid string."""
        builder = _BaseBuilder()
        builder._set_property("string_key", "hello")

        result = builder._validate_string_length(
            "string_key",
            min_length=3,
            max_length=10,
        )

        if not (result):
            error_msg_33: str = f"Expected True, got {result}"
            raise AssertionError(error_msg_33)
        if builder.error_count != 0:
            error_msg_34: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_34)

    def test_validate_string_length_too_short(self) -> None:
        """Test string length validation with too short string."""
        builder = _BaseBuilder()
        builder._set_property("string_key", "hi")

        result = builder._validate_string_length("string_key", min_length=5)

        if result:
            error_msg_35: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_35)
        assert builder.error_count == 1
        if "must be at least 5 characters" not in builder._get_errors()[0]:
            error_msg_36: str = (
                f"Expected must be at least 5 characters in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                error_msg_36,
            )

    def test_validate_string_length_too_long(self) -> None:
        """Test string length validation with too long string."""
        builder = _BaseBuilder()
        builder._set_property("string_key", "very long string")

        result = builder._validate_string_length("string_key", max_length=5)

        if result:
            error_msg_37: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_37)
        assert builder.error_count == 1
        if "must be at most 5 characters" not in builder._get_errors()[0]:
            error_msg_38: str = (
                f"Expected must be at most 5 characters in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                error_msg_38,
            )

    def test_validate_string_length_not_string(self) -> None:
        """Test string length validation on non-string value."""
        builder = _BaseBuilder()
        builder._set_property("int_key", 42)

        result = builder._validate_string_length("int_key", min_length=1)

        if result:
            non_string_result_error_msg_39: str = f"Expected False, got {result}"
            raise AssertionError(non_string_result_error_msg_39)
        assert builder.error_count == 0  # No error added for non-string

    def test_validate_string_length_missing_property(self) -> None:
        """Test string length validation on missing property."""
        builder = _BaseBuilder()

        result = builder._validate_string_length("missing_key", min_length=1)

        if result:
            missing_property_length_result_error_msg_40: str = (
                f"Expected False, got {result}"
            )
            raise AssertionError(missing_property_length_result_error_msg_40)
        assert builder.error_count == 0  # No error added for missing property

    def test_validate_string_length_custom_message(self) -> None:
        """Test string length validation with custom error message."""
        builder = _BaseBuilder()
        builder._set_property("string_key", "short")

        result = builder._validate_string_length(
            "string_key",
            min_length=10,
            error_message="Custom length error",
        )

        if result:
            error_msg_41: str = f"Expected False, got {result}"
            raise AssertionError(error_msg_41)
        assert builder.error_count == 1
        if "Custom length error" not in builder._get_errors()[0]:
            error_msg_42: str = (
                f"Expected Custom length error in {builder._get_errors()[0]}"
            )
            raise AssertionError(
                error_msg_42,
            )

    def test_clear_errors(self) -> None:
        """Test clearing validation errors."""
        builder = _BaseBuilder()
        builder._validate_required("missing_key1")
        builder._validate_required("missing_key2")
        if builder.error_count != EXPECTED_BULK_SIZE:
            error_msg_43: str = f"Expected {2}, got {builder.error_count}"
            raise AssertionError(error_msg_43)

        builder._clear_errors()

        if builder.error_count != 0:
            error_msg_44: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_44)
        assert builder._get_errors() == []

    def test_add_error_valid(self) -> None:
        """Test adding valid error message."""
        builder = _BaseBuilder()

        builder._add_error("Custom error message")

        if builder.error_count != 1:
            error_msg_45: str = f"Expected {1}, got {builder.error_count}"
            raise AssertionError(error_msg_45)
        if "Custom error message" not in builder._get_errors():
            error_msg_46: str = (
                f"Expected Custom error message in {builder._get_errors()}"
            )
            raise AssertionError(
                error_msg_46,
            )

    def test_add_error_empty_string(self) -> None:
        """Test adding empty error message."""
        builder = _BaseBuilder()

        builder._add_error("")

        if builder.error_count != 0:
            empty_error_count_error_msg_47: str = (
                f"Expected {0}, got {builder.error_count}"
            )
            raise AssertionError(empty_error_count_error_msg_47)

    def test_add_error_none(self) -> None:
        """Test adding None error message."""
        builder = _BaseBuilder()

        builder._add_error(None)  # type: ignore[arg-type] # Intentional type error for testing

        if builder.error_count != 0:
            none_error_count_error_msg_48: str = (
                f"Expected {0}, got {builder.error_count}"
            )
            raise AssertionError(none_error_count_error_msg_48)

    def test_is_valid_no_errors(self) -> None:
        """Test is_valid when there are no errors."""
        builder = _BaseBuilder()

        if not (builder._is_valid()):
            error_msg_49: str = f"Expected True, got {builder._is_valid()}"
            raise AssertionError(error_msg_49)

    def test_is_valid_with_errors(self) -> None:
        """Test is_valid when there are errors."""
        builder = _BaseBuilder()
        builder._add_error("Test error")

        if builder._is_valid():
            error_msg_50: str = f"Expected False, got {builder._is_valid()}"
            raise AssertionError(error_msg_50)

    def test_get_errors_empty(self) -> None:
        """Test getting errors when list is empty."""
        builder = _BaseBuilder()

        errors = builder._get_errors()

        if errors != []:
            error_msg_51: str = f"Expected {[]}, got {errors}"
            raise AssertionError(error_msg_51)
        assert isinstance(errors, list)

    def test_get_errors_with_errors(self) -> None:
        """Test getting errors when errors exist."""
        builder = _BaseBuilder()
        builder._add_error("Error 1")
        builder._add_error("Error 2")

        errors = builder._get_errors()

        if len(errors) != EXPECTED_BULK_SIZE:
            error_msg_52: str = f"Expected {2}, got {len(errors)}"
            raise AssertionError(error_msg_52)
        if "Error 1" not in errors:
            error_msg_53: str = f"Expected {'Error 1'} in {errors}"
            raise AssertionError(error_msg_53)
        assert "Error 2" in errors

    def test_get_errors_returns_copy(self) -> None:
        """Test that get_errors returns a copy, not the original list."""
        builder = _BaseBuilder()
        builder._add_error("Original error")

        errors = builder._get_errors()
        errors.append("Modified error")

        # Original errors should be unchanged
        assert builder.error_count == 1
        assert "Modified error" not in builder._get_errors()
        assert "Original error" in builder._get_errors()

    def test_reset(self) -> None:
        """Test resetting builder state."""
        builder = _BaseBuilder()
        builder._set_property("key1", "value1")
        builder._add_error("Test error")
        builder._mark_built()

        result = builder._reset()

        assert result is builder
        if builder.property_count != 0:
            error_msg_54: str = f"Expected {0}, got {builder.property_count}"
            raise AssertionError(error_msg_54)
        assert builder.error_count == 0
        if builder.is_built:
            error_msg_55: str = f"Expected False, got {builder.is_built}"
            raise AssertionError(error_msg_55)
        assert builder.property_keys == []

    def test_mark_built(self) -> None:
        """Test marking builder as built."""
        builder = _BaseBuilder()
        if builder.is_built:
            error_msg_56: str = f"Expected False, got {builder.is_built}"
            raise AssertionError(error_msg_56)

        builder._mark_built()

        if not (builder.is_built):
            error_msg_57: str = f"Expected True, got {builder.is_built}"
            raise AssertionError(error_msg_57)

    def test_properties_immutable_after_built(self) -> None:
        """Test that properties cannot be modified after marking as built."""
        builder = _BaseBuilder()
        builder._set_property("before_built", "value1")
        builder._mark_built()

        # Try to set after built
        builder._set_property("after_built", "value2")

        # Should have error and no new property
        if builder.property_count != 1:
            error_msg_58: str = f"Expected {1}, got {builder.property_count}"
            raise AssertionError(error_msg_58)
        assert builder.error_count == 1
        assert not builder._has_property("after_built")


@pytest.mark.unit
class TestBaseFluentBuilder:
    """Test _BaseFluentBuilder functionality."""

    def test_fluent_builder_initialization(self) -> None:
        """Test fluent builder initialization."""
        builder = _BaseFluentBuilder("fluent_test")

        if builder.builder_name != "fluent_test":
            error_msg_59: str = f"Expected fluent_test, got {builder.builder_name}"
            raise AssertionError(
                error_msg_59,
            )
        assert builder.property_count == 0
        if builder.error_count != 0:
            error_msg_60: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_60)
        if builder.is_built:
            error_msg_61: str = f"Expected False, got {builder.is_built}"
            raise AssertionError(error_msg_61)

    def test_with_property_fluent(self) -> None:
        """Test fluent property setting."""
        builder = _BaseFluentBuilder()

        result = builder.with_property("key1", "value1")

        assert result is builder
        if builder._get_property("key1") != "value1":
            error_msg_62: str = f"Expected value1, got {builder._get_property('key1')}"
            raise AssertionError(
                error_msg_62,
            )

    def test_with_property_chaining(self) -> None:
        """Test fluent property setting with chaining."""
        builder = _BaseFluentBuilder()

        result = (
            builder.with_property("key1", "value1")
            .with_property("key2", "value2")
            .with_property("key3", "value3")
        )

        assert result is builder
        if builder.property_count != EXPECTED_DATA_COUNT:
            error_msg_63: str = f"Expected {3}, got {builder.property_count}"
            raise AssertionError(error_msg_63)
        assert builder._get_property("key1") == "value1"
        if builder._get_property("key2") != "value2":
            error_msg_64: str = f"Expected value2, got {builder._get_property('key2')}"
            raise AssertionError(
                error_msg_64,
            )
        assert builder._get_property("key3") == "value3"

    def test_when_condition_true(self) -> None:
        """Test conditional builder with true condition."""
        builder = _BaseFluentBuilder()

        result = builder.when(condition=True)

        assert result is builder
        if not (builder._get_property("_last_condition")):
            error_msg_65: str = (
                f"Expected True, got {builder._get_property('_last_condition')}"
            )
            raise AssertionError(
                error_msg_65,
            )

    def test_when_condition_false(self) -> None:
        """Test conditional builder with false condition."""
        builder = _BaseFluentBuilder()

        result = builder.when(condition=False)

        assert result is builder
        if builder._get_property("_last_condition"):
            error_msg_66: str = (
                f"Expected False, got {builder._get_property('_last_condition')}"
            )
            raise AssertionError(
                error_msg_66,
            )

    def test_then_set_after_true_condition(self) -> None:
        """Test then_set after true condition."""
        builder = _BaseFluentBuilder()

        result = builder.when(condition=True).then_set(
            "conditional_key",
            "conditional_value",
        )

        assert result is builder
        if builder._get_property("conditional_key") != "conditional_value":
            error_msg_67: str = f"Expected conditional_value, got {builder._get_property('conditional_key')}"
            raise AssertionError(
                error_msg_67,
            )

    def test_then_set_after_false_condition(self) -> None:
        """Test then_set after false condition."""
        builder = _BaseFluentBuilder()

        result = builder.when(condition=False).then_set(
            "conditional_key",
            "conditional_value",
        )

        assert result is builder
        assert builder._get_property("conditional_key") is None
        assert not builder._has_property("conditional_key")

    def test_then_set_without_when(self) -> None:
        """Test then_set without previous when (defaults to True)."""
        builder = _BaseFluentBuilder()

        result = builder.then_set("key", "value")

        assert result is builder
        if builder._get_property("key") != "value":
            error_msg_68: str = f"Expected value, got {builder._get_property('key')}"
            raise AssertionError(
                error_msg_68,
            )

    def test_then_set_invalid_condition_type(self) -> None:
        """Test then_set with non-boolean condition."""
        builder = _BaseFluentBuilder()
        builder._set_property("_last_condition", "not_a_bool")

        result = builder.then_set("key", "value")

        assert result is builder
        assert not builder._has_property(
            "key",
        )  # Should not set when condition is not boolean

    def test_conditional_chaining_complex(self) -> None:
        """Test complex conditional chaining."""
        builder = _BaseFluentBuilder()

        result = (
            builder.when(condition=True)
            .then_set("key1", "value1")
            .when(condition=False)
            .then_set("key2", "value2")
            .when(condition=True)
            .then_set("key3", "value3")
        )

        assert result is builder
        if builder._get_property("key1") != "value1":
            error_msg_69: str = (
                f"Expected {'value1'}, got {builder._get_property('key1')}"
            )
            raise AssertionError(error_msg_69)
        assert not builder._has_property("key2")
        if builder._get_property("key3") != "value3":
            error_msg_70: str = (
                f"Expected {'value3'}, got {builder._get_property('key3')}"
            )
            raise AssertionError(error_msg_70)

    def test_validate_fluent(self) -> None:
        """Test fluent validate method."""
        builder = _BaseFluentBuilder()

        result = builder.validate()

        assert result is builder
        # Base implementation does nothing, but should return self

    def test_clear_errors_fluent(self) -> None:
        """Test fluent clear_errors method."""
        builder = _BaseFluentBuilder()
        builder._add_error("Test error")
        if builder.error_count != 1:
            error_msg_71: str = f"Expected {1}, got {builder.error_count}"
            raise AssertionError(error_msg_71)

        result = builder.clear_errors()

        assert result is builder
        if builder.error_count != 0:
            error_msg_72: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_72)

    def test_reset_fluent(self) -> None:
        """Test fluent reset method."""
        builder = _BaseFluentBuilder()
        builder.with_property("key", "value")
        builder._add_error("Test error")

        result = builder.reset()

        assert result is builder
        if builder.property_count != 0:
            error_msg_73: str = f"Expected {0}, got {builder.property_count}"
            raise AssertionError(error_msg_73)
        assert builder.error_count == 0
        if builder.is_built:
            error_msg_74: str = f"Expected False, got {builder.is_built}"
            raise AssertionError(error_msg_74)

    def test_fluent_complex_workflow(self) -> None:
        """Test complex fluent builder workflow."""
        builder = (
            _BaseFluentBuilder("complex_workflow")
            .with_property("base_key", "base_value")
            .when(condition=True)
            .then_set("feature_enabled", value=True)
            .with_property("count", 10)
            .when(condition=False)
            .then_set("disabled_feature", "should_not_exist")
            .validate()
            .clear_errors()
        )

        if builder.builder_name != "complex_workflow":
            error_msg_75: str = (
                f"Expected {'complex_workflow'}, got {builder.builder_name}"
            )
            raise AssertionError(error_msg_75)
        assert (
            builder.property_count >= 3
        )  # base_key, feature_enabled, count, _last_condition
        if builder._get_property("base_key") != "base_value":
            base_value_error_msg: str = (
                f"Expected base_value, got {builder._get_property('base_key')}"
            )
            raise AssertionError(base_value_error_msg)
        if not (builder._get_property("feature_enabled")):
            error_msg_76: str = (
                f"Expected True, got {builder._get_property('feature_enabled')}"
            )
            raise AssertionError(error_msg_76)
        if builder._get_property("count") != 10:
            error_msg_77: str = f"Expected {10}, got {builder._get_property('count')}"
            raise AssertionError(error_msg_77)
        assert not builder._has_property("disabled_feature")
        if builder.error_count != 0:
            error_msg_78: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_78)


@pytest.mark.unit
class TestBuilderFactories:
    """Test builder factory functions."""

    def test_create_builder_default(self) -> None:
        """Test creating builder with default name."""
        builder = _create_builder()

        assert isinstance(builder, _BaseBuilder)
        if builder.builder_name != "unnamed":
            error_msg_79: str = f"Expected {'unnamed'}, got {builder.builder_name}"
            raise AssertionError(error_msg_79)

    def test_create_builder_with_name(self) -> None:
        """Test creating builder with custom name."""
        builder = _create_builder("custom_builder")

        assert isinstance(builder, _BaseBuilder)
        if builder.builder_name != "custom_builder":
            error_msg_80: str = f"Expected custom_builder, got {builder.builder_name}"
            raise AssertionError(error_msg_80)

    def test_create_fluent_builder_default(self) -> None:
        """Test creating fluent builder with default name."""
        builder = _create_fluent_builder()

        assert isinstance(builder, _BaseFluentBuilder)
        if builder.builder_name != "unnamed":
            error_msg_81: str = f"Expected unnamed, got {builder.builder_name}"
            raise AssertionError(error_msg_81)

    def test_create_fluent_builder_with_name(self) -> None:
        """Test creating fluent builder with custom name."""
        builder = _create_fluent_builder("custom_fluent")

        assert isinstance(builder, _BaseFluentBuilder)
        if builder.builder_name != "custom_fluent":
            error_msg_82: str = f"Expected custom_fluent, got {builder.builder_name}"
            raise AssertionError(error_msg_82)


@pytest.mark.unit
class TestBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def test_property_overwrite(self) -> None:
        """Test overwriting existing property."""
        builder = _BaseBuilder()
        builder._set_property("key", "original_value")

        builder._set_property("key", "new_value")

        if builder._get_property("key") != "new_value":
            error_msg_83: str = (
                f"Expected new_value, got {builder._get_property('key')}"
            )
            raise AssertionError(error_msg_83)
        assert builder.property_count == 1

    def test_multiple_validation_errors(self) -> None:
        """Test accumulating multiple validation errors."""
        builder = _BaseBuilder()

        builder._validate_required("missing1")
        builder._validate_required("missing2")
        builder._validate_type("non_existent", str)
        builder._add_error("Custom error")

        assert (
            builder.error_count == EXPECTED_DATA_COUNT
        )  # Only required validations and custom error add errors
        errors = builder._get_errors()
        assert any("missing1" in error for error in errors)
        assert any("missing2" in error for error in errors)
        assert "Custom error" in errors

    def test_property_with_complex_values(self) -> None:
        """Test storing complex property values."""
        builder = _BaseBuilder()

        builder._set_property(
            "complex", "complex_test_value"
        )  # Use string for type compatibility

        retrieved = builder._get_property("complex")
        assert retrieved == "complex_test_value"
        assert isinstance(retrieved, str)
        # Test simplified since we're using string value
        assert len(retrieved) > 0

    def test_builder_name_immutable(self) -> None:
        """Test that builder name is immutable."""
        builder = _BaseBuilder("original_name")

        # Should not be able to modify builder_name through property
        if builder.builder_name != "original_name":
            error_msg_84: str = f"Expected original_name, got {builder.builder_name}"
            raise AssertionError(
                error_msg_84,
            )
        # No public setter exists for builder_name

    def test_string_length_validation_edge_cases(self) -> None:
        """Test string length validation with edge cases."""
        builder = _BaseBuilder()

        # Empty string
        builder._set_property("empty", "")
        if not (builder._validate_string_length("empty", min_length=0)):
            error_msg_85: str = f"Expected True, got {builder._validate_string_length('empty', min_length=0)}"
            raise AssertionError(error_msg_85)
        if builder._validate_string_length("empty", min_length=1):
            error_msg_86: str = f"Expected False, got {builder._validate_string_length('empty', min_length=1)}"
            raise AssertionError(error_msg_86)

        # Single character
        builder._set_property("single", "a")
        assert (
            builder._validate_string_length(
                "single",
                min_length=1,
                max_length=1,
            )
            is True
        )

        # Long string
        builder._set_property("long_string", "this is a long string")
        # Max length None (no maximum)
        assert (
            builder._validate_string_length(
                "long_string",
                min_length=1,
                max_length=None,
            )
            is True
        )

    def test_validation_with_built_state(self) -> None:
        """Test validation methods work even after builder is built."""
        builder = _BaseBuilder()
        builder._set_property("test_key", "test_value")
        builder._mark_built()

        # Validation should still work
        if not (builder._validate_required("test_key")):
            error_msg_87: str = (
                f"Expected True, got {builder._validate_required('test_key')}"
            )
            raise AssertionError(error_msg_87)
        assert builder._validate_type("test_key", str) is True
        if not (builder._validate_string_length("test_key", min_length=1)):
            error_msg_88: str = f"Expected True, got {builder._validate_string_length('test_key', min_length=1)}"
            raise AssertionError(error_msg_88)


@pytest.mark.integration
class TestBuilderIntegration:
    """Integration tests for builder functionality."""

    def test_full_builder_lifecycle(self) -> None:
        """Test complete builder lifecycle."""
        builder = _BaseBuilder("lifecycle_test")

        # Configuration phase
        builder._set_property("name", "test_entity")
        builder._set_property("version", "1.0.0")
        builder._set_property("enabled", value=True)

        # Validation phase
        if not (builder._validate_required("name")):
            error_msg_89: str = (
                f"Expected True, got {builder._validate_required('name')}"
            )
            raise AssertionError(error_msg_89)
        assert builder._validate_required("version") is True
        if not (builder._validate_type("name", str)):
            error_msg_90: str = (
                f"Expected True, got {builder._validate_type('name', str)}"
            )
            raise AssertionError(error_msg_90)
        assert builder._validate_type("enabled", bool) is True
        if not (builder._validate_string_length("name", min_length=3)):
            error_msg_91: str = f"Expected True, got {builder._validate_string_length('name', min_length=3)}"
            raise AssertionError(error_msg_91)

        # State checks
        if not (builder._is_valid()):
            error_msg_92: str = f"Expected True, got {builder._is_valid()}"
            raise AssertionError(error_msg_92)
        if builder.property_count != EXPECTED_DATA_COUNT:
            error_msg_93: str = f"Expected {3}, got {builder.property_count}"
            raise AssertionError(error_msg_93)
        assert builder.error_count == 0

        # Build phase
        builder._mark_built()
        if not (builder.is_built):
            error_msg_94: str = f"Expected True, got {builder.is_built}"
            raise AssertionError(error_msg_94)

        # Post-build immutability
        builder._set_property("new_key", "should_fail")
        if builder.error_count != 1:
            error_msg_95: str = f"Expected {1}, got {builder.error_count}"
            raise AssertionError(error_msg_95)
        assert not builder._has_property("new_key")

    def test_fluent_builder_complex_workflow(self) -> None:
        """Test complex fluent builder workflow."""
        config_enabled = True
        debug_mode = False

        builder = (
            _BaseFluentBuilder("complex_config")
            .with_property("app_name", "TestApp")
            .with_property("version", "2.0.0")
            .when(condition=config_enabled)
            .then_set("config_file", "config.json")
            .then_set("load_config", value=True)
            .when(condition=debug_mode)
            .then_set("debug_level", "verbose")
            .then_set("log_file", "debug.log")
            .with_property("port", FlextConstants.Platform.FLEXCORE_PORT)
            .validate()
        )

        # Verify results
        if builder.builder_name != "complex_config":
            error_msg_96: str = f"Expected complex_config, got {builder.builder_name}"
            raise AssertionError(error_msg_96)
        assert builder._get_property("app_name") == "TestApp"
        if builder._get_property("version") != "2.0.0":
            error_msg_97: str = (
                f"Expected 2.0.0, got {builder._get_property('version')}"
            )
            raise AssertionError(error_msg_97)
        assert builder._get_property("config_file") == "config.json"
        if not (builder._get_property("load_config")):
            error_msg_98: str = (
                f"Expected True, got {builder._get_property('load_config')}"
            )
            raise AssertionError(error_msg_98)
        if builder._get_property("port") != FlextConstants.Platform.FLEXCORE_PORT:
            error_msg_99: str = f"Expected FlextConstants.Platform.FLEXCORE_PORT, got {builder._get_property('port')}"
            raise AssertionError(error_msg_99)

        # Debug properties should not be set
        assert not builder._has_property("debug_level")
        assert not builder._has_property("log_file")

        # Should be valid
        if not (builder._is_valid()):
            error_msg_100: str = f"Expected True, got {builder._is_valid()}"
            raise AssertionError(error_msg_100)

    def test_builder_error_recovery(self) -> None:
        """Test builder error recovery workflow."""
        builder = _BaseFluentBuilder("error_recovery")

        # Introduce errors
        builder.with_property("", "invalid_key")  # Invalid key
        builder._validate_required("missing_field")  # Missing required field
        builder._set_property("number_field", "not_a_number")
        builder._validate_type("number_field", int)  # Type mismatch

        # Check errors exist
        initial_errors = builder.error_count
        assert initial_errors > 0
        assert not builder._is_valid()

        # Clear errors and fix issues
        builder.clear_errors()
        builder.with_property("valid_key", "valid_value")
        builder.with_property("missing_field", "now_present")
        builder.with_property("number_field", 42)

        # Revalidate
        if not (builder._validate_required("missing_field")):
            missing_field_validation_error_msg: str = (
                f"Expected True, got {builder._validate_required('missing_field')}"
            )
            raise AssertionError(missing_field_validation_error_msg)
        assert builder._validate_type("number_field", int) is True

        # Should now be valid
        if not (builder._is_valid()):
            error_msg_101: str = f"Expected True, got {builder._is_valid()}"
            raise AssertionError(error_msg_101)
        if builder.error_count != 0:
            error_msg_102: str = f"Expected {0}, got {builder.error_count}"
            raise AssertionError(error_msg_102)

    def test_builder_inheritance_compatibility(self) -> None:
        """Test that fluent builder maintains base builder functionality."""
        base_builder = _BaseBuilder("base_test")
        fluent_builder = _BaseFluentBuilder("fluent_test")

        # Both should support same base operations
        base_builder._set_property("key", "value")
        fluent_builder.with_property("key", "value")

        if base_builder._get_property("key") != fluent_builder._get_property("key"):
            error_msg_103: str = f"Expected {fluent_builder._get_property('key')}, got {base_builder._get_property('key')}"
            raise AssertionError(error_msg_103)
        assert base_builder._has_property("key") == fluent_builder._has_property("key")
        assert base_builder._validate_required(
            "key",
        ) == fluent_builder._validate_required("key")

        # Both should have same property interface
        if base_builder.property_count != fluent_builder.property_count:
            error_msg_104: str = f"Expected {fluent_builder.property_count}, got {base_builder.property_count}"
            raise AssertionError(error_msg_104)
        assert base_builder.error_count == fluent_builder.error_count
        if base_builder.is_built != fluent_builder.is_built:
            is_built_comparison_error_msg: str = (
                f"Expected {fluent_builder.is_built}, got {base_builder.is_built}"
            )
            raise AssertionError(is_built_comparison_error_msg)
