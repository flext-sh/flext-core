"""Tests for FLEXT Core Builder Base module."""

from __future__ import annotations

import pytest

from flext_core._builder_base import _BaseBuilder

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestBaseBuilder:
    """Test _BaseBuilder functionality."""

    def test_builder_creation_success(self) -> None:
        """Test successful builder creation."""
        builder = _BaseBuilder("test_builder")

        if builder._builder_name != "test_builder":
            raise AssertionError(
                f"Expected {'test_builder'}, got {builder._builder_name}"
            )
        assert builder._properties == {}
        if builder._validation_errors != []:
            raise AssertionError(f"Expected {[]}, got {builder._validation_errors}")
        if builder._is_built:
            raise AssertionError(f"Expected False, got {builder._is_built}")

    def test_builder_creation_default_name(self) -> None:
        """Test builder creation with default name."""
        builder = _BaseBuilder()

        if builder._builder_name != "unnamed":
            raise AssertionError(f"Expected {'unnamed'}, got {builder._builder_name}")
        assert builder._properties == {}
        if builder._validation_errors != []:
            raise AssertionError(f"Expected {[]}, got {builder._validation_errors}")
        if builder._is_built:
            raise AssertionError(f"Expected False, got {builder._is_built}")

    def test_set_property_success(self) -> None:
        """Test successful property setting."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property("key1", "value1")

        assert result is builder  # Should return self for chaining
        if builder._properties["key1"] != "value1":
            raise AssertionError(
                f"Expected {'value1'}, got {builder._properties['key1']}"
            )
        assert len(builder._validation_errors) == 0

    def test_set_property_multiple(self) -> None:
        """Test setting multiple properties."""
        builder = _BaseBuilder("test_builder")

        builder._set_property("key1", "value1")
        builder._set_property("key2", 42)
        builder._set_property("key3", {"nested": "dict"})

        if builder._properties["key1"] != "value1":
            raise AssertionError(
                f"Expected {'value1'}, got {builder._properties['key1']}"
            )
        assert builder._properties["key2"] == 42
        if builder._properties["key3"] != {"nested": "dict"}:
            raise AssertionError(
                f"Expected {{'nested': 'dict'}}, got {builder._properties['key3']}"
            )
        assert len(builder._validation_errors) == 0

    def test_set_property_invalid_key_empty(self) -> None:
        """Test setting property with empty key."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property("", "value")

        assert result is builder
        if "Invalid property key: " not in builder._validation_errors[0]:
            raise AssertionError(
                f"Expected {'Invalid property key: '} in {builder._validation_errors[0]}"
            )
        assert "" not in builder._properties

    def test_set_property_invalid_key_none(self) -> None:
        """Test setting property with None key."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property(None, "value")  # type: ignore[arg-type]

        assert result is builder
        if "Invalid property key: None" not in builder._validation_errors[0]:
            raise AssertionError(
                f"Expected {'Invalid property key: None'} in {builder._validation_errors[0]}"
            )

    def test_set_property_invalid_key_whitespace(self) -> None:
        """Test setting property with whitespace-only key."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property("   ", "value")

        assert result is builder
        if "Invalid property key:    " not in builder._validation_errors[0]:
            raise AssertionError(
                f"Expected {'Invalid property key:    '} in {builder._validation_errors[0]}"
            )

    def test_set_property_after_built(self) -> None:
        """Test setting property after builder is marked as built."""
        builder = _BaseBuilder("test_builder")
        builder._is_built = True  # Simulate built state

        result = builder._set_property("key", "value")

        assert result is builder
        if "Cannot modify built object" not in builder._validation_errors:
            raise AssertionError(
                f"Expected {'Cannot modify built object'} in {builder._validation_errors}"
            )
        assert "key" not in builder._properties

    def test_property_chaining(self) -> None:
        """Test fluent API property chaining."""
        builder = _BaseBuilder("test_builder")

        enabled_value = True
        result = (
            builder._set_property("name", "test")
            ._set_property("value", 42)
            ._set_property("enabled", enabled_value)
        )

        assert result is builder
        if builder._properties["name"] != "test":
            raise AssertionError(
                f"Expected {'test'}, got {builder._properties['name']}"
            )
        assert builder._properties["value"] == 42
        if not (builder._properties["enabled"]):
            raise AssertionError(f"Expected True, got {builder._properties['enabled']}")

    def test_validation_errors_accumulate(self) -> None:
        """Test that validation errors accumulate."""
        builder = _BaseBuilder("test_builder")

        builder._set_property("", "value1")
        builder._set_property("   ", "value2")
        builder._set_property(None, "value3")  # type: ignore[arg-type]

        if len(builder._validation_errors) != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {len(builder._validation_errors)}")
        assert all(
            "Invalid property key:" in error for error in builder._validation_errors
        )

    def test_builder_state_immutable_after_built(self) -> None:
        """Test that builder becomes immutable after built."""
        builder = _BaseBuilder("test_builder")
        builder._set_property("initial", "value")

        # Mark as built
        builder._is_built = True

        # Try to modify
        builder._set_property("after_built", "should_fail")

        if "after_built" not in builder._properties:
            raise AssertionError(
                f"Expected {'after_built'} not in {builder._properties}"
            )
        assert "Cannot modify built object" in builder._validation_errors
        if builder._properties["initial"] != "value":  # Original property preserved:
            raise AssertionError(
                f"Expected {'value'}, got {builder._properties['initial']}"
            )

    def test_builder_name_access(self) -> None:
        """Test accessing builder name."""
        name = "my_custom_builder"
        builder = _BaseBuilder(name)

        if builder._builder_name != name:
            raise AssertionError(f"Expected {name}, got {builder._builder_name}")

    def test_properties_access(self) -> None:
        """Test accessing properties dict."""
        builder = _BaseBuilder("test_builder")
        builder._set_property("prop1", "val1")
        builder._set_property("prop2", "val2")

        properties = builder._properties
        if properties["prop1"] != "val1":
            raise AssertionError(f"Expected {'val1'}, got {properties['prop1']}")
        assert properties["prop2"] == "val2"
        if len(properties) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(properties)}")

    def test_validation_errors_access(self) -> None:
        """Test accessing validation errors."""
        builder = _BaseBuilder("test_builder")
        builder._set_property("", "invalid_key")

        errors = builder._validation_errors
        if len(errors) != 1:
            raise AssertionError(f"Expected {1}, got {len(errors)}")
        if "Invalid property key:" not in errors[0]:
            raise AssertionError(f"Expected {'Invalid property key:'} in {errors[0]}")

    def test_is_built_flag(self) -> None:
        """Test the is_built flag functionality."""
        builder = _BaseBuilder("test_builder")

        # Initially not built
        if builder._is_built:
            raise AssertionError(f"Expected False, got {builder._is_built}")
        # Can set properties
        builder._set_property("test", "value")
        if len(builder._validation_errors) != 0:
            raise AssertionError(f"Expected {0}, got {len(builder._validation_errors)}")

        # Mark as built
        builder._is_built = True
        if not (builder._is_built):
            raise AssertionError(f"Expected True, got {builder._is_built}")

        # Cannot set properties anymore
        builder._set_property("test2", "value2")
        if "Cannot modify built object" not in builder._validation_errors:
            raise AssertionError(
                f"Expected {'Cannot modify built object'} in {builder._validation_errors}"
            )
