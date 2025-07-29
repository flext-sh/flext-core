"""Tests for FLEXT Core Builder Base module."""

from __future__ import annotations

import pytest

from flext_core._builder_base import _BaseBuilder

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestBaseBuilder:
    """Test _BaseBuilder functionality."""

    def test_builder_creation_success(self) -> None:
        """Test successful builder creation."""
        builder = _BaseBuilder("test_builder")

        assert builder._builder_name == "test_builder"
        assert builder._properties == {}
        assert builder._validation_errors == []
        assert builder._is_built is False

    def test_builder_creation_default_name(self) -> None:
        """Test builder creation with default name."""
        builder = _BaseBuilder()

        assert builder._builder_name == "unnamed"
        assert builder._properties == {}
        assert builder._validation_errors == []
        assert builder._is_built is False

    def test_set_property_success(self) -> None:
        """Test successful property setting."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property("key1", "value1")

        assert result is builder  # Should return self for chaining
        assert builder._properties["key1"] == "value1"
        assert len(builder._validation_errors) == 0

    def test_set_property_multiple(self) -> None:
        """Test setting multiple properties."""
        builder = _BaseBuilder("test_builder")

        builder._set_property("key1", "value1")
        builder._set_property("key2", 42)
        builder._set_property("key3", {"nested": "dict"})

        assert builder._properties["key1"] == "value1"
        assert builder._properties["key2"] == 42
        assert builder._properties["key3"] == {"nested": "dict"}
        assert len(builder._validation_errors) == 0

    def test_set_property_invalid_key_empty(self) -> None:
        """Test setting property with empty key."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property("", "value")

        assert result is builder
        assert "Invalid property key: " in builder._validation_errors[0]
        assert "" not in builder._properties

    def test_set_property_invalid_key_none(self) -> None:
        """Test setting property with None key."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property(None, "value")  # type: ignore[arg-type]

        assert result is builder
        assert "Invalid property key: None" in builder._validation_errors[0]

    def test_set_property_invalid_key_whitespace(self) -> None:
        """Test setting property with whitespace-only key."""
        builder = _BaseBuilder("test_builder")

        result = builder._set_property("   ", "value")

        assert result is builder
        assert "Invalid property key:    " in builder._validation_errors[0]

    def test_set_property_after_built(self) -> None:
        """Test setting property after builder is marked as built."""
        builder = _BaseBuilder("test_builder")
        builder._is_built = True  # Simulate built state

        result = builder._set_property("key", "value")

        assert result is builder
        assert "Cannot modify built object" in builder._validation_errors
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
        assert builder._properties["name"] == "test"
        assert builder._properties["value"] == 42
        assert builder._properties["enabled"] is True

    def test_validation_errors_accumulate(self) -> None:
        """Test that validation errors accumulate."""
        builder = _BaseBuilder("test_builder")

        builder._set_property("", "value1")
        builder._set_property("   ", "value2")
        builder._set_property(None, "value3")  # type: ignore[arg-type]

        assert len(builder._validation_errors) == 3
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

        assert "after_built" not in builder._properties
        assert "Cannot modify built object" in builder._validation_errors
        assert builder._properties["initial"] == "value"  # Original property preserved

    def test_builder_name_access(self) -> None:
        """Test accessing builder name."""
        name = "my_custom_builder"
        builder = _BaseBuilder(name)

        assert builder._builder_name == name

    def test_properties_access(self) -> None:
        """Test accessing properties dict."""
        builder = _BaseBuilder("test_builder")
        builder._set_property("prop1", "val1")
        builder._set_property("prop2", "val2")

        properties = builder._properties
        assert properties["prop1"] == "val1"
        assert properties["prop2"] == "val2"
        assert len(properties) == 2

    def test_validation_errors_access(self) -> None:
        """Test accessing validation errors."""
        builder = _BaseBuilder("test_builder")
        builder._set_property("", "invalid_key")

        errors = builder._validation_errors
        assert len(errors) == 1
        assert "Invalid property key:" in errors[0]

    def test_is_built_flag(self) -> None:
        """Test the is_built flag functionality."""
        builder = _BaseBuilder("test_builder")

        # Initially not built
        assert builder._is_built is False

        # Can set properties
        builder._set_property("test", "value")
        assert len(builder._validation_errors) == 0

        # Mark as built
        builder._is_built = True
        assert builder._is_built is True

        # Cannot set properties anymore
        builder._set_property("test2", "value2")
        assert "Cannot modify built object" in builder._validation_errors
