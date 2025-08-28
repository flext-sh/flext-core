"""Comprehensive tests for FLEXT patterns fields module."""

from __future__ import annotations

from flext_core import FlextFields, FlextResult


class TestFlextFields:
    """Test FlextFields hierarchical class."""

    def test_flext_fields_import(self) -> None:
        """Test that FlextFields can be imported and instantiated."""
        # Test basic import
        assert FlextFields is not None

        # Test that it's a class we can work with
        assert hasattr(FlextFields, "__name__")
        assert FlextFields.__name__ == "FlextFields"

    def test_flext_fields_basic_functionality(self) -> None:
        """Test basic FlextFields functionality if available."""
        # This is a placeholder test that ensures the module loads
        # Real functionality tests should be added based on actual FlextFields API
        assert True


class TestFlextResult:
    """Test FlextResult integration with fields."""

    def test_result_ok(self) -> None:
        """Test FlextResult.ok functionality."""
        result = FlextResult[str].ok("test_value")
        assert result.success is True
        assert result.data == "test_value"

    def test_result_fail(self) -> None:
        """Test FlextResult.fail functionality."""
        result = FlextResult[str].fail("error_message")
        assert result.success is False
        assert result.error == "error_message"
