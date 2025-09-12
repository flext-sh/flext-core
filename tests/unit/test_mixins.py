"""Simplified mixins tests."""

from flext_core import FlextMixins


class TestMixinsSimple:
    """Test mixins functionality."""

    def test_serializable_exists(self) -> None:
        """Test that Serializable mixin exists."""
        assert hasattr(FlextMixins, "Serializable")

    def test_loggable_exists(self) -> None:
        """Test that Loggable mixin exists."""
        assert hasattr(FlextMixins, "Loggable")

    def test_to_json_works(self) -> None:
        """Test that to_json method works."""
        result = FlextMixins.to_json({"test": "data"})
        assert "test" in result
