"""Simple test for FlextTypeAdapters to boost coverage."""

from flext_core import FlextTypeAdapters


class TestFlextTypeAdaptersSimple:
    """Test FlextTypeAdapters basic functionality."""

    def test_adapt_to_dict_with_model_dump(self) -> None:
        """Test adapt_to_dict with model_dump method."""

        class MockModel:
            def model_dump(self) -> dict[str, str]:
                return {"test": "value"}

        result = FlextTypeAdapters.adapt_to_dict(MockModel())
        assert result == {"test": "value"}

    def test_adapt_to_dict_with_dict_method(self) -> None:
        """Test adapt_to_dict with dict method."""

        class MockModel:
            def dict(self) -> dict[str, str]:
                return {"legacy": "value"}

        result = FlextTypeAdapters.adapt_to_dict(MockModel())
        assert result == {"legacy": "value"}

    def test_adapt_to_dict_fallback(self) -> None:
        """Test adapt_to_dict fallback for regular objects."""
        result = FlextTypeAdapters.adapt_to_dict("simple_string")
        assert result == {"value": "simple_string"}

    def test_adapt_to_dict_invalid_return_types(self) -> None:
        """Test adapt_to_dict with invalid return types from methods."""

        class MockModel:
            def model_dump(self) -> str:  # Returns non-dict
                return "not_a_dict"

        result = FlextTypeAdapters.adapt_to_dict(MockModel())
        assert result == {"value": "not_a_dict"}

        class MockModel2:
            def dict(self) -> int:  # Returns non-dict
                return 42

        result2 = FlextTypeAdapters.adapt_to_dict(MockModel2())
        assert result2 == {"value": 42}
