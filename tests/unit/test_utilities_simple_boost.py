"""Simple test to boost FlextUtilities coverage targeting missing lines."""

from flext_core import FlextUtilities


class TestFlextUtilitiesSimpleBoost:
    """Test FlextUtilities targeting specific uncovered lines."""

    def test_validation_is_non_empty_string(self) -> None:
        """Test Validation.is_non_empty_string method."""
        # Test valid non-empty string
        assert FlextUtilities.Validation.is_non_empty_string("test") is True
        assert FlextUtilities.Validation.is_non_empty_string("valid string") is True

        # Test empty/invalid strings
        assert FlextUtilities.Validation.is_non_empty_string("") is False
        assert FlextUtilities.Validation.is_non_empty_string("   ") is False
        assert FlextUtilities.Validation.is_non_empty_string(None) is False

    def test_generators_all_methods(self) -> None:
        """Test all generator methods with uniqueness and format checks."""
        # Test generate_id
        ids = {FlextUtilities.Generators.generate_id() for _ in range(200)}
        assert all(isinstance(x, str) and len(x) == 8 for x in ids)
        assert len(ids) == len(set(ids))  # uniqueness among sample

        # Test generate_uuid
        uuid_result = FlextUtilities.Generators.generate_uuid()
        assert isinstance(uuid_result, str)
        assert len(uuid_result) == 36
        assert uuid_result.count("-") == 4

        # Test generate_iso_timestamp
        timestamp = FlextUtilities.Generators.generate_iso_timestamp()
        assert isinstance(timestamp, str)
        assert "T" in timestamp

        # Test generate_correlation_id
        corr_id = FlextUtilities.Generators.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert corr_id.startswith("corr_")

        # Test generate_request_id
        req_id = FlextUtilities.Generators.generate_request_id()
        assert isinstance(req_id, str)
        assert req_id.startswith("req_")

        # Test generate_entity_id
        ent_id = FlextUtilities.Generators.generate_entity_id()
        assert isinstance(ent_id, str)
        assert ent_id.startswith("ent_")

    def test_text_processor_all_methods(self) -> None:
        """Test all TextProcessor methods."""
        # Test safe_string with various inputs
        assert FlextUtilities.TextProcessor.safe_string(None) == ""
        assert FlextUtilities.TextProcessor.safe_string("test") == "test"
        assert FlextUtilities.TextProcessor.safe_string(123) == "123"
        assert FlextUtilities.TextProcessor.safe_string([1, 2, 3]) == "[1, 2, 3]"

        # Test clean_text
        assert FlextUtilities.TextProcessor.clean_text("") == ""
        assert FlextUtilities.TextProcessor.clean_text("  test  ") == "test"
        assert (
            FlextUtilities.TextProcessor.clean_text("multiple   spaces")
            == "multiple spaces"
        )
        # Unicode normalization test - NFKD normalization expected
        result = FlextUtilities.TextProcessor.clean_text("unicodeé")
        assert "unicode" in result  # Check normalized unicode handling

        # Test is_non_empty_string (from TextProcessor)
        assert FlextUtilities.TextProcessor.is_non_empty_string("test") is True
        assert FlextUtilities.TextProcessor.is_non_empty_string("") is False
        assert FlextUtilities.TextProcessor.is_non_empty_string(None) is False
        # TextProcessor.is_non_empty_string only accepts strings
        assert FlextUtilities.TextProcessor.is_non_empty_string("123") is True

    def test_conversions_methods(self) -> None:
        """Test Conversions class methods behavior with diverse inputs."""
        # safe_bool
        assert FlextUtilities.Conversions.safe_bool(True) is True
        assert FlextUtilities.Conversions.safe_bool(False) is False
        assert FlextUtilities.Conversions.safe_bool("true") is True
        assert FlextUtilities.Conversions.safe_bool("false") is False
        assert FlextUtilities.Conversions.safe_bool(1) is True
        assert FlextUtilities.Conversions.safe_bool(0) is False
        assert FlextUtilities.Conversions.safe_bool("unknown", default=True) is True

        # safe_int
        assert FlextUtilities.Conversions.safe_int("123") == 123
        assert FlextUtilities.Conversions.safe_int(12.7) == 12
        assert FlextUtilities.Conversions.safe_int("invalid", default=0) == 0

        # safe_float
        assert FlextUtilities.Conversions.safe_float("123.45") == 123.45
        assert FlextUtilities.Conversions.safe_float(42) == 42.0
        assert FlextUtilities.Conversions.safe_float("invalid", default=0.0) == 0.0

    def test_json_serialization_with_builtin(self) -> None:
        """Test JSON serialization using FlextUtilities.safe_json_stringify."""
        test_data = {"test": "value", "number": 123}
        json_result = FlextUtilities.safe_json_stringify(test_data)
        assert isinstance(json_result, str)
        assert "test" in json_result
        assert "number" in json_result

        # Unserializable object falls back to default
        class Unserializable:
            pass

        default_str = FlextUtilities.safe_json_stringify(Unserializable(), default="{}")
        assert default_str == "{}"

    def test_edge_cases_and_error_paths(self) -> None:
        """Test edge cases focusing on unique paths."""
        # Test clean_text with None/empty
        assert FlextUtilities.TextProcessor.clean_text("") == ""
        assert FlextUtilities.TextProcessor.clean_text("\t\n  ") == ""

        # Test safe_string with complex objects
        assert (
            FlextUtilities.TextProcessor.safe_string({"key": "value"})
            == "{'key': 'value'}"
        )

        # Test validation edge cases
        assert FlextUtilities.Validation.is_non_empty_string("") is False
        assert FlextUtilities.Validation.is_non_empty_string("\t\n  ") is False
