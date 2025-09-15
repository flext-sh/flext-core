"""Tests for refactored FlextUtilities - Testing REAL API, not implementation.

Tests only the actually used functionality across the FLEXT ecosystem.
Focuses on API behavior and functionality, not internal implementation details.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities


class TestFlextUtilitiesAPI:
    """Test FlextUtilities API functionality based on real ecosystem usage."""

    def test_generators_api_functionality(self) -> None:
        """Test Generators API - 99 real usages across ecosystem."""
        # Test generate_id functionality
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert len(id1) == 8  # Short IDs are 8 chars
        assert id1 != id2  # Should be unique

        # Test generate_uuid functionality
        uuid1 = FlextUtilities.Generators.generate_uuid()
        uuid2 = FlextUtilities.Generators.generate_uuid()
        assert isinstance(uuid1, str)
        assert isinstance(uuid2, str)
        assert len(uuid1) == 36  # Standard UUID format
        assert uuid1 != uuid2  # Should be unique

        # Test generate_iso_timestamp functionality
        timestamp = FlextUtilities.Generators.generate_iso_timestamp()
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format has T separator
        assert (
            timestamp.endswith(("Z", "+00:00")) or ":" in timestamp[-6:]
        )  # Timezone info

        # Test generate_correlation_id functionality
        corr_id = FlextUtilities.Generators.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert corr_id.startswith("corr_")
        assert len(corr_id) == 17  # "corr_" + 12 char UUID

    def test_text_processor_api_functionality(self) -> None:
        """Test TextProcessor API - 96 real usages across ecosystem."""
        # Test safe_string functionality
        assert FlextUtilities.TextProcessor.safe_string(None) == ""
        assert FlextUtilities.TextProcessor.safe_string("") == ""
        assert FlextUtilities.TextProcessor.safe_string("test") == "test"
        assert FlextUtilities.TextProcessor.safe_string(123) == "123"
        assert FlextUtilities.TextProcessor.safe_string([1, 2, 3]) == "[1, 2, 3]"

        # Test clean_text functionality
        assert FlextUtilities.TextProcessor.clean_text("") == ""
        assert FlextUtilities.TextProcessor.clean_text("  test   text  ") == "test text"
        assert FlextUtilities.TextProcessor.clean_text("test\n\ttext") == "test text"
        # Note: NFKD normalization keeps accented chars but may change representation
        cleaned_cafe = FlextUtilities.TextProcessor.clean_text("café")
        assert "cafe" in cleaned_cafe.lower() or "café" in cleaned_cafe

    def test_conversions_api_functionality(self) -> None:
        """Test Conversions API - 37 real usages across ecosystem."""
        # Test safe_int functionality
        assert FlextUtilities.Conversions.safe_int("123") == 123
        assert FlextUtilities.Conversions.safe_int("invalid") == 0
        assert FlextUtilities.Conversions.safe_int("invalid", default=42) == 42
        assert FlextUtilities.Conversions.safe_int(123.7) == 123
        assert FlextUtilities.Conversions.safe_int("invalid") == 0

        # Test safe_bool functionality
        assert FlextUtilities.Conversions.safe_bool(True) is True
        assert FlextUtilities.Conversions.safe_bool(False) is False
        assert FlextUtilities.Conversions.safe_bool("true") is True
        assert FlextUtilities.Conversions.safe_bool("1") is True
        assert FlextUtilities.Conversions.safe_bool("yes") is True
        assert FlextUtilities.Conversions.safe_bool("on") is True
        assert FlextUtilities.Conversions.safe_bool("false") is False
        assert FlextUtilities.Conversions.safe_bool("invalid") is False
        assert FlextUtilities.Conversions.safe_bool("invalid", default=True) is True
        assert FlextUtilities.Conversions.safe_bool(1) is True
        assert FlextUtilities.Conversions.safe_bool(0) is False

        # Test safe_float functionality
        assert FlextUtilities.Conversions.safe_float("123.45") == 123.45
        assert FlextUtilities.Conversions.safe_float("invalid") == 0.0
        assert FlextUtilities.Conversions.safe_float("invalid", default=42.5) == 42.5
        assert FlextUtilities.Conversions.safe_float(123) == 123.0
        assert FlextUtilities.Conversions.safe_float("invalid") == 0.0

    def test_type_guards_api_functionality(self) -> None:
        """Test TypeGuards API - 24 real usages across ecosystem."""
        # Test is_dict_non_empty functionality
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty("string") is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty([]) is False

        # Test is_string_non_empty functionality
        assert FlextUtilities.TypeGuards.is_string_non_empty("test") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert (
            FlextUtilities.TypeGuards.is_string_non_empty("   ") is False
        )  # Only whitespace
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(123) is False

        # Test is_list_non_empty functionality
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty("string") is False
        assert FlextUtilities.TypeGuards.is_list_non_empty({}) is False

    def test_direct_utility_functions(self) -> None:
        """Test direct utility functions - 20+ real usages across ecosystem."""
        # Test generate_iso_timestamp direct usage
        timestamp = FlextUtilities.generate_iso_timestamp()
        assert isinstance(timestamp, str)
        assert "T" in timestamp

        # Test safe_json_stringify functionality
        assert FlextUtilities.safe_json_stringify({"test": True}) == '{"test":true}'
        assert FlextUtilities.safe_json_stringify({"key": "value"}) == '{"key":"value"}'
        # Test safe_json_stringify with None returns valid JSON "null"
        assert FlextUtilities.safe_json_stringify(None) == "null"  # Valid JSON for None
        assert (
            FlextUtilities.safe_json_stringify(object()) == "{}"
        )  # Default for un-serializable

    def test_api_consistency(self) -> None:
        """Test that API remains consistent with ecosystem expectations."""
        # Verify main class exists
        assert hasattr(FlextUtilities, "Generators")
        assert hasattr(FlextUtilities, "TextProcessor")
        assert hasattr(FlextUtilities, "Conversions")
        assert hasattr(FlextUtilities, "TypeGuards")

        # Verify all expected methods exist
        assert hasattr(FlextUtilities.Generators, "generate_id")
        assert hasattr(FlextUtilities.Generators, "generate_uuid")
        assert hasattr(FlextUtilities.Generators, "generate_iso_timestamp")
        assert hasattr(FlextUtilities.Generators, "generate_correlation_id")

        assert hasattr(FlextUtilities.TextProcessor, "safe_string")
        assert hasattr(FlextUtilities.TextProcessor, "clean_text")

        assert hasattr(FlextUtilities.Conversions, "safe_int")
        assert hasattr(FlextUtilities.Conversions, "safe_bool")
        assert hasattr(FlextUtilities.Conversions, "safe_float")

        assert hasattr(FlextUtilities.TypeGuards, "is_dict_non_empty")
        assert hasattr(FlextUtilities.TypeGuards, "is_string_non_empty")
        assert hasattr(FlextUtilities.TypeGuards, "is_list_non_empty")

        # Verify direct methods exist
        assert hasattr(FlextUtilities, "generate_iso_timestamp")
        assert hasattr(FlextUtilities, "safe_json_stringify")


# Test for ecosystem compatibility - this is what matters in the real world
class TestEcosystemCompatibility:
    """Test compatibility with real ecosystem usage patterns."""

    def test_import_compatibility(self) -> None:
        """Test that imports work as expected in ecosystem."""
        # Test most common usage patterns found in ecosystem analysis
        id_val = FlextUtilities.Generators.generate_id()
        safe_str = FlextUtilities.TextProcessor.safe_string(None)
        safe_bool = FlextUtilities.Conversions.safe_bool("true")
        is_dict = FlextUtilities.TypeGuards.is_dict_non_empty({"test": 1})
        timestamp = FlextUtilities.generate_iso_timestamp()

        assert all([id_val, safe_str == "", safe_bool, is_dict, timestamp])

    def test_backward_compatibility(self) -> None:
        """Test that existing ecosystem code patterns still work."""
        # These are patterns found in actual ecosystem usage

        # Pattern 1: Direct timestamp usage (20+ occurrences)
        result = FlextUtilities.generate_iso_timestamp()
        assert isinstance(result, str)

        # Pattern 2: ID generation (99+ occurrences)
        result = FlextUtilities.Generators.generate_uuid()
        assert isinstance(result, str)
        assert len(result) == 36

        # Pattern 3: Safe conversions (37+ occurrences)
        bool_result = FlextUtilities.Conversions.safe_bool("invalid", default=True)
        assert bool_result is True

        # Pattern 4: Type guards (24+ occurrences)
        guard_result = FlextUtilities.TypeGuards.is_dict_non_empty({})
        assert guard_result is False

        # Pattern 5: Text processing (96+ occurrences)
        result = FlextUtilities.TextProcessor.safe_string(None)
        assert result == ""
