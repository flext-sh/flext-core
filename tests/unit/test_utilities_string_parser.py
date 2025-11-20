"""Comprehensive tests for FlextUtilitiesStringParser - 100% coverage target.

This module provides real tests (no mocks) for all string parsing functions
in FlextUtilitiesStringParser to achieve 100% code coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities

# ============================================================================
# Test Parse Delimited
# ============================================================================


class TestFlextUtilitiesStringParserParseDelimited:
    """Test parse_delimited method."""

    def test_parse_delimited_basic(self) -> None:
        """Test basic delimited string parsing."""
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a,b,c", ",")
        assert result.is_success
        assert result.unwrap() == ["a", "b", "c"]

    def test_parse_delimited_with_spaces(self) -> None:
        """Test parsing with spaces."""
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a, b, c", ",")
        assert result.is_success
        assert result.unwrap() == ["a", "b", "c"]

    def test_parse_delimited_empty_string(self) -> None:
        """Test parsing empty string."""
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("", ",")
        assert result.is_success
        assert result.unwrap() == []

    def test_parse_delimited_with_options(self) -> None:
        """Test parsing with ParseOptions."""
        parser = FlextUtilities.StringParser()
        options = FlextUtilities.StringParser.ParseOptions(
            strip=True, remove_empty=True
        )
        result = parser.parse_delimited("a, b, c", ",", options=options)
        assert result.is_success
        assert result.unwrap() == ["a", "b", "c"]

    def test_parse_delimited_options_no_strip(self) -> None:
        """Test parsing with options, strip=False."""
        options = FlextUtilities.StringParser.ParseOptions(
            strip=False, remove_empty=True
        )
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a, b, c", ",", options=options)
        assert result.is_success
        assert result.unwrap() == ["a", " b", " c"]

    def test_parse_delimited_options_no_remove_empty(self) -> None:
        """Test parsing with options, remove_empty=False."""
        options = FlextUtilities.StringParser.ParseOptions(
            strip=True, remove_empty=False
        )
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a,,c", ",", options=options)
        assert result.is_success
        assert result.unwrap() == ["a", "", "c"]

    def test_parse_delimited_with_validator_success(self) -> None:
        """Test parsing with validator that passes."""

        def is_valid(s: str) -> bool:
            return len(s) > 0

        options = FlextUtilities.StringParser.ParseOptions(
            strip=True, remove_empty=True, validator=is_valid
        )
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a,b,c", ",", options=options)
        assert result.is_success

    def test_parse_delimited_with_validator_failure(self) -> None:
        """Test parsing with validator that fails."""

        def is_valid(s: str) -> bool:
            return len(s) > 1  # Fail for single char

        options = FlextUtilities.StringParser.ParseOptions(
            strip=True, remove_empty=True, validator=is_valid
        )
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a,b", ",", options=options)
        assert result.is_failure
        assert "Invalid component" in result.error

    def test_parse_delimited_legacy_params(self) -> None:
        """Test parsing with legacy parameters (no options)."""
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a, b, c", ",", strip=True, remove_empty=True)
        assert result.is_success
        assert result.unwrap() == ["a", "b", "c"]

    def test_parse_delimited_legacy_validator(self) -> None:
        """Test parsing with legacy validator parameter."""

        def is_valid(s: str) -> bool:
            return "x" not in s

        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited("a,b,c", ",", validator=is_valid)
        assert result.is_success

    def test_parse_delimited_exception_handling(self) -> None:
        """Test parsing exception handling."""

        # Create object that will fail on split
        class BadString:
            def split(self, delimiter: str) -> list[str]:
                msg = "Split failed"
                raise RuntimeError(msg)

        bad = BadString()
        parser = FlextUtilities.StringParser()
        result = parser.parse_delimited(bad, ",")
        assert result.is_failure
        assert "Failed to parse" in result.error


# ============================================================================
# Test Split With Escape
# ============================================================================


class TestFlextUtilitiesStringParserSplitWithEscape:
    """Test split_on_char_with_escape method."""

    def test_split_with_escape_basic(self) -> None:
        """Test basic split with escape."""
        parser = FlextUtilities.StringParser()
        result = parser.split_on_char_with_escape("a,b,c", ",")
        assert result.is_success
        assert result.unwrap() == ["a", "b", "c"]

    def test_split_with_escape_escaped_delimiter(self) -> None:
        """Test split with escaped delimiter."""
        parser = FlextUtilities.StringParser()
        result = parser.split_on_char_with_escape("a\\,b,c", ",")
        assert result.is_success
        assert result.unwrap() == ["a\\,b", "c"]

    def test_split_with_escape_empty_string(self) -> None:
        """Test split with empty string."""
        parser = FlextUtilities.StringParser()
        result = parser.split_on_char_with_escape("", ",")
        assert result.is_success
        assert result.unwrap() == []

    def test_split_with_escape_custom_escape_char(self) -> None:
        """Test split with custom escape character."""
        parser = FlextUtilities.StringParser()
        result = parser.split_on_char_with_escape("a#b,c", ",", escape_char="#")
        assert result.is_success
        assert result.unwrap() == ["a#b", "c"]

    def test_split_with_escape_at_end(self) -> None:
        """Test split with escape at end of string."""
        parser = FlextUtilities.StringParser()
        result = parser.split_on_char_with_escape("a,b\\", ",")
        assert result.is_success
        # Escape at end is treated as regular char
        assert "b\\" in result.unwrap()[-1]

    def test_split_with_escape_exception_handling(self) -> None:
        """Test split exception handling."""

        # Create object that will fail on indexing
        class BadString:
            def __len__(self) -> int:
                return 5

            def __getitem__(self, key: int) -> str:
                msg = "Index failed"
                raise KeyError(msg)

        bad = BadString()
        parser = FlextUtilities.StringParser()
        result = parser.split_on_char_with_escape(
            bad,
            ",",
        )
        assert result.is_failure
        assert "Failed to split" in result.error


# ============================================================================
# Test Normalize Whitespace
# ============================================================================


class TestFlextUtilitiesStringParserNormalizeWhitespace:
    """Test normalize_whitespace method."""

    def test_normalize_whitespace_basic(self) -> None:
        """Test basic whitespace normalization."""
        parser = FlextUtilities.StringParser()
        result = parser.normalize_whitespace("  hello   world  ")
        assert result.is_success
        assert result.unwrap() == "hello world"

    def test_normalize_whitespace_empty_string(self) -> None:
        """Test normalization of empty string."""
        parser = FlextUtilities.StringParser()
        result = parser.normalize_whitespace("")
        assert result.is_success
        assert result.unwrap() == ""

    def test_normalize_whitespace_custom_pattern(self) -> None:
        """Test normalization with custom pattern."""
        parser = FlextUtilities.StringParser()
        result = parser.normalize_whitespace(
            "hello---world", pattern=r"-+", replacement="-"
        )
        assert result.is_success
        assert result.unwrap() == "hello-world"

    def test_normalize_whitespace_custom_replacement(self) -> None:
        """Test normalization with custom replacement."""
        parser = FlextUtilities.StringParser()
        result = parser.normalize_whitespace("hello   world", replacement="_")
        assert result.is_success
        assert "_" in result.unwrap()

    def test_normalize_whitespace_exception_handling(self) -> None:
        """Test normalization exception handling."""

        # Create object that will fail on re.sub
        class BadString:
            def __str__(self) -> str:
                msg = "String conversion failed"
                raise RuntimeError(msg)

        bad = BadString()
        parser = FlextUtilities.StringParser()
        result = parser.normalize_whitespace(bad)
        assert result.is_failure
        assert "Failed to normalize" in result.error


# ============================================================================
# Test Regex Pipeline
# ============================================================================


class TestFlextUtilitiesStringParserRegexPipeline:
    """Test apply_regex_pipeline method."""

    def test_apply_regex_pipeline_basic(self) -> None:
        """Test basic regex pipeline."""
        patterns = [
            (r"\s+", " "),
            (r"=", "="),
        ]
        parser = FlextUtilities.StringParser()
        result = parser.apply_regex_pipeline("hello   world", patterns)
        assert result.is_success
        assert result.unwrap() == "hello world"

    def test_apply_regex_pipeline_empty_string(self) -> None:
        """Test pipeline with empty string."""
        patterns = [(r"\s+", " ")]
        parser = FlextUtilities.StringParser()
        result = parser.apply_regex_pipeline("", patterns)
        assert result.is_success
        assert result.unwrap() == ""

    def test_apply_regex_pipeline_multiple_patterns(self) -> None:
        """Test pipeline with multiple patterns."""
        patterns = [
            (r"\s+=", "="),
            (r",\s+", ","),
            (r"\s+", " "),
        ]
        parser = FlextUtilities.StringParser()
        result = parser.apply_regex_pipeline("cn = REDACTED_LDAP_BIND_PASSWORD , ou = users", patterns)
        assert result.is_success
        assert "=" in result.unwrap()
        assert "," in result.unwrap()

    def test_apply_regex_pipeline_exception_handling(self) -> None:
        """Test pipeline exception handling."""
        # Pass invalid patterns to trigger exception
        patterns: list[tuple[str | None, str]] = [(None, "replacement")]
        parser = FlextUtilities.StringParser()
        result = parser.apply_regex_pipeline("test", patterns)
        assert result.is_failure
        assert "Failed to apply" in result.error


# ============================================================================
# Test Get Object Key
# ============================================================================


class TestFlextUtilitiesStringParserGetObjectKey:
    """Test get_object_key method."""

    def test_get_object_key_type(self) -> None:
        """Test getting key from type."""
        parser = FlextUtilities.StringParser()
        key = parser.get_object_key(int)
        assert key == "int"

    def test_get_object_key_class(self) -> None:
        """Test getting key from class."""

        class TestClass:
            pass

        parser = FlextUtilities.StringParser()
        key = parser.get_object_key(TestClass)
        assert key == "TestClass"

    def test_get_object_key_function(self) -> None:
        """Test getting key from function."""

        def test_function() -> None:
            pass

        parser = FlextUtilities.StringParser()
        key = parser.get_object_key(test_function)
        assert key == "test_function"

    def test_get_object_key_instance(self) -> None:
        """Test getting key from instance."""
        obj = object()
        parser = FlextUtilities.StringParser()
        key = parser.get_object_key(obj)
        assert isinstance(key, str)
        assert "object" in key

    def test_get_object_key_string(self) -> None:
        """Test getting key from string."""
        parser = FlextUtilities.StringParser()
        key = parser.get_object_key("test")
        assert key == "test"

    def test_get_object_key_no_str_method(self) -> None:
        """Test getting key from object without __str__."""

        class NoStr:
            def __str__(self) -> str:
                msg = "Cannot convert to string"
                raise TypeError(msg)

        parser = FlextUtilities.StringParser()
        key = parser.get_object_key(NoStr())
        assert isinstance(key, str)
        assert "NoStr" in key
