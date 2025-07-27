"""Tests for _constants_base module."""

import re

import pytest

from flext_core._constants_base import (
    _BaseConstants,
    _FlextEnvironment,
    _FlextFieldType,
    _FlextLogLevel,
)


class TestFlextEnvironment:
    """Test _FlextEnvironment enum."""

    def test_environment_values(self) -> None:
        """Test all environment values."""
        assert _FlextEnvironment.DEVELOPMENT == "development"
        assert _FlextEnvironment.TESTING == "testing"
        assert _FlextEnvironment.STAGING == "staging"
        assert _FlextEnvironment.PRODUCTION == "production"

    def test_environment_enumeration(self) -> None:
        """Test environment enum iteration."""
        environments = list(_FlextEnvironment)
        assert len(environments) == 4
        assert _FlextEnvironment.DEVELOPMENT in environments
        assert _FlextEnvironment.TESTING in environments
        assert _FlextEnvironment.STAGING in environments
        assert _FlextEnvironment.PRODUCTION in environments


class TestFlextLogLevel:
    """Test _FlextLogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test all log level values."""
        assert _FlextLogLevel.CRITICAL == "CRITICAL"
        assert _FlextLogLevel.ERROR == "ERROR"
        assert _FlextLogLevel.WARNING == "WARNING"
        assert _FlextLogLevel.INFO == "INFO"
        assert _FlextLogLevel.DEBUG == "DEBUG"
        assert _FlextLogLevel.TRACE == "TRACE"

    def test_log_level_enumeration(self) -> None:
        """Test log level enum iteration."""
        log_levels = list(_FlextLogLevel)
        assert len(log_levels) == 6
        assert _FlextLogLevel.CRITICAL in log_levels
        assert _FlextLogLevel.ERROR in log_levels
        assert _FlextLogLevel.WARNING in log_levels
        assert _FlextLogLevel.INFO in log_levels
        assert _FlextLogLevel.DEBUG in log_levels
        assert _FlextLogLevel.TRACE in log_levels


class TestFlextFieldType:
    """Test _FlextFieldType enum."""

    def test_field_type_values(self) -> None:
        """Test all field type values."""
        assert _FlextFieldType.STRING == "string"
        assert _FlextFieldType.INTEGER == "integer"
        assert _FlextFieldType.FLOAT == "float"
        assert _FlextFieldType.BOOLEAN == "boolean"
        assert _FlextFieldType.DATE == "date"
        assert _FlextFieldType.DATETIME == "datetime"
        assert _FlextFieldType.UUID == "uuid"
        assert _FlextFieldType.EMAIL == "email"
        assert _FlextFieldType.URL == "url"
        assert _FlextFieldType.JSON == "json"
        assert _FlextFieldType.BINARY == "binary"
        assert _FlextFieldType.ENUM == "enum"
        assert _FlextFieldType.LIST == "list"
        assert _FlextFieldType.DICT == "dict"
        assert _FlextFieldType.CUSTOM == "custom"

    def test_field_type_enumeration(self) -> None:
        """Test field type enum iteration."""
        field_types = list(_FlextFieldType)
        assert len(field_types) == 15
        assert _FlextFieldType.STRING in field_types
        assert _FlextFieldType.INTEGER in field_types
        assert _FlextFieldType.FLOAT in field_types
        assert _FlextFieldType.BOOLEAN in field_types
        assert _FlextFieldType.DATE in field_types
        assert _FlextFieldType.DATETIME in field_types
        assert _FlextFieldType.UUID in field_types
        assert _FlextFieldType.EMAIL in field_types
        assert _FlextFieldType.URL in field_types
        assert _FlextFieldType.JSON in field_types
        assert _FlextFieldType.BINARY in field_types
        assert _FlextFieldType.ENUM in field_types
        assert _FlextFieldType.LIST in field_types
        assert _FlextFieldType.DICT in field_types
        assert _FlextFieldType.CUSTOM in field_types


class TestBaseConstants:
    """Test _BaseConstants class."""

    def test_project_metadata(self) -> None:
        """Test project metadata constants."""
        assert isinstance(_BaseConstants.VERSION, str)
        assert _BaseConstants.NAME == "flext-core"
        assert len(_BaseConstants.VERSION) > 0

    def test_system_defaults(self) -> None:
        """Test system default constants."""
        assert isinstance(_BaseConstants.DEFAULT_TIMEOUT, int)
        assert isinstance(_BaseConstants.DEFAULT_RETRIES, int)
        assert isinstance(_BaseConstants.DEFAULT_PAGE_SIZE, int)
        assert isinstance(_BaseConstants.DEFAULT_LOG_LEVEL, str)

        # Validate reasonable ranges
        assert _BaseConstants.DEFAULT_TIMEOUT > 0
        assert _BaseConstants.DEFAULT_RETRIES >= 0
        assert _BaseConstants.DEFAULT_PAGE_SIZE > 0
        assert _BaseConstants.DEFAULT_LOG_LEVEL in _FlextLogLevel

    def test_validation_patterns(self) -> None:
        """Test validation pattern constants."""
        # Test UUID pattern
        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        invalid_uuid = "invalid-uuid"
        assert re.match(_BaseConstants.UUID_PATTERN, valid_uuid) is not None
        assert re.match(_BaseConstants.UUID_PATTERN, invalid_uuid) is None

        # Test email pattern
        valid_email = "test@example.com"
        invalid_email = "invalid-email"
        assert re.match(_BaseConstants.EMAIL_PATTERN, valid_email) is not None
        assert re.match(_BaseConstants.EMAIL_PATTERN, invalid_email) is None

        # Test service name pattern
        valid_service = "my-service"
        invalid_service = "my service"
        assert re.match(_BaseConstants.SERVICE_NAME_PATTERN, valid_service) is not None
        assert re.match(_BaseConstants.SERVICE_NAME_PATTERN, invalid_service) is None

        # Test identifier pattern
        valid_identifier = "my_identifier"
        invalid_identifier = "123identifier"
        assert re.match(_BaseConstants.IDENTIFIER_PATTERN, valid_identifier) is not None
        assert re.match(_BaseConstants.IDENTIFIER_PATTERN, invalid_identifier) is None

        # Test URL pattern
        valid_url = "https://example.com/path?param=value"
        invalid_url = "not-a-url"
        assert re.match(_BaseConstants.URL_PATTERN, valid_url) is not None
        assert re.match(_BaseConstants.URL_PATTERN, invalid_url) is None

    def test_pattern_consistency(self) -> None:
        """Test that patterns are consistent and valid regex."""
        patterns = [
            _BaseConstants.UUID_PATTERN,
            _BaseConstants.EMAIL_PATTERN,
            _BaseConstants.SERVICE_NAME_PATTERN,
            _BaseConstants.IDENTIFIER_PATTERN,
            _BaseConstants.URL_PATTERN,
        ]

        for pattern in patterns:
            # Test that pattern is a valid regex
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Invalid regex pattern: {pattern}")

            # Test that pattern is a string
            assert isinstance(pattern, str)
            assert len(pattern) > 0

        def test_constants_immutability(self: object) -> None:
            """Test that constants are immutable (Final)."""
            # Note: Final is a type checker annotation, not runtime enforcement
            # In Python, these can be modified at runtime, but it's bad practice
            # We test that the constants have the expected values and types

            # Test that constants have expected types and values
            assert isinstance(_BaseConstants.VERSION, str)
            assert isinstance(_BaseConstants.NAME, str)
            assert isinstance(_BaseConstants.DEFAULT_TIMEOUT, int)

            # Test that constants are not None
            assert _BaseConstants.VERSION is not None
            assert _BaseConstants.NAME is not None
            assert _BaseConstants.DEFAULT_TIMEOUT is not None
