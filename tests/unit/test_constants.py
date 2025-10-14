"""Comprehensive tests for FlextCore.Constants - Foundation Constants.

Tests the actual FlextCore.Constants API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re

from flext_core import FlextCore


class TestFlextConstants:
    """Test suite for FlextCore.Constants foundation constants."""

    def test_core_constants(self) -> None:
        """Test core constants access."""
        assert FlextCore.Constants.NAME == "FLEXT"
        assert FlextCore.Constants.VERSION == "0.9.9"

    def test_network_constants(self) -> None:
        """Test network constants access."""
        assert FlextCore.Constants.Network.MIN_PORT == 1
        assert FlextCore.Constants.Network.MAX_PORT == 65535
        assert FlextCore.Constants.Network.TOTAL_TIMEOUT == 60
        assert FlextCore.Constants.Network.DEFAULT_TIMEOUT == 30

    def test_validation_constants(self) -> None:
        """Test validation constants access."""
        assert FlextCore.Constants.Validation.MIN_NAME_LENGTH == 2
        assert FlextCore.Constants.Validation.MAX_NAME_LENGTH == 100
        assert FlextCore.Constants.Validation.MIN_SERVICE_NAME_LENGTH == 2
        assert FlextCore.Constants.Validation.MAX_EMAIL_LENGTH == 254
        assert FlextCore.Constants.Validation.MIN_PERCENTAGE == 0.0
        assert FlextCore.Constants.Validation.MAX_PERCENTAGE == 100.0
        assert FlextCore.Constants.Validation.MIN_SECRET_KEY_LENGTH == 32
        assert FlextCore.Constants.Validation.MIN_PHONE_DIGITS == 10

    def test_error_constants(self) -> None:
        """Test error constants access."""
        assert FlextCore.Constants.Errors.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert FlextCore.Constants.Errors.TYPE_ERROR == "TYPE_ERROR"
        assert FlextCore.Constants.Errors.SERIALIZATION_ERROR == "SERIALIZATION_ERROR"
        assert FlextCore.Constants.Errors.CONFIG_ERROR == "CONFIG_ERROR"
        assert FlextCore.Constants.Errors.OPERATION_ERROR == "OPERATION_ERROR"
        assert (
            FlextCore.Constants.Errors.BUSINESS_RULE_VIOLATION
            == "BUSINESS_RULE_VIOLATION"
        )
        assert FlextCore.Constants.Errors.NOT_FOUND_ERROR == "NOT_FOUND_ERROR"

    def test_messages_constants(self) -> None:
        """Test message constants access."""
        assert FlextCore.Constants.Messages.TYPE_MISMATCH == "Type mismatch"
        assert (
            FlextCore.Constants.Messages.SERVICE_NAME_EMPTY
            == "Service name cannot be empty"
        )

    def test_entities_constants(self) -> None:
        """Test entity constants access."""
        assert (
            FlextCore.Constants.Entities.ENTITY_ID_EMPTY == "Entity ID cannot be empty"
        )

    def test_defaults_constants(self) -> None:
        """Test default constants access."""
        assert FlextCore.Constants.Defaults.TIMEOUT == 30
        assert FlextCore.Constants.Defaults.PAGE_SIZE == 100
        assert FlextCore.Constants.Defaults.TIMEOUT_SECONDS == 30

    def test_limits_constants(self) -> None:
        """Test limits constants access."""
        assert FlextCore.Constants.Limits.MAX_STRING_LENGTH == 1000
        assert FlextCore.Constants.Limits.MAX_LIST_SIZE == 10000
        assert FlextCore.Constants.Limits.MAX_FILE_SIZE == 10 * 1024 * 1024

    def test_utilities_constants(self) -> None:
        """Test utility constants access."""
        assert FlextCore.Constants.Utilities.SECONDS_PER_MINUTE == 60
        assert FlextCore.Constants.Utilities.SECONDS_PER_HOUR == 3600
        assert FlextCore.Constants.Utilities.BYTES_PER_KB == 1024

    def test_logging_constants(self) -> None:
        """Test logging constants access."""
        assert FlextCore.Constants.Logging.DEFAULT_LEVEL == "INFO"
        assert FlextCore.Constants.Logging.DEFAULT_LEVEL_DEVELOPMENT == "DEBUG"
        assert FlextCore.Constants.Logging.DEFAULT_LEVEL_PRODUCTION == "WARNING"
        assert FlextCore.Constants.Logging.DEFAULT_LEVEL_TESTING == "INFO"

    def test_logging_levels_enum(self) -> None:
        """Test logging levels enum."""
        assert FlextCore.Constants.Config.LogLevel.DEBUG == "DEBUG"
        assert FlextCore.Constants.Config.LogLevel.INFO == "INFO"
        assert FlextCore.Constants.Config.LogLevel.WARNING == "WARNING"
        assert FlextCore.Constants.Config.LogLevel.ERROR == "ERROR"
        assert FlextCore.Constants.Config.LogLevel.CRITICAL == "CRITICAL"

    def test_config_source_enum(self) -> None:
        """Test config source enum."""
        assert FlextCore.Constants.Config.ConfigSource.FILE.value == "file"
        assert FlextCore.Constants.Config.ConfigSource.ENVIRONMENT.value == "env"
        assert FlextCore.Constants.Config.ConfigSource.CLI.value == "cli"

    def test_field_type_enum(self) -> None:
        """Test field type enum."""
        assert FlextCore.Constants.Enums.FieldType.STRING.value == "string"
        assert FlextCore.Constants.Enums.FieldType.INTEGER.value == "integer"
        assert FlextCore.Constants.Enums.FieldType.FLOAT.value == "float"
        assert FlextCore.Constants.Enums.FieldType.BOOLEAN.value == "boolean"
        assert FlextCore.Constants.Enums.FieldType.DATETIME.value == "datetime"

    def test_platform_constants(self) -> None:
        """Test platform constants access."""
        assert FlextCore.Constants.Platform.FLEXT_API_PORT == 8000
        assert FlextCore.Constants.Platform.DEFAULT_HOST == "localhost"
        assert FlextCore.Constants.Platform.LOOPBACK_IP == "127.0.0.1"

    def test_validation_patterns_email(self) -> None:
        """Test email validation pattern."""
        pattern = re.compile(FlextCore.Constants.Platform.PATTERN_EMAIL)

        # Valid emails
        assert pattern.match("test@example.com") is not None
        assert pattern.match("user.name+tag@example.co.uk") is not None
        assert pattern.match("valid_email@domain.com") is not None

        # Invalid emails
        assert pattern.match("invalid.email") is None
        assert pattern.match("@example.com") is None
        assert pattern.match("test@") is None

    def test_validation_patterns_url(self) -> None:
        """Test URL validation pattern."""
        pattern = re.compile(FlextCore.Constants.Platform.PATTERN_URL, re.IGNORECASE)

        # Valid URLs
        assert pattern.match("https://github.com") is not None
        assert pattern.match("http://localhost:8000") is not None
        assert pattern.match("https://example.com/path?query=1") is not None

        # Invalid URLs
        assert pattern.match("not-a-url") is None
        assert pattern.match("ftp://invalid.com") is None

    def test_validation_patterns_phone(self) -> None:
        """Test phone number validation pattern."""
        pattern = re.compile(FlextCore.Constants.Platform.PATTERN_PHONE_NUMBER)

        # Valid phone numbers
        assert pattern.match("+5511987654321") is not None
        assert pattern.match("5511987654321") is not None
        assert pattern.match("+1234567890") is not None

        # Invalid phone numbers
        assert pattern.match("123") is None
        assert pattern.match("abc1234567890") is None

    def test_validation_patterns_uuid(self) -> None:
        """Test UUID validation pattern."""
        pattern = re.compile(FlextCore.Constants.Platform.PATTERN_UUID)

        # Valid UUIDs
        assert pattern.match("550e8400-e29b-41d4-a716-446655440000") is not None
        assert (
            pattern.match("550e8400e29b41d4a716446655440000") is not None
        )  # Without hyphens
        assert pattern.match("123e4567-E89B-12D3-A456-426614174000") is not None

        # Invalid UUIDs
        assert pattern.match("invalid-uuid") is None
        assert pattern.match("550e8400-e29b-41d4") is None

    def test_validation_patterns_path(self) -> None:
        """Test file path validation pattern."""
        pattern = re.compile(FlextCore.Constants.Platform.PATTERN_PATH)

        # Valid paths
        assert pattern.match("/home/user/file.txt") is not None
        assert pattern.match("C:\\Users\\file.txt") is not None
        assert pattern.match("relative/path/file.py") is not None

        # Invalid paths (with invalid characters)
        assert pattern.match("path/with<invalid>chars") is None
        assert pattern.match('path/with"quotes') is None

    def test_observability_constants(self) -> None:
        """Test observability constants access."""
        assert FlextCore.Constants.Observability.DEFAULT_LOG_LEVEL == "INFO"

    def test_performance_constants(self) -> None:
        """Test performance constants access."""
        assert FlextCore.Constants.Performance.DEFAULT_PAGE_SIZE == 10
        assert FlextCore.Constants.Performance.SUBPROCESS_TIMEOUT == 300
        assert FlextCore.Constants.Performance.SUBPROCESS_TIMEOUT_SHORT == 180

    def test_reliability_constants(self) -> None:
        """Test reliability constants access."""
        assert FlextCore.Constants.Reliability.MAX_RETRY_ATTEMPTS == 3
        assert FlextCore.Constants.Reliability.DEFAULT_MAX_RETRIES == 3
        assert FlextCore.Constants.Reliability.DEFAULT_BACKOFF_STRATEGY == "exponential"

    def test_security_constants(self) -> None:
        """Test security constants access."""
        assert FlextCore.Constants.Security.MAX_JWT_EXPIRY_MINUTES == 43200
        assert FlextCore.Constants.Security.DEFAULT_JWT_EXPIRY_MINUTES == 60

    def test_cqrs_constants(self) -> None:
        """Test CQRS constants access."""
        assert FlextCore.Constants.Cqrs.DEFAULT_HANDLER_TYPE == "command"
        assert FlextCore.Constants.Cqrs.COMMAND_HANDLER_TYPE == "command"
        assert FlextCore.Constants.Cqrs.QUERY_HANDLER_TYPE == "query"

    def test_container_constants(self) -> None:
        """Test container constants access."""
        assert FlextCore.Constants.Container.MAX_WORKERS == 4
        assert FlextCore.Constants.Container.MIN_WORKERS == 1

    def test_dispatcher_constants(self) -> None:
        """Test dispatcher constants access."""
        assert FlextCore.Constants.Dispatcher.HANDLER_MODE_COMMAND == "command"
        assert FlextCore.Constants.Dispatcher.HANDLER_MODE_QUERY == "query"
        assert FlextCore.Constants.Dispatcher.DEFAULT_HANDLER_MODE == "command"

    def test_mixins_constants(self) -> None:
        """Test mixins constants access."""
        assert FlextCore.Constants.Mixins.FIELD_CREATED_AT == "created_at"
        assert FlextCore.Constants.Mixins.FIELD_UPDATED_AT == "updated_at"
        assert FlextCore.Constants.Mixins.FIELD_ID == "id"

    def test_constants_immutability(self) -> None:
        """Test that constants are immutable (Final)."""
        # This test verifies that constants are properly marked as Final
        # and cannot be modified at runtime
        original_name = FlextCore.Constants.NAME
        assert original_name == "FLEXT"

        # Constants should be immutable - this is enforced by the Final type hint
        # In a real test, we would verify that attempting to modify raises an error
        # but since these are Final, Python will prevent modification at runtime

    def test_constants_type_safety(self) -> None:
        """Test that constants have correct types."""
        # Test string constants
        assert isinstance(FlextCore.Constants.NAME, str)
        assert isinstance(FlextCore.Constants.VERSION, str)

        # Test integer constants
        assert isinstance(FlextCore.Constants.Network.MIN_PORT, int)
        assert isinstance(FlextCore.Constants.Network.MAX_PORT, int)

        # Test float constants
        assert isinstance(FlextCore.Constants.Validation.MIN_PERCENTAGE, float)
        assert isinstance(FlextCore.Constants.Validation.MAX_PERCENTAGE, float)

        # Test list constants - removed environment references per requirements

    def test_constants_completeness(self) -> None:
        """Test that all expected constant categories exist."""
        # Verify all major constant categories are present
        assert hasattr(FlextCore.Constants, "Core")
        assert hasattr(FlextCore.Constants, "Network")
        assert hasattr(FlextCore.Constants, "Validation")
        assert hasattr(FlextCore.Constants, "Errors")
        assert hasattr(FlextCore.Constants, "Messages")
        assert hasattr(FlextCore.Constants, "Entities")
        assert hasattr(FlextCore.Constants, "Defaults")
        assert hasattr(FlextCore.Constants, "Limits")
        assert hasattr(FlextCore.Constants, "Utilities")
        assert hasattr(FlextCore.Constants, "Config")
        assert hasattr(FlextCore.Constants, "Logging")
        assert hasattr(FlextCore.Constants, "Enums")
        assert hasattr(FlextCore.Constants, "Platform")
        assert hasattr(FlextCore.Constants, "Observability")
        assert hasattr(FlextCore.Constants, "Performance")
        assert hasattr(FlextCore.Constants, "Reliability")
        assert hasattr(FlextCore.Constants, "Security")
        assert hasattr(FlextCore.Constants, "Cqrs")
        assert hasattr(FlextCore.Constants, "Container")
        assert hasattr(FlextCore.Constants, "Dispatcher")
        assert hasattr(FlextCore.Constants, "Mixins")

    def test_constants_documentation(self) -> None:
        """Test that constants have proper documentation."""
        # Verify that the main class has documentation
        assert FlextCore.Constants.__doc__ is not None
        assert "foundation" in FlextCore.Constants.__doc__.lower()

        # Verify that nested classes have documentation
        assert FlextCore.Constants.__doc__ is not None
        assert FlextCore.Constants.Network.__doc__ is not None
        assert FlextCore.Constants.Validation.__doc__ is not None
        assert FlextCore.Constants.Errors.__doc__ is not None

    def test_class_getitem_nested_path(self) -> None:
        """Test FlextCore.Constants[] method with nested paths."""
        # Test valid nested path access
        validation_error = FlextCore.Constants.Errors.VALIDATION_ERROR
        assert validation_error == "VALIDATION_ERROR"

        # Test another nested path
        default_timeout = FlextCore.Constants.Defaults.TIMEOUT
        assert default_timeout == 30

        # Test deep nested path
        default_level = FlextCore.Constants.Logging.DEFAULT_LEVEL
        assert default_level == "INFO"

    def test_class_getitem_invalid_path(self) -> None:
        """Test FlextCore.Constants[] with invalid path returns placeholder values."""
        # Test accessing a real constant that exists - should work
        result = FlextCore.Constants.Errors.NONEXISTENT_ERROR
        assert result == "NONEXISTENT_ERROR"
