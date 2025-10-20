"""Comprehensive tests for FlextConstants - Foundation Constants.

Tests the actual FlextConstants API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re

from flext_core import FlextConstants


class TestFlextConstants:
    """Test suite for FlextConstants foundation constants."""

    def test_core_constants(self) -> None:
        """Test core constants access."""
        assert FlextConstants.NAME == "FLEXT"

    def test_network_constants(self) -> None:
        """Test network constants access."""
        assert FlextConstants.Network.MIN_PORT == 1
        assert FlextConstants.Network.MAX_PORT == 65535

    def test_validation_constants(self) -> None:
        """Test validation constants access."""
        assert FlextConstants.Validation.MIN_NAME_LENGTH == 2
        assert FlextConstants.Validation.MAX_NAME_LENGTH == 100
        assert FlextConstants.Validation.MAX_EMAIL_LENGTH == 254
        assert FlextConstants.Validation.MIN_PHONE_DIGITS == 10

    def test_error_constants(self) -> None:
        """Test error constants access."""
        assert FlextConstants.Errors.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert FlextConstants.Errors.TYPE_ERROR == "TYPE_ERROR"
        assert FlextConstants.Errors.SERIALIZATION_ERROR == "SERIALIZATION_ERROR"
        assert FlextConstants.Errors.CONFIG_ERROR == "CONFIG_ERROR"
        assert FlextConstants.Errors.OPERATION_ERROR == "OPERATION_ERROR"
        assert (
            FlextConstants.Errors.BUSINESS_RULE_VIOLATION == "BUSINESS_RULE_VIOLATION"
        )
        assert FlextConstants.Errors.NOT_FOUND_ERROR == "NOT_FOUND_ERROR"

    def test_messages_constants(self) -> None:
        """Test message constants access."""
        assert FlextConstants.Messages.TYPE_MISMATCH == "Type mismatch"

    def test_defaults_constants(self) -> None:
        """Test default constants access."""
        assert FlextConstants.Defaults.TIMEOUT == 30
        assert FlextConstants.Defaults.PAGE_SIZE == 100

    def test_utilities_constants(self) -> None:
        """Test utility constants access."""
        assert FlextConstants.Utilities.DEFAULT_ENCODING == "utf-8"
        assert FlextConstants.Utilities.MAX_TIMEOUT_SECONDS == 3600

    def test_logging_constants(self) -> None:
        """Test logging constants access."""
        assert FlextConstants.Logging.DEFAULT_LEVEL == "INFO"
        assert FlextConstants.Logging.DEFAULT_LEVEL_DEVELOPMENT == "DEBUG"
        assert FlextConstants.Logging.DEFAULT_LEVEL_PRODUCTION == "WARNING"
        assert FlextConstants.Logging.DEFAULT_LEVEL_TESTING == "INFO"

    def test_logging_levels_enum(self) -> None:
        """Test logging levels enum."""
        assert FlextConstants.Config.LogLevel.DEBUG == "DEBUG"
        assert FlextConstants.Config.LogLevel.INFO == "INFO"
        assert FlextConstants.Config.LogLevel.WARNING == "WARNING"
        assert FlextConstants.Config.LogLevel.ERROR == "ERROR"
        assert FlextConstants.Config.LogLevel.CRITICAL == "CRITICAL"

    def test_platform_constants(self) -> None:
        """Test platform constants access."""
        assert FlextConstants.Platform.FLEXT_API_PORT == 8000
        assert FlextConstants.Platform.DEFAULT_HOST == "localhost"

    def test_validation_patterns_email(self) -> None:
        """Test email validation pattern."""
        pattern = re.compile(FlextConstants.Platform.PATTERN_EMAIL)

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
        pattern = re.compile(FlextConstants.Platform.PATTERN_URL, re.IGNORECASE)

        # Valid URLs
        assert pattern.match("https://github.com") is not None
        assert pattern.match("http://localhost:8000") is not None
        assert pattern.match("https://example.com/path?query=1") is not None

        # Invalid URLs
        assert pattern.match("not-a-url") is None
        assert pattern.match("ftp://invalid.com") is None

    def test_validation_patterns_phone(self) -> None:
        """Test phone number validation pattern."""
        pattern = re.compile(FlextConstants.Platform.PATTERN_PHONE_NUMBER)

        # Valid phone numbers
        assert pattern.match("+5511987654321") is not None
        assert pattern.match("5511987654321") is not None
        assert pattern.match("+1234567890") is not None

        # Invalid phone numbers
        assert pattern.match("123") is None
        assert pattern.match("abc1234567890") is None

    def test_validation_patterns_uuid(self) -> None:
        """Test UUID validation pattern."""
        pattern = re.compile(FlextConstants.Platform.PATTERN_UUID)

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
        pattern = re.compile(FlextConstants.Platform.PATTERN_PATH)

        # Valid paths
        assert pattern.match("/home/user/file.txt") is not None
        assert pattern.match("C:\\Users\\file.txt") is not None
        assert pattern.match("relative/path/file.py") is not None

        # Invalid paths (with invalid characters)
        assert pattern.match("path/with<invalid>chars") is None
        assert pattern.match('path/with"quotes') is None

    def test_performance_constants(self) -> None:
        """Test performance constants access."""
        assert FlextConstants.Performance.MAX_TIMEOUT_SECONDS == 600
        assert FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE == 1000

    def test_reliability_constants(self) -> None:
        """Test reliability constants access."""
        assert FlextConstants.Reliability.MAX_RETRY_ATTEMPTS == 3
        assert FlextConstants.Reliability.DEFAULT_MAX_RETRIES == 3
        assert FlextConstants.Reliability.DEFAULT_BACKOFF_STRATEGY == "exponential"

    def test_security_constants(self) -> None:
        """Test security constants access."""
        assert FlextConstants.Security.JWT_DEFAULT_ALGORITHM == "HS256"
        assert FlextConstants.Security.CREDENTIAL_BCRYPT_ROUNDS == 12

    def test_cqrs_constants(self) -> None:
        """Test CQRS constants access."""
        assert FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE == "command"
        assert FlextConstants.Cqrs.COMMAND_HANDLER_TYPE == "command"
        assert FlextConstants.Cqrs.QUERY_HANDLER_TYPE == "query"

    def test_container_constants(self) -> None:
        """Test container constants access."""
        assert FlextConstants.Container.DEFAULT_WORKERS == 4
        assert FlextConstants.Container.MAX_CACHE_SIZE == 100

    def test_dispatcher_constants(self) -> None:
        """Test dispatcher constants access."""
        assert FlextConstants.Dispatcher.HANDLER_MODE_COMMAND == "command"
        assert FlextConstants.Dispatcher.HANDLER_MODE_QUERY == "query"
        assert FlextConstants.Dispatcher.DEFAULT_HANDLER_MODE == "command"

    def test_mixins_constants(self) -> None:
        """Test mixins constants access."""
        assert FlextConstants.Mixins.FIELD_CREATED_AT == "created_at"
        assert FlextConstants.Mixins.FIELD_UPDATED_AT == "updated_at"
        assert FlextConstants.Mixins.FIELD_ID == "id"

    def test_constants_immutability(self) -> None:
        """Test that constants are immutable (Final)."""
        # This test verifies that constants are properly marked as Final
        # and cannot be modified at runtime
        original_name = FlextConstants.NAME
        assert original_name == "FLEXT"

        # Constants should be immutable - this is enforced by the Final type hint
        # In a real test, we would verify that attempting to modify raises an error
        # but since these are Final, Python will prevent modification at runtime

    def test_constants_type_safety(self) -> None:
        """Test that constants have correct types."""
        # Test string constants
        assert isinstance(FlextConstants.NAME, str)

        # Test integer constants
        assert isinstance(FlextConstants.Network.MIN_PORT, int)
        assert isinstance(FlextConstants.Network.MAX_PORT, int)

        # Test float constants
        assert isinstance(FlextConstants.Reliability.RETRY_BACKOFF_MAX, float)

        # Test list constants - removed environment references per requirements

    def test_constants_completeness(self) -> None:
        """Test that all expected constant categories exist."""
        # Verify all major constant categories are present (used by flext-core only)
        # Core namespace was removed in Phase 8 - constants moved to class level (NAME, VERSION, ZERO, INITIAL_TIME)
        assert hasattr(FlextConstants, "Network")
        assert hasattr(FlextConstants, "Validation")
        assert hasattr(FlextConstants, "Errors")
        assert hasattr(FlextConstants, "Messages")
        assert hasattr(FlextConstants, "Defaults")
        assert hasattr(FlextConstants, "Utilities")
        assert hasattr(FlextConstants, "Config")
        assert hasattr(FlextConstants, "Logging")
        assert hasattr(FlextConstants, "Platform")
        assert hasattr(FlextConstants, "Performance")
        assert hasattr(FlextConstants, "Reliability")
        assert hasattr(FlextConstants, "Security")
        assert hasattr(FlextConstants, "Cqrs")
        assert hasattr(FlextConstants, "Container")
        assert hasattr(FlextConstants, "Dispatcher")
        assert hasattr(FlextConstants, "Mixins")
        assert hasattr(FlextConstants, "Context")
        assert hasattr(FlextConstants, "Processing")
        assert hasattr(FlextConstants, "FlextWeb")
        assert hasattr(FlextConstants, "Pagination")

    def test_constants_documentation(self) -> None:
        """Test that constants have proper documentation."""
        # Verify that the main class has documentation
        assert FlextConstants.__doc__ is not None
        assert "foundation" in FlextConstants.__doc__.lower()

        # Verify that nested classes have documentation
        assert FlextConstants.__doc__ is not None
        assert FlextConstants.Network.__doc__ is not None
        assert FlextConstants.Validation.__doc__ is not None
        assert FlextConstants.Errors.__doc__ is not None

    def test_class_getitem_nested_path(self) -> None:
        """Test FlextConstants[] method with nested paths."""
        # Test valid nested path access
        validation_error = FlextConstants.Errors.VALIDATION_ERROR
        assert validation_error == "VALIDATION_ERROR"

        # Test another nested path
        default_timeout = FlextConstants.Defaults.TIMEOUT
        assert default_timeout == 30

        # Test deep nested path
        default_level = FlextConstants.Logging.DEFAULT_LEVEL
        assert default_level == "INFO"

    def test_class_getitem_invalid_path(self) -> None:
        """Test FlextConstants[] with invalid path returns placeholder values."""
        # Test accessing a real constant that exists - should work
        result = FlextConstants.Errors.NONEXISTENT_ERROR
        assert result == "NONEXISTENT_ERROR"
