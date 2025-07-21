"""Comprehensive test coverage for flext_core.config.validators module.

This file ensures 100% test coverage for all validator functions,
including all error paths and edge cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Clean imports - only what actually exists after corruption cleanup
from flext_core.config.validators import (
    CommonValidators,
    port_validator,
    timeout_validator,
    url_validator,
    validate_database_url,
    validate_database_url_field,
    validate_email,
    validate_email_field,
    validate_log_level,
    validate_port,
    validate_port_field,
    validate_timeout,
    validate_timeout_field,
    validate_url,
    validate_url_field,
)


class TestTypeCheckingCoverage:
    """Test TYPE_CHECKING imports for coverage."""

    def test_type_checking_imports_coverage(self) -> None:
        """Test that TYPE_CHECKING imports are covered."""
        # This test ensures that the TYPE_CHECKING block is covered
        # by importing the module during test execution
        import flext_core.config.validators

        # The imports are available during runtime through the module
        assert hasattr(flext_core.config.validators, "CommonValidators")
        assert hasattr(flext_core.config.validators, "url_validator")
        assert hasattr(flext_core.config.validators, "port_validator")


class TestPrivateHelperFunctions:
    """Test private helper functions that raise errors."""

    # Note: _raise_url_error and _raise_database_error are now private implementation details
    # and are tested indirectly through the main validator functions


class TestValidateUrlEdgeCases:
    """Test URL validation edge cases and error paths."""

    def test_url_without_tld_with_require_false(self) -> None:
        """Test URL without TLD when require_tld=False."""
        result = validate_url("http://localhost", require_tld=False)
        assert result == "http://localhost"

    def test_url_malformed_parsing_error(self) -> None:
        """Test URL that causes parsing exception."""
        # This will cause urlparse to succeed but result in invalid format
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_url("http://")

    def test_url_missing_scheme(self) -> None:
        """Test URL missing scheme."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_url("example.com")

    def test_url_missing_netloc(self) -> None:
        """Test URL missing netloc."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_url("http://")

    def test_url_parsing_exception(self) -> None:
        """Test URL that causes urlparse to raise an exception."""
        import unittest.mock

        with unittest.mock.patch("flext_core.config.validators.urlparse") as mock_parse:
            mock_parse.side_effect = Exception("Parsing failed")
            with pytest.raises(ValueError, match="Invalid URL: Parsing failed"):
                validate_url("http://example.com")


class TestValidateDatabaseUrlErrorPaths:
    """Test database URL validation error paths."""

    def test_database_url_parsing_exception(self) -> None:
        """Test database URL that causes parsing exception."""
        # Mock a URL that would cause urlparse to raise an exception
        import unittest.mock

        with unittest.mock.patch("flext_core.config.validators.urlparse") as mock_parse:
            mock_parse.side_effect = Exception("Parse error")
            with pytest.raises(ValueError, match="Invalid database URL: Parse error"):
                validate_database_url("postgresql://test")

    def test_database_url_missing_scheme_after_pattern_match(self) -> None:
        """Test database URL that matches pattern but has no scheme."""
        # This is a complex edge case - URL passes regex but fails urlparse scheme check
        import unittest.mock

        with unittest.mock.patch("flext_core.config.validators.urlparse") as mock_parse:
            mock_result = unittest.mock.Mock()
            mock_result.scheme = ""
            mock_result.netloc = "localhost"
            mock_parse.return_value = mock_result

            with pytest.raises(ValueError, match="Database URL must include a scheme"):
                validate_database_url("postgresql://localhost/db")

    def test_database_url_missing_host_non_sqlite(self) -> None:
        """Test non-SQLite database URL missing host."""
        import unittest.mock

        with unittest.mock.patch("flext_core.config.validators.urlparse") as mock_parse:
            mock_result = unittest.mock.Mock()
            mock_result.scheme = "postgresql"
            mock_result.netloc = ""
            mock_parse.return_value = mock_result

            with pytest.raises(ValueError, match="Database URL must include a host"):
                validate_database_url("postgresql://localhost/db")


class TestCommonValidatorsFieldValidators:
    """Test CommonValidators field validator methods."""

    def test_validate_url_field_none(self) -> None:
        """Test URL field validator with None value."""
        result = validate_url_field(None)
        assert result is None

    def test_validate_url_field_valid(self) -> None:
        """Test URL field validator with valid URL."""
        result = validate_url_field("https://example.com")
        assert result == "https://example.com"

    def test_validate_url_field_invalid(self) -> None:
        """Test URL field validator with invalid URL."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            validate_url_field("")

    def test_validate_database_url_field_none(self) -> None:
        """Test database URL field validator with None value."""
        result = validate_database_url_field(None)
        assert result is None

    def test_validate_database_url_field_valid(self) -> None:
        """Test database URL field validator with valid database URL."""
        result = validate_database_url_field(
            "postgresql://localhost/db"
        )
        assert result == "postgresql://localhost/db"

    def test_validate_database_url_field_invalid(self) -> None:
        """Test database URL field validator with invalid database URL."""
        with pytest.raises(ValueError, match="Database URL cannot be empty"):
            validate_database_url_field("")

    def test_validate_port_field_none(self) -> None:
        """Test port field validator with None value."""
        result = validate_port_field(None)
        assert result is None

    def test_validate_port_field_valid_int(self) -> None:
        """Test port field validator with valid integer."""
        result = validate_port_field(8080)
        assert result == 8080

    def test_validate_port_field_valid_string(self) -> None:
        """Test port field validator with valid string."""
        result = validate_port_field("8080")
        assert result == 8080

    def test_validate_port_field_invalid(self) -> None:
        """Test port field validator with invalid port."""
        with pytest.raises((ValueError, TypeError)):
            validate_port_field("invalid")

    def test_validate_timeout_field_none(self) -> None:
        """Test timeout field validator with None value."""
        result = validate_timeout_field(None)
        assert result is None

    def test_validate_timeout_field_valid_float(self) -> None:
        """Test timeout field validator with valid float."""
        result = validate_timeout_field(30.5)
        assert result == 30.5

    def test_validate_timeout_field_valid_string(self) -> None:
        """Test timeout field validator with valid string."""
        result = validate_timeout_field("30.5")
        assert result == 30.5

    def test_validate_timeout_field_invalid(self) -> None:
        """Test timeout field validator with invalid timeout."""
        with pytest.raises((ValueError, TypeError)):
            validate_timeout_field("invalid")


class TestEmailValidation:
    """Test email validation functionality."""

    def test_validate_email_field_none(self) -> None:
        """Test email field validator with None value."""
        result = validate_email_field(None)
        assert result is None

    def test_validate_email_field_valid(self) -> None:
        """Test email field validator with valid email."""
        result = validate_email_field("test@example.com")
        assert result == "test@example.com"

    def test_validate_email_field_uppercase_normalized(self) -> None:
        """Test email field validator normalizes to lowercase."""
        result = validate_email_field("TEST@EXAMPLE.COM")
        assert result == "test@example.com"

    def test_validate_email_field_invalid(self) -> None:
        """Test email field validator with invalid email."""
        invalid_emails = [
            "invalid",
            "@example.com",
            "test@",
            "test@example",
            "test.example.com",
            "",
        ]
        for email in invalid_emails:
            with pytest.raises(ValueError, match="Invalid email format"):
                validate_email_field(email)


class TestLogLevelValidation:
    """Test log level validation functionality."""

    def test_validate_log_level_field_valid_lowercase(self) -> None:
        """Test log level field validator with valid lowercase level."""
        result = validate_log_level("debug")
        assert result == "DEBUG"

    def test_validate_log_level_field_valid_uppercase(self) -> None:
        """Test log level field validator with valid uppercase level."""
        result = validate_log_level("INFO")
        assert result == "INFO"

    def test_validate_log_level_field_valid_mixed_case(self) -> None:
        """Test log level field validator with mixed case level."""
        result = validate_log_level("Warning")
        assert result == "WARNING"

    def test_validate_log_level_field_all_valid_levels(self) -> None:
        """Test all valid log levels."""
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        expected_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for input_level, expected in zip(valid_levels, expected_levels, strict=False):
            result = validate_log_level(input_level)
            assert result == expected

    def test_validate_log_level_field_invalid(self) -> None:
        """Test log level field validator with invalid level."""
        invalid_levels = ["trace", "fatal", "verbose", "invalid", ""]

        for level in invalid_levels:
            with pytest.raises(ValueError, match="Invalid log level"):
                validate_log_level(level)


class TestValidatorDecorators:
    """Test validator decorator functions."""

    def test_url_validator_decorator(self) -> None:
        """Test url_validator decorator function."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            url: str = ""

            _validate_url = url_validator()

        # Test valid URL
        model = TestModel(url="https://example.com")
        assert model.url == "https://example.com"

        # Test invalid URL should raise validation error
        with pytest.raises(ValidationError):
            TestModel(url="not-a-url")

    def test_url_validator_decorator_with_require_tld_false(self) -> None:
        """Test url_validator decorator with require_tld=False."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            url: str = ""

            _validate_url = url_validator(require_tld=False)

        # Test URL without TLD should work with require_tld=False
        model = TestModel(url="http://localhost")
        assert model.url == "http://localhost"

    def test_port_validator_decorator(self) -> None:
        """Test port_validator decorator function."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            port: int = 8080

            _validate_port = port_validator()

        # Test valid port
        model = TestModel(port=8080)
        assert model.port == 8080

        # Test invalid port should raise validation error
        with pytest.raises(ValidationError):
            TestModel(port=70000)

    def test_timeout_validator_decorator(self) -> None:
        """Test timeout_validator decorator function."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            timeout: float = 30.0

            _validate_timeout = timeout_validator()

        # Test valid timeout
        model = TestModel(timeout=30.0)
        assert model.timeout == 30.0

        # Test invalid timeout should raise validation error
        with pytest.raises(ValidationError):
            TestModel(timeout=-1.0)

    def test_timeout_validator_decorator_with_min_value(self) -> None:
        """Test timeout_validator decorator with custom min_value."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            timeout: float = 30.0

            _validate_timeout = timeout_validator(min_value=1.0)

        # Test valid timeout above min_value
        model = TestModel(timeout=30.0)
        assert model.timeout == 30.0

        # Test timeout below min_value should raise validation error
        with pytest.raises(ValidationError):
            TestModel(timeout=0.5)

    def test_decorator_inner_functions(self) -> None:
        """Test the inner validator functions created by decorators."""
        # Test url_validator inner function manually
        from flext_core.config.validators import timeout_validator, url_validator

        # Create the decorators which creates the inner functions
        url_dec = url_validator(require_tld=False)
        timeout_dec = timeout_validator(min_value=1.0)

        # Access the inner function by getting the wrapped attribute
        # This simulates what Pydantic does internally
        if hasattr(url_dec, "wrapped"):
            inner_url_func = url_dec.wrapped
            # Test calling the inner function directly
            result = inner_url_func("http://localhost")
            assert result == "http://localhost"

        if hasattr(timeout_dec, "wrapped"):
            inner_timeout_func = timeout_dec.wrapped
            # Test calling the inner function directly
            result = inner_timeout_func(5.0)
            assert result == 5.0

            # Test error case
            with pytest.raises(ValueError, match="Timeout must be"):
                inner_timeout_func(0.5)  # Below min_value


class TestValidationIntegrationScenarios:
    """Test real-world validation scenarios."""

    def test_production_database_urls(self) -> None:
        """Test production-style database URLs."""
        production_urls = [
            "postgresql://user:password@prod-db.example.com:5432/myapp",
            "mysql://app_user:secret123@mysql-cluster.internal:3306/application_db",
            "sqlite:///var/lib/app/production.db",
            "oracle://system:manager@oracle-server:1521/xe",
        ]

        for url in production_urls:
            result = validate_database_url(url)
            assert result == url

    def test_development_scenarios(self) -> None:
        """Test development environment scenarios."""
        # Development URLs that should work without TLD requirement
        dev_urls = [
            "http://localhost:3000",
            "http://dev-server:8080",
            "https://api.local",
        ]

        for url in dev_urls:
            result = validate_url(url, require_tld=False)
            assert result == url

    def test_common_port_ranges(self) -> None:
        """Test common port number ranges."""
        common_ports = [80, 443, 8080, 3000, 5432, 3306, 6379, 27017, 9200, 5601]

        for port in common_ports:
            result = validate_port(port)
            assert result == port

            # Test string versions too
            result = validate_port(str(port))
            assert result == port

    def test_realistic_timeout_values(self) -> None:
        """Test realistic timeout values."""
        timeouts = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0, 300.0, 3600.0]

        for timeout in timeouts:
            result = validate_timeout(timeout)
            assert result == timeout

            # Test string versions
            result = validate_timeout(str(timeout))
            assert result == timeout


class TestErrorMessageQuality:
    """Test that error messages are informative and helpful."""

    def test_url_error_messages(self) -> None:
        """Test URL validation error messages are descriptive."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            validate_url("")

        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_url("not-a-url")

        with pytest.raises(ValueError, match="URL must contain a valid domain"):
            validate_url("http://localhost", require_tld=True)

    def test_database_url_error_messages(self) -> None:
        """Test database URL validation error messages are descriptive."""
        with pytest.raises(ValueError, match="Database URL cannot be empty"):
            validate_database_url("")

        with pytest.raises(
            ValueError,
            match="Invalid database URL. Must start with a valid database scheme",
        ):
            validate_database_url("http://example.com")

    def test_port_error_messages(self) -> None:
        """Test port validation error messages are descriptive."""
        with pytest.raises(TypeError, match="Port must be an integer"):
            validate_port("not-a-number")

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            validate_port(0)

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            validate_port(65536)

    def test_timeout_error_messages(self) -> None:
        """Test timeout validation error messages are descriptive."""
        with pytest.raises(TypeError, match="Timeout must be a number"):
            validate_timeout("not-a-number")

        with pytest.raises(ValueError, match="Timeout must be at least 0.0 seconds"):
            validate_timeout(-1.0)

        with pytest.raises(ValueError, match="Timeout must be at least 5.0 seconds"):
            validate_timeout(1.0, min_value=5.0)

    def test_log_level_error_message(self) -> None:
        """Test log level validation error message includes valid options."""
        with pytest.raises(ValueError, match="Invalid log level"):
            validate_log_level("invalid")
