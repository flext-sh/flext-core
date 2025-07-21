"""Simplified tests for flext_core.config.validators module."""

# fmt: off

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core.config.validators import (
    validate_database_url,
    validate_port,
    validate_timeout,
    validate_url,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestDatabaseUrlValidator:
    """Test database URL validation functionality."""

    def test_validate_postgresql_url(self) -> None:
        """Test validation of PostgreSQL URLs."""
        valid_urls = [
            "postgresql://user:pass@localhost:5432/dbname",
            "postgres://user@localhost/dbname",
            "postgresql://localhost/dbname",
        ]

        for url in valid_urls:
            result = validate_database_url(url)
            assert result == url

    def test_validate_mysql_url(self) -> None:
        """Test validation of MySQL URLs."""
        valid_urls = [
            "mysql://user:pass@localhost:3306/dbname",
            "mysql://user@localhost/dbname",
        ]

        for url in valid_urls:
            result = validate_database_url(url)
            assert result == url

    def test_validate_sqlite_url(self) -> None:
        """Test validation of SQLite URLs."""
        valid_urls = [
            "sqlite:///path/to/database.db",
            "sqlite:///:memory:",
        ]

        for url in valid_urls:
            result = validate_database_url(url)
            assert result == url

    def test_invalid_database_urls(self) -> None:
        """Test validation of invalid database URLs."""
        # Test empty URL
        with pytest.raises(ValueError, match="Database URL cannot be empty"):
            validate_database_url("")

        # Test invalid schemes
        invalid_scheme_urls = [
            "invalid://url",  # Invalid scheme
            "http://example.com",  # Not a database URL
            "mysql:",  # Incomplete
        ]

        for url in invalid_scheme_urls:
            with pytest.raises(ValueError, match="(Invalid database URL|Database URL must include)"):
                validate_database_url(url)


class TestUrlValidator:
    """Test URL validation functionality."""

    def test_validate_valid_urls(self) -> None:
        """Test validation of valid URLs."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://example.com:8080",
            "https://example.com/path",
            "https://user:pass@example.com",
        ]

        for url in valid_urls:
            result = validate_url(url)
            assert result == url

    def test_validate_invalid_urls(self) -> None:
        """Test validation of invalid URLs."""
        invalid_urls = [
            "",  # Empty
            "not-a-url",  # No scheme
            "http://",  # No host
            "https://",  # No host
            "http://localhost",  # No TLD (by default require_tld=True)
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError, match="(Invalid URL|URL cannot be empty|URL must contain a valid domain)"):
                validate_url(url)


class TestPortValidator:
    """Test port validation functionality."""

    def test_validate_valid_ports(self) -> None:
        """Test validation of valid port numbers."""
        valid_ports: list[int | str] = [80, 443, 8080, 3000, 5432, "80", "443"]

        for port in valid_ports:
            result = validate_port(port)
            assert isinstance(result, int), (
                f"Expected int, got {type(result)} for port {port}"
            )
            assert 1 <= result <= 65535

    def test_validate_invalid_ports(self) -> None:
        """Test validation of invalid port numbers."""
        invalid_ports: list[int | str] = [0, -1, 65536, 100000, "invalid", ""]

        for port in invalid_ports:
            with pytest.raises((ValueError, TypeError)):
                validate_port(port)

    def test_port_string_conversion(self) -> None:
        """Test port string to int conversion."""
        result = validate_port("8080")
        assert isinstance(result, int)
        assert result == 8080


class TestTimeoutValidator:
    """Test timeout validation functionality."""

    def test_validate_valid_timeouts(self) -> None:
        """Test validation of valid timeout values."""
        valid_timeouts: list[float | str] = [1.0, 30.0, 60.5, 300, "30", "60.5"]

        for timeout in valid_timeouts:
            result = validate_timeout(timeout)
            assert isinstance(result, float), (
                f"Expected float, got {type(result)} for timeout {timeout}"
            )
            assert result >= 0

    def test_validate_invalid_timeouts(self) -> None:
        """Test validation of invalid timeout values."""
        invalid_timeouts: list[float | str] = [-1, -10.5, "invalid", ""]

        for timeout in invalid_timeouts:
            with pytest.raises((ValueError, TypeError)):
                validate_timeout(timeout)

    def test_timeout_string_conversion(self) -> None:
        """Test timeout string to float conversion."""
        result = validate_timeout("30.5")
        assert result == 30.5
        assert isinstance(result, float)

    def test_timeout_minimum_value(self) -> None:
        """Test timeout minimum value validation."""
        # Default minimum is 0.0
        result = validate_timeout(5.0, min_value=1.0)
        assert result == 5.0

        # Should fail if below minimum
        with pytest.raises(ValueError, match="Timeout must be"):
            validate_timeout(0.5, min_value=1.0)


class TestValidatorIntegration:
    """Test validator integration and combined usage."""

    def test_multiple_validators_usage(self) -> None:
        """Test using multiple validators together."""
        # Test valid values
        db_url = validate_database_url("postgresql://localhost/test")
        url = validate_url("https://example.com")
        port = validate_port(8080)
        timeout = validate_timeout(30.0)

        assert db_url == "postgresql://localhost/test"
        assert url == "https://example.com"
        assert port == 8080
        assert timeout == 30.0

    def test_validator_error_propagation(self) -> None:
        """Test that validator errors are properly propagated."""
        # All these should raise ValueError or TypeError
        with pytest.raises(ValueError, match="Database URL cannot be empty"):
            validate_database_url("")

        with pytest.raises(ValueError, match="URL cannot be empty"):
            validate_url("")

        with pytest.raises((ValueError, TypeError)):
            validate_port("invalid")

        with pytest.raises((ValueError, TypeError)):
            validate_timeout("invalid")


class TestValidatorErrorHandling:
    """Test error handling in validators."""

    def test_validators_handle_none_input(self) -> None:
        """Test validators handle None input gracefully."""
        from typing import Any

        validators_and_inputs: list[tuple[Callable[[Any], Any], None]] = [
            (validate_database_url, None),
            (validate_url, None),
            (validate_port, None),
            (validate_timeout, None),
        ]

        for validator_func, input_val in validators_and_inputs:
            with pytest.raises((ValueError, TypeError)):
                validator_func(input_val)

    def test_validators_handle_wrong_type_input(self) -> None:
        """Test validators handle wrong input types."""
        string_validators = [validate_database_url, validate_url]
        wrong_inputs = [123, [], {}, True]

        for validator_func in string_validators:
            for wrong_input in wrong_inputs:
                with pytest.raises((ValueError, TypeError)):
                    validator_func(wrong_input)  # type: ignore[operator]
