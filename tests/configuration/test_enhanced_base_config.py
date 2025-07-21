"""Tests for Enhanced Base Configuration.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from flext_core.configuration import (
    APIConfig,
    DatabaseConfig,
    EnhancedBaseConfig,
    Environment,
    LogLevel,
    ObservabilityConfig,
)


class TestLogLevel:
    """Test LogLevel enumeration."""

    def test_log_level_values(self) -> None:
        """Test all log level values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_log_level_string_inheritance(self) -> None:
        """Test LogLevel inherits from str."""
        assert isinstance(LogLevel.INFO, str)
        assert LogLevel.INFO.value == "INFO"


class TestEnvironment:
    """Test Environment enumeration."""

    def test_environment_values(self) -> None:
        """Test all environment values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TESTING.value == "testing"

    def test_environment_string_inheritance(self) -> None:
        """Test Environment inherits from str."""
        assert isinstance(Environment.PRODUCTION, str)
        assert Environment.PRODUCTION.value == "production"


class TestEnhancedBaseConfig:
    """Test EnhancedBaseConfig class."""

    def test_default_configuration(self) -> None:
        """Test default configuration values."""
        # Clear any FLEXT environment variables to test true defaults
        with patch.dict(os.environ, {}, clear=True):
            # Create config without loading .env file to test true defaults
            from pydantic_settings import SettingsConfigDict

            class TestEnhancedBaseConfigLocal(EnhancedBaseConfig):
                model_config = SettingsConfigDict(
                    env_file=None,  # Don't load .env file
                    env_file_encoding="utf-8",
                    extra="allow",
                    validate_assignment=True,
                    case_sensitive=False,
                    env_prefix="FLEXT_",
                )

            config = TestEnhancedBaseConfigLocal()

            assert config.debug is False
            assert config.log_level == LogLevel.INFO
            assert config.environment == Environment.DEVELOPMENT
            assert config.app_name == "flext-application"
            assert config.app_version == "1.0.0"
            assert (
                config.secret_key
                == "default-development-key-that-is-long-enough-for-validation"
            )
            assert config.allowed_hosts == ["localhost", "127.0.0.1"]
            assert config.database_url is None
            assert config.database_pool_size == 5
            assert config.database_timeout == 30
            assert config.enable_metrics is True
            assert config.enable_tracing is False
            assert config.metrics_port == 9090

    def test_environment_properties(self) -> None:
        """Test environment check properties."""
        # Development environment
        dev_config = EnhancedBaseConfig(environment=Environment.DEVELOPMENT)
        assert dev_config.is_development is True
        assert dev_config.is_production is False
        assert dev_config.is_testing is False

        # Production environment
        prod_config = EnhancedBaseConfig(environment=Environment.PRODUCTION)
        assert prod_config.is_development is False
        assert prod_config.is_production is True
        assert prod_config.is_testing is False

        # Testing environment
        test_config = EnhancedBaseConfig(environment=Environment.TESTING)
        assert test_config.is_development is False
        assert test_config.is_production is False
        assert test_config.is_testing is True

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        # Valid string conversion
        config = EnhancedBaseConfig(log_level=LogLevel.DEBUG)
        assert config.log_level == LogLevel.DEBUG

        config = EnhancedBaseConfig(log_level=LogLevel.INFO)
        assert config.log_level == LogLevel.INFO

        # LogLevel enum directly
        config = EnhancedBaseConfig(log_level=LogLevel.ERROR)
        assert config.log_level == LogLevel.ERROR

        # Invalid log level - we need to test the validator directly since enum type prevents invalid values
        # Testing string conversion directly
        with pytest.raises(ValueError, match="Invalid log level"):
            # This will test the field validator
            EnhancedBaseConfig.validate_log_level("INVALID")

    def test_environment_validation(self) -> None:
        """Test environment validation."""
        # Valid string conversion
        config = EnhancedBaseConfig(environment=Environment.PRODUCTION)
        assert config.environment == Environment.PRODUCTION

        config = EnhancedBaseConfig(environment=Environment.DEVELOPMENT)
        assert config.environment == Environment.DEVELOPMENT

        # Environment enum directly
        config = EnhancedBaseConfig(environment=Environment.STAGING)
        assert config.environment == Environment.STAGING

        # Invalid environment - test validator directly
        with pytest.raises(ValueError, match="Invalid environment"):
            EnhancedBaseConfig.validate_environment("invalid")

    def test_secret_key_validation(self) -> None:
        """Test secret key validation."""
        # Valid secret key (32+ characters)
        long_key = "a" * 32
        config = EnhancedBaseConfig(secret_key=long_key)
        assert config.secret_key == long_key

        # Invalid secret key (too short)
        with pytest.raises(
            ValueError, match="Secret key must be at least 32 characters"
        ):
            EnhancedBaseConfig(secret_key="short")

    def test_database_url_validation(self) -> None:
        """Test database URL validation."""
        # Valid URLs
        valid_urls = [
            "postgresql://user:pass@host:5432/db",
            "sqlite:///path/to/db.sqlite",
            "mysql://user:pass@host:3306/db",
            "oracle://user:pass@host:1521/db",
        ]

        for url in valid_urls:
            config = EnhancedBaseConfig(database_url=url)
            assert config.database_url == url

        # None is valid
        config = EnhancedBaseConfig(database_url=None)
        assert config.database_url is None

        # Invalid URL scheme
        with pytest.raises(ValueError, match="Unsupported database URL scheme"):
            EnhancedBaseConfig(database_url="redis://localhost:6379")

    def test_get_config_summary(self) -> None:
        """Test configuration summary with hidden sensitive data."""
        with patch.dict(os.environ, {}, clear=True):
            from pydantic_settings import SettingsConfigDict

            class TestEnhancedBaseConfigLocal(EnhancedBaseConfig):
                model_config = SettingsConfigDict(
                    env_file=None,  # Don't load .env file
                    env_file_encoding="utf-8",
                    extra="allow",
                    validate_assignment=True,
                    case_sensitive=False,
                    env_prefix="FLEXT_",
                )

            config = TestEnhancedBaseConfigLocal(
                secret_key="super-secret-key-that-is-long-enough",
                database_url="postgresql://user:pass@host:5432/db",
            )

        summary = config.get_config_summary()

        assert summary["secret_key"] == "[HIDDEN]"
        assert summary["database_url"] == "[HIDDEN]"
        assert summary["debug"] is False
        assert summary["log_level"] == LogLevel.INFO

    def test_validate_configuration_development(self) -> None:
        """Test configuration validation in development."""
        config = EnhancedBaseConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            log_level=LogLevel.DEBUG,
            secret_key="default-development-key-that-is-long-enough-for-validation",
        )

        issues = config.validate_configuration()
        assert len(issues) == 0  # No issues in development

    def test_validate_configuration_production(self) -> None:
        """Test configuration validation in production."""
        config = EnhancedBaseConfig(
            environment=Environment.PRODUCTION,
            debug=True,
            log_level=LogLevel.DEBUG,
            secret_key="default-development-key-that-is-long-enough-for-validation",
            database_url="postgresql://user:pass@localhost:5432/db",
        )

        issues = config.validate_configuration()

        expected_issues = [
            "Debug mode should not be enabled in production",
            "Default secret key detected in production",
            "Debug logging should not be used in production",
            "Production should not use localhost database",
        ]

        assert len(issues) == len(expected_issues)
        for issue in expected_issues:
            assert issue in issues

    def test_setup_environment(self, tmp_path: Path) -> None:
        """Test environment setup."""
        # Change to temporary directory
        original_cwd = Path.cwd()
        os.chdir(tmp_path)

        try:
            config = EnhancedBaseConfig(
                debug=True,
                log_level=LogLevel.DEBUG,
                environment=Environment.TESTING,
                enable_metrics=True,
            )

            config.setup_environment()

            # Check environment variables
            assert os.environ["FLEXT_DEBUG"] == "True"
            assert os.environ["FLEXT_LOG_LEVEL"] == "DEBUG"
            assert os.environ["FLEXT_ENVIRONMENT"] == "testing"

            # Check directories
            assert (tmp_path / "logs").exists()
            assert (tmp_path / "metrics").exists()

        finally:
            os.chdir(original_cwd)

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading configuration from file."""
        # Create temporary config file
        config_file = tmp_path / ".env"
        config_content = """
FLEXT_DEBUG=true
FLEXT_LOG_LEVEL=DEBUG
FLEXT_ENVIRONMENT=production
FLEXT_APP_NAME=test-app
FLEXT_SECRET_KEY=this-is-a-very-long-secret-key-for-testing-purposes
        """.strip()

        config_file.write_text(config_content)

        # Clear existing environment variables to avoid conflicts
        with patch.dict(os.environ, {}, clear=True):
            # Load configuration
            config = EnhancedBaseConfig.load_from_file(config_file)

            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            # File defines FLEXT_ENVIRONMENT=production so we check for that
            assert config.environment == Environment.PRODUCTION
            assert config.app_name == "test-app"
            assert (
                config.secret_key
                == "this-is-a-very-long-secret-key-for-testing-purposes"
            )

    def test_load_from_nonexistent_file(self) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            EnhancedBaseConfig.load_from_file("/nonexistent/config.env")


class TestDatabaseConfig:
    """Test DatabaseConfig class."""

    def test_database_config_defaults(self) -> None:
        """Test database configuration defaults."""
        # Clear environment to avoid conflicts
        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig(
                database_url="postgresql://user:pass@host:5432/db",
                secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
                # Explicitly set to override any inheritance issues
                database_pool_size=10,
            )

            assert config.database_url == "postgresql://user:pass@host:5432/db"
            assert config.database_pool_size == 10
            assert config.database_timeout == 60
            assert config.database_echo is False
            assert config.pool_pre_ping is True
            assert config.pool_recycle == 3600

    def test_database_url_required(self) -> None:
        """Test that database URL is required."""
        # Clear environment to avoid conflicts
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(
                ValueError,
                match="Database URL is required|Unsupported database URL scheme",
            ),
        ):
            DatabaseConfig(
                database_url="",
                secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
            )


class TestAPIConfig:
    """Test APIConfig class."""

    def test_api_config_defaults(self) -> None:
        """Test API configuration defaults."""
        # Clear environment to avoid conflicts
        with patch.dict(os.environ, {}, clear=True):
            config = APIConfig(
                secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
            )

            assert config.host == "127.0.0.1"
            assert config.port == 8000
            assert config.workers == 1
            assert config.enable_cors is True
            assert config.cors_origins == ["*"]
            assert config.api_key_required is False
            assert config.rate_limit_enabled is True
            assert config.rate_limit_requests == 100
            assert config.rate_limit_window == 60

    def test_port_validation(self) -> None:
        """Test port validation."""
        # Clear environment to avoid conflicts
        with patch.dict(os.environ, {}, clear=True):
            # Valid ports
            for port in [1, 80, 443, 8000, 65535]:
                config = APIConfig(
                    port=port,
                    secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
                )
                assert config.port == port

            # Invalid ports
            for port in [0, -1, 65536, 100000]:
                with pytest.raises(
                    ValueError, match="Port must be between 1 and 65535"
                ):
                    APIConfig(
                        port=port,
                        secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
                    )


class TestObservabilityConfig:
    """Test ObservabilityConfig class."""

    def test_observability_config_defaults(self) -> None:
        """Test observability configuration defaults."""
        config = ObservabilityConfig(
            secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
        )

        assert config.enable_metrics is True
        assert config.metrics_port == 9090
        assert config.metrics_path == "/metrics"
        assert config.enable_tracing is False
        assert config.tracing_endpoint is None
        assert config.tracing_sample_rate == 0.1
        assert config.log_format == "json"
        assert config.log_file is None
        assert config.log_rotation is True
        assert config.log_retention_days == 30

    def test_sample_rate_validation(self) -> None:
        """Test tracing sample rate validation."""
        # Clear environment to avoid conflicts
        with patch.dict(os.environ, {}, clear=True):
            # Valid sample rates
            for rate in [0.0, 0.1, 0.5, 1.0]:
                config = ObservabilityConfig(
                    tracing_sample_rate=rate,
                    secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
                )
                assert config.tracing_sample_rate == rate

            # Invalid sample rates
            for rate in [-0.1, 1.1, 2.0]:
                with pytest.raises(
                    ValueError, match="Sample rate must be between 0.0 and 1.0"
                ):
                    ObservabilityConfig(
                        tracing_sample_rate=rate,
                        secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
                    )


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    def test_environment_variable_override(self) -> None:
        """Test environment variable override."""
        # Set environment variables
        os.environ["FLEXT_DEBUG"] = "true"
        os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
        os.environ["FLEXT_APP_NAME"] = "env-test-app"

        try:
            config = EnhancedBaseConfig()

            assert config.debug is True
            assert config.log_level == LogLevel.ERROR
            assert config.app_name == "env-test-app"

        finally:
            # Clean up environment variables
            for key in ["FLEXT_DEBUG", "FLEXT_LOG_LEVEL", "FLEXT_APP_NAME"]:
                os.environ.pop(key, None)

    def test_configuration_inheritance(self) -> None:
        """Test configuration inheritance patterns."""
        # Base configuration
        base_config = EnhancedBaseConfig(
            debug=True,
            log_level=LogLevel.DEBUG,
            secret_key="this-is-a-very-long-secret-key-for-testing-purposes",
        )

        # API configuration inherits from base
        api_config = APIConfig(
            debug=base_config.debug,
            log_level=base_config.log_level,
            secret_key=base_config.secret_key,
            port=3000,
        )

        assert api_config.debug is True
        assert api_config.log_level == LogLevel.DEBUG
        assert api_config.port == 3000

        # Database configuration inherits from base
        db_config = DatabaseConfig(
            debug=base_config.debug,
            log_level=base_config.log_level,
            secret_key=base_config.secret_key,
            database_url="postgresql://user:pass@host:5432/db",
            database_pool_size=20,
        )

        assert db_config.debug is True
        assert db_config.log_level == LogLevel.DEBUG
        assert db_config.database_pool_size == 20

    def test_production_ready_configuration(self) -> None:
        """Test production-ready configuration."""
        config = EnhancedBaseConfig(
            debug=False,
            log_level=LogLevel.WARNING,
            environment=Environment.PRODUCTION,
            secret_key="super-secure-production-key-with-sufficient-length",
            allowed_hosts=["api.example.com", "app.example.com"],
            database_url="postgresql://user:pass@db.example.com:5432/proddb",
            enable_metrics=True,
            enable_tracing=True,
        )

        # Should have no configuration issues
        issues = config.validate_configuration()
        assert len(issues) == 0

        # Verify production characteristics
        assert config.is_production is True
        assert config.debug is False
        assert config.log_level != LogLevel.DEBUG
        assert config.database_url is None or "localhost" not in config.database_url
        assert len(config.secret_key) >= 32
