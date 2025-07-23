"""Unit tests for FlextCoreSettings - Modern pytest patterns.

Tests for the pydantic-settings based configuration management system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from flext_core.config import FlextCoreSettings
from flext_core.config import configure_settings
from flext_core.config import get_settings
from flext_core.constants import FlextConstants
from flext_core.constants import FlextEnvironment
from flext_core.constants import FlextLogLevel

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestFlextCoreSettings:
    """Unit tests for FlextCoreSettings pydantic model."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        settings = FlextCoreSettings()

        assert settings.environment == FlextConstants.DEFAULT_ENVIRONMENT
        assert settings.log_level == FlextConstants.DEFAULT_LOG_LEVEL
        assert settings.service_timeout == FlextConstants.DEFAULT_TIMEOUT
        assert settings.max_retries == 3
        assert settings.debug is False

    @pytest.mark.parametrize(
        "environment",
        [
            FlextEnvironment.DEVELOPMENT,
            FlextEnvironment.TESTING,
            FlextEnvironment.STAGING,
            FlextEnvironment.PRODUCTION,
        ],
    )
    def test_environment_validation(
        self,
        environment: FlextEnvironment,
    ) -> None:
        """Test environment enum validation."""
        settings = FlextCoreSettings(environment=environment)
        assert settings.environment == environment

    @pytest.mark.parametrize(
        "log_level",
        [
            FlextLogLevel.DEBUG,
            FlextLogLevel.INFO,
            FlextLogLevel.WARNING,
            FlextLogLevel.ERROR,
            FlextLogLevel.CRITICAL,
        ],
    )
    def test_log_level_validation(self, log_level: FlextLogLevel) -> None:
        """Test log level enum validation."""
        settings = FlextCoreSettings(log_level=log_level)
        assert settings.log_level == log_level

    @pytest.mark.parametrize(
        ("timeout", "expected"),
        [
            (1, 1),  # Minimum valid
            (30, 30),  # Default
            (300, 300),  # Maximum valid
        ],
    )
    def test_service_timeout_validation(
        self,
        timeout: int,
        expected: int,
    ) -> None:
        """Test service timeout validation with valid values."""
        settings = FlextCoreSettings(service_timeout=timeout)
        assert settings.service_timeout == expected

    @pytest.mark.parametrize("invalid_timeout", [0, -1, 301, 1000])
    def test_service_timeout_invalid(self, invalid_timeout: int) -> None:
        """Test service timeout validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextCoreSettings(service_timeout=invalid_timeout)

    @pytest.mark.parametrize(
        ("retries", "expected"),
        [
            (0, 0),  # Minimum valid
            (3, 3),  # Default
            (10, 10),  # Maximum valid
        ],
    )
    def test_max_retries_validation(self, retries: int, expected: int) -> None:
        """Test max retries validation with valid values."""
        settings = FlextCoreSettings(max_retries=retries)
        assert settings.max_retries == expected

    @pytest.mark.parametrize("invalid_retries", [-1, 11, 100])
    def test_max_retries_invalid(self, invalid_retries: int) -> None:
        """Test max retries validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextCoreSettings(max_retries=invalid_retries)

    def test_directory_path_resolution(self, tmp_path: Path) -> None:
        """Test that directory paths are resolved to absolute paths."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"
        logs_dir = tmp_path / "logs"

        settings = FlextCoreSettings(
            config_dir=config_dir,
            data_dir=data_dir,
            logs_dir=logs_dir,
        )

        # All paths should be absolute
        assert settings.config_dir.is_absolute()
        assert settings.data_dir.is_absolute()
        assert settings.logs_dir.is_absolute()

        # Should be resolved paths
        assert settings.config_dir == config_dir.resolve()
        assert settings.data_dir == data_dir.resolve()
        assert settings.logs_dir == logs_dir.resolve()

    def test_log_format_validation(self) -> None:
        """Test log format string validation."""
        valid_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        settings = FlextCoreSettings(log_format=valid_format)
        assert settings.log_format == valid_format

    def test_log_format_empty_invalid(self) -> None:
        """Test that empty log format raises validation error."""
        with pytest.raises(ValidationError):
            FlextCoreSettings(log_format="")

        with pytest.raises(ValidationError):
            FlextCoreSettings(log_format="   ")

    def test_environment_helper_methods(self) -> None:
        """Test environment helper methods."""
        dev_settings = FlextCoreSettings(
            environment=FlextEnvironment.DEVELOPMENT,
        )
        prod_settings = FlextCoreSettings(
            environment=FlextEnvironment.PRODUCTION,
        )
        test_settings = FlextCoreSettings(environment=FlextEnvironment.TESTING)

        # Development
        assert dev_settings.is_development() is True
        assert dev_settings.is_production() is False
        assert dev_settings.is_testing() is False

        # Production
        assert prod_settings.is_development() is False
        assert prod_settings.is_production() is True
        assert prod_settings.is_testing() is False

        # Testing
        assert test_settings.is_development() is False
        assert test_settings.is_production() is False
        assert test_settings.is_testing() is True

    def test_model_dump_safe(self) -> None:
        """Test safe model dumping without sensitive data."""
        settings = FlextCoreSettings(debug=True)
        safe_data = settings.model_dump_safe()

        # Should contain normal configuration
        assert "environment" in safe_data
        assert "log_level" in safe_data
        assert "debug" in safe_data

        # Should not contain private fields (none in this case, but
        # method should work)
        for key in safe_data:
            assert not key.startswith("_")

    def test_settings_immutability(
        self,
        sample_settings: FlextCoreSettings,
    ) -> None:
        """Test that settings are immutable after creation."""
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            sample_settings.environment = FlextEnvironment.PRODUCTION  # type: ignore[misc]


@pytest.mark.unit
class TestSettingsGlobalFunctions:
    """Unit tests for global settings management functions."""

    def test_get_settings_singleton(self) -> None:
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
        assert isinstance(settings1, FlextCoreSettings)

    def test_configure_settings_with_instance(
        self,
        sample_settings: FlextCoreSettings,
    ) -> None:
        """Test configuring settings with a custom instance."""
        configured = configure_settings(sample_settings)

        assert configured is sample_settings
        assert get_settings() is sample_settings
        assert get_settings().debug is True

    def test_configure_settings_with_none(self) -> None:
        """Test configuring settings with None creates new default
        instance."""
        configured = configure_settings(None)

        assert isinstance(configured, FlextCoreSettings)
        assert get_settings() is configured
        assert configured.environment == FlextConstants.DEFAULT_ENVIRONMENT

    def test_settings_reset_between_tests(self) -> None:
        """Test that settings are properly isolated between tests."""
        # This test should pass if global state is properly managed
        current_settings = get_settings()
        assert isinstance(current_settings, FlextCoreSettings)


@pytest.mark.integration
class TestSettingsEnvironmentIntegration:
    """Integration tests for environment variable loading."""

    def test_environment_prefix_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test loading settings from environment variables with
        FLEXT_ prefix."""
        monkeypatch.setenv("FLEXT_ENVIRONMENT", "production")
        monkeypatch.setenv("FLEXT_LOG_LEVEL", "ERROR")
        monkeypatch.setenv("FLEXT_DEBUG", "true")
        monkeypatch.setenv("FLEXT_SERVICE_TIMEOUT", "60")

        settings = FlextCoreSettings()

        assert settings.environment == FlextEnvironment.PRODUCTION
        assert settings.log_level == FlextLogLevel.ERROR
        assert settings.debug is True
        assert settings.service_timeout == 60

    def test_case_insensitive_env_vars(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that environment variables are case insensitive."""
        monkeypatch.setenv("flext_environment", "testing")  # Lowercase
        monkeypatch.setenv("FLEXT_LOG_LEVEL", "DEBUG")  # Uppercase

        settings = FlextCoreSettings()

        assert settings.environment == FlextEnvironment.TESTING
        assert settings.log_level == FlextLogLevel.DEBUG

    def test_env_file_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test loading settings from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FLEXT_ENVIRONMENT=staging\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n",
        )

        # Change to the temp directory so .env is found
        monkeypatch.chdir(tmp_path)

        # Create settings with explicit .env file path
        settings = FlextCoreSettings(_env_file=str(env_file))

        assert settings.environment == FlextEnvironment.STAGING
        assert settings.log_level == FlextLogLevel.WARNING
        assert settings.debug is True
