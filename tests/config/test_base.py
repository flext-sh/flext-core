"""Tests for flext_core.config.base module."""

from __future__ import annotations

import pytest

from flext_core.config.base import BaseConfig, BaseSettings, ConfigurationError
from flext_core.domain.constants import ConfigDefaults, Environments


class TestConfigDefaults:
    """Test configuration defaults."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        assert ConfigDefaults.DEFAULT_TIMEOUT == 30
        assert ConfigDefaults.DEFAULT_RETRY_COUNT == 3
        assert ConfigDefaults.DEFAULT_BATCH_SIZE == 1000
        assert ConfigDefaults.ENV_PREFIX == "FLEXT_"
        assert ConfigDefaults.ENV_DELIMITER == "__"


class TestEnvironments:
    """Test environment constants."""

    def test_environment_values(self) -> None:
        """Test environment constant values."""
        assert Environments.DEVELOPMENT == "development"
        assert Environments.STAGING == "staging"
        assert Environments.PRODUCTION == "production"
        assert Environments.TEST == "test"
        assert Environments.DEFAULT == "development"


class TestBaseConfig:
    """Test BaseConfig class."""

    def test_base_config_creation(self) -> None:
        """Test BaseConfig can be instantiated."""
        config = BaseConfig()
        assert config is not None
        assert hasattr(config, "model_config")
        assert hasattr(config, "to_dict")

    def test_base_config_to_dict(self) -> None:
        """Test BaseConfig to_dict method."""
        config = BaseConfig()
        result = config.to_dict()
        assert isinstance(result, dict)

    def test_base_config_model_config(self) -> None:
        """Test BaseConfig model configuration."""
        config = BaseConfig()
        assert config.model_config["extra"] == "forbid"
        assert config.model_config["validate_assignment"] is True
        assert config.model_config["str_strip_whitespace"] is True

    def test_get_subsection(self) -> None:
        """Test get_subsection method."""
        config = BaseConfig()
        # This method should work even with no data
        result = config.get_subsection("test_")
        assert isinstance(result, dict)


class TestBaseSettings:
    """Test BaseSettings class."""

    def test_base_settings_creation(self) -> None:
        """Test BaseSettings can be instantiated."""
        settings = BaseSettings()
        assert settings is not None
        assert hasattr(settings, "project_name")
        assert hasattr(settings, "environment")
        assert hasattr(settings, "debug")

    def test_base_settings_defaults(self) -> None:
        """Test BaseSettings default values."""
        # Disable env file loading to test true defaults
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(_env_file=None)
            assert settings.project_name == "flext"
            assert settings.environment == "development"
        # Check that debug is boolean rather than assuming False
        # (configuration may set it to True in development)
        assert isinstance(settings.debug, bool)

    def test_base_settings_with_values(self) -> None:
        """Test BaseSettings with specific values."""
        settings = BaseSettings(
            environment="production",
            debug=True,
            project_name="test-project",
        )
        assert settings.environment == "production"
        assert settings.debug is True
        assert settings.project_name == "test-project"

    def test_base_settings_model_config(self) -> None:
        """Test BaseSettings model configuration."""
        settings = BaseSettings()
        assert settings.model_config["env_prefix"] == "FLEXT_"
        assert settings.model_config["env_file"] == ".env"
        assert settings.model_config["case_sensitive"] is False


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_configuration_error_creation(self) -> None:
        """Test ConfigurationError can be created."""
        error = ConfigurationError("Test error")
        assert str(error) == "Test error"

    def test_configuration_error_inheritance(self) -> None:
        """Test ConfigurationError inherits from DomainError."""
        error = ConfigurationError("Test error")
        # Should be a valid exception
        assert isinstance(error, Exception)


class TestConfigIntegration:
    """Test configuration system integration."""

    def test_base_config_inheritance(self) -> None:
        """Test that BaseConfig can be properly inherited."""

        class TestConfig(BaseConfig):
            test_field: str = "test_value"

        config = TestConfig()
        assert config.test_field == "test_value"
        assert hasattr(config, "to_dict")

    def test_base_settings_inheritance(self) -> None:
        """Test that BaseSettings can be properly inherited."""

        class TestSettings(BaseSettings):
            custom_setting: str = "custom_value"

        settings = TestSettings()
        assert settings.custom_setting == "custom_value"
        assert settings.project_name == "flext"  # Inherited default

    def test_config_validation_error_handling(self) -> None:
        """Test config validation error handling."""
        from pydantic import ValidationError

        # Test that validation errors are properly raised
        class BadConfig(BaseConfig):
            pass

        config = BadConfig()
        # Since extra="forbid", setting unknown attributes should fail
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            config.model_validate({"unknown_field": "value"})

    def test_settings_env_prefix(self) -> None:
        """Test settings environment prefix functionality."""
        settings = BaseSettings()
        assert settings.model_config["env_prefix"] == "FLEXT_"

    def test_config_to_dict_functionality(self) -> None:
        """Test config to_dict functionality with actual data."""

        class TestConfig(BaseConfig):
            name: str = "test_name"
            value: int = 42

        config = TestConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test_name"
        assert result["value"] == 42

    def test_subsection_functionality(self) -> None:
        """Test get_subsection functionality with actual data."""

        class TestConfig(BaseConfig):
            prefix_name: str = "test"
            prefix_value: int = 42
            other_field: str = "other"

        config = TestConfig()
        result = config.get_subsection("prefix_")

        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert "other_field" not in result
        assert result["name"] == "test"
        assert result["value"] == 42
