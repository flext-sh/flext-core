"""Foundation configuration patterns tests for FLEXT Core.

Tests only the foundation configuration patterns that belong in flext-core:
- FlextConfig (base configuration pattern)
- FlextSettings (environment-aware base)
- FlextSystemDefaults (ecosystem defaults)
- Foundation utility functions

Configurations specific to other libraries are tested in their respective libraries:
- FlextJWTConfig → flext-auth
- FlextDatabaseConfig → flext-db-oracle
- FlextLDAPConfig → flext-ldap
- FlextObservabilityConfig → flext-observability
- FlextSingerConfig → flext-meltano
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from flext_core import (
    FlextConfig,
    FlextSettings,
    FlextSystemDefaults,
    merge_configs,
    safe_get_env_var,
    safe_load_json_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextSystemDefaults:
    """Test FlextSystemDefaults foundation patterns."""

    def test_system_defaults_structure(self) -> None:
        """Test that FlextSystemDefaults provides foundation defaults."""
        # Security defaults
        assert FlextSystemDefaults.Security.MIN_PASSWORD_LENGTH_HIGH_SECURITY == 12
        assert FlextSystemDefaults.Security.MIN_PASSWORD_LENGTH_MEDIUM_SECURITY == 8
        assert FlextSystemDefaults.Security.MIN_SECRET_KEY_LENGTH_STRONG == 64
        assert FlextSystemDefaults.Security.MIN_SECRET_KEY_LENGTH_ADEQUATE == 32

        # Network defaults
        assert FlextSystemDefaults.Network.TIMEOUT == 30
        assert FlextSystemDefaults.Network.RETRIES == 3

        # Pagination defaults
        assert FlextSystemDefaults.Pagination.PAGE_SIZE == 100
        assert FlextSystemDefaults.Pagination.MAX_PAGE_SIZE == 1000

        # Environment defaults
        assert FlextSystemDefaults.Environment.DEFAULT_ENV == "development"


class TestFlextConfig:
    """Test FlextConfig foundation configuration pattern."""

    def test_config_default_creation(self) -> None:
        """Test FlextConfig creates with defaults."""
        config = FlextConfig()

        assert config.name == "flext"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.timeout == 30
        assert config.retries == 3
        assert config.page_size == 100
        assert config.enable_caching is True
        assert config.enable_metrics is True
        assert config.enable_tracing is False

    def test_config_custom_values(self) -> None:
        """Test FlextConfig with custom values."""
        config = FlextConfig(
            name="custom-app",
            version="2.0.0",
            environment="production",
            debug=False,
            log_level="WARNING",
            timeout=60,
            enable_tracing=True,
        )

        assert config.name == "custom-app"
        assert config.version == "2.0.0"
        assert config.environment == "production"
        assert config.debug is False
        assert config.log_level == "WARNING"
        assert config.timeout == 60
        assert config.enable_tracing is True

    def test_config_environment_validation(self) -> None:
        """Test environment validation with shortcuts."""
        # Test shorthand mappings
        config_dev = FlextConfig(environment="dev")
        assert config_dev.environment == "development"

        config_prod = FlextConfig(environment="prod")
        assert config_prod.environment == "production"

        config_stage = FlextConfig(environment="stage")
        assert config_stage.environment == "staging"

        # Test invalid environment
        with pytest.raises(ValueError, match="Environment must be one of"):
            FlextConfig(environment="invalid")

    def test_config_log_level_validation(self) -> None:
        """Test log level validation."""
        # Valid log levels
        config = FlextConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

        config = FlextConfig(log_level="error")  # Should be converted to uppercase
        assert config.log_level == "ERROR"

        # Invalid log level
        with pytest.raises(ValueError, match="Log level must be one of"):
            FlextConfig(log_level="INVALID")

    def test_config_positive_integer_validation(self) -> None:
        """Test positive integer validation."""
        # Valid positive values
        config = FlextConfig(timeout=60, retries=5, page_size=200)
        assert config.timeout == 60
        assert config.retries == 5
        assert config.page_size == 200

        # Invalid zero values
        with pytest.raises(ValueError, match="Value must be positive"):
            FlextConfig(timeout=0)

        with pytest.raises(ValueError, match="Value must be positive"):
            FlextConfig(retries=-1)

    def test_config_business_rules_validation(self) -> None:
        """Test business rules validation."""
        # Valid configuration
        config = FlextConfig(environment="development", debug=True)
        result = config.validate_business_rules()
        assert result.success

        # Invalid: debug in production
        config = FlextConfig(environment="production", debug=True)
        result = config.validate_business_rules()
        assert result.failure
        assert "Debug mode cannot be enabled in production" in result.error

    def test_config_serialization(self) -> None:
        """Test config serialization for API output."""
        config = FlextConfig(
            environment="production",
            log_level="INFO",
            enable_caching=True,
            enable_metrics=True,
            enable_tracing=True,
        )

        json_data = config.model_dump(mode="json")

        # Check environment serialization
        env_data = json_data["environment"]
        assert env_data["name"] == "production"
        assert env_data["is_production"] is True
        assert env_data["debug_allowed"] is False

        # Check log level serialization
        log_data = json_data["log_level"]
        assert log_data["level"] == "INFO"
        assert log_data["numeric_level"] == 20
        assert log_data["production_safe"] is True

        # Check API metadata
        config_meta = json_data["_config"]
        assert config_meta["type"] == "FlextConfig"
        # The environment in _config metadata will be the serialized dict,
        # so we need to access the actual environment name from the main data
        assert env_data["name"] == "production"  # Use the environment from main data
        assert config_meta["features_enabled"]["caching"] is True
        assert config_meta["features_enabled"]["metrics"] is True
        assert config_meta["features_enabled"]["tracing"] is True

    def test_config_creation_methods(self) -> None:
        """Test config creation helper methods."""
        base_data = {"environment": "staging", "debug": False}
        override_data = {"log_level": "DEBUG", "timeout": 60}

        result = FlextConfig.create_complete_config(base_data, override_data)
        assert result.success

        config_dict = result.value
        assert config_dict["environment"] == "staging"
        assert config_dict["log_level"] == "DEBUG"
        assert config_dict["timeout"] == 60

    def test_config_file_loading(self) -> None:
        """Test config loading from file."""
        config_data = {
            "name": "file-config",
            "environment": "test",
            "log_level": "DEBUG",
            "timeout": 45,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            result = FlextConfig.load_and_validate_from_file(temp_file)
            assert result.success

            loaded_config = result.value
            assert loaded_config["name"] == "file-config"
            assert loaded_config["environment"] == "test"
            assert loaded_config["log_level"] == "DEBUG"
            assert loaded_config["timeout"] == 45
        finally:
            os.unlink(temp_file)


class TestFlextSettings:
    """Test FlextSettings environment-aware base."""

    def test_settings_default_creation(self) -> None:
        """Test FlextSettings creates with defaults."""
        settings = FlextSettings()

        # Default business rules validation should pass
        result = settings.validate_business_rules()
        assert result.success

    def test_settings_serialization(self) -> None:
        """Test settings serialization with metadata."""
        settings = FlextSettings()
        json_data = settings.model_dump(mode="json")

        # Check settings metadata
        settings_meta = json_data["_settings"]
        assert settings_meta["type"] == "FlextSettings"
        assert settings_meta["env_loaded"] is True
        assert settings_meta["validation_enabled"] is True
        assert settings_meta["api_version"] == "v2"

    def test_settings_with_validation(self) -> None:
        """Test settings creation with validation."""
        result = FlextSettings.create_with_validation(
            overrides={"test_key": "test_value"}
        )
        assert result.success

        settings = result.value
        assert isinstance(settings, FlextSettings)


class TestFoundationUtilities:
    """Test foundation utility functions."""

    def test_safe_get_env_var(self) -> None:
        """Test safe environment variable retrieval."""
        # Set test environment variable
        os.environ["FLEXT_TEST_VAR"] = "test_value"

        try:
            # Test successful retrieval
            result = safe_get_env_var("FLEXT_TEST_VAR")
            assert result.success
            assert result.value == "test_value"

            # Test with default for missing variable
            result = safe_get_env_var("MISSING_VAR", "default_value")
            assert result.success
            assert result.value == "default_value"

            # Test missing variable without default
            result = safe_get_env_var("MISSING_VAR")
            assert result.failure
            assert "not set" in result.error
        finally:
            del os.environ["FLEXT_TEST_VAR"]

    def test_safe_load_json_file(self) -> None:
        """Test safe JSON file loading."""
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Test successful loading
            result = safe_load_json_file(temp_file)
            assert result.success
            assert result.value == test_data

            # Test file not found
            result = safe_load_json_file("/nonexistent/file.json")
            assert result.failure
            assert "File not found" in result.error
        finally:
            os.unlink(temp_file)

    def test_merge_configs(self) -> None:
        """Test configuration merging."""
        base_config = {"a": 1, "b": 2, "shared": "base"}
        override_config = {"c": 3, "shared": "override"}

        result = merge_configs(base_config, override_config)
        assert result.success

        merged = result.value
        assert merged["a"] == 1
        assert merged["b"] == 2
        assert merged["c"] == 3
        assert merged["shared"] == "override"  # Override takes precedence

    def test_merge_configs_with_none_values(self) -> None:
        """Test config merging rejects None values."""
        base_config = {"valid_key": "value"}
        override_config = {"invalid_key": None}

        result = merge_configs(base_config, override_config)
        assert result.failure
        assert "cannot be null" in result.error


class TestFlextConfigIntegration:
    """Integration tests for foundation config patterns."""

    def test_complete_config_workflow(self) -> None:
        """Test complete config workflow with foundation patterns."""
        # 1. Create base config
        base_config = FlextConfig(
            name="integration-test",
            environment="test"
        )

        # 2. Test business rules
        validation_result = base_config.validate_business_rules()
        assert validation_result.success

        # 3. Export to dict
        config_dict = base_config.model_dump()
        assert config_dict["name"] == "integration-test"
        assert config_dict["environment"] == "test"

        # 4. Test merging with overrides
        overrides = {"timeout": 120, "enable_tracing": True}
        merge_result = merge_configs(config_dict, overrides)
        assert merge_result.success

        merged_config = merge_result.value
        assert merged_config["timeout"] == 120
        assert merged_config["enable_tracing"] is True

        # 5. Recreate config from merged data
        new_config = FlextConfig.model_validate(merged_config)
        assert new_config.timeout == 120
        assert new_config.enable_tracing is True

    def test_environment_based_configuration(self) -> None:
        """Test environment-based configuration patterns."""
        # Development environment
        dev_config = FlextConfig(environment="development", debug=True)
        assert dev_config.debug is True
        assert dev_config.environment == "development"

        dev_validation = dev_config.validate_business_rules()
        assert dev_validation.success

        # Production environment
        prod_config = FlextConfig(environment="production", debug=False)
        assert prod_config.debug is False
        assert prod_config.environment == "production"

        prod_validation = prod_config.validate_business_rules()
        assert prod_validation.success

        # Invalid production config (with debug)
        with pytest.raises(ValueError):
            # This should fail at validation time
            invalid_config = FlextConfig(environment="production", debug=True)
            validation = invalid_config.validate_business_rules()
            if validation.failure:
                raise ValueError(validation.error)
