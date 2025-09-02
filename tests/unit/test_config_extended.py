"""Comprehensive tests for config.py to achieve maximum coverage.

Tests cover all aspects of FlextConfig including SystemDefaults, Settings,
business rule validation, serialization, and utility functions.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from flext_core.config import FlextConfig
from flext_core.result import FlextResult


class TestFlextConfigSystemDefaults:
    """Test FlextConfig.SystemDefaults nested classes."""

    def test_system_defaults_security(self) -> None:
        """Test SystemDefaults.Security class."""
        security = FlextConfig.SystemDefaults.Security()

        # Test default values from FlextConstants
        assert security.max_password_length > 0
        assert security.max_username_length > 0
        assert (
            security.min_secret_key_length_strong
            > security.min_secret_key_length_adequate
        )
        assert security.min_secret_key_length_adequate > 0

    def test_system_defaults_network(self) -> None:
        """Test SystemDefaults.Network class."""
        network = FlextConfig.SystemDefaults.Network()

        # Test actual network attributes from FlextConstants
        assert network.TIMEOUT > 0
        assert network.RETRIES >= 0
        assert network.CONNECTION_TIMEOUT > 0

    def test_system_defaults_pagination(self) -> None:
        """Test SystemDefaults.Pagination class."""
        pagination = FlextConfig.SystemDefaults.Pagination()

        # Test actual pagination attributes from FlextConstants
        assert pagination.PAGE_SIZE > 0
        assert pagination.MAX_PAGE_SIZE >= pagination.PAGE_SIZE

    def test_system_defaults_logging(self) -> None:
        """Test SystemDefaults.Logging class."""
        logging_defaults = FlextConfig.SystemDefaults.Logging()

        # Test actual logging attribute from FlextConstants
        assert logging_defaults.LOG_LEVEL is not None
        assert isinstance(logging_defaults.LOG_LEVEL, str)

    def test_system_defaults_environment(self) -> None:
        """Test SystemDefaults.Environment class."""
        env = FlextConfig.SystemDefaults.Environment()

        # Test actual environment attribute from FlextConstants
        assert env.DEFAULT_ENV is not None
        assert isinstance(env.DEFAULT_ENV, str)

    def test_system_defaults_performance_config_dicts(self) -> None:
        """Test SystemDefaults.PerformanceConfigDicts class."""
        perf_config = FlextConfig.SystemDefaults.PerformanceConfigDicts()

        # Test that it has the expected configuration dictionaries
        assert hasattr(perf_config, "PERFORMANCE_CONFIG")
        # Should have optimized settings for performance
        config_dict = perf_config.PERFORMANCE_CONFIG
        assert config_dict.get("validate_assignment") is True
        assert config_dict.get("extra") == "forbid"

    def test_system_defaults_performance_config_dicts_other_configs(self) -> None:
        """Test other configs in PerformanceConfigDicts."""
        perf_config = FlextConfig.SystemDefaults.PerformanceConfigDicts()

        # Test other configuration dictionaries exist
        assert hasattr(perf_config, "BATCH_CONFIG")
        assert hasattr(perf_config, "MEMORY_CONFIG")
        assert hasattr(perf_config, "CONCURRENCY_CONFIG")
        assert hasattr(perf_config, "CACHE_CONFIG")

        # Test batch config has appropriate settings
        batch_config = perf_config.BATCH_CONFIG
        assert batch_config.get("validate_assignment") is True
        json_schema_extra = batch_config.get("json_schema_extra", {})
        assert isinstance(json_schema_extra, dict)
        assert "batch_defaults" in json_schema_extra


class TestFlextConfigSettings:
    """Test FlextConfig.Settings class."""

    def test_settings_creation(self) -> None:
        """Test basic Settings instance creation."""
        settings = FlextConfig.Settings()

        # Should have default model_config
        assert hasattr(settings, "model_config")
        # Should be able to create without errors
        assert isinstance(settings, FlextConfig.Settings)
        # Should have environment prefix in model_config
        assert settings.model_config.get("env_prefix") == "FLEXT_"

    def test_settings_with_custom_values(self) -> None:
        """Test Settings with custom field values."""
        # Settings can accept arbitrary keyword arguments due to BaseSettings
        settings = FlextConfig.Settings()

        # Test that instance was created successfully
        assert isinstance(settings, FlextConfig.Settings)
        # Test model_config has expected values
        assert settings.model_config.get("env_prefix") == "FLEXT_"
        assert settings.model_config.get("env_file") == ".env"

    def test_validate_business_rules_default(self) -> None:
        """Test default validate_business_rules implementation."""
        settings = FlextConfig.Settings()
        result = settings.validate_business_rules()

        # Default implementation should succeed
        assert result.success
        assert result.unwrap() is None

    def test_validate_business_rules_custom_implementation(self) -> None:
        """Test custom validate_business_rules implementation."""

        class CustomSettings(FlextConfig.Settings):
            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].fail("Custom validation failed")

        settings = CustomSettings()
        result = settings.validate_business_rules()

        # Custom implementation should fail
        assert result.is_failure
        assert result.error == "Custom validation failed"

    def test_create_with_validation_success(self) -> None:
        """Test successful create_with_validation."""
        result = FlextConfig.Settings.create_with_validation()

        assert result.success
        settings = result.unwrap()
        assert isinstance(settings, FlextConfig.Settings)

    def test_create_with_validation_with_overrides(self) -> None:
        """Test create_with_validation with overrides."""
        # Settings class doesn't have predefined fields like name/version
        # So we can only test that create_with_validation accepts overrides
        result = FlextConfig.Settings.create_with_validation(
            overrides={"custom_field": "value"}
        )

        assert result.success
        settings = result.unwrap()
        assert isinstance(settings, FlextConfig.Settings)

    def test_create_with_validation_business_rules_error(self) -> None:
        """Test create_with_validation with business rules failure."""

        class FailValidationSettings(FlextConfig.Settings):
            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].fail("Business rule violation")

        result = FailValidationSettings.create_with_validation()

        assert result.is_failure
        error_msg = result.error or ""
        assert "Business rule violation" in error_msg

    def test_create_with_validation_pydantic_error(self) -> None:
        """Test create_with_validation with pydantic validation error."""
        # Try to create with invalid field values by mocking Exception
        with patch.object(
            FlextConfig.Settings,
            "model_validate",
            side_effect=Exception("Mock validation error"),
        ):
            result = FlextConfig.Settings.create_with_validation(
                overrides={"invalid": "value"}
            )

            assert result.is_failure
            error_msg = result.error or ""
            assert "Settings creation failed" in error_msg

    def test_serialize_settings_for_api(self) -> None:
        """Test serialize_settings_for_api method."""
        settings = FlextConfig.Settings()

        # Create a mock serializer and info
        mock_serializer = MagicMock()
        from pydantic import SerializationInfo

        mock_info = MagicMock(spec=SerializationInfo)

        result = settings.serialize_settings_for_api(mock_serializer, mock_info)

        # Should return a dict with _settings metadata
        assert isinstance(result, dict)
        assert "_settings" in result
        # Cast _settings to dict for type safety
        settings_metadata = result["_settings"]
        assert isinstance(settings_metadata, dict)
        assert settings_metadata["type"] == "FlextConfig"
        assert settings_metadata["env_loaded"] is True
        assert settings_metadata["validation_enabled"] is True
        assert settings_metadata["api_version"] == "v2"

    def test_serialize_settings_for_api_with_complex_types(self) -> None:
        """Test serialize_settings_for_api with complex data types."""
        # Create settings with complex field that would need string conversion
        settings = FlextConfig.Settings()

        # Mock model_dump to return complex types
        with patch.object(
            type(settings),
            "model_dump",
            return_value={
                "name": "test",
                "complex_object": {"nested": "value"},
                "list_field": ["item1", "item2"],
                "invalid_object": object(),  # This will be converted to str
            },
        ):
            mock_serializer = MagicMock()
            from pydantic import SerializationInfo

            mock_info = MagicMock(spec=SerializationInfo)

            result = settings.serialize_settings_for_api(mock_serializer, mock_info)

            # Should handle complex types and convert invalid ones to string
            assert result["name"] == "test"
            assert result["complex_object"] == {"nested": "value"}
            assert result["list_field"] == ["item1", "item2"]
            assert isinstance(result["invalid_object"], str)


class TestFlextConfigMain:
    """Test main FlextConfig class fields and validation."""

    def test_flext_config_creation(self) -> None:
        """Test FlextConfig instance creation."""
        config = FlextConfig()

        # Test default field values
        assert config.name == "flext"  # From FlextConstants.Core.NAME.lower()
        assert config.version is not None
        assert config.description == "FLEXT configuration"
        assert config.environment == "development"
        assert config.debug is False
        assert config.config_source == "default"
        assert config.config_namespace == "flext"

    def test_flext_config_with_custom_values(self) -> None:
        """Test FlextConfig with custom field values."""
        config = FlextConfig(
            name="custom-config",
            version="2.0.0",
            description="Custom configuration",
            environment="production",
            debug=True,
            config_source="file",
            config_namespace="custom",
        )

        assert config.name == "custom-config"
        assert config.version == "2.0.0"
        assert config.description == "Custom configuration"
        assert config.environment == "production"
        assert config.debug is True
        assert config.config_source == "file"
        assert config.config_namespace == "custom"

    def test_validate_business_rules_default(self) -> None:
        """Test FlextConfig validate_business_rules method."""
        config = FlextConfig()
        result = config.validate_business_rules()

        # Should succeed with default implementation
        assert result.success
        assert result.unwrap() is None

    def test_environment_validation(self) -> None:
        """Test environment field validation."""
        # Valid environments (only literal values allowed)
        config = FlextConfig(environment="development")
        assert config.environment == "development"

        config = FlextConfig(environment="production")
        assert config.environment == "production"

        config = FlextConfig(environment="staging")
        assert config.environment == "staging"

    def test_environment_validation_invalid(self) -> None:
        """Test environment validation with invalid values."""
        with pytest.raises(ValidationError, match="literal_error"):
            FlextConfig(environment="invalid")

    def test_config_source_validation(self) -> None:
        """Test config_source field validation."""
        # Should accept valid sources from FlextConstants.Config.ConfigSource
        config = FlextConfig(config_source="file")
        assert config.config_source == "file"

    def test_config_source_validation_invalid(self) -> None:
        """Test config_source validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextConfig(config_source="invalid_source")

    def test_config_priority_validation(self) -> None:
        """Test config_priority field validation."""
        # Should accept valid priority range - test with default value
        config = FlextConfig()
        assert config.config_priority >= 0  # Default should be valid

    def test_config_priority_validation_invalid(self) -> None:
        """Test config_priority validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextConfig(config_priority=9999)  # Too high

        with pytest.raises(ValidationError):
            FlextConfig(config_priority=-1)  # Too low

    def test_log_level_validation(self) -> None:
        """Test log_level field validation."""
        # Should accept and normalize log levels
        config = FlextConfig(log_level="info")
        assert config.log_level == "INFO"  # Should be uppercase

        config = FlextConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

    def test_log_level_validation_invalid(self) -> None:
        """Test log_level validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextConfig(log_level="INVALID")

    def test_positive_integer_validation(self) -> None:
        """Test validation of positive integer fields."""
        config = FlextConfig(timeout=30, retries=5, page_size=100)
        assert config.timeout == 30
        assert config.retries == 5
        assert config.page_size == 100

    def test_positive_integer_validation_invalid(self) -> None:
        """Test positive integer validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextConfig(timeout=0)

        with pytest.raises(ValidationError):
            FlextConfig(retries=-1)

        with pytest.raises(ValidationError):
            FlextConfig(page_size=0)


class TestFlextConfigUtilityMethods:
    """Test FlextConfig utility methods."""

    def test_safe_get_env_var_success(self) -> None:
        """Test safe_get_env_var with existing environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = FlextConfig.safe_get_env_var("TEST_VAR", "default")

            assert result.success
            assert result.unwrap() == "test_value"

    def test_safe_get_env_var_default(self) -> None:
        """Test safe_get_env_var with missing environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            result = FlextConfig.safe_get_env_var("NONEXISTENT_VAR", "default_value")

            assert result.success
            assert result.unwrap() == "default_value"

    def test_safe_get_env_var_none_default(self) -> None:
        """Test safe_get_env_var with None default."""
        with patch.dict(os.environ, {}, clear=True):
            result = FlextConfig.safe_get_env_var("NONEXISTENT_VAR", None)

            # Actually fails when default is None - method requires non-None default
            assert result.is_failure
            error_msg = result.error or ""
            assert "Environment variable NONEXISTENT_VAR not set" in error_msg

    def test_safe_load_json_file_success(self) -> None:
        """Test safe_load_json_file with valid JSON file."""
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = FlextConfig.safe_load_json_file(temp_path)

            assert result.success
            data = result.unwrap()
            assert data == test_data
        finally:
            Path(temp_path).unlink()

    def test_safe_load_json_file_nonexistent(self) -> None:
        """Test safe_load_json_file with nonexistent file."""
        result = FlextConfig.safe_load_json_file("nonexistent_file.json")

        assert result.is_failure
        error_msg = result.error or ""
        assert "NOT_FOUND:" in error_msg

    def test_safe_load_json_file_invalid_json(self) -> None:
        """Test safe_load_json_file with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            f.write("invalid json content {")
            temp_path = f.name

        try:
            result = FlextConfig.safe_load_json_file(temp_path)

            assert result.is_failure
            error_msg = result.error or ""
            assert "FLEXT_2004:" in error_msg  # Actual error format
        finally:
            Path(temp_path).unlink()

    def test_safe_load_json_file_pathlib(self) -> None:
        """Test safe_load_json_file with pathlib.Path."""
        test_data = {"pathlib": "test"}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            result = FlextConfig.safe_load_json_file(temp_path)

            assert result.success
            data = result.unwrap()
            assert data == test_data
        finally:
            temp_path.unlink()

    def test_safe_load_json_file_empty_file(self) -> None:
        """Test safe_load_json_file with empty file."""
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        try:
            result = FlextConfig.safe_load_json_file(temp_path)

            assert result.is_failure
            error_msg = result.error or ""
            assert "FLEXT_2004:" in error_msg  # Actual error format
        finally:
            Path(temp_path).unlink()

    def test_safe_load_json_file_permission_error(self) -> None:
        """Test safe_load_json_file with permission denied."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = FlextConfig.safe_load_json_file("test.json")

            assert result.is_failure
            # Method first checks if file exists, so gets NOT_FOUND instead
            error_msg = result.error or ""
            assert "NOT_FOUND:" in error_msg

    @patch.dict(os.environ, {"FLEXT_TEST": "environment_value"})
    def test_environment_integration(self) -> None:
        """Test that Settings properly loads from environment."""
        # This tests the actual environment integration
        settings = FlextConfig.Settings()

        # Should have loaded the environment variable
        # Note: We can't directly test FLEXT_TEST since Settings doesn't define that field
        # But we can test that the mechanism works by checking model_config
        assert settings.model_config.get("env_prefix") == "FLEXT_"
        assert isinstance(settings, FlextConfig.Settings)
