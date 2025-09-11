"""Comprehensive tests for FlextConfig.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from flext_core import FlextConfig, FlextResult, FlextUtilities
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextConfigComprehensive:
    """Comprehensive tests for FlextConfig class."""

    def test_config_initialization_defaults(self) -> None:
        """Test FlextConfig initialization with defaults."""
        config = FlextConfig()

        # Check default values
        assert config.app_name == "flext-app"
        assert config.environment == "development"
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.config_source == "default"
        assert config.config_priority == 5
        assert config.max_workers == 4
        assert config.timeout_seconds == 30
        assert config.enable_metrics is True
        assert config.enable_caching is True

    def test_config_initialization_with_values(self) -> None:
        """Test FlextConfig initialization with custom values."""
        config = FlextConfig(
            app_name="test-app",
            environment="development",  # Use development to allow debug=True
            debug=True,
            log_level="INFO",
            max_workers=8,
            timeout_seconds=60,
        )

        assert config.app_name == "test-app"
        assert config.environment == "development"
        assert config.debug is True
        assert config.log_level == "INFO"
        assert config.max_workers == 8
        assert config.timeout_seconds == 60

    def test_validate_environment_valid_values(self) -> None:
        """Test environment validation with valid values."""
        valid_environments: list[
            Literal["development", "production", "test", "staging"]
        ] = ["development", "production", "test", "staging"]

        for env in valid_environments:
            config = FlextConfig(environment=env)
            assert config.environment == env

    def test_validate_environment_invalid_value(self) -> None:
        """Test environment validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig(environment="invalid_env")
        msg = str(exc_info.value)
        assert "Environment must be one of" in msg or "Input should be" in msg

    def test_validate_log_level_valid_values(self) -> None:
        """Test log level validation with valid values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            config = FlextConfig(log_level=level)
            assert config.log_level == level

    def test_validate_log_level_invalid_value(self) -> None:
        """Test log level validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig(log_level="INVALID")

        assert "Log level must be one of" in str(exc_info.value)

    def test_validate_positive_integers_valid_values(self) -> None:
        """Test positive integer validation with valid values."""
        config = FlextConfig(max_workers=10, timeout_seconds=120, config_priority=5)
        assert config.max_workers == 10
        assert config.timeout_seconds == 120
        assert config.config_priority == 5

    def test_validate_positive_integers_invalid_values(self) -> None:
        """Test positive integer validation with invalid values."""
        with pytest.raises(ValidationError):
            FlextConfig(max_workers=0)

        with pytest.raises(ValidationError):
            FlextConfig(timeout_seconds=-1)

        with pytest.raises(ValidationError):
            FlextConfig(config_priority=-5)

    def test_validate_config_source_valid_values(self) -> None:
        """Test config source validation with valid values."""
        # Using actual values from FlextConstants.Config.ConfigSource
        valid_sources = ["file", "env", "cli", "default", "dotenv", "yaml", "json"]

        for source in valid_sources:
            config = FlextConfig(config_source=source)
            assert config.config_source == source

    def test_validate_config_source_invalid_value(self) -> None:
        """Test config source validation with invalid value."""
        with pytest.raises(ValidationError, match="Config source must be one of"):
            FlextConfig(config_source="invalid_source")

    def test_validate_business_rules_success(self) -> None:
        """Test business rules validation success."""
        config = FlextConfig()
        config.environment = "production"
        config.debug = False
        config.max_workers = 8
        config.timeout_seconds = 60
        result = config.validate_business_rules()
        assert result.success

    def test_validate_business_rules_debug_in_production(self) -> None:
        """Regra removida: base aceita debug em production."""
        result = FlextConfig.create(
            constants={"environment": "production", "debug": True}
        )
        assert result.success
        cfg = result.unwrap()
        rules = cfg.validate_business_rules()
        assert rules.success

    def test_validate_business_rules_high_timeout_low_workers(self) -> None:
        """Regra removida: base não impõe combinação timeout/workers."""
        result = FlextConfig.create(
            constants={"timeout_seconds": 120, "max_workers": 1}
        )
        assert result.success
        rules = result.unwrap().validate_business_rules()
        assert rules.success

    def test_environment_basic_value(self) -> None:
        result = FlextConfig.create(constants={"environment": "production"})
        assert result.success
        assert result.unwrap().environment == "production"

    def test_log_level_basic_normalization(self) -> None:
        result = FlextConfig.create(constants={"log_level": "info"})
        assert result.success
        assert result.unwrap().log_level == "INFO"

    def test_as_api_payload(self) -> None:
        result = FlextConfig.create(
            constants={
                "app_name": "test-app",
                "environment": "development",
                "debug": True,
            }
        )
        assert result.success
        cfg = result.unwrap()
        result = cfg.as_api_payload()
        assert result.success
        assert result.unwrap() == {
            "app_name": "test-app",
            "environment": "development",
            "debug": True,
        }

    def test_create_complete_config_success(self) -> None:
        """Test creating complete configuration using create method."""
        base_config = {"app_name": "base-app"}
        override_config = {"environment": "production", "debug": False}

        # Use declarative create method
        result = FlextConfig.create(
            constants=base_config, cli_overrides=override_config
        )
        assert result.success

        config = result.unwrap()
        assert config.app_name == "base-app"
        assert config.environment == "production"
        assert config.debug is False

    def test_create_complete_config_validation_error(self) -> None:
        """Test create complete config with validation error."""
        base_config: FlextTypes.Core.Dict = {"environment": "invalid_env"}

        # Use declarative create method
        result = FlextConfig.create(constants=base_config)
        assert result.is_failure
        assert result.error is not None
        assert "Configuration creation failed" in (result.error or "")

    def test_load_and_validate_from_file_success(self) -> None:
        """Test loading and validating from file using create method."""
        config_data = {
            "app_name": "file-app",
            "environment": "test",
            "debug": True,
            "log_level": "DEBUG",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Use declarative create method
            result = FlextConfig.create(env_file=temp_path)
            assert result.success

            config = result.unwrap()
            assert isinstance(config, FlextConfig)
        finally:
            Path(temp_path).unlink()

    def test_load_and_validate_from_file_not_found(self) -> None:
        """Test load from non-existent file using create method."""
        # Use declarative create method
        result = FlextConfig.create(env_file="non_existent_file.json")
        assert result.is_failure
        assert result.error is not None

    def test_load_and_validate_from_file_invalid_json(self) -> None:
        """Test loading from file with invalid env file content."""
        import os

        # Save original environment and clear FLEXT_ENVIRONMENT for test isolation
        original_env = os.environ.get("FLEXT_ENVIRONMENT")
        if "FLEXT_ENVIRONMENT" in os.environ:
            del os.environ["FLEXT_ENVIRONMENT"]

        try:
            with tempfile.NamedTemporaryFile(
                encoding="utf-8",
                mode="w",
                suffix=".env",
                delete=False,
            ) as f:
                # Write content that causes Pydantic validation failure
                f.write("FLEXT_ENVIRONMENT=invalid_environment_value\n")
                temp_path = f.name

            try:
                # Use declarative create method
                result = FlextConfig.create(env_file=temp_path)
                assert result.is_failure
                assert result.error is not None
                # The error should mention invalid environment
                assert "Invalid environment" in (result.error or "")
            finally:
                Path(temp_path).unlink()
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["FLEXT_ENVIRONMENT"] = original_env

    def test_safe_load_from_dict_success(self) -> None:
        """Test safe loading from dictionary."""
        config_dict = {
            "app_name": "dict-app",
            "environment": "staging",
            "max_workers": 6,
        }

        # Use safe_load_from_dict method
        result = FlextConfig.create(constants=config_dict)
        assert result.success
        config = result.unwrap()
        assert config.app_name == "dict-app"
        assert config.environment == "staging"

    def test_safe_load_from_dict_validation_error(self) -> None:
        """Test safe loading from dict with validation error."""
        invalid_dict: FlextTypes.Core.Dict = {"environment": "invalid_env"}

        # Use safe_load_from_dict method
        result = FlextConfig.create(constants=invalid_dict)
        assert result.is_failure

    def test_merge_and_validate_configs_success(self) -> None:
        """Test merging and validating two configs using create method."""
        base_config = {"app_name": "base-app", "environment": "development"}
        override_config = {
            "environment": "production",
            "debug": False,
            "max_workers": 8,
        }

        # Use declarative create method
        result = FlextConfig.create(
            constants=base_config, cli_overrides=override_config
        )
        assert result.success

        config = result.unwrap()
        assert config.app_name == "base-app"
        assert config.environment == "production"
        assert config.debug is False

    def test_merge_and_validate_configs_validation_error(self) -> None:
        """Test merging configs with validation error using create method."""
        config1 = {"environment": "development"}
        config2 = {"environment": "invalid_env"}

        # Use declarative create method
        result = FlextConfig.create(constants=config1, cli_overrides=config2)
        assert result.is_failure
        assert result.error is not None

    def test_get_env_with_validation_success(self) -> None:
        """Test getting environment variable with validation."""
        # Set up test environment variable
        os.environ["TEST_CONFIG_VAR"] = "test_value"

        try:
            # Use FlextUtilities directly instead of removed method
            result = FlextUtilities.EnvironmentUtils.safe_get_env_var(
                "TEST_CONFIG_VAR", "default"
            )
            assert result.success
            assert result.value == "test_value"
        finally:
            if "TEST_CONFIG_VAR" in os.environ:
                del os.environ["TEST_CONFIG_VAR"]

    def test_get_env_with_validation_not_found(self) -> None:
        """Test getting non-existent environment variable."""
        # Use FlextConfig.get_env_var method
        result = FlextConfig.get_env_var("NON_EXISTENT_VAR")
        assert result.is_failure
        assert "not found" in (result.error or "")

    def test_get_env_with_validation_validator_failure(self) -> None:
        """Test getting environment variable with type validation failure."""
        # Test is no longer relevant since the method was removed
        # Just test basic env var retrieval
        result = FlextUtilities.EnvironmentUtils.safe_get_env_var(
            "NON_EXISTENT_VAR", "default_value"
        )
        assert result.success
        assert result.value == "default_value"

    def test_validate_config_value_success(self) -> None:
        """Test config value validation success using direct validation."""
        # Use validate_config_value static method
        result = FlextConfig.validate_config_value("test_value", str)
        assert result.success
        assert result.unwrap() is True

    def test_validate_config_value_type_error(self) -> None:
        """Test config value validation type error using direct validation."""
        # Use validate_config_value static method
        result = FlextConfig.validate_config_value(123, str)
        assert result.success
        assert result.unwrap() is False

    def test_get_model_config(self) -> None:
        """Test getting model configuration using direct access."""
        config = FlextConfig()
        # Use direct access to model_config
        model_config = config.model_config

        assert model_config is not None
        assert "validate_assignment" in model_config
        assert model_config["validate_assignment"] is True

    def test_system_defaults_class(self) -> None:
        """Test system defaults - check default values."""
        # Test that we can create a config with defaults
        config = FlextConfig()
        assert config.app_name == "flext-app"  # Default value
        assert config.environment == "development"  # Default value


class TestFlextConfigFunctionality:
    """Test standalone config functions."""

    def test_get_env_var_success(self) -> None:
        """Test getting environment variable."""
        os.environ["TEST_GET_VAR"] = "test_value"

        try:
            result = FlextConfig.get_env_var("TEST_GET_VAR")
            assert result.success
            assert result.unwrap() == "test_value"
        finally:
            if "TEST_GET_VAR" in os.environ:
                del os.environ["TEST_GET_VAR"]

    def test_get_env_var_not_found(self) -> None:
        """Test getting non-existent environment variable."""
        result = FlextConfig.get_env_var("NON_EXISTENT_GET_VAR")
        assert result.is_failure
        assert result.error is not None
        assert "Environment variable NON_EXISTENT_GET_VAR not found" in (
            result.error or ""
        )

    def test_load_json_file_success(self) -> None:
        """Test loading JSON file."""
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result: FlextResult[FlextTypes.Core.Dict] = (
                FlextUtilities.EnvironmentUtils.safe_load_json_file(temp_path)
            )
            assert result.success

            data = result.unwrap()
            assert data["key"] == "value"
            assert data["number"] == 42
        finally:
            Path(temp_path).unlink()

    def test_load_json_file_not_found(self) -> None:
        """Test loading non-existent JSON file."""
        result = FlextUtilities.EnvironmentUtils.safe_load_json_file(
            "non_existent.json"
        )
        assert result.is_failure
        assert result.error is not None
        assert "NOT_FOUND:" in (result.error or "")

    def test_load_json_file_invalid_json(self) -> None:
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            f.write("invalid json")
            temp_path = f.name

        try:
            result = FlextUtilities.EnvironmentUtils.safe_load_json_file(temp_path)
            assert result.is_failure
            assert result.error is not None
            assert "FLEXT_2004:" in (result.error or "")
        finally:
            Path(temp_path).unlink()

    def test_merge_config_dicts_success(self) -> None:
        """Test merging configuration dictionaries."""
        dict1: FlextTypes.Core.Dict = {"a": 1, "b": 2}
        dict2: FlextTypes.Core.Dict = {"b": 3, "c": 4}
        dict3: FlextTypes.Core.Dict = {"c": 5, "d": 6}

        # First merge dict1 and dict2
        intermediate_result = FlextConfig.merge_configs(dict1, dict2)
        assert intermediate_result.is_success
        # Then merge with dict3
        result = FlextConfig.merge_configs(intermediate_result.value, dict3)
        assert result.success

        merged = result.unwrap()
        assert merged["a"] == 1
        assert merged["b"] == 3  # Later values override
        assert merged["c"] == 5  # Later values override
        assert merged["d"] == 6

    def test_merge_config_dicts_empty_list(self) -> None:
        """Test merging empty dictionaries."""
        result = FlextConfig.merge_configs({}, {})
        assert result.success
        assert result.unwrap() == {}

    def test_create_settings_success(self) -> None:
        """Test creating settings."""
        config_dict = {"app_name": "settings-app", "environment": "test", "debug": True}

        # Use declarative create method with constants parameter
        result = FlextConfig.create(constants=config_dict)
        assert result.success
        config = result.unwrap()
        assert config.app_name == "settings-app"
        assert config.environment == "test"
        assert config.debug is True

    def test_create_settings_validation_error(self) -> None:
        """Test creating settings with validation error."""
        # Use declarative create with invalid data
        result = FlextConfig.create(constants={"environment": "invalid_env"})
        assert result.is_failure  # Should fail validation

    def test_create_validated_settings_success(self) -> None:
        """Test creating validated settings."""
        config_dict = {
            "app_name": "validated-app",
            "environment": "production",
            "debug": False,
        }

        # Use declarative create method with constants parameter
        result = FlextConfig.create(constants=config_dict)
        assert result.success
        config = result.unwrap()
        assert config.app_name == "validated-app"
        assert config.environment == "production"
        assert config.debug is False

    def test_create_validated_settings_debug_in_production_allowed(self) -> None:
        """Regra de negócio removida: deve aceitar debug em production."""
        result = FlextConfig.create(
            constants={"environment": "production", "debug": True}
        )
        assert result.success

    def test_safe_get_env_var_success(self) -> None:
        """Test safely getting environment variable."""
        os.environ["SAFE_TEST_VAR"] = "safe_value"

        try:
            result = FlextUtilities.EnvironmentUtils.safe_get_env_var(
                "SAFE_TEST_VAR", "default"
            )
            assert result.success
            assert result.unwrap() == "safe_value"
        finally:
            if "SAFE_TEST_VAR" in os.environ:
                del os.environ["SAFE_TEST_VAR"]

    def test_safe_get_env_var_with_default(self) -> None:
        """Test safely getting environment variable with default."""
        result = FlextUtilities.EnvironmentUtils.safe_get_env_var(
            "NON_EXISTENT_SAFE_VAR", "default_value"
        )
        assert result.success
        assert result.unwrap() == "default_value"

    def test_safe_load_json_file_success(self) -> None:
        """Test safely loading JSON file."""
        test_data = {"safe": True, "data": 123}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = FlextUtilities.EnvironmentUtils.safe_load_json_file(temp_path)
            assert result.success

            data = result.unwrap()
            assert data["safe"] is True
            assert data["data"] == 123
        finally:
            Path(temp_path).unlink()

    def test_safe_load_json_file_path_object(self) -> None:
        """Test safely loading JSON file with Path object."""
        test_data = {"path_test": True}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            result = FlextUtilities.EnvironmentUtils.safe_load_json_file(temp_path)
            assert result.success
            assert result.unwrap()["path_test"] is True
        finally:
            temp_path.unlink()

    def test_safe_load_json_file_not_found(self) -> None:
        """Test safely loading non-existent JSON file."""
        result = FlextUtilities.EnvironmentUtils.safe_load_json_file(
            "safe_non_existent.json"
        )
        assert result.is_failure
        assert result.error is not None
        assert "NOT_FOUND:" in (result.error or "")

    def test_merge_configs_success(self) -> None:
        """Test merging configurations."""
        config1_data = {"app_name": "merge1"}
        config2_data = {"app_name": "merge2", "debug": True}

        # Use declarative create for individual configs
        result1 = FlextConfig.create(constants=config1_data)
        result2 = FlextConfig.create(constants=config2_data)

        assert result1.success
        assert result2.success

        config1 = result1.unwrap()
        config2 = result2.unwrap()

        # Use to_dict() method instead of model_dump()
        result = FlextConfig.merge_configs(config1.to_dict(), config2.to_dict())
        assert result.success

        merged = result.unwrap()
        assert merged["app_name"] == "merge2"  # Later config overrides
        assert merged["environment"] == "test"  # From .env file
        assert merged["debug"] is True  # From second config


class TestConfigEdgeCases:
    """Test edge cases and error scenarios."""

    def test_config_with_none_values(self) -> None:
        """Test config handling with None values."""
        # Test that None values are handled appropriately
        config = FlextConfig()

        # These should use defaults
        assert config.app_name is not None
        assert config.environment is not None

    def test_config_serialization_exception(self) -> None:
        """Test config serialization com API payload mínimo."""
        config = FlextConfig()
        result = config.as_api_payload()
        assert result.success
        serialized = result.unwrap()
        assert isinstance(serialized, dict)
        assert serialized["app_name"] == config.app_name

    def test_file_operations_with_permissions_error(self) -> None:
        """Test file operations with permission errors."""
        # Create a directory instead of file to cause permission error
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "directory_as_file"
            Path(invalid_path).mkdir(parents=True)

            result = FlextUtilities.EnvironmentUtils.safe_load_json_file(invalid_path)
            assert result.is_failure
            assert result.error is not None
            assert "CONFIG_ERROR:" in (result.error or "")

    def test_environment_variable_with_unicode(self) -> None:
        """Test environment variables with unicode characters."""
        unicode_value = "test_ñíçódé_value"
        os.environ["UNICODE_TEST_VAR"] = unicode_value

        try:
            result = FlextConfig.get_env_var("UNICODE_TEST_VAR")
            assert result.success
            assert result.unwrap() == unicode_value
        finally:
            if "UNICODE_TEST_VAR" in os.environ:
                del os.environ["UNICODE_TEST_VAR"]

    def test_large_config_file(self) -> None:
        """Test loading large configuration file."""
        # Create a large config file
        large_config = {f"key_{i}": f"value_{i}" for i in range(1000)}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(large_config, f)
            temp_path = f.name

        try:
            result = FlextUtilities.EnvironmentUtils.safe_load_json_file(temp_path)
            assert result.success

            data = result.unwrap()
            assert len(data) == 1000
            assert data["key_999"] == "value_999"
        finally:
            Path(temp_path).unlink()

    def test_config_validation_with_custom_types(self) -> None:
        """Test config validation with custom type scenarios."""
        # Test with various data types
        result = FlextConfig.create(
            constants={
                "app_name": "complex-app",
                "environment": "development",
                "debug": True,
                "max_workers": 8,
                "timeout_seconds": 120,
            }
        )
        assert result.success
        cfg = result.unwrap()
        assert cfg.app_name == "complex-app"

    def test_validate_type_value_exception_handling(self) -> None:
        """Test validation with exception in type validation."""
        # Normal validation should succeed
        result = FlextConfig.validate_config_value("test", str)
        assert result.success
        assert result.unwrap() is True

    def test_merge_configs_with_empty_list(self) -> None:
        """Test merging empty list of configs."""
        result = FlextConfig.merge_configs({}, {})
        assert result.success

        # Should return a default config
        merged = result.unwrap()
        assert isinstance(merged, dict)
        # Empty merge should result in empty dict

    def test_config_with_extreme_values(self) -> None:
        """Campos extremos removidos do núcleo: teste mantido mínimo."""
        result = FlextConfig.create(
            constants={"environment": "production", "debug": True}
        )
        assert result.success
