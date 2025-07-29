"""Comprehensive tests for FlextConfig and configuration functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_settings import SettingsConfigDict

from flext_core.config import (
    FlextBaseSettings,
    FlextConfig,
    FlextConfigDefaults,
    FlextConfigOps,
    FlextConfigValidation,
    merge_configs,
    safe_get_env_var,
    safe_load_json_file,
)

if TYPE_CHECKING:
    import pytest


class TestFlextConfig:
    """Test FlextConfig composition-based configuration management."""

    def test_create_complete_config_basic(self) -> None:
        """Test basic complete configuration creation."""
        config_data = {"app_name": "test", "custom_setting": "value"}

        result = FlextConfig.create_complete_config(config_data)

        assert result.is_success
        if result.data["app_name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {result.data['app_name']}")
        assert result.data["custom_setting"] == "value"

    def test_create_complete_config_with_defaults(self) -> None:
        """Test complete configuration creation with defaults applied."""
        config_data = {"app_name": "test"}

        result = FlextConfig.create_complete_config(
            config_data,
            apply_defaults=True,
        )

        assert result.is_success
        if result.data["app_name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {result.data['app_name']}")
        # Check that defaults were applied
        assert (
            "debug" in result.data or "timeout" in result.data or "port" in result.data
        )

    def test_create_complete_config_without_validation(self) -> None:
        """Test complete configuration creation without validation."""
        config_data = {"app_name": "test", "some_key": "valid_value"}

        result = FlextConfig.create_complete_config(
            config_data,
            validate_all=False,
        )

        assert result.is_success
        if result.data["app_name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {result.data['app_name']}")

    def test_create_complete_config_validation_failure(self) -> None:
        """Test complete configuration creation with validation failure."""
        config_data = {"app_name": "test", "invalid_key": None}

        result = FlextConfig.create_complete_config(
            config_data,
            validate_all=True,
        )

        # Should fail because of None value when validation is enabled
        assert result.is_failure
        if "Config validation failed" not in result.error:
            raise AssertionError(
                f"Expected {'Config validation failed'} in {result.error}"
            )

    def test_load_and_validate_from_file_success(self) -> None:
        """Test successful file loading and validation."""
        test_config = {"app_name": "test", "debug": False}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(test_config, f)
            config_path = f.name

        try:
            result = FlextConfig.load_and_validate_from_file(config_path)

            assert result.is_success
            if result.data["app_name"] != "test":
                raise AssertionError(
                    f"Expected {'test'}, got {result.data['app_name']}"
                )
        finally:
            Path(config_path).unlink()

    def test_load_and_validate_from_file_not_found(self) -> None:
        """Test file loading with non-existent file."""
        result = FlextConfig.load_and_validate_from_file("non_existent.json")

        assert result.is_failure
        if "Configuration file not found" not in result.error:
            raise AssertionError(
                f"Expected {'Configuration file not found'} in {result.error}"
            )

    def test_merge_and_validate_configs(self) -> None:
        """Test configuration merging and validation."""
        config1 = {"app_name": "test", "debug": False}
        config2 = {"debug": True, "database_host": "localhost"}

        result = FlextConfig.merge_and_validate_configs(config1, config2)

        assert result.is_success
        if result.data["app_name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {result.data['app_name']}")
        assert result.data["debug"] is True  # config2 overrides config1
        if result.data["database_host"] != "localhost":
            raise AssertionError(
                f"Expected {'localhost'}, got {result.data['database_host']}"
            )

    def test_get_env_with_validation_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test environment variable retrieval with validation."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        result = FlextConfig.get_env_with_validation("TEST_VAR")

        assert result.is_success
        if result.data != "test_value":
            raise AssertionError(f"Expected {'test_value'}, got {result.data}")

    def test_get_env_with_validation_not_found(self) -> None:
        """Test environment variable retrieval when not found."""
        result = FlextConfig.get_env_with_validation("NON_EXISTENT_VAR")

        assert result.is_failure
        if "Environment variable 'NON_EXISTENT_VAR' not found" not in result.error:
            raise AssertionError(
                f"Expected {"Environment variable 'NON_EXISTENT_VAR' not found"} in {result.error}"
            )


class TestFlextConfigDefaults:
    """Test FlextConfigDefaults functionality."""

    def test_apply_defaults_simple(self) -> None:
        """Test applying simple defaults."""
        config = {"app": {"name": "test"}}
        defaults = {"app": {"debug": False}, "database": {"host": "localhost"}}

        result = FlextConfigDefaults.apply_defaults(config, defaults)

        assert result.is_success
        merged_config = result.data
        if merged_config["app"]["name"] != "test":
            raise AssertionError(
                f"Expected {'test'}, got {merged_config['app']['name']}"
            )

    def test_merge_configs_basic(self) -> None:
        """Test basic config merging."""
        config = {"existing": "value"}
        defaults = {"new": "default"}

        result = FlextConfigDefaults.merge_configs(config, defaults)

        assert result.is_success
        if result.data["existing"] != "value":
            raise AssertionError(f"Expected {'value'}, got {result.data['existing']}")
        assert result.data["new"] == "default"


class TestFlextConfigOps:
    """Test FlextConfigOps functionality."""

    def test_safe_load_from_dict(self) -> None:
        """Test safe dictionary loading."""
        test_dict = {"key": "value", "number": 42}

        result = FlextConfigOps.safe_load_from_dict(test_dict)

        assert result.is_success
        if result.data != test_dict:
            raise AssertionError(f"Expected {test_dict}, got {result.data}")


class TestFlextConfigValidation:
    """Test FlextConfigValidation functionality."""

    def test_validate_config_value_success(self) -> None:
        """Test successful config value validation."""
        value = "test_value"

        def validator(x: object) -> bool:
            return isinstance(x, str)

        result = FlextConfigValidation.validate_config_value(
            value,
            validator,
            "Value must be string",
        )

        assert result.is_success

    def test_validate_config_value_failure(self) -> None:
        """Test config value validation failure."""
        value = 123

        def validator(x: object) -> bool:
            return isinstance(x, str)

        result = FlextConfigValidation.validate_config_value(
            value,
            validator,
            "Value must be string",
        )

        assert result.is_failure
        if "Value must be string" not in result.error:
            raise AssertionError(f"Expected {'Value must be string'} in {result.error}")


class TestFlextBaseSettings:
    """Test FlextBaseSettings Pydantic integration."""

    def test_base_settings_creation(self) -> None:
        """Test FlextBaseSettings creation."""

        class TestSettings(FlextBaseSettings):
            model_config = SettingsConfigDict(
                env_file=None,  # Don't read any .env file
                extra="ignore",  # Ignore extra environment variables
                case_sensitive=False,
                validate_assignment=True,
            )

            app_name: str = "test"
            debug: bool = False

        settings = TestSettings()
        if settings.app_name != "test":
            raise AssertionError(f"Expected {'test'}, got {settings.app_name}")
        if settings.debug:
            raise AssertionError(f"Expected False, got {settings.debug}")

    def test_base_settings_with_values(self) -> None:
        """Test FlextBaseSettings with explicit values."""

        class TestSettings(FlextBaseSettings):
            model_config = SettingsConfigDict(
                env_file=None,  # Don't read any .env file
                extra="ignore",  # Ignore extra environment variables
                case_sensitive=False,
                validate_assignment=True,
            )

            app_name: str = "test"
            port: int = 8000

        settings = TestSettings(app_name="custom", port=9000)
        if settings.app_name != "custom":
            raise AssertionError(f"Expected {'custom'}, got {settings.app_name}")
        assert settings.port == 9000


class TestStandaloneFunctions:
    """Test standalone configuration functions."""

    def test_merge_configs_function(self) -> None:
        """Test standalone merge_configs function."""
        config1 = {"app_name": "test", "database_host": "localhost"}
        config2 = {"app_debug": True, "database_port": 5432}

        result = merge_configs(config1, config2)

        # This returns a dict, not FlextResult
        assert isinstance(result, dict)
        if result["app_name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {result['app_name']}")
        if not (result["app_debug"]):
            raise AssertionError(f"Expected True, got {result['app_debug']}")
        if result["database_host"] != "localhost":
            raise AssertionError(
                f"Expected {'localhost'}, got {result['database_host']}"
            )
        assert result["database_port"] == 5432

    def test_safe_get_env_var_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test safe environment variable retrieval when exists."""
        monkeypatch.setenv("TEST_ENV_VAR", "test_value")

        result = safe_get_env_var("TEST_ENV_VAR")

        assert result.is_success
        if result.data != "test_value":
            raise AssertionError(f"Expected {'test_value'}, got {result.data}")

    def test_safe_get_env_var_not_found(self) -> None:
        """Test safe environment variable retrieval when not found."""
        result = safe_get_env_var("NON_EXISTENT_VAR")

        assert result.is_failure

    def test_safe_get_env_var_with_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test safe environment variable retrieval with default."""
        # Make sure the var doesn't exist
        monkeypatch.delenv("NON_EXISTENT_VAR", raising=False)

        result = safe_get_env_var("NON_EXISTENT_VAR", "default_value")

        assert result.is_success
        if result.data != "default_value":
            raise AssertionError(f"Expected {'default_value'}, got {result.data}")

    def test_safe_load_json_file_success(self) -> None:
        """Test successful JSON file loading."""
        test_data = {"test": "value", "number": 42}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(test_data, f)
            file_path = f.name

        try:
            result = safe_load_json_file(file_path)

            assert result.is_success
            if result.data != test_data:
                raise AssertionError(f"Expected {test_data}, got {result.data}")
        finally:
            Path(file_path).unlink()

    def test_safe_load_json_file_not_found(self) -> None:
        """Test JSON file loading with non-existent file."""
        result = safe_load_json_file("non_existent.json")

        assert result.is_failure

    def test_safe_load_json_file_invalid_json(self) -> None:
        """Test JSON file loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            file_path = f.name

        try:
            result = safe_load_json_file(file_path)

            assert result.is_failure
        finally:
            Path(file_path).unlink()


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_configuration_workflow(self) -> None:
        """Test complete configuration workflow."""
        # Create test config file
        test_config = {
            "app": {"name": "test_app"},
            "database": {"host": "localhost"},
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(test_config, f)
            config_path = f.name

        try:
            # Load configuration
            load_result = safe_load_json_file(config_path)
            assert load_result.is_success

            # Create complete configuration
            complete_result = FlextConfig.create_complete_config(
                load_result.data,
                apply_defaults=True,
                validate_all=True,
            )

            assert complete_result.is_success
            final_config = complete_result.data

            # Validate final configuration
            if final_config["app"]["name"] != "test_app":
                raise AssertionError(
                    f"Expected {'test_app'}, got {final_config['app']['name']}"
                )
            assert final_config["database"]["host"] == "localhost"

        finally:
            Path(config_path).unlink()

    def test_configuration_error_handling(self) -> None:
        """Test configuration error handling throughout the system."""
        # Test file loading error
        load_result = safe_load_json_file("non_existent.json")
        assert load_result.is_failure

        # Test validation error
        validation_result = FlextConfig.create_complete_config(
            {"invalid_key": None},
            validate_all=True,
        )
        assert validation_result.is_failure

        # Test environment variable error
        env_result = safe_get_env_var("NON_EXISTENT_ENV_VAR")
        assert env_result.is_failure
