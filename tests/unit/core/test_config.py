"""Comprehensive tests for FlextConfig and configuration functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
        config_data: dict[str, object] = {"app_name": "test", "custom_setting": "value"}

        result = FlextConfig.create_complete_config(config_data)

        assert result.success
        assert (result.data or {})["app_name"] == "test", (
            f"Expected {'test'}, got {(result.data or {})['app_name']}"
        )
        assert (result.data or {})["custom_setting"] == "value"

    def test_create_complete_config_with_defaults(self) -> None:
        """Test complete configuration creation with defaults applied."""
        config_data: dict[str, object] = {"app_name": "test"}

        result = FlextConfig.create_complete_config(
            config_data,
            apply_defaults=True,
        )

        assert result.success
        assert (result.data or {})["app_name"] == "test", (
            f"Expected {'test'}, got {(result.data or {})['app_name']}"
        )
        # Check that defaults were applied
        assert (
            "debug" in (result.data or {})
            or "timeout" in (result.data or {})
            or "port" in (result.data or {})
        )

    def test_create_complete_config_without_validation(self) -> None:
        """Test complete configuration creation without validation."""
        config_data: dict[str, object] = {"app_name": "test", "some_key": "valid_value"}

        result = FlextConfig.create_complete_config(
            config_data,
            validate_all=False,
        )

        assert result.success
        assert (result.data or {})["app_name"] == "test", (
            f"Expected {'test'}, got {(result.data or {})['app_name']}"
        )

    def test_create_complete_config_validation_failure(self) -> None:
        """Test complete configuration creation with validation failure."""
        config_data: dict[str, object] = {"app_name": "test", "invalid_key": None}

        result = FlextConfig.create_complete_config(
            config_data,
            validate_all=True,
        )

        # Should fail because of None value when validation is enabled
        assert result.is_failure
        assert "Config validation failed" in (result.error or ""), (
            f"Expected 'Config validation failed' in {result.error}"
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

            assert result.success
            assert (result.data or {})["app_name"] == "test", (
                f"Expected {'test'}, got {(result.data or {})['app_name']}"
            )
        finally:
            Path(config_path).unlink()

    def test_load_and_validate_from_file_not_found(self) -> None:
        """Test file loading with non-existent file."""
        result = FlextConfig.load_and_validate_from_file("non_existent.json")

        assert result.is_failure
        assert "Configuration file not found" in (result.error or ""), (
            f"Expected 'Configuration file not found' in {result.error}"
        )

    def test_merge_and_validate_configs(self) -> None:
        """Test configuration merging and validation."""
        config1 = {"app_name": "test", "debug": False}
        config2 = {"debug": True, "database_host": "localhost"}

        result = FlextConfig.merge_and_validate_configs(config1, config2)

        assert result.success
        assert (result.data or {})["app_name"] == "test", (
            f"Expected {'test'}, got {(result.data or {})['app_name']}"
        )
        assert (result.data or {})["debug"] is True  # config2 overrides config1
        assert (result.data or {})["database_host"] == "localhost", (
            f"Expected {'localhost'}, got {(result.data or {})['database_host']}"
        )

    def test_get_env_with_validation_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test environment variable retrieval with validation."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        result = FlextConfig.get_env_with_validation("TEST_VAR")

        assert result.success
        assert result.data == "test_value", (
            f"Expected {'test_value'}, got {result.data}"
        )

    def test_get_env_with_validation_not_found(self) -> None:
        """Test environment variable retrieval when not found."""
        result = FlextConfig.get_env_with_validation("NON_EXISTENT_VAR")

        assert result.is_failure
        assert "Environment variable 'NON_EXISTENT_VAR' not found" in (
            result.error or ""
        ), (
            f"Expected 'Environment variable \\'NON_EXISTENT_VAR\\' not found' in {result.error}"
        )


class TestFlextConfigDefaults:
    """Test FlextConfigDefaults functionality."""

    def test_apply_defaults_simple(self) -> None:
        """Test applying simple defaults."""
        config: dict[str, object] = {"app": {"name": "test"}}
        defaults: dict[str, object] = {
            "app": {"debug": False},
            "database": {"host": "localhost"},
        }

        result = FlextConfigDefaults.apply_defaults(config, defaults)

        assert result.success
        merged_config = result.data
        assert merged_config is not None

        # Type cast to access nested structure
        app_config = cast("dict[str, object]", merged_config["app"])
        assert app_config["name"] == "test", (
            f"Expected {'test'}, got {app_config['name']}"
        )

    def test_merge_configs_basic(self) -> None:
        """Test basic config merging."""
        config: dict[str, object] = {"existing": "value"}
        defaults: dict[str, object] = {"new": "default"}

        result = FlextConfigDefaults.merge_configs(config, defaults)

        assert result.success
        assert (result.data or {})["existing"] == "value", (
            f"Expected {'value'}, got {(result.data or {})['existing']}"
        )
        assert (result.data or {})["new"] == "default"


class TestFlextConfigOps:
    """Test FlextConfigOps functionality."""

    def test_safe_load_from_dict(self) -> None:
        """Test safe dictionary loading."""
        test_dict = {"key": "value", "number": 42}

        result = FlextConfigOps.safe_load_from_dict(test_dict)

        assert result.success
        assert result.data == test_dict, f"Expected {test_dict}, got {result.data}"


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

        assert result.success

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
        assert result.error is not None
        assert "Value must be string" in result.error


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
        assert settings.app_name == "test", (
            f"Expected {'test'}, got {settings.app_name}"
        )
        assert not settings.debug, f"Expected False, got {settings.debug}"

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
        assert settings.app_name == "custom", (
            f"Expected {'custom'}, got {settings.app_name}"
        )
        assert settings.port == 9000


class TestStandaloneFunctions:
    """Test standalone configuration functions."""

    def test_merge_configs_function(self) -> None:
        """Test standalone merge_configs function."""
        config1: dict[str, object] = {"app_name": "test", "database_host": "localhost"}
        config2: dict[str, object] = {"app_debug": True, "database_port": 5432}

        result = merge_configs(config1, config2)

        # This returns a dict, not FlextResult
        assert isinstance(result, dict)
        assert result["app_name"] == "test", (
            f"Expected {'test'}, got {result['app_name']}"
        )
        assert result["app_debug"], f"Expected True, got {result['app_debug']}"
        assert result["database_host"] == "localhost", (
            f"Expected {'localhost'}, got {result['database_host']}"
        )
        assert result["database_port"] == 5432

    def test_safe_get_env_var_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test safe environment variable retrieval when exists."""
        monkeypatch.setenv("TEST_ENV_VAR", "test_value")

        result = safe_get_env_var("TEST_ENV_VAR")

        assert result.success
        assert result.data == "test_value", (
            f"Expected {'test_value'}, got {result.data}"
        )

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

        assert result.success
        assert result.data == "default_value", (
            f"Expected {'default_value'}, got {result.data}"
        )

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

            assert result.success
            assert result.data == test_data, f"Expected {test_data}, got {result.data}"
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
            assert load_result.success
            assert load_result.data is not None

            # Create complete configuration
            complete_result = FlextConfig.create_complete_config(
                load_result.data,
                apply_defaults=True,
                validate_all=True,
            )

            assert complete_result.success
            final_config = complete_result.data
            assert final_config is not None

            # Validate final configuration - proper type guards for nested dict access
            app_config = cast("dict[str, object]", final_config["app"])
            assert app_config["name"] == "test_app", (
                f"Expected {'test_app'}, got {app_config['name']}"
            )

            database_config = cast("dict[str, object]", final_config["database"])
            assert database_config["host"] == "localhost"

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
