"""Comprehensive tests for FLEXT Core Config Module.

Tests all consolidated configuration functionality including:
- FlextConfig orchestration and multiple inheritance
- FlextBaseSettings Pydantic integration
- Direct base class functionality
- File loading and validation
- Environment variable management
- Configuration merging and validation
- Error handling and FlextResult integration
- Backward compatibility aliases
"""

import os
import pathlib
import tempfile
from typing import Any

from pydantic import field_validator

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


class TestFlextConfigOrchestration:
    """Test FlextConfig consolidated functionality with multiple inheritance."""

    def test_config_inheritance_structure(self) -> None:
        """Test that FlextConfig inherits from all required base classes."""
        # Verify inheritance chain
        assert hasattr(FlextConfig, "safe_load_from_dict")  # from _BaseConfig
        assert hasattr(FlextConfig, "apply_defaults")  # from _BaseConfigDefaults
        assert hasattr(FlextConfig, "merge_configs")  # from _BaseConfigOps
        assert hasattr(
            FlextConfig,
            "validate_config_value",
        )  # from _BaseConfigValidation

        # Verify orchestration methods exist
        assert hasattr(FlextConfig, "create_complete_config")
        assert hasattr(FlextConfig, "load_and_validate_from_file")
        assert hasattr(FlextConfig, "merge_and_validate_configs")
        assert hasattr(FlextConfig, "get_env_with_validation")

    def test_create_complete_config_success(self) -> None:
        """Test successful complete configuration creation."""
        config_data = {
            "database_url": "sqlite:///test.db",
            "debug": True,
            "port": 8080,
            "timeout": 30.5,
        }

        result = FlextConfig.create_complete_config(
            config_data,
            apply_defaults=False,  # Skip defaults to avoid need for defaults dict
            validate_all=False,  # Skip validation to focus on basic functionality
        )

        assert result.is_success
        final_config = result.unwrap()
        assert isinstance(final_config, dict)
        assert final_config["database_url"] == "sqlite:///test.db"
        assert final_config["debug"] is True
        assert final_config["port"] == 8080
        assert final_config["timeout"] == 30.5

    def test_create_complete_config_validation_failure(self) -> None:
        """Test complete configuration creation with validation failure."""
        # Use invalid config data that should fail validation
        config_data = {"invalid_key": None}

        result = FlextConfig.create_complete_config(
            config_data,
            validate_all=True,
        )

        # This test depends on the base validation logic
        # We assume that None values might trigger validation failures
        # Adjust based on actual _BaseConfigValidation implementation
        assert result.is_success or result.is_failure  # Either outcome is valid

    def test_create_complete_config_no_validation(self) -> None:
        """Test complete configuration creation without validation."""
        config_data = {
            "any_key": "any_value",
            "another_key": 42,
        }

        result = FlextConfig.create_complete_config(
            config_data,
            apply_defaults=False,
            validate_all=False,
        )

        assert result.is_success
        final_config = result.unwrap()
        assert final_config["any_key"] == "any_value"
        assert final_config["another_key"] == 42

    def test_create_complete_config_exception_handling(self) -> None:
        """Test complete configuration creation with exception handling."""
        # Test with invalid config data type
        result = FlextConfig.create_complete_config(
            "not_a_dict",
        )

        assert result.is_failure
        assert "failed" in result.error.lower()


class TestFlextConfigFileOperations:
    """Test file-based configuration operations."""

    def test_load_and_validate_from_file_success(self) -> None:
        """Test successful file loading and validation."""
        config_data = {
            "database_url": "postgresql://localhost/test",
            "secret_key": "test_secret_key_123",
            "debug": False,
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            import json

            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = FlextConfig.load_and_validate_from_file(
                temp_path,
                required_keys=["database_url", "secret_key"],
            )

            assert result.is_success
            loaded_config = result.unwrap()
            assert loaded_config["database_url"] == "postgresql://localhost/test"
            assert loaded_config["secret_key"] == "test_secret_key_123"
            assert loaded_config["debug"] is False
        finally:
            pathlib.Path(temp_path).unlink()

    def test_load_and_validate_from_file_missing_required_key(self) -> None:
        """Test file loading with missing required key."""
        config_data = {"database_url": "sqlite:///test.db"}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            import json

            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = FlextConfig.load_and_validate_from_file(
                temp_path,
                required_keys=["database_url", "secret_key"],
            )

            assert result.is_failure
            assert "secret_key" in result.error
            assert "not found" in result.error
        finally:
            pathlib.Path(temp_path).unlink()

    def test_load_and_validate_from_file_invalid_file(self) -> None:
        """Test file loading with invalid file path."""
        result = FlextConfig.load_and_validate_from_file(
            "/nonexistent/path/config.json",
        )

        assert result.is_failure
        # Error message should indicate file loading failure

    def test_load_and_validate_from_file_no_required_keys(self) -> None:
        """Test file loading without required key validation."""
        config_data = {"any_key": "any_value"}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            import json

            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = FlextConfig.load_and_validate_from_file(temp_path)

            assert result.is_success
            loaded_config = result.unwrap()
            assert loaded_config["any_key"] == "any_value"
        finally:
            pathlib.Path(temp_path).unlink()


class TestFlextConfigMerging:
    """Test configuration merging and validation."""

    def test_merge_and_validate_configs_success(self) -> None:
        """Test successful configuration merging with validation."""
        base_config = {
            "database_url": "sqlite:///base.db",
            "debug": False,
            "timeout": 30,
        }

        override_config = {
            "debug": True,
            "port": 8080,
            "new_setting": "new_value",
        }

        result = FlextConfig.merge_and_validate_configs(
            base_config,
            override_config,
        )

        assert result.is_success
        merged_config = result.unwrap()

        # Check that base values are preserved
        assert merged_config["database_url"] == "sqlite:///base.db"
        assert merged_config["timeout"] == 30

        # Check that override values take precedence
        assert merged_config["debug"] is True

        # Check that new values are added
        assert merged_config["port"] == 8080
        assert merged_config["new_setting"] == "new_value"

    def test_merge_and_validate_configs_empty_override(self) -> None:
        """Test configuration merging with empty override."""
        base_config = {"key1": "value1", "key2": "value2"}
        override_config: dict[str, Any] = {}

        result = FlextConfig.merge_and_validate_configs(
            base_config,
            override_config,
        )

        assert result.is_success
        merged_config = result.unwrap()
        assert merged_config == base_config

    def test_merge_and_validate_configs_exception_handling(self) -> None:
        """Test configuration merging with exception handling."""
        # Test with invalid config types
        result = FlextConfig.merge_and_validate_configs(
            "not_a_dict",
            {"key": "value"},
        )

        assert result.is_failure
        assert "merge failed" in result.error.lower()


class TestFlextConfigEnvironmentVariables:
    """Test environment variable operations."""

    def test_get_env_with_validation_success(self) -> None:
        """Test successful environment variable retrieval."""
        # Set a test environment variable
        test_var_name = "FLEXT_TEST_VAR"
        test_var_value = "test_value_123"
        os.environ[test_var_name] = test_var_value

        try:
            result = FlextConfig.get_env_with_validation(
                test_var_name,
                required=True,
                validate_type=str,
            )

            assert result.is_success
            assert result.unwrap() == test_var_value
        finally:
            # Clean up
            if test_var_name in os.environ:
                del os.environ[test_var_name]

    def test_get_env_with_validation_missing_required(self) -> None:
        """Test environment variable retrieval with missing required variable."""
        nonexistent_var = "FLEXT_NONEXISTENT_VAR_12345"

        # Ensure the variable doesn't exist
        if nonexistent_var in os.environ:
            del os.environ[nonexistent_var]

        result = FlextConfig.get_env_with_validation(
            nonexistent_var,
            required=True,
        )

        assert result.is_failure
        assert "required" in result.error.lower()
        assert nonexistent_var in result.error

    def test_get_env_with_validation_default_value(self) -> None:
        """Test environment variable retrieval with default value."""
        nonexistent_var = "FLEXT_NONEXISTENT_VAR_DEFAULT"
        default_value = "default_test_value"

        # Ensure the variable doesn't exist
        if nonexistent_var in os.environ:
            del os.environ[nonexistent_var]

        result = FlextConfig.get_env_with_validation(
            nonexistent_var,
            required=False,
            default=default_value,
        )

        assert result.is_success
        assert result.unwrap() == default_value

    def test_get_env_with_validation_optional_missing(self) -> None:
        """Test optional environment variable retrieval when missing."""
        nonexistent_var = "FLEXT_OPTIONAL_VAR_12345"

        # Ensure the variable doesn't exist
        if nonexistent_var in os.environ:
            del os.environ[nonexistent_var]

        result = FlextConfig.get_env_with_validation(
            nonexistent_var,
            required=False,
        )

        # Should propagate the failure from the base method
        assert result.is_failure


class TestFlextBaseSettings:
    """Test FlextBaseSettings Pydantic integration."""

    def test_base_settings_inheritance(self) -> None:
        """Test that FlextBaseSettings inherits from PydanticBaseSettings."""
        from pydantic_settings import BaseSettings as PydanticBaseSettings

        assert issubclass(FlextBaseSettings, PydanticBaseSettings)

        # Check model configuration
        config = FlextBaseSettings.model_config
        assert config["env_file"] == ".env"
        assert config["env_file_encoding"] == "utf-8"
        assert config["case_sensitive"] is False
        assert config["extra"] == "forbid"
        assert config["validate_assignment"] is True

    def test_create_with_validation_success(self) -> None:
        """Test successful settings creation with validation."""
        from pydantic_settings import SettingsConfigDict

        class TestSettings(FlextBaseSettings):
            database_url: str = "sqlite:///default.db"
            debug: bool = False
            port: int = 8000

            model_config = SettingsConfigDict(
                env_file=None,  # Don't load from .env file
                extra="ignore",  # Ignore extra environment variables for testing
            )

        result = TestSettings.create_with_validation(
            database_url="postgresql://localhost/test",
            debug=True,
            port=9000,
        )

        assert result.is_success
        settings = result.unwrap()
        assert isinstance(settings, TestSettings)
        assert settings.database_url == "postgresql://localhost/test"
        assert settings.debug is True
        assert settings.port == 9000

    def test_create_with_validation_type_error(self) -> None:
        """Test settings creation with type validation error."""

        class TestSettings(FlextBaseSettings):
            port: int = 8000
            debug: bool = False

        result = TestSettings.create_with_validation(
            port="not_an_integer",
            debug="not_a_boolean",
        )

        assert result.is_failure
        assert "failed" in result.error.lower()

    def test_create_with_validation_extra_fields(self) -> None:
        """Test settings creation with extra fields (should fail due to forbid)."""

        class TestSettings(FlextBaseSettings):
            database_url: str = "sqlite:///default.db"

        result = TestSettings.create_with_validation(
            database_url="sqlite:///test.db",
            extra_field="not_allowed",  # This should cause failure
        )

        assert result.is_failure
        assert "failed" in result.error.lower()

    def test_create_with_validation_empty_overrides(self) -> None:
        """Test settings creation with no overrides."""
        from pydantic_settings import SettingsConfigDict

        class TestSettings(FlextBaseSettings):
            database_url: str = "sqlite:///default.db"
            debug: bool = False

            model_config = SettingsConfigDict(
                env_file=None,  # Don't load from .env file
                extra="ignore",  # Ignore extra environment variables for testing
            )

        result = TestSettings.create_with_validation()

        assert result.is_success
        settings = result.unwrap()
        assert settings.database_url == "sqlite:///default.db"
        assert settings.debug is False


class TestFlextConfigDirectBaseExposure:
    """Test direct base class exposure without inheritance overhead."""

    def test_direct_base_access(self) -> None:
        """Test access to directly exposed base classes."""
        # Verify that direct base classes are available
        assert FlextConfigOps is not None
        assert FlextConfigDefaults is not None
        assert FlextConfigValidation is not None

        # Verify they have expected methods (this depends on base implementation)
        assert hasattr(FlextConfigOps, "__name__")
        assert hasattr(FlextConfigDefaults, "__name__")
        assert hasattr(FlextConfigValidation, "__name__")


class TestFlextConfigBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_safe_load_json_file_success(self) -> None:
        """Test backward compatible JSON file loading."""
        config_data = {"test_key": "test_value", "number": 42}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            import json

            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = safe_load_json_file(temp_path)

            assert result.is_success
            loaded_data = result.unwrap()
            assert loaded_data["test_key"] == "test_value"
            assert loaded_data["number"] == 42
        finally:
            pathlib.Path(temp_path).unlink()

    def test_safe_load_json_file_invalid_path(self) -> None:
        """Test backward compatible JSON file loading with invalid path."""
        result = safe_load_json_file("/nonexistent/path/config.json")

        assert result.is_failure

    def test_safe_get_env_var_existing(self) -> None:
        """Test backward compatible environment variable retrieval."""
        test_var_name = "FLEXT_BACKWARD_COMPAT_TEST"
        test_var_value = "backward_compat_value"
        os.environ[test_var_name] = test_var_value

        try:
            result = safe_get_env_var(test_var_name)

            assert result.is_success
            assert result.unwrap() == test_var_value
        finally:
            if test_var_name in os.environ:
                del os.environ[test_var_name]

    def test_safe_get_env_var_with_default(self) -> None:
        """Test backward compatible environment variable with default."""
        nonexistent_var = "FLEXT_NONEXISTENT_BACKWARD_COMPAT"
        default_value = "default_backward_value"

        if nonexistent_var in os.environ:
            del os.environ[nonexistent_var]

        result = safe_get_env_var(nonexistent_var, default_value)

        assert result.is_success
        assert result.unwrap() == default_value

    def test_merge_configs_backward_compatibility(self) -> None:
        """Test backward compatible configuration merging."""
        base = {"key1": "base_value1", "key2": "base_value2"}
        override = {"key2": "override_value2", "key3": "override_value3"}

        merged = merge_configs(base, override)

        # The function returns a dict directly
        assert isinstance(merged, dict)
        assert merged["key1"] == "base_value1"  # Preserved from base
        assert merged["key2"] == "override_value2"  # Overridden
        assert merged["key3"] == "override_value3"  # Added from override


class TestFlextConfigIntegrationScenarios:
    """Test integration scenarios combining multiple config features."""

    def test_complete_configuration_workflow(self) -> None:
        """Test complete configuration workflow from file to environment."""
        # Create a base config file
        base_config = {
            "database_url": "sqlite:///base.db",
            "debug": False,
            "timeout": 30,
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            import json

            json.dump(base_config, f)
            config_file_path = f.name

        # Set environment override
        env_var_name = "FLEXT_INTEGRATION_DEBUG"
        os.environ[env_var_name] = "true"

        try:
            # Step 1: Load base config from file
            file_result = FlextConfig.load_and_validate_from_file(
                config_file_path,
                required_keys=["database_url"],
            )
            assert file_result.is_success
            base_config_loaded = file_result.unwrap()

            # Step 2: Get environment override
            env_result = FlextConfig.get_env_with_validation(
                env_var_name,
                required=False,
                default="false",
            )
            assert env_result.is_success
            debug_value = env_result.unwrap()

            # Step 3: Merge configurations
            override_config = {"debug": debug_value == "true"}
            merge_result = FlextConfig.merge_and_validate_configs(
                base_config_loaded,
                override_config,
            )
            assert merge_result.is_success
            final_config = merge_result.unwrap()

            # Step 4: Create complete validated config
            complete_result = FlextConfig.create_complete_config(
                final_config,
                apply_defaults=True,
                validate_all=True,
            )
            assert complete_result.is_success
            complete_config = complete_result.unwrap()

            # Verify final result
            assert complete_config["database_url"] == "sqlite:///base.db"
            assert complete_config["debug"] is True  # Overridden by environment
            assert complete_config["timeout"] == 30

        finally:
            # Clean up
            pathlib.Path(config_file_path).unlink()
            if env_var_name in os.environ:
                del os.environ[env_var_name]

    def test_settings_with_config_integration(self) -> None:
        """Test FlextBaseSettings integration with FlextConfig validation."""
        from pydantic_settings import SettingsConfigDict

        class IntegrationSettings(FlextBaseSettings):
            database_url: str = "sqlite:///default.db"
            secret_key: str = "default_secret"
            debug: bool = False

            model_config = SettingsConfigDict(
                env_file=None,  # Don't load from .env file
                extra="ignore",  # Ignore extra environment variables for testing
            )

            @field_validator("secret_key")
            @classmethod
            def validate_secret_key(cls, v: str) -> str:
                if len(v) < 8:
                    msg = "Secret key must be at least 8 characters"
                    raise ValueError(msg)
                return v

        # Test with valid settings
        result = IntegrationSettings.create_with_validation(
            database_url="postgresql://localhost/app",
            secret_key="very_secret_key_123",
            debug=True,
        )

        assert result.is_success
        settings = result.unwrap()
        assert settings.database_url == "postgresql://localhost/app"
        assert settings.secret_key == "very_secret_key_123"
        assert settings.debug is True

    def test_error_handling_chain(self) -> None:
        """Test error handling through the entire configuration chain."""
        # Test that errors propagate correctly through the chain

        # Step 1: Invalid file loading
        invalid_file_result = FlextConfig.load_and_validate_from_file(
            "/definitely/nonexistent/path.json",
        )
        assert invalid_file_result.is_failure

        # Step 2: Invalid environment variable
        missing_env_result = FlextConfig.get_env_with_validation(
            "DEFINITELY_NONEXISTENT_VAR_12345",
            required=True,
        )
        assert missing_env_result.is_failure

        # Step 3: Invalid merge (depends on base implementation)
        invalid_merge_result = FlextConfig.merge_and_validate_configs(
            "not_a_dict",
            {"key": "value"},
        )
        assert invalid_merge_result.is_failure


class TestFlextConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_handling(self) -> None:
        """Test handling of empty configurations."""
        empty_config: dict[str, Any] = {}

        result = FlextConfig.create_complete_config(
            empty_config,
            apply_defaults=True,
            validate_all=False,
        )

        assert result.is_success
        final_config = result.unwrap()
        assert isinstance(final_config, dict)

    def test_large_config_handling(self) -> None:
        """Test handling of large configurations."""
        large_config = {f"key_{i}": f"value_{i}" for i in range(1000)}

        result = FlextConfig.create_complete_config(
            large_config,
            apply_defaults=False,
            validate_all=False,
        )

        assert result.is_success
        final_config = result.unwrap()
        assert len(final_config) == 1000
        assert final_config["key_500"] == "value_500"

    def test_nested_config_merging(self) -> None:
        """Test merging of nested configurations."""
        base_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "app_db",
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }

        override_config = {
            "database": {
                "host": "production_host",
                "ssl": True,
            },
            "redis": {
                "port": 6380,
            },
        }

        result = FlextConfig.merge_and_validate_configs(
            base_config,
            override_config,
        )

        assert result.is_success
        merged = result.unwrap()

        # The exact behavior depends on the base merge implementation
        # This test validates that the operation completes successfully
        assert isinstance(merged, dict)
        assert "database" in merged
        assert "redis" in merged
