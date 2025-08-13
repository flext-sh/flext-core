"""Comprehensive tests for config.py module.

# Constants
EXPECTED_TOTAL_PAGES = 8
EXPECTED_DATA_COUNT = 3

This test suite provides complete coverage of the configuration system,
testing all aspects including FlextConfig, FlextSettings, configuration
operations, validation, and integration patterns to achieve near 100% coverage.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from flext_core.config import (
    FlextConfig,
    FlextConfigDefaults,
    FlextConfigOps,
    FlextConfigValidation,
    FlextSettings,
    merge_configs,
    safe_get_env_var,
    safe_load_json_file,
)
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Generator

    from flext_core.typings import TAnyDict

pytestmark = [pytest.mark.unit, pytest.mark.core]


# Test data fixtures
@pytest.fixture
def sample_config() -> TAnyDict:
    """Sample configuration data for testing."""
    return {
        "database_url": "postgresql://localhost/test",
        "debug": True,
        "port": FlextConstants.Platform.FLEXCORE_PORT,
        "timeout": FlextConstants.DEFAULT_TIMEOUT,
    }


@pytest.fixture
def sample_defaults() -> TAnyDict:
    """Sample default configuration values."""
    return {
        "debug": False,
        "timeout": 60,
        "port": 8000,
        "max_connections": 100,
    }


# temp_json_file fixture now centralized in conftest.py


@pytest.fixture
def invalid_json_file() -> Generator[str]:
    """Create a temporary invalid JSON file for testing."""
    with tempfile.NamedTemporaryFile(
        encoding="utf-8",
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        f.write('{"invalid": json syntax}')
        temp_path = f.name

    yield temp_path

    # Cleanup
    with contextlib.suppress(OSError):
        Path(temp_path).unlink()


@pytest.mark.unit
class TestFlextConfig:
    """Test FlextConfig class functionality."""

    def test_create_complete_config_success(self, sample_config: TAnyDict) -> None:
        """Test successful complete config creation."""
        result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=True,
            validate_all=True,
        )

        assert result.success
        assert result.data is not None
        config = result.data
        if config["database_url"] != "postgresql://localhost/test":
            raise AssertionError(
                f"Expected 'postgresql://localhost/test', got {config['database_url']}",
            )
        if not config["debug"]:
            raise AssertionError(f"Expected True, got {config['debug']}")
        if config["port"] != FlextConstants.Platform.FLEXCORE_PORT:
            raise AssertionError(
                f"Expected {FlextConstants.Platform.FLEXCORE_PORT}, got {config['port']}",
            )
        # Should apply defaults
        if "timeout" not in config:
            raise AssertionError(f"Expected 'timeout' in {config}")

    def test_create_complete_config_no_defaults(self, sample_config: TAnyDict) -> None:
        """Test complete config creation without applying defaults."""
        result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=False,
            validate_all=True,
        )

        assert result.success
        assert result.data is not None
        config = result.data
        if config["database_url"] != "postgresql://localhost/test":
            raise AssertionError(
                f"Expected {'postgresql://localhost/test'}, got {config['database_url']}",
            )
        if not (config["debug"]):
            raise AssertionError(f"Expected True, got {config['debug']}")

    def test_create_complete_config_no_validation(
        self,
        sample_config: TAnyDict,
    ) -> None:
        """Test complete config creation without validation."""
        result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=True,
            validate_all=False,
        )

        assert result.success
        assert result.data is not None
        config = result.data
        if config["database_url"] != "postgresql://localhost/test":
            raise AssertionError(
                f"Expected {'postgresql://localhost/test'}, got {config['database_url']}",
            )

    def test_create_complete_config_with_none_values(self) -> None:
        """Test complete config creation with None values (should fail validation)."""
        config_with_none = {
            "database_url": None,
            "debug": True,
        }

        result = FlextConfig.create_complete_config(
            cast("TAnyDict", config_with_none),
            apply_defaults=True,
            validate_all=True,
        )

        assert result.is_failure
        if "Config validation failed for database_url" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Config validation failed for database_url' in {result.error}",
            )

    def test_create_complete_config_validation_failure(self) -> None:
        """Test complete config creation with validation failure."""
        # Mock the validation to fail
        config_data = {"key": "value"}

        with patch(
            "flext_core.config.FlextConfigValidation.validate_config_value",
        ) as mock_validate:
            mock_validate.return_value = FlextResult.fail("Validation failed")

            result = FlextConfig.create_complete_config(
                cast("TAnyDict", config_data),
                validate_all=True,
            )

            assert result.is_failure
            if "Config validation failed for key" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Config validation failed for key' in {result.error}",
                )

    def test_create_complete_config_load_failure(self) -> None:
        """Test complete config creation with load failure."""
        # Mock safe_load_from_dict to fail - use correct module path
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_from_dict",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("Load failed")

            result = FlextConfig.create_complete_config({"key": "value"})

            assert result.is_failure
            if "Config load failed" not in (result.error or ""):
                raise AssertionError(f"Expected 'Config load failed' in {result.error}")

    def test_create_complete_config_defaults_failure(self) -> None:
        """Test complete config creation with defaults application failure."""
        # Mock apply_defaults to fail - use correct module path
        with patch(
            "flext_core.config.FlextConfigDefaults.apply_defaults",
        ) as mock_defaults:
            mock_defaults.return_value = FlextResult.fail("Defaults failed")

            result = FlextConfig.create_complete_config(
                {"key": "value"},
                apply_defaults=True,
            )

            assert result.is_failure
            if "Applying defaults failed" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Applying defaults failed' in {result.error}",
                )

    def test_create_complete_config_exception_handling(self) -> None:
        """Test complete config creation with exception handling."""
        # Create config data that will cause an exception - use correct module path
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_from_dict",
        ) as mock_load:
            mock_load.side_effect = TypeError("Type error")

            result = FlextConfig.create_complete_config({"key": "value"})

            assert result.is_failure
            if "Complete config creation failed" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Complete config creation failed' in {result.error}",
                )

    def test_load_and_validate_from_file_success(self, temp_json_file: str) -> None:
        """Test successful file loading and validation."""
        result = FlextConfig.load_and_validate_from_file(
            temp_json_file,
            required_keys=["database_url", "secret_key"],
        )

        assert result.success
        assert result.data is not None
        config = result.data
        if config["database_url"] != "sqlite:///test.db":
            raise AssertionError(
                f"Expected {'sqlite:///test.db'}, got {config['database_url']}",
            )
        assert config["secret_key"] == "test-secret-key"

    def test_load_and_validate_from_file_no_required_keys(
        self,
        temp_json_file: str,
    ) -> None:
        """Test file loading without required keys validation."""
        result = FlextConfig.load_and_validate_from_file(temp_json_file)

        assert result.success
        assert result.data is not None
        config = result.data
        # Config is a dict; ensure key presence instead of attribute access
        if "database_url" not in config:
            raise AssertionError(f"Expected 'database_url' in {config}")

    def test_load_and_validate_from_file_missing_required_key(
        self,
        temp_json_file: str,
    ) -> None:
        """Test file loading with missing required key."""
        result = FlextConfig.load_and_validate_from_file(
            temp_json_file,
            required_keys=["database_url", "missing_key"],
        )

        assert result.is_failure
        if "Required config key 'missing_key' not found" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Required config key \\'missing_key\\' not found' in {result.error}",
            )

    def test_load_and_validate_from_file_none_value(self) -> None:
        """Test file loading with None value validation failure."""
        config_with_none = {"key": None}

        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.ok(config_with_none)

            result = FlextConfig.load_and_validate_from_file(
                "dummy.json",
                required_keys=["key"],
            )

            assert result.is_failure
            if "Invalid config value for 'key'" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Invalid config value for \\'key\\'' in {result.error}",
                )

    def test_load_and_validate_from_file_load_failure(self) -> None:
        """Test file loading with load failure."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("File load failed")

            result = FlextConfig.load_and_validate_from_file("nonexistent.json")

            assert result.is_failure
            if "File load failed" not in (result.error or ""):
                raise AssertionError(f"Expected 'File load failed' in {result.error}")

    def test_load_and_validate_from_file_empty_error(self) -> None:
        """Test file loading with empty error message."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("")

            result = FlextConfig.load_and_validate_from_file("dummy.json")

            assert result.is_failure
            # FlextResult converts empty errors to "Unknown error occurred"
            if "Unknown error occurred" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Unknown error occurred' in {result.error}",
                )

    def test_merge_and_validate_configs_success(
        self,
        sample_config: TAnyDict,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test successful config merging and validation."""
        result = FlextConfig.merge_and_validate_configs(sample_defaults, sample_config)

        assert result.success
        assert result.data is not None
        merged = result.data
        # Override values should be preserved
        if not (merged["debug"]):
            raise AssertionError(f"Expected True, got {merged['debug']}")
        if merged["port"] != FlextConstants.Platform.FLEXCORE_PORT:
            raise AssertionError(
                f"Expected {FlextConstants.Platform.FLEXCORE_PORT}, got {merged['port']}",
            )
        # Base values should be included
        if merged["max_connections"] != 100:
            raise AssertionError(f"Expected {100}, got {merged['max_connections']}")

    def test_merge_and_validate_configs_merge_failure(self) -> None:
        """Test config merging with merge failure."""
        with patch(
            "flext_core.config.FlextConfigDefaults.merge_configs",
        ) as mock_merge:
            mock_merge.return_value = FlextResult.fail("Merge failed")

            result = FlextConfig.merge_and_validate_configs({}, {})

            assert result.is_failure
            if "Config merge failed" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Config merge failed' in {result.error}",
                )

    def test_merge_and_validate_configs_validation_failure(self) -> None:
        """Test config merging with validation failure."""
        base_config = {"key": "value"}
        override_config = {"key2": "value2"}

        with patch(
            "flext_core.config.FlextConfigValidation.validate_config_value",
        ) as mock_validate:
            mock_validate.return_value = FlextResult.fail("Validation failed")

            result = FlextConfig.merge_and_validate_configs(
                cast("TAnyDict", base_config),
                cast("TAnyDict", override_config),
            )

            assert result.is_failure
            if "Merged config validation failed" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Merged config validation failed' in {result.error}",
                )

    def test_merge_and_validate_configs_exception_handling(self) -> None:
        """Test config merging with exception handling."""
        with patch(
            "flext_core.config.FlextConfigDefaults.merge_configs",
        ) as mock_merge:
            mock_merge.side_effect = TypeError("Type error")

            result = FlextConfig.merge_and_validate_configs({}, {})

            assert result.is_failure
            if "Config merge failed" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Config merge failed' in {result.error}",
                )

    def test_get_env_with_validation_success(self) -> None:
        """Test successful environment variable access with validation."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = FlextConfig.get_env_with_validation(
                "TEST_VAR",
                required=True,
                validate_type=str,
            )

            assert result.success
            if result.data != "test_value":
                raise AssertionError(f"Expected {'test_value'}, got {result.data}")

    def test_get_env_with_validation_not_required(self) -> None:
        """Test environment variable access when not required."""
        # Clear any existing env var
        with patch.dict(os.environ, {}, clear=True):
            result = FlextConfig.get_env_with_validation(
                "NONEXISTENT_VAR",
                required=False,
                default="default_value",
            )

            assert result.success
            if result.data != "default_value":
                raise AssertionError(f"Expected {'default_value'}, got {result.data}")

    def test_get_env_with_validation_env_failure(self) -> None:
        """Test environment variable access with env access failure."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_get_env_var",
        ) as mock_env:
            mock_env.return_value = FlextResult.fail("Env access failed")

            result = FlextConfig.get_env_with_validation("TEST_VAR")

            assert result.is_failure
            if "Env access failed" not in (result.error or ""):
                raise AssertionError(f"Expected 'Env access failed' in {result.error}")

    def test_get_env_with_validation_empty_error(self) -> None:
        """Test environment variable access with empty error message."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_get_env_var",
        ) as mock_env:
            mock_env.return_value = FlextResult.fail("")

            result = FlextConfig.get_env_with_validation("TEST_VAR")

            assert result.is_failure
            # FlextResult converts empty errors to "Unknown error occurred"
            if "Unknown error occurred" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Unknown error occurred' in {result.error}",
                )

    def test_safe_get_env_var_empty_error(self) -> None:
        """Test safe_get_env_var wrapper with empty error."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_get_env_var",
        ) as mock_env:
            mock_env.return_value = FlextResult.fail("")

            result = safe_get_env_var("TEST_VAR")

            assert result.is_failure
            # FlextResult converts empty errors to "Unknown error occurred"
            if "Unknown error occurred" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Unknown error occurred' in {result.error}",
                )

    def test_safe_load_json_file_success(self, temp_json_file: str) -> None:
        """Test safe_load_json_file wrapper function success."""
        result = safe_load_json_file(temp_json_file)

        assert result.success
        assert result.data is not None
        config = result.data
        if "database_url" not in config:
            raise AssertionError(f"Expected {'database_url'} in {config}")

    def test_safe_load_json_file_path_object(self, temp_json_file: str) -> None:
        """Test safe_load_json_file with Path object."""
        path_obj = Path(temp_json_file)
        result = safe_load_json_file(path_obj)

        assert result.success
        assert result.data is not None
        config = result.data
        if "database_url" not in config:
            raise AssertionError(f"Expected {'database_url'} in {config}")

    def test_safe_load_json_file_failure(self) -> None:
        """Test safe_load_json_file wrapper with failure."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("File error")

            result = safe_load_json_file("dummy.json")

            assert result.is_failure
            if "File error" not in (result.error or ""):
                raise AssertionError(f"Expected 'File error' in {result.error}")

    def test_safe_load_json_file_empty_error(self) -> None:
        """Test safe_load_json_file wrapper with empty error."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("")

            result = safe_load_json_file("dummy.json")

            assert result.is_failure
            # Module-level safe_load_json_file returns "File error" for any failure
            if "File error" not in (result.error or ""):
                raise AssertionError(f"Expected 'File error' in {result.error}")

    def test_merge_configs_proxy(
        self,
        sample_config: TAnyDict,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test merge_configs proxy method."""
        result = FlextConfig.merge_configs(sample_defaults, sample_config)

        assert result.success
        assert result.data is not None
        merged = result.data
        # Simple merge implementation
        assert merged["debug"] is True
        if merged["max_connections"] != 100:
            msg: str = f"Expected {100}, got {merged['max_connections']}"
            raise AssertionError(msg)

    def test_validate_config_value_success(self) -> None:
        """Test validate_config_value with successful validation."""

        def validator(value: object) -> bool:
            return isinstance(value, str) and len(str(value)) > 0

        result = FlextConfig.validate_config_value("test_value", validator)

        assert result.success

    def test_validate_config_value_failure(self) -> None:
        """Test validate_config_value with validation failure."""

        def validator(value: object) -> bool:
            return isinstance(value, int)

        result = FlextConfig.validate_config_value(
            "string_value",
            validator,
            "Must be integer",
        )

        assert result.is_failure
        if "Must be integer" not in (result.error or ""):
            raise AssertionError(f"Expected 'Must be integer' in {result.error}")

    def test_validate_config_value_exception_in_validator(self) -> None:
        """Test validate_config_value with exception in validator."""

        def failing_validator(value: object) -> bool:
            msg = "Validator error"
            raise ValueError(msg)

        result = FlextConfig.validate_config_value("test", failing_validator)

        assert result.is_failure
        if "Validation error" not in (result.error or ""):
            raise AssertionError(f"Expected 'Validation error' in {result.error}")

    def test_validate_config_value_non_callable_validator(self) -> None:
        """Test validate_config_value with non-callable validator."""
        result = FlextConfig.validate_config_value("test", "not_callable")

        assert result.is_failure
        if "Validator must be callable" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Validator must be callable' in {result.error}",
            )


@pytest.mark.unit
class TestFlextSettings:
    """Test FlextSettings class functionality."""

    def test_create_settings_basic(self) -> None:
        """Test basic settings creation."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            settings = TestSettings(_env_file=None)
            if settings.debug:
                msg: str = f"Expected False, got {settings.debug}"
                raise AssertionError(msg)
            assert settings.timeout == 30

    def test_create_with_validation_success(self) -> None:
        """Test create_with_validation with valid data."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = TestSettings.create_with_validation(debug=True, timeout=60)
            assert result.success
            assert result.data is not None
            settings = cast("TestSettings", result.data)
            if not (settings.debug):
                raise AssertionError(f"Expected True, got {settings.debug}")
            if settings.timeout != 60:
                raise AssertionError(f"Expected {60}, got {settings.timeout}")

    def test_create_with_validation_overrides_only(self) -> None:
        """Test create_with_validation with overrides only."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            overrides = {"debug": True, "timeout": 60}
            result = TestSettings.create_with_validation(
                overrides=cast("TAnyDict", overrides),
            )
            assert result.success
            assert result.data is not None
            settings = cast("TestSettings", result.data)
            if not (settings.debug):
                raise AssertionError(f"Expected True, got {settings.debug}")
            if settings.timeout != 60:
                raise AssertionError(f"Expected {60}, got {settings.timeout}")

    def test_create_with_validation_kwargs_only(self) -> None:
        """Test create_with_validation with kwargs only."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = TestSettings.create_with_validation(debug=True, timeout=60)
            assert result.success
            assert result.data is not None
            settings = cast("TestSettings", result.data)
            if not (settings.debug):
                raise AssertionError(f"Expected True, got {settings.debug}")
            if settings.timeout != 60:
                raise AssertionError(f"Expected {60}, got {settings.timeout}")

    def test_create_with_validation_no_params(self) -> None:
        """Test create_with_validation with no parameters."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = TestSettings.create_with_validation()
            assert result.success
            assert result.data is not None
            settings = cast("TestSettings", result.data)
            # The debug value might be affected by environment, so we check the timeout instead
            assert settings.timeout == 30

    def test_create_with_validation_merging_priority(self) -> None:
        """Test create_with_validation merging priority."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            overrides = {"debug": True}
            result = TestSettings.create_with_validation(
                overrides=cast("TAnyDict", overrides),
                timeout=60,
            )
            assert result.success
            assert result.data is not None
            settings = cast("TestSettings", result.data)
            assert settings.debug is True  # From overrides
            if settings.timeout != 60:  # From kwargs (higher priority)
                raise AssertionError(f"Expected {60}, got {settings.timeout}")

    def test_settings_model_config(self) -> None:
        """Test settings model configuration."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Disable .env file loading and isolate environment variables
        with patch.dict(os.environ, {}, clear=True):
            settings = TestSettings(_env_file=None)
            if settings.model_config["env_file"] != ".env":
                raise AssertionError(
                    f"Expected {'.env'}, got {settings.model_config['env_file']}",
                )
            assert settings.model_config["env_file_encoding"] == "utf-8"
            if settings.model_config["case_sensitive"]:
                raise AssertionError(
                    f"Expected False, got {settings.model_config['case_sensitive']}",
                )
            assert settings.model_config["extra"] == "ignore"
            if not (settings.model_config["validate_assignment"]):
                raise AssertionError(
                    f"Expected True, got {settings.model_config['validate_assignment']}",
                )

    def test_settings_environment_integration(self) -> None:
        """Test settings environment integration."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Set specific environment variables for this test
        test_env = {"DEBUG": "true", "TIMEOUT": "60"}
        with patch.dict(os.environ, test_env, clear=True):
            settings = TestSettings(_env_file=None)
            if not (settings.debug):
                raise AssertionError(f"Expected True, got {settings.debug}")
            if settings.timeout != 60:
                raise AssertionError(f"Expected {60}, got {settings.timeout}")


@pytest.mark.unit
class TestConfigAliases:
    """Test configuration aliases and direct exports."""

    def test_flext_config_ops_alias(self, sample_config: TAnyDict) -> None:
        """Test FlextConfigOps alias works correctly."""
        result = FlextConfigOps.safe_load_from_dict(sample_config)

        assert result.success
        if result.data != sample_config:
            raise AssertionError(f"Expected {sample_config}, got {result.data}")

    def test_flext_config_defaults_alias(
        self,
        sample_config: TAnyDict,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test FlextConfigDefaults alias works correctly."""
        result = FlextConfigDefaults.apply_defaults(sample_config, sample_defaults)

        assert result.success
        assert result.data is not None
        # Should have both original and default values
        if "debug" not in result.data:
            raise AssertionError(f"Expected {'debug'} in {result.data}")
        assert "max_connections" in result.data

    def test_flext_config_validation_alias(self) -> None:
        """Test FlextConfigValidation alias works correctly."""

        def validator(value: object) -> bool:
            return isinstance(value, str)

        result = FlextConfigValidation.validate_config_value(
            "test",
            validator,
            "Must be string",
        )

        assert result.success


@pytest.mark.unit
class TestModuleLevelFunctions:
    """Test module-level wrapper functions."""

    def test_merge_configs_function_success(
        self,
        sample_config: TAnyDict,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test merge_configs function success."""
        merged = merge_configs(sample_defaults, sample_config)

        assert isinstance(merged, dict)
        # Should have values from both configs
        assert merged["debug"] is True  # From sample_config
        if merged["max_connections"] != 100:  # From sample_defaults
            raise AssertionError(f"Expected {100}, got {merged['max_connections']}")

    def test_merge_configs_function_failure(self) -> None:
        """Test merge_configs function with failure."""
        with patch(
            "flext_core.config.FlextConfigDefaults.merge_configs",
        ) as mock_merge:
            mock_merge.return_value = FlextResult.fail("Merge failed")

            result = merge_configs({}, {})

            # Should return empty dict on failure
            if result != {}:
                raise AssertionError(f"Expected {{}}, got {result}")

    def test_safe_get_env_var_success(self) -> None:
        """Test safe_get_env_var wrapper function success."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = safe_get_env_var("TEST_VAR")

            assert result.success
            if result.data != "test_value":
                raise AssertionError(f"Expected {'test_value'}, got {result.data}")

    def test_safe_get_env_var_with_default(self) -> None:
        """Test safe_get_env_var with default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = safe_get_env_var("NONEXISTENT_VAR", default="default_value")

            assert result.success
            if result.data != "default_value":
                raise AssertionError(f"Expected {'default_value'}, got {result.data}")

    def test_safe_get_env_var_required_missing(self) -> None:
        """Test safe_get_env_var with required missing variable."""
        with patch.dict(os.environ, {}, clear=True):
            result = safe_get_env_var("REQUIRED_VAR", required=True)

            assert result.is_failure
            if "required" not in (result.error or "").lower():
                raise AssertionError(
                    f"Expected {'required'} in {(result.error or '').lower()}",
                )

    def test_safe_get_env_var_failure(self) -> None:
        """Test safe_get_env_var wrapper with failure."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_get_env_var",
        ) as mock_env:
            mock_env.return_value = FlextResult.fail("Env error")

            result = safe_get_env_var("TEST_VAR")

            assert result.is_failure
            if "Env error" not in (result.error or ""):
                raise AssertionError(f"Expected 'Env error' in {result.error}")

    def test_safe_get_env_var_empty_error(self) -> None:
        """Test safe_get_env_var wrapper with empty error."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_get_env_var",
        ) as mock_env:
            mock_env.return_value = FlextResult.fail("")

            result = safe_get_env_var("TEST_VAR")

            assert result.is_failure
            # FlextResult converts empty errors to "Unknown error occurred"
            if "Unknown error occurred" not in (result.error or ""):
                raise AssertionError(
                    f"Expected 'Unknown error occurred' in {result.error}",
                )

    def test_safe_load_json_file_success(self, temp_json_file: str) -> None:
        """Test safe_load_json_file wrapper function success."""
        result = safe_load_json_file(temp_json_file)

        assert result.success
        assert result.data is not None
        config = result.data
        if "database_url" not in config:
            raise AssertionError(f"Expected {'database_url'} in {config}")

    def test_safe_load_json_file_path_object(self, temp_json_file: str) -> None:
        """Test safe_load_json_file with Path object."""
        path_obj = Path(temp_json_file)
        result = safe_load_json_file(path_obj)

        assert result.success
        assert result.data is not None
        config = result.data
        if "database_url" not in config:
            raise AssertionError(f"Expected {'database_url'} in {config}")

    def test_safe_load_json_file_failure(self) -> None:
        """Test safe_load_json_file wrapper with failure."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("File error")

            result = safe_load_json_file("dummy.json")

            assert result.is_failure
            if "File error" not in (result.error or ""):
                raise AssertionError(f"Expected 'File error' in {result.error}")

    def test_safe_load_json_file_empty_error(self) -> None:
        """Test safe_load_json_file wrapper with empty error."""
        with patch(
            "flext_core.config.FlextConfigOps.safe_load_json_file",
        ) as mock_load:
            mock_load.return_value = FlextResult.fail("")

            result = safe_load_json_file("dummy.json")

            assert result.is_failure
            # Module-level wrapper returns "File error" when underlying operation fails
            if "File error" not in (result.error or ""):
                raise AssertionError(f"Expected 'File error' in {result.error}")


@pytest.mark.unit
class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_complete_configuration_workflow(
        self,
        sample_config: TAnyDict,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test complete configuration workflow integration."""
        # Step 1: Create complete config
        complete_result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=True,
            validate_all=True,
        )
        assert complete_result.success

        # Step 2: Merge with additional config
        additional_config = {"new_setting": "new_value"}
        assert complete_result.data is not None
        merge_result = FlextConfig.merge_and_validate_configs(
            complete_result.data,
            cast("TAnyDict", additional_config),
        )
        assert merge_result.success

        # Step 3: Verify final configuration
        assert merge_result.data is not None
        final_config = merge_result.data
        if final_config["database_url"] != "postgresql://localhost/test":
            raise AssertionError(
                f"Expected {'postgresql://localhost/test'}, got {final_config['database_url']}",
            )
        assert final_config["new_setting"] == "new_value"
        if "timeout" not in final_config:  # From defaults
            raise AssertionError(f"Expected {'timeout'} in {final_config}")

    def test_settings_with_config_integration(self) -> None:
        """Test settings integration with FlextConfig."""

        class TestSettings(FlextSettings):
            debug: bool = False
            timeout: int = 30

        # Isolate environment variables for this test
        with patch.dict(os.environ, {}, clear=True):
            _ = TestSettings(_env_file=None)  # Create settings to test initialization

            # Test integration with FlextConfig
            config_data = {"debug": True, "timeout": 60}
            result = FlextConfig.create_complete_config(cast("TAnyDict", config_data))
            assert result.success

            # Test that settings can be created from config
            settings_result = TestSettings.create_with_validation(
                overrides=cast("TAnyDict", config_data),
            )
            assert settings_result.success
            assert settings_result.data is not None
            updated_settings = cast("TestSettings", settings_result.data)
            if not (updated_settings.debug):
                raise AssertionError(f"Expected True, got {updated_settings.debug}")
            if updated_settings.timeout != 60:
                raise AssertionError(f"Expected {60}, got {updated_settings.timeout}")

    def test_file_loading_with_environment_override(self, temp_json_file: str) -> None:
        """Test file loading with environment variable override."""
        # Load config from file
        file_result = FlextConfig.load_and_validate_from_file(temp_json_file)
        assert file_result.success

        base_config = file_result.data
        assert base_config is not None

        # Get environment override
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://prod/app"}):
            env_result = FlextConfig.get_env_with_validation(
                "DATABASE_URL",
                validate_type=str,
            )
            assert env_result.success
            assert env_result.data is not None

            # Merge file config with env override
            env_override: dict[str, object] = {"database_url": env_result.data}
            final_result = FlextConfig.merge_and_validate_configs(
                base_config,
                env_override,
            )
            assert final_result.success

            final_config = final_result.data
            assert final_config is not None
            if final_config["database_url"] != "postgresql://prod/app":
                raise AssertionError(
                    f"Expected {'postgresql://prod/app'}, got {final_config['database_url']}",
                )

    def test_error_handling_cascade(self) -> None:
        """Test error handling cascades through integration."""
        # Start with invalid config
        invalid_config: dict[str, object] = {"key": None}

        # This should fail validation
        result = FlextConfig.create_complete_config(
            invalid_config,
            validate_all=True,
        )
        assert result.is_failure

        # Error should be descriptive
        if "Config validation failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Config validation failed' in {result.error}",
            )
        assert "key" in (result.error or "")

    def test_proxy_methods_consistency(self, sample_config: TAnyDict) -> None:
        """Test that proxy methods are consistent with direct calls."""
        # Test safe_load_from_dict consistency
        direct_result = FlextConfigOps.safe_load_from_dict(sample_config)
        proxy_result = FlextConfig.safe_load_from_dict(sample_config)

        if direct_result.success != proxy_result.success:
            raise AssertionError(
                f"Expected {proxy_result.success}, got {direct_result.success}",
            )
        assert direct_result.data == proxy_result.data

        # Test validation consistency
        def validator(value: object) -> bool:
            return isinstance(value, str)

        direct_validation = FlextConfigValidation.validate_config_value(
            "test",
            validator,
        )
        proxy_validation = FlextConfig.validate_config_value("test", validator)

        if direct_validation.success != proxy_validation.success:
            raise AssertionError(
                f"Expected {proxy_validation.success}, got {direct_validation.success}",
            )


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test configuration edge cases and error conditions."""

    def test_empty_configurations(self) -> None:
        """Test handling of empty configurations."""
        empty_config: TAnyDict = {}

        result = FlextConfig.create_complete_config(empty_config)
        assert result.success

        # Should still apply defaults
        config = result.data
        assert config is not None
        assert config["debug"] is False  # Default value

    def test_nested_configuration_handling(self) -> None:
        """Test handling of nested configuration structures."""
        nested_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
            },
            "cache": {
                "redis": {
                    "host": "redis-server",
                    "port": 6379,
                },
            },
        }

        result = FlextConfig.create_complete_config(nested_config)
        assert result.success

        config = result.data
        assert config is not None

        # Type cast to access nested structure
        database_config = cast("dict[str, object]", config["database"])
        cache_config = cast("dict[str, object]", config["cache"])
        redis_config = cast("dict[str, object]", cache_config["redis"])

        if database_config["host"] != "localhost":
            raise AssertionError(
                f"Expected {'localhost'}, got {database_config['host']}",
            )
        assert redis_config["port"] == 6379

    def test_type_validation_edge_cases(self) -> None:
        """Test type validation with edge cases."""
        # Test with None validator
        result = FlextConfig.validate_config_value("test", None)
        assert result.is_failure
        if "Validator must be callable" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Validator must be callable' in {result.error}",
            )

        # Test with validator that returns non-boolean
        def bad_validator(value: object) -> str:
            return "not_boolean"

        # This should still work because Python treats "not_boolean" as truthy
        result = FlextConfig.validate_config_value("test", bad_validator)
        assert result.success

    def test_environment_variable_edge_cases(self) -> None:
        """Test environment variable handling edge cases."""
        # Test with empty string environment variable
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = FlextConfig.get_env_with_validation("EMPTY_VAR")
            assert result.success
            if result.data != "":
                raise AssertionError(f"Expected {''}, got {result.data}")

        # Test with whitespace-only environment variable
        with patch.dict(os.environ, {"WHITESPACE_VAR": "   "}):
            result = FlextConfig.get_env_with_validation("WHITESPACE_VAR")
            assert result.success
            if result.data != "   ":
                raise AssertionError(f"Expected {'   '}, got {result.data}")

    def test_file_loading_edge_cases(self, temp_json_file: str) -> None:
        """Test file loading edge cases."""
        # Test with non-existent file
        result = FlextConfig.load_and_validate_from_file("nonexistent.json")
        assert result.is_failure

        # Test with empty required keys list
        result = FlextConfig.load_and_validate_from_file(
            temp_json_file,
            required_keys=[],
        )
        assert result.success

    def test_merge_configs_with_none_values(self) -> None:
        """Test config merging with None values."""
        base_config: dict[str, object] = {"key1": "value1", "key2": None}
        override_config: dict[str, object] = {"key2": "new_value", "key3": None}

        result = FlextConfig.merge_and_validate_configs(base_config, override_config)
        # This should fail validation due to None values
        assert result.is_failure

    def test_settings_validation_edge_cases(self) -> None:
        """Test settings validation edge cases."""

        class StrictSettings(FlextSettings):
            required_int: int
            optional_str: str = "default"

        # Missing required field should fail
        result = StrictSettings.create_with_validation(optional_str="custom")
        assert result.is_failure

        # Wrong type should fail
        result = StrictSettings.create_with_validation(required_int="not_an_int")
        assert result.is_failure
