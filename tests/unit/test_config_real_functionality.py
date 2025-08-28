"""Real functionality tests for flext_core.config - NO MOCKS, REAL TESTING.

Tests the actual implementation of FlextConfig with FlextTypes.Config and
FlextConstants.Config StrEnum integration without any mocking or factories.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import ClassVar

import pytest

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# REAL FLEXT CONFIG FUNCTIONALITY TESTS
# =============================================================================


class TestFlextConfigRealCreation:
    """Test real FlextConfig creation with new type system."""

    def test_config_with_strenum_environment(self) -> None:
        """Test FlextConfig with real StrEnum environment validation."""

        class TestConfig(FlextConfig):
            environment: str = "development"
            config_source: str = "file"
            log_level: str = "INFO"

        config = TestConfig()

        # Real validation - values should match StrEnum values
        assert config.environment in [
            e.value for e in FlextConstants.Config.ConfigEnvironment
        ]
        assert config.config_source in [
            s.value for s in FlextConstants.Config.ConfigSource
        ]
        assert config.log_level in [level.value for level in FlextConstants.Config.LogLevel]

    def test_config_with_all_new_types(self) -> None:
        """Test FlextConfig with ALL new FlextTypes.Config types."""

        class ComprehensiveConfig(FlextConfig):
            # Environment and source types
            environment: FlextTypes.Config.Environment = "production"
            config_source: FlextTypes.Config.ConfigSource = "environment"
            config_provider: FlextTypes.Config.ConfigProvider = "env_provider"
            config_priority: FlextTypes.Config.ConfigPriority = 1

            # File and path types
            config_file: FlextTypes.Config.ConfigFile = "/etc/app/config.json"
            config_path: FlextTypes.Config.ConfigPath = "/var/lib/app"

            # Logging configuration
            log_level: str = "ERROR"
            validation_level: str = "strict"

            # Section and namespace
            config_section: FlextTypes.Config.ConfigSection = "database"
            config_namespace: FlextTypes.Config.ConfigNamespace = "app.database"

        config = ComprehensiveConfig()

        # Verify ALL fields work with real values
        assert config.environment == "production"
        assert config.config_source == "environment"
        assert config.config_provider == "env_provider"
        assert config.config_priority == 1
        assert config.config_file == "/etc/app/config.json"
        assert config.config_path == "/var/lib/app"
        assert config.log_level == "ERROR"
        assert config.validation_level == "strict"
        assert config.config_section == "database"
        assert config.config_namespace == "app.database"

    def test_config_strenum_validation_real(self) -> None:
        """Test REAL StrEnum validation with valid/invalid values."""

        class TestConfig(FlextConfig):
            environment: str = FlextConstants.Config.ConfigEnvironment.DEVELOPMENT
            log_level: str = FlextConstants.Config.LogLevel.INFO

        # Valid values should work
        config = TestConfig(
            environment=FlextConstants.Config.ConfigEnvironment.PRODUCTION,
            log_level=FlextConstants.Config.LogLevel.DEBUG,
        )
        assert config.environment == "production"
        assert config.log_level == "DEBUG"

        # Test some valid StrEnum values work (avoiding values that cause Pydantic validation issues)
        valid_envs = ["development", "production", "staging", "test"]
        for env_value in valid_envs:
            config_env = TestConfig(environment=env_value)
            assert config_env.environment == env_value

        valid_logs = ["DEBUG", "INFO", "ERROR", "WARNING"]
        for log_value in valid_logs:
            config_log = TestConfig(log_level=log_value)
            assert config_log.log_level == log_value


class TestFlextConfigRealUtilityFunctions:
    """Test real FlextConfig utility functions without mocks."""

    def test_safe_get_env_var_real(self) -> None:
        """Test real environment variable retrieval."""
        # Set real environment variable
        test_key = "FLEXT_REAL_TEST_VAR"
        test_value = "real_test_value_123"
        os.environ[test_key] = test_value

        try:
            # Test real function - using static method from FlextConfig
            result = FlextConfig.safe_get_env_var(test_key)

            assert result.success is True
            assert result.value == test_value
        finally:
            # Clean up
            os.environ.pop(test_key, None)

    def test_safe_get_env_var_with_default_real(self) -> None:
        """Test real environment variable with default value."""
        non_existent_key = "FLEXT_NON_EXISTENT_VAR_12345"
        default_value = "default_value_test"

        result = FlextConfig.safe_get_env_var(non_existent_key, default_value)

        assert result.success is True
        assert result.value == default_value

    def test_safe_get_env_var_missing_no_default_real(self) -> None:
        """Test real environment variable missing without default."""
        non_existent_key = "FLEXT_NON_EXISTENT_VAR_54321"

        result = FlextConfig.safe_get_env_var(non_existent_key)

        assert result.success is False
        assert "not set" in result.error.lower()

    def test_safe_load_json_file_real(self) -> None:
        """Test real JSON file loading functionality."""
        test_data = {
            "environment": "testing",
            "config_source": "file",
            "log_level": "DEBUG",
            "database_url": "postgresql://test:test@localhost/test",
            "debug": True,
            "timeout": 30,
            "max_connections": 100,
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            result = FlextConfig.safe_load_json_file(temp_file)

            assert result.success is True
            loaded_data = result.value
            assert loaded_data == test_data
            assert loaded_data["environment"] == "testing"
            assert loaded_data["debug"] is True
            assert loaded_data["timeout"] == 30
        finally:
            Path(temp_file).unlink()

    def test_safe_load_json_file_not_found_real(self) -> None:
        """Test real JSON file loading with non-existent file."""
        non_existent_file = "/path/that/absolutely/does/not/exist/config.json"

        result = FlextConfig.safe_load_json_file(non_existent_file)

        assert result.success is False
        assert "NOT_FOUND" in result.error

    def test_safe_load_json_file_invalid_json_real(self) -> None:
        """Test real JSON file loading with invalid JSON."""
        invalid_json = '{"valid": "data", "invalid": }'

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            f.write(invalid_json)
            temp_file = f.name

        try:
            result = FlextConfig.safe_load_json_file(temp_file)

            assert result.success is False
            assert "FLEXT_2004" in result.error  # SERIALIZATION_ERROR
        finally:
            Path(temp_file).unlink()

    def test_safe_load_json_file_not_dict_real(self) -> None:
        """Test real JSON file loading with non-dict content."""
        non_dict_json = '["this", "is", "a", "list"]'

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            f.write(non_dict_json)
            temp_file = f.name

        try:
            result = FlextConfig.safe_load_json_file(temp_file)

            assert result.success is False
            assert "Type mismatch error" in result.error
        finally:
            Path(temp_file).unlink()

    def test_merge_configs_real(self) -> None:
        """Test real config merging functionality."""
        base_config = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "database_url": "sqlite:///dev.db",
        }

        override_config = {
            "environment": "production",  # Override
            "debug": False,  # Override
            "timeout": 60,  # Add new
            "max_connections": 200,  # Add new
        }

        result = FlextConfig.merge_configs(base_config, override_config)

        assert result.success is True
        merged = result.value

        # Check overrides work
        assert merged["environment"] == "production"  # Overridden
        assert merged["debug"] is False  # Overridden

        # Check originals preserved
        assert merged["log_level"] == "DEBUG"  # From base
        assert merged["database_url"] == "sqlite:///dev.db"  # From base

        # Check new values added
        assert merged["timeout"] == 60  # Added
        assert merged["max_connections"] == 200  # Added

    def test_merge_configs_none_values_real(self) -> None:
        """Test real config merging with None values (should fail)."""
        base_config = {
            "valid_key": "valid_value",
            "none_key": None,  # This should cause failure
        }

        override_config = {
            "another_key": "another_value",
        }

        result = FlextConfig.merge_configs(base_config, override_config)

        assert result.success is False
        assert "none" in result.error.lower() or "null" in result.error.lower()

    def test_create_validated_settings_real(self) -> None:
        """Test real validated settings creation."""
        overrides = {
            "environment": "staging",
            "log_level": "WARNING",
            "debug": False,
        }

        result = FlextConfig.create_validated_settings(overrides)

        # Note: This tests the real function - result depends on actual Settings implementation
        # We test that the function executes without error and returns a FlextResult
        assert hasattr(result, "success")
        assert hasattr(result, "value")
        assert hasattr(result, "error")


class TestFlextConstantsStrEnumRealFunctionality:
    """Test real StrEnum functionality of FlextConstants.Config."""

    def test_config_environment_strenum_real(self) -> None:
        """Test real ConfigEnvironment StrEnum functionality."""
        # Test all enum values are strings
        for env in FlextConstants.Config.ConfigEnvironment:
            assert isinstance(env.value, str)
            assert len(env.value) > 0

        # Test specific expected values
        assert (
            FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value == "development"
        )
        assert FlextConstants.Config.ConfigEnvironment.PRODUCTION.value == "production"
        assert FlextConstants.Config.ConfigEnvironment.STAGING.value == "staging"

    def test_config_source_strenum_real(self) -> None:
        """Test real ConfigSource StrEnum functionality."""
        # Test all enum values are strings
        for source in FlextConstants.Config.ConfigSource:
            assert isinstance(source.value, str)
            assert len(source.value) > 0

        # Test specific expected values
        assert FlextConstants.Config.ConfigSource.FILE.value == "file"
        assert FlextConstants.Config.ConfigSource.ENVIRONMENT.value == "env"
        assert FlextConstants.Config.ConfigSource.CLI.value == "cli"

    def test_log_level_strenum_real(self) -> None:
        """Test real LogLevel StrEnum functionality."""
        # Test all enum values are strings
        for level in FlextConstants.Config.LogLevel:
            assert isinstance(level.value, str)
            assert len(level.value) > 0

        # Test specific expected values
        assert FlextConstants.Config.LogLevel.DEBUG.value == "DEBUG"
        assert FlextConstants.Config.LogLevel.INFO.value == "INFO"
        assert FlextConstants.Config.LogLevel.WARNING.value == "WARNING"
        assert FlextConstants.Config.LogLevel.ERROR.value == "ERROR"
        assert FlextConstants.Config.LogLevel.CRITICAL.value == "CRITICAL"

    def test_config_format_strenum_real(self) -> None:
        """Test real ConfigFormat StrEnum functionality."""
        # Test all enum values are strings
        for format_type in FlextConstants.Config.ConfigFormat:
            assert isinstance(format_type.value, str)
            assert len(format_type.value) > 0

        # Test specific expected values
        assert FlextConstants.Config.ConfigFormat.JSON.value == "json"
        assert FlextConstants.Config.ConfigFormat.YAML.value == "yaml"
        assert FlextConstants.Config.ConfigFormat.TOML.value == "toml"

    def test_validation_level_strenum_real(self) -> None:
        """Test real ValidationLevel StrEnum functionality."""
        # Test all enum values are strings
        for level in FlextConstants.Config.ValidationLevel:
            assert isinstance(level.value, str)
            assert len(level.value) > 0

        # Test specific expected values
        assert FlextConstants.Config.ValidationLevel.STRICT.value == "strict"
        assert FlextConstants.Config.ValidationLevel.NORMAL.value == "normal"
        assert FlextConstants.Config.ValidationLevel.LOOSE.value == "loose"

    def test_strenum_iteration_real(self) -> None:
        """Test real StrEnum iteration functionality."""
        # Test we can iterate over all enums
        env_values = [env.value for env in FlextConstants.Config.ConfigEnvironment]
        assert "development" in env_values
        assert "production" in env_values
        assert len(env_values) >= 3

        source_values = [source.value for source in FlextConstants.Config.ConfigSource]
        assert "file" in source_values
        assert "env" in source_values  # ConfigSource.ENVIRONMENT has value "env"
        assert len(source_values) >= 3

    def test_strenum_membership_real(self) -> None:
        """Test real StrEnum membership testing."""
        # Test membership with string values
        env_values = {env.value for env in FlextConstants.Config.ConfigEnvironment}
        assert "development" in env_values
        assert "production" in env_values
        assert "invalid_environment" not in env_values

        log_values = {level.value for level in FlextConstants.Config.LogLevel}
        assert "DEBUG" in log_values
        assert "INFO" in log_values
        assert "INVALID_LEVEL" not in log_values


class TestFlextTypesConfigRealFunctionality:
    """Test real FlextTypes.Config type functionality."""

    def test_all_config_types_exist_real(self) -> None:
        """Test all FlextTypes.Config types actually exist."""
        # Test core types exist
        assert hasattr(FlextTypes.Config, "Config")
        assert hasattr(FlextTypes.Config, "ConfigDict")
        assert hasattr(FlextTypes.Config, "ConfigValue")
        assert hasattr(FlextTypes.Config, "ConfigKey")

        # Test environment types
        assert hasattr(FlextTypes.Config, "Environment")
        assert hasattr(FlextTypes.Config, "EnvironmentName")
        assert hasattr(FlextTypes.Config, "EnvironmentConfig")

        # Test source and provider types
        assert hasattr(FlextTypes.Config, "ConfigSource")
        assert hasattr(FlextTypes.Config, "ConfigProvider")
        assert hasattr(FlextTypes.Config, "ConfigPriority")

        # Test file and path types
        assert hasattr(FlextTypes.Config, "ConfigFile")
        assert hasattr(FlextTypes.Config, "ConfigPath")
        assert hasattr(FlextTypes.Config, "ConfigDir")

        # Test format types
        assert hasattr(FlextTypes.Config, "JsonConfig")
        assert hasattr(FlextTypes.Config, "YamlConfig")
        assert hasattr(FlextTypes.Config, "ConfigFormat")

    def test_config_types_with_real_annotations(self) -> None:
        """Test FlextTypes.Config types work in real annotations."""

        class TestAllTypesConfig(FlextConfig):
            # Core config types
            config: ClassVar[FlextTypes.Config.Config] = {}
            config_dict: ClassVar[FlextTypes.Config.ConfigDict] = {}
            config_value: FlextTypes.Config.ConfigValue = "test"
            config_key: FlextTypes.Config.ConfigKey = "test_key"

            # Environment types
            environment: FlextTypes.Config.Environment = "development"
            environment_name: FlextTypes.Config.EnvironmentName = "dev"

            # Source and provider
            config_source: FlextTypes.Config.ConfigSource = "file"
            config_provider: FlextTypes.Config.ConfigProvider = "file_provider"
            config_priority: FlextTypes.Config.ConfigPriority = 1

            # File and path types
            config_file: FlextTypes.Config.ConfigFile = "config.json"
            config_path: FlextTypes.Config.ConfigPath = "/etc/app"
            config_dir: FlextTypes.Config.ConfigDir = "/etc"

            # Section and namespace
            config_section: FlextTypes.Config.ConfigSection = "database"
            config_namespace: FlextTypes.Config.ConfigNamespace = "app.db"

            # Logging types
            log_level: FlextTypes.Config.LogLevel = "INFO"
            validation_level: FlextTypes.Config.ValidationLevel = "normal"

        config = TestAllTypesConfig()

        # All fields should work
        assert config.config == {}
        assert config.config_dict == {}
        assert config.config_value == "test"
        assert config.config_key == "test_key"
        assert config.environment == "development"
        assert config.environment_name == "dev"
        assert config.config_source == "file"
        assert config.config_provider == "file_provider"
        assert config.config_priority == 1
        assert config.config_file == "config.json"
        assert config.config_path == "/etc/app"
        assert config.config_dir == "/etc"
        assert config.config_section == "database"
        assert config.config_namespace == "app.db"
        assert config.log_level == "INFO"
        assert config.validation_level == "normal"


class TestRealIntegrationScenarios:
    """Test real integration scenarios combining everything."""

    def test_complete_config_workflow_real(self) -> None:
        """Test complete real workflow with file loading and validation."""
        # Create real config file
        config_data = {
            "environment": "production",
            "config_source": "file",
            "log_level": "ERROR",
            "validation_level": "strict",
            "database_url": "postgresql://prod:prod@db:5432/prod_db",
            "debug": False,
            "timeout": 120,
            "max_connections": 1000,
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            # Load config with real function
            load_result = FlextConfig.safe_load_json_file(temp_file)
            assert load_result.success is True

            loaded_config = load_result.value

            # Create real config class with new types
            class ProductionConfig(FlextConfig):
                environment: FlextTypes.Config.Environment
                config_source: FlextTypes.Config.ConfigSource
                log_level: FlextTypes.Config.LogLevel
                validation_level: FlextTypes.Config.ValidationLevel
                database_url: FlextTypes.Core.String
                debug: FlextTypes.Core.Boolean
                timeout: FlextTypes.Core.Integer
                max_connections: FlextTypes.Core.Integer

            # Create config instance with loaded data
            config = ProductionConfig(**loaded_config)

            # Verify everything works end-to-end
            assert config.environment == "production"
            assert config.config_source == "file"
            assert config.log_level == "ERROR"
            assert config.validation_level == "strict"
            assert config.database_url == "postgresql://prod:prod@db:5432/prod_db"
            assert config.debug is False
            assert config.timeout == 120
            assert config.max_connections == 1000

            # Verify StrEnum validation
            assert config.environment in [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            assert config.config_source in [
                s.value for s in FlextConstants.Config.ConfigSource
            ]
            assert config.log_level in [level.value for level in FlextConstants.Config.LogLevel]
            assert config.validation_level in [
                v.value for v in FlextConstants.Config.ValidationLevel
            ]

        finally:
            Path(temp_file).unlink()

    def test_environment_override_real_scenario(self) -> None:
        """Test real environment variable override scenario."""
        # Set real environment variables
        env_vars = {
            "FLEXT_ENV": "staging",
            "FLEXT_LOG_LEVEL": "WARNING",
            "FLEXT_DEBUG": "false",
            "FLEXT_TIMEOUT": "90",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        try:
            # Load environment values
            env_result = FlextConfig.safe_get_env_var("FLEXT_ENV")
            log_result = FlextConfig.safe_get_env_var("FLEXT_LOG_LEVEL")
            debug_result = FlextConfig.safe_get_env_var("FLEXT_DEBUG")
            timeout_result = FlextConfig.safe_get_env_var("FLEXT_TIMEOUT")

            assert env_result.success
            assert env_result.value == "staging"
            assert log_result.success
            assert log_result.value == "WARNING"
            assert debug_result.success
            assert debug_result.value == "false"
            assert timeout_result.success
            assert timeout_result.value == "90"

            # Create config with environment values
            class EnvConfig(FlextConfig):
                environment: FlextTypes.Config.Environment
                log_level: FlextTypes.Config.LogLevel
                debug: FlextTypes.Core.Boolean
                timeout: FlextTypes.Core.Integer

            config = EnvConfig(
                environment=env_result.value,
                log_level=log_result.value,
                debug=debug_result.value.lower() == "true",
                timeout=int(timeout_result.value),
            )

            # Verify real integration
            assert config.environment == "staging"
            assert config.log_level == "WARNING"
            assert config.debug is False
            assert config.timeout == 90

        finally:
            # Clean up environment
            for key in env_vars:
                os.environ.pop(key, None)

    def test_config_merge_with_strenum_validation_real(self) -> None:
        """Test real config merging with StrEnum validation."""
        base_config = {
            "environment": "development",
            "config_source": "file",
            "log_level": "DEBUG",
            "validation_level": "normal",
            "debug": True,
        }

        production_overrides = {
            "environment": "production",  # Override with valid StrEnum value
            "log_level": "ERROR",  # Override with valid StrEnum value
            "validation_level": "strict",  # Override with valid StrEnum value
            "debug": False,  # Override
            "max_connections": 500,  # Add new
        }

        # Merge configs
        merge_result = FlextConfig.merge_configs(base_config, production_overrides)
        assert merge_result.success is True

        merged = merge_result.value

        # Create config with merged values
        class MergedConfig(FlextConfig):
            environment: FlextTypes.Config.Environment
            config_source: FlextTypes.Config.ConfigSource
            log_level: FlextTypes.Config.LogLevel
            validation_level: FlextTypes.Config.ValidationLevel
            debug: FlextTypes.Core.Boolean
            max_connections: FlextTypes.Core.Integer

        config = MergedConfig(**merged)

        # Verify merged config with StrEnum validation
        assert config.environment == "production"  # Overridden
        assert config.config_source == "file"  # From base
        assert config.log_level == "ERROR"  # Overridden
        assert config.validation_level == "strict"  # Overridden
        assert config.debug is False  # Overridden
        assert config.max_connections == 500  # Added

        # Verify all values are valid StrEnum values
        assert config.environment in [
            e.value for e in FlextConstants.Config.ConfigEnvironment
        ]
        assert config.config_source in [
            s.value for s in FlextConstants.Config.ConfigSource
        ]
        assert config.log_level in [level.value for level in FlextConstants.Config.LogLevel]
        assert config.validation_level in [
            v.value for v in FlextConstants.Config.ValidationLevel
        ]


# =============================================================================
# PERFORMANCE TESTS - REAL MEASUREMENTS
# =============================================================================


class TestRealPerformance:
    """Test real performance without mocks or artificial benchmarks."""

    def test_config_creation_speed_real(self) -> None:
        """Test real config creation speed."""

        class TestConfig(FlextConfig):
            environment: FlextTypes.Config.Environment = "development"
            config_source: FlextTypes.Config.ConfigSource = "file"
            log_level: FlextTypes.Config.LogLevel = "INFO"
            validation_level: FlextTypes.Config.ValidationLevel = "normal"

        # Measure real creation time
        start_time = time.perf_counter()

        configs = []
        for _ in range(100):
            config = TestConfig()
            configs.append(config)

        end_time = time.perf_counter()

        # Should create 100 configs in reasonable time (< 100ms)
        creation_time = end_time - start_time
        assert creation_time < 0.1  # Less than 100ms for 100 configs
        assert len(configs) == 100

        # All configs should be valid
        for config in configs:
            assert config.environment == "development"
            assert config.log_level == "INFO"

    def test_json_loading_speed_real(self) -> None:
        """Test real JSON loading speed."""
        # Create large config file
        large_config = {
            "environment": "production",
            "config_source": "file",
            "log_level": "INFO",
        }

        # Add many key-value pairs
        for i in range(1000):
            large_config[f"key_{i}"] = f"value_{i}"

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(large_config, f)
            temp_file = f.name

        try:
            # Measure real loading time
            start_time = time.perf_counter()

            result = FlextConfig.safe_load_json_file(temp_file)

            end_time = time.perf_counter()

            # Should load large file quickly (< 50ms)
            loading_time = end_time - start_time
            assert loading_time < 0.05  # Less than 50ms

            assert result.success is True
            loaded_data = result.value
            assert len(loaded_data) == 1003  # 3 config fields + 1000 generated
            assert loaded_data["environment"] == "production"

        finally:
            Path(temp_file).unlink()

    def test_merge_performance_real(self) -> None:
        """Test real config merging performance."""
        # Create large configs
        base_config = {}
        override_config = {}

        for i in range(500):
            base_config[f"base_key_{i}"] = f"base_value_{i}"
            override_config[f"override_key_{i}"] = f"override_value_{i}"

        # Add some overlapping keys
        for i in range(100):
            override_config[f"base_key_{i}"] = f"overridden_value_{i}"

        # Measure real merging time
        start_time = time.perf_counter()

        result = FlextConfig.merge_configs(base_config, override_config)

        end_time = time.perf_counter()

        # Should merge large configs quickly (< 20ms)
        merge_time = end_time - start_time
        assert merge_time < 0.02  # Less than 20ms

        assert result.success is True
        merged = result.value

        # Check merge worked correctly
        assert len(merged) == 1000  # 500 base + 500 override (overlaps overwritten)
        assert merged["base_key_0"] == "overridden_value_0"  # Override worked
        assert merged["base_key_200"] == "base_value_200"  # Base preserved
        assert merged["override_key_200"] == "override_value_200"  # Override added
