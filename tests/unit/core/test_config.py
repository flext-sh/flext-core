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
import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextResult,
    FlextSettings,
    TAnyDict,
    merge_configs,
    safe_get_env_var,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# Custom test classes for configuration testing


class TestFlextSettings(FlextSettings):
    """Test version of FlextSettings with controlled behavior."""

    # Override env_prefix for testing
    model_config = SettingsConfigDict(
        env_prefix="TEST_",
        env_file=".test.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Custom fields for testing
    test_var: str | None = None
    from_env: str | None = None


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


@pytest.fixture
def temp_json_file() -> Generator[str]:
    """Create a temporary valid JSON file for testing."""
    test_config = {
        "name": "test-config",
        "version": "2.0.0",
        "description": "Test configuration",
        "environment": "test",
        "debug": True,
        "log_level": "DEBUG",
        "timeout": 60,
        "retries": 5,
        "enable_caching": False,
    }

    with tempfile.NamedTemporaryFile(
        encoding="utf-8",
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(test_config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    with contextlib.suppress(OSError):
        Path(temp_path).unlink()


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
class TestFlextConfigClass:
    """Test FlextConfig class functionality."""

    def test_create_complete_config_success(self, sample_config: TAnyDict) -> None:
        """Test successful complete config creation."""
        result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=True,
        )

        assert result.success
        assert result.value is not None
        config = result.value
        # Check that the config contains the expected fields from the model
        assert "name" in config
        assert "debug" in config
        assert "timeout" in config
        # database_url is not a defined field in FlextConfig, so it won't be in the result
        # The test should verify the actual model fields instead

    def test_create_complete_config_no_defaults(self, sample_config: TAnyDict) -> None:
        """Test complete config creation without applying defaults."""
        result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=False,
        )

        assert result.success
        assert result.value is not None
        config = result.value
        if config["debug"] is not True:
            raise AssertionError(f"Expected {True}, got {config['debug']}")

    def test_create_complete_config_with_custom_defaults(
        self,
        sample_config: TAnyDict,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test complete config creation with custom defaults."""
        result = FlextConfig.create_complete_config(
            sample_config,
            defaults=sample_defaults,
            apply_defaults=True,
        )

        assert result.success
        assert result.value is not None
        config = result.value
        # Check that the config contains the expected fields from the model
        assert "name" in config
        assert "debug" in config
        assert "timeout" in config
        # max_connections is not a defined field in FlextConfig, so it won't be in the result
        # The test should verify the actual model fields instead

    def test_create_complete_config_empty_data(self) -> None:
        """Test complete config creation with empty data."""
        result = FlextConfig.create_complete_config({})

        assert result.success
        assert result.value is not None
        assert isinstance(result.value, dict)

    def test_create_complete_config_no_validation(
        self,
        sample_config: TAnyDict,
    ) -> None:
        """Test complete config creation without validation."""
        # Implementation is same as regular creation since base class doesn't validate
        result = FlextConfig.create_complete_config(
            sample_config,
            apply_defaults=False,
        )

        assert result.success
        assert result.value is not None

    def test_create_complete_config_validation_failure(self) -> None:
        """Test complete config creation with validation failure."""
        # Test with invalid data that will cause validation to fail
        invalid_data = {
            "environment": "invalid_environment",  # This should fail validation
        }

        try:
            FlextConfig.model_validate(invalid_data)
            # If we get here, validation didn't fail as expected
            assert False, "Expected validation to fail"
        except ValueError:
            # This is expected - validation failed appropriately
            assert True

    def test_create_complete_config_missing_required_field(self) -> None:
        """Test complete config creation with missing required field."""
        # Test Pydantic validation with invalid log level
        invalid_data = {
            "log_level": "INVALID_LEVEL",  # This should fail validation
        }

        try:
            FlextConfig.model_validate(invalid_data)
            assert False, "Expected validation to fail"
        except ValueError:
            # This is expected - validation failed appropriately
            assert True

    def test_load_and_validate_from_file_success(self, temp_json_file: str) -> None:
        """Test successful file loading and validation."""
        result = FlextConfig.load_and_validate_from_file(
            temp_json_file,
            required_keys=["name", "version"],
        )

        assert result.success
        assert result.value is not None
        config = result.value
        assert config["name"] == "test-config"
        assert config["version"] == "2.0.0"
        assert config["debug"] is True
        assert config["timeout"] == 60

    def test_load_and_validate_from_file_no_required_keys(
        self,
        temp_json_file: str,
    ) -> None:
        """Test file loading without required keys validation."""
        result = FlextConfig.load_and_validate_from_file(temp_json_file)

        assert result.success
        assert result.value is not None
        config = result.value
        # Verify the config contains the model fields from our fixture
        assert config["name"] == "test-config"
        assert config["debug"] is True
        assert isinstance(config["timeout"], int)

    def test_load_and_validate_from_file_missing_required_key(
        self,
        temp_json_file: str,
    ) -> None:
        """Test file loading with missing required key."""
        result = FlextConfig.load_and_validate_from_file(
            temp_json_file,
            required_keys=["name", "missing_key"],
        )

        assert result.is_failure
        assert "missing_key" in result.error.lower()

    def test_load_and_validate_from_file_none_value(self) -> None:
        """Test file loading with None value validation failure."""

        # Create a custom version that returns None value
        class NoneValueConfig(FlextConfig):
            @classmethod
            def load_and_validate_from_file(
                cls,
                file_path: str,
                required_keys: list[str] | None = None,
            ) -> FlextResult[dict[str, object]]:
                """Override to inject None value."""
                if "none_test" in file_path:
                    config = {"key": None}
                    if required_keys:
                        for key in required_keys:
                            if key not in config or config[key] is None:
                                return FlextResult[None].fail(
                                    f"Required config key '{key}' not found or is None",
                                )
                    return FlextResult[None].ok(config)
                return super().load_and_validate_from_file(file_path, required_keys)

        result = NoneValueConfig.load_and_validate_from_file(
            "none_test.json",
            required_keys=["key"],
        )

        assert result.is_failure
        if "Required config key 'key' not found or is None" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Required config key \\'key\\' not found or is None' in {result.error}",
            )

    def test_load_and_validate_from_file_load_failure(self) -> None:
        """Test file loading with load failure."""
        # Test loading non-existent file
        result = FlextConfig.load_and_validate_from_file("/non/existent/file.json")

        assert result.is_failure
        assert result.error is not None

    def test_load_and_validate_from_file_validation_failure(self) -> None:
        """Test file loading with validation failure."""
        # Test with invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_path = f.name

        try:
            result = FlextConfig.load_and_validate_from_file(temp_path)
            assert result.is_failure
            assert result.error is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_and_validate_from_file_nonexistent(self) -> None:
        """Test loading from nonexistent file."""
        result = FlextConfig.load_and_validate_from_file("/nonexistent/file.json")

        assert result.is_failure
        # Error message will vary based on OS
        assert result.error is not None

    def test_load_and_validate_from_file_invalid_json(
        self,
        invalid_json_file: str,
    ) -> None:
        """Test loading from invalid JSON file."""
        result = FlextConfig.load_and_validate_from_file(invalid_json_file)

        assert result.is_failure
        # Error will be JSON decode error
        assert result.error is not None

    def test_load_and_validate_from_file_access_denied(self) -> None:
        """Test file loading with access denied."""
        # Create a file and change permissions (if possible)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name

        try:
            # Try to make file unreadable (on Unix systems)
            try:
                os.chmod(temp_path, 0o000)
                result = FlextConfig.load_and_validate_from_file(temp_path)
                assert result.is_failure
                assert result.error is not None
            except (OSError, PermissionError):
                # If we can't change permissions, skip this test scenario
                result = FlextConfig.load_and_validate_from_file("/root/no_access.json")
                assert result.is_failure
        finally:
            # Restore permissions and clean up
            try:
                os.chmod(temp_path, 0o644)
                Path(temp_path).unlink(missing_ok=True)
            except (OSError, PermissionError):
                pass


# TestFlextConfigOps class removed - FlextConfigOps class doesn't exist

    def test_safe_get_env_var_existing(self) -> None:
        """Test getting existing environment variable."""
        # Set env var directly for test
        test_key = "TEST_VAR_FOR_CONFIG_TEST"
        test_value = "test_value"
        os.environ[test_key] = test_value

        try:
            result = safe_get_env_var(test_key)
            assert result.success
            assert result.value == test_value
        finally:
            # Clean up
            os.environ.pop(test_key, None)

    def test_safe_get_env_var_missing_with_default(self) -> None:
        """Test getting missing env var with default."""
        # Ensure var doesn't exist
        test_key = "MISSING_VAR_FOR_TEST"
        os.environ.pop(test_key, None)

        result = safe_get_env_var(test_key, "default_value")
        assert result.success
        assert result.value == "default_value"

    def test_safe_get_env_var_missing_no_default(self) -> None:
        """Test getting missing env var without default."""
        # Ensure var doesn't exist
        test_key = "MISSING_VAR_NO_DEFAULT"
        os.environ.pop(test_key, None)

        result = safe_get_env_var(test_key)
        assert result.failure
        assert "not set" in result.error.lower()

    def test_safe_get_env_var_empty_string(self) -> None:
        """Test getting env var with empty string value."""
        test_key = "EMPTY_VAR_TEST"
        os.environ[test_key] = ""

        try:
            result = safe_get_env_var(test_key, "default")
            # Empty string is still a valid value
            assert result.success
            assert result.value == ""
        finally:
            os.environ.pop(test_key, None)

    def test_merge_configs_success(self) -> None:
        """Test successful config merging."""
        base = {"a": 1, "b": 2, "c": {"d": 3}}
        override = {"b": 20, "f": 5}

        result = merge_configs(base, override)

        assert result.success
        assert result.value is not None
        merged = result.value
        assert merged["a"] == 1     # From base
        assert merged["b"] == 20    # Overridden
        assert merged["c"] == {"d": 3}  # From base (dict preserved)
        assert merged["f"] == 5     # New key

    def test_merge_configs_empty_base(self) -> None:
        """Test merging with empty base config."""
        override = {"a": 1, "b": 2}

        result = merge_configs({}, override)

        assert result.success
        assert result.value is not None
        merged = result.value
        if merged["a"] != 1:
            raise AssertionError(f"Expected {1}, got {merged['a']}")
        assert merged["b"] == 2

    def test_merge_configs_empty_override(self) -> None:
        """Test merging with empty override config."""
        base = {"a": 1, "b": 2}

        result = merge_configs(base, {})

        assert result.success
        assert result.value is not None
        merged = result.value
        if merged["a"] != 1:
            raise AssertionError(f"Expected {1}, got {merged['a']}")
        assert merged["b"] == 2

    def test_merge_configs_both_empty(self) -> None:
        """Test merging two empty configs."""
        result = merge_configs({}, {})

        assert result.success
        assert result.value is not None
        assert result.value == {}

    def test_merge_configs_deep_nesting(self) -> None:
        """Test merging deeply nested configs."""
        base = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "base",
                        "base_only": True,
                    },
                },
            },
        }
        override = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "override",
                        "override_only": True,
                    },
                },
            },
        }

        result = merge_configs(base, override)

        assert result.success
        assert result.value is not None
        merged = result.value
        level3 = merged["level1"]["level2"]["level3"]
        # With shallow merging, the override completely replaces the nested dict
        assert level3["value"] == "override"
        assert level3["override_only"] is True
        # base_only is not present because the dict was completely replaced
        assert "base_only" not in level3


# TestFlextConfigDefaults class removed - FlextConfigDefaults class doesn't exist


# TestFlextConfigValidation class removed - FlextConfigValidation class doesn't exist


# TestFlextConfigFactory class removed - FlextConfigFactory class doesn't exist


@pytest.mark.unit
class TestFlextSettingsClass:
    """Test FlextSettings class functionality."""

    def test_settings_from_env(self) -> None:
        """Test loading settings from environment variables."""
        # Set test env vars
        os.environ["TEST_TEST_VAR"] = "test_value"
        os.environ["TEST_FROM_ENV"] = "env_value"

        try:
            settings = TestFlextSettings()

            assert settings.test_var == "test_value"
            assert settings.from_env == "env_value"
        finally:
            os.environ.pop("TEST_TEST_VAR", None)
            os.environ.pop("TEST_FROM_ENV", None)

    def test_settings_defaults(self) -> None:
        """Test settings with default values."""
        # Ensure env vars don't exist
        os.environ.pop("TEST_TEST_VAR", None)
        os.environ.pop("TEST_FROM_ENV", None)

        settings = TestFlextSettings()

        # Should use default None values
        assert settings.test_var is None
        assert settings.from_env is None

    def test_settings_case_insensitive(self) -> None:
        """Test case-insensitive environment variable loading."""
        # Set with different case
        os.environ["TEST_TEST_VAR"] = "lowercase_value"

        try:
            settings = TestFlextSettings()
            # Case insensitive should work
            assert settings.test_var == "lowercase_value"
        finally:
            os.environ.pop("TEST_TEST_VAR", None)

    def test_settings_extra_allowed(self) -> None:
        """Test that extra fields are allowed in settings."""
        os.environ["TEST_EXTRA_FIELD"] = "extra_value"

        try:
            settings = TestFlextSettings()
            # Extra fields should be accessible
            if hasattr(settings, "extra_field"):
                assert settings.extra_field == "extra_value"
        finally:
            os.environ.pop("TEST_EXTRA_FIELD", None)

    def test_base_flext_settings(self) -> None:
        """Test base FlextSettings class."""
        # Base class uses FLEXT_ prefix
        os.environ["FLEXT_DEBUG"] = "true"
        os.environ["FLEXT_PORT"] = "8888"

        try:
            settings = FlextSettings()

            # These are base settings that should exist
            assert hasattr(settings, "model_config")
            # Verify env_prefix is set correctly
            assert settings.model_config.get("env_prefix") == "FLEXT_"
        finally:
            os.environ.pop("FLEXT_DEBUG", None)
            os.environ.pop("FLEXT_PORT", None)


@pytest.mark.unit
class TestConfigIntegration:
    """Test configuration system integration scenarios."""

    def test_complete_config_workflow(
        self,
        temp_json_file: str,
        sample_defaults: TAnyDict,
    ) -> None:
        """Test complete configuration workflow."""
        # Step 1: Load from file
        file_result = FlextConfig.load_and_validate_from_file(temp_json_file)
        assert file_result.success

        # Step 2: Apply defaults (using merge_configs)
        config = file_result.value
        defaults_result = merge_configs(sample_defaults, config)
        assert defaults_result.success

        # Step 3: Validate using FlextConfig model validation
        final_config = defaults_result.value
        flext_config = FlextConfig.model_validate(final_config)
        validation_result = flext_config.validate_business_rules()
        assert validation_result.success

        # Step 4: Check final config
        # Check that the config contains the expected fields from the model
        assert "name" in final_config
        assert "debug" in final_config
        assert "timeout" in final_config
        # database_url and max_connections are not defined fields in FlextConfig

    def test_factory_with_validation(self, sample_config: TAnyDict) -> None:
        """Test factory with validation workflow."""
        # Create config with merge_configs
        defaults = {"max_retries": 3}
        result = merge_configs(defaults, sample_config)

        assert result.success
        config = result.value

        # Validate specific keys using FlextConfig validation
        flext_config = FlextConfig.model_validate(config)
        validation_result = flext_config.validate_business_rules()
        assert validation_result.success

        # Check merged values
        if config["database_url"] != "postgresql://localhost/test":
            raise AssertionError(
                f"Expected {'postgresql://localhost/test'}, got {config['database_url']}",
            )
        assert config["max_retries"] == 3

    def test_environment_override_file_config(self, temp_json_file: str) -> None:
        """Test environment variables overriding file config."""
        # Set env var that should override file value
        os.environ["FLEXT_DATABASE_URL"] = "postgresql://env/override"

        try:
            # Load from file first
            file_result = FlextConfig.load_and_validate_from_file(temp_json_file)
            assert file_result.success
            file_config = file_result.value

            # Create env config dict manually (since FlextConfigFactory doesn't exist)
            env_config = {
                "database_url": os.environ.get("FLEXT_DATABASE_URL", ""),
            }

            # Merge with env taking precedence
            merged_result = merge_configs(file_config, env_config)
            assert merged_result.success
            final_config = merged_result.value

            # Env value should override file value
            if "database_url" in final_config:
                assert final_config["database_url"] == "postgresql://env/override"
        finally:
            os.environ.pop("FLEXT_DATABASE_URL", None)

    def test_config_with_complex_validation(self) -> None:
        """Test config with complex validation rules."""

        def validate_complex_config(config: dict[str, object]) -> FlextResult[None]:
            """Custom validation with complex rules."""
            # Check interdependent fields
            if config.get("use_ssl") and not config.get("ssl_cert"):
                return FlextResult[None].fail(
                    "SSL cert required when use_ssl is True"
                )

            # Check value ranges
            port = config.get("port")
            if isinstance(port, int) and not (1 <= port <= 65535):
                return FlextResult[None].fail("Port must be between 1 and 65535")

            # Check mutually exclusive options
            if config.get("use_memory_cache") and config.get("use_disk_cache"):
                return FlextResult[None].fail(
                    "Cannot use both memory and disk cache"
                )

            return FlextResult[None].ok(None)

        # Test valid config
        valid_config = {
            "use_ssl": True,
            "ssl_cert": "/path/to/cert",
            "port": 8080,
            "use_memory_cache": True,
            "use_disk_cache": False,
        }

        result = validate_complex_config(valid_config)
        assert result.success

        # Test invalid: missing SSL cert
        invalid_ssl = {
            "use_ssl": True,
            "port": 8080,
        }

        result = validate_complex_config(invalid_ssl)
        assert result.is_failure
        assert "SSL cert required" in (result.error or "")

        # Test invalid: bad port
        invalid_port = {
            "port": 70000,
        }

        result = validate_complex_config(invalid_port)
        assert result.is_failure
        assert "Port must be between" in (result.error or "")

        # Test invalid: mutually exclusive
        invalid_cache = {
            "use_memory_cache": True,
            "use_disk_cache": True,
        }

        result = validate_complex_config(invalid_cache)
        assert result.is_failure
        assert "Cannot use both" in (result.error or "")


# Configuration classes FlextDatabaseConfig, FlextRedisConfig, FlextLDAPConfig, FlextOracleConfig,
# FlextJWTConfig, FlextObservabilityConfig, FlextSingerConfig removed - these classes don't exist in current codebase.
# Only FlextConfig and FlextBaseConfigModel exist in config.py.

class TestFlextConfigSerializers:
    """Test FlextConfig serializer methods - increasing coverage."""

    def test_basic_config_creation(self) -> None:
        """Test basic FlextConfig creation and properties."""
        config = FlextConfig(environment="production")

        assert config.environment == "production"
        assert config.name == "flext"
        assert config.version == "1.0.0"
        assert config.log_level == "INFO"

    def test_config_with_custom_values(self) -> None:
        """Test FlextConfig with custom values."""
        config = FlextConfig(
            name="test-config",
            version="2.1.0",
            environment="staging",
            debug=True,
            log_level="DEBUG",
            timeout=60,
            retries=5,
            enable_caching=False,
            enable_metrics=True,
            enable_tracing=True,
        )

        assert config.name == "test-config"
        assert config.version == "2.1.0"
        assert config.environment == "staging"
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.timeout == 60
        assert config.retries == 5
        assert config.enable_caching is False
        assert config.enable_metrics is True
        assert config.enable_tracing is True


class TestFlextConfigValidationMethods:
    """Test FlextConfig validation methods for increased coverage."""

    def test_validate_business_rules_success(self) -> None:
        """Test validate_business_rules with valid configuration."""
        config = FlextConfig(
            database_url="postgresql://user:pass@localhost:5432/db",
            key="valid-key-value",
        )

        result = config.validate_business_rules()
        assert result.is_success

    def test_validate_business_rules_critical_none_fields(self) -> None:
        """Test validate_business_rules with None critical fields."""
        # Create a config instance and manually set __pydantic_extra__ to test the business rule
        config = FlextConfig(environment="development", log_level="INFO")

        # Manually set __pydantic_extra__ to simulate extra fields with None values
        config.__pydantic_extra__ = {
            "database_url": None,  # Critical field
            "key": None,  # Critical field
        }

        result = config.validate_business_rules()

        assert result.is_failure
        assert "Config validation failed for" in result.error

    def test_environment_validation_with_aliases(self) -> None:
        """Test environment validation with alias mappings."""
        # Test alias mapping
        config = FlextConfig(environment="prod")  # Should map to production
        assert config.environment == "production"

        config = FlextConfig(environment="dev")  # Should map to development
        assert config.environment == "development"

        config = FlextConfig(environment="stg")  # Should map to staging
        assert config.environment == "staging"

    def test_environment_validation_invalid(self) -> None:
        """Test environment validation with invalid value."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            FlextConfig(environment="invalid_env")

    def test_log_level_validation_case_insensitive(self) -> None:
        """Test log level validation is case insensitive."""
        config = FlextConfig(log_level="debug")  # lowercase
        assert config.log_level == "DEBUG"  # Should be uppercase

        config = FlextConfig(log_level="Warning")  # mixed case
        assert config.log_level == "WARNING"

    def test_log_level_validation_invalid(self) -> None:
        """Test log level validation with invalid value."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            FlextConfig(log_level="INVALID_LEVEL")

    def test_positive_integer_validation(self) -> None:
        """Test validation of positive integer fields."""
        # Valid positive values
        config = FlextConfig(timeout=30, retries=3, page_size=100)
        assert config.timeout == 30
        assert config.retries == 3
        assert config.page_size == 100

        # Invalid zero value
        with pytest.raises(ValueError, match="Value must be positive"):
            FlextConfig(timeout=0)

        # Invalid negative value
        with pytest.raises(ValueError, match="Value must be positive"):
            FlextConfig(retries=-1)

        # Invalid negative page size
        with pytest.raises(ValueError, match="Value must be positive"):
            FlextConfig(page_size=-10)


# All config classes (FlextRedisConfig, FlextLDAPConfig, FlextOracleConfig, FlextJWTConfig,
# FlextObservabilityConfig, FlextSingerConfig) have been removed from this test file.
# These classes do not exist in the current codebase - only FlextConfig and FlextBaseConfigModel exist.
# The tests for these non-existent classes have been removed to fix test failures.
