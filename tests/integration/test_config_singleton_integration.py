"""Integration tests for FlextConfig singleton pattern and multi-source configuration.

This test validates that:
1. FlextConfig works as a singleton across all modules
2. Configuration can be loaded from multiple sources (.env, JSON, YAML, TOML)
3. Environment variables override file configurations
4. All modules use the same configuration instance
"""

import json
import os
import tempfile
import threading
from pathlib import Path

import yaml

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextCore,
    FlextProcessors,
    FlextValidations,
)
from flext_core.mixins import FlextMixins


class TestFlextConfigSingletonIntegration:
    """Test FlextConfig singleton pattern and integration with all modules."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        FlextConfig.clear_global_instance()
        FlextCore.reset_instance()

    def teardown_method(self) -> None:
        """Clear singleton instances after each test."""
        FlextConfig.clear_global_instance()
        FlextCore.reset_instance()

    def test_singleton_pattern(self) -> None:
        """Test that FlextConfig.get_global_instance() returns the same instance."""
        # Get config instance multiple times
        config1 = FlextConfig.get_global_instance()
        config2 = FlextConfig.get_global_instance()
        config3 = FlextConfig.get_global_instance()

        # All should be the same instance
        assert config1 is config2
        assert config2 is config3
        assert id(config1) == id(config2) == id(config3)

        # Test that the config has expected attributes
        assert hasattr(config1, "app_name")
        assert hasattr(config1, "environment")
        assert hasattr(config1, "log_level")
        assert hasattr(config1, "max_name_length")
        assert hasattr(config1, "min_phone_digits")

    def test_config_in_flext_core(self) -> None:
        """Test that FlextCore uses the global config singleton."""
        # Get global config
        global_config = FlextConfig.get_global_instance()

        # Get FlextCore instance
        core = FlextCore.get_instance()

        # FlextCore should use the same config
        assert core._config is global_config
        assert core.get_config() is global_config

    def test_config_in_flext_container(self) -> None:
        """Test that FlextContainer uses the global config singleton."""
        # Get global config
        global_config = FlextConfig.get_global_instance()

        # Create new container
        container = FlextContainer()

        # Container should have reference to global config
        assert container._global_config is global_config

    def test_config_in_validation_classes(self) -> None:
        """Test that validation classes use global config."""
        # Get global config
        global_config = FlextConfig.get_global_instance()

        # Create validator
        validator = FlextValidations.Domain.BaseValidator()

        # Validator should use config values
        assert validator._config is global_config
        assert validator.max_name_length == global_config.max_name_length
        assert validator.max_email_length == global_config.max_email_length

    def test_config_in_processors(self) -> None:
        """Test that processors use global config."""
        # Get global config
        global_config = FlextConfig.get_global_instance()

        # Create entry validator
        validator = FlextProcessors.EntryValidator()

        # Validator should use config values
        assert validator._config is global_config
        assert validator.max_name_length == global_config.max_name_length

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override default config."""
        # Clear any existing instance
        FlextConfig.clear_global_instance()

        # Set environment variables
        os.environ["FLEXT_APP_NAME"] = "test-app-from-env"
        os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"
        os.environ["FLEXT_MAX_NAME_LENGTH"] = "200"
        os.environ["FLEXT_MIN_PHONE_DIGITS"] = "7"
        os.environ["FLEXT_DEBUG"] = "true"

        try:
            # Get config (should load from env vars)
            config = FlextConfig.get_global_instance()

            # Check that env vars were loaded
            assert config.app_name == "test-app-from-env"
            assert config.log_level == "DEBUG"
            assert config.max_name_length == 200
            assert config.min_phone_digits == 7
            assert config.debug is True

        finally:
            # Cleanup
            del os.environ["FLEXT_APP_NAME"]
            del os.environ["FLEXT_LOG_LEVEL"]
            del os.environ["FLEXT_MAX_NAME_LENGTH"]
            del os.environ["FLEXT_MIN_PHONE_DIGITS"]
            del os.environ["FLEXT_DEBUG"]
            FlextConfig.clear_global_instance()

    def test_json_config_file_loading(self) -> None:
        """Test loading configuration from JSON file."""
        # Clear any existing instance
        FlextConfig.clear_global_instance()

        # Save and clear environment variable that might override
        saved_env = os.environ.pop("FLEXT_ENVIRONMENT", None)

        # Create temporary JSON config file
        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "app_name": "test-app-from-json",
                "environment": "test",
                "log_level": "WARNING",
                "max_name_length": 150,
                "cache_enabled": False,
            }
            json.dump(config_data, f)
            config_file = f.name

        # Initialize variables before try block to avoid unbound variable warnings
        original_dir = Path.cwd()
        temp_dir: str | None = None

        try:
            # Change to temp directory and create config.json
            temp_dir = tempfile.mkdtemp()
            os.chdir(temp_dir)

            # Copy config to config.json in current directory
            Path("config.json").write_text(json.dumps(config_data), encoding="utf-8")

            # Get config (should load from JSON)
            config = FlextConfig.get_global_instance()

            # Check that JSON values were loaded
            assert config.app_name == "test-app-from-json"
            assert config.environment == "test"
            assert config.log_level == "WARNING"
            assert config.max_name_length == 150
            assert config.cache_enabled is False

        finally:
            # Cleanup
            os.chdir(original_dir)
            Path(config_file).unlink(missing_ok=True)
            if temp_dir is not None:
                Path(temp_dir).joinpath("config.json").unlink(missing_ok=True)
                Path(temp_dir).rmdir()
            # Restore environment variable if it was set
            if saved_env is not None:
                os.environ["FLEXT_ENVIRONMENT"] = saved_env
            FlextConfig.clear_global_instance()

    def test_yaml_config_file_loading(self) -> None:
        """Test loading configuration from YAML file."""
        # Clear any existing instance
        FlextConfig.clear_global_instance()

        # Save and clear environment variable that might override
        saved_env = os.environ.pop("FLEXT_ENVIRONMENT", None)

        # Create temporary directory
        original_dir = Path.cwd()
        temp_dir = tempfile.mkdtemp()

        try:
            os.chdir(temp_dir)

            # Create YAML config file
            config_data = {
                "app_name": "test-app-from-yaml",
                "environment": "production",
                "debug": False,
                "command_timeout": 60,
                "validation_strict_mode": True,
            }

            with Path("config.yaml").open("w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            # Get config (should load from YAML)
            config = FlextConfig.get_global_instance()

            # Check that YAML values were loaded
            assert config.app_name == "test-app-from-yaml"
            assert config.environment == "production"
            assert config.debug is False
            assert config.command_timeout == 60
            assert config.validation_strict_mode is True

        finally:
            # Cleanup
            os.chdir(original_dir)
            Path(temp_dir).joinpath("config.yaml").unlink(missing_ok=True)
            Path(temp_dir).rmdir()
            # Restore environment variable if it was set
            if saved_env is not None:
                os.environ["FLEXT_ENVIRONMENT"] = saved_env
            FlextConfig.clear_global_instance()

    def test_config_priority_order(self) -> None:
        """Test that configuration sources have correct priority."""
        # Clear any existing instance
        FlextConfig.clear_global_instance()

        original_dir = Path.cwd()
        temp_dir = tempfile.mkdtemp()

        try:
            os.chdir(temp_dir)

            # 1. Create JSON config (lower priority)
            json_config = {"app_name": "from-json", "port": 3000}
            with Path("config.json").open("w", encoding="utf-8") as f:
                json.dump(json_config, f)

            # 2. Create .env file (medium priority)
            with Path(".env").open("w", encoding="utf-8") as f:
                f.write("FLEXT_APP_NAME=from-env\n")
                f.write("FLEXT_HOST=env-host\n")

            # 3. Set environment variable (highest priority)
            os.environ["FLEXT_APP_NAME"] = "from-env-var"

            # Get config
            config = FlextConfig.get_global_instance()

            # Check priority: env var > .env > json
            assert config.app_name == "from-env-var"  # From env var (highest)
            assert config.host == "env-host"  # From .env (no env var)
            assert config.port == 3000  # From JSON (no .env or env var)

        finally:
            # Cleanup
            os.chdir(original_dir)
            if "FLEXT_APP_NAME" in os.environ:
                del os.environ["FLEXT_APP_NAME"]
            Path(temp_dir).joinpath("config.json").unlink(missing_ok=True)
            Path(temp_dir).joinpath(".env").unlink(missing_ok=True)
            Path(temp_dir).rmdir()
            FlextConfig.clear_global_instance()

    def test_config_values_used_in_validation(self) -> None:
        """Test that config values are actually used in validation logic."""
        # Clear any existing instance
        FlextConfig.clear_global_instance()

        # Set custom config values via env vars
        os.environ["FLEXT_MAX_NAME_LENGTH"] = "50"
        os.environ["FLEXT_MIN_PHONE_DIGITS"] = "5"

        try:
            # Get fresh config
            config = FlextConfig.get_global_instance()
            assert config.max_name_length == 50
            assert config.min_phone_digits == 5

            # Test processor validation uses config
            validator = FlextProcessors.EntryValidator()

            # Create entry with name longer than configured max
            entry = FlextProcessors.Entry(
                entry_type="test",
                identifier="a" * 51,  # 51 chars, over limit of 50
                original_content="test data",
                clean_content="test content"
            )

            result = validator.validate_entry(entry)
            assert result.is_failure
            assert "50 characters or less" in str(result.error)

            # Test phone validation uses config
            # Phone with 4 digits (under minimum of 5)
            result = FlextMixins.validate_phone("1234")
            assert result.is_failure

            # Phone with 5 digits (meeting minimum)
            result = FlextMixins.validate_phone("12345")
            assert result.is_success

        finally:
            # Cleanup
            del os.environ["FLEXT_MAX_NAME_LENGTH"]
            del os.environ["FLEXT_MIN_PHONE_DIGITS"]
            FlextConfig.clear_global_instance()

    def test_config_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe."""
        # Clear any existing instance
        FlextConfig.clear_global_instance()

        configs = []

        def get_config() -> None:
            config = FlextConfig.get_global_instance()
            configs.append(config)

        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_config)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All configs should be the same instance
        assert len(configs) == 10
        first_config = configs[0]
        for config in configs[1:]:
            assert config is first_config

        # Cleanup
        FlextConfig.clear_global_instance()
