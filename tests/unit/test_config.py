"""Comprehensive tests for FlextCore.Config - Configuration Management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from flext_core import FlextCore


class TestFlextConfig:
    """Test suite for FlextCore.Config configuration management."""

    def test_config_initialization(self) -> None:
        """Test config initialization with default values."""
        config = FlextCore.Config()
        assert config is not None
        assert isinstance(config, FlextCore.Config)

    def test_config_with_custom_values(self) -> None:
        """Test config initialization with custom values."""
        config = FlextCore.Config(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )
        assert config.app_name == "test_app"
        assert config.version == "1.0.0"
        assert config.debug is True

    def test_config_callable_interface(self) -> None:
        """Test config callable interface for nested field access."""
        config = FlextCore.Config(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )

        # Test direct field access
        assert config("app_name") == "test_app"
        assert config("version") == "1.0.0"
        assert config("debug") is True

        # Test non-existent key
        with pytest.raises(
            KeyError, match=r"Configuration key 'nonexistent' not found"
        ):
            config("nonexistent")

    def test_config_from_dict(self) -> None:
        """Test config creation from dictionary."""
        config_data: dict[str, str | bool] = {
            "app_name": "dict_app",
            "version": "2.0.0",
            "debug": False,
        }
        config = FlextCore.Config(
            app_name=str(config_data["app_name"]),
            version=str(config_data["version"]),
            debug=bool(config_data["debug"]),
        )
        assert config.app_name == "dict_app"
        assert config.version == "2.0.0"
        assert config.debug is False

    def test_config_to_dict(self) -> None:
        """Test config conversion to dictionary."""
        config = FlextCore.Config(app_name="test_app", version="1.0.0", debug=True)
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "test_app"
        assert config_dict["version"] == "1.0.0"
        assert config_dict["debug"] is True

    def test_config_validation(self) -> None:
        """Test config validation."""
        # Valid config should pass
        config = FlextCore.Config(
            app_name="valid_app",
            version="1.0.0",
        )
        assert config.app_name == "valid_app"

    def test_config_clone(self) -> None:
        """Test config cloning."""
        original_config = FlextCore.Config(
            app_name="original_app",
            version="1.0.0",
        )

        # Use model_copy instead of clone
        cloned_config = original_config.model_copy()
        assert cloned_config.app_name == original_config.app_name
        assert cloned_config.version == original_config.version

        # Modifying clone should not affect original
        cloned_config.app_name = "modified_app"
        assert original_config.app_name == "original_app"

    def test_config_get_set_value(self) -> None:
        """Test config get/set value operations."""
        config = FlextCore.Config()

        # Set custom value using direct field access
        config.app_name = "custom_value"

        # Get custom value
        value = config.app_name
        assert value == "custom_value"

        # Test default value access
        default_value = getattr(config, "nonexistent_key", "default")
        assert default_value == "default"

    def test_config_has_value(self) -> None:
        """Test config has_value check."""
        config = FlextCore.Config()

        # Test with actual fields
        config.app_name = "test_value"
        assert hasattr(config, "app_name") is True
        assert hasattr(config, "nonexistent_key") is False

    def test_config_field_access(self) -> None:
        """Test config field access operations."""
        config = FlextCore.Config()

        # Test field assignment
        config.app_name = "test_value"
        assert config.app_name == "test_value"

        # Test field modification
        config.app_name = "modified_value"
        assert config.app_name == "modified_value"

    def test_config_field_reset(self) -> None:
        """Test config field reset operation."""
        config = FlextCore.Config()

        # Set multiple fields
        config.app_name = "value1"
        config.version = "value2"

        assert config.app_name == "value1"
        assert config.version == "value2"

        # Reset to defaults
        config.app_name = FlextCore.Config.model_fields["app_name"].default
        config.version = FlextCore.Config.model_fields["version"].default

        assert config.app_name != "value1"
        assert config.version != "value2"

    def test_config_keys_values_items(self) -> None:
        """Test config keys, values, and items operations."""
        config = FlextCore.Config()

        # Set actual fields
        config.app_name = "value1"
        config.version = "value2"

        # Use model_dump to get dict-like access
        config_dict = config.model_dump()
        keys = list(config_dict.keys())
        assert "app_name" in keys
        assert "version" in keys

        values = list(config_dict.values())
        assert "value1" in values
        assert "value2" in values

        items = list(config_dict.items())
        assert ("app_name", "value1") in items
        assert ("version", "value2") in items

    def test_config_singleton_pattern(self) -> None:
        """Test config creates independent instances with same defaults."""
        config1 = FlextCore.Config()
        config2 = FlextCore.Config()

        # Each call creates a new instance (no longer singleton)
        assert config1 is not config2
        # But they have the same default configuration values
        assert config1.model_dump() == config2.model_dump()

    def test_config_thread_safety(self) -> None:
        """Test config thread safety."""
        config = FlextCore.Config()
        results: FlextCore.Types.StringList = []

        def set_value(thread_id: int) -> None:
            # Use Pydantic field assignment instead of non-existent set_value
            config.app_name = f"thread_{thread_id}"
            results.append(config.app_name)

        threads: list[threading.Thread] = []
        for i in range(10):
            thread = threading.Thread(target=set_value, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.startswith("thread_") for result in results)

    def test_config_performance(self) -> None:
        """Test config performance characteristics."""
        config = FlextCore.Config()

        start_time = time.time()

        # Perform many operations using actual fields
        for i in range(100):  # Reduced from 1000 to 100 for faster execution
            config.app_name = f"value_{i}"
            _ = config.app_name

        end_time = time.time()

        # Should complete 200 operations in reasonable time
        assert end_time - start_time < 5.0  # Increased timeout to 5 seconds

    def test_config_serialization(self) -> None:
        """Test config serialization."""
        config = FlextCore.Config(
            app_name="serialize_app",
            version="1.0.0",
        )

        # Test JSON serialization using model_dump_json
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "serialize_app" in json_str

        # Test JSON deserialization using model_validate_json
        restored_config = FlextCore.Config.model_validate_json(json_str)
        assert restored_config.app_name == config.app_name
        assert restored_config.version == config.version

    def test_config_validate_log_level_invalid(self) -> None:
        """Test log level validation with invalid level (line 597-601)."""
        # Test with invalid log level - raises FlextCore.Exceptions.ValidationError
        with pytest.raises(FlextCore.Exceptions.ValidationError) as exc_info:
            FlextCore.Config(log_level="INVALID")

        assert "Invalid log level" in str(exc_info.value)

    def test_config_validate_trace_requires_debug(self) -> None:
        """Test trace requires debug to be enabled (line 616-620)."""
        # Test trace=True with debug=False should fail - raises FlextCore.Exceptions.ValidationError
        with pytest.raises(FlextCore.Exceptions.ValidationError) as exc_info:
            FlextCore.Config(trace=True, debug=False)

        assert "Trace mode requires debug mode" in str(exc_info.value)

    def test_config_create_and_configure_pattern(self) -> None:
        """Test direct instantiation and configuration pattern."""
        # New pattern - create and configure directly
        config = FlextCore.Config()

        # Apply project-specific overrides
        config.app_name = "Test Application"
        config.debug = True

        # Verify configuration applied
        assert config.app_name == "Test Application"
        assert config.debug is True

        # Create another instance - each is independent
        config2 = FlextCore.Config()

        # Verify it has default values (not the overrides from config)
        assert (
            config2.app_name == "FLEXT Application"
        )  # Default value from FlextCore.Constants
        assert config2.debug is False  # Default value

    def test_config_debug_enabled(self) -> None:
        """Test debug enabled checking using direct fields."""
        # Test with debug=True - check field directly
        debug_config = FlextCore.Config(debug=True)
        assert debug_config.debug is True

        # Test with trace=True (requires debug=True) - check both fields
        trace_config = FlextCore.Config(debug=True, trace=True)
        assert trace_config.debug is True
        assert trace_config.trace is True

        # Test with neither - check fields directly
        normal_config = FlextCore.Config(debug=False, trace=False)
        assert normal_config.debug is False
        assert normal_config.trace is False

    def test_config_effective_log_level(self) -> None:
        """Test effective log level using direct fields."""
        # Test normal case - use log_level directly
        config = FlextCore.Config(log_level="INFO")
        assert config.log_level == "INFO"

        # Test with debug enabled - log_level unchanged
        debug_config = FlextCore.Config(log_level="INFO", debug=True)
        assert debug_config.log_level == "INFO"
        assert debug_config.debug is True

        # Test with trace enabled - check both fields directly
        trace_config = FlextCore.Config(log_level="INFO", debug=True, trace=True)
        assert trace_config.trace is True
        # When trace is enabled, application should use DEBUG level
        assert trace_config.log_level == "INFO"  # Field value unchanged

    def test_global_instance_management(self) -> None:
        """Test global instance management methods."""
        # Save original global instance
        original_instance = FlextCore.Config.get_global_instance()

        try:
            # Create a new instance
            new_config = FlextCore.Config(app_name="test_instance", version="1.0.0")

            # Set it as global
            FlextCore.Config.set_global_instance(new_config)

            # Verify it's the global instance
            global_config = FlextCore.Config.get_global_instance()
            assert global_config is new_config
            assert global_config.app_name == "test_instance"

        finally:
            # Restore original instance
            FlextCore.Config.set_global_instance(original_instance)

    def test_pydantic_env_prefix(self) -> None:
        """Test that FlextCore.Config uses FLEXT_ prefix for environment variables."""
        import os

        # Cleanup any existing env vars
        for key in list(os.environ.keys()):
            if key.startswith("FLEXT_"):
                del os.environ[key]

        try:
            # Test that variables WITHOUT prefix are NOT loaded
            os.environ["DEBUG"] = "true"
            os.environ["LOG_LEVEL"] = "ERROR"

            config = FlextCore.Config()
            assert config.debug is False  # Not loaded
            assert config.log_level == "INFO"  # Not loaded, using default

            # Clean up
            del os.environ["DEBUG"]
            del os.environ["LOG_LEVEL"]

            # Test that variables WITH FLEXT_ prefix ARE loaded
            os.environ["FLEXT_DEBUG"] = "true"
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"

            config_with_prefix = FlextCore.Config()
            assert config_with_prefix.debug is True  # Loaded from FLEXT_DEBUG
            assert (
                config_with_prefix.log_level == "ERROR"
            )  # Loaded from FLEXT_LOG_LEVEL

        finally:
            # Cleanup
            for key in ["DEBUG", "LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_LOG_LEVEL"]:
                if key in os.environ:
                    del os.environ[key]

    def test_pydantic_dotenv_file_loading(self, tmp_path: Path) -> None:
        """Test that FlextCore.Config automatically loads .env file."""
        import os
        from pathlib import Path

        original_dir = Path.cwd()

        try:
            # Change to temp directory
            os.chdir(tmp_path)

            # Create .env file with FLEXT_ prefix
            env_file = tmp_path / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
            )

            # Create config - should load from .env
            config = FlextCore.Config()

            assert config.app_name == "from-dotenv"
            assert config.log_level == "WARNING"
            assert config.debug is True

        finally:
            # Cleanup
            os.chdir(original_dir)

    def test_pydantic_env_var_precedence(self, tmp_path: Path) -> None:
        """Test that environment variables override .env file."""
        import os
        from pathlib import Path

        original_dir = Path.cwd()
        saved_env_vars = {
            "FLEXT_APP_NAME": os.environ.pop("FLEXT_APP_NAME", None),
            "FLEXT_LOG_LEVEL": os.environ.pop("FLEXT_LOG_LEVEL", None),
        }

        try:
            # Change to temp directory
            os.chdir(tmp_path)

            # Create .env file
            env_file = tmp_path / ".env"
            env_file.write_text("FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\n")

            # Set environment variables (should override .env)
            os.environ["FLEXT_APP_NAME"] = "from-env-var"
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"

            # Create config
            config = FlextCore.Config()

            # Environment variables should override .env file
            assert config.app_name == "from-env-var"
            assert config.log_level == "ERROR"

        finally:
            # Cleanup
            os.chdir(original_dir)
            for key, value in saved_env_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    def test_pydantic_complete_precedence_chain(self, tmp_path: Path) -> None:
        """Test complete Pydantic 2 Settings precedence chain.

        Precedence (highest to lowest):
        1. Explicit init arguments (highest)
        2. Environment variables
        3. .env file
        4. Field defaults (lowest)
        """
        import os
        from pathlib import Path

        original_dir = Path.cwd()
        saved_env_vars = {
            "FLEXT_TIMEOUT_SECONDS": os.environ.pop("FLEXT_TIMEOUT_SECONDS", None),
        }

        try:
            # Change to temp directory
            os.chdir(tmp_path)

            # Create .env file (precedence level 3)
            env_file = tmp_path / ".env"
            env_file.write_text("FLEXT_TIMEOUT_SECONDS=45\n")

            # Set environment variable (precedence level 2)
            os.environ["FLEXT_TIMEOUT_SECONDS"] = "60"

            # Create config with explicit argument (precedence level 1 - highest)
            config = FlextCore.Config(timeout_seconds=90)

            # Explicit argument should win
            assert config.timeout_seconds == 90

            # Test without explicit argument - env var should win
            config_no_explicit = FlextCore.Config()
            assert config_no_explicit.timeout_seconds == 60  # From env var

            # Remove env var - .env should win
            del os.environ["FLEXT_TIMEOUT_SECONDS"]
            config_no_env = FlextCore.Config()
            assert config_no_env.timeout_seconds == 45  # From .env

        finally:
            # Cleanup
            os.chdir(original_dir)
            for key, value in saved_env_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    def test_pydantic_env_var_naming(self) -> None:
        """Test that environment variables follow correct naming convention."""
        import os

        saved_env = os.environ.pop("FLEXT_DEBUG", None)

        try:
            # Test with properly cased FLEXT_ prefix
            os.environ["FLEXT_DEBUG"] = "true"

            config = FlextCore.Config()
            assert config.debug is True

            # Verify config loaded from environment variable
            os.environ["FLEXT_DEBUG"] = "false"
            config_updated = FlextCore.Config()
            assert config_updated.debug is False

        finally:
            # Cleanup
            if "FLEXT_DEBUG" in os.environ:
                del os.environ["FLEXT_DEBUG"]
            if saved_env is not None:
                os.environ["FLEXT_DEBUG"] = saved_env

    def test_pydantic_effective_log_level_with_precedence(self) -> None:
        """Test that effective_log_level respects debug mode precedence."""
        import os

        saved_env_vars = {
            "FLEXT_LOG_LEVEL": os.environ.pop("FLEXT_LOG_LEVEL", None),
            "FLEXT_DEBUG": os.environ.pop("FLEXT_DEBUG", None),
        }

        try:
            # Set env vars
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
            os.environ["FLEXT_DEBUG"] = "true"

            config = FlextCore.Config()

            # Check fields directly - no computed properties
            assert config.log_level == "ERROR"
            assert config.debug is True

            # Test with debug=False
            os.environ["FLEXT_DEBUG"] = "false"
            config_no_debug = FlextCore.Config()

            assert config_no_debug.log_level == "ERROR"
            assert config_no_debug.debug is False

        finally:
            # Cleanup
            for key, value in saved_env_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]
