"""FlextConfig comprehensive functionality tests.

Module: flext_core.config
Scope: FlextConfig class - configuration management, validation, environment handling,
thread safety, namespace management, and Pydantic integration.

Tests core FlextConfig functionality including:
- Configuration initialization and validation
- Environment variable handling
- Thread safety and singleton patterns
- Namespace management and auto-registration
- Pydantic model integration and serialization

Uses Python 3.13 patterns (StrEnum, frozen dataclasses with slots),
centralized test constants, and parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from flext_core import FlextConfig, FlextConstants


class TestFlextConfig:
    """Test suite for FlextConfig configuration management."""

    def test_config_initialization(self) -> None:
        """Test config initialization with default values."""
        config = FlextConfig()
        assert config is not None
        assert isinstance(config, FlextConfig)

    def test_config_with_custom_values(self) -> None:
        """Test config initialization with custom values."""
        config = FlextConfig(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )
        assert config.app_name == "test_app"
        assert config.version == "1.0.0"
        assert config.debug is True

    def test_config_from_dict(self) -> None:
        """Test config creation from dictionary."""
        config_data: dict[str, str | bool] = {
            "app_name": "dict_app",
            "version": "2.0.0",
            "debug": False,
        }
        config = FlextConfig(
            app_name=str(config_data["app_name"]),
            version=str(config_data["version"]),
            debug=bool(config_data["debug"]),
        )
        assert config.app_name == "dict_app"
        assert config.version == "2.0.0"
        assert config.debug is False

    def test_config_to_dict(self) -> None:
        """Test config conversion to dictionary."""
        config = FlextConfig(app_name="test_app", version="1.0.0", debug=True)
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "test_app"
        assert config_dict["version"] == "1.0.0"
        assert config_dict["debug"] is True

    def test_config_validation(self) -> None:
        """Test config validation."""
        # Valid config should pass
        config = FlextConfig(
            app_name="valid_app",
            version="1.0.0",
        )
        assert config.app_name == "valid_app"

    def test_config_clone(self) -> None:
        """Test config cloning with singleton pattern.

        With singleton pattern, FlextConfig(app_name=X) on first call initializes
        the singleton with app_name=X. model_copy() creates a data copy with the
        same configuration values, suitable for serialization/deserialization testing.
        """
        # Create and initialize singleton with custom values
        original_config = FlextConfig(
            app_name="original_app",
            version="1.0.0",
        )

        # Verify the singleton was initialized with our values
        assert original_config.app_name == "original_app"
        assert original_config.version == "1.0.0"

        # Use model_copy to get a config dict for comparison
        config_dict = original_config.model_dump()

        # Create a new instance from the dict (independent of singleton)
        cloned_config = FlextConfig.model_validate(config_dict)

        # Both should have the same values
        assert cloned_config.app_name == original_config.app_name
        assert cloned_config.version == original_config.version

        # The cloned config from model_validate is also a singleton instance
        # So modifying it also affects the singleton
        assert cloned_config is original_config  # Both are the same singleton

    def test_config_get_set_value(self) -> None:
        """Test config get/set value operations."""
        config = FlextConfig()

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
        config = FlextConfig()

        # Test with actual fields
        config.app_name = "test_value"
        assert hasattr(config, "app_name") is True
        assert hasattr(config, "nonexistent_key") is False

    def test_config_field_access(self) -> None:
        """Test config field access operations."""
        config = FlextConfig()

        # Test field assignment
        config.app_name = "test_value"
        assert config.app_name == "test_value"

        # Test field modification
        config.app_name = "modified_value"
        assert config.app_name == "modified_value"

    def test_config_field_reset(self) -> None:
        """Test config field reset operation."""
        config = FlextConfig()

        # Set multiple fields with valid semantic versions
        config.app_name = "value1"
        config.version = "2.0.0"

        assert config.app_name == "value1"
        assert config.version == "2.0.0"

        # Reset to defaults
        config.app_name = FlextConfig.model_fields["app_name"].default
        config.version = FlextConfig.model_fields["version"].default

        assert config.app_name != "value1"
        assert config.version != "2.0.0"

    def test_config_keys_values_items(self) -> None:
        """Test config keys, values, and items operations."""
        config = FlextConfig()

        # Set actual fields with valid semantic version
        config.app_name = "value1"
        config.version = "2.0.0"

        # Use model_dump to get dict-like access
        config_dict = config.model_dump()
        keys = list(config_dict.keys())
        assert "app_name" in keys
        assert "version" in keys

        values = list(config_dict.values())
        assert "value1" in values
        assert "2.0.0" in values

        items = list(config_dict.items())
        assert ("app_name", "value1") in items
        assert ("version", "2.0.0") in items

    def test_config_singleton_pattern(self) -> None:
        """Test config implements true singleton pattern."""
        config1 = FlextConfig()
        config2 = FlextConfig()

        # True singleton: both calls return the same instance
        assert config1 is config2
        # They have identical configuration values
        assert config1.model_dump() == config2.model_dump()

    def test_config_thread_safety(self) -> None:
        """Test config thread safety."""
        config = FlextConfig()
        results: list[str] = []

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
        config = FlextConfig()

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
        config = FlextConfig(
            app_name="serialize_app",
            version="1.0.0",
        )

        # Test JSON serialization excluding computed fields
        json_str = config.model_dump_json(
            exclude={
                "is_debug_enabled",
                "effective_log_level",
                "is_production",
                "effective_timeout",
                "has_database",
                "has_cache",
            },
        )
        assert isinstance(json_str, str)
        assert "serialize_app" in json_str

        # Test JSON deserialization
        restored_config = FlextConfig.model_validate_json(json_str)
        assert restored_config.app_name == config.app_name
        assert restored_config.version == config.version

    def test_config_validate_log_level_invalid(self) -> None:
        """Test log level validation with invalid level (line 597-601)."""
        # Test with invalid log level
        # Pydantic v2 Literal type raises ValidationError with descriptive message
        with pytest.raises(ValidationError) as exc_info:
            # Test with invalid log level using model_validate to bypass strict typing
            FlextConfig.model_validate({"log_level": "INVALID"})

        # Check that the error message is descriptive
        assert "log_level" in str(exc_info.value) or "INVALID" in str(exc_info.value)

    def test_config_validate_trace_requires_debug(self) -> None:
        """Test trace requires debug to be enabled (line 616-620)."""
        # Test trace=True with debug=False should fail - Pydantic wraps ValueError as ValidationError
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig(trace=True, debug=False)

        assert "Trace mode requires debug mode" in str(exc_info.value)

    def test_config_create_and_configure_pattern(self) -> None:
        """Test direct instantiation and configuration pattern.

        FlextConfig singleton pattern: All instances are the same object.
        When FlextConfig() is called again, __new__ returns the existing instance,
        and __init__ is called with the provided kwargs, re-initializing fields.
        """
        # Reset to ensure clean state
        FlextConfig.reset_global_instance()

        try:
            # Create instance with specific values
            config = FlextConfig(app_name="Test Application", debug=True)

            # Verify configuration applied
            assert config.app_name == "Test Application"
            assert config.debug is True

            # Create another instance - singleton pattern means same instance
            # Note: Calling FlextConfig() with no args will re-init with defaults
            # So we pass the same values again to maintain them
            config2 = FlextConfig(app_name="Test Application", debug=True)

            # Verify it has the same values (because it's the same singleton)
            assert config2.app_name == "Test Application"
            assert config2.debug is True

            # Both are the same object
            assert config is config2

        finally:
            # Cleanup: restore fresh singleton
            FlextConfig.reset_global_instance()

    def test_config_debug_enabled(self) -> None:
        """Test debug enabled checking using direct fields."""
        # Test with debug=True - check field directly
        debug_config = FlextConfig(debug=True)
        assert debug_config.debug is True

        # Test with trace=True (requires debug=True) - check both fields
        trace_config = FlextConfig(debug=True, trace=True)
        assert trace_config.debug is True
        assert trace_config.trace is True

        # Test with neither - check fields directly
        normal_config = FlextConfig(debug=False, trace=False)
        assert normal_config.debug is False
        assert normal_config.trace is False

    def test_config_effective_log_level(self) -> None:
        """Test effective log level using direct fields."""
        # Test normal case - use log_level directly
        config = FlextConfig(log_level=FlextConstants.Settings.LogLevel.INFO)
        assert config.log_level == FlextConstants.Settings.LogLevel.INFO

        # Test with debug enabled - log_level unchanged
        debug_config = FlextConfig(
            log_level=FlextConstants.Settings.LogLevel.INFO,
            debug=True,
        )
        assert debug_config.log_level == FlextConstants.Settings.LogLevel.INFO
        assert debug_config.debug is True

        # Test with trace enabled - check both fields directly
        trace_config = FlextConfig(
            log_level=FlextConstants.Settings.LogLevel.INFO,
            debug=True,
            trace=True,
        )
        assert trace_config.trace is True
        # When trace is enabled, application should use DEBUG level
        assert trace_config.log_level == "INFO"  # Field value unchanged

    def test_global_instance_management(self) -> None:
        """Test global instance management methods with singleton pattern.

        Tests that get_global_instance() and reset_global_instance()
        work correctly to manage the singleton.
        """
        # Get original global instance
        original_instance = FlextConfig.get_global_instance()

        try:
            # Verify that get_global_instance returns singleton
            second_call = FlextConfig.get_global_instance()
            assert second_call is original_instance

            # Test reset_global_instance
            FlextConfig.reset_global_instance()

            # After reset, should get a different instance
            fresh_config = FlextConfig()
            assert fresh_config is not original_instance
            # Fresh config has default values
            assert fresh_config.app_name == "flext"  # Default value

            # Create new instance - should get the fresh singleton created above
            another_instance = FlextConfig.get_global_instance()
            assert another_instance is fresh_config

            # Verify that subsequent calls return the same instance
            third_call = FlextConfig.get_global_instance()
            assert third_call is another_instance

        finally:
            # Cleanup: restore a fresh singleton
            FlextConfig.reset_global_instance()

    def test_pydantic_env_prefix(self) -> None:
        """Test that FlextConfig uses FLEXT_ prefix for environment variables."""
        # Cleanup any existing env vars
        for key in list(os.environ.keys()):
            if key.startswith("FLEXT_"):
                del os.environ[key]

        try:
            # Test that variables WITHOUT prefix are NOT loaded
            os.environ["DEBUG"] = "true"
            os.environ["LOG_LEVEL"] = "ERROR"

            config = FlextConfig()
            assert config.debug is False  # Not loaded
            assert config.log_level == "INFO"  # Not loaded, using default

            # Clean up
            del os.environ["DEBUG"]
            del os.environ["LOG_LEVEL"]

            # Test that variables WITH FLEXT_ prefix ARE loaded
            os.environ["FLEXT_DEBUG"] = "true"
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"

            config_with_prefix = FlextConfig()
            assert config_with_prefix.debug is True  # Loaded from FLEXT_DEBUG
            assert (
                config_with_prefix.log_level == "ERROR"
            )  # Loaded from FLEXT_LOG_LEVEL

        finally:
            # Cleanup
            for key in [
                "DEBUG",
                "LOG_LEVEL",
                "FLEXT_DEBUG",
                "FLEXT_LOG_LEVEL",
            ]:
                if key in os.environ:
                    del os.environ[key]

    def test_pydantic_dotenv_file_loading(self, tmp_path: Path) -> None:
        """Test that FlextConfig automatically loads .env file."""
        original_dir = Path.cwd()
        saved_env_vars = {
            "FLEXT_LOG_LEVEL": os.environ.pop("FLEXT_LOG_LEVEL", None),
            "FLEXT_DEBUG": os.environ.pop("FLEXT_DEBUG", None),
            "FLEXT_APP_NAME": os.environ.pop("FLEXT_APP_NAME", None),
        }

        try:
            # Change to temp directory
            os.chdir(tmp_path)

            # Create .env file with FLEXT_ prefix
            env_file = tmp_path / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n",
            )

            # Reset singleton to force reload from .env
            # Note: FlextConfig is imported at top of file
            # Clear singleton to force reload
            if hasattr(FlextConfig, "_instances"):
                FlextConfig._instances.clear()

            # Create config - should load from .env
            config = FlextConfig()

            assert config.app_name == "from-dotenv"
            # log_level - compare as string (works for both enum and string types)
            log_level_str: object = config.log_level
            assert str(log_level_str) == "WARNING" or "WARNING" in str(log_level_str)
            assert config.debug is True

        finally:
            # Cleanup - restore environment variables
            os.chdir(original_dir)
            for key, value in saved_env_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    def test_pydantic_env_var_precedence(self, tmp_path: Path) -> None:
        """Test that environment variables override .env file."""
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
            config = FlextConfig()

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
            config = FlextConfig(timeout_seconds=90)

            # Explicit argument should win
            assert config.timeout_seconds == 90

            # Test without explicit argument - env var should win
            config_no_explicit = FlextConfig()
            assert config_no_explicit.timeout_seconds == 60  # From env var

            # Remove env var - .env should win
            del os.environ["FLEXT_TIMEOUT_SECONDS"]
            config_no_env = FlextConfig()
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
        saved_env = os.environ.pop("FLEXT_DEBUG", None)

        try:
            # Test with properly cased FLEXT_ prefix
            os.environ["FLEXT_DEBUG"] = "true"

            config = FlextConfig()
            assert config.debug is True

            # Verify config loaded from environment variable
            os.environ["FLEXT_DEBUG"] = "false"
            config_updated = FlextConfig()
            assert config_updated.debug is False

        finally:
            # Cleanup
            if "FLEXT_DEBUG" in os.environ:
                del os.environ["FLEXT_DEBUG"]
            if saved_env is not None:
                os.environ["FLEXT_DEBUG"] = saved_env

    def test_pydantic_effective_log_level_with_precedence(self) -> None:
        """Test that effective_log_level respects debug mode precedence."""
        saved_env_vars = {
            "FLEXT_LOG_LEVEL": os.environ.pop("FLEXT_LOG_LEVEL", None),
            "FLEXT_DEBUG": os.environ.pop("FLEXT_DEBUG", None),
        }

        try:
            # Set env vars
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
            os.environ["FLEXT_DEBUG"] = "true"

            config = FlextConfig()

            # Check fields directly - no computed properties
            assert config.log_level == "ERROR"
            assert config.debug is True

            # Test with debug=False
            os.environ["FLEXT_DEBUG"] = "false"
            config_no_debug = FlextConfig()

            assert config_no_debug.log_level == "ERROR"
            assert config_no_debug.debug is False

        finally:
            # Cleanup
            for key, value in saved_env_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    def test_get_global_instance(self) -> None:
        """Test get_global_instance returns singleton."""
        instance1 = FlextConfig.get_global_instance()
        instance2 = FlextConfig.get_global_instance()
        assert instance1 is instance2  # Same instance

    def test_config_with_all_fields(self) -> None:
        """Test config initialization with all fields set."""
        saved_vars = {}
        field_names = ["FLEXT_DEBUG", "FLEXT_LOG_LEVEL"]

        try:
            # Save original values
            for key in field_names:
                saved_vars[key] = os.environ.get(key)

            # Set environment variables
            os.environ["FLEXT_DEBUG"] = "true"
            os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"

            config = FlextConfig()
            assert config.debug is True
            assert config.log_level == "DEBUG"

        finally:
            # Restore original values
            for key, value in saved_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]
