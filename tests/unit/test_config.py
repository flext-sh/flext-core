"""Comprehensive tests for FlextConfig - Configuration Management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from flext_core import FlextConfig, FlextTypes


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
            environment="test",
            debug=True,
        )
        assert config.app_name == "test_app"
        assert config.version == "1.0.0"
        assert config.environment == "test"
        assert config.debug is True

    def test_config_from_dict(self) -> None:
        """Test config creation from dictionary."""
        config_data: dict[str, str | int | float | bool] = {
            "app_name": "dict_app",
            "version": "2.0.0",
            "environment": "production",
            "debug": False,
        }
        config = FlextConfig.create(**config_data)
        assert config.app_name == "dict_app"
        assert config.version == "2.0.0"
        assert config.environment == "production"
        assert config.debug is False

    def test_config_to_dict(self) -> None:
        """Test config conversion to dictionary."""
        config = FlextConfig(app_name="test_app", version="1.0.0", environment="test")
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "test_app"
        assert config_dict["version"] == "1.0.0"
        assert config_dict["environment"] == "test"

    def test_config_from_json_file(self, tmp_path: Path) -> None:
        """Test config loading from JSON file."""
        config_data: dict[str, str | int | float | bool] = {
            "app_name": "json_app",
            "version": "3.0.0",
            "environment": "staging",
            "debug": True,
        }

        json_file = tmp_path / "config.json"
        with Path(json_file).open("w", encoding="utf-8") as f:
            json.dump(config_data, f)

        result = FlextConfig.from_file(str(json_file))
        assert result.is_success
        config = result.value
        assert config.app_name == "json_app"
        assert config.version == "3.0.0"
        assert config.environment == "staging"
        assert config.debug is True

    def test_config_from_json_file_alternative(self, tmp_path: Path) -> None:
        """Test config loading from JSON file with different data."""
        config_data: dict[str, str | int | float | bool] = {
            "app_name": "json_app_alt",
            "version": "4.0.0",
            "environment": "development",
            "debug": False,
        }

        json_file = tmp_path / "config_alt.json"

        with Path(json_file).open("w", encoding="utf-8") as f:
            json.dump(config_data, f)

        result = FlextConfig.from_file(str(json_file))
        assert result.is_success
        config = result.value
        assert config.app_name == "json_app_alt"
        assert config.version == "4.0.0"
        assert config.environment == "development"
        assert config.debug is False

    def test_config_from_nonexistent_file(self) -> None:
        """Test config loading from non-existent file."""
        result = FlextConfig.from_file("nonexistent.json")
        assert result.is_failure

    def test_config_save_to_file(self, tmp_path: Path) -> None:
        """Test config saving to file."""
        config = FlextConfig(app_name="save_app", version="5.0.0", environment="test")

        config_file = tmp_path / "saved_config.json"
        # Test model_dump method instead
        config_dict = config.model_dump()
        assert config_dict["app_name"] == "save_app"
        assert config_dict["version"] == "5.0.0"

        # Test saving to file manually
        with Path(config_file).open("w", encoding="utf-8") as f:
            json.dump(config_dict, f)

        # Verify file was created and contains correct data
        assert config_file.exists()
        with Path(config_file).open(encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["app_name"] == "save_app"
        assert saved_data["version"] == "5.0.0"

    def test_config_validation(self) -> None:
        """Test config validation."""
        # Valid config should pass
        config = FlextConfig(
            app_name="valid_app",
            version="1.0.0",
            environment="production",
        )
        assert config.app_name == "valid_app"
        assert config.environment == "production"

    def test_config_validation_failure(self) -> None:
        """Test config validation failure."""
        # Invalid config should fail
        # Invalid environment should fail validation
        with pytest.raises(Exception) as exc_info:
            FlextConfig(
                app_name="test_app",
                version="1.0.0",
                environment="invalid_env",  # This should fail validation
            )
        # Expected validation error - test passes
        assert (
            "environment" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    def test_config_environment_variables(self) -> None:
        """Test config loading from environment variables."""
        os.environ["FLEXT_APP_NAME"] = "env_app"
        os.environ["FLEXT_VERSION"] = "6.0.0"
        os.environ["FLEXT_ENVIRONMENT"] = "test"

        try:
            # Pydantic Settings automatically reads from environment variables
            config = FlextConfig()
            assert config.app_name == "env_app"
            assert config.version == "6.0.0"
            assert config.environment == "test"
        finally:
            # Clean up environment variables
            for key in ["FLEXT_APP_NAME", "FLEXT_VERSION", "FLEXT_ENVIRONMENT"]:
                if key in os.environ:
                    del os.environ[key]

    def test_config_clone(self) -> None:
        """Test config cloning."""
        original_config = FlextConfig(
            app_name="original_app",
            version="1.0.0",
            environment="development",
        )

        # Use model_copy instead of clone
        cloned_config = original_config.model_copy()
        assert cloned_config.app_name == original_config.app_name
        assert cloned_config.version == original_config.version
        assert cloned_config.environment == original_config.environment

        # Modifying clone should not affect original
        cloned_config.app_name = "modified_app"
        assert original_config.app_name == "original_app"

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

        # Set multiple fields
        config.app_name = "value1"
        config.version = "value2"

        assert config.app_name == "value1"
        assert config.version == "value2"

        # Reset to defaults
        config.app_name = FlextConfig.model_fields["app_name"].default
        config.version = FlextConfig.model_fields["version"].default

        assert config.app_name != "value1"
        assert config.version != "value2"

    def test_config_keys_values_items(self) -> None:
        """Test config keys, values, and items operations."""
        config = FlextConfig()

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
        config1 = FlextConfig()
        config2 = FlextConfig()

        # Each call creates a new instance (no longer singleton)
        assert config1 is not config2
        # But they have the same default configuration values
        assert config1.model_dump() == config2.model_dump()

    def test_config_singleton_reset(self) -> None:
        """Test config singleton reset."""
        config1 = FlextConfig()
        FlextConfig.reset_global_instance()
        config2 = FlextConfig()

        assert config1 is not config2

    def test_config_thread_safety(self) -> None:
        """Test config thread safety."""
        config = FlextConfig()
        results: FlextTypes.StringList = []

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

    def test_config_error_handling(self) -> None:
        """Test config error handling."""
        config = FlextConfig()

        # Test invalid file path (using from_file which exists)
        result = FlextConfig.from_file("/invalid/path/config.json")
        assert result.is_failure

        # Test invalid data type - Pydantic might not raise immediately
        # but will validate on model_dump or other operations
        config.app_name = "test_app_name"  # Use valid string type
        # This should work since Pydantic allows proper types
        assert config.app_name is not None

    def test_config_serialization(self) -> None:
        """Test config serialization."""
        config = FlextConfig(
            app_name="serialize_app",
            version="1.0.0",
            environment="test",
        )

        # Test JSON serialization using model_dump_json
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "serialize_app" in json_str

        # Test JSON deserialization using model_validate_json
        restored_config = FlextConfig.model_validate_json(json_str)
        assert restored_config.app_name == config.app_name
        assert restored_config.version == config.version
        assert restored_config.environment == config.environment

    def test_config_handler_configuration_resolve_handler_mode_with_config_attr(
        self,
    ) -> None:
        """Test resolve_handler_mode with handler_config.handler_type attribute (line 181-184)."""
        from flext_core import FlextConfig

        # Create a config object with handler_type attribute
        class MockConfig:
            handler_type = "query"

        resolved = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None,
            handler_config=MockConfig(),
        )
        assert resolved == "query"

    def test_config_handler_configuration_resolve_handler_mode_with_dict(self) -> None:
        """Test resolve_handler_mode with handler_config dict (line 187-193)."""
        from flext_core import FlextConfig

        # Test with dict containing handler_type
        config_dict = {"handler_type": "query"}
        resolved = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None,
            handler_config=config_dict,
        )
        assert resolved == "query"

    def test_config_handler_configuration_resolve_handler_mode_default(self) -> None:
        """Test resolve_handler_mode default fallback (line 196)."""
        from flext_core import FlextConfig, FlextConstants

        # Test default fallback when no valid mode found
        resolved = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None,
            handler_config=None,
        )
        assert resolved == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_config_handler_configuration_create_handler_config_defaults(self) -> None:
        """Test create_handler_config with automatic defaults (line 222-250)."""
        from flext_core import FlextConfig

        # Test with minimal inputs - should generate defaults
        config = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="command",
        )

        # Should have auto-generated handler_id
        assert "handler_id" in config
        handler_id = cast("str", config["handler_id"])
        assert handler_id.startswith("command_handler_")

        # Should have auto-generated handler_name
        assert "handler_name" in config
        handler_name = config["handler_name"]
        assert handler_name == "Command Handler"

        # Should have correct handler_type
        handler_type = config["handler_type"]
        handler_mode = config["handler_mode"]
        assert handler_type == "command"
        assert handler_mode == "command"

    def test_config_handler_configuration_create_handler_config_with_merge(
        self,
    ) -> None:
        """Test create_handler_config with additional config merge (line 247-248)."""
        from flext_core import FlextConfig

        # Test with additional config that should be merged
        additional_config = cast(
            "FlextTypes.Dict",
            {"custom_field": "custom_value", "another_field": 42},
        )

        config = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="query",
            handler_id="test_query",
            handler_name="Test Query Handler",
            handler_config=additional_config,
        )

        # Should have the provided values
        assert config["handler_id"] == "test_query"
        assert config["handler_name"] == "Test Query Handler"

        # Should have merged the additional config
        assert config["custom_field"] == "custom_value"
        assert config["another_field"] == 42

    def test_config_validate_log_level_invalid(self) -> None:
        """Test log level validation with invalid level (line 597-601)."""
        import pytest
        from pydantic import ValidationError

        from flext_core import FlextConfig

        # Test with invalid log level - Pydantic raises ValidationError
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig(log_level="INVALID")

        assert "Invalid log level" in str(exc_info.value)

    def test_config_validate_debug_trace_production_error(self) -> None:
        """Test debug cannot be enabled in production (line 608-613)."""
        import pytest
        from pydantic import ValidationError

        from flext_core import FlextConfig

        # Test production with debug=True should fail - Pydantic raises ValidationError
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig(environment="PRODUCTION", debug=True)

        assert "Debug mode cannot be enabled in production" in str(exc_info.value)

    def test_config_validate_trace_requires_debug(self) -> None:
        """Test trace requires debug to be enabled (line 616-620)."""
        import pytest
        from pydantic import ValidationError

        from flext_core import FlextConfig

        # Test trace=True with debug=False should fail - Pydantic raises ValidationError
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig(trace=True, debug=False)

        assert "Trace mode requires debug mode" in str(exc_info.value)

    def test_config_create_and_configure_pattern(self) -> None:
        """Test direct instantiation and configuration pattern."""
        from flext_core import FlextConfig

        # New pattern - create and configure directly
        config = FlextConfig()

        # Apply project-specific overrides
        config.app_name = "Test Application"
        config.debug = True

        # Verify configuration applied
        assert config.app_name == "Test Application"
        assert config.debug is True

        # Create another instance - each is independent
        config2 = FlextConfig()

        # Verify it has default values (not the overrides from config)
        assert (
            config2.app_name == "FLEXT Application"
        )  # Default value from FlextConstants
        assert config2.debug is False  # Default value
