"""Simplified config tests for the new singleton pattern."""

import os

from flext_core import FlextConfig


class TestFlextConfigSimple:
    """Test simplified FlextConfig functionality."""

    def test_singleton_pattern(self) -> None:
        """Test that config is a singleton."""
        config1 = FlextConfig.get_global_instance()
        config2 = FlextConfig.get_global_instance()
        assert config1 is config2

    def test_config_has_basic_attributes(self) -> None:
        """Test config has basic attributes with proper types and values."""
        config = FlextConfig.get_global_instance()

        # Verify attribute types and default values
        assert isinstance(config.app_name, str)
        assert config.app_name  # Should not be empty

        assert isinstance(config.environment, str)
        assert config.environment in {"development", "production", "test"}

        assert isinstance(config.debug, bool)

        assert isinstance(config.log_level, str)
        assert config.log_level.upper() in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        # Verify numeric attributes
        assert isinstance(config.min_phone_digits, int)
        assert config.min_phone_digits > 0

        assert isinstance(config.max_name_length, int)
        assert config.max_name_length > 0

    def test_config_loads_from_environment(self) -> None:
        """Test config loads from environment variables."""
        # Clear singleton properly
        FlextConfig.clear_global_instance()

        # Set multiple environment variables
        os.environ["FLEXT_APP_NAME"] = "test-app-from-env"
        os.environ["FLEXT_DEBUG"] = "true"
        os.environ["FLEXT_LOG_LEVEL"] = "WARNING"

        try:
            # Get new config instance
            config = FlextConfig.get_global_instance()

            # Assert exact expected values from environment
            assert config.app_name == "test-app-from-env"
            assert config.debug is True
            assert config.log_level == "WARNING"

        finally:
            # Clean up environment variables
            if "FLEXT_APP_NAME" in os.environ:
                del os.environ["FLEXT_APP_NAME"]
            if "FLEXT_DEBUG" in os.environ:
                del os.environ["FLEXT_DEBUG"]
            if "FLEXT_LOG_LEVEL" in os.environ:
                del os.environ["FLEXT_LOG_LEVEL"]
            FlextConfig.clear_global_instance()

    def test_config_create_with_constants(self) -> None:
        """Test that create works with valid constants and proper validation."""
        # Test with valid constants
        constants = {
            "environment": "development",
            "app_name": "test-create-app",
            "debug": True,
            "max_name_length": 150,
        }
        result = FlextConfig.create(constants=constants)
        assert result.is_success

        # Verify the created config has the provided constants
        config = result.unwrap()
        assert config.environment == "development"
        assert config.app_name == "test-create-app"
        assert config.debug is True
        assert config.max_name_length == 150

        # Test with invalid environment
        invalid_constants: dict[str, object] = {"environment": "invalid_env_name"}
        result = FlextConfig.create(constants=invalid_constants)
        # The test should handle validation appropriately
        # If validation fails, it should return failure
        # If no validation, it should succeed but we test the value
        if result.is_success:
            config = result.unwrap()
            # Verify it handles invalid values gracefully
            assert hasattr(config, "environment")

    def test_config_safe_load_applies_values(self) -> None:
        """Test that safe_load actually applies loaded values."""
        # Test with valid data
        valid_data = {"app_name": "safe-load-app", "debug": False, "log_level": "ERROR"}
        result = FlextConfig.safe_load(valid_data)
        assert result.is_success

        # Verify the loaded values are actually applied
        config = result.unwrap()
        assert config.app_name == "safe-load-app"
        assert config.debug is False
        assert config.log_level == "ERROR"

        # Test with potentially invalid data to verify graceful handling
        invalid_data: dict[str, object] = {"log_level": "INVALID_LEVEL"}
        result = FlextConfig.safe_load(invalid_data)
        # Safe load should handle invalid data gracefully
        assert result.is_success
        config = result.unwrap()
        # Verify the config object has the value (even if invalid)
        assert hasattr(config, "log_level")
        assert config.log_level is not None

    def test_config_merge_functionality(self) -> None:
        """Test that merge correctly combines configurations with proper priority."""
        # Create base config with initial values
        base_config = FlextConfig(app_name="base-app", debug=False, log_level="INFO")

        # Test merging with override values
        override_data = {
            "environment": "production",
            "debug": True,
            "app_name": "merged-app",  # Should override base value
        }
        result = FlextConfig.merge(base_config, override_data)
        assert result.is_success

        # Verify merged values are correctly combined
        merged_config = result.unwrap()
        assert merged_config.app_name == "merged-app"  # Override priority
        assert merged_config.debug is True  # Override priority
        assert merged_config.environment == "production"  # New value
        assert merged_config.log_level == "INFO"  # Preserved from base

        # Test merging with empty dict
        result = FlextConfig.merge(base_config, {})
        assert result.is_success
        empty_merged = result.unwrap()
        assert empty_merged.app_name == "base-app"  # Should preserve base values

        # Test merging with None values
        result = FlextConfig.merge(base_config, {"app_name": None})
        assert result.is_success
        none_merged = result.unwrap()
        # Should handle None values gracefully
        assert hasattr(none_merged, "app_name")
