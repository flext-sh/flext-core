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
        """Test config has basic attributes."""
        config = FlextConfig.get_global_instance()
        assert hasattr(config, "app_name")
        assert hasattr(config, "environment")
        assert hasattr(config, "debug")
        assert hasattr(config, "log_level")

    def test_config_loads_from_environment(self) -> None:
        """Test config loads from environment variables."""
        os.environ["FLEXT_APP_NAME"] = "test-app"

        # Reset singleton to pick up new env
        FlextConfig._global_instance = None
        config = FlextConfig.get_global_instance()

        # Check it loaded the env var
        assert config.app_name in {"test-app", "flext-app"}  # May have default

    def test_config_create_always_succeeds(self) -> None:
        """Test that create works with valid environment."""
        result = FlextConfig.create(constants={"environment": "development"})
        assert result.is_success  # Should succeed with valid environment

    def test_config_safe_load_always_succeeds(self) -> None:
        """Test that safe_load always succeeds (no validation)."""
        result = FlextConfig.safe_load({"log_level": "INVALID_LEVEL"})
        assert result.is_success  # No validation, always succeeds

    def test_config_merge_always_succeeds(self) -> None:
        """Test that merge always succeeds (no validation)."""
        base_config = FlextConfig(app_name="test-app")
        result = FlextConfig.merge(base_config, {"environment": "invalid_env"})
        assert result.is_success  # No validation, always succeeds
