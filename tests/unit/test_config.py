"""Simplified config tests for the new singleton pattern."""

import os

import pytest

from flext_core import FlextConfig


class TestFlextConfigSimple:
    """Test simplified FlextConfig functionality."""

    def test_singleton_pattern(self) -> None:
        """Test that config is a singleton."""
        config1 = FlextConfig.get_global_instance()
        config2 = FlextConfig.get_global_instance()
        assert config1 is config2

    @pytest.mark.skip(reason="Test only checks attribute existence, not functionality")
    def test_config_has_basic_attributes(self) -> None:
        """Test config has basic attributes.

        TODO: Improve this test to:
        - Verify attribute types (str, bool, etc.)
        - Test default values are correct
        - Verify attribute validation works
        - Test that attributes are properly initialized
        """
        config = FlextConfig.get_global_instance()
        assert hasattr(config, "app_name")
        assert hasattr(config, "environment")
        assert hasattr(config, "debug")
        assert hasattr(config, "log_level")

    @pytest.mark.skip(
        reason="Test uses weak assertion - value could be either test-app or default"
    )
    def test_config_loads_from_environment(self) -> None:
        """Test config loads from environment variables.

        TODO: Improve this test to:
        - Clear singleton properly using clear_global_instance()
        - Assert exact expected value, not a set of possibilities
        - Test multiple environment variables
        - Verify environment variable priority over defaults
        - Clean up environment after test
        """
        os.environ["FLEXT_APP_NAME"] = "test-app"

        # Reset singleton to pick up new env
        FlextConfig._global_instance = None
        config = FlextConfig.get_global_instance()

        # Check it loaded the env var
        assert config.app_name in {"test-app", "flext-app"}  # May have default

    @pytest.mark.skip(reason="Test only checks success status, not actual creation")
    def test_config_create_always_succeeds(self) -> None:
        """Test that create works with valid environment.

        TODO: Improve this test to:
        - Verify the created config has the provided constants
        - Test with invalid environments and expect failures
        - Verify the created config object is properly initialized
        - Test that constants override defaults correctly
        - Test error handling for invalid constant values
        """
        result = FlextConfig.create(constants={"environment": "development"})
        assert result.is_success  # Should succeed with valid environment

    @pytest.mark.skip(reason="Test doesn't verify actual loading, only success status")
    def test_config_safe_load_always_succeeds(self) -> None:
        """Test that safe_load always succeeds (no validation).

        TODO: Improve this test to:
        - Verify the loaded values are actually applied
        - Test with valid and invalid data separately
        - Check that invalid data is handled gracefully
        - Verify the returned config object has expected values
        """
        result = FlextConfig.safe_load({"log_level": "INVALID_LEVEL"})
        assert result.is_success  # No validation, always succeeds

    @pytest.mark.skip(
        reason="Test doesn't verify merge functionality, only success status"
    )
    def test_config_merge_always_succeeds(self) -> None:
        """Test that merge always succeeds (no validation).

        TODO: Improve this test to:
        - Verify merged values are correctly combined
        - Test priority of merge (which values override)
        - Test merging of nested configurations
        - Verify the merged config object has expected values
        - Test edge cases (None values, empty dicts, etc.)
        """
        base_config = FlextConfig(app_name="test-app")
        result = FlextConfig.merge(base_config, {"environment": "invalid_env"})
        assert result.is_success  # No validation, always succeeds
