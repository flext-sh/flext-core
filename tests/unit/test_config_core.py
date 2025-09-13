"""Core functionality tests for FlextConfig to increase coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from flext_core import FlextConfig


class TestFlextApiConfig:
    """Test FlextApiConfig functionality."""

    def setup_method(self) -> None:
        """Clear config before each test."""
        FlextConfig.clear_global_instance()

    def teardown_method(self) -> None:
        """Clear config after each test."""
        FlextConfig.clear_global_instance()

    def test_config_creation_defaults(self) -> None:
        """Test config creation with default values."""
        config = FlextConfig.get_global_instance()

        # Test defaults exist and are accessible
        assert hasattr(config, "app_name")
        assert hasattr(config, "environment")
        assert hasattr(config, "debug")
        assert hasattr(config, "max_workers")
        assert hasattr(config, "timeout_seconds")

        # Test default types
        assert isinstance(config.app_name, str)
        assert isinstance(config.environment, str)
        assert isinstance(config.debug, bool)
        assert isinstance(config.max_workers, int)
        assert isinstance(config.timeout_seconds, int)

    def test_config_field_info(self) -> None:
        """Test config field information and constraints."""
        config = FlextConfig.get_global_instance()

        # Test that config has reasonable defaults
        assert config.max_workers > 0, "max_workers should be positive"
        assert config.timeout_seconds > 0, "timeout_seconds should be positive"
        assert len(config.app_name) > 0, "app_name should not be empty"
        assert len(config.environment) > 0, "environment should not be empty"

    def test_config_timeout_validation(self) -> None:
        """Test timeout validation scenarios."""
        config = FlextConfig.get_global_instance()

        # Test accessing timeout field - should work without errors
        timeout_value = config.timeout_seconds
        assert isinstance(timeout_value, int)
        assert timeout_value >= 0

    def test_config_base_url_validation(self) -> None:
        """Test base URL validation if present."""
        config = FlextConfig.get_global_instance()

        # Test that config can be accessed - basic functionality
        if hasattr(config, "base_url"):
            base_url = config.base_url
            assert isinstance(base_url, str)

    def test_config_validation_error_details(self) -> None:
        """Test validation error scenarios for coverage."""
        # Test environment variable error handling using correct API
        env_helper = FlextConfig.DefaultEnvironmentAdapter()

        # Test getting non-existent environment variable
        result = env_helper.get_env_var("FLEXT_NONEXISTENT_VAR_12345")
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error.lower()

        # Test getting environment variables with prefix
        prefix_result = env_helper.get_env_vars_with_prefix("FLEXT_TEST_")
        assert prefix_result.is_success
        assert isinstance(prefix_result.unwrap(), dict)

    def test_config_business_rule_validation_production_debug(self) -> None:
        """Test production environment with debug enabled validation."""
        # Create a FlextConfig instance with problematic configuration
        config = FlextConfig(
            app_name="test-app",
            name="test-name",
            version="1.0.0",
            environment="production",
            debug=True,
            config_source="file",  # Not default (valid values: cli, default, dotenv, env, file)
            max_workers=4,
            timeout_seconds=30,
        )

        # Test business rule validator
        validator = FlextConfig.BusinessValidator()
        result = validator.validate_business_rules(config)

        # Should fail due to debug in production
        assert result.is_failure
        assert result.error is not None
        assert "debug mode in production" in result.error.lower()

    def test_config_business_rule_validation_production_workers(self) -> None:
        """Test production environment with insufficient workers."""
        config = FlextConfig(
            app_name="test-app",
            name="test-name",
            version="1.0.0",
            environment="production",
            debug=False,
            max_workers=1,  # Too few for production
            timeout_seconds=30,
        )

        validator = FlextConfig.BusinessValidator()
        result = validator.validate_business_rules(config)

        assert result.is_failure
        assert result.error is not None
        assert "should have at least" in result.error.lower()

    def test_config_business_rule_validation_high_timeout_low_workers(self) -> None:
        """Test high timeout with low worker count."""
        config = FlextConfig(
            app_name="test-app",
            name="test-name",
            version="1.0.0",
            environment="development",
            max_workers=1,  # Too few
            timeout_seconds=150,  # High timeout
        )

        validator = FlextConfig.BusinessValidator()
        result = validator.validate_business_rules(config)

        assert result.is_failure
        assert result.error is not None
        assert "performance issues" in result.error.lower()

    def test_config_business_rule_validation_excessive_workers(self) -> None:
        """Test excessive worker count."""
        config = FlextConfig(
            app_name="test-app",
            name="test-name",
            version="1.0.0",
            environment="development",
            max_workers=60,  # Excessive
            timeout_seconds=30,
        )

        validator = FlextConfig.BusinessValidator()
        result = validator.validate_business_rules(config)

        assert result.is_failure
        assert result.error is not None
        assert "resource exhaustion" in result.error.lower()

    def test_config_business_rule_validation_auth_without_key(self) -> None:
        """Test authentication enabled without API key."""
        config = FlextConfig(
            app_name="test-app",
            name="test-name",
            version="1.0.0",
            environment="development",
            enable_auth=True,
            api_key="   ",  # Empty/whitespace
            max_workers=2,
            timeout_seconds=30,
        )

        validator = FlextConfig.BusinessValidator()
        result = validator.validate_business_rules(config)

        assert result.is_failure
        assert result.error is not None
        assert "api key required" in result.error.lower()

    def test_config_save_unsupported_format(self) -> None:
        """Test saving config to unsupported format."""
        config = FlextConfig.get_global_instance()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            file_manager = FlextConfig.FilePersistence()
            result = file_manager.save_to_file(config, temp_path)

            assert result.is_failure
            assert result.error is not None
            assert "unsupported format" in result.error.lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_config_save_yaml_format_list_values(self) -> None:
        """Test saving config with list values to YAML."""
        config_data = {
            "app_name": "test-app",
            "environments": ["dev", "staging", "prod"],
            "features": ["auth", "logging"],
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            file_manager = FlextConfig.FilePersistence()
            result = file_manager.save_to_file(config_data, temp_path)

            assert result.is_success

            # Verify YAML content
            content = Path(temp_path).read_text(encoding="utf-8")
            assert "environments:" in content
            assert "- dev" in content
            assert "features:" in content
            assert "- auth" in content
        finally:
            Path(temp_path).unlink(missing_ok=True)
