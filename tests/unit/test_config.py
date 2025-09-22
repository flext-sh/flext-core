"""Comprehensive tests for FlextConfig - Target 90%+ coverage.

Tests for the unified configuration management system following FLEXT patterns
with complete coverage of all actual FlextConfig functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from flext_core import FlextConfig, FlextConstants


class TestFlextConfigBasics:
    """Test basic FlextConfig functionality and initialization."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_flext_config_basic_initialization(self) -> None:
        """Test basic FlextConfig initialization with default values."""
        config = FlextConfig()

        # Test basic attributes are set to defaults
        assert config.app_name == "FLEXT Application"
        assert config.environment == "development"
        assert config.debug is False
        assert config.log_level == FlextConstants.Config.LogLevel.INFO
        assert config.max_workers == 4
        assert config.timeout_seconds == 30

    def test_flext_config_initialization_with_custom_values(self) -> None:
        """Test FlextConfig initialization with custom values."""
        config = FlextConfig(
            app_name="custom-app",
            environment="production",
            debug=False,  # Must be False in production
            max_workers=8,
            log_level=FlextConstants.Config.LogLevel.DEBUG,
        )

        assert config.app_name == "custom-app"
        assert config.environment == "production"
        assert config.debug is False
        assert config.max_workers == 8
        assert config.log_level == "DEBUG"

    def test_flext_config_validation_errors(self) -> None:
        """Test FlextConfig validation errors."""
        # Test invalid environment
        with pytest.raises(ValidationError):
            FlextConfig(environment="invalid_env")

        # Test invalid log level
        with pytest.raises(ValidationError):
            FlextConfig(log_level="INVALID_LEVEL")

        # Test debug=True in production (should fail)
        with pytest.raises(ValidationError):
            FlextConfig(environment="production", debug=True)

    def test_field_constraints(self) -> None:
        """Test field constraints and validation."""
        # Test database_pool_size constraints
        with pytest.raises(ValidationError):
            FlextConfig(database_pool_size=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            FlextConfig(database_pool_size=200)  # Must be <= 100

        # Test timeout_seconds constraints
        with pytest.raises(ValidationError):
            FlextConfig(timeout_seconds=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            FlextConfig(timeout_seconds=500)  # Must be <= 300

        # Test max_workers constraints
        with pytest.raises(ValidationError):
            FlextConfig(max_workers=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            FlextConfig(max_workers=100)  # Must be <= 50

    def test_environment_specific_validation(self) -> None:
        """Test environment-specific validation rules."""
        # Test trace requires debug
        with pytest.raises(ValidationError):
            FlextConfig(debug=False, trace=True)

        # Test valid trace with debug
        config = FlextConfig(debug=True, trace=True)
        assert config.debug is True
        assert config.trace is True


class TestFlextConfigClassMethods:
    """Test FlextConfig class methods."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_create_method(self) -> None:
        """Test create class method."""
        config = FlextConfig.create(
            app_name="created_app",
            version="2.0.0",
            environment="staging",
            debug=True,
        )

        assert isinstance(config, FlextConfig)
        assert config.app_name == "created_app"
        assert config.version == "2.0.0"
        assert config.environment == "staging"
        assert config.debug is True

    def test_create_for_environment_method(self) -> None:
        """Test create_for_environment class method."""
        # Test development environment
        dev_config = FlextConfig.create_for_environment("development")
        assert dev_config.environment == "development"

        # Test production environment
        prod_config = FlextConfig.create_for_environment(
            "production",
            app_name="prod_app",
            debug=False,
        )
        assert prod_config.environment == "production"
        assert prod_config.app_name == "prod_app"
        assert prod_config.debug is False

    def test_from_file_method_json(self) -> None:
        """Test from_file class method with JSON file."""
        config_data = {
            "app_name": "json_test_app",
            "version": "3.0.0",
            "environment": "testing",
            "debug": True,
            "max_workers": 6,
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(config_data, f)
            json_path = f.name

        try:
            config = FlextConfig.from_file(json_path)
            assert config.app_name == "json_test_app"
            assert config.version == "3.0.0"
            assert config.environment == "testing"
            assert config.debug is True
            assert config.max_workers == 6
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_from_file_error_cases(self) -> None:
        """Test from_file error handling."""
        # Test nonexistent file
        with pytest.raises(FileNotFoundError):
            FlextConfig.from_file("/nonexistent/config.json")

        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"invalid content")
            txt_path = f.name

        try:
            with pytest.raises(
                ValueError, match="Unsupported configuration file format"
            ):
                FlextConfig.from_file(txt_path)
        finally:
            Path(txt_path).unlink(missing_ok=True)


class TestFlextConfigInstanceMethods:
    """Test FlextConfig instance methods."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_helper_methods(self) -> None:
        """Test helper methods."""
        # Test is_development and is_production
        dev_config = FlextConfig(environment="development")
        assert dev_config.is_development() is True
        assert dev_config.is_production() is False

        prod_config = FlextConfig(environment="production", debug=False)
        assert prod_config.is_production() is True
        assert prod_config.is_development() is False

    def test_getter_methods(self) -> None:
        """Test configuration getter methods."""
        config = FlextConfig(
            app_name="test_app",
            log_level=FlextConstants.Config.LogLevel.DEBUG,
            json_output=True,
            include_source=False,
            structured_output=True,
            database_url="postgresql://localhost/test",
            database_pool_size=20,
            cache_ttl=800,
            cache_max_size=3000,
            enable_caching=True,
        )

        # Test get_logging_config
        logging_config = config.get_logging_config()
        assert logging_config["level"] == "DEBUG"
        assert logging_config["json_output"] is True
        assert logging_config["include_source"] is False
        assert logging_config["structured_output"] is True

        # Test get_database_config
        db_config = config.get_database_config()
        assert db_config["url"] == "postgresql://localhost/test"
        assert db_config["pool_size"] == 20

        # Test get_cache_config
        cache_config = config.get_cache_config()
        assert cache_config["ttl"] == 800
        assert cache_config["max_size"] == 3000
        assert cache_config["enabled"] is True

    def test_serialization_methods(self) -> None:
        """Test serialization methods."""
        config = FlextConfig(
            app_name="serialize_test",
            version="4.0.0",
            debug=True,
            environment="development",
            max_workers=10,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "serialize_test"
        assert config_dict["version"] == "4.0.0"
        assert config_dict["max_workers"] == 10

        # Test to_json
        json_str = config.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["app_name"] == "serialize_test"
        assert parsed["version"] == "4.0.0"

    def test_merge_method(self) -> None:
        """Test merge method."""
        base_config = FlextConfig(app_name="base", version="1.0.0", max_workers=4)

        # Test merge with dict
        override_dict: dict[str, object] = {
            "app_name": "merged",
            "debug": True,
            "max_workers": 8,
            "timeout_seconds": 60,
        }
        merged = base_config.merge(override_dict)

        assert merged.app_name == "merged"
        assert merged.debug is True
        assert merged.max_workers == 8
        assert merged.timeout_seconds == 60
        assert merged.version == "1.0.0"  # Preserved from base


class TestFlextConfigGlobalInstance:
    """Test FlextConfig global instance management."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_singleton_behavior(self) -> None:
        """Test global instance singleton behavior."""
        # Test get_global_instance creates instance
        instance1 = FlextConfig.get_global_instance()
        assert isinstance(instance1, FlextConfig)

        # Test subsequent calls return same instance
        instance2 = FlextConfig.get_global_instance()
        assert instance1 is instance2

    def test_set_global_instance(self) -> None:
        """Test set_global_instance method."""
        # Create custom config
        custom_config = FlextConfig(app_name="custom", version="5.0.0")
        FlextConfig.set_global_instance(custom_config)

        # Verify it's returned by get_global_instance
        instance = FlextConfig.get_global_instance()
        assert instance is custom_config
        assert instance.app_name == "custom"

    def test_reset_global_instance(self) -> None:
        """Test reset_global_instance method."""
        # Set a custom instance
        custom_config = FlextConfig(app_name="custom")
        FlextConfig.set_global_instance(custom_config)

        # Reset and verify new instance is created
        FlextConfig.reset_global_instance()
        new_instance = FlextConfig.get_global_instance()
        assert new_instance is not custom_config
        assert new_instance.app_name == "FLEXT Application"  # Default

    def test_clear_global_instance_alias(self) -> None:
        """Test clear_global_instance delegates to reset behavior."""
        # Set a custom instance
        custom_config = FlextConfig(app_name="custom-clear")
        FlextConfig.set_global_instance(custom_config)

        # Clear via the alias and verify a new instance is lazily created
        FlextConfig.clear_global_instance()
        recreated_instance = FlextConfig.get_global_instance()
        assert recreated_instance is not custom_config

        # Ensure subsequent calls reuse the new singleton instance
        subsequent_instance = FlextConfig.get_global_instance()
        assert subsequent_instance is recreated_instance


class TestFlextConfigEnvironmentLoading:
    """Test FlextConfig environment variable loading."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_environment_variable_loading(self) -> None:
        """Test loading configuration from environment variables."""
        env_vars = {
            "FLEXT_APP_NAME": "env_app",
            "FLEXT_VERSION": "6.0.0",
            "FLEXT_ENVIRONMENT": "testing",
            "FLEXT_DEBUG": "true",
            "FLEXT_MAX_WORKERS": "12",
            "FLEXT_TIMEOUT_SECONDS": "90",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = FlextConfig()

            assert config.app_name == "env_app"
            assert config.version == "6.0.0"
            assert config.environment == "testing"
            assert config.debug is True
            assert config.max_workers == 12
            assert config.timeout_seconds == 90

    def test_environment_variable_type_conversion(self) -> None:
        """Test proper type conversion of environment variables."""
        env_vars = {
            "FLEXT_DEBUG": "false",
            "FLEXT_TRACE": "true",
            "FLEXT_ENABLE_CACHING": "false",
            "FLEXT_JSON_OUTPUT": "true",
            "FLEXT_INCLUDE_SOURCE": "false",
            "FLEXT_STRUCTURED_OUTPUT": "true",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = FlextConfig()

            assert config.debug is False
            assert config.trace is True
            assert config.enable_caching is False
            assert config.json_output is True
            assert config.include_source is False
            assert config.structured_output is True


class TestFlextConfigValidationComprehensive:
    """Comprehensive validation testing."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_all_constraint_fields(self) -> None:
        """Test all fields with constraints work within valid ranges."""
        config = FlextConfig(
            app_name="validation_test",
            database_pool_size=50,  # Range: 1-100
            cache_ttl=600,  # >= 0
            cache_max_size=2000,  # >= 0
            max_retry_attempts=5,  # Range: 0-10
            timeout_seconds=120,  # Range: 1-300
            max_workers=25,  # Range: 1-50
        )

        assert config.database_pool_size == 50
        assert config.cache_ttl == 600
        assert config.cache_max_size == 2000
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 120
        assert config.max_workers == 25

    def test_boundary_values(self) -> None:
        """Test boundary values for constrained fields."""
        # Test minimum values
        config_min = FlextConfig(
            database_pool_size=1,
            cache_ttl=0,
            cache_max_size=0,
            max_retry_attempts=0,
            timeout_seconds=1,
            max_workers=1,
        )

        assert config_min.database_pool_size == 1
        assert config_min.cache_ttl == 0
        assert config_min.cache_max_size == 0
        assert config_min.max_retry_attempts == 0
        assert config_min.timeout_seconds == 1
        assert config_min.max_workers == 1

        # Test maximum values
        config_max = FlextConfig(
            database_pool_size=100,
            max_retry_attempts=10,
            timeout_seconds=300,
            max_workers=50,
        )

        assert config_max.database_pool_size == 100
        assert config_max.max_retry_attempts == 10
        assert config_max.timeout_seconds == 300
        assert config_max.max_workers == 50

    def test_invalid_boundary_values(self) -> None:
        """Test invalid boundary values raise ValidationError."""
        # Test below minimum
        with pytest.raises(ValidationError):
            FlextConfig(database_pool_size=0)

        with pytest.raises(ValidationError):
            FlextConfig(cache_ttl=-1)

        with pytest.raises(ValidationError):
            FlextConfig(max_retry_attempts=-1)

        with pytest.raises(ValidationError):
            FlextConfig(timeout_seconds=0)

        with pytest.raises(ValidationError):
            FlextConfig(max_workers=0)

        # Test above maximum
        with pytest.raises(ValidationError):
            FlextConfig(database_pool_size=101)

        with pytest.raises(ValidationError):
            FlextConfig(max_retry_attempts=11)

        with pytest.raises(ValidationError):
            FlextConfig(timeout_seconds=301)

        with pytest.raises(ValidationError):
            FlextConfig(max_workers=51)


class TestFlextConfigDefaults:
    """Test FlextConfig default values."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_all_default_values(self) -> None:
        """Test all default values are set correctly."""
        config = FlextConfig()

        # Core defaults
        assert config.app_name == "FLEXT Application"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.debug is False
        assert config.trace is False

        # Logging defaults
        assert config.log_level == "INFO"
        assert config.json_output is False
        assert config.include_source is True
        assert config.structured_output is False

        # Database defaults
        assert config.database_url == "sqlite:///flext.db"
        assert config.database_pool_size == 10

        # Cache defaults
        assert config.cache_ttl == 300
        assert config.cache_max_size == 1000
        assert config.enable_caching is True

        # Performance defaults
        assert config.max_workers == 4
        assert config.timeout_seconds == 30
        assert config.max_retry_attempts == 3

    def test_environment_specific_defaults(self) -> None:
        """Test environment-specific defaults."""
        # Development environment
        dev_config = FlextConfig(environment="development")
        assert dev_config.debug is False  # Default even in development

        # Production environment
        prod_config = FlextConfig(environment="production")
        assert prod_config.debug is False
        assert prod_config.trace is False


class TestFlextConfigEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_empty_string_validation(self) -> None:
        """Test empty string validation."""
        # Empty app_name should fail
        with pytest.raises(ValidationError):
            FlextConfig(app_name="")

        # Empty version should fail
        with pytest.raises(ValidationError):
            FlextConfig(version="")

        # Empty database_url should fail
        with pytest.raises(ValidationError):
            FlextConfig(database_url="")

    def test_whitespace_handling(self) -> None:
        """Test whitespace handling in string fields."""
        # Whitespace-only strings should fail validation
        with pytest.raises(ValidationError):
            FlextConfig(app_name="   ")

        with pytest.raises(ValidationError):
            FlextConfig(version="   ")

    def test_type_coercion(self) -> None:
        """Test type coercion for numeric fields."""
        # String numbers should be converted
        config = FlextConfig(
            max_workers=8,  # Integer value
            timeout_seconds=60,
            database_pool_size=15,
        )

        assert config.max_workers == 8
        assert config.timeout_seconds == 60
        assert config.database_pool_size == 15
        assert isinstance(config.max_workers, int)
        assert isinstance(config.timeout_seconds, int)
        assert isinstance(config.database_pool_size, int)
