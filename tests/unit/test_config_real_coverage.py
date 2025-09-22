"""Real tests for FlextConfig to increase coverage.

This test suite focuses on actually implemented functionality in config.py
to increase coverage from 44% to a higher percentage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from flext_core import FlextConfig
from flext_core.constants import FlextConstants


class TestFlextConfigRealCoverage:
    """Real tests for implemented FlextConfig functionality."""

    def setup_method(self) -> None:
        """Reset global instance before each test."""
        FlextConfig.reset_global_instance()

    def test_basic_initialization(self) -> None:
        """Test basic FlextConfig initialization."""
        config = FlextConfig(app_name="test_app")
        assert config.app_name == "test_app"
        assert config.environment in {
            FlextConstants.Environment.ConfigEnvironment.DEVELOPMENT,
            FlextConstants.Environment.ConfigEnvironment.TESTING,
            FlextConstants.Environment.ConfigEnvironment.STAGING,
            FlextConstants.Environment.ConfigEnvironment.PRODUCTION,
        }
        assert isinstance(config.debug, bool)

    def test_initialization_with_all_fields(self) -> None:
        """Test initialization with various field types."""
        config = FlextConfig(
            app_name="full_test_app",
            version="1.2.3",
            environment=FlextConstants.Environment.ConfigEnvironment.PRODUCTION,
            debug=False,  # Must be False in production
            trace=False,
            log_level=FlextConstants.Config.LogLevel.INFO,
            max_workers=8,
            timeout_seconds=60,
            enable_metrics=True,
            enable_caching=False,
            database_url="sqlite:///test.db",
            api_key="test_key",
        )

        assert config.app_name == "full_test_app"
        assert config.version == "1.2.3"
        assert config.environment == "production"
        assert config.debug is False
        assert config.trace is False
        assert config.log_level == "INFO"
        assert config.max_workers == 8
        assert config.timeout_seconds == 60
        assert config.enable_metrics is True
        assert config.enable_caching is False
        assert config.database_url == "sqlite:///test.db"
        assert config.api_key == "test_key"

    def test_global_instance_management(self) -> None:
        """Test global instance singleton pattern."""
        # Test initial get creates instance
        instance1 = FlextConfig.get_global_instance()
        assert instance1 is not None
        assert isinstance(instance1, FlextConfig)

        # Test singleton behavior
        instance2 = FlextConfig.get_global_instance()
        assert instance1 is instance2

        # Test set_global_instance
        custom_config = FlextConfig(app_name="custom")
        FlextConfig.set_global_instance(custom_config)

        instance3 = FlextConfig.get_global_instance()
        assert instance3 is custom_config
        assert instance3.app_name == "custom"

        # Test reset_global_instance
        FlextConfig.reset_global_instance()
        instance4 = FlextConfig.get_global_instance()
        assert instance4 is not instance3

    def test_environment_validation(self) -> None:
        """Test environment field validation."""
        # Valid environments
        valid_envs = ["development", "test", "staging", "production"]
        for env in valid_envs:
            config = FlextConfig(app_name="test", environment=env)
            assert config.environment == env

        # Invalid environment should raise ValidationError
        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", environment="invalid_env")

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        # Valid log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = FlextConfig(app_name="test", log_level=level)
            assert config.log_level == level.upper()

        # Case insensitive
        config = FlextConfig(app_name="test", log_level="info")
        assert config.log_level == "INFO"

        # Invalid log level should raise ValidationError
        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", log_level="INVALID")

    def test_configuration_consistency_validation(self) -> None:
        """Test configuration consistency validation."""
        # Valid configuration - debug False in production
        config = FlextConfig(app_name="test", environment="production", debug=False)
        assert config.environment == "production"
        assert config.debug is False

        # Invalid configuration - debug True in production should fail
        with pytest.raises(
            ValidationError, match="Debug mode cannot be enabled in production"
        ):
            FlextConfig(app_name="test", environment="production", debug=True)

        # Invalid configuration - trace True without debug should fail
        with pytest.raises(ValidationError, match="Trace mode requires debug mode"):
            FlextConfig(
                app_name="test", environment="development", debug=False, trace=True
            )

        # Valid configuration - trace True with debug True should work
        config = FlextConfig(
            app_name="test", environment="development", debug=True, trace=True
        )
        assert config.debug is True
        assert config.trace is True

    def test_helper_methods(self) -> None:
        """Test helper methods."""
        # Test is_development
        dev_config = FlextConfig(app_name="test", environment="development")
        assert dev_config.is_development() is True
        assert dev_config.is_production() is False

        # Test is_production
        prod_config = FlextConfig(
            app_name="test", environment="production", debug=False
        )
        assert prod_config.is_production() is True
        assert prod_config.is_development() is False

    def test_get_config_methods(self) -> None:
        """Test configuration getter methods."""
        config = FlextConfig(
            app_name="test",
            log_level="DEBUG",
            json_output=True,
            include_source=False,
            structured_output=True,
            database_url="postgresql://user:pass@localhost/db",
            database_pool_size=20,
            cache_ttl=600,
            cache_max_size=2000,
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
        assert db_config["url"] == "postgresql://user:pass@localhost/db"
        assert db_config["pool_size"] == 20

        # Test get_cache_config
        cache_config = config.get_cache_config()
        assert cache_config["ttl"] == 600
        assert cache_config["max_size"] == 2000
        assert cache_config["enabled"] is True

        # Test get_cqrs_bus_config
        cqrs_config = config.get_cqrs_bus_config()
        assert isinstance(cqrs_config, dict)

    def test_environment_specific_initialization(self) -> None:
        """Direct initialization replaces create_for_environment."""
        # Test basic creation
        config = FlextConfig(environment="development")
        assert config.environment == "development"

        # Test with overrides
        config = FlextConfig(
            environment="staging",
            app_name="staging_app",
            debug=True,
            max_workers=16,
        )
        assert config.environment == "staging"
        assert config.app_name == "staging_app"
        assert config.debug is True
        assert config.max_workers == 16

    def test_from_file_json(self) -> None:
        """Test from_file class method with JSON."""
        # Create temporary JSON config file
        config_data = {
            "app_name": "json_app",
            "version": "2.0.0",
            "environment": "test",
            "debug": True,
            "log_level": "DEBUG",
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
            assert config.app_name == "json_app"
            assert config.version == "2.0.0"
            assert config.environment == "test"
            assert config.debug is True
            assert config.log_level == "DEBUG"
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_from_file_nonexistent(self) -> None:
        """Test from_file with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            FlextConfig.from_file("/nonexistent/config.json")

    def test_from_file_unsupported_format(self) -> None:
        """Test from_file with unsupported format."""
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

    def test_serialization_methods(self) -> None:
        """Test to_dict and to_json methods."""
        config = FlextConfig(
            app_name="serialize_test",
            version="2.0.0",
            debug=True,
            environment="development",
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "serialize_test"
        assert config_dict["version"] == "2.0.0"
        assert config_dict["debug"] is True

        # Test to_json
        json_str = config.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["app_name"] == "serialize_test"
        assert parsed["version"] == "2.0.0"
        assert parsed["debug"] is True

    def test_model_copy_update_replaces_merge(self) -> None:
        """Test model_copy(update=...) replaces merge helper."""
        base_config = FlextConfig(app_name="base", version="1.0.0", debug=False)

        # Test merge with dict
        override_dict: dict[str, object] = {
            "app_name": "merged",
            "debug": True,
            "max_workers": 16,
        }
        merged = base_config.model_copy(update=override_dict)

        assert merged.app_name == "merged"
        assert merged.debug is True
        assert merged.max_workers == 16
        assert merged.version == "1.0.0"  # Preserved from base

        # Test merge with another FlextConfig
        override_config = FlextConfig(
            app_name="override", trace=True, debug=True, environment="development"
        )
        merged2 = base_config.model_copy(update=override_config.model_dump())

        assert merged2.app_name == "override"
        assert merged2.trace is True
        assert merged2.environment == "development"
        assert merged2.version == "0.9.0"  # Default version from override_config

    def test_direct_initialization_replaces_create(self) -> None:
        """Direct initialization replaces the removed create helper."""
        config = FlextConfig(
            app_name="created_app",
            version="3.0.0",
            environment="staging",
            debug=True,
        )

        assert isinstance(config, FlextConfig)
        assert config.app_name == "created_app"
        assert config.version == "3.0.0"
        assert config.environment == "staging"
        assert config.debug is True

    def test_field_constraints(self) -> None:
        """Test field constraints and validation."""
        # Test positive integer constraints
        config = FlextConfig(
            app_name="test",
            database_pool_size=50,
            max_retry_attempts=5,
            timeout_seconds=120,
            max_workers=25,
        )
        assert config.database_pool_size == 50
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 120
        assert config.max_workers == 25

        # Test constraint violations
        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", database_pool_size=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", database_pool_size=200)  # Must be <= 100

        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", timeout_seconds=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", timeout_seconds=500)  # Must be <= 300

        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", max_workers=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            FlextConfig(app_name="test", max_workers=100)  # Must be <= 50

    def test_default_values(self) -> None:
        """Test default field values."""
        config = FlextConfig()

        # Test core defaults
        assert config.app_name == "FLEXT Application"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.debug is False
        assert config.trace is False
        assert config.log_level == "INFO"

        # Test feature flag defaults
        assert config.enable_caching is True
        assert config.enable_metrics is False
        assert config.enable_tracing is False
        assert config.enable_circuit_breaker is False

        # Test numeric defaults
        assert config.database_pool_size == 10
        assert config.cache_ttl == 300
        assert config.cache_max_size == 1000
        assert config.max_retry_attempts == 3
        assert config.timeout_seconds == 30
        assert config.max_workers == 4

        # Test optional defaults
        assert config.database_url is None
        assert config.secret_key is None
        assert config.api_key is None

    def test_comprehensive_field_coverage(self) -> None:
        """Test all available fields to ensure coverage."""
        config = FlextConfig(
            # Core fields
            app_name="comprehensive_test",
            version="1.2.3",
            environment="test",
            debug=True,
            trace=True,
            # Logging fields
            log_level="DEBUG",
            json_output=True,
            include_source=False,
            structured_output=True,
            # Database fields
            database_url="postgresql://localhost/test",
            database_pool_size=15,
            # Cache fields
            cache_ttl=600,
            cache_max_size=2000,
            # Security fields
            secret_key="test_secret",
            api_key="test_api_key",
            # Service fields
            max_retry_attempts=5,
            timeout_seconds=60,
            # Feature flags
            enable_caching=False,
            enable_metrics=True,
            enable_tracing=True,
            enable_circuit_breaker=True,
            # Container fields
            max_workers=8,
            # Validation fields
            validation_strict_mode=True,
            # Serialization fields
            serialization_encoding="utf-16",
            # Dispatcher fields
            dispatcher_auto_context=False,
            dispatcher_timeout_seconds=45,
            dispatcher_enable_metrics=False,
            dispatcher_enable_logging=False,
            # JSON fields
            json_indent=4,
            json_sort_keys=True,
            ensure_json_serializable=False,
        )

        # Verify all fields are set correctly
        assert config.app_name == "comprehensive_test"
        assert config.version == "1.2.3"
        assert config.environment == "test"
        assert config.debug is True
        assert config.trace is True
        assert config.log_level == "DEBUG"
        assert config.json_output is True
        assert config.include_source is False
        assert config.structured_output is True
        assert config.database_url == "postgresql://localhost/test"
        assert config.database_pool_size == 15
        assert config.cache_ttl == 600
        assert config.cache_max_size == 2000
        assert config.secret_key == "test_secret"
        assert config.api_key == "test_api_key"
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 60
        assert config.enable_caching is False
        assert config.enable_metrics is True
        assert config.enable_tracing is True
        assert config.enable_circuit_breaker is True
        assert config.max_workers == 8
        assert config.validation_strict_mode is True
        assert config.serialization_encoding == "utf-16"
        assert config.dispatcher_auto_context is False
        assert config.dispatcher_timeout_seconds == 45
        assert config.dispatcher_enable_metrics is False
        assert config.dispatcher_enable_logging is False
        assert config.json_indent == 4
        assert config.json_sort_keys is True
        assert config.ensure_json_serializable is False
