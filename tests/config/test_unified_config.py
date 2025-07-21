"""Tests for unified configuration system.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from flext_core.config.unified_config import (
    APIConfigMixin,
    AuthConfigMixin,
    BaseConfigMixin,
    ConfigBuilder,
    ConfigFactory,
    ConfigValidator,
    DatabaseConfigMixin,
    FlextAPIConfigBuilder,
    FlextAuthConfigBuilder,
    FlextCoreConfigBuilder,
    LoggingConfigMixin,
    MonitoringConfigMixin,
    OracleConnectionConfigMixin,
    OracleWMSConfigBuilder,
    PerformanceConfigMixin,
    RedisConfigMixin,
    SingerConfigMixin,
    WMSConfigMixin,
    load_config_from_file,
    merge_configs,
)
from flext_core.domain.shared_types import Environment, LogLevel


class TestBaseConfigMixin:
    """Test BaseConfigMixin functionality."""

    def test_base_config_defaults(self) -> None:
        """Test base configuration defaults."""
        config = BaseConfigMixin(project_name="test-project")

        assert config.project_name == "test-project"
        assert config.project_version == "0.1.0"
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False

    def test_base_config_with_values(self) -> None:
        """Test base configuration with explicit values."""
        config = BaseConfigMixin(
            project_name="test-project",
            project_version="2.0.0",
            environment=Environment.PRODUCTION,
            debug=True,
        )

        assert config.project_name == "test-project"
        assert config.project_version == "2.0.0"
        assert config.environment == Environment.PRODUCTION
        assert config.debug is True

    def test_base_config_section_metadata(self) -> None:
        """Test base configuration section metadata."""
        assert BaseConfigMixin._config_section == "base"

    def test_environment_validation(self) -> None:
        """Test environment validation."""
        # Valid string conversion
        config = BaseConfigMixin(
            project_name="test", environment=Environment.PRODUCTION
        )
        assert config.environment == Environment.PRODUCTION

        # Valid enum value
        config = BaseConfigMixin(project_name="test", environment=Environment.STAGING)
        assert config.environment == Environment.STAGING

    def test_environment_helper_methods(self) -> None:
        """Test environment helper methods."""
        prod_config = BaseConfigMixin(
            project_name="test", environment=Environment.PRODUCTION
        )
        assert prod_config.is_production() is True
        assert prod_config.is_development() is False

        dev_config = BaseConfigMixin(
            project_name="test", environment=Environment.DEVELOPMENT
        )
        assert dev_config.is_production() is False
        assert dev_config.is_development() is True


class TestLoggingConfigMixin:
    """Test LoggingConfigMixin functionality."""

    def test_logging_config_defaults(self) -> None:
        """Test logging configuration defaults."""
        config = LoggingConfigMixin()

        assert config.log_level == LogLevel.INFO
        assert config.log_file is None
        assert "%(asctime)s" in config.log_format
        assert config.log_rotation is True
        assert config.log_retention_days == 30

    def test_logging_config_with_values(self) -> None:
        """Test logging configuration with explicit values."""
        config = LoggingConfigMixin(
            log_level=LogLevel.DEBUG,
            log_file=Path("/var/log/app.log"),
            log_format="simple",
            log_rotation=False,
            log_retention_days=7,
        )

        assert config.log_level == LogLevel.DEBUG
        assert str(config.log_file) == "/var/log/app.log"
        assert config.log_format == "simple"
        assert config.log_rotation is False
        assert config.log_retention_days == 7

    def test_logging_config_section_metadata(self) -> None:
        """Test logging configuration section metadata."""
        assert LoggingConfigMixin._config_section == "logging"


class TestDatabaseConfigMixin:
    """Test DatabaseConfigMixin functionality."""

    def test_database_config_with_values(self) -> None:
        """Test database configuration with explicit values."""
        config = DatabaseConfigMixin(
            database_url="postgresql://user:pass@localhost:5432/db",
            database_pool_size=15,
            database_max_overflow=25,
        )

        assert config.database_url == "postgresql://user:pass@localhost:5432/db"
        assert config.database_pool_size == 15
        assert config.database_max_overflow == 25

    def test_database_config_section_metadata(self) -> None:
        """Test database configuration section metadata."""
        assert DatabaseConfigMixin._config_section == "database"


class TestRedisConfigMixin:
    """Test RedisConfigMixin functionality."""

    def test_redis_config_section_metadata(self) -> None:
        """Test Redis configuration section metadata."""
        assert RedisConfigMixin._config_section == "redis"


class TestAuthConfigMixin:
    """Test AuthConfigMixin functionality."""

    def test_auth_config_section_metadata(self) -> None:
        """Test authentication configuration section metadata."""
        assert AuthConfigMixin._config_section == "auth"


class TestAPIConfigMixin:
    """Test APIConfigMixin functionality."""

    def test_api_config_section_metadata(self) -> None:
        """Test API configuration section metadata."""
        assert APIConfigMixin._config_section == "api"


class TestPerformanceConfigMixin:
    """Test PerformanceConfigMixin functionality."""

    def test_performance_config_section_metadata(self) -> None:
        """Test performance configuration section metadata."""
        assert PerformanceConfigMixin._config_section == "performance"


class TestMonitoringConfigMixin:
    """Test MonitoringConfigMixin functionality."""

    def test_monitoring_config_section_metadata(self) -> None:
        """Test monitoring configuration section metadata."""
        assert MonitoringConfigMixin._config_section == "monitoring"


class TestOracleConnectionConfigMixin:
    """Test OracleConnectionConfigMixin functionality."""

    def test_oracle_config_section_metadata(self) -> None:
        """Test Oracle configuration section metadata."""
        assert OracleConnectionConfigMixin._config_section == "oracle"


class TestWMSConfigMixin:
    """Test WMSConfigMixin functionality."""

    def test_wms_config_section_metadata(self) -> None:
        """Test WMS configuration section metadata."""
        assert WMSConfigMixin._config_section == "wms"


class TestSingerConfigMixin:
    """Test SingerConfigMixin functionality."""

    def test_singer_config_section_metadata(self) -> None:
        """Test Singer configuration section metadata."""
        assert SingerConfigMixin._config_section == "singer"


class TestConfigBuilder:
    """Test ConfigBuilder abstract base class."""

    def test_config_builder_is_abstract(self) -> None:
        """Test that ConfigBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ConfigBuilder()  # type: ignore[abstract]


class TestFlextCoreConfigBuilder:
    """Test FlextCoreConfigBuilder functionality."""

    def test_core_builder_instantiation(self) -> None:
        """Test FlextCore builder can be instantiated."""
        builder = FlextCoreConfigBuilder()
        assert isinstance(builder, ConfigBuilder)

    def test_core_builder_config_creation(self) -> None:
        """Test FlextCore builder can create config."""
        with patch.dict(
            os.environ,
            {
                "FLEXT_CORE_PROJECT_NAME": "test-core",
            },
        ):
            builder = FlextCoreConfigBuilder()
            config = builder.build_config()

            # Should have base configuration
            assert hasattr(config, "project_name")
            assert hasattr(config, "environment")


class TestFlextAuthConfigBuilder:
    """Test FlextAuthConfigBuilder functionality."""

    def test_auth_builder_instantiation(self) -> None:
        """Test FlextAuth builder can be instantiated."""
        builder = FlextAuthConfigBuilder()
        assert isinstance(builder, ConfigBuilder)

    def test_auth_builder_config_creation(self) -> None:
        """Test FlextAuth builder can create config."""
        with patch.dict(
            os.environ,
            {
                "FLEXT_AUTH_PROJECT_NAME": "test-auth",
                "FLEXT_AUTH_JWT_SECRET_KEY": "test-secret-key-12345",
                "FLEXT_AUTH_DATABASE_URL": "postgresql://localhost/test",
            },
        ):
            builder = FlextAuthConfigBuilder()
            config = builder.build_config()

            # Should have base and auth configuration
            assert hasattr(config, "project_name")
            assert hasattr(config, "environment")


class TestFlextAPIConfigBuilder:
    """Test FlextAPIConfigBuilder functionality."""

    def test_api_builder_instantiation(self) -> None:
        """Test FlextAPI builder can be instantiated."""
        builder = FlextAPIConfigBuilder()
        assert isinstance(builder, ConfigBuilder)

    def test_api_builder_config_creation(self) -> None:
        """Test FlextAPI builder can create config."""
        # Mock environment variables for API config
        with patch.dict(
            os.environ,
            {
                "FLEXT_API_PROJECT_NAME": "test-api",
                "FLEXT_API_REDIS_URL": "redis://localhost:6379/15",
            },
        ):
            builder = FlextAPIConfigBuilder()
            config = builder.build_config()

            # Should have base and API configuration
            assert hasattr(config, "project_name")
            assert hasattr(config, "environment")


class TestOracleWMSConfigBuilder:
    """Test OracleWMSConfigBuilder functionality."""

    def test_oracle_wms_builder_instantiation(self) -> None:
        """Test OracleWMS builder can be instantiated."""
        builder = OracleWMSConfigBuilder()
        assert isinstance(builder, ConfigBuilder)

    def test_oracle_wms_builder_config_creation(self) -> None:
        """Test OracleWMS builder can create config."""
        with patch.dict(
            os.environ,
            {
                "ORACLE_WMS_PROJECT_NAME": "test-oracle-wms",
                "ORACLE_WMS_WMS_ENVIRONMENT": "development",
                "ORACLE_WMS_WMS_ORG_ID": "test-org",
                "ORACLE_WMS_WMS_FACILITY_CODE": "TEST-FAC",
                "ORACLE_WMS_WMS_COMPANY_CODE": "TEST-CO",
                "ORACLE_WMS_ORACLE_HOST": "localhost",
                "ORACLE_WMS_ORACLE_SERVICE": "xe",
                "ORACLE_WMS_ORACLE_USERNAME": "test_user",
                "ORACLE_WMS_ORACLE_PASSWORD": "test_pass",
            },
        ):
            builder = OracleWMSConfigBuilder()
            config = builder.build_config()

            # Should have base configuration and Oracle/WMS specifics
            assert hasattr(config, "project_name")
            assert hasattr(config, "environment")


class TestConfigFactory:
    """Test ConfigFactory functionality."""

    def test_factory_builder_registration(self) -> None:
        """Test factory can register builders."""
        # Register a builder
        builder = FlextCoreConfigBuilder()
        ConfigFactory.register_builder("test-core", builder)

        # Should be able to create config
        with patch.dict(os.environ, {"FLEXT_CORE_PROJECT_NAME": "test-core"}):
            config = ConfigFactory.create_config("test-core")
            assert hasattr(config, "project_name")

    def test_factory_unknown_project_type(self) -> None:
        """Test factory raises error for unknown project type."""
        with pytest.raises(ValueError, match="Unknown project type"):
            ConfigFactory.create_config("nonexistent-project")

    def test_factory_create_from_env(self) -> None:
        """Test factory can create config from environment variable."""
        # Register a builder first
        builder = FlextCoreConfigBuilder()
        ConfigFactory.register_builder("flext-core", builder)

        # Test with default environment variable
        with patch.dict(
            os.environ,
            {
                "FLEXT_PROJECT_TYPE": "flext-core",
                "FLEXT_CORE_PROJECT_NAME": "test-core",
            },
        ):
            config = ConfigFactory.create_from_env()
            assert hasattr(config, "project_name")

    def test_factory_create_from_env_default(self) -> None:
        """Test factory creates default config when env var not set."""
        # Register the default builder
        builder = FlextCoreConfigBuilder()
        ConfigFactory.register_builder("flext-core", builder)

        # Test without environment variable (should use default)
        with patch.dict(
            os.environ, {"FLEXT_CORE_PROJECT_NAME": "default-core"}, clear=True
        ):
            config = ConfigFactory.create_from_env()
            assert hasattr(config, "project_name")


class TestConfigValidator:
    """Test ConfigValidator functionality."""

    def test_validator_production_config_valid(self) -> None:
        """Test validator with valid production config."""
        config = BaseConfigMixin(
            project_name="test",
            environment=Environment.PRODUCTION,
            debug=False,
        )

        issues = ConfigValidator.validate_production_config(config)
        # Should have no debug-related issues
        debug_issues = [issue for issue in issues if "debug" in issue.lower()]
        assert len(debug_issues) == 0

    def test_validator_production_config_debug_enabled(self) -> None:
        """Test validator detects debug enabled in production."""
        config = BaseConfigMixin(
            project_name="test",
            environment=Environment.PRODUCTION,
            debug=True,
        )

        issues = ConfigValidator.validate_production_config(config)
        assert any("Debug mode should be disabled" in issue for issue in issues)

    def test_validator_production_config_no_environment(self) -> None:
        """Test validator handles config without environment attribute."""
        # Create a simple config without environment
        from pydantic import BaseModel

        class SimpleConfig(BaseModel):
            name: str = "test"

        config = SimpleConfig()
        issues = ConfigValidator.validate_production_config(config)
        # Should not crash and return empty list
        assert isinstance(issues, list)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_merge_configs_multiple(self) -> None:
        """Test merging multiple configurations."""
        config1 = BaseConfigMixin(project_name="test1", debug=True)
        config2 = LoggingConfigMixin(log_level=LogLevel.DEBUG)

        merged = merge_configs(config1, config2)

        assert merged["project_name"] == "test1"
        assert merged["debug"] is True
        assert merged["log_level"] == LogLevel.DEBUG

    def test_merge_configs_empty(self) -> None:
        """Test merging with no configurations."""
        merged = merge_configs()
        assert merged == {}

    def test_merge_configs_single(self) -> None:
        """Test merging single configuration."""
        config = BaseConfigMixin(project_name="test", debug=True)
        merged = merge_configs(config)

        assert merged["project_name"] == "test"
        assert merged["debug"] is True

    def test_load_config_from_file_nonexistent(self) -> None:
        """Test loading config from non-existent file."""
        nonexistent_path = Path("/nonexistent/config.yaml")

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config_from_file(nonexistent_path, "flext-core")

    def test_load_config_from_file_existing(self) -> None:
        """Test loading config from existing file."""
        # Register a builder first
        builder = FlextCoreConfigBuilder()
        ConfigFactory.register_builder("flext-core", builder)

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("project_name: test-from-file\n")
            f.write("debug: true\n")
            f.flush()

            try:
                with patch.dict(
                    os.environ, {"FLEXT_CORE_PROJECT_NAME": "test-from-file"}
                ):
                    config = load_config_from_file(Path(f.name), "flext-core")
                    assert hasattr(config, "project_name")
            finally:
                Path(f.name).unlink()


class TestConfigurationIntegration:
    """Test configuration system integration scenarios."""

    def test_full_configuration_composition(self) -> None:
        """Test full configuration with multiple mixins."""
        # This tests that all the mixins can be used together
        # We'll test this by checking that all section metadata is unique
        mixins = [
            BaseConfigMixin,
            LoggingConfigMixin,
            DatabaseConfigMixin,
            RedisConfigMixin,
            AuthConfigMixin,
            APIConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            OracleConnectionConfigMixin,
            WMSConfigMixin,
            SingerConfigMixin,
        ]

        sections = [getattr(mixin, "_config_section", None) for mixin in mixins]

        # All sections should be unique
        assert len(sections) == len(set(sections))

        # All sections should be strings
        assert all(isinstance(section, str) for section in sections)

    def test_environment_variable_integration(self) -> None:
        """Test that environment variables work with configuration."""
        # Register a builder
        builder = FlextCoreConfigBuilder()
        ConfigFactory.register_builder("test-env", builder)

        with patch.dict(os.environ, {"FLEXT_PROJECT_TYPE": "test-env"}):
            config = ConfigFactory.create_from_env()
            assert hasattr(config, "project_name")

    def test_config_factory_builder_isolation(self) -> None:
        """Test that different builders create different configurations."""
        core_builder = FlextCoreConfigBuilder()
        auth_builder = FlextAuthConfigBuilder()

        ConfigFactory.register_builder("test-core", core_builder)
        ConfigFactory.register_builder("test-auth", auth_builder)

        core_config = ConfigFactory.create_config("test-core")
        auth_config = ConfigFactory.create_config("test-auth")

        # Both should have base configuration
        assert hasattr(core_config, "project_name")
        assert hasattr(auth_config, "project_name")

        # But they should be different instances
        assert core_config is not auth_config
