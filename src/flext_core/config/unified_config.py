"""Unified configuration system using composition and Pydantic v2.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a unified configuration system that eliminates duplication
across all FLEXT projects. Uses composition over inheritance, Pydantic v2 with
field composition, ClassVars, and modern Python 3.13 patterns.
"""

from __future__ import annotations

import json
import os
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import yaml
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from flext_core.domain.shared_types import URL
from flext_core.domain.shared_types import ApiKey
from flext_core.domain.shared_types import BatchSize
from flext_core.domain.shared_types import DatabaseURL
from flext_core.domain.shared_types import DurationSeconds
from flext_core.domain.shared_types import Environment
from flext_core.domain.shared_types import FilePath
from flext_core.domain.shared_types import LogLevel
from flext_core.domain.shared_types import MemoryMB
from flext_core.domain.shared_types import Password
from flext_core.domain.shared_types import Port
from flext_core.domain.shared_types import PositiveInt
from flext_core.domain.shared_types import ProjectName
from flext_core.domain.shared_types import RedisURL
from flext_core.domain.shared_types import RetryCount
from flext_core.domain.shared_types import RetryDelay
from flext_core.domain.shared_types import TimeoutSeconds
from flext_core.domain.shared_types import Username
from flext_core.domain.shared_types import Version

if TYPE_CHECKING:
    from pathlib import Path

# ==============================================================================
# CONFIGURATION COMPONENT MIXINS - COMPOSABLE PIECES
# ==============================================================================


class BaseConfigMixin(BaseModel):
    """Base configuration mixin with common fields."""

    project_name: ProjectName = Field(description="Project name")
    project_version: Version = Field(default="0.1.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)

    # ClassVars for configuration metadata
    _config_section: ClassVar[str] = "base"
    _required_env_vars: ClassVar[list[str]] = []
    _optional_env_vars: ClassVar[list[str]] = ["DEBUG"]

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate environment value."""
        if isinstance(v, str):
            return Environment(v.lower())
        if isinstance(v, Environment):
            return v
        msg = f"Invalid environment value: {v}"
        raise ValueError(msg)

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


class LoggingConfigMixin(BaseModel):
    """Logging configuration mixin."""

    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_file: FilePath | None = Field(default=None)
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log_rotation: bool = Field(default=True)
    log_retention_days: PositiveInt = Field(default=30)

    _config_section: ClassVar[str] = "logging"


class DatabaseConfigMixin(BaseModel):
    """Database configuration mixin."""

    database_url: DatabaseURL = Field(description="Database connection URL")
    database_pool_size: PositiveInt = Field(default=10)
    database_max_overflow: PositiveInt = Field(default=20)
    database_timeout: TimeoutSeconds = Field(default=30.0)
    database_echo: bool = Field(default=False)

    _config_section: ClassVar[str] = "database"
    _required_env_vars: ClassVar[list[str]] = ["DATABASE_URL"]


class RedisConfigMixin(BaseModel):
    """Redis configuration mixin."""

    redis_url: RedisURL = Field(default="redis://localhost:6379/0")
    redis_pool_size: PositiveInt = Field(default=10)
    redis_max_connections: PositiveInt = Field(default=100)
    redis_timeout: TimeoutSeconds = Field(default=5.0)
    redis_decode_responses: bool = Field(default=True)

    _config_section: ClassVar[str] = "redis"


class AuthConfigMixin(BaseModel):
    """Authentication configuration mixin."""

    jwt_secret_key: str = Field(description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: PositiveInt = Field(default=30)
    jwt_refresh_token_expire_days: PositiveInt = Field(default=7)

    password_min_length: PositiveInt = Field(default=8)
    password_require_uppercase: bool = Field(default=True)
    password_require_lowercase: bool = Field(default=True)
    password_require_numbers: bool = Field(default=True)
    password_require_symbols: bool = Field(default=False)
    password_require_special: bool = Field(default=False)
    password_bcrypt_rounds: PositiveInt = Field(default=12, ge=4, le=31)
    # Legacy alias for backwards compatibility
    bcrypt_rounds: PositiveInt = Field(default=12, ge=4, le=31)

    max_failed_login_attempts: PositiveInt = Field(default=5)
    account_lockout_duration_minutes: PositiveInt = Field(default=30)

    # Email verification settings
    require_email_verification: bool = Field(default=True)
    email_verification_token_expire_hours: PositiveInt = Field(default=24)

    # Password reset settings
    password_reset_token_expire_hours: PositiveInt = Field(default=1)

    # Session settings
    session_expire_hours: PositiveInt = Field(default=24)
    session_extend_on_activity: bool = Field(default=True)

    _config_section: ClassVar[str] = "auth"
    _required_env_vars: ClassVar[list[str]] = ["JWT_SECRET_KEY"]


class APIConfigMixin(BaseModel):
    """API configuration mixin."""

    api_host: str = Field(default="0.0.0.0")  # noqa: S104
    api_port: Port = Field(default=8000)
    api_workers: PositiveInt = Field(default=1)
    api_timeout: TimeoutSeconds = Field(default=60.0)
    api_max_request_size: MemoryMB = Field(default=10)
    api_cors_origins: list[str] = Field(default_factory=list)

    _config_section: ClassVar[str] = "api"


class PerformanceConfigMixin(BaseModel):
    """Performance configuration mixin."""

    batch_size: BatchSize = Field(default=1000)
    timeout_seconds: TimeoutSeconds = Field(default=30.0)
    max_retries: RetryCount = Field(default=3)
    retry_delay: RetryDelay = Field(default=1.0)
    connection_pool_size: PositiveInt = Field(default=10)
    memory_limit_mb: MemoryMB | None = Field(default=None)

    _config_section: ClassVar[str] = "performance"


class MonitoringConfigMixin(BaseModel):
    """Monitoring configuration mixin."""

    metrics_enabled: bool = Field(default=True)
    metrics_port: Port = Field(default=9090)
    health_check_enabled: bool = Field(default=True)
    health_check_interval: DurationSeconds = Field(default=30.0)
    tracing_enabled: bool = Field(default=False)
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    _config_section: ClassVar[str] = "monitoring"


# ==============================================================================
# ORACLE WMS SPECIFIC CONFIGURATION MIXINS
# ==============================================================================


class OracleConnectionConfigMixin(BaseModel):
    """Oracle database connection configuration mixin."""

    oracle_host: str = Field(description="Oracle database host")
    oracle_port: Port = Field(default=1521)
    oracle_service: str = Field(description="Oracle service name")
    oracle_username: Username = Field(description="Oracle username")
    oracle_password: Password = Field(description="Oracle password")
    oracle_pool_size: PositiveInt = Field(default=10)
    oracle_timeout: TimeoutSeconds = Field(default=30.0)

    _config_section: ClassVar[str] = "oracle"
    _required_env_vars: ClassVar[list[str]] = [
        "ORACLE_HOST",
        "ORACLE_SERVICE",
        "ORACLE_USERNAME",
        "ORACLE_PASSWORD",
    ]


class WMSConfigMixin(BaseModel):
    """Oracle WMS specific configuration mixin."""

    wms_environment: Environment = Field(description="WMS environment")
    wms_org_id: str = Field(description="WMS organization ID")
    wms_facility_code: str = Field(description="WMS facility code")
    wms_company_code: str = Field(description="WMS company code")

    wms_api_url: URL | None = Field(default=None)
    wms_api_username: Username | None = Field(default=None)
    wms_api_password: Password | None = Field(default=None)
    wms_api_key: ApiKey | None = Field(default=None)
    wms_api_timeout: TimeoutSeconds = Field(default=30.0)

    _config_section: ClassVar[str] = "wms"
    _required_env_vars: ClassVar[list[str]] = [
        "WMS_ORG_ID",
        "WMS_FACILITY_CODE",
        "WMS_COMPANY_CODE",
    ]


class SingerConfigMixin(BaseModel):
    """Singer protocol configuration mixin."""

    singer_catalog_path: FilePath | None = Field(default=None)
    singer_state_path: FilePath | None = Field(default=None)
    singer_config_path: FilePath | None = Field(default=None)
    singer_batch_size: BatchSize = Field(default=1000)
    singer_max_batches: PositiveInt | None = Field(default=None)

    _config_section: ClassVar[str] = "singer"


# ==============================================================================
# CONFIGURATION BUILDERS USING COMPOSITION
# ==============================================================================


class ConfigBuilder(ABC):
    """Abstract configuration builder using composition."""

    @abstractmethod
    def build_config(self, env_prefix: str = "") -> BaseModel:
        """Build configuration from components."""
        ...


class FlextCoreConfigBuilder(ConfigBuilder):
    """FLEXT Core configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_CORE_") -> BaseModel:
        """Build FLEXT Core configuration."""

        class FlextCoreConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT Core unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextCoreConfig(project_name="flext-core")


class FlextAuthConfigBuilder(ConfigBuilder):
    """FLEXT Auth configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_AUTH_") -> BaseModel:
        """Build FLEXT Auth configuration."""

        class FlextAuthConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            DatabaseConfigMixin,
            RedisConfigMixin,
            AuthConfigMixin,
            APIConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT Auth unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextAuthConfig(
            project_name="flext-auth",
            jwt_secret_key="dev-secret-key-change-in-production",
            database_url="sqlite:///./test.db",
        )


class FlextAPIConfigBuilder(ConfigBuilder):
    """FLEXT API configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_API_") -> BaseModel:
        """Build FLEXT API configuration."""

        class FlextAPIConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            DatabaseConfigMixin,
            RedisConfigMixin,
            APIConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT API unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextAPIConfig(
            project_name="flext-api",
            database_url="sqlite:///./test.db",
        )


class FlextCLIConfigBuilder(ConfigBuilder):
    """FLEXT CLI configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_CLI_") -> BaseModel:
        """Build FLEXT CLI configuration."""

        class FlextCLIConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            APIConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT CLI unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextCLIConfig(project_name="flext-cli")


class FlextWebConfigBuilder(ConfigBuilder):
    """FLEXT Web (Django) configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_WEB_") -> BaseModel:
        """Build FLEXT Web configuration."""

        class FlextWebConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            DatabaseConfigMixin,
            RedisConfigMixin,
            AuthConfigMixin,
            APIConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT Web unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextWebConfig(
            project_name="flext-web",
            jwt_secret_key="dev-secret-key-change-in-production",
            database_url="sqlite:///./test.db",
        )


class FlextGRPCConfigBuilder(ConfigBuilder):
    """FLEXT gRPC configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_GRPC_") -> BaseModel:
        """Build FLEXT gRPC configuration."""

        class FlextGRPCConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            APIConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT gRPC unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextGRPCConfig(project_name="flext-grpc")


class FlextPluginConfigBuilder(ConfigBuilder):
    """FLEXT Plugin configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_PLUGIN_") -> BaseModel:
        """Build FLEXT Plugin configuration."""

        class FlextPluginConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT Plugin unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextPluginConfig(project_name="flext-plugin")


class FlextMeltanoConfigBuilder(ConfigBuilder):
    """FLEXT Meltano configuration builder."""

    def build_config(self, env_prefix: str = "FLEXT_MELTANO_") -> BaseModel:
        """Build FLEXT Meltano configuration."""

        class FlextMeltanoConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            SingerConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """FLEXT Meltano unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return FlextMeltanoConfig(project_name="flext-meltano")


class SingerTapConfigBuilder(ConfigBuilder):
    """Singer Tap configuration builder (for all tap projects)."""

    def build_config(self, env_prefix: str = "SINGER_TAP_") -> BaseModel:
        """Build Singer Tap configuration."""

        class SingerTapConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            SingerConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """Singer Tap unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return SingerTapConfig(project_name="singer-tap")


class SingerTargetConfigBuilder(ConfigBuilder):
    """Singer Target configuration builder (for all target projects)."""

    def build_config(self, env_prefix: str = "SINGER_TARGET_") -> BaseModel:
        """Build Singer Target configuration."""

        class SingerTargetConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            SingerConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """Singer Target unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return SingerTargetConfig(project_name="singer-target")


class DBTAdapterConfigBuilder(ConfigBuilder):
    """dbt Adapter configuration builder (for all dbt projects)."""

    def build_config(self, env_prefix: str = "DBT_ADAPTER_") -> BaseModel:
        """Build dbt Adapter configuration."""

        class DBTAdapterConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            DatabaseConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """dbt Adapter unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return DBTAdapterConfig(
            project_name="dbt-adapter",
            database_url="sqlite:///./test.db",
        )


class OracleWMSConfigBuilder(ConfigBuilder):
    """Oracle WMS configuration builder."""

    def build_config(self, env_prefix: str = "ORACLE_WMS_") -> BaseModel:
        """Build Oracle WMS configuration."""

        class OracleWMSConfig(
            BaseConfigMixin,
            LoggingConfigMixin,
            OracleConnectionConfigMixin,
            WMSConfigMixin,
            SingerConfigMixin,
            PerformanceConfigMixin,
            MonitoringConfigMixin,
            BaseSettings,
        ):
            """Oracle WMS unified configuration."""

            model_config = SettingsConfigDict(
                env_prefix=env_prefix,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

        return OracleWMSConfig(
            project_name="oracle-wms",
            oracle_host="localhost",
            oracle_service="xe",
            oracle_username="test_user",
            oracle_password=os.getenv("ORACLE_TEST_PASSWORD", "changeme"),
            wms_environment=Environment.DEVELOPMENT,
            wms_org_id="test-org",
            wms_facility_code="TEST-FAC",
            wms_company_code="TEST-CO",
        )


# ==============================================================================
# CONFIGURATION FACTORY - SINGLE POINT OF CONFIGURATION CREATION
# ==============================================================================


class ConfigFactory:
    """Configuration factory using composition."""

    _builders: ClassVar[dict[str, ConfigBuilder]] = {
        # Core FLEXT Framework
        "flext-core": FlextCoreConfigBuilder(),
        "flext-auth": FlextAuthConfigBuilder(),
        "flext-api": FlextAPIConfigBuilder(),
        "flext-cli": FlextCLIConfigBuilder(),
        "flext-web": FlextWebConfigBuilder(),
        "flext-grpc": FlextGRPCConfigBuilder(),
        "flext-plugin": FlextPluginConfigBuilder(),
        "flext-meltano": FlextMeltanoConfigBuilder(),
        # Singer Protocol Projects (reusable configs)
        "singer-tap": SingerTapConfigBuilder(),
        "singer-target": SingerTargetConfigBuilder(),
        # dbt Adapter Projects (reusable configs)
        "dbt-adapter": DBTAdapterConfigBuilder(),
        # Oracle/WMS Specific
        "oracle-wms": OracleWMSConfigBuilder(),
    }

    @classmethod
    def register_builder(cls, name: str, builder: ConfigBuilder) -> None:
        """Register a new configuration builder."""
        cls._builders[name] = builder

    @classmethod
    def get_available_project_types(cls) -> list[str]:
        """Get list of available project types."""
        return list(cls._builders.keys())

    @classmethod
    def get_builder(cls, project_type: str) -> ConfigBuilder:
        """Get builder for specific project type."""
        if project_type not in cls._builders:
            available = cls.get_available_project_types()
            msg = f"Unknown project type: {project_type}. Available: {available}"
            raise ValueError(msg)
        return cls._builders[project_type]

    @classmethod
    def create_config(
        cls,
        project_type: str,
        env_prefix: str | None = None,
    ) -> BaseModel:
        """Create configuration for specific project type."""
        builder = cls.get_builder(project_type)
        if env_prefix is None:
            # Let builder use its default prefix
            return builder.build_config()
        return builder.build_config(env_prefix)

    @classmethod
    def create_from_env(cls, env_var: str = "FLEXT_PROJECT_TYPE") -> BaseModel:
        """Create configuration from environment variable."""
        project_type = os.getenv(env_var, "flext-core")
        # Don't pass env_prefix to let each builder use its default
        builder = cls.get_builder(project_type)
        return builder.build_config()


# ==============================================================================
# CONFIGURATION VALIDATOR
# ==============================================================================


class ConfigValidator:
    """Configuration validator for production readiness."""

    @staticmethod
    def validate_production_config(config: BaseModel) -> list[str]:
        """Validate configuration for production deployment."""
        issues = []

        # Check base configuration
        if (
            hasattr(config, "environment")
            and config.environment == Environment.PRODUCTION
            and hasattr(config, "debug")
            and config.debug
        ):
            issues.append("Debug mode should be disabled in production")

        # Check JWT configuration
        if hasattr(config, "jwt_secret_key") and (
            not config.jwt_secret_key or config.jwt_secret_key == "your-secret-key"
        ):
            issues.append(
                "JWT secret key must be set to a secure value in production",
            )

        # Check database configuration
        if (
            hasattr(config, "database_url")
            and hasattr(config, "environment")
            and "localhost" in config.database_url
            and config.environment == Environment.PRODUCTION
        ):
            issues.append("Database should not use localhost in production")

        # Check password security
        if (
            hasattr(config, "password_bcrypt_rounds")
            and config.password_bcrypt_rounds < 12
        ):
            issues.append(
                "Password bcrypt rounds should be at least 12 in production",
            )

        return issues


# ==============================================================================
# CONFIGURATION UTILITIES
# ==============================================================================


def load_config_from_file(file_path: Path, project_type: str) -> BaseModel:
    """Load configuration from file."""
    if not file_path.exists():
        msg = f"Configuration file not found: {file_path}"
        raise FileNotFoundError(msg)

    # Load and parse the YAML file
    with file_path.open(encoding="utf-8") as f:
        file_config = yaml.safe_load(f)

    # Get the builder for the project type
    available = ConfigFactory.get_available_project_types()
    if project_type not in available:
        msg = f"Unknown project type: {project_type}. Available: {available}"
        raise ValueError(
            msg,
        )

    builder = ConfigFactory.get_builder(project_type)

    # Build config with file values merged with environment
    config_class = type(builder.build_config())

    # Merge file config with environment variables
    merged_config = {**file_config}
    env_prefix = config_class.model_config.get("env_prefix", "")
    for key in file_config:
        env_key = f"{env_prefix}{key}".upper()
        if env_key in os.environ:
            merged_config[key] = os.environ[env_key]

    return config_class(**merged_config)


def merge_configs(*configs: BaseModel) -> dict[str, Any]:
    """Merge multiple configurations into a single dictionary."""
    merged = {}
    for config in configs:
        if hasattr(config, "model_dump"):
            merged.update(config.model_dump())
        elif hasattr(config, "dict"):
            merged.update(config.dict())
    return merged


def export_config_schema(project_type: str, output_path: Path) -> None:
    """Export configuration schema to JSON file."""
    config = ConfigFactory.create_config(project_type)
    schema = config.model_json_schema() if hasattr(config, "model_json_schema") else {}

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Configuration Mixins
    "APIConfigMixin",
    "AuthConfigMixin",
    "BaseConfigMixin",
    # Configuration Builders
    "ConfigBuilder",
    # Factory and Utilities
    "ConfigFactory",
    "ConfigValidator",
    "DBTAdapterConfigBuilder",
    "DatabaseConfigMixin",
    "FlextAPIConfigBuilder",
    "FlextAuthConfigBuilder",
    "FlextCLIConfigBuilder",
    "FlextCoreConfigBuilder",
    "FlextGRPCConfigBuilder",
    "FlextMeltanoConfigBuilder",
    "FlextPluginConfigBuilder",
    "FlextWebConfigBuilder",
    "LoggingConfigMixin",
    "MonitoringConfigMixin",
    "OracleConnectionConfigMixin",
    "OracleWMSConfigBuilder",
    "PerformanceConfigMixin",
    "RedisConfigMixin",
    "SingerConfigMixin",
    "SingerTapConfigBuilder",
    "SingerTargetConfigBuilder",
    "WMSConfigMixin",
    "export_config_schema",
    "load_config_from_file",
    "merge_configs",
]
