"""FLEXT Core configuration using the new system.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides the configuration for the FLEXT Core application.
It is used to configure the application's settings and dependencies.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from flext_core.config.base import BaseConfig
from flext_core.config.base import BaseSettings


class FlextDatabaseConfig(BaseConfig):
    """FLEXT database configuration."""

    url: str = Field(
        default="postgresql://flext:flext@localhost:5432/flext",
        description="Database connection URL",
    )
    pool_size: int = Field(default=20, description="Connection pool size", ge=1, le=100)
    pool_timeout: float = Field(
        default=30.0, description="Pool timeout in seconds", gt=0
    )
    echo: bool = Field(default=False, description="Echo SQL statements")

    @property
    def async_url(self) -> str:
        """Get async database URL."""
        return self.url.replace("postgresql://", "postgresql+asyncpg://")


class FlextCacheConfig(BaseConfig):
    """FLEXT cache configuration."""

    backend: str = Field(default="redis", description="Cache backend (redis, memory)")
    redis_url: str | None = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    max_connections: int = Field(default=50, description="Max Redis connections")


class FlextAPIConfig(BaseConfig):
    """FLEXT API configuration."""

    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="CORS allowed origins",
    )
    docs_url: str = Field(default="/docs", description="API documentation URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI schema URL")


class FlextObservabilityConfig(BaseConfig):
    """FLEXT observability configuration."""

    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics port")

    tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )
    tracing_endpoint: str | None = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector endpoint",
    )

    logging_level: str = Field(default="INFO", description="Logging level")
    logging_format: str = Field(
        default="json", description="Logging format (json, text)"
    )


class FlextSecurityConfig(BaseConfig):
    """FLEXT security configuration."""

    jwt_secret: str = Field(..., description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, description="JWT expiration in seconds")

    bcrypt_rounds: int = Field(default=12, description="Bcrypt hashing rounds")

    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")


class FlextSettings(BaseSettings):
    """FLEXT Core settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="FLEXT_",
        env_nested_delimiter="__",
    )

    # Component configurations
    database: FlextDatabaseConfig = FlextDatabaseConfig()
    cache: FlextCacheConfig = FlextCacheConfig()
    api: FlextAPIConfig = FlextAPIConfig()
    observability: FlextObservabilityConfig = FlextObservabilityConfig()
    security: FlextSecurityConfig = Field(
        default_factory=lambda: FlextSecurityConfig(
            jwt_secret="change-me-in-production",  # noqa: S106
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            bcrypt_rounds=12,
            rate_limit_requests=100,
            rate_limit_window=60,
        ),
        description="Security configuration",
    )

    # Paths
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".flext" / "data",
        description="Data directory",
    )
    plugins_dir: Path = Field(
        default_factory=lambda: Path.home() / ".flext" / "plugins",
        description="Plugins directory",
    )

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
_settings: FlextSettings | None = None


def get_flext_settings(*, reload: bool = False) -> FlextSettings:
    """Get FLEXT settings instance.

    Args:
        reload: Force reload settings

    Returns:
        FLEXT settings instance

    """
    global _settings  # noqa: PLW0603

    if _settings is None or reload:
        _settings = FlextSettings(
            project_name="flext",
            project_version="0.7.0",
            environment="development",
        )
        _settings.ensure_directories()

    return _settings


# Example settings.toml for Dynaconf
EXAMPLE_SETTINGS_TOML = """
# FLEXT Core Settings

[default]
project_name = "flext-core"
environment = "development"
debug = true

[default.database]
url = "postgresql://flext:flext@localhost:5432/flext_dev"
pool_size = 10
echo = true

[default.cache]
backend = "redis"
redis_url = "redis://localhost:6379/0"

[default.api]
host = "127.0.0.1"
port = 8000
reload = true

[default.observability]
logging_level = "DEBUG"
metrics_enabled = true
tracing_enabled = false

[default.security]
jwt_secret = "dev-secret-change-in-production"
rate_limit_enabled = false

# Production overrides
[production]
debug = false

[production.database]
url = "@format {env[DATABASE_URL]}"
pool_size = 50
echo = false

[production.cache]
redis_url = "@format {env[REDIS_URL]}"

[production.api]
host = "0.0.0.0"
reload = false
workers = "@int @format {env[API_WORKERS]}"

[production.observability]
logging_level = "INFO"
tracing_enabled = true
tracing_endpoint = "@format {env[OTEL_EXPORTER_OTLP_ENDPOINT]}"

[production.security]
jwt_secret = "@format {env[JWT_SECRET]}"
rate_limit_enabled = true
"""
