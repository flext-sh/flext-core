"""Unified configuration mixins for FLEXT projects.

This module provides reusable configuration mixins that eliminate duplication
across different FLEXT projects while maintaining clean architecture principles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from flext_core.domain.shared_types import EnvironmentLiteral
    from flext_core.domain.shared_types import ProjectName

# ==============================================================================
# UNIFIED CONFIGURATION MIXINS - ABSTRACT AND REUSABLE
# ==============================================================================


class BaseConfigMixin(BaseModel):
    """Base configuration mixin with common fields."""

    project_name: ProjectName = Field(description="Project name")
    project_version: str = Field(default="0.1.0", description="Project version")
    environment: EnvironmentLiteral = Field(
        default="development",
        description="Environment type",
    )
    debug: bool = Field(default=False, description="Debug mode")

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"


class LoggingConfigMixin(BaseModel):
    """Logging configuration mixin."""

    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    log_to_file: bool = Field(default=False, description="Enable file logging")
    log_file_path: str | None = Field(default=None, description="Log file path")


class DatabaseConfigMixin(BaseModel):
    """Database configuration mixin - generic."""

    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(description="Database name")
    db_user: str = Field(description="Database user")
    db_password: str = Field(description="Database password")
    db_pool_size: int = Field(default=5, description="Connection pool size")


class RedisConfigMixin(BaseModel):
    """Redis configuration mixin - generic."""

    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: str | None = Field(default=None, description="Redis password")


class AuthConfigMixin(BaseModel):
    """Authentication configuration mixin - generic."""

    auth_secret_key: str = Field(description="Authentication secret key")
    auth_algorithm: str = Field(default="HS256", description="JWT algorithm")
    auth_token_expire_minutes: int = Field(
        default=30,
        description="Token expiration in minutes",
    )


class MonitoringConfigMixin(BaseModel):
    """Monitoring configuration mixin - generic."""

    monitoring_enabled: bool = Field(default=False, description="Enable monitoring")
    metrics_port: int = Field(default=8080, description="Metrics port")
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds",
    )


class PerformanceConfigMixin(BaseModel):
    """Performance configuration mixin - generic."""

    max_workers: int = Field(default=4, description="Maximum worker processes")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    batch_size: int = Field(default=100, description="Batch processing size")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL")


class APIConfigMixin(BaseModel):
    """API configuration mixin - generic."""

    api_host: str = Field(default="127.0.0.1", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    api_cors_enabled: bool = Field(default=True, description="Enable CORS")
    api_rate_limit: int = Field(default=100, description="Rate limit per minute")


# ==============================================================================
# EXPORTS - ONLY ABSTRACT/GENERIC CONFIGURATION MIXINS
# ==============================================================================

__all__ = [
    "APIConfigMixin",
    "AuthConfigMixin",
    "BaseConfigMixin",
    "DatabaseConfigMixin",
    "LoggingConfigMixin",
    "MonitoringConfigMixin",
    "PerformanceConfigMixin",
    "RedisConfigMixin",
]
