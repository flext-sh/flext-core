"""Enhanced Base Configuration with enterprise features.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Implements Phase 2 Advanced Configuration Patterns.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class LogLevel(StrEnum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(StrEnum):
    """Environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class EnhancedBaseConfig(BaseSettings):
    """Universal configuration with enterprise features.

    Provides a standardized configuration base class that all FLEXT
    projects can inherit from to ensure consistent configuration
    patterns across the entire platform.

    Features:
    - Hierarchical configuration loading
    - Environment variable support
    - Validation with Pydantic v2
    - Type safety
    - Secret management
    - Environment-specific defaults
    """

    # Core application settings
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    environment: Environment = Environment.DEVELOPMENT

    # Application metadata
    app_name: str = "flext-application"
    app_version: str = "1.0.0"

    # Security settings
    secret_key: str = "default-development-key-that-is-long-enough-for-validation"  # noqa: S105
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]

    # Database settings (optional - projects can override)
    database_url: str | None = None
    database_pool_size: int = 5
    database_timeout: int = 30

    # Observability settings
    enable_metrics: bool = True
    enable_tracing: bool = False
    metrics_port: int = 9090

    # Configuration validation
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # Changed to allow for testing compatibility
        validate_assignment=True,
        case_sensitive=False,
        env_prefix="FLEXT_",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str | LogLevel) -> LogLevel:
        """Validate and convert log level."""
        if isinstance(v, LogLevel):
            return v
        # Must be string due to type hints
        upper_value = v.upper()
        if upper_value in LogLevel.__members__:
            return LogLevel(upper_value)
        msg = f"Invalid log level: {v}"
        raise ValueError(msg)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str | Environment) -> Environment:
        """Validate and convert environment."""
        if isinstance(v, Environment):
            return v
        # Must be string due to type hints
        lower_value = v.lower()
        # Handle common aliases
        if lower_value == "test":
            lower_value = "testing"

        if lower_value in Environment.__members__.values():
            return Environment(lower_value)
        msg = f"Invalid environment: {v}"
        raise ValueError(msg)

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key security."""
        if len(v) < 32:
            msg = "Secret key must be at least 32 characters long"
            raise ValueError(msg)

        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str | None) -> str | None:
        """Validate database URL format."""
        if v is None:
            return v

        # Basic URL validation
        if not v.startswith(("postgresql://", "sqlite://", "mysql://", "oracle://")):
            msg = f"Unsupported database URL scheme: {v}"
            raise ValueError(msg)

        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING

    def get_config_summary(self) -> dict[str, Any]:
        """Get sanitized configuration summary for logging."""
        config_dict: dict[str, Any] = self.model_dump()

        # Remove sensitive information
        sensitive_keys = {"secret_key", "database_url"}
        for key in sensitive_keys:
            if key in config_dict:
                config_dict[key] = "[HIDDEN]"

        return config_dict

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return any issues."""
        issues = []

        # Production-specific validations
        if self.is_production:
            if self.debug:
                issues.append("Debug mode should not be enabled in production")

            if (
                self.secret_key
                == "default-development-key-that-is-long-enough-for-validation"  # noqa: S105
            ):  # Security check comparison
                issues.append("Default secret key detected in production")

            if self.log_level == LogLevel.DEBUG:
                issues.append("Debug logging should not be used in production")

        # Database validations
        if (
            self.database_url
            and self.is_production
            and ("localhost" in self.database_url or "127.0.0.1" in self.database_url)
        ):
            issues.append("Production should not use localhost database")

        return issues

    def setup_environment(self) -> None:
        """Set up environment-specific configuration."""
        # Set environment variables for consistency
        os.environ["FLEXT_DEBUG"] = str(self.debug)
        os.environ["FLEXT_LOG_LEVEL"] = self.log_level.value
        os.environ["FLEXT_ENVIRONMENT"] = self.environment.value

        # Create necessary directories
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        if self.enable_metrics:
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)

    @classmethod
    def load_from_file(cls, config_file: Path | str) -> EnhancedBaseConfig:
        """Load configuration from a specific file."""
        config_path = Path(config_file)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        # Load environment variables from the specified file
        # We'll use os.environ to temporarily override env_file loading
        original_env = os.environ.copy()

        try:
            # Read the file and set environment variables
            if config_path.suffix in {".env", ".txt"}:
                with config_path.open("r", encoding="utf-8") as f:
                    for file_line in f:
                        line = file_line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ[key.strip()] = value.strip()

            # Create instance with current environment
            return cls()
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


class DatabaseConfig(EnhancedBaseConfig):
    """Database-specific configuration."""

    # Database is required for this config
    database_url: str
    database_pool_size: int = 10
    database_timeout: int = 60
    database_echo: bool = False

    # Connection pool settings
    pool_pre_ping: bool = True
    pool_recycle: int = 3600

    @field_validator("database_url")
    @classmethod
    def validate_database_url_required(cls, v: str) -> str:
        """Validate that database URL is provided."""
        if not v:
            msg = "Database URL is required"
            raise ValueError(msg)

        # Apply same validation as parent class
        if not v.startswith(("postgresql://", "sqlite://", "mysql://", "oracle://")):
            msg = f"Unsupported database URL scheme: {v}"
            raise ValueError(msg)

        return v


class APIConfig(EnhancedBaseConfig):
    """API-specific configuration."""

    # API server settings - defaults to localhost for security
    host: str = "127.0.0.1"  # Use localhost by default for security
    port: int = 8000
    workers: int = 1

    # API security
    enable_cors: bool = True
    cors_origins: list[str] = ["*"]
    api_key_required: bool = False

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            msg = f"Port must be between 1 and 65535, got: {v}"
            raise ValueError(msg)

        return v


class ObservabilityConfig(EnhancedBaseConfig):
    """Observability and monitoring configuration."""

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # Tracing
    enable_tracing: bool = False
    tracing_endpoint: str | None = None
    tracing_sample_rate: float = 0.1

    # Logging
    log_format: str = "json"
    log_file: str | None = None
    log_rotation: bool = True
    log_retention_days: int = 30

    @field_validator("tracing_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        """Validate tracing sample rate."""
        if not 0.0 <= v <= 1.0:
            msg = f"Sample rate must be between 0.0 and 1.0, got: {v}"
            raise ValueError(msg)

        return v
