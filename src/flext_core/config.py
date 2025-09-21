"""Configuration subsystem delivering the FLEXT 1.0.0 alignment pillar.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import threading
import tomllib
from pathlib import Path
from typing import ClassVar, Self

import yaml
from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class FlextConfig(BaseSettings):
    """Unified configuration system for FLEXT v1.0.0 based on Pydantic Settings.

    Provides consolidated configuration management aligned with the enterprise
    requirements established in ``../CLAUDE.md`` and the domain-driven-design
    practices of the FLEXT 1.0.0 stable release target.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="FLEXT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        validate_default=True,
        frozen=False,  # Allow runtime configuration updates
    )

    # Core application configuration
    app_name: str = Field(
        default="FLEXT Application",
        description="Application name for configuration identification",
    )

    version: str = Field(
        default="1.0.0",
        description="Application version identifier",
    )

    environment: str = Field(
        default="development",
        description="Deployment environment identifier",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode and verbose logging",
    )

    trace: bool = Field(
        default=False,
        description="Enable trace mode for detailed debugging",
    )

    # Logging Configuration Properties - Single Source of Truth
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    json_output: bool | None = Field(
        default=None,
        description="Use JSON output format for logging (auto-detected if None)",
        validate_default=True,
    )
    include_source: bool = Field(
        default=True,
        description="Include source code location info in log entries",
        validate_default=True,
    )
    structured_output: bool = Field(
        default=True,
        description="Use structured logging format with enhanced context",
        validate_default=True,
    )

    # Database configuration
    database_url: str | None = Field(
        default=None,
        description="Database connection URL",
    )

    database_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size",
    )

    # Cache configuration
    cache_ttl: int = Field(
        default=300,
        ge=0,
        description="Default cache TTL in seconds",
    )

    cache_max_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum cache size",
    )

    # Security configuration
    secret_key: str | None = Field(
        default=None,
        description="Secret key for security operations",
    )

    api_key: str | None = Field(
        default=None,
        description="API key for external service authentication",
    )

    # Service configuration
    max_retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for operations",
    )

    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default timeout for operations in seconds",
    )

    # Feature flags
    enable_caching: bool = Field(
        default=True,
        description="Enable caching functionality",
    )

    enable_metrics: bool = Field(
        default=False,
        description="Enable metrics collection",
    )

    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing",
    )

    # Container configuration
    max_workers: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Maximum number of workers",
    )

    # Circuit breaker configuration
    enable_circuit_breaker: bool = Field(
        default=False,
        description="Enable circuit breaker functionality",
    )

    # Validation configuration
    validation_strict_mode: bool = Field(
        default=False,
        description="Enable strict validation mode",
    )

    # Serialization configuration
    serialization_encoding: str = Field(
        default="utf-8",
        description="Default encoding for serialization",
    )

    # Dispatcher configuration
    dispatcher_auto_context: bool = Field(
        default=True,
        description="Enable automatic context propagation in dispatcher",
    )

    dispatcher_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Default dispatcher timeout in seconds",
    )

    dispatcher_enable_metrics: bool = Field(
        default=True,
        description="Enable dispatcher metrics collection",
    )

    dispatcher_enable_logging: bool = Field(
        default=True,
        description="Enable dispatcher logging",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a supported value."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            msg = f"Log level must be one of {valid_levels}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is a supported value."""
        valid_environments = {
            "development",
            "testing",
            "staging",
            "production",
        }
        if v.lower() not in valid_environments:
            msg = f"Environment must be one of {valid_environments}"
            raise ValueError(msg)
        return v.lower()

    @model_validator(mode="after")
    def validate_configuration_consistency(self) -> Self:
        """Validate overall configuration consistency."""
        # Ensure debug mode is only enabled in development
        if self.debug and self.environment == "production":
            msg = "Debug mode cannot be enabled in production environment"
            raise ValueError(msg)

        # Ensure trace mode requires debug mode
        if self.trace and not self.debug:
            msg = "Trace mode requires debug mode to be enabled"
            raise ValueError(msg)

        return self

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def get_logging_config(self) -> dict[str, object]:
        """Get logging configuration dictionary."""
        return {
            "level": self.log_level,
            "json_output": self.json_output,
            "include_source": self.include_source,
            "structured_output": self.structured_output,
        }

    def get_database_config(self) -> dict[str, object]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
        }

    def get_cache_config(self) -> dict[str, object]:
        """Get cache configuration dictionary."""
        return {
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
            "enabled": self.enable_caching,
        }

    @classmethod
    def create_for_environment(
        cls, environment: str, **overrides: object
    ) -> FlextConfig:
        """Create configuration for specific environment with overrides."""
        config_data = {"environment": environment, **overrides}
        return cls.model_validate(config_data)

    @classmethod
    def from_file(cls, config_file: Path | str) -> FlextConfig:
        """Load configuration from file (TOML, JSON, or YAML)."""
        config_path = Path(config_file)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        # Load based on file extension
        content = config_path.read_text(encoding="utf-8")

        if config_path.suffix.lower() == ".toml":
            config_data = tomllib.loads(content)
        elif config_path.suffix.lower() == ".json":
            config_data = json.loads(content)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            config_data = yaml.safe_load(content)
        else:
            msg = f"Unsupported configuration file format: {config_path.suffix}"
            raise ValueError(msg)

        return cls.model_validate(config_data)

    def to_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return self.model_dump_json(indent=2)

    def merge(self, other: FlextConfig | dict[str, object]) -> FlextConfig:
        """Merge with another configuration, returning new instance."""
        other_data = other.to_dict() if isinstance(other, FlextConfig) else other

        current_data = self.to_dict()
        merged_data = {**current_data, **other_data}
        return self.__class__.model_validate(merged_data)

    # Global instance management
    _global_instance: ClassVar[FlextConfig | None] = None
    _global_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_global_instance(cls) -> FlextConfig:
        """Get or create the global FlextConfig instance (singleton pattern)."""
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    # Create default instance from environment
                    cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def set_global_instance(cls, instance: FlextConfig) -> None:
        """Set the global FlextConfig instance."""
        with cls._global_lock:
            cls._global_instance = instance

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset the global FlextConfig instance (mainly for testing)."""
        with cls._global_lock:
            cls._global_instance = None


__all__ = ["FlextConfig"]
