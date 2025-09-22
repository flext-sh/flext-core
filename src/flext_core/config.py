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

from flext_core.constants import FlextConstants


class FlextConfig(BaseSettings):
    """Unified configuration system for FLEXT v1.0.0 based on Pydantic Settings.

    Provides consolidated configuration management aligned with the enterprise
    requirements established in ``../CLAUDE.md`` and the domain-driven-design
    practices of the FLEXT 1.0.0 stable release target.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix=FlextConstants.Platform.ENV_PREFIX,
        env_file=FlextConstants.Platform.ENV_FILE_DEFAULT,
        env_file_encoding=FlextConstants.Mixins.DEFAULT_ENCODING,
        env_nested_delimiter=FlextConstants.Platform.ENV_NESTED_DELIMITER,
        extra="ignore",  # Changed from "forbid" to "ignore" to allow extra env vars
        use_enum_values=True,
        validate_assignment=True,
        validate_default=True,
        frozen=False,  # Allow runtime configuration updates
    )

    # Core application configuration - using FlextConstants for defaults
    app_name: str = Field(
        default=FlextConstants.Core.NAME + " Application",
        description="Application name for configuration identification",
    )

    version: str = Field(
        default=FlextConstants.Core.VERSION,
        description="Application version identifier",
    )

    environment: str = Field(
        default=FlextConstants.Defaults.ENVIRONMENT,
        description="Deployment environment identifier",
    )

    debug: bool = Field(
        default=False,  # Keep as False for production safety
        description="Enable debug mode and verbose logging",
    )

    trace: bool = Field(
        default=False,  # Keep as False for production safety
        description="Enable trace mode for detailed debugging",
    )

    # Logging Configuration Properties - Single Source of Truth using FlextConstants.Logging.
    log_level: str = Field(
        default=FlextConstants.Logging.DEFAULT_LEVEL,
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    json_output: bool | None = Field(
        default=FlextConstants.Logging.JSON_OUTPUT_DEFAULT,
        description="Use JSON output format for logging (auto-detected if None)",
        validate_default=True,
    )

    include_source: bool = Field(
        default=FlextConstants.Logging.INCLUDE_SOURCE,
        description="Include source code location info in log entries",
        validate_default=True,
    )

    structured_output: bool = Field(
        default=FlextConstants.Logging.STRUCTURED_OUTPUT,
        description="Use structured logging format with enhanced context",
        validate_default=True,
    )

    log_verbosity: str = Field(
        default=FlextConstants.Logging.VERBOSITY,
        description="Console logging verbosity (compact, detailed, full)",
    )

    # Additional logging configuration fields using FlextConstants.Logging.
    log_format: str = Field(
        default=FlextConstants.Logging.DEFAULT_FORMAT,
        description="Log message format string",
    )

    log_file: str | None = Field(
        default=None,
        description="Log file path (None for console only)",
    )

    log_file_max_size: int = Field(
        default=FlextConstants.Logging.MAX_FILE_SIZE,
        description="Maximum log file size in bytes",
    )

    log_file_backup_count: int = Field(
        default=FlextConstants.Logging.BACKUP_COUNT,
        description="Number of backup log files to keep",
    )

    console_enabled: bool = Field(
        default=FlextConstants.Logging.CONSOLE_ENABLED,
        description="Enable console logging output",
    )

    console_color_enabled: bool = Field(
        default=FlextConstants.Logging.CONSOLE_COLOR_ENABLED,
        description="Enable colored console output",
    )

    track_performance: bool = Field(
        default=FlextConstants.Logging.TRACK_PERFORMANCE,
        description="Enable performance tracking in logs",
    )

    track_timing: bool = Field(
        default=FlextConstants.Logging.TRACK_TIMING,
        description="Enable timing information in logs",
    )

    include_context: bool = Field(
        default=FlextConstants.Logging.INCLUDE_CONTEXT,
        description="Include execution context in log messages",
    )

    include_correlation_id: bool = Field(
        default=FlextConstants.Logging.INCLUDE_CORRELATION_ID,
        description="Include correlation ID in log messages",
    )

    mask_sensitive_data: bool = Field(
        default=FlextConstants.Logging.MASK_SENSITIVE_DATA,
        description="Mask sensitive data in log messages",
    )

    # Database configuration
    database_url: str | None = Field(
        default=None,
        description="Database connection URL",
    )

    database_pool_size: int = Field(
        default=FlextConstants.Performance.DEFAULT_DB_POOL_SIZE,
        ge=FlextConstants.Performance.MIN_DB_POOL_SIZE,
        le=FlextConstants.Performance.MAX_DB_POOL_SIZE,
        description="Database connection pool size",
    )

    # Cache configuration using FlextConstants
    cache_ttl: int = Field(
        default=FlextConstants.Defaults.TIMEOUT * 10,  # 300 seconds
        ge=0,
        description="Default cache TTL in seconds",
    )

    cache_max_size: int = Field(
        default=FlextConstants.Defaults.PAGE_SIZE * 10,  # 1000
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

    # Service configuration using FlextConstants
    max_retry_attempts: int = Field(
        default=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        ge=0,
        le=FlextConstants.Performance.MAX_RETRY_ATTEMPTS_LIMIT,
        description="Maximum retry attempts for operations",
    )

    timeout_seconds: int = Field(
        default=FlextConstants.Defaults.TIMEOUT,
        ge=1,
        le=FlextConstants.Performance.DEFAULT_TIMEOUT_LIMIT,
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

    # Authentication configuration
    jwt_expiry_minutes: int = Field(
        default=60,
        description="JWT token expiry time in minutes",
    )

    bcrypt_rounds: int = Field(
        default=12,
        description="BCrypt hashing rounds",
    )

    jwt_secret: str = Field(
        default="default-jwt-secret-change-in-production",
        description="JWT secret key for token signing",
    )

    # Container configuration using FlextConstants
    max_workers: int = Field(
        default=FlextConstants.Container.MAX_WORKERS,
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
        default=FlextConstants.Mixins.DEFAULT_ENCODING,
        description="Default encoding for serialization",
    )

    # Dispatcher configuration using FlextConstants
    dispatcher_auto_context: bool = Field(
        default=FlextConstants.Dispatcher.DEFAULT_AUTO_CONTEXT,
        description="Enable automatic context propagation in dispatcher",
    )

    dispatcher_timeout_seconds: int = Field(
        default=FlextConstants.Dispatcher.DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=600,
        description="Default dispatcher timeout in seconds",
    )

    dispatcher_enable_metrics: bool = Field(
        default=FlextConstants.Dispatcher.DEFAULT_ENABLE_METRICS,
        description="Enable dispatcher metrics collection",
    )

    dispatcher_enable_logging: bool = Field(
        default=FlextConstants.Dispatcher.DEFAULT_ENABLE_LOGGING,
        description="Enable dispatcher logging",
    )

    # JSON serialization configuration using FlextConstants
    json_indent: int = Field(
        default=FlextConstants.Mixins.DEFAULT_JSON_INDENT,
        ge=0,
        description="JSON indentation level",
    )

    json_sort_keys: bool = Field(
        default=FlextConstants.Mixins.DEFAULT_SORT_KEYS,
        description="Sort keys in JSON output",
    )

    ensure_json_serializable: bool = Field(
        default=True,  # Keep as True for safety
        description="Ensure JSON output is serializable",
    )

    # Timestamp configuration using FlextConstants
    use_utc_timestamps: bool = Field(
        default=FlextConstants.Mixins.DEFAULT_USE_UTC,
        description="Use UTC timestamps for all datetime fields",
    )

    timestamp_auto_update: bool = Field(
        default=FlextConstants.Mixins.DEFAULT_AUTO_UPDATE,
        description="Automatically update timestamps on changes",
    )

    # Validation configuration - using FlextConstants for defaults
    max_name_length: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum allowed name length for validation",
    )

    min_phone_digits: int = Field(
        default=10,
        ge=7,
        le=15,
        description="Minimum phone number digit count for validation",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a supported value using FlextConstants.Logging.

        Returns:
            The validated and normalized (uppercase) log level.

        Raises:
            ValueError: If the log level is not in the list of valid levels.

        """
        valid_levels = FlextConstants.Logging.VALID_LEVELS
        if v.upper() not in valid_levels:
            msg = f"Log level must be one of {valid_levels}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is a supported value using FlextConstants.

        Returns:
            The validated and normalized (lowercase) environment.

        Raises:
            ValueError: If the environment is not in the list of valid environments.

        """
        valid_environments = set(FlextConstants.Config.ENVIRONMENTS)
        if v.lower() not in valid_environments:
            msg = f"Environment must be one of {valid_environments}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("log_verbosity")
    @classmethod
    def validate_log_verbosity(cls, v: str) -> str:
        """Validate log verbosity against supported options."""
        valid_levels = FlextConstants.Logging.VALID_VERBOSITY_LEVELS
        normalized = v.lower()
        if normalized not in valid_levels:
            msg = f"Log verbosity must be one of {valid_levels}"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_configuration_consistency(self) -> Self:
        """Validate overall configuration consistency.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If configuration has inconsistent settings.

        """
        # Ensure debug mode is only enabled in development
        if (
            self.debug
            and self.environment
            == FlextConstants.Environment.ConfigEnvironment.PRODUCTION
        ):
            msg = "Debug mode cannot be enabled in production environment"
            raise ValueError(msg)

        # Ensure trace mode requires debug mode
        if self.trace and not self.debug:
            msg = "Trace mode requires debug mode to be enabled"
            raise ValueError(msg)

        return self

    def is_development(self) -> bool:
        """Check if running in development environment.

        Returns:
            bool: True if running in development environment.

        """
        return (
            self.environment == FlextConstants.Environment.ConfigEnvironment.DEVELOPMENT
        )

    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            bool: True if running in production environment.

        """
        return (
            self.environment == FlextConstants.Environment.ConfigEnvironment.PRODUCTION
        )

    def get_logging_config(self) -> dict[str, object]:
        """Get logging configuration dictionary using FlextConstants.Logging defaults.

        Returns:
            dict[str, object]: Logging configuration dictionary.

        """
        return {
            "level": self.log_level,
            "json_output": self.json_output,
            "include_source": self.include_source,
            "structured_output": self.structured_output,
            "log_verbosity": self.log_verbosity,
            "format": self.log_format,
            "log_file": self.log_file,
            "log_file_max_size": self.log_file_max_size,
            "log_file_backup_count": self.log_file_backup_count,
            "console_enabled": self.console_enabled,
            "console_color_enabled": self.console_color_enabled,
            "track_performance": self.track_performance,
            "track_timing": self.track_timing,
            "include_context": self.include_context,
            "include_correlation_id": self.include_correlation_id,
            "mask_sensitive_data": self.mask_sensitive_data,
        }

    def get_database_config(self) -> dict[str, object]:
        """Get database configuration dictionary.

        Returns:
            dict[str, object]: Database configuration dictionary.

        """
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
        }

    def get_cache_config(self) -> dict[str, object]:
        """Get cache configuration dictionary.

        Returns:
            dict[str, object]: Cache configuration dictionary.

        """
        return {
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
            "enabled": self.enable_caching,
        }

    @classmethod
    def create_for_environment(
        cls, environment: str, **overrides: object
    ) -> FlextConfig:
        """Create configuration for specific environment with overrides.

        Args:
            environment: The target environment name.
            **overrides: Configuration parameter overrides.

        Returns:
            FlextConfig: New configuration instance for the specified environment.

        """
        config_data = {"environment": environment, **overrides}
        return cls.model_validate(config_data)

    @classmethod
    def from_file(cls, config_file: Path | str) -> FlextConfig:
        """Load configuration from file (TOML, JSON, or YAML).

        Args:
            config_file: Path to the configuration file.

        Returns:
            FlextConfig: Configuration instance loaded from file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file format is not supported or contains invalid data.

        """
        config_path = Path(config_file)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        # Load based on file extension
        content = config_path.read_text(encoding=FlextConstants.Mixins.DEFAULT_ENCODING)

        if config_path.suffix.lower() == FlextConstants.Platform.EXT_TOML:
            config_data = tomllib.loads(content)
        elif config_path.suffix.lower() == FlextConstants.Platform.EXT_JSON:
            config_data = json.loads(content)
        elif config_path.suffix.lower() in {
            FlextConstants.Platform.EXT_YAML,
            FlextConstants.Platform.EXT_YML,
        }:
            config_data = yaml.safe_load(content)
        else:
            msg = f"Unsupported configuration file format: {config_path.suffix}"
            raise ValueError(msg)

        return cls.model_validate(config_data)

    def to_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary.

        Returns:
            dict[str, object]: Configuration as dictionary.

        """
        return self.model_dump()

    def to_json(self) -> str:
        """Convert configuration to JSON string.

        Returns:
            str: Configuration as formatted JSON string.

        """
        return self.model_dump_json(indent=2)

    def merge(self, other: FlextConfig | dict[str, object]) -> FlextConfig:
        """Merge with another configuration, returning new instance.

        Args:
            other: Another FlextConfig instance or dictionary to merge with.

        Returns:
            FlextConfig: New configuration instance with merged values.

        """
        other_data = other.to_dict() if isinstance(other, FlextConfig) else other

        current_data = self.to_dict()
        merged_data = {**current_data, **other_data}
        return self.__class__.model_validate(merged_data)

    # Global instance management
    _global_instance: ClassVar[FlextConfig | None] = None
    _global_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _auto_load_config_files(cls) -> dict[str, object] | None:
        """Auto-load configuration from files in current directory.
        
        File precedence: .env < config.yaml < config.toml < config.json
        
        Returns:
            dict[str, object] | None: Merged configuration data or None if no files found.

        """
        config_data: dict[str, object] = {}
        current_dir = Path.cwd()

        # Define file precedence order (lowest to highest priority)
        config_files = [
            current_dir / "config.yaml",
            current_dir / "config.yml",
            current_dir / "config.toml",
            current_dir / "config.json",
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    file_data = cls._load_config_file(config_file)
                    if file_data:
                        # Merge with existing data (higher precedence overwrites)
                        config_data.update(file_data)
                except Exception as e:
                    # Log configuration file errors but continue with other files
                    # This maintains backward compatibility while providing debugging info
                    import logging
                    logging.getLogger(__name__).debug(
                        f"Failed to load config file {config_file}: {e}"
                    )
                    continue

        return config_data or None

    @classmethod
    def _load_config_file(cls, config_file: Path) -> dict[str, object] | None:
        """Load and normalize configuration from a single file.
        
        Args:
            config_file: Path to the configuration file to load.
            
        Returns:
            dict[str, object] | None: Normalized configuration data or None if failed.

        """
        try:
            content = config_file.read_text(encoding=FlextConstants.Mixins.DEFAULT_ENCODING)

            if config_file.suffix.lower() == FlextConstants.Platform.EXT_TOML:
                raw_data = tomllib.loads(content)
            elif config_file.suffix.lower() == FlextConstants.Platform.EXT_JSON:
                raw_data = json.loads(content)
            elif config_file.suffix.lower() in {
                FlextConstants.Platform.EXT_YAML,
                FlextConstants.Platform.EXT_YML,
            }:
                raw_data = yaml.safe_load(content)
            else:
                return None

            # Normalize keys for FlextConfig compatibility
            return cls._normalize_config_keys(raw_data) if raw_data else None

        except Exception:
            return None

    @classmethod
    def _normalize_config_keys(cls, config_data: dict[str, object]) -> dict[str, object]:
        """Normalize configuration keys for FlextConfig compatibility.
        
        Handles common aliases like command_timeout -> timeout_seconds.
        
        Args:
            config_data: Raw configuration data with potentially mismatched keys.
            
        Returns:
            dict[str, object]: Configuration data with normalized keys.

        """
        normalized = dict(config_data)

        # Define key mappings for common aliases
        key_mappings = {
            "command_timeout": "timeout_seconds",
            "cmd_timeout": "timeout_seconds",
            "request_timeout": "timeout_seconds",
            "execution_timeout": "timeout_seconds",
            "log_level": "log_level",  # Keep as-is
            "logging_level": "log_level",
            "debug_mode": "debug",
            "trace_mode": "trace",
            "env": "environment",
            "stage": "environment",
        }

        # Apply key mappings
        for old_key, new_key in key_mappings.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)

        return normalized

    @classmethod
    def get_global_instance(cls) -> FlextConfig:
        """Get or create the global FlextConfig instance (singleton pattern).

        Automatically loads configuration from files if available in current directory.
        File precedence: .env < config.yaml < config.toml < config.json

        Returns:
            FlextConfig: The global singleton configuration instance.

        """
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    # Try to auto-load configuration from files
                    config_data = cls._auto_load_config_files()

                    # Create instance with loaded configuration data
                    if config_data:
                        cls._global_instance = cls.model_validate(config_data)
                    else:
                        # Create default instance from environment only
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

    @classmethod
    def clear_global_instance(cls) -> None:
        """Clear the global FlextConfig instance (alias for reset_global_instance)."""
        cls.reset_global_instance()

    def get_cqrs_bus_config(self) -> object:
        """Get CQRS bus configuration.

        Returns:
            dict: CQRS bus configuration dictionary aligned with FlextModels.CqrsConfig.Bus.

        """
        return {
            "execution_timeout": self.dispatcher_timeout_seconds,
            "enable_metrics": self.dispatcher_enable_metrics,
            "enable_logging": self.dispatcher_enable_logging,
            "enable_middleware": True,  # Default middleware enabled
            "enable_caching": True,     # Default caching enabled
        }

    @classmethod
    def create(cls, **kwargs: object) -> FlextConfig:
        """Create configuration with provided parameters.

        Args:
            **kwargs: Configuration parameters.

        Returns:
            FlextConfig: New configuration instance.

        """
        return cls.model_validate(kwargs)

    def get_metadata(self) -> dict[str, object]:
        """Get configuration metadata.

        Returns:
            dict[str, object]: Configuration metadata dictionary.

        """
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "loaded_from_file": getattr(self, "_loaded_from_file", False),
            "created_at": getattr(self, "_created_at", None),
        }


__all__ = ["FlextConfig"]
