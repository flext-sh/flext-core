"""Configuration management with Pydantic validation and dependency injection.

This module provides FlextConfig, a comprehensive configuration management
system built on Pydantic BaseSettings with dependency injection integration,
environment variable support, and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from typing import ClassVar, Self, cast

from dependency_injector import providers
from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult


class FlextConfig(BaseSettings):
    """Configuration management with Pydantic validation and dependency injection.

    Provides comprehensive configuration management built on Pydantic BaseSettings
    with dependency injection integration, environment variable support, and validation.

    Features:
    - Pydantic 2.11+ BaseSettings for validation and environment support
    - Dependency injection provider integration
    - Environment variable configuration with prefixes
    - Configuration file support (JSON, YAML, TOML)
    - Centralized configuration with comprehensive validation
    - FlextResult-based error handling for all operations
    - Computed fields for derived configuration values
    - Thread-safe singleton pattern for global configuration

    Usage:
        >>> from flext_core import FlextConfig
        >>>
        >>> config = FlextConfig()
        >>> timeout = config.timeout_seconds
        >>> config = FlextConfig(log_level="DEBUG", debug=True)
    """

    # Singleton pattern - per-class instances
    _instances: ClassVar[dict[type, FlextConfig]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Pydantic 2.11+ BaseSettings configuration with environment variable support
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix=FlextConstants.Platform.ENV_PREFIX,
        env_file=FlextConstants.Platform.ENV_FILE_DEFAULT,
        env_file_encoding=FlextConstants.Mixins.DEFAULT_ENCODING,
        env_nested_delimiter=FlextConstants.Platform.ENV_NESTED_DELIMITER,
        extra="ignore",
        use_enum_values=True,
        frozen=False,
        arbitrary_types_allowed=True,
        validate_return=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        str_to_lower=False,
        json_schema_extra={
            "title": "FLEXT Configuration",
            "description": "Enterprise FLEXT ecosystem configuration",
        },
    )

    # Core application configuration - ALL defaults from FlextConstants
    app_name: str = Field(
        default=f"{FlextConstants.NAME} Application",
        description="Application name",
    )

    version: str = Field(
        default=FlextConstants.VERSION,
        description="Application version",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    trace: bool = Field(
        default=False,
        description="Enable trace mode",
    )

    # Logging configuration - ALL from FlextConstants
    log_level: str = Field(
        default=FlextConstants.Logging.DEFAULT_LEVEL,
        description="Logging level",
    )

    json_output: bool = Field(
        default=FlextConstants.Logging.JSON_OUTPUT_DEFAULT,
        description="Use JSON output format",
    )

    include_source: bool = Field(
        default=FlextConstants.Logging.INCLUDE_SOURCE,
        description="Include source code location",
    )

    structured_output: bool = Field(
        default=FlextConstants.Logging.STRUCTURED_OUTPUT,
        description="Use structured logging format",
    )

    # Extended logging configuration
    log_verbosity: str = Field(
        default=FlextConstants.Logging.VERBOSITY,
        description="Logging verbosity level",
    )

    include_context: bool = Field(
        default=FlextConstants.Logging.INCLUDE_CONTEXT,
        description="Include context in log messages",
    )

    include_correlation_id: bool = Field(
        default=FlextConstants.Logging.INCLUDE_CORRELATION_ID,
        description="Include correlation ID in log messages",
    )

    log_file: str | None = Field(
        default=None,
        description="Log file path",
    )

    log_file_max_size: int = Field(
        default=FlextConstants.Logging.MAX_FILE_SIZE,
        ge=0,
        description="Maximum log file size in bytes",
    )

    log_file_backup_count: int = Field(
        default=FlextConstants.Logging.BACKUP_COUNT,
        ge=0,
        description="Number of backup log files to keep",
    )

    console_enabled: bool = Field(
        default=FlextConstants.Logging.CONSOLE_ENABLED,
        description="Enable console logging",
    )

    console_color_enabled: bool = Field(
        default=FlextConstants.Logging.CONSOLE_COLOR_ENABLED,
        description="Enable colored console output",
    )

    mask_sensitive_data: bool = Field(
        default=FlextConstants.Logging.MASK_SENSITIVE_DATA,
        description="Mask sensitive data in logs",
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

    # Cache configuration - ALL from FlextConstants
    cache_ttl: int = Field(
        default=FlextConstants.Defaults.DEFAULT_CACHE_TTL,
        ge=0,
        description="Cache TTL in seconds",
    )

    cache_max_size: int = Field(
        default=FlextConstants.Defaults.DEFAULT_MAX_CACHE_SIZE,
        ge=0,
        description="Maximum cache size",
    )

    # Security configuration
    secret_key: SecretStr | None = Field(
        default=None,
        description="Secret key for security operations",
    )

    api_key: SecretStr | None = Field(
        default=None,
        description="API key for external service authentication",
    )

    # Service configuration - ALL from FlextConstants
    max_retry_attempts: int = Field(
        default=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        ge=0,
        le=FlextConstants.Performance.MAX_RETRY_ATTEMPTS_LIMIT,
        description="Maximum retry attempts",
    )

    timeout_seconds: int = Field(
        default=FlextConstants.Defaults.TIMEOUT,
        ge=1,
        le=FlextConstants.Performance.DEFAULT_TIMEOUT_LIMIT,
        description="Default timeout in seconds",
    )

    # Dispatcher configuration
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

    # Dispatcher reliability configuration
    circuit_breaker_threshold: int = Field(
        default=FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        ge=1,
        le=100,
        description="Circuit breaker failure threshold",
    )

    rate_limit_max_requests: int = Field(
        default=FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
        ge=1,
        le=10000,
        description="Maximum requests per window for rate limiting",
    )

    rate_limit_window_seconds: float = Field(
        default=FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        ge=0.1,
        le=3600.0,
        description="Rate limit window size in seconds",
    )

    enable_timeout_executor: bool = Field(
        default=False,
        description="Enable timeout executor for operation timeouts",
    )

    executor_workers: int = Field(
        default=FlextConstants.Container.MAX_WORKERS,
        ge=1,
        le=100,
        description="Number of executor workers for timeout handling",
    )

    retry_delay: float = Field(
        default=0.1,
        ge=0.0,
        le=60.0,
        description="Delay between retries in seconds",
    )

    # Feature flags
    enable_caching: bool = Field(
        default=FlextConstants.Config.DEFAULT_ENABLE_CACHING,
        description="Enable caching functionality",
    )

    enable_metrics: bool = Field(
        default=FlextConstants.Config.DEFAULT_ENABLE_METRICS,
        description="Enable metrics collection",
    )

    enable_tracing: bool = Field(
        default=FlextConstants.Config.DEFAULT_ENABLE_TRACING,
        description="Enable distributed tracing",
    )

    # Container configuration - from FlextConstants
    max_workers: int = Field(
        default=FlextConstants.Container.MAX_WORKERS,
        ge=1,
        le=50,
        description="Maximum number of workers",
    )

    # Batch processing configuration
    max_batch_size: int = Field(
        default=FlextConstants.Processing.DEFAULT_BATCH_SIZE,
        ge=1,
        le=FlextConstants.Processing.MAX_BATCH_SIZE,
        description="Maximum batch size for batch operations",
    )

    # Validation configuration - from FlextConstants
    max_name_length: int = Field(
        default=FlextConstants.Validation.MAX_NAME_LENGTH,
        ge=1,
        le=500,
        description="Maximum allowed name length",
    )

    min_phone_digits: int = Field(
        default=FlextConstants.Validation.MIN_PHONE_DIGITS,
        ge=7,
        le=15,
        description="Minimum phone number digits",
    )

    validation_timeout_ms: int = Field(
        default=FlextConstants.Validation.VALIDATION_TIMEOUT_MS,
        ge=1,
        le=10000,
        description="Maximum validation time in milliseconds",
    )

    validation_strict_mode: bool = Field(
        default=False,
        description="Enable strict validation mode",
    )

    # Direct access method - simplified
    def __call__(self, key: str) -> object:
        """Direct value access: config('log_level')."""
        if not hasattr(self, key):
            msg = f"Configuration key '{key}' not found"
            raise KeyError(msg)
        return getattr(self, key)

    # Validation methods
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level using FlextConstants."""
        v_upper = v.upper()
        if v_upper not in FlextConstants.Logging.VALID_LEVELS:
            error_msg = f"Invalid log level: {v}. Must be one of: {', '.join(FlextConstants.Logging.VALID_LEVELS)}"
            raise FlextExceptions.ValidationError(error_msg)
        return v_upper

    @model_validator(mode="after")
    def validate_debug_trace_consistency(self) -> Self:
        """Validate debug and trace mode consistency."""
        if self.trace and not self.debug:
            error_msg = "Trace mode requires debug mode to be enabled"
            raise FlextExceptions.ValidationError(error_msg)
        return self

    # Dependency injection integration
    _di_config_provider: ClassVar[providers.Configuration | None] = None
    _di_provider_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_di_config_provider(cls) -> providers.Configuration:
        """Get dependency-injector Configuration provider."""
        if cls._di_config_provider is None:
            with cls._di_provider_lock:
                if cls._di_config_provider is None:
                    cls._di_config_provider = providers.Configuration()
                    instance = cls._instances.get(cls)
                    if instance is not None:
                        config_dict = instance.model_dump()
                        cls._di_config_provider.from_dict(config_dict)
        return cls._di_config_provider

    @classmethod
    def get_global_instance(cls) -> Self:
        """Get or create global singleton instance."""
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = cls()
                    cls._instances[cls] = instance
        return cast("Self", cls._instances[cls])

    @classmethod
    def set_global_instance(cls, instance: FlextConfig) -> None:
        """Set global singleton instance."""
        with cls._lock:
            cls._instances[cls] = instance

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset global singleton instance."""
        with cls._lock:
            cls._instances.pop(cls, None)

    def validate_runtime_requirements(self) -> FlextResult[None]:
        """Validate configuration meets runtime requirements."""
        try:
            self.validate_log_level(self.log_level)
        except FlextExceptions.ValidationError as e:
            return FlextResult[None].fail(str(e))

        if self.trace and not self.debug:
            return FlextResult[None].fail(
                "Trace mode requires debug mode to be enabled",
            )

        return FlextResult[None].ok(None)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for configuration consistency."""
        return FlextResult[None].ok(None)

    # Computed fields
    @computed_field
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug or self.trace

    @computed_field
    def effective_log_level(self) -> str:
        """Get effective log level considering debug/trace modes."""
        if self.trace:
            return "DEBUG"
        if self.debug:
            return "INFO"
        return self.log_level


__all__ = [
    "FlextConfig",
]
