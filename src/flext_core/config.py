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
from pydantic import (
    Field,
    SecretStr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
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
        """Get or create global singleton instance.

        Note: Uses FlextConfig as the singleton key to ensure all subclasses
        (including FlextBase.Config and FlextCore.Config) share the same instance.
        This is intentional to maintain a single global configuration across the
        entire flext-core ecosystem, regardless of access pattern.

        If a more derived subclass is requested after a base class instance was
        created, the singleton is upgraded to the more derived type to maintain
        type compatibility with isinstance checks.

        Returns:
            Self: The global singleton instance

        """
        # Use base class (FlextConfig) as key to ensure single global singleton
        # Subclasses (FlextBase.Config, FlextCore.Config) share this instance
        base_class = FlextConfig

        if base_class not in cls._instances:
            # No instance exists - create one
            with cls._lock:
                if base_class not in cls._instances:
                    instance = cls()
                    cls._instances[base_class] = instance
        else:
            # Instance exists - check if it's compatible with requested class
            stored = cls._instances[base_class]
            if not isinstance(stored, cls):
                # Stored instance is less derived than requested class
                # Upgrade singleton to more derived type for isinstance compatibility
                with cls._lock:
                    # Double-check after acquiring lock
                    stored = cls._instances[base_class]
                    if not isinstance(stored, cls):
                        instance = cls()
                        cls._instances[base_class] = instance

        return cast("Self", cls._instances[base_class])

    @classmethod
    def set_global_instance(cls, instance: FlextConfig) -> None:
        """Set global singleton instance.

        Uses FlextConfig as the key to match get_global_instance() behavior,
        ensuring all subclasses (FlextBase.Config, FlextCore.Config) properly
        set the shared singleton.

        Args:
            instance: The config instance to set as global singleton

        """
        with cls._lock:
            base_class = FlextConfig
            cls._instances[base_class] = instance  # Use base_class key, not cls

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset global singleton instance.

        Uses FlextConfig as the key to match get_global_instance() behavior,
        ensuring all subclasses (FlextBase.Config, FlextCore.Config) properly
        reset the shared singleton.
        """
        with cls._lock:
            base_class = FlextConfig
            cls._instances.pop(base_class, None)  # Use base_class key, not cls

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

    # Computed fields - Enhanced Pydantic 2 features
    @computed_field
    @property
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug or self.trace

    @computed_field
    @property
    def effective_log_level(self) -> str:
        """Get effective log level considering debug/trace modes."""
        if self.trace:
            return "DEBUG"
        if self.debug:
            return "INFO"
        return self.log_level

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production mode (not debug/trace)."""
        return not (self.debug or self.trace)

    @computed_field
    @property
    def effective_timeout(self) -> int:
        """Get effective timeout considering debug mode (longer timeout for debugging)."""
        if self.debug or self.trace:
            return self.timeout_seconds * 3  # 3x timeout for debugging
        return self.timeout_seconds

    @computed_field
    @property
    def has_database(self) -> bool:
        """Check if database is configured."""
        return self.database_url is not None and len(self.database_url) > 0

    @computed_field
    @property
    def has_cache(self) -> bool:
        """Check if caching is enabled and configured."""
        return self.enable_caching and self.cache_max_size > 0

    # Field serializers for SecretStr masking
    @field_serializer("secret_key", "api_key", when_used="json")
    def serialize_secrets(self, value: SecretStr | None) -> str:
        """Mask secret values in JSON serialization."""
        if value is None:
            return ""
        return "***MASKED***"

    # =========================================================================
    # REUSABLE VALIDATORS - For ecosystem-wide consistency
    # =========================================================================

    @classmethod
    def validate_log_level_field(cls, v: str) -> str:
        """Reusable validator for log level fields.

        Validates log levels against standard set (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Can be used by subclasses via field_validator.

        Args:
            v: Log level string to validate

        Returns:
            str: Validated and normalized log level (uppercase)

        Raises:
            ValueError: If log level is invalid

        Example:
            >>> @field_validator("log_level")
            >>> @classmethod
            >>> def validate_log_level(cls, v: str) -> str:
            ...     return cls.validate_log_level_field(v)

        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level_upper = v.upper()
        if level_upper not in valid_levels:
            msg = f"Invalid log level '{v}'. Must be one of: {', '.join(sorted(valid_levels))}"
            raise ValueError(msg)
        return level_upper

    @classmethod
    def validate_log_verbosity_field(cls, v: str) -> str:
        """Reusable validator for log verbosity fields.

        Validates verbosity against standard set (compact, detailed, full).
        Can be used by subclasses via field_validator.

        Args:
            v: Verbosity string to validate

        Returns:
            str: Validated and normalized verbosity (lowercase)

        Raises:
            ValueError: If verbosity is invalid

        Example:
            >>> @field_validator("log_verbosity")
            >>> @classmethod
            >>> def validate_verbosity(cls, v: str) -> str:
            ...     return cls.validate_log_verbosity_field(v)

        """
        valid_verbosity = {"compact", "detailed", "full"}
        verbosity_lower = v.lower()
        if verbosity_lower not in valid_verbosity:
            msg = f"Invalid log verbosity '{v}'. Must be one of: {', '.join(sorted(valid_verbosity))}"
            raise ValueError(msg)
        return verbosity_lower

    @classmethod
    def validate_environment_field(cls, v: str) -> str:
        """Reusable validator for environment fields.

        Validates environment against standard set (development, staging, production, test).
        Can be used by subclasses via field_validator.

        Args:
            v: Environment string to validate

        Returns:
            str: Validated and normalized environment (lowercase)

        Raises:
            ValueError: If environment is invalid

        Example:
            >>> @field_validator("environment")
            >>> @classmethod
            >>> def validate_env(cls, v: str) -> str:
            ...     return cls.validate_environment_field(v)

        """
        valid_environments = {"development", "staging", "production", "test"}
        env_lower = v.lower()
        if env_lower not in valid_environments:
            msg = f"Invalid environment '{v}'. Must be one of: {', '.join(sorted(valid_environments))}"
            raise ValueError(msg)
        return env_lower

    # =========================================================================
    # CONFIGURATION UTILITY METHODS - For ecosystem-wide reuse
    # =========================================================================

    def update_from_dict(self, **kwargs: object) -> FlextResult[None]:
        """Update configuration from dictionary with validation.

        Allows dynamic override of configuration values with Pydantic validation.
        Only valid configuration fields are updated.

        Args:
            **kwargs: Configuration key-value pairs to update

        Returns:
            FlextResult[None]: Success or validation error

        Example:
            >>> config = FlextConfig()
            >>> result = config.update_from_dict(log_level="DEBUG", debug=True)
            >>> result.is_success
            True

        """
        try:
            # Filter only valid configuration fields
            valid_updates = {
                key: value for key, value in kwargs.items() if hasattr(self, key)
            }

            # Apply updates using Pydantic's validation
            for key, value in valid_updates.items():
                setattr(self, key, value)

            # Re-validate entire model to ensure consistency
            self.model_validate(self.model_dump())

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Configuration update failed: {e}")

    def merge_with_env_vars(self) -> FlextResult[None]:
        """Re-load environment variables and merge with current config.

        Useful when environment variables change during runtime.
        Existing config values take precedence over environment variables.

        Returns:
            FlextResult[None]: Success or error

        Example:
            >>> config = FlextConfig()
            >>> # Environment changes
            >>> import os
            >>> os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"
            >>> result = config.merge_with_env_vars()
            >>> config.log_level  # Will be "DEBUG" if not explicitly set

        """
        try:
            # Get current config snapshot
            current_config = self.model_dump()

            # Create new instance from environment
            env_config = self.__class__()

            # Merge: current config overrides env
            for key, value in current_config.items():
                if value != getattr(self.__class__(), key, None):
                    # Value was explicitly set, keep it
                    setattr(env_config, key, value)

            # Copy merged config back
            for key in current_config:
                setattr(self, key, getattr(env_config, key))

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Environment merge failed: {e}")

    def validate_overrides(self, **overrides: object) -> FlextResult[dict[str, object]]:
        """Validate configuration overrides without applying them.

        Useful for checking if overrides are valid before applying.

        Args:
            **overrides: Configuration overrides to validate

        Returns:
            FlextResult[dict[str, object]]: Valid overrides or validation errors

        Example:
            >>> config = FlextConfig()
            >>> result = config.validate_overrides(log_level="DEBUG", max_workers=10)
            >>> if result.is_success:
            ...     config.update_from_dict(**result.unwrap())

        """
        try:
            valid_overrides: dict[str, object] = {}
            errors: list[str] = []

            for key, value in overrides.items():
                # Check if field exists
                if not hasattr(self, key):
                    errors.append(f"Unknown configuration field: '{key}'")
                    continue

                # Try to validate the value
                try:
                    # Create test instance with override
                    test_config = self.model_copy()
                    setattr(test_config, key, value)
                    test_config.model_validate(test_config.model_dump())
                    valid_overrides[key] = value
                except Exception as e:
                    errors.append(f"Invalid value for '{key}': {e}")

            if errors:
                return FlextResult[dict[str, object]].fail(
                    f"Validation errors: {'; '.join(errors)}"
                )

            return FlextResult[dict[str, object]].ok(valid_overrides)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Validation failed: {e}")


__all__ = [
    "FlextConfig",
]
