"""FLEXT Configuration System - Pydantic 2.11+ BaseSettings with dependency_injector.

MANDATORY PATTERNS:
- Pydantic 2.11+ BaseSettings with dependency_injector integration
- ALL defaults from FlextConstants (ZERO module-level constants)
- Centralized config classes with comprehensive validation logic
- FlextResult for ALL operations (railway pattern)
- Sync-only operations (temporary requirement)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import ClassVar, Self

from dependency_injector import providers
from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextConfig(BaseSettings):
    """FLEXT Configuration - Pydantic 2.11+ BaseSettings with dependency_injector integration.

    MANDATORY PATTERNS:
    - Pydantic 2.11+ BaseSettings for validation
    - dependency_injector integration for service injection
    - ALL defaults from FlextConstants (ZERO module-level constants)
    - Centralized configuration with comprehensive validation logic
    - FlextResult for ALL operations (railway pattern)
    - Sync-only operations (temporary requirement)

    Core Features:
    - Pydantic validation with FlextConstants defaults
    - Dependency injection provider integration
    - Computed fields for derived configuration
    - Railway pattern error handling

    Usage:
        config = FlextConfig()
        # All defaults from FlextConstants

        # Direct access
        level = config.log_level
        timeout = config.timeout_seconds

        # Callable access
        level = config("log_level")

        # DI integration
        provider = FlextConfig.get_di_config_provider()
    """

    # Singleton pattern - per-class instances
    _instances: ClassVar[dict[type, FlextConfig]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Pydantic 2.11+ BaseSettings configuration - NO environment variables
    model_config = SettingsConfigDict(
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
        default=f"{FlextConstants.Core.NAME} Application",
        description="Application name",
    )

    version: str = Field(
        default=FlextConstants.Core.VERSION,
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
        return cls._instances[cls]  # type: ignore[return-value]

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset global singleton instance."""
        with cls._lock:
            cls._instances.pop(cls, None)

    # File operations with FlextResult
    @classmethod
    def from_file(cls, file_path: str | Path) -> FlextResult[FlextConfig]:
        """Load configuration from JSON file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return FlextResult[FlextConfig].fail(
                    f"Configuration file not found: {file_path}"
                )

            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                config = cls.model_validate(data)
                return FlextResult[FlextConfig].ok(config)

            return FlextResult[FlextConfig].fail(
                f"Unsupported file format: {path.suffix}"
            )

        except json.JSONDecodeError as e:
            return FlextResult[FlextConfig].fail(
                f"Invalid JSON in configuration file: {e}"
            )
        except Exception as e:
            return FlextResult[FlextConfig].fail(f"Failed to load configuration: {e}")

    def save_to_file(
        self, file_path: str | Path, **kwargs: object
    ) -> FlextResult[None]:
        """Save configuration to JSON file."""
        try:
            path = Path(file_path)
            config_data = self.model_dump()

            # Mask sensitive fields
            if config_data.get("secret_key"):
                config_data["secret_key"] = FlextConstants.Messages.REDACTED_SECRET
            if config_data.get("api_key"):
                config_data["api_key"] = FlextConstants.Messages.REDACTED_SECRET

            indent_value = kwargs.get(
                "indent", FlextConstants.Mixins.DEFAULT_JSON_INDENT
            )
            sort_keys_value = kwargs.get(
                "sort_keys", FlextConstants.Mixins.DEFAULT_SORT_KEYS
            )

            # Type check and convert
            indent: int | str | None = (
                int(indent_value)
                if isinstance(indent_value, (int, str))
                else FlextConstants.Mixins.DEFAULT_JSON_INDENT
            )
            sort_keys: bool = (
                bool(sort_keys_value)
                if isinstance(sort_keys_value, bool)
                else FlextConstants.Mixins.DEFAULT_SORT_KEYS
            )

            with path.open("w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=indent, sort_keys=sort_keys)

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Failed to save configuration: {e}")

    # Infrastructure protocol implementations
    def configure(self, config: FlextTypes.Dict) -> FlextResult[None]:
        """Configure component with provided settings."""
        try:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            return self.validate_runtime_requirements()
        except Exception as e:
            return FlextResult[None].fail(f"Configuration failed: {e}")

    def validate_runtime_requirements(self) -> FlextResult[None]:
        """Validate configuration meets runtime requirements."""
        try:
            self.validate_log_level(self.log_level)
        except FlextExceptions.ValidationError as e:
            return FlextResult[None].fail(str(e))

        if self.trace and not self.debug:
            return FlextResult[None].fail(
                "Trace mode requires debug mode to be enabled"
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

    @computed_field
    def cache_config(self) -> FlextTypes.Dict:
        """Get cache configuration."""
        return {
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
            "enabled": self.cache_ttl > 0,
        }

    @computed_field
    def security_config(self) -> FlextTypes.Dict:
        """Get security configuration."""
        return {
            "secret_key_configured": self.secret_key is not None,
            "api_key_configured": self.api_key is not None,
        }

    @computed_field
    def database_config(self) -> FlextTypes.Dict:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "connection_config": {
                "min_size": FlextConstants.Performance.MIN_DB_POOL_SIZE,
                "max_size": self.database_pool_size,
                "timeout_seconds": self.timeout_seconds,
                "retry_attempts": self.max_retry_attempts,
            },
        }

    @computed_field
    def logging_config(self) -> FlextTypes.Dict:
        """Get logging configuration."""
        return {
            "level": self.effective_log_level,
            "json_output": self.json_output,
            "include_source": self.include_source,
            "structured": self.structured_output,
            "verbosity": self.log_verbosity,
            "include_context": self.include_context,
            "include_correlation_id": self.include_correlation_id,
            "file": {
                "path": self.log_file,
                "max_size": self.log_file_max_size,
                "backup_count": self.log_file_backup_count,
            },
            "console": {
                "enabled": self.console_enabled,
                "color_enabled": self.console_color_enabled,
            },
            "security": {
                "mask_sensitive_data": self.mask_sensitive_data,
            },
        }

    @computed_field
    def metadata_config(self) -> FlextTypes.Dict:
        """Get application metadata."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "debug_mode": self.debug,
            "trace_mode": self.trace,
            "effective_log_level": self.effective_log_level,
            "is_debug_enabled": self.is_debug_enabled,
        }

    @computed_field
    @property
    def batch_size(self) -> int:
        """Get batch size (alias for max_batch_size)."""
        return self.max_batch_size

    def get_component_config(self, component: str) -> FlextResult[FlextTypes.Dict]:
        """Get configuration for specific flext-core component."""
        component_configs: dict[str, FlextTypes.Dict] = {
            "container": {
                "max_workers": self.max_workers,
                "enable_metrics": self.enable_metrics,
                "timeout": self.timeout_seconds,
            },
            "dispatcher": {
                "auto_context": self.dispatcher_auto_context,
                "timeout_seconds": self.dispatcher_timeout_seconds,
                "enable_metrics": self.dispatcher_enable_metrics,
                "enable_logging": self.dispatcher_enable_logging,
            },
            "logger": {
                "level": self.effective_log_level,
                "structured": self.structured_output,
                "include_context": self.include_context,
                "include_correlation_id": self.include_correlation_id,
                "mask_sensitive_data": self.mask_sensitive_data,
            },
            "cache": self.cache_config,
            "database": self.database_config,
        }

        if component not in component_configs:
            return FlextResult[FlextTypes.Dict].fail(
                f"Unknown component: {component}. Available: {list(component_configs.keys())}"
            )

        return FlextResult[FlextTypes.Dict].ok(component_configs[component])


FlextConfig.model_rebuild()

__all__ = [
    "FlextConfig",
]
