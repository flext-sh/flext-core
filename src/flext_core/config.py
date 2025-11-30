"""FlextConfig - Configuration Management Module.

This module provides comprehensive configuration management for the FLEXT ecosystem,
implementing Pydantic v2 BaseSettings with dependency injection, environment variable support,
and runtime validation. Serves as the foundation layer (0.5) controlling all other layers.

Scope: Global configuration management, singleton pattern, DI integration, validation,
environment variable handling, thread-safe operations, and dynamic config updates.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self, TypeVar

from dependency_injector import providers
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.__version__ import __version__
from flext_core.constants import FlextConstants
from flext_core.runtime import FlextRuntime, StructlogLogger
from flext_core.typings import FlextTypes, T_Namespace

# TypeVar for decorator type preservation - bound to BaseSettings
_TSettings = TypeVar("_TSettings", bound=BaseSettings)


def _resolve_env_file_impl() -> str | None:
    """Internal implementation of env file resolution.

    This function is called at module load time, before FlextConfig class exists.
    """
    # Check for custom env file path
    custom_env_file = os.environ.get(FlextConstants.Platform.ENV_FILE_ENV_VAR)
    if custom_env_file:
        custom_path = Path(custom_env_file)
        if custom_path.exists():
            return str(custom_path.resolve())
        # If custom path doesn't exist, return it anyway (Pydantic will handle gracefully)
        return custom_env_file

    # Default: use .env from current directory
    default_path = Path.cwd() / FlextConstants.Platform.ENV_FILE_DEFAULT
    if default_path.exists():
        return str(default_path.resolve())

    # Fallback: use default value (Pydantic handles missing file gracefully)
    return FlextConstants.Platform.ENV_FILE_DEFAULT


class FlextConfig(BaseSettings):
    """Configuration management with Pydantic validation and dependency injection.

    Architecture: Layer 0.5 (Configuration Foundation)
    Provides enterprise-grade configuration management for the FLEXT ecosystem
    through Pydantic v2 BaseSettings, implementing structural typing via
    FlextProtocols.Configurable (duck typing - no inheritance required).

    Core Features:
    - Pydantic v2 BaseSettings with type-safe configuration
    - Environment variable support with FLEXT_ prefix
    - Thread-safe singleton pattern
    - Dependency injection integration
    - Runtime configuration updates
    """

    # Singleton pattern
    _instances: ClassVar[dict[type[BaseSettings], BaseSettings]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    @property
    def logger(self) -> StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods.
        """
        return FlextRuntime.get_logger(__name__)

    # Configuration fields
    # env_file resolved at module load time via FLEXT_ENV_FILE env var
    model_config = SettingsConfigDict(
        env_prefix=FlextConstants.Platform.ENV_PREFIX,
        env_nested_delimiter=FlextConstants.Platform.ENV_NESTED_DELIMITER,
        env_file=_resolve_env_file_impl(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # ENV FILE RESOLUTION - Public API for namespace configs
    # =========================================================================

    @staticmethod
    def resolve_env_file() -> str | None:
        """Resolve .env file path from FLEXT_ENV_FILE environment variable.

        This method is the PUBLIC API for all namespace configs to use.
        It ensures all FLEXT ecosystem configs use the same .env resolution logic.

        Precedence (highest to lowest):
        1. FLEXT_ENV_FILE environment variable (custom path)
        2. Default .env file from current directory

        Returns:
            str | None: Path to .env file or None if not found

        Example:
            # In namespace config classes (e.g., FlextLdapConfig)
            model_config = SettingsConfigDict(
                env_prefix="FLEXT_LDAP_",
                env_file=FlextConfig.resolve_env_file(),
                ...
            )

        """
        return _resolve_env_file_impl()

    # Core configuration
    app_name: str = Field(default="flext", description="Application name")
    version: str = Field(default=__version__, description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    trace: bool = Field(default=False, description="Enable trace mode")

    # Logging configuration
    log_level: FlextConstants.Literals.LogLevelLiteral = Field(
        default=FlextConstants.Settings.LogLevel.INFO,
        description="Log level",
    )
    json_output: bool = Field(
        default=FlextConstants.Logging.JSON_OUTPUT_DEFAULT,
        description="JSON log output",
    )
    include_source: bool = Field(
        default=FlextConstants.Logging.INCLUDE_SOURCE,
        description="Include source in logs",
    )
    log_verbosity: str = Field(
        default=FlextConstants.Logging.VERBOSITY,
        description="Log verbosity",
    )
    include_context: bool = Field(
        default=FlextConstants.Logging.INCLUDE_CONTEXT,
        description="Include context in logs",
    )
    include_correlation_id: bool = Field(
        default=FlextConstants.Logging.INCLUDE_CORRELATION_ID,
        description="Include correlation ID in logs",
    )
    log_file: str | None = Field(default=None, description="Log file path")
    log_file_max_size: int = Field(
        default=FlextConstants.Logging.MAX_FILE_SIZE,
        description="Max log file size",
    )
    log_file_backup_count: int = Field(
        default=FlextConstants.Logging.BACKUP_COUNT,
        description="Log file backup count",
    )
    console_enabled: bool = Field(
        default=FlextConstants.Logging.CONSOLE_ENABLED,
        description="Enable console logging",
    )
    console_color_enabled: bool = Field(
        default=FlextConstants.Logging.CONSOLE_COLOR_ENABLED,
        description="Enable console colors",
    )

    # Cache configuration
    enable_caching: bool = Field(
        default=FlextConstants.Settings.DEFAULT_ENABLE_CACHING,
        description="Enable caching",
    )
    cache_ttl: int = Field(
        default=FlextConstants.Defaults.CACHE_TTL,
        description="Cache TTL",
    )

    # Database configuration
    database_url: str | None = Field(default=None, description="Database URL")
    database_pool_size: int = Field(
        default=FlextConstants.Performance.DEFAULT_DB_POOL_SIZE,
        description="Database pool size",
    )

    # Reliability configuration
    circuit_breaker_threshold: int = Field(
        default=FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD,
        description="Circuit breaker threshold",
    )
    rate_limit_max_requests: int = Field(
        default=FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
        description="Rate limit max requests",
    )
    rate_limit_window_seconds: int = Field(
        default=FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        description="Rate limit window",
    )
    retry_delay: int = Field(
        default=FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
        description="Retry delay",
    )
    max_retry_attempts: int = Field(
        default=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        description="Max retry attempts",
    )

    # Dispatcher configuration
    enable_timeout_executor: bool = Field(
        default=True,
        description="Enable timeout executor",
    )
    dispatcher_enable_logging: bool = Field(
        default=FlextConstants.Dispatcher.DEFAULT_ENABLE_LOGGING,
        description="Enable dispatcher logging",
    )
    dispatcher_auto_context: bool = Field(
        default=FlextConstants.Dispatcher.DEFAULT_AUTO_CONTEXT,
        description="Auto context in dispatcher",
    )
    dispatcher_timeout_seconds: float = Field(
        default=FlextConstants.Dispatcher.DEFAULT_TIMEOUT_SECONDS,
        description="Dispatcher timeout",
    )
    dispatcher_enable_metrics: bool = Field(
        default=FlextConstants.Dispatcher.DEFAULT_ENABLE_METRICS,
        description="Enable dispatcher metrics",
    )
    executor_workers: int = Field(
        default=FlextConstants.Container.DEFAULT_WORKERS,
        description="Executor workers",
    )

    # Processing configuration
    timeout_seconds: float = Field(
        default=FlextConstants.Network.DEFAULT_TIMEOUT,
        description="Default timeout",
    )
    max_workers: int = Field(
        default=FlextConstants.Processing.DEFAULT_MAX_WORKERS,
        description="Max workers",
    )
    max_batch_size: int = Field(
        default=FlextConstants.Processing.MAX_BATCH_SIZE,
        description="Max batch size",
    )

    # Security configuration
    api_key: str | None = Field(default=None, description="API key")
    mask_sensitive_data: bool = Field(
        default=FlextConstants.Logging.MASK_SENSITIVE_DATA,
        description="Mask sensitive data",
    )

    # Exception configuration
    # Note: Using FailureLevel StrEnum directly for type safety
    exception_failure_level: FlextConstants.Exceptions.FailureLevel = Field(
        default=FlextConstants.Exceptions.FAILURE_LEVEL_DEFAULT,
        description="Exception failure level",
    )

    def __new__(cls, **_kwargs: object) -> Self:
        """Create singleton instance.

        Note: BaseSettings.__init__ accepts **values: Any internally.
        We override __new__ to implement singleton pattern while allowing
        kwargs to be passed for testing and configuration via model_validate.
        """
        base_class = cls
        if base_class not in cls._instances:
            with cls._lock:
                if base_class not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[base_class] = instance
        raw_instance = cls._instances[base_class]
        if not isinstance(raw_instance, cls):
            msg = f"Singleton instance is not of expected type {cls.__name__}"
            raise TypeError(msg)
        return raw_instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance for testing purposes.

        This method is intended for use in tests only to allow
        clean state between test runs.
        """
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]

    def __init__(self, **kwargs: str | int | bool | None) -> None:
        """Initialize config with data.

        Note: BaseSettings handles initialization from environment variables,
        .env files, and other sources automatically. Kwargs can be passed for
        testing and explicit configuration (used by model_validate).
        """
        # Check if already initialized (singleton pattern)
        if hasattr(self, "_di_provider"):
            # Instance already initialized, just update fields from kwargs
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            return

        # First initialization - call BaseSettings.__init__() then apply kwargs
        # BaseSettings loads from environment/files, then we apply explicit kwargs
        # Resolve env_file dynamically to support directory changes in tests
        env_file_path = _resolve_env_file_impl()
        # Temporarily update model_config to use resolved path
        original_env_file = self.model_config.get("env_file")
        if env_file_path != original_env_file:
            # Update model_config for this instance
            self.model_config = SettingsConfigDict(**{
                **self.model_config,
                "env_file": env_file_path,
            })
        super().__init__()
        self._di_provider: providers.Singleton[Self] | None = None

        # Apply explicit kwargs after BaseSettings initialization (overrides env values)
        # Validate kwargs using Pydantic validators before applying
        if kwargs:
            # Apply kwargs directly - BaseSettings.__init__ already called above
            # Validate kwargs by applying them individually
            for key, value in kwargs.items():
                if key in self.model_fields:
                    # Validate using field validator if exists
                    validated_value = value
                    # Check if there's a field validator for this field
                    if hasattr(self.__class__, f"validate_{key}"):
                        validator = getattr(self.__class__, f"validate_{key}")
                        validated_value = validator(value)
                    # Apply validated value
                    object.__setattr__(self, key, validated_value)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(
        cls,
        v: str | FlextConstants.Settings.LogLevel,
    ) -> FlextConstants.Literals.LogLevelLiteral:
        """Validate and normalize log level against allowed values.

        Accepts string or LogLevel StrEnum, normalizes to uppercase string.
        """
        if isinstance(v, FlextConstants.Settings.LogLevel):
            # v is already LogLevel enum member, which is compatible with LogLevelLiteral
            # LogLevelLiteral is Literal[LogLevel.DEBUG, LogLevel.INFO, ...]
            # Return the enum member directly (compatible with Literal type)
            return v  # type: ignore[return-value]
        normalized = v.upper()
        # Validate against StrEnum values and return the enum value (Literal type)
        try:
            # Return LogLevel enum member, compatible with LogLevelLiteral
            # LogLevelLiteral is Literal[LogLevel.DEBUG, LogLevel.INFO, ...]
            return FlextConstants.Settings.LogLevel(normalized)  # type: ignore[return-value]
        except ValueError:
            log_level_enum = FlextConstants.Settings.LogLevel
            allowed_values = [
                level.value for level in log_level_enum.__members__.values()
            ]
            msg = f"Invalid log level: {v}. Must be one of {allowed_values}"
            raise ValueError(msg) from None

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Validate configuration."""
        # Check database URL scheme if provided
        if self.database_url and not self.database_url.startswith((
            "postgresql://",
            "mysql://",
            "sqlite://",
        )):
            msg = "Invalid database URL scheme"
            raise ValueError(msg)

        # Check that trace mode requires debug mode
        if self.trace and not self.debug:
            msg = "Trace mode requires debug mode"
            raise ValueError(msg)

        return self

    @property
    def effective_log_level(self) -> FlextConstants.Literals.LogLevelLiteral:
        """Get effective log level based on debug/trace flags."""
        if self.trace:
            # LogLevel.DEBUG is already compatible with LogLevelLiteral
            return FlextConstants.Settings.LogLevel.DEBUG
        if self.debug:
            # LogLevel.INFO is already compatible with LogLevelLiteral
            return FlextConstants.Settings.LogLevel.INFO
        # self.log_level is already LogLevelLiteral (from field_validator)
        return self.log_level

    @computed_field
    def is_production(self) -> bool:
        """Check if running in production environment."""
        env_value = os.getenv("ENVIRONMENT", "").lower()
        return env_value == FlextConstants.Settings.Environment.PRODUCTION.value

    @classmethod
    def get_global_instance(cls) -> Self:
        """Get the global singleton instance."""
        return cls()

    def get_di_config_provider(self) -> providers.Singleton[Self]:
        """Get dependency injection provider for this config."""
        if self._di_provider is None:
            self._di_provider = providers.Singleton(lambda: self)
        return self._di_provider

    def update_from_env(self) -> None:
        """Update configuration from current environment variables."""
        # Implementation would reload from env

    def validate_override(
        self,
        key: str,
        _value: FlextTypes.FlexibleValue,
    ) -> bool:
        """Validate if an override is acceptable."""
        # Basic validation - could be extended
        return key in self.model_fields

    def apply_override(
        self,
        key: str,
        value: FlextTypes.FlexibleValue,
    ) -> None:
        """Apply a validated configuration override."""
        if self.validate_override(key, value):
            setattr(self, key, value)

    class AutoConfig(BaseModel):
        """Auto-configuration model for dynamic config creation."""

        config_class: type[BaseSettings]
        env_prefix: str = Field(default=FlextConstants.Platform.ENV_PREFIX)
        env_file: str | None = None

        def create_config(self) -> BaseSettings:
            """Create configuration instance."""
            return self.config_class()

    # Registry for namespaced configurations
    _namespace_registry: ClassVar[dict[str, type[BaseSettings]]] = {}

    @staticmethod
    def auto_register(
        namespace: str,
    ) -> Callable[[type[_TSettings]], type[_TSettings]]:
        """Decorator for auto-registering configuration classes.

        Uses TypeVar to preserve the original class type through the decorator,
        ensuring type checkers (pyright/mypy) see the specific class type, not BaseSettings.

        Args:
            namespace: Namespace identifier for the configuration

        Returns:
            Decorator function that registers the class while preserving its type

        """

        def decorator(cls: type[_TSettings]) -> type[_TSettings]:
            """Register the configuration class while preserving type."""
            # Register in namespace registry (namespace stored in registry key, not on class)
            FlextConfig._namespace_registry[namespace] = cls
            return cls

        return decorator

    @classmethod
    def register_namespace(
        cls,
        namespace: str,
        config_class: type[BaseSettings],
    ) -> None:
        """Register a configuration class for a namespace.

        Args:
            namespace: Namespace identifier
            config_class: Configuration class to register

        """
        cls._namespace_registry[namespace] = config_class

    @classmethod
    def get_namespace_config(cls, namespace: str) -> type[BaseSettings] | None:
        """Get configuration class for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Configuration class or None if not found

        """
        return cls._namespace_registry.get(namespace)

    def get_namespace(
        self,
        namespace: str,
        config_type: type[T_Namespace],
    ) -> T_Namespace:
        """Get configuration instance for a namespace.

        Args:
            namespace: Namespace identifier
            config_type: Expected configuration type

        Returns:
            Configuration instance

        Raises:
            ValueError: If namespace not found
            TypeError: If type mismatch

        """
        config_class_raw = self.get_namespace_config(namespace)
        if config_class_raw is None:
            msg = f"Namespace '{namespace}' not registered"
            raise ValueError(msg)
        if not issubclass(config_class_raw, config_type):
            msg = f"Namespace '{namespace}' config class {config_class_raw} is not subclass of {config_type}"
            raise TypeError(msg)
        # Instantiate the config class properly - Pydantic models need regular instantiation
        # config_class is already validated as subclass of config_type, safe to instantiate
        config_class: type[T_Namespace] = config_class_raw
        return config_class()

    def __getattr__(self, name: str) -> BaseSettings:
        """Auto-resolve registered namespaces via attribute access.

        Enables `config.ldif` instead of `config.get_namespace("ldif", FlextLdifConfig)`.

        Args:
            name: Namespace name to resolve

        Returns:
            Configuration instance for the namespace

        Raises:
            AttributeError: If namespace not registered

        Example:
            config = FlextConfig.get_global_instance()
            ldif_config = config.ldif  # Auto-resolves "ldif" namespace

        """
        config_class = self._namespace_registry.get(name)
        if config_class is not None:
            return config_class()
        msg = f"'{type(self).__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset the global singleton instance for testing."""
        cls._instances.clear()

    # =========================================================================
    # LOGGING INITIALIZATION - Centralized logging setup for FLEXT ecosystem
    # =========================================================================

    _logging_initialized: ClassVar[bool] = False

    @classmethod
    def _log_level_to_numeric(cls, level: str) -> int:
        """Convert string log level to numeric value.

        Args:
            level: String log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            int: Numeric log level for structlog

        """
        level_map: dict[str, int] = {
            FlextConstants.Settings.LogLevel.DEBUG.value: logging.DEBUG,
            FlextConstants.Settings.LogLevel.INFO.value: logging.INFO,
            FlextConstants.Settings.LogLevel.WARNING.value: logging.WARNING,
            FlextConstants.Settings.LogLevel.ERROR.value: logging.ERROR,
            FlextConstants.Settings.LogLevel.CRITICAL.value: logging.CRITICAL,
        }
        return level_map.get(level.upper(), logging.INFO)

    def initialize_logging(self, *, force: bool = False) -> None:
        """Initialize logging for the FLEXT ecosystem.

        Configures structlog using effective_log_level from configuration.
        This method should be called once at application startup.

        Precedence (highest to lowest):
        1. CLI parameters (via apply_cli_logging_params)
        2. Environment variables (FLEXT_LOG_LEVEL, FLEXT_DEBUG, FLEXT_TRACE)
        3. .env file values
        4. Default values from FlextConstants

        Args:
            force: Force reinitialization even if already initialized

        Example:
            # At application startup
            config = FlextConfig.get_global_instance()
            config.initialize_logging()

            # Force reinitialize (e.g., after CLI params override)
            config.initialize_logging(force=True)

        """
        # Import here to avoid circular import (FlextRuntime imports from config)

        if FlextConfig._logging_initialized and not force:
            return

        # Get effective log level (respects debug/trace flags)
        log_level_str = self.effective_log_level
        log_level_numeric = self._log_level_to_numeric(log_level_str)

        # Configure structlog with effective settings
        if force:
            FlextRuntime.reconfigure_structlog(
                log_level=log_level_numeric,
                console_renderer=not self.json_output,
            )
        else:
            FlextRuntime.configure_structlog(
                log_level=log_level_numeric,
                console_renderer=not self.json_output,
            )

        FlextConfig._logging_initialized = True
        # Use self.logger property - StructlogLogger has debug method
        self.logger.debug(
            "Logging initialized with level=%s (numeric=%d)",
            log_level_str,
            log_level_numeric,
        )

    def apply_cli_logging_params(
        self,
        *,
        log_level: FlextConstants.Literals.LogLevelLiteral | None = None,
        debug: bool | None = None,
        trace: bool | None = None,
        _verbose: bool | None = None,
        _quiet: bool | None = None,
        json_output: bool | None = None,
    ) -> None:
        """Apply CLI logging parameters and reinitialize logging.

        CLI parameters have highest precedence and override all other
        configuration sources (env vars, .env file, defaults).

        Args:
            log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            debug: Enable debug mode (sets effective_log_level to INFO)
            trace: Enable trace mode (sets effective_log_level to DEBUG, requires debug=True)
            _verbose: Reserved for CLI-specific use (not used in logging initialization)
            _quiet: Reserved for CLI-specific use (not used in logging initialization)
            json_output: Use JSON output instead of console renderer

        Example:
            # In CLI command handler
            config = FlextConfig.get_global_instance()
            config.apply_cli_logging_params(
                log_level="DEBUG",
                debug=True,
                json_output=False,
            )

        """
        # Apply overrides if provided
        if log_level is not None:
            # Validate and normalize log level using the same validator
            self.log_level = self.validate_log_level(log_level)

        if debug is not None:
            self.debug = debug

        if trace is not None:
            # Trace requires debug
            if trace and not (debug if debug is not None else self.debug):
                self.debug = True
            self.trace = trace

        if json_output is not None:
            self.json_output = json_output

        # Store verbose/quiet for application use (not directly used in logging)
        # These are typically handled by CLI-specific config classes

        # Reinitialize logging with new settings
        self.initialize_logging(force=True)

    def get(
        self, key: str, default: FlextTypes.FlexibleValue | None = None
    ) -> FlextTypes.FlexibleValue | None:
        """Retrieve a configuration value with a default fallback."""
        return getattr(self, key, default)

    def set(self, key: str, value: FlextTypes.FlexibleValue) -> None:
        """Set a configuration value in place."""
        object.__setattr__(self, key, value)

    @classmethod
    def reset_logging(cls) -> None:
        """Reset logging initialization state for testing.

        This allows tests to reinitialize logging with different settings.
        """
        cls._logging_initialized = False


__all__ = ["FlextConfig"]
