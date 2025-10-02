"""Configuration subsystem delivering the FLEXT 1.0.0 alignment pillar.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, cast, runtime_checkable

from pydantic import (
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Backward compatibility: re-export from protocols.py
# Protocol now centralized in FlextProtocols.Foundation
if TYPE_CHECKING:
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class HasHandlerType(Protocol):
        """Protocol for config objects with handler_type attribute."""

        handler_type: str | None

else:
    from flext_core.protocols import FlextProtocols

    HasHandlerType = FlextProtocols.Foundation.HasHandlerType


class FlextConfig(BaseSettings):
    """Configuration management system for FLEXT ecosystem.

    FlextConfig provides centralized configuration management using
    Pydantic Settings with environment variable support. Global singleton
    pattern ensures consistent configuration across all 32+ dependent
    FLEXT projects. Access via FlextConfig.get_global_instance().

    **Function**: Centralized configuration with environment support
        - Global singleton configuration instance
        - Environment variable mapping with FLEXT_ prefix
        - Pydantic Settings validation and type safety
        - Configuration profiles (dev, staging, prod)
        - Secret management with SecretStr for sensitive data
        - Logging configuration (level, format, output)
        - Database configuration (URL, pool size)
        - Cache configuration (TTL, max size, enabled)
        - CQRS bus configuration (timeout, metrics)
        - Security configuration (keys, JWT, bcrypt)
        - Metadata configuration (app name, version)
        - Computed fields for derived configuration
        - Validation methods for configuration integrity

    **Uses**: Pydantic Settings for configuration management
        - BaseSettings for environment-based configuration
        - Field for default values and validation rules
        - SecretStr for sensitive data protection
        - field_validator for custom validation logic
        - model_validator for cross-field validation
        - computed_field for derived properties
        - FlextConstants for configuration defaults
        - FlextResult[T] for operation results
        - FlextTypes for type definitions
        - threading.Lock for singleton thread safety
        - json module for configuration serialization

    **How to use**: Access and configure via singleton
        ```python
        from flext_core import FlextConfig

        # Example 1: Get global configuration instance
        config = FlextConfig.get_global_instance()

        # Example 2: Access configuration values
        log_level = config.log_level
        timeout = config.timeout_seconds
        environment = config.environment

        # Example 3: Environment variable override
        # Set FLEXT_LOG_LEVEL=DEBUG in environment
        # config.log_level will be "DEBUG"

        # Example 4: Check configuration validity
        validation_result = config.validate_environment()
        if validation_result.is_success:
            print("Configuration valid")

        # Example 5: Access computed fields
        metadata = config.metadata_config
        print(f"App: {metadata['app_name']}")

        # Example 6: Configuration profiles
        if config.environment == "production":
            # Production-specific configuration
            config.debug = False
            config.trace = False

        # Example 7: Secret configuration (from environment)
        # Set FLEXT_SECRET_KEY in environment
        secret = config.secret_key  # Returns SecretStr
        ```

    Args:
        **data: Configuration values as keyword arguments.

    Attributes:
        environment (str): Runtime environment (dev/staging/prod).
        debug (bool): Debug mode flag for development.
        trace (bool): Trace mode for detailed logging.
        log_level (str): Logging level (DEBUG, INFO, WARN, ERROR).
        timeout_seconds (int): Default operation timeout.
        max_workers (int): Thread pool maximum workers.
        secret_key (SecretStr): Application secret key.
        api_key (SecretStr): API authentication key.
        database_url (str): Database connection URL.
        cache_ttl (int): Cache time-to-live in seconds.

    Returns:
        FlextConfig: Singleton configuration instance.

    Raises:
        ValidationError: When configuration validation fails.
        ValueError: When required configuration missing.

    Note:
        Global singleton pattern - use get_global_instance() not
        direct instantiation. Environment variables prefixed with
        FLEXT_ override defaults. SecretStr protects sensitive data.
        Configuration validated on load. Thread-safe singleton.

    Warning:
        Never instantiate FlextConfig directly - use singleton.
        Never commit secrets to source control. Always use
        environment variables for production secrets. Configuration
        changes require application restart (no hot-reload yet).

    Example:
        Complete configuration management workflow:

        >>> config = FlextConfig.get_global_instance()
        >>> print(config.environment)
        development
        >>> print(config.log_level)
        INFO
        >>> validation = config.validate_environment()
        >>> print(validation.is_success)
        True

    See Also:
        FlextConstants: For configuration defaults.
        FlextContainer: For dependency injection integration.
        FlextLogger: For logging configuration usage.

    """

    class HandlerConfiguration:
        """Handler configuration resolution utilities."""

        @staticmethod
        def resolve_handler_mode(
            handler_mode: str | None = None,
            handler_config: object = None,
        ) -> str:
            """Resolve handler mode from various sources.

            Args:
                handler_mode: Explicit handler mode
                handler_config: Config object or dict containing handler_type

            Returns:
                str: Resolved handler mode (command or query)

            """
            # Use explicit handler_mode if provided and valid
            if handler_mode in {"command", "query"}:
                return handler_mode

            # Try to extract from config object
            if handler_config is not None:
                # Try attribute access
                if isinstance(handler_config, HasHandlerType):
                    config_mode: str | None = handler_config.handler_type
                    if config_mode in {"command", "query"}:
                        return str(config_mode)

                # Try dict access
                if isinstance(handler_config, dict):
                    config_mode_dict = handler_config.get("handler_type")
                    if isinstance(config_mode_dict, str) and config_mode_dict in {
                        "command",
                        "query",
                    }:
                        return config_mode_dict

            # Default to command
            return FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

        @staticmethod
        def create_handler_config(
            handler_mode: str | None = None,
            handler_name: str | None = None,
            handler_id: str | None = None,
            handler_config: FlextTypes.Core.Dict | None = None,
            command_timeout: int = 0,
            max_command_retries: int = 0,
        ) -> FlextTypes.Core.Dict:
            """Create handler configuration dictionary.

            Args:
                handler_mode: Handler mode (command or query)
                handler_name: Handler name
                handler_id: Handler ID
                handler_config: Additional configuration to merge
                command_timeout: Command timeout in milliseconds
                max_command_retries: Maximum command retries

            Returns:
                dict: Handler configuration dictionary

            """
            # Resolve handler mode
            resolved_mode = FlextConfig.HandlerConfiguration.resolve_handler_mode(
                handler_mode=handler_mode,
                handler_config=handler_config,
            )

            # Generate default handler_id if not provided or empty
            if not handler_id:
                unique_suffix = uuid.uuid4().hex[:8]
                handler_id = f"{resolved_mode}_handler_{unique_suffix}"

            # Generate default handler_name if not provided or empty
            if not handler_name:
                handler_name = f"{resolved_mode.capitalize()} Handler"

            # Create base config
            config: FlextTypes.Core.Dict = {
                "handler_id": handler_id,
                "handler_name": handler_name,
                "handler_type": resolved_mode,
                "handler_mode": resolved_mode,
                "command_timeout": command_timeout,
                "max_command_retries": max_command_retries,
                "metadata": {},
            }

            # Merge additional config if provided
            if handler_config:
                config.update(handler_config)

            return config

    # Singleton pattern implementation - per-class instances to support inheritance
    _instances: ClassVar[dict[type, FlextConfig]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix=FlextConstants.Platform.ENV_PREFIX,
        env_file=FlextConstants.Platform.ENV_FILE_DEFAULT,
        env_file_encoding=FlextConstants.Mixins.DEFAULT_ENCODING,
        env_nested_delimiter=FlextConstants.Platform.ENV_NESTED_DELIMITER,
        extra="ignore",  # Changed from "forbid" to "ignore" to allow extra env vars
        use_enum_values=True,
        frozen=False,  # Allow runtime configuration updates
        # Pydantic 2.11+ enhanced features
        arbitrary_types_allowed=True,
        validate_return=True,
        validate_assignment=True,  # Validate on assignment
        # Enhanced settings features
        cli_parse_args=False,  # Disable CLI parsing by default
        cli_avoid_json=True,  # Avoid JSON CLI options for complex types
        enable_decoding=True,  # Enable JSON decoding for environment variables
        nested_model_default_partial_update=True,  # Allow partial updates to nested models
        # Advanced Pydantic 2.11+ features
        str_strip_whitespace=True,  # Strip whitespace from strings
        str_to_lower=False,  # Keep original case
        json_schema_extra={
            "title": "FLEXT Configuration",
            "description": "Enterprise FLEXT ecosystem configuration with singleton pattern support",
        },
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

    # Security configuration using SecretStr for sensitive data
    secret_key: SecretStr | None = Field(
        default=None,
        description="Secret key for security operations (sensitive)",
    )

    api_key: SecretStr | None = Field(
        default=None,
        description="API key for external service authentication (sensitive)",
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
        default=FlextConstants.Security.SHORT_JWT_EXPIRY_MINUTES,
        description="JWT token expiry time in minutes",
    )

    bcrypt_rounds: int = Field(
        default=FlextConstants.Security.DEFAULT_BCRYPT_ROUNDS,
        description="BCrypt hashing rounds",
    )

    jwt_secret: str = Field(
        default=FlextConstants.Security.DEFAULT_JWT_SECRET,
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
        default=FlextConstants.Validation.MAX_NAME_LENGTH,
        ge=1,
        le=500,
        description="Maximum allowed name length for validation",
    )

    min_phone_digits: int = Field(
        default=FlextConstants.Validation.MIN_PHONE_DIGITS,
        ge=7,
        le=15,
        description="Minimum phone number digit count for validation",
    )

    # Field validators
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate that environment is one of the allowed values."""
        valid_environments = {
            "development",
            "dev",
            "local",
            "staging",
            "test",
            "production",
        }
        if v.lower() not in valid_environments:
            msg = f"Invalid environment: {v}. Must be one of: {', '.join(sorted(valid_environments))}"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log_level is one of the allowed values (case-insensitive)."""
        v_upper = v.upper()
        if v_upper not in FlextConstants.Logging.VALID_LEVELS:
            msg = f"Invalid log level: {v}. Must be one of: {', '.join(FlextConstants.Logging.VALID_LEVELS)}"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return v_upper

    @model_validator(mode="after")
    def validate_debug_trace_consistency(self) -> FlextConfig:
        """Validate debug and trace mode consistency."""
        # Production cannot have debug=True
        if self.environment.lower() == "production" and self.debug:
            msg = "Debug mode cannot be enabled in production environment"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Trace requires debug
        if self.trace and not self.debug:
            msg = "Trace mode requires debug mode to be enabled"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return self

    # Singleton pattern methods
    @classmethod
    def get_global_instance(cls) -> Self:
        """Get the global singleton instance per class (supports inheritance).

        Each subclass gets its own singleton instance, preventing conflicts
        in the FLEXT ecosystem where multiple config classes may coexist.

        Returns:
            Singleton instance of the specific config class.

        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = cls()
        return cast("Self", cls._instances[cls])

    @classmethod
    def set_global_instance(cls, instance: FlextConfig) -> None:
        """Set the global singleton instance per class."""
        with cls._lock:
            cls._instances[cls] = instance

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset the global instance for this specific class (mainly for testing)."""
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]

    @classmethod
    def get_or_create_shared_instance(
        cls,
        project_name: str | None = None,
        **overrides: FlextTypes.Core.Value,
    ) -> FlextConfig:
        """Get or create a shared singleton instance with project-specific overrides.

        This method supports inverse dependency injection where multiple projects
        can share the same FlextConfig singleton instance while allowing project-specific
        configuration overrides to be applied.

        Args:
            project_name: Optional project name for logging and identification
            **overrides: Project-specific configuration overrides

        Returns:
            FlextConfig: The shared singleton instance with any overrides applied

        """
        instance = cls.get_global_instance()

        # If overrides are provided, create a merged instance but keep using the singleton
        if overrides:
            # Apply overrides without breaking the singleton pattern
            for key, value in overrides.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

        # Log project access for debugging if project_name provided
        if project_name:
            logger = logging.getLogger(__name__)
            logger.debug(
                "Project '%s' accessing shared FlextConfig instance",
                project_name,
            )

        return instance

    @classmethod
    def create_project_config(
        cls,
        project_name: str,
        **project_defaults: FlextTypes.Core.Value,
    ) -> FlextConfig:
        """Create a project-specific configuration that inherits from the global singleton.

        This factory method is designed for projects that need their own Config class
        but want to maintain compatibility with the shared FlextConfig singleton.

        Args:
            project_name: Name of the project (e.g., 'flext-api', 'flext-auth')
            **project_defaults: Project-specific default values

        Returns:
            FlextConfig: Project configuration instance based on global singleton

        """
        # Get the base configuration from singleton
        base_config = cls.get_global_instance()

        # Create project-specific overrides
        project_overrides: FlextTypes.Core.Dict = {
            "app_name": f"{project_name} Application",
            **project_defaults,
        }

        # Use the merge method to create a new instance with project specifics
        return base_config.merge(project_overrides)

    # Class methods for creating instances
    @classmethod
    def create(cls, **kwargs: FlextTypes.Core.ConfigValue) -> FlextConfig:
        """Create a new FlextConfig instance with the given parameters.

        Args:
            **kwargs: Configuration parameters. Pydantic BaseSettings handles
                flexible kwargs with proper validation and type conversion.

        """
        # Pydantic BaseSettings handles kwargs validation and type conversion automatically
        return cls.model_validate(kwargs)

    @classmethod
    def create_for_environment(cls, environment: str, **kwargs: object) -> FlextConfig:
        """Create a FlextConfig instance for a specific environment.

        Args:
            environment: The environment name (development, production, etc.)
            **kwargs: Additional configuration parameters. Pydantic BaseSettings
                handles flexible kwargs with proper validation and type conversion.

        """
        # Pydantic BaseSettings handles kwargs validation and type conversion automatically
        return cls(environment=environment, **kwargs)

    @classmethod
    def from_file(cls, file_path: str | Path) -> FlextResult[FlextConfig]:
        """Load configuration from a file.

        Supports JSON format files.

        Args:
            file_path: Path to the configuration file

        Returns:
            FlextResult containing the loaded configuration or error

        """
        try:
            path = Path(file_path)
            if not path.exists():
                return FlextResult[FlextConfig].fail(
                    f"Failed to load config: Configuration file not found: {file_path}",
                )

            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                config = cls(**data)
                return FlextResult[FlextConfig].ok(config)
            return FlextResult[FlextConfig].fail(
                f"Unsupported file format: {path.suffix}",
            )

        except json.JSONDecodeError as e:
            return FlextResult[FlextConfig].fail(
                f"Failed to parse config: Invalid JSON in configuration file: {e}",
            )
        except Exception as e:
            return FlextResult[FlextConfig].fail(
                f"Failed to load config: Error loading configuration: {e}",
            )

    # Instance methods for configuration access
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in {"development", "dev", "local"}

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def get_logging_config(self) -> FlextTypes.Core.Dict:
        """Get logging configuration as dictionary."""
        return {
            "level": self.log_level,
            "json_output": self.json_output,
            "include_source": self.include_source,
            "structured_output": self.structured_output,
            "log_verbosity": self.log_verbosity,
            "log_format": self.log_format,
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

    def get_database_config(self) -> FlextTypes.Core.Dict:
        """Get database configuration as dictionary."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
        }

    def get_cache_config(self) -> FlextTypes.Core.Dict:
        """Get cache configuration as dictionary."""
        return {
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
            "enabled": self.enable_caching,
        }

    def get_cqrs_bus_config(self) -> FlextTypes.Core.Dict:
        """Get CQRS bus configuration as dictionary."""
        return {
            "auto_context": self.dispatcher_auto_context,
            "timeout_seconds": self.dispatcher_timeout_seconds,
            "enable_metrics": self.dispatcher_enable_metrics,
            "enable_logging": self.dispatcher_enable_logging,
        }

    def get_metadata(self) -> FlextTypes.Core.Dict:
        """Get application metadata as dictionary."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "debug": self.debug,
            "trace": self.trace,
        }

    def merge(self, overrides: FlextTypes.Core.Dict | FlextConfig) -> FlextConfig:
        """Create a new FlextConfig instance with merged values.

        Args:
            overrides: Dictionary of values to override or another FlextConfig instance

        Returns:
            New FlextConfig instance with merged values

        """
        current_data = self.model_dump()

        if isinstance(overrides, FlextConfig):
            # If it's another FlextConfig, get its data as dict
            override_data = overrides.model_dump()
        else:
            # If it's a dict, use it directly
            override_data = overrides

        current_data.update(override_data)
        # Use model_validate to ensure proper type checking
        return FlextConfig.model_validate(current_data)

    # Computed fields for derived configuration

    @computed_field
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled (debug or trace)."""
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
    def cache_config(self) -> FlextTypes.Core.Dict:
        """Get cache configuration as dictionary."""
        return {
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
            "enabled": self.cache_ttl > 0,
        }

    @computed_field
    def security_config(self) -> FlextTypes.Core.Dict:
        """Get security configuration as dictionary."""
        return {
            "secret_key_configured": self.secret_key is not None,
            "api_key_configured": self.api_key is not None,
            "jwt_expiry_minutes": self.jwt_expiry_minutes,
            "bcrypt_rounds": self.bcrypt_rounds,
        }

    # SecretStr accessor methods for sensitive configuration
    def get_secret_key_value(self) -> str | None:
        """Get the actual secret key value (safely extract from SecretStr)."""
        if self.secret_key is not None:
            return self.secret_key.get_secret_value()
        return None

    def get_api_key_value(self) -> str | None:
        """Get the actual API key value (safely extract from SecretStr)."""
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return None


# This resolves any forward reference issues that may occur during model construction
FlextConfig.model_rebuild()

# Direct class access - no legacy aliases

__all__ = [
    "FlextConfig",
]
