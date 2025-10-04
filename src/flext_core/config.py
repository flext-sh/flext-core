"""Configuration subsystem delivering the FLEXT 1.0.0 alignment pillar.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import ClassVar, Self, cast

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
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextConfig(
    BaseSettings,
):
    """Configuration management system for FLEXT ecosystem - OPTIMIZATION SHOWCASE.

    FLEXT-CORE OPTIMIZATION PATTERNS DEMONSTRATED:

    🚀 NAMESPACE CLASS PATTERN
    Single unified class with nested helper classes and comprehensive functionality:
    - Nested HandlerConfiguration class for CQRS handler config resolution
    - All configuration logic centralized in one class
    - No loose helper functions - everything properly organized

    🔧 FLEXT-CORE INTEGRATION
    Complete integration with flext-core ecosystem:
    - FlextConstants for all default values and validation limits
    - FlextResult for all operation results (railway pattern)
    - FlextExceptions for structured error handling
    - FlextProtocols for interface compliance
    - FlextTypes for type definitions

    ⚙️ PYDANTIC 2.11+ BASESETTINGS
    Modern configuration management:
    - Environment variable support with FLEXT_ prefix
    - Field validators for custom validation logic
    - Model validators for cross-field validation
    - Computed fields for derived configuration values
    - SecretStr for sensitive data protection
    - SettingsConfigDict for advanced configuration options

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

    **Uses**: Core infrastructure patterns
        - FlextConstants for configuration defaults and limits
        - FlextResult[T] for all operation results (railway pattern)
        - FlextExceptions for structured error handling
        - FlextTypes for type definitions
        - threading.Lock for singleton thread safety
        - json module for configuration serialization

    OPTIMIZATION EXAMPLES:

    ```python
    # ✅ CORRECT - Complete flext-core integration with optimization patterns
    from flext_core import FlextConfig, FlextConstants, FlextResult

    # Example 1: Enhanced configuration with computed fields
    config = FlextConfig()

    # Use computed fields for derived configuration
    print(f"Debug enabled: {config.is_debug_enabled}")  # From debug or trace
    print(f"Effective log level: {config.effective_log_level}")  # Considers debug/trace

    # Access component-specific configuration
    container_config = config.get_component_config("container")
    if container_config.is_success:
        settings = container_config.unwrap()
        print(f"Max workers: {settings['max_workers']}")


    # Example 2: Railway pattern for configuration operations
    def validate_and_create_config() -> FlextResult[FlextConfig]:
        config = FlextConfig()

        # Validate integration with railway pattern
        validation = config.validate_flext_core_integration()
        if validation.is_failure:
            return FlextResult[FlextConfig].fail(
                f"Integration failed: {validation.error}"
            )

        return FlextResult[FlextConfig].ok(config)


    # Example 3: Service configuration with flext-core integration
    service_config = config.create_service_config(
        "user_service", timeout_seconds=60, enable_caching=True
    )
    if service_config.is_success:
        settings = service_config.unwrap()
        print(f"Service timeout: {settings['timeout_seconds']}")

    # Example 4: Integration validation
    setup_result = config.validate_flext_core_integration()
    if setup_result.is_success:
        print("✅ All flext-core components properly integrated")
    else:
        print(f"❌ Integration issues: {setup_result.error}")

    # Example 5: Practical integration example
    example = config.get_integration_example("service")
    if example.is_success:
        print(f"Service integration pattern: {example.unwrap()[:100]}...")

    # Example 6: Nested helper class usage (optimization pattern)
    handler_mode = FlextConfig.HandlerConfiguration.resolve_handler_mode(
        handler_mode="command", handler_config={"handler_type": "query"}
    )
    print(f"Resolved handler mode: {handler_mode}")
    ```

    BEFORE vs AFTER OPTIMIZATION:

    ```python
    # ❌ BEFORE - Scattered configuration, no integration
    class OldConfig:
        def __init__(self):
            self.log_level = "INFO"
            self.timeout = 30

        def get_logging_config(self):
            return {"level": self.log_level}


    # ✅ AFTER - Unified configuration with flext-core integration
    class FlextConfig(BaseSettings):  # Single class pattern
        log_level: str = Field(default="INFO")  # Pydantic field

        @computed_field  # Derived configuration
        def logging_config(self) -> dict:
            return {
                "level": self.effective_log_level,  # Considers debug/trace
                "format": self.log_format,
                "structured": self.structured_output,
            }

        def get_component_config(self, component: str) -> FlextResult[FlextTypes.Dict]:
            # Integration with flext-core components
            return FlextResult[FlextTypes.Dict].ok({
                "component": component,
                "config": "enhanced",
            })
    ```

    **Args**:
        **data: Configuration values as keyword arguments.

    **Attributes**:
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

    **Returns**:
        FlextConfig: Optimized configuration instance with full flext-core integration.

    **Raises**:
        ValidationError: When configuration validation fails.
        ValueError: When required configuration missing.

    **Note**:
        Direct instantiation pattern - create with FlextConfig().
        Environment variables prefixed with FLEXT_ override defaults.
        SecretStr protects sensitive data. Configuration validated on load.
        All operations use FlextResult for consistency.

    **Warning**:
        Never commit secrets to source control.
        Configuration changes require application restart.
        Always validate integration before production deployment.

    **Example**:
        Complete optimization showcase:

        >>> config = FlextConfig()
        >>> print(f"Environment: {config.environment}")
        development
        >>> print(f"Debug enabled: {config.is_debug_enabled}")
        False
        >>> print(f"Effective log level: {config.effective_log_level}")
        INFO
        >>> integration = config.validate_flext_core_integration()
        >>> print(f"Integration valid: {integration.is_success}")
        True

    **See Also**:
        FlextConstants: For configuration defaults and validation limits.
        FlextContainer: For dependency injection integration.
        FlextLogger: For logging configuration usage.
        FlextUtilities: For configuration validation utilities.
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
                if isinstance(handler_config, FlextProtocols.Foundation.HasHandlerType):
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
            handler_config: FlextTypes.Dict | None = None,
            command_timeout: int = 0,
            max_command_retries: int = 0,
        ) -> FlextTypes.Dict:
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
            config: FlextTypes.Dict = {
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
        default=f"{FlextConstants.Core.NAME} Application",
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
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Circuit breaker failure threshold before opening",
    )

    # Rate limiting configuration
    rate_limit_max_requests: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum requests allowed in rate limit window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit time window in seconds",
    )

    # Batch processing configuration
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Default batch size for batch operations",
    )
    cache_size: int = Field(
        default=1000,
        ge=1,
        le=1000000,
        description="Default cache size for caching operations",
    )

    # Retry configuration
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=300.0,
        description="Default delay between retry attempts in seconds",
    )

    # Validation configuration
    validation_timeout_ms: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum validation time in milliseconds",
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

    def __call__(self, key: str) -> object:
        """Direct value access: config('log_level') with Pydantic 2 enhancements.

        Enables simplified configuration access by field name with advanced
        Pydantic 2 Settings features including nested field access and
        type-safe value retrieval.

        Pydantic 2 Settings features:
        - Direct field access with dot notation support
        - Nested configuration access (e.g., 'cache_config.ttl')
        - Computed field support
        - Type-safe value extraction with proper validation

        Args:
            key: Configuration field name (e.g., 'log_level', 'timeout_seconds')
                 Supports dot notation for nested access (e.g., 'cache_config.ttl')

        Returns:
            The configuration value for the specified field

        Raises:
            KeyError: If the configuration key doesn't exist

        Example:
            >>> config = FlextConfig()
            >>> config("log_level")
            'INFO'
            >>> config("timeout_seconds")
            30
            >>> # Nested access with dot notation
            >>> config("cache_config.ttl")
            300
            >>> # Computed field access
            >>> config("is_debug_enabled")
            False

        """
        # Support nested field access with dot notation
        if "." in key:
            parts = key.split(".", 1)
            first_key = parts[0]
            remaining_key = parts[1]

            if not hasattr(self, first_key):
                msg = f"Configuration key '{first_key}' not found"
                raise KeyError(msg)

            # Get the nested object
            nested_obj = getattr(self, first_key)

            # Handle dict access
            if isinstance(nested_obj, dict):
                if remaining_key not in nested_obj:
                    msg = f"Configuration key '{key}' not found in nested config"
                    raise KeyError(msg)
                return nested_obj[remaining_key]

            # Handle object attribute access
            if hasattr(nested_obj, remaining_key):
                return getattr(nested_obj, remaining_key)

            msg = f"Configuration key '{remaining_key}' not found in '{first_key}'"
            raise KeyError(msg)

        # Direct field access
        if not hasattr(self, key):
            msg = f"Configuration key '{key}' not found"
            raise KeyError(msg)

        return getattr(self, key)

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

    @model_validator(mode="after")
    def synchronize_models_config(self) -> FlextConfig:
        """Synchronize models module configuration when FlextConfig is created/updated.

        This ensures that FlextModels classes use the current FlextConfig instance
        as their configuration source, enabling the newer pattern where FlextConfig
        serves as the central source of configuration for all model classes.
        """
        # Update models config to use this instance as the source
        FlextConfig._update_models_config(self)
        return self

    # Singleton pattern implementation
    _global_instance: ClassVar[FlextConfig | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> Self:
        """Implement singleton pattern for FlextConfig instances."""
        if cls._global_instance is None:
            with cls._instance_lock:
                if cls._global_instance is None:
                    cls._global_instance = super().__new__(cls)
                    # Update models module config when global instance is created
                    cls._update_models_config(cls._global_instance)
        return cls._global_instance

    @classmethod
    def get_global_instance(cls) -> Self:
        """Get or create the global singleton configuration instance.

        Returns:
            FlextConfig: The global configuration instance.

        Example:
            >>> config = FlextConfig.get_global_instance()
            >>> config2 = FlextConfig.get_global_instance()
            >>> assert config is config2  # Same instance

        """
        return cls()

    @classmethod
    def set_global_instance(cls, instance: FlextConfig) -> None:
        """Set the global singleton instance per class."""
        with cls._instance_lock:
            cls._global_instance = instance
            # Update models module config when FlextConfig changes
            cls._update_models_config(instance)

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset the global instance for this specific class (mainly for testing)."""
        with cls._instance_lock:
            cls._global_instance = None
            # Reset models config to default when global instance is reset
            cls._reset_models_config()

    @classmethod
    def _update_models_config(cls, config_instance: FlextConfig) -> None:
        """Update the models module configuration to use the current FlextConfig instance.

        This ensures that FlextModels classes use the correct configuration source.
        The models module maintains a module-level _config instance that gets updated
        when the global FlextConfig changes.
        """
        try:
            # Import models module and update its config
            # NOTE: This access to private _config is necessary for the current
            # architecture where models use global config for field defaults.
            # TODO(#1): Refactor to use proper dependency injection pattern.  # noqa: FIX002
            import flext_core.models as models_module

            models_module._config = config_instance  # noqa: SLF001
        except ImportError:
            # Models module not yet imported, will use default when imported
            pass

    @classmethod
    def _reset_models_config(cls) -> None:
        """Reset the models module configuration to default FlextConfig instance."""
        try:
            # Import models module and update its config
            # NOTE: This access to private _config is necessary for the current
            # architecture where models use global config for field defaults.
            # TODO(#1): Refactor to use proper dependency injection pattern.  # noqa: FIX002
            import flext_core.models as models_module

            models_module._config = FlextConfig()  # noqa: SLF001
        except ImportError:
            # Models module not yet imported
            pass

    @classmethod
    def get_or_create_shared_instance(
        cls,
        project_name: str | None = None,
        **overrides: FlextTypes.Value,
    ) -> FlextConfig:
        """REMOVED: Create config directly and apply overrides.

        Migration:
            # Old pattern
            config = FlextConfig.get_or_create_shared_instance(
                project_name="my-project",
                debug=True
            )

            # New pattern - create and configure
            config = FlextConfig()
            config.debug = True

            # For logging project access
            import logging
            logger = logging.getLogger(__name__)
            logger.debug("Project 'my-project' using FlextConfig instance")

        """
        msg = (
            "FlextConfig.get_or_create_shared_instance() has been removed. "
            "Create FlextConfig() directly and set attributes as needed."
        )
        raise NotImplementedError(msg)

    # Class methods for creating instances
    @classmethod
    def create(cls, **kwargs: FlextTypes.ConfigValue) -> FlextConfig:
        """Create a new FlextConfig instance with the given parameters.

        Args:
            **kwargs: Configuration parameters. Pydantic BaseSettings handles
                flexible kwargs with proper validation and type conversion.

        """
        # Pydantic BaseSettings handles kwargs validation and type conversion automatically
        return cls.model_validate(kwargs)

    @classmethod
    def create_for_environment(
        cls, environment: str, **kwargs: FlextTypes.ConfigValue
    ) -> FlextConfig:
        """Create a FlextConfig instance for a specific environment.

        Args:
            environment: The environment name (development, production, etc.)
            **kwargs: Additional configuration parameters. Pydantic BaseSettings
                handles flexible kwargs with proper validation and type conversion.

        """
        # Pydantic BaseSettings handles kwargs validation and type conversion automatically
        # Include environment in the validation data
        config_data = {"environment": environment, **kwargs}
        return cls.model_validate(config_data)

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

    def get_logging_config(self) -> FlextTypes.Dict:
        """REMOVED: Access config attributes directly.

        Migration:
            # Old pattern
            logging_config = config.get_logging_config()

            # New pattern - direct attribute access
            logging_config = {
                "level": config.log_level,
                "json_output": config.json_output,
                "include_source": config.include_source,
                # ... other logging attributes as needed
            }

        """
        msg = (
            "FlextConfig.get_logging_config() has been removed. "
            "Access logging configuration attributes directly."
        )
        raise NotImplementedError(msg)

    def get_database_config(self) -> FlextTypes.Dict:
        """REMOVED: Access config attributes directly.

        Migration:
            # Old pattern
            db_config = config.get_database_config()

            # New pattern - direct attribute access
            db_config = {
                "url": config.database_url,
                "pool_size": config.database_pool_size,
            }

        """
        msg = (
            "FlextConfig.get_database_config() has been removed. "
            "Access database configuration attributes directly."
        )
        raise NotImplementedError(msg)

    def get_cache_config(self) -> FlextTypes.Dict:
        """REMOVED: Access config attributes directly.

        Migration:
            # Old pattern
            cache_config = config.get_cache_config()

            # New pattern - direct attribute access
            cache_config = {
                "ttl": config.cache_ttl,
                "max_size": config.cache_max_size,
                "enabled": config.enable_caching,
            }

        """
        msg = (
            "FlextConfig.get_cache_config() has been removed. "
            "Access cache configuration attributes directly."
        )
        raise NotImplementedError(msg)

    def get_cqrs_bus_config(self) -> FlextTypes.Dict:
        """REMOVED: Access config attributes directly.

        Migration:
            # Old pattern
            bus_config = config.get_cqrs_bus_config()

            # New pattern - direct attribute access
            bus_config = {
                "auto_context": config.dispatcher_auto_context,
                "timeout_seconds": config.dispatcher_timeout_seconds,
                "enable_metrics": config.dispatcher_enable_metrics,
                "enable_logging": config.dispatcher_enable_logging,
            }

        """
        msg = (
            "FlextConfig.get_cqrs_bus_config() has been removed. "
            "Access CQRS bus configuration attributes directly."
        )
        raise NotImplementedError(msg)

    def get_metadata(self) -> FlextTypes.Dict:
        """REMOVED: Access config attributes directly.

        Migration:
            # Old pattern
            metadata = config.get_metadata()

            # New pattern - direct attribute access
            metadata = {
                "app_name": config.app_name,
                "version": config.version,
                "environment": config.environment,
                "debug": config.debug,
                "trace": config.trace,
            }

        """
        msg = (
            "FlextConfig.get_metadata() has been removed. "
            "Access metadata attributes directly."
        )
        raise NotImplementedError(msg)

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
    def cache_config(self) -> FlextTypes.Dict:
        """Get cache configuration as dictionary."""
        return {
            "ttl": self.cache_ttl,
            "max_size": self.cache_max_size,
            "enabled": self.cache_ttl > 0,
        }

    @computed_field
    def security_config(self) -> FlextTypes.Dict:
        """Get security configuration as dictionary."""
        return {
            "secret_key_configured": self.secret_key is not None,
            "api_key_configured": self.api_key is not None,
            "jwt_expiry_minutes": self.jwt_expiry_minutes,
            "bcrypt_rounds": self.bcrypt_rounds,
        }

    @computed_field
    def database_config(self) -> FlextTypes.Dict:
        """Get database configuration as dictionary with enhanced flext-core integration."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "connection_config": {
                "min_size": FlextConstants.Performance.MIN_DB_POOL_SIZE,
                "max_size": self.database_pool_size,
                "timeout_seconds": self.timeout_seconds,
                "retry_attempts": self.max_retry_attempts,
            },
            "health_check_enabled": self.enable_metrics,
        }

    @computed_field
    def dispatcher_config(self) -> FlextTypes.Dict:
        """Get dispatcher configuration with flext-core integration patterns."""
        return {
            "auto_context": self.dispatcher_auto_context,
            "timeout_seconds": self.dispatcher_timeout_seconds,
            "enable_metrics": self.dispatcher_enable_metrics,
            "enable_logging": self.dispatcher_enable_logging,
            "performance_config": {
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
                "enable_circuit_breaker": self.enable_circuit_breaker,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
            },
        }

    @computed_field
    def logging_config(self) -> FlextTypes.Dict:
        """Get comprehensive logging configuration with flext-core patterns."""
        return {
            "level": self.effective_log_level,
            "format": self.log_format,
            "output": {
                "console": self.console_enabled,
                "console_color": self.console_color_enabled,
                "file": self.log_file,
                "json": self.json_output,
            },
            "features": {
                "include_source": self.include_source,
                "structured": self.structured_output,
                "performance_tracking": self.track_performance,
                "timing_tracking": self.track_timing,
                "context_inclusion": self.include_context,
                "correlation_id": self.include_correlation_id,
                "sensitive_data_masking": self.mask_sensitive_data,
            },
            "verbosity": self.log_verbosity,
        }

    @computed_field
    def performance_config(self) -> FlextTypes.Dict:
        """Get performance configuration with flext-core optimization patterns."""
        return {
            "caching": {
                "enabled": self.enable_caching,
                "ttl": self.cache_ttl,
                "max_size": self.cache_max_size,
            },
            "processing": {
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
                "timeout_seconds": self.timeout_seconds,
            },
            "reliability": {
                "retry_attempts": self.max_retry_attempts,
                "retry_delay_seconds": self.retry_delay_seconds,
                "circuit_breaker_enabled": self.enable_circuit_breaker,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
            },
            "rate_limiting": {
                "max_requests": self.rate_limit_max_requests,
                "window_seconds": self.rate_limit_window_seconds,
            },
        }

    @computed_field
    def metadata_config(self) -> FlextTypes.Dict:
        """Get application metadata with flext-core integration."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "debug_mode": self.debug,
            "trace_mode": self.trace,
            "effective_log_level": self.effective_log_level,
            "is_debug_enabled": self.is_debug_enabled,
            "is_production": self.is_production(),
            "is_development": self.is_development(),
        }

    @computed_field
    def validation_config(self) -> FlextTypes.Dict:
        """Get validation configuration with flext-core patterns."""
        return {
            "strict_mode": self.validation_strict_mode,
            "timeout_ms": self.validation_timeout_ms,
            "name_length_limits": {
                "max_length": self.max_name_length,
                "min_length": 1,
            },
            "phone_validation": {
                "min_digits": self.min_phone_digits,
                "max_digits": 15,
            },
            "serialization": {
                "encoding": self.serialization_encoding,
                "use_utc": self.use_utc_timestamps,
                "auto_update": self.timestamp_auto_update,
            },
        }

    # =========================================================================
    # Infrastructure Protocol Implementations
    # =========================================================================

    # Infrastructure.Configurable protocol methods
    def configure(self, config: FlextTypes.Dict) -> FlextResult[None]:
        """Configure component with provided settings.

        Implements Infrastructure.Configurable protocol.

        Args:
            config: Configuration dictionary to apply

        Returns:
            FlextResult[None]: Success if configuration valid, failure otherwise

        """
        try:
            # Update current instance with provided config
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # Validate after configuration
            return self.validate_runtime_requirements()
        except Exception as e:
            return FlextResult[None].fail(f"Configuration failed: {e}")

    # Infrastructure.ConfigValidator protocol methods
    def validate_runtime_requirements(self) -> FlextResult[None]:
        """Validate configuration meets runtime requirements.

        Implements Infrastructure.ConfigValidator protocol.

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        # Validate environment
        env_validation = self.validate_environment(self.environment)
        if not env_validation:
            return FlextResult[None].fail(f"Invalid environment: {self.environment}")

        # Validate log level
        try:
            self.validate_log_level(self.log_level)
        except FlextExceptions.ValidationError as e:
            return FlextResult[None].fail(str(e))

        # Debug/trace consistency is validated by model_validator automatically

        return FlextResult[None].ok(None)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for configuration consistency.

        Implements Infrastructure.ConfigValidator protocol.

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        # Production environment checks
        if self.is_production():
            if self.debug:
                return FlextResult[None].fail(
                    "Debug mode cannot be enabled in production"
                )
            if self.trace:
                return FlextResult[None].fail(
                    "Trace mode cannot be enabled in production"
                )

        # Security configuration checks
        if self.is_production() and not self.secret_key:
            return FlextResult[None].fail(
                "Secret key required in production environment"
            )

        return FlextResult[None].ok(None)

    # Infrastructure.ConfigPersistence protocol methods
    def save_to_file(
        self,
        file_path: str | Path,
        **kwargs: FlextTypes.ConfigValue,
    ) -> FlextResult[None]:
        """Save configuration to file.

        Implements Infrastructure.ConfigPersistence protocol.

        Args:
            file_path: Path to save configuration file
            **kwargs: Additional save options (e.g., indent, sort_keys)

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            path = Path(file_path)

            # Get configuration data
            config_data = self.model_dump()

            # Handle SecretStr fields - don't serialize actual values
            if config_data.get("secret_key"):
                config_data["secret_key"] = FlextConstants.Messages.REDACTED_SECRET
            if config_data.get("api_key"):
                config_data["api_key"] = FlextConstants.Messages.REDACTED_SECRET

            # Determine format from extension
            if path.suffix.lower() == ".json":
                indent_value = kwargs.get("indent", self.json_indent)
                indent = (
                    int(indent_value)
                    if indent_value is not None and isinstance(indent_value, (int, str))
                    else self.json_indent
                )
                sort_keys_value = kwargs.get("sort_keys", self.json_sort_keys)
                sort_keys = (
                    bool(sort_keys_value)
                    if sort_keys_value is not None
                    else self.json_sort_keys
                )

                with path.open("w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=indent, sort_keys=sort_keys)

                return FlextResult[None].ok(None)

            return FlextResult[None].fail(f"Unsupported file format: {path.suffix}")

        except Exception as e:
            return FlextResult[None].fail(f"Save failed: {e}")

    # =========================================================================
    # Enhanced flext-core Integration Methods
    # =========================================================================

    def get_component_config(self, component: str) -> FlextResult[FlextTypes.Dict]:
        """Get configuration for specific flext-core component with integration patterns.

        Demonstrates advanced flext-core integration by providing component-specific
        configuration that integrates with FlextContainer, FlextBus, FlextDispatcher, etc.

        Args:
            component: Component name ('container', 'bus', 'dispatcher', 'logger', 'context')

        Returns:
            FlextResult containing component configuration or error

        Example:
            >>> config = FlextConfig()
            >>> container_config = config.get_component_config("container")
            >>> if container_config.is_success:
            ...     print(
            ...         f"Container max workers: {container_config.unwrap()['max_workers']}"
            ...     )

        """
        component_configs = {
            "container": {
                "max_workers": self.max_workers,
                "enable_circuit_breaker": self.enable_circuit_breaker,
                "performance_monitoring": self.enable_metrics,
                "health_check_interval": self.timeout_seconds // 4,
                "registry": {
                    "auto_register": True,
                    "enable_metrics": self.dispatcher_enable_metrics,
                    "enable_logging": self.dispatcher_enable_logging,
                },
            },
            "bus": {
                "auto_context": self.dispatcher_auto_context,
                "timeout_seconds": self.dispatcher_timeout_seconds,
                "enable_metrics": self.dispatcher_enable_metrics,
                "enable_logging": self.dispatcher_enable_logging,
                "performance_config": {
                    "batch_size": self.batch_size,
                    "max_workers": self.max_workers,
                },
            },
            "dispatcher": {
                "auto_context": self.dispatcher_auto_context,
                "timeout_seconds": self.dispatcher_timeout_seconds,
                "enable_metrics": self.dispatcher_enable_metrics,
                "enable_logging": self.dispatcher_enable_logging,
                "circuit_breaker": {
                    "enabled": self.enable_circuit_breaker,
                    "threshold": self.circuit_breaker_threshold,
                },
            },
            "logger": {
                "level": self.effective_log_level,
                "structured": self.structured_output,
                "include_context": self.include_context,
                "include_correlation_id": self.include_correlation_id,
                "performance_tracking": self.track_performance,
                "timing_tracking": self.track_timing,
            },
            "context": {
                "auto_propagation": self.dispatcher_auto_context,
                "correlation_id_enabled": self.include_correlation_id,
                "performance_tracking": self.track_performance,
                "metadata_inclusion": self.include_context,
            },
        }

        if component not in component_configs:
            return FlextResult[FlextTypes.Dict].fail(
                f"Unknown component: {component}. Available: {list(component_configs.keys())}"
            )

        from typing import cast

        return FlextResult[FlextTypes.Dict].ok(
            cast("FlextTypes.Dict", component_configs[component])
        )

    def create_service_config(
        self, service_name: str, **overrides: FlextTypes.ConfigValue
    ) -> FlextResult[FlextTypes.Dict]:
        """Create service configuration with flext-core integration patterns.

        Demonstrates how FlextConfig integrates with FlextService, FlextContainer,
        and other flext-core components to provide comprehensive service configuration.

        Args:
            service_name: Name of the service to configure
            **overrides: Service-specific configuration overrides

        Returns:
            FlextResult containing complete service configuration

        Example:
            >>> config = FlextConfig()
            >>> service_config = config.create_service_config(
            ...     "user_service", timeout_seconds=60, enable_caching=True
            ... )
            >>> if service_config.is_success:
            ...     print(f"Service config: {service_config.unwrap()}")

        """
        # Base service configuration using flext-core patterns
        base_config = {
            "service_name": service_name,
            "timeout_seconds": self.timeout_seconds,
            "max_retry_attempts": self.max_retry_attempts,
            "enable_caching": self.enable_caching,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "performance_config": {
                "batch_size": self.batch_size,
                "max_workers": min(self.max_workers, 4),  # Service-specific limit
                "circuit_breaker_enabled": self.enable_circuit_breaker,
            },
            "logging_config": {
                "level": self.effective_log_level,
                "include_context": self.include_context,
                "structured": self.structured_output,
            },
            "context_config": {
                "auto_context": self.dispatcher_auto_context,
                "correlation_id": self.include_correlation_id,
            },
        }

        # Apply overrides
        base_config.update(overrides)  # type: ignore[arg-type]

        # Validate service configuration
        timeout_value = base_config.get("timeout_seconds", self.timeout_seconds)
        timeout_seconds = (
            int(timeout_value)
            if isinstance(timeout_value, (int, str))
            else self.timeout_seconds
        )

        retry_value = base_config.get("max_retry_attempts", self.max_retry_attempts)
        max_retry_attempts = (
            int(retry_value)
            if isinstance(retry_value, (int, str))
            else self.max_retry_attempts
        )

        if timeout_seconds < 1:
            return FlextResult[FlextTypes.Dict].fail(
                "Service timeout must be at least 1 second"
            )

        if max_retry_attempts < 0:
            return FlextResult[FlextTypes.Dict].fail(
                "Retry attempts cannot be negative"
            )

        return FlextResult[FlextTypes.Dict].ok(cast("FlextTypes.Dict", base_config))

    def validate_flext_core_integration(self) -> FlextResult[None]:
        """Validate flext-core integration configuration with comprehensive checks.

        Demonstrates advanced flext-core integration by validating that all
        components work together properly with proper error handling and logging.

        Returns:
            FlextResult indicating success or detailed failure information

        Example:
            >>> config = FlextConfig()
            >>> validation = config.validate_flext_core_integration()
            >>> if validation.is_failure:
            ...     print(f"Integration issues: {validation.error}")

        """
        issues = []

        # Validate component compatibility
        if self.enable_circuit_breaker and not self.enable_metrics:
            issues.append("Circuit breaker requires metrics to be enabled")

        if self.enable_tracing and not self.include_correlation_id:
            issues.append("Tracing works best with correlation ID enabled")

        if self.trace and not self.debug:
            issues.append("Trace mode requires debug mode to be enabled")

        # Validate resource limits
        if self.max_workers > FlextConstants.Container.MAX_WORKERS:
            issues.append(
                f"Max workers {self.max_workers} exceeds recommended limit {FlextConstants.Container.MAX_WORKERS}"
            )

        if self.database_pool_size > FlextConstants.Performance.MAX_DB_POOL_SIZE:
            issues.append(
                f"Database pool size {self.database_pool_size} exceeds maximum {FlextConstants.Performance.MAX_DB_POOL_SIZE}"
            )

        # Validate timeout consistency
        if self.dispatcher_timeout_seconds > self.timeout_seconds * 2:
            issues.append(
                "Dispatcher timeout should not exceed general timeout by more than 2x"
            )

        if issues:
            return FlextResult[None].fail(
                f"Flext-core integration issues: {'; '.join(issues)}"
            )

        return FlextResult[None].ok(None)

    def get_integration_example(self, pattern: str) -> FlextResult[str]:
        """Get flext-core integration examples for common patterns.

        Provides practical examples of how to integrate FlextConfig with other
        flext-core components, demonstrating best practices for the ecosystem.

        Args:
            pattern: Integration pattern ('service', 'handler', 'middleware', 'pipeline')

        Returns:
            FlextResult containing example code or error

        Example:
            >>> config = FlextConfig()
            >>> example = config.get_integration_example("service")
            >>> if example.is_success:
            ...     print(f"Service integration example: {example.unwrap()}")

        """
        examples = {
            "service": """
# Example: Service with FlextConfig integration
from flext_core import FlextConfig, FlextService, FlextLogger, FlextResult

class MyService(FlextService):
    def __init__(self, config: FlextConfig | None = None):
        super().__init__()
        self._config = config or FlextConfig()
        self._logger = FlextLogger(__name__)

    async def process(self, data: dict) -> FlextResult[FlextTypes.Dict]:
        # Use config for timeout and validation
        timeout = self._config.timeout_seconds
        max_retries = self._config.max_retry_attempts

        # Validate using config settings
        if len(data) > self._config.batch_size:
            return FlextResult[FlextTypes.Dict].fail("Data batch too large")

        return FlextResult[FlextTypes.Dict].ok({"processed": True})
""",
            "handler": """
# Example: Handler with FlextConfig integration
from flext_core import FlextConfig, FlextHandlers, FlextResult

class MyHandler(FlextHandlers[dict, dict]):
    def __init__(self, config: FlextConfig | None = None):
        super().__init__()
        self._config = config or FlextConfig()

    def handle(self, command: dict) -> FlextResult[FlextTypes.Dict]:
        # Use config for validation and processing
        if not command:
            return FlextResult[FlextTypes.Dict].fail("Command cannot be empty")

        # Process with config-driven settings
        result = {"handled": True, "timestamp": "now"}
        return FlextResult[FlextTypes.Dict].ok(result)
""",
            "middleware": """
# Example: Middleware with FlextConfig integration
from flext_core import FlextConfig, FlextResult

class MyMiddleware:
    def __init__(self, config: FlextConfig | None = None):
        self._config = config or FlextConfig()

    def process_request(self, request: dict) -> FlextResult[FlextTypes.Dict]:
        # Use config for rate limiting and validation
        if self._config.rate_limit_max_requests <= 0:
            return FlextResult[FlextTypes.Dict].fail("Rate limiting disabled")

        # Validate request size
        if len(str(request)) > self._config.validation_timeout_ms:
            return FlextResult[FlextTypes.Dict].fail("Request too large")

        return FlextResult[FlextTypes.Dict].ok(request)
""",
            "pipeline": """
# Example: Pipeline with FlextConfig integration
from flext_core import FlextConfig, FlextResult

class MyPipeline:
    def __init__(self, config: FlextConfig | None = None):
        self._config = config or FlextConfig()
        self._steps = []

    def add_step(self, step_func):
        self._steps.append(step_func)
        return self

    def execute(self, data: dict) -> FlextResult[FlextTypes.Dict]:
        # Use config for pipeline settings
        if len(self._steps) > self._config.batch_size:
            return FlextResult[FlextTypes.Dict].fail("Too many pipeline steps")

        # Execute with config-driven error handling
        for step in self._steps:
            result = step(data)
            if result.is_failure:
                return result

        return FlextResult[FlextTypes.Dict].ok({"pipeline_completed": True})
""",
        }

        if pattern not in examples:
            available = ", ".join(examples.keys())
            return FlextResult[str].fail(
                f"Unknown pattern: {pattern}. Available: {available}"
            )

        return FlextResult[str].ok(examples[pattern].strip())


FlextConfig.model_rebuild()

__all__ = [
    "FlextConfig",
]
