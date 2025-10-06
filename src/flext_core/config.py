"""Configuration subsystem delivering the FLEXT 1.0.0 alignment pillar.

Bidirectional sync between Pydantic BaseSettings and DI Configuration:
    - FlextConfig values injectable through DI container
    - Configuration provider for dependency injection
    - Maintains Pydantic validation while enabling DI patterns
    - Clean Layer 2 dependency

Dependency Layer: 2 (Foundation Configuration)
Dependencies: dependency_injector, constants, exceptions, result, typings
Used by: All Flext modules requiring configuration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from dependency_injector import providers
from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

structlog = FlextRuntime.structlog()


class FlextConfig(BaseSettings):
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

    🔌 DEPENDENCY INJECTION INTEGRATION (v1.1.0+)
    Bidirectional sync with dependency-injector:
    - Internal Configuration provider for DI container
    - Pydantic validation + DI injectable values
    - Automatic sync on config creation/update
    - Register config instance in FlextContainer for injection
    - Access config values via DI: container.config.log_level()

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
                # Lazy import to avoid circular dependency
                if TYPE_CHECKING:
                    from flext_core.protocols import FlextProtocols

                try:
                    from flext_core.protocols import FlextProtocols

                    if isinstance(
                        handler_config, FlextProtocols.Foundation.HasHandlerType
                    ):
                        config_mode: str | None = handler_config.handler_type
                        if config_mode in {"command", "query"}:
                            return str(config_mode)
                except ImportError:
                    # Handle case where protocols module not available
                    pass

                # Try dict access
                if isinstance(handler_config, dict):
                    config_mode_dict: object = handler_config.get("handler_type")
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

    class Providers:
        """Dependency injection provider factory utilities.

        Provides factory methods for creating dependency_injector providers
        for FlextConfig integration with the DI container.

        This class demonstrates Phase 2 enhancement patterns:
        - Singleton providers for global config access
        - Factory providers for config instance creation
        - Callable providers for computed configuration values
        - Configuration providers for settings injection
        """

        @staticmethod
        def create_singleton_provider(
            config_instance: FlextConfig | None = None,
        ) -> providers.Singleton[FlextConfig]:
            """Create a Singleton provider for FlextConfig.

            Args:
                config_instance: Optional config instance to use. If None,
                    uses get_global_instance() to get/create singleton.

            Returns:
                providers.Singleton: Singleton provider for config instance

            Example:
                >>> from flext_core import FlextConfig
                >>> config_provider = FlextConfig.Providers.create_singleton_provider()
                >>> # Use in container
                >>> container.config = config_provider

            """
            if config_instance is None:
                config_instance = FlextConfig.get_global_instance()

            return providers.Singleton(lambda: config_instance)

        @staticmethod
        def create_factory_provider(
            **default_kwargs: FlextTypes.ConfigValue,
        ) -> providers.Factory[FlextConfig]:
            """Create a Factory provider for FlextConfig instances.

            Args:
                **default_kwargs: Default configuration values to use

            Returns:
                providers.Factory: Factory provider for creating config instances

            Example:
                >>> factory = FlextConfig.Providers.create_factory_provider(
                ...     debug=True, environment="test"
                ... )
                >>> # Creates new config with defaults
                >>> config = factory()

            """
            return providers.Factory(
                FlextConfig.create,
                **default_kwargs,
            )

        @staticmethod
        def create_callable_provider(
            config_instance: FlextConfig,
            field_name: str,
        ) -> providers.Callable[object]:
            """Create a Callable provider for a specific config field.

            Args:
                config_instance: Config instance to read from
                field_name: Name of the field to provide

            Returns:
                providers.Callable: Callable provider for field value

            Example:
                >>> config = FlextConfig()
                >>> log_level_provider = FlextConfig.Providers.create_callable_provider(
                ...     config, "log_level"
                ... )
                >>> # Access field value
                >>> level = log_level_provider()

            """
            return providers.Callable(lambda: getattr(config_instance, field_name))

        @staticmethod
        def create_configuration_provider(
            config_instance: FlextConfig | None = None,
        ) -> Any:  # providers.Configuration
            """Create a Configuration provider from FlextConfig.

            Args:
                config_instance: Optional config instance. If None,
                    uses get_global_instance().

            Returns:
                providers.Configuration: Configuration provider with all config values

            Example:
                >>> config_provider = (
                ...     FlextConfig.Providers.create_configuration_provider()
                ... )
                >>> # Access via DI
                >>> log_level = config_provider.log_level()
                >>> timeout = config_provider.timeout_seconds()

            """
            if config_instance is None:
                config_instance = FlextConfig.get_global_instance()

            config_provider = providers.Configuration()
            config_provider.from_dict(config_instance.model_dump())
            return config_provider

        @staticmethod
        def create_component_provider(
            component_name: str,
            config_instance: FlextConfig | None = None,
        ) -> FlextResult[Any]:  # FlextResult[providers.Callable]
            """Create a provider for component-specific configuration.

            Args:
                component_name: Name of component ('container', 'bus', etc.)
                config_instance: Optional config instance

            Returns:
                FlextResult containing Callable provider or error

            Example:
                >>> result = FlextConfig.Providers.create_component_provider(
                ...     "container"
                ... )
                >>> if result.is_success:
                ...     container_config_provider = result.unwrap()
                ...     config = container_config_provider()

            """
            if config_instance is None:
                config_instance = FlextConfig.get_global_instance()

            # Validate component exists
            component_config_result = config_instance.get_component_config(
                component_name
            )
            if component_config_result.is_failure:
                return FlextResult[Any].fail(
                    f"Component provider creation failed: {component_config_result.error}"
                )

            # Create callable provider for component config
            provider = providers.Callable(
                lambda: config_instance.get_component_config(component_name).unwrap()
            )

            return FlextResult[Any].ok(provider)

        @staticmethod
        def register_in_container(
            container: Any,  # FlextContainer or dependency_injector.containers.Container
            config_instance: FlextConfig | None = None,
        ) -> FlextResult[None]:
            """Register FlextConfig providers in a DI container.

            Args:
                container: DI container to register providers in
                config_instance: Optional config instance to use

            Returns:
                FlextResult[None]: Success or failure result

            Example:
                >>> from flext_core import FlextContainer, FlextConfig
                >>> container = FlextContainer.get_global()
                >>> result = FlextConfig.Providers.register_in_container(
                ...     container, FlextConfig()
                ... )

            """
            try:
                if config_instance is None:
                    config_instance = FlextConfig.get_global_instance()

                # Check if container has _di_container attribute (FlextContainer)
                if hasattr(container, "_di_container"):
                    # FlextContainer with internal dependency-injector container
                    di_container = container._di_container

                    # Register configuration provider
                    di_container.config = (
                        FlextConfig.Providers.create_configuration_provider(
                            config_instance
                        )
                    )

                    # Register singleton config instance
                    config_singleton = FlextConfig.Providers.create_singleton_provider(
                        config_instance
                    )
                    container.register("flext_config", config_singleton)

                    return FlextResult[None].ok(None)

                # Direct dependency-injector container
                if hasattr(container, "config"):
                    container.config = (
                        FlextConfig.Providers.create_configuration_provider(
                            config_instance
                        )
                    )
                    return FlextResult[None].ok(None)

                return FlextResult[None].fail(
                    "Container does not support config provider registration"
                )

            except Exception as e:
                return FlextResult[None].fail(f"Provider registration failed: {e}")

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

    max_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum batch size for batch operations",
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
            nested_obj: object = getattr(self, first_key)

            # Handle dict access
            if isinstance(nested_obj, dict):
                nested_dict = cast("dict[str, object]", nested_obj)
                if remaining_key not in nested_dict:
                    msg = f"Configuration key '{key}' not found in nested config"
                    raise KeyError(msg)
                return nested_dict[remaining_key]

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

    # Field validators with structured logging
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate that environment is one of the allowed values with structured logging."""
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

            # Structured validation error logging
            try:
                logger = structlog.get_logger()
                logger.warning(
                    "config_validation_failed",
                    event_type="validation_error",
                    validator="validate_environment",
                    field="environment",
                    invalid_value=v,
                    valid_values=sorted(valid_environments),
                )
            except Exception:
                # Don't fail validation if logging fails, but log to stderr for diagnostics
                import logging as python_logging

                stderr_logger = python_logging.getLogger("flext.config.validation")
                stderr_logger.exception(
                    "Validation logging failed for field '%s'. Original validation error: %s",
                    "environment",
                    msg,
                )

            raise FlextExceptions.ValidationError(
                message=msg,
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log_level is one of the allowed values with structured logging."""
        v_upper = v.upper()
        if v_upper not in FlextConstants.Logging.VALID_LEVELS:
            msg = f"Invalid log level: {v}. Must be one of: {', '.join(FlextConstants.Logging.VALID_LEVELS)}"

            # Structured validation error logging
            try:
                logger = structlog.get_logger()
                logger.warning(
                    "config_validation_failed",
                    event_type="validation_error",
                    validator="validate_log_level",
                    field="log_level",
                    invalid_value=v,
                    valid_values=list(FlextConstants.Logging.VALID_LEVELS),
                )
            except Exception:
                # Don't fail validation if logging fails, but log to stderr for diagnostics
                import logging as python_logging

                stderr_logger = python_logging.getLogger("flext.config.validation")
                stderr_logger.exception(
                    "Validation logging failed for field '%s'. Original validation error: %s",
                    "log_level",
                    msg,
                )

            raise FlextExceptions.ValidationError(
                message=msg,
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
            )

        # Trace requires debug
        if self.trace and not self.debug:
            msg = "Trace mode requires debug mode to be enabled"
            raise FlextExceptions.ValidationError(
                message=msg,
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

    @classmethod
    def _update_models_config(cls, _config_instance: FlextConfig) -> None:
        """Update the models module configuration to use the current FlextConfig instance.

        This ensures that FlextModels classes use the current FlextConfig instance
        as their configuration source, enabling the newer pattern where FlextConfig
        serves as the central source of configuration for all model classes.
        """
        with contextlib.suppress(ImportError):
            # Import models module and update its config
            # NOTE: This access to private _config is necessary for the current
            # architecture where models use global config for field defaults.
            pass

            # Configuration is handled through dependency injection

    # NOTE: Removed synchronize_di_config validator to avoid circular dependency
    # The DI Configuration provider is created lazily when first accessed
    # and automatically syncs with the Pydantic settings instance

    # Dependency Injection integration (v1.1.0+)
    _di_config_provider: ClassVar[Any | None] = None  # providers.Configuration
    _di_provider_lock: ClassVar[threading.Lock] = threading.Lock()

    # Singleton pattern implementation - per-class singletons
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, **_data: object) -> Self:
        """Create new FlextConfig instance.

        Normal Pydantic instantiation - each call creates a new instance.
        Use get_global_instance() for singleton behavior.
        """
        return super().__new__(cls)

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
        if cls not in cls._instances:
            with cls._instance_lock:
                if cls not in cls._instances:
                    instance = cls()
                    cls._instances[cls] = instance
                    # Update models module config when global instance is created
                    cls._update_models_config(instance)
        return cast("Self", cls._instances[cls])

    @classmethod
    def set_global_instance(cls, instance: FlextConfig) -> None:
        """Set the global singleton instance for this class.

        Args:
            instance: Configuration instance to set as singleton for this class.

        """
        with cls._instance_lock:
            cls._instances[cls] = instance
            # Update models module config when FlextConfig changes
            cls._update_models_config(instance)

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset the global singleton instance for this class.

        Removes the singleton instance, so the next call to get_global_instance()
        will create a new instance.
        """
        with cls._instance_lock:
            cls._instances.pop(cls, None)

    @classmethod
    def _get_or_create_di_provider(cls) -> Any:  # providers.Configuration
        """Get or create the dependency-injector Configuration provider.

        Creates a Configuration provider linked to Pydantic settings as described in:
        https://python-dependency-injector.ets-labs.org/providers/configuration.html

        Returns:
            Any: The DI Configuration provider instance (providers.Configuration)

        """
        if cls._di_config_provider is None:
            with cls._di_provider_lock:
                if cls._di_config_provider is None:
                    # Create Configuration provider
                    cls._di_config_provider = providers.Configuration()

                    # Populate with Pydantic settings if instance exists for this class
                    instance = cls._instances.get(cls)
                    if instance is not None:
                        # Convert Pydantic model to dict and populate DI provider
                        config_dict = instance.model_dump()
                        if cls._di_config_provider is not None:
                            cls._di_config_provider.from_dict(config_dict)
        return cls._di_config_provider

    @classmethod
    def _sync_to_di_provider(cls, config_instance: FlextConfig) -> None:
        """Sync FlextConfig Pydantic settings to DI Configuration provider.

        Implements bidirectional sync pattern from dependency-injector docs.
        The Configuration provider automatically reads values from Pydantic settings.

        Args:
            config_instance: FlextConfig instance to sync to DI provider.

        """
        # Get or create the DI Configuration provider
        di_provider = cls._get_or_create_di_provider()

        # Update the provider with current Pydantic settings instance
        # The provider automatically reads values from the Pydantic settings
        if di_provider is not None:
            di_provider.set_pydantic_settings([config_instance])

    @classmethod
    def get_di_config_provider(cls) -> Any:  # providers.Configuration
        """Get the dependency-injector Configuration provider for FlextConfig.

        This provider can be used in FlextContainer to make configuration
        values injectable through dependency injection.

        Returns:
            Any: Configuration provider for DI container (providers.Configuration)

        Example:
            >>> config_provider = FlextConfig.get_di_config_provider()
            >>> # Access config values through DI
            >>> log_level = config_provider.log_level()
            >>> timeout = config_provider.timeout_seconds()

        Example (in FlextContainer):
            >>> # Register in DI container
            >>> container._di_container.config = FlextConfig.get_di_config_provider()
            >>> # Access via container
            >>> log_level = container._di_container.config.log_level()

        """
        # Get or create the provider
        provider = cls._get_or_create_di_provider()

        # Ensure it's synced with current instance for this class
        instance = cls._instances.get(cls)
        if instance is not None:
            provider.set_pydantic_settings([instance])

        return provider

    @classmethod
    def reset_di_config_provider(cls) -> None:
        """Reset the DI Configuration provider (mainly for testing).

        Clears the cached Configuration provider, forcing recreation on next access.
        """
        with cls._di_provider_lock:
            cls._di_config_provider = None

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
        config_data: dict[str, object] = {"environment": environment, **kwargs}
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
                # Create instance using model_validate to bypass singleton __new__
                config = cls.model_validate(data)
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

    # =========================================================================
    # Advanced Structlog Integration (Phase 1 Enhancement)
    # =========================================================================

    @classmethod
    def audit_config_change(
        cls,
        *,
        field: str,
        old_value: object,
        new_value: object,
        change_reason: str | None = None,
    ) -> FlextResult[None]:
        """Audit configuration changes with structured logging.

        Args:
            field: Configuration field that changed
            old_value: Previous field value (masked if sensitive)
            new_value: New field value (masked if sensitive)
            change_reason: Optional reason for the change

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> FlextConfig.audit_config_change(
            ...     field="log_level",
            ...     old_value="INFO",
            ...     new_value="DEBUG",
            ...     change_reason="Troubleshooting production issue",
            ... )

        """
        try:
            # Mask sensitive fields
            sensitive_fields = {"secret_key", "api_key", "jwt_secret", "database_url"}
            is_sensitive = field in sensitive_fields

            masked_old = (
                FlextConstants.Messages.REDACTED_SECRET if is_sensitive else old_value
            )
            masked_new = (
                FlextConstants.Messages.REDACTED_SECRET if is_sensitive else new_value
            )

            # Structured audit log
            logger = structlog.get_logger()
            logger.info(
                "config_change_audit",
                event_type="configuration_change",
                field=field,
                old_value=masked_old,
                new_value=masked_new,
                change_reason=change_reason,
                is_sensitive=is_sensitive,
            )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Audit logging failed: {e}")

    def track_validation_error(
        self,
        *,
        validator: str,
        field: str,
        error: str,
        value: object = None,
    ) -> FlextResult[None]:
        """Track validation errors with structured logging.

        Args:
            validator: Name of the validator that failed
            field: Field that failed validation
            error: Validation error message
            value: Value that failed (masked if sensitive)

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> config.track_validation_error(
            ...     validator="validate_log_level",
            ...     field="log_level",
            ...     error="Invalid log level: TRACE",
            ...     value="TRACE",
            ... )

        """
        try:
            # Mask sensitive field values
            sensitive_fields = {"secret_key", "api_key", "jwt_secret", "database_url"}
            is_sensitive = field in sensitive_fields
            masked_value = (
                FlextConstants.Messages.REDACTED_SECRET if is_sensitive else value
            )

            logger = structlog.get_logger()
            logger.warning(
                "config_validation_error",
                event_type="validation_failure",
                validator=validator,
                field=field,
                error=error,
                value=masked_value if value is not None else "not_provided",
                is_sensitive=is_sensitive,
            )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Validation tracking failed: {e}")

    def generate_config_diff(
        self,
        other: FlextConfig,
    ) -> FlextResult[FlextTypes.Dict]:
        """Generate structured diff between two configurations.

        Args:
            other: Other configuration to compare with

        Returns:
            FlextResult containing diff dictionary or error

        Example:
            >>> config1 = FlextConfig(debug=False, log_level="INFO")
            >>> config2 = FlextConfig(debug=True, log_level="DEBUG")
            >>> diff_result = config1.generate_config_diff(config2)
            >>> if diff_result.is_success:
            ...     print(diff_result.unwrap())

        """
        try:
            current_dict = self.model_dump()
            other_dict = other.model_dump()

            # Track differences with proper type annotations
            added_fields: dict[str, object] = {}
            removed_fields: dict[str, object] = {}
            changed_fields: dict[str, dict[str, object]] = {}

            # Find added and changed fields
            for key, value in other_dict.items():
                if key not in current_dict:
                    added_fields[key] = value
                elif current_dict[key] != value:
                    changed_fields[key] = {
                        "old": current_dict[key],
                        "new": value,
                    }

            # Find removed fields
            for key in current_dict:
                if key not in other_dict:
                    removed_fields[key] = current_dict[key]

            # Create differences dict
            differences: FlextTypes.Dict = {
                "added": added_fields,
                "removed": removed_fields,
                "changed": changed_fields,
            }

            # Type assertions for mypy
            added_dict: dict[str, object] = cast(
                "dict[str, object]", differences["added"]
            )
            removed_dict: dict[str, object] = cast(
                "dict[str, object]", differences["removed"]
            )
            changed_dict: dict[str, dict[str, object]] = cast(
                "dict[str, dict[str, object]]", differences["changed"]
            )

            # Mask sensitive fields
            sensitive_fields = {"secret_key", "api_key", "jwt_secret", "database_url"}
            for section_data in differences.values():
                if not isinstance(section_data, dict):
                    continue
                section_data_dict = cast("dict[str, object]", section_data)
                for field in section_data_dict:
                    if field in sensitive_fields:
                        if isinstance(section_data_dict[field], dict):
                            section_data_dict[field] = {
                                "old": FlextConstants.Messages.REDACTED_SECRET,
                                "new": FlextConstants.Messages.REDACTED_SECRET,
                            }
                        else:
                            section_data_dict[field] = (
                                FlextConstants.Messages.REDACTED_SECRET
                            )

            # Log the diff
            logger = structlog.get_logger()
            logger.info(
                "config_diff_generated",
                event_type="configuration_diff",
                changes_count=len(changed_dict),
                additions_count=len(added_dict),
                removals_count=len(removed_dict),
            )

            return FlextResult[FlextTypes.Dict].ok(differences)
        except Exception as e:
            return FlextResult[FlextTypes.Dict].fail(f"Diff generation failed: {e}")

    def log_config_state(
        self,
        *,
        event: str = "config_snapshot",
        include_sensitive: bool = False,
    ) -> FlextResult[None]:
        """Log current configuration state with structured logging.

        Args:
            event: Event name for the log entry
            include_sensitive: Whether to include sensitive fields (NOT RECOMMENDED)

        Returns:
            FlextResult[None]: Success or failure result

        Example:
            >>> config.log_config_state(event="application_startup")

        """
        try:
            config_dict = self.model_dump()

            # Mask sensitive fields unless explicitly requested
            if not include_sensitive:
                sensitive_fields = {
                    "secret_key",
                    "api_key",
                    "jwt_secret",
                    "database_url",
                }
                for field in sensitive_fields:
                    if field in config_dict and config_dict[field] is not None:
                        config_dict[field] = FlextConstants.Messages.REDACTED_SECRET

            logger = structlog.get_logger()
            logger.info(
                event,
                event_type="configuration_snapshot",
                environment=self.environment,
                debug_enabled=self.debug,
                trace_enabled=self.trace,
                log_level=self.effective_log_level,
                app_name=self.app_name,
                version=self.version,
                config_summary={
                    "environment": self.environment,
                    "debug": self.debug,
                    "log_level": self.log_level,
                    "max_workers": self.max_workers,
                    "timeout_seconds": self.timeout_seconds,
                },
            )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Config state logging failed: {e}")

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
                indent: int | str | None = (
                    int(indent_value)
                    if indent_value is not None and isinstance(indent_value, (int, str))
                    else self.json_indent
                )
                sort_keys_value = kwargs.get("sort_keys", self.json_sort_keys)
                sort_keys: bool = (
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
        component_configs: dict[str, FlextTypes.Dict] = {
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

        return FlextResult[FlextTypes.Dict].ok(component_configs[component])

    def get_config_with_fallback(
        self,
        primary_key: str,
        *fallback_keys: str,
        default: object | None = None,
    ) -> object:
        """Get configuration value trying multiple keys using alt pattern.

        Demonstrates alt pattern for configuration fallback chains.

        Args:
            primary_key: Primary configuration key to try
            *fallback_keys: Fallback keys to try in order
            default: Default value if all keys fail

        Returns:
            Configuration value from first found key, or default

        Example:
            >>> config = FlextConfig()
            >>> host = config.get_config_with_fallback(
            ...     "db_host", "database_host", default="localhost"
            ... )

        """

        def get_value(key: str) -> FlextResult[object]:
            if hasattr(self, key):
                value = getattr(self, key)
                if value is not None:
                    return FlextResult[object].ok(value)
            return FlextResult[object].fail(f"Key '{key}' not found or None")

        result = get_value(primary_key)
        for fallback_key in fallback_keys:
            result = result.alt(get_value(fallback_key))

        return (
            result.unwrap_or(default) if default is not None else result.unwrap_or(None)
        )

    def validate_config_pipeline(self) -> FlextResult[None]:
        """Validate complete configuration using flow_through pattern.

        Demonstrates flow_through for validation pipeline composition.

        Returns:
            FlextResult[None]: Success if all validations pass

        Example:
            >>> config = FlextConfig()
            >>> result = config.validate_config_pipeline()
            >>> if result.is_success:
            ...     print("Configuration valid")

        """
        return (
            FlextResult[None]
            .ok(None)
            .flow_through(
                lambda _: self.validate_runtime_requirements(),
                lambda _: self.validate_business_rules(),
            )
        )

    def safe_get_component_config(
        self,
        component: str,
        default_config: FlextTypes.Dict | None = None,
    ) -> FlextTypes.Dict:
        """Get component config with fallback using lash pattern.

        Demonstrates lash for error recovery with fallback.

        Args:
            component: Component name
            default_config: Default configuration if component not found

        Returns:
            Component configuration or default

        Example:
            >>> config = FlextConfig()
            >>> logger_config = config.safe_get_component_config(
            ...     "logger", {"level": "INFO"}
            ... )

        """

        def provide_default(_error: str) -> FlextResult[FlextTypes.Dict]:
            if default_config is not None:
                return FlextResult[FlextTypes.Dict].ok(default_config)
            return FlextResult[FlextTypes.Dict].ok({})

        return self.get_component_config(component).lash(provide_default).unwrap()


FlextConfig.model_rebuild()

__all__ = [
    "FlextConfig",
]
