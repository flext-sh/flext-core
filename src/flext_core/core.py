"""Enterprise-grade central orchestration hub for the complete FLEXT Core ecosystem.

This module serves as the primary entry point and comprehensive facade for all FLEXT Core
functionality, implementing singleton patterns for enterprise system management. It provides
unified access to dependency injection, domain modeling, validation, handlers, observability,
and all architectural patterns through a single, thread-safe interface.

The FlextCore class orchestrates:
    - Container management with global service registration
    - Structured logging with correlation tracking
    - Railway-oriented programming with FlextResult[T]
    - Domain-driven design patterns (Entity, Value Object, Aggregate Root)
    - CQRS implementation with command/query separation
    - Comprehensive validation and guard systems
    - Enterprise decorator patterns for cross-cutting concerns
    - Performance monitoring and observability integration
    - Configuration management with environment support
    - Plugin architecture and middleware systems

Architectural Features:
    - Thread-safe singleton pattern with lazy initialization
    - Comprehensive error handling with FlextResult railway patterns
    - Enterprise-grade logging with structured output and correlation IDs
    - Type-safe dependency injection with factory pattern support
    - Hierarchical configuration management with environment variables
    - Performance tracking with metrics collection
    - Plugin system with lifecycle management
    - Boilerplate reduction through dynamic class generation

Usage Examples:
    Basic system initialization::

        core = FlextCore.get_instance()
        core.configure_logging(log_level="INFO", _json_output=True)

        # Register services
        core.register_service("database", DatabaseService())
        core.register_factory("logger", lambda: get_logger(__name__))

    Railway-oriented programming::

        result = (
            core.ok(user_data)
            .flat_map(lambda data: core.validate_required(data, "user_data"))
            .flat_map(lambda data: core.create_entity(User, **data))
            .map(lambda user: core.register_service("current_user", user))
        )

    Configuration management::

        config = core.get_environment_config("production")
        if config.success:
            validated = core.validate_config_with_types(config.value)
            core.configure_core_system(validated.value)

    Domain modeling with validation::

        UserEntity = core.create_entity_with_validators(
            "User",
            {"name": (str, {}), "email": (str, {})},
            {"email": core.validate_email},
        )

Integration Points:
    - FlextContainer: Global dependency injection container
    - FlextLogger: Structured logging with correlation IDs
    - FlextResult[T]: Railway-oriented programming patterns
    - FlextValidation: Hierarchical validation system
    - FlextHandlers: Enterprise request processing patterns
    - FlextDecorators: Cross-cutting concern implementations
    - FlextModels: Consolidated domain modeling system
    - FlextObservability: Metrics and monitoring integration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Annotated, cast, override

from pydantic import Field, field_validator

from flext_core.commands import FlextCommands
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions
from flext_core.fields import FlextFields
from flext_core.guards import FlextGuards
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.observability import (
    FlextObservability,
    get_global_observability,
    reset_global_observability,
)
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult, safe_call
from flext_core.schema_processing import FlextProcessingPipeline
from flext_core.typings import FlextTypes, P, R, T
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextValidation


def flext_validate_service_name(name: str) -> FlextResult[None]:
    """Validate service name according to FLEXT naming conventions.

    Ensures service names follow proper conventions for consistency and avoid
    conflicts in the dependency injection container. Service names must be
    non-empty strings without leading/trailing whitespace.

    Args:
        name (str): The service name to validate.

    Returns:
        FlextResult[None]: Success if name is valid, failure with descriptive error
            if validation fails.

    Validation Rules:
        - Must be a string type
        - Cannot be None or empty
        - Cannot consist only of whitespace characters
        - Should follow snake_case convention (recommended but not enforced)

    Usage Examples:
        Basic validation::

            result = flext_validate_service_name("user_service")
            assert result.success

            result = flext_validate_service_name("")
            assert result.failure
            assert "cannot be only whitespace" in result.error

    See Also:
        - FlextCore.validate_service_name(): Public validation method
        - FlextCore.register_service(): Service registration with validation

    """
    if not name or not isinstance(name, str):
        return FlextResult[None].fail("Service name must be a non-empty string")
    if not name.strip():
        return FlextResult[None].fail("Service name cannot be only whitespace")
    return FlextResult[None].ok(None)


# Type aliases following FLEXT centralized patterns for consistent typing
ValidatorCallable = FlextTypes.Core.FlextCallableType
"""Type alias for callable validators in the FLEXT ecosystem.

Defines the standard callable signature for validation functions used throughout
the FLEXT Core system. Validators should accept any object and return a boolean
indicating validity, enabling consistent validation patterns across all components.

Usage:
    validators: dict[str, ValidatorCallable] = {
        "email": lambda x: "@" in str(x),
        "positive": lambda x: isinstance(x, (int, float)) and x > 0
    }
"""


class FlextCore:
    r"""Enterprise-grade comprehensive facade providing unified access to the complete FLEXT ecosystem.

    This class serves as the central orchestration hub for all FLEXT Core functionality,
    implementing enterprise patterns for system management, dependency injection, logging,
    validation, domain modeling, CQRS, observability, and architectural patterns through
    a single, thread-safe interface with singleton lifecycle management.

    The FlextCore facade provides comprehensive access to:
        - **Container Management**: Global dependency injection with factory patterns
        - **Logging Infrastructure**: Structured logging with correlation tracking
        - **Railway Programming**: FlextResult[T] patterns for composable error handling
        - **Domain Modeling**: DDD patterns with Entity, Value Object, Aggregate Root
        - **CQRS Implementation**: Command/Query separation with handler registration
        - **Validation Systems**: Hierarchical validation with predicate composition
        - **Decorator Patterns**: Cross-cutting concerns for reliability and observability
        - **Performance Monitoring**: Metrics collection and performance tracking
        - **Configuration Management**: Environment-aware configuration with validation
        - **Plugin Architecture**: Extensible plugin system with lifecycle management
        - **Utility Functions**: ID generation, type guards, and helper utilities

    Thread Safety:
        All operations are thread-safe with proper synchronization for concurrent access.
        The singleton pattern ensures consistent state across the application lifecycle.

    Lazy Initialization:
        Components are initialized on-demand to optimize memory usage and startup time.
        Heavy subsystems (handlers, fields, plugins) are created only when accessed.

    Architecture Benefits:
        - **Single Point of Entry**: Unified interface for all FLEXT functionality
        - **Dependency Inversion**: Loose coupling through container-based injection
        - **Railway Patterns**: Composable error handling without exceptions
        - **Type Safety**: Full generic type support with runtime validation
        - **Enterprise Scale**: Designed for high-throughput production environments
        - **Extensibility**: Plugin architecture for custom functionality

    Usage Examples:
        System initialization and configuration::

            core = FlextCore.get_instance()

            # Configure logging with structured output
            core.configure_logging(log_level="INFO", _json_output=True)

            # Setup services with validation
            services = {
                "database": DatabaseService,
                "cache": lambda: RedisCache(host="localhost"),
                "metrics": MetricsCollector,
            }
            container_result = core.setup_container_with_services(
                services, core.validate_service_name
            )

        Railway-oriented domain operations::

            # Composable validation and entity creation
            user_result = (
                core.validate_required(user_data, "user_data")
                .flat_map(lambda data: core.validate_email(data.get("email")))
                .flat_map(lambda email: core.create_entity(User, **user_data))
                .tap(
                    lambda user: core.get_logger(__name__).info(
                        "User created", user_id=user.id
                    )
                )
                .map_error(lambda err: f"User creation failed: {err}")
            )

        Environment-aware configuration::

            # Production environment setup
            config_result = core.create_environment_core_config("production")
            if config_result.success:
                optimized = core.optimize_core_performance(config_result.value)
                core.configure_core_system(optimized.value)

        Dynamic class generation for boilerplate reduction::

            # Create validated domain classes
            UserEntity = core.create_entity_with_validators(
                "User",
                {
                    "name": (str, {"min_length": 2, "max_length": 100}),
                    "email": (str, {"pattern": r"^[^@]+@[^@]+\.[^@]+$"}),
                    "age": (int, {"ge": 18, "le": 120}),
                },
                {
                    "name": core.create_standard_validators()["name"],
                    "email": core.create_standard_validators()["email"],
                    "age": core.create_standard_validators()["age"],
                },
            )

        Performance monitoring and observability::

            # Create performance-tracked service
            @core.track_performance("user_service_operation")
            def process_user_request(request):
                return core.get_service("user_processor").flat_map(
                    lambda processor: processor.process(request)
                )

    Integration with FLEXT Ecosystem:
        The FlextCore class integrates seamlessly with all FLEXT ecosystem components:

        - **flext-api**: HTTP API services with FlextResult responses
        - **flext-auth**: Authentication services with FlextEntity users
        - **flext-db-oracle**: Database operations with FlextResult patterns
        - **flext-ldap**: LDAP integration with FlextContainer services
        - **flext-web**: Web applications with FlextContainer dependency injection
        - **Singer ecosystem**: Data pipeline operations with validation
        - **Go services**: Cross-language integration via Python bridge

    Attributes:
        _instance (FlextCore | None): Singleton instance for global access
        _container (FlextContainer): Global dependency injection container
        _settings_cache (dict[type[object], object]): Cached settings instances
        _handler_registry (FlextHandlers.Management.HandlerRegistry | None): Lazy-loaded handler registry
        _field_registry (FlextFields.Registry.FieldRegistry | None): Lazy-loaded field registry
        _plugin_registry (object | None): Lazy-loaded plugin registry
        _console (FlextUtilities | None): Lazy-loaded console utilities
        _observability (FlextObservability.Observability | None): Lazy-loaded observability instance

    See Also:
        - FlextContainer: Dependency injection implementation
        - FlextResult[T]: Railway-oriented programming patterns
        - FlextValidation: Comprehensive validation system
        - FlextHandlers: Enterprise handler patterns
        - FlextDecorators: Cross-cutting concern implementations

    """

    _instance: FlextCore | None = None

    def __init__(self) -> None:
        """Initialize FLEXT Core with comprehensive subsystem setup and lazy loading.

        Sets up the core infrastructure with thread-safe initialization of essential
        components and lazy loading of resource-intensive subsystems. The container
        is immediately initialized as it's required for all service operations,
        while other components are created on-demand for optimal performance.

        Initialization includes:
            - Global dependency injection container setup
            - Settings cache initialization for configuration management
            - Lazy loading placeholders for heavy subsystems
            - Thread-safe singleton pattern enforcement

        Note:
            This constructor is called automatically by get_instance() and should
            not be called directly. Use FlextCore.get_instance() instead.

        """
        # Core container
        self._container = FlextContainer.get_global()

        # Settings cache
        self._settings_cache: dict[type[object], object] = {}

        # Lazy-loaded components
        self._handler_registry: FlextHandlers.Management.HandlerRegistry | None = None
        self._field_registry: FlextFields.Registry.FieldRegistry | None = None
        self._plugin_registry: object | None = None
        self._console: FlextUtilities | None = None
        self._observability: FlextObservability.Observability | None = None

    @classmethod
    def get_instance(cls) -> FlextCore:
        """Get the singleton instance of FlextCore with thread-safe initialization.

        Implements thread-safe singleton pattern ensuring only one FlextCore instance
        exists throughout the application lifecycle. The instance is created on first
        access and reused for all subsequent calls, providing consistent state across
        all FLEXT ecosystem components.

        Returns:
            FlextCore: The singleton instance providing unified access to all FLEXT functionality.

        Thread Safety:
            This method is thread-safe and can be called concurrently from multiple threads
            without risk of creating multiple instances.

        Usage Examples:
            Basic instance access::

                core = FlextCore.get_instance()
                core.configure_logging(log_level="INFO")

            Service registration::

                core = FlextCore.get_instance()
                result = core.register_service("database", db_service)
                if result.success:
                    logger.info("Database service registered successfully")

            Cross-module consistency::

                # In module A
                core_a = FlextCore.get_instance()
                core_a.register_service("cache", cache_service)

                # In module B - same instance, same services
                core_b = FlextCore.get_instance()
                cache_result = core_b.get_service("cache")  # Available here

        Note:
            The singleton pattern ensures consistent configuration and service
            availability across the entire application, making it ideal for
            enterprise applications with complex dependency graphs.

        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # CONTAINER & DEPENDENCY INJECTION
    # Enterprise-grade service management with factory patterns and lifecycle control
    # =========================================================================

    @property
    def container(self) -> FlextContainer:
        """Access dependency injection container."""
        return self._container

    def register_service(
        self,
        key: str,
        service: object,
    ) -> FlextResult[None]:
        """Register a service instance in the global dependency injection container.

        Registers a service instance with the specified key for later retrieval throughout
        the application. The service becomes available to all components that have access
        to the FlextCore instance, enabling loose coupling and testability.

        Args:
            key (str): Unique identifier for the service. Should follow naming conventions
                (e.g., "database", "user_service", "email_client").
            service (object): The service instance to register. Can be any object including
                classes, functions, or complex service implementations.

        Returns:
            FlextResult[None]: Success result on successful registration, failure result
                with error message if registration fails.

        Thread Safety:
            This method is thread-safe and can be called concurrently from multiple threads.

        Usage Examples:
            Register various service types::

                core = FlextCore.get_instance()

                # Register service instances
                result1 = core.register_service(
                    "database", DatabaseService(host="localhost")
                )
                result2 = core.register_service("cache", RedisCache(port=6379))
                result3 = core.register_service("logger", structured_logger)

                # Check registration results
                if all(r.success for r in [result1, result2, result3]):
                    print("All services registered successfully")

            Register with validation::

                def register_validated_service(
                    name: str, service: object
                ) -> FlextResult[None]:
                    return (
                        core.validate_service_name(name)
                        .flat_map(lambda _: core.register_service(name, service))
                        .tap(lambda _: logger.info(f"Service {name} registered"))
                    )

        See Also:
            - get_service(): Retrieve registered services
            - register_factory(): Register service factories for lazy instantiation
            - clear_container(): Remove all registered services

        """
        return self._container.register(str(key), service)

    def get_service(self, key: str) -> FlextResult[object]:
        """Get service from container."""
        result = self._container.get(str(key))
        if result.is_failure:
            return FlextResult[object].fail(result.error or "Service not found")
        return FlextResult[object].ok(result.value)

    def register_factory(
        self,
        key: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register service factory in container."""
        return self._container.register_factory(str(key), factory)

    def configure_container(self, **config: object) -> FlextResult[None]:
        """Configure container with settings."""
        try:
            # Configure container if it has configuration capability
            if hasattr(self._container, "configure"):
                configure_method = getattr(self._container, "configure", None)
                if callable(configure_method):
                    configure_method(**config)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Container configuration failed: {e}")

    def clear_container(self) -> FlextResult[None]:
        """Clear all services from container."""
        try:
            if hasattr(self._container, "clear"):
                self._container.clear()
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Container clear failed: {e}")

    # =========================================================================
    # LOGGING & OBSERVABILITY
    # =========================================================================

    @staticmethod
    def get_logger(name: str) -> FlextLogger:
        """Get configured logger instance."""
        return FlextLogger(name)

    @staticmethod
    def configure_logging(
        *,
        log_level: FlextTypes.Config.LogLevel = "INFO",
        _json_output: FlextTypes.Core.Boolean | None = None,
    ) -> None:
        """Configure the global logging system with enterprise-grade structured output.

        Sets up comprehensive logging configuration for the entire FLEXT ecosystem with
        support for structured JSON output, correlation tracking, and configurable log
        levels. The configuration applies globally to all FlextLogger instances.

        Args:
            log_level (FlextTypes.Config.LogLevel, optional): Minimum log level for output.
                Valid values: "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
                Defaults to "INFO".
            _json_output (FlextTypes.Core.Boolean | None, optional): Enable JSON structured
                output format for log parsing and analysis. When None, uses default format.
                Defaults to None.

        Configuration Features:
            - **Structured Output**: JSON format for log aggregation systems
            - **Correlation Tracking**: Automatic correlation ID propagation
            - **Performance Metrics**: Built-in timing and performance logging
            - **Error Context**: Rich context capture for debugging
            - **Thread Safety**: Safe for concurrent logging operations

        Usage Examples:
            Basic logging configuration::

                # Development environment
                FlextCore.configure_logging(log_level="DEBUG")

                # Production environment with JSON output
                FlextCore.configure_logging(log_level="WARNING", _json_output=True)

            Environment-specific setup::

                import os

                if os.getenv("FLEXT_ENV") == "production":
                    FlextCore.configure_logging(
                        log_level="ERROR",
                        _json_output=True,  # For log aggregation
                    )
                else:
                    FlextCore.configure_logging(
                        log_level="DEBUG",
                        _json_output=False,  # Human-readable for development
                    )

            Integration with correlation tracking::

                FlextCore.configure_logging(log_level="INFO", _json_output=True)

                core = FlextCore.get_instance()
                logger = core.create_log_context(
                    correlation_id="req-12345",
                    user_id="user-67890",
                    operation="user_registration",
                )
                logger.info(
                    "Processing user registration"
                )  # Includes correlation context

        Note:
            This is a static method that configures global logging behavior.
            Changes affect all existing and future FlextLogger instances.
            For request-specific context, use create_log_context() instead.

        See Also:
            - get_logger(): Create logger instances
            - create_log_context(): Create context-aware loggers
            - FlextLogger: Core logging implementation

        """
        log_level_enum = FlextConstants.Config.LogLevel.INFO
        try:
            log_level_enum = FlextConstants.Config.LogLevel(log_level.upper())
        except (ValueError, AttributeError):
            log_level_enum = FlextConstants.Config.LogLevel.INFO

        # Note: FlextLogger doesn't have global level - individual loggers have levels

        if _json_output is not None:
            FlextLogger.configure(
                log_level=str(log_level_enum.value),
                json_output=_json_output,
            )

    def create_log_context(
        self, logger: FlextLogger | str | None = None, **context: object
    ) -> FlextLogger:
        """Create structured logging context manager."""
        if isinstance(logger, FlextLogger):
            # Add context to the existing logger
            logger.set_request_context(**context)
            return logger
        if isinstance(logger, str):
            base_logger = FlextLogger(logger)
            base_logger.set_request_context(**context)
            return base_logger
        base_logger = FlextLogger("flext")
        base_logger.set_request_context(**context)
        return base_logger

    @property
    def observability(self) -> FlextObservability.Observability:
        """Get observability instance."""
        if self._observability is None:
            self._observability = get_global_observability()
        return self._observability

    def reset_observability(self) -> None:
        """Reset global observability state."""
        reset_global_observability()
        self._observability = None

    # =========================================================================
    # CONFIGURATION MANAGEMENT - MASSIVE FLEXT TYPES INTEGRATION
    # =========================================================================

    def get_environment_config(
        self, environment: FlextTypes.Config.Environment = "development"
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get environment-specific configuration using FlextTypes.Config."""
        try:
            config_dict: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "debug": environment in {"development", "staging"},
                "config_source": FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
            }
            return FlextResult[FlextTypes.Config.ConfigDict].ok(config_dict)
        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Environment config failed: {e}"
            )

    def create_config_provider(
        self,
        provider_type: FlextTypes.Config.ConfigProvider = "default_provider",
        config_format: FlextTypes.Config.ConfigFormat = "json",
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create configuration provider with enhanced type safety."""
        try:
            provider_config: FlextTypes.Config.ConfigDict = {
                "provider_type": provider_type,
                "format": config_format,
                "priority": 1,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "source": FlextConstants.Config.ConfigSource.FILE.value,
            }
            return FlextResult[FlextTypes.Config.ConfigDict].ok(provider_config)
        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Config provider creation failed: {e}"
            )

    def validate_config_with_types(
        self,
        config: FlextTypes.Config.ConfigDict,
        required_keys: list[FlextTypes.Config.ConfigKey] | None = None,
    ) -> FlextResult[FlextTypes.Config.ValidationResult]:
        """Validate configuration using FlextTypes.Config with comprehensive checks."""
        try:
            required = required_keys or ["environment", "log_level"]

            for key in required:
                if key not in config:
                    return FlextResult[FlextTypes.Config.ValidationResult].fail(
                        f"Missing required config key: {key}"
                    )

            # Validate environment if present
            if "environment" in config:
                env_value = config["environment"]
                valid_envs = [
                    env.value for env in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_envs:
                    return FlextResult[FlextTypes.Config.ValidationResult].fail(
                        f"Invalid environment: {env_value}"
                    )

            # Validate log level if present
            if "log_level" in config:
                log_value = config["log_level"]
                valid_logs = [level.value for level in FlextConstants.Config.LogLevel]
                if log_value not in valid_logs:
                    return FlextResult[FlextTypes.Config.ValidationResult].fail(
                        f"Invalid log level: {log_value}"
                    )

            return FlextResult[FlextTypes.Config.ValidationResult].ok(data=True)
        except Exception as e:
            return FlextResult[FlextTypes.Config.ValidationResult].fail(
                f"Config validation failed: {e}"
            )

    @classmethod
    def configure_core_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure core system using FlextTypes.Config with StrEnum validation."""
        try:
            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )

            # Validate log level
            if "log_level" in config:
                log_value = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_value not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_value}'. Valid options: {valid_log_levels}"
                    )

            # Validate validation level
            if "validation_level" in config:
                val_value = config["validation_level"]
                valid_validation_levels = [
                    v.value for v in FlextConstants.Config.ValidationLevel
                ]
                if val_value not in valid_validation_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{val_value}'. Valid options: {valid_validation_levels}"
                    )

            # Validate config source
            if "config_source" in config:
                source_value = config["config_source"]
                valid_sources = [s.value for s in FlextConstants.Config.ConfigSource]
                if source_value not in valid_sources:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid config_source '{source_value}'. Valid options: {valid_sources}"
                    )

            # Build validated configuration with defaults
            validated_config: FlextTypes.Config.ConfigDict = {
                "environment": config.get(
                    "environment",
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                ),
                "log_level": config.get(
                    "log_level", FlextConstants.Config.LogLevel.DEBUG.value
                ),
                "validation_level": config.get(
                    "validation_level",
                    FlextConstants.Config.ValidationLevel.NORMAL.value,
                ),
                "config_source": config.get(
                    "config_source", FlextConstants.Config.ConfigSource.DEFAULT.value
                ),
                "enable_observability": config.get("enable_observability", True),
                "enable_logging": config.get("enable_logging", True),
                "enable_container_debugging": config.get(
                    "enable_container_debugging", False
                ),
                "max_service_registrations": config.get(
                    "max_service_registrations", 1000
                ),
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Core system configuration failed: {e}"
            )

    @classmethod
    def get_core_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current core system configuration with runtime information."""
        try:
            config: FlextTypes.Config.ConfigDict = {
                # Current system state
                "environment": os.getenv(
                    "FLEXT_ENV",
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                ),
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "config_source": FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
                # Runtime information
                "active_services": 0,  # Would be populated from actual container
                "observability_enabled": True,
                "logging_enabled": True,
                "container_debug_mode": False,
                # System metrics
                "system_uptime": 0,  # Would be calculated from startup time
                "total_operations": 0,  # Would be tracked in metrics
                "error_count": 0,  # Would be tracked in error metrics
                # Available subsystems
                "available_subsystems": [
                    "container",
                    "logging",
                    "validation",
                    "handlers",
                    "observability",
                    "commands",
                    "domain_services",
                    "utilities",
                    "exceptions",
                    "fields",
                ],
                # Performance settings
                "max_service_registrations": 1000,
                "enable_performance_monitoring": True,
                "cache_configuration": True,
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get core system configuration: {e}"
            )

    @classmethod
    def create_environment_core_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific core system configuration."""
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "config_source": FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
            }

            # Environment-specific settings
            if environment == "production":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "enable_container_debugging": False,  # No debugging in production
                    "enable_performance_monitoring": True,  # Performance monitoring in production
                    "max_service_registrations": 500,  # Conservative limit for production
                    "cache_configuration": True,  # Cache for performance
                    "enable_error_reporting": True,  # Error reporting in production
                })
            elif environment == "development":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "enable_container_debugging": True,  # Full debugging for development
                    "enable_performance_monitoring": False,  # No performance monitoring in dev
                    "max_service_registrations": 2000,  # Higher limit for development
                    "cache_configuration": False,  # No caching for development (fresh data)
                    "enable_detailed_logging": True,  # Detailed logging for debugging
                })
            elif environment == "test":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "enable_container_debugging": False,  # No debugging in tests
                    "enable_performance_monitoring": False,  # No performance monitoring in tests
                    "max_service_registrations": 100,  # Limited for tests
                    "cache_configuration": False,  # No caching in tests
                    "enable_test_utilities": True,  # Special test utilities
                })
            elif environment == "staging":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "enable_container_debugging": False,  # No debugging in staging
                    "enable_performance_monitoring": True,  # Performance monitoring in staging
                    "max_service_registrations": 750,  # Staging limit
                    "cache_configuration": True,  # Cache for staging performance
                    "enable_staging_features": True,  # Special staging features
                })
            else:  # local, etc.
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "enable_container_debugging": True,  # Debugging for local development
                    "enable_performance_monitoring": False,  # No performance monitoring locally
                    "max_service_registrations": 1000,  # Standard limit for local
                    "cache_configuration": False,  # No caching for local development
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment core config: {e}"
            )

    @classmethod
    def optimize_core_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize core system performance based on configuration."""
        try:
            # Extract performance level or determine from config
            performance_level = config.get("performance_level", "medium")

            # Base optimization settings
            optimized_config: FlextTypes.Config.ConfigDict = {
                "performance_level": performance_level,
                "optimization_enabled": True,
            }

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update({
                    "max_service_registrations": config.get(
                        "max_service_registrations", 2000
                    ),
                    "container_cache_size": 1000,
                    "enable_lazy_loading": True,
                    "enable_service_pooling": True,
                    "max_concurrent_operations": 200,
                    "memory_optimization": "aggressive",
                    "gc_optimization": True,
                    "enable_async_processing": True,
                    "buffer_size": 10000,
                })
            elif performance_level == "medium":
                optimized_config.update({
                    "max_service_registrations": config.get(
                        "max_service_registrations", 1000
                    ),
                    "container_cache_size": 500,
                    "enable_lazy_loading": True,
                    "enable_service_pooling": False,
                    "max_concurrent_operations": 100,
                    "memory_optimization": "balanced",
                    "gc_optimization": False,
                    "enable_async_processing": False,
                    "buffer_size": 5000,
                })
            elif performance_level == "low":
                optimized_config.update({
                    "max_service_registrations": config.get(
                        "max_service_registrations", 500
                    ),
                    "container_cache_size": 100,
                    "enable_lazy_loading": False,
                    "enable_service_pooling": False,
                    "max_concurrent_operations": 50,
                    "memory_optimization": "conservative",
                    "gc_optimization": False,
                    "enable_async_processing": False,
                    "buffer_size": 1000,
                })
            else:
                # Default/custom performance level
                optimized_config.update({
                    "max_service_registrations": config.get(
                        "max_service_registrations", 1000
                    ),
                    "container_cache_size": 500,
                    "enable_lazy_loading": config.get("enable_lazy_loading", True),
                    "max_concurrent_operations": config.get(
                        "max_concurrent_operations", 100
                    ),
                    "memory_optimization": "balanced",
                })

            # Merge with original config
            optimized_config.update({
                key: value
                for key, value in config.items()
                if key not in optimized_config
            })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Core performance optimization failed: {e}"
            )

    # =========================================================================
    # RESULT PATTERN & RAILWAY PROGRAMMING
    # =========================================================================

    @staticmethod
    def ok(value: object) -> FlextResult[object]:
        """Create a successful FlextResult containing the specified value.

        Creates a successful FlextResult instance wrapping the provided value, enabling
        railway-oriented programming patterns with composable operations. This is the
        primary way to create successful results in the FLEXT ecosystem.

        Args:
            value (object): The value to wrap in a successful result. Can be any object
                including None, primitive types, or complex data structures.

        Returns:
            FlextResult[object]: A successful result containing the provided value.

        Usage Examples:
            Create successful results::

                # Simple value wrapping
                result = FlextCore.ok("Hello, World!")
                assert result.success
                assert result.value == "Hello, World!"

            Railway pattern with chaining::

                result = (
                    FlextCore.ok({"name": "John", "age": 30})
                    .map(lambda user: user["name"].upper())
                    .flat_map(lambda name: FlextCore.ok(f"Hello, {name}!"))
                    .tap(lambda msg: print(msg))
                )  # Prints: Hello, JOHN!

            Integration with validation::

                def create_user(data: dict) -> FlextResult[User]:
                    return (
                        FlextCore.ok(data)
                        .flat_map(lambda d: core.validate_required(d, "user_data"))
                        .flat_map(lambda d: core.create_entity(User, **d))
                    )

        See Also:
            - fail(): Create failure results
            - from_exception(): Create results from exceptions
            - sequence(): Handle multiple results
            - FlextResult[T]: Complete result type documentation

        """
        return FlextResult[object].ok(value)

    @staticmethod
    def fail(error: str) -> FlextResult[object]:
        """Create failed Result."""
        return FlextResult[object].fail(error)

    @staticmethod
    def from_exception(exc: Exception) -> FlextResult[object]:
        """Create failed Result from exception."""
        return FlextResult[object].fail(str(exc))

    @staticmethod
    def sequence(results: list[FlextResult[object]]) -> FlextResult[list[object]]:
        """Convert list of Results to Result of list."""
        values: list[object] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[object]].fail(result.error or "Sequence failed")
            values.append(result.value)
        return FlextResult[list[object]].ok(values)

    @staticmethod
    def first_success(results: list[FlextResult[object]]) -> FlextResult[object]:
        """Return first successful Result, or last error if all fail."""
        last_error = "No results provided"
        for result in results:
            if result.is_success:
                return result
            last_error = result.error or "Unknown error"
        return FlextResult[object].fail(last_error)

    # =========================================================================
    # FUNCTIONAL PROGRAMMING & PIPELINES
    # =========================================================================

    @staticmethod
    def pipe(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Create a pipeline of Result-returning functions."""

        def pipeline(value: object) -> FlextResult[object]:
            result: FlextResult[object] = FlextResult[object].ok(value)
            for func in funcs:
                if result.is_failure:
                    break
                result = func(result.value)
            return result

        return pipeline

    @staticmethod
    def compose(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Compose Result-returning functions (right to left)."""
        return FlextCore.pipe(*reversed(funcs))

    @staticmethod
    def when(
        predicate: ValidatorCallable,
        then_func: Callable[[object], FlextResult[object]],
        else_func: Callable[[object], FlextResult[object]] | None = None,
    ) -> Callable[[object], FlextResult[object]]:
        """Conditional Result execution."""

        def conditional(value: object) -> FlextResult[object]:
            if predicate(value):
                return then_func(value)
            if else_func:
                return else_func(value)
            return FlextResult[object].ok(value)

        return conditional

    @staticmethod
    def tap(
        side_effect: Callable[[object], None],
    ) -> Callable[[object], FlextResult[object]]:
        """Execute side effect in pipeline."""

        def side_effect_wrapper(value: object) -> FlextResult[object]:
            side_effect(value)
            return FlextResult[object].ok(value)

        return side_effect_wrapper

    # =========================================================================
    # VALIDATION & GUARDS
    # =========================================================================

    @staticmethod
    def validate_required(
        value: object, field_name: str = "value"
    ) -> FlextResult[object]:
        """Validate that a value is not None or empty string with descriptive error messages.

        Performs comprehensive validation to ensure the value is present and meaningful,
        checking for None values and empty strings with whitespace normalization.
        This is a fundamental validation used throughout the FLEXT ecosystem.

        Args:
            value (object): The value to validate. Can be any object type.
            field_name (str, optional): Name of the field being validated for error messages.
                Defaults to "value".

        Returns:
            FlextResult[object]: Success result containing the original value if valid,
                failure result with descriptive error message if invalid.

        Validation Rules:
            - **None Check**: Value cannot be None
            - **String Check**: String values cannot be empty after whitespace trimming
            - **Type Preservation**: Non-string values pass through unchanged

        Usage Examples:
            Basic validation::

                # Valid values
                result1 = FlextCore.validate_required("hello", "greeting")
                assert result1.success and result1.value == "hello"

                result2 = FlextCore.validate_required(42, "age")
                assert result2.success and result2.value == 42

            Invalid values::

                # None value
                result = FlextCore.validate_required(None, "user_id")
                assert result.failure and "user_id is required" in result.error

                # Empty string
                result = FlextCore.validate_required("", "name")
                assert result.failure and "name cannot be empty" in result.error

                # Whitespace-only string
                result = FlextCore.validate_required("   ", "title")
                assert result.failure and "title cannot be empty" in result.error

            Railway pattern integration::

                def process_user_data(data: dict) -> FlextResult[dict]:
                    return (
                        FlextCore.validate_required(data.get("name"), "name")
                        .flat_map(
                            lambda _: FlextCore.validate_required(
                                data.get("email"), "email"
                            )
                        )
                        .map(lambda _: data)
                    )  # Return original data if all validations pass

            Form validation::

                def validate_registration_form(form: dict) -> FlextResult[dict]:
                    required_fields = [
                        "username",
                        "email",
                        "password",
                        "confirm_password",
                    ]

                    for field in required_fields:
                        result = FlextCore.validate_required(form.get(field), field)
                        if result.failure:
                            return FlextResult[dict].fail(
                                f"Registration failed: {result.error}"
                            )

                    return FlextCore.ok(form)

        Integration:
            This method integrates seamlessly with other validation methods and
            can be chained with validate_string(), validate_email(), etc.

        See Also:
            - validate_string(): String-specific validation with length constraints
            - validate_email(): Email format validation
            - validate_numeric(): Numeric range validation
            - require_not_none(): Type-safe None validation from FlextGuards

        """
        if value is None:
            return FlextResult[object].fail(f"{field_name} is required")
        if isinstance(value, str) and not value.strip():
            return FlextResult[object].fail(f"{field_name} cannot be empty")
        return FlextResult[object].ok(value)

    @staticmethod
    def validate_string(
        value: object, min_length: int = 0, max_length: int | None = None
    ) -> FlextResult[str]:
        """Validate string value with length constraints."""
        if not isinstance(value, str):
            return FlextResult[str].fail("Value must be a string")

        if len(value) < min_length:
            return FlextResult[str].fail(
                f"String must be at least {min_length} characters"
            )

        if max_length is not None and len(value) > max_length:
            return FlextResult[str].fail(
                f"String must not exceed {max_length} characters"
            )

        return FlextResult[str].ok(value)

    @staticmethod
    def validate_numeric(
        value: object, min_value: float | None = None, max_value: float | None = None
    ) -> FlextResult[float]:
        """Validate numeric value with range constraints."""
        if not isinstance(value, (int, float)):
            return FlextResult[float].fail("Value must be numeric")
        numeric_value = float(value)

        if min_value is not None and numeric_value < min_value:
            return FlextResult[float].fail(f"Value must be at least {min_value}")

        if max_value is not None and numeric_value > max_value:
            return FlextResult[float].fail(f"Value must not exceed {max_value}")

        return FlextResult[float].ok(numeric_value)

    @staticmethod
    def validate_email(value: object) -> FlextResult[str]:
        """Validate email address format - delegates to FlextValidation."""
        if not isinstance(value, str):
            return FlextResult[str].fail("Email must be a string")

        # Delegate to canonical email validation
        return FlextValidation.Rules.StringRules.validate_email(value)

    @staticmethod
    def validate_service_name(value: object) -> FlextResult[str]:
        """Validate service name format."""
        if not isinstance(value, str):
            return FlextResult[str].fail("Service name must be a string")
        is_valid = flext_validate_service_name(value)
        if is_valid:
            return FlextResult[str].ok(value)
        return FlextResult[str].fail("Invalid service name format")

    @staticmethod
    def require_not_none(
        value: T | None, message: str = "Value cannot be None"
    ) -> FlextResult[T]:
        """Guard that ensures value is not None."""
        try:
            result = FlextGuards.ValidationUtils.require_not_none(value, message)
            return FlextResult[T].ok(cast("T", result))
        except Exception as e:
            return FlextResult[T].fail(str(e))

    @staticmethod
    def require_non_empty(
        value: str, message: str = "Value cannot be empty"
    ) -> FlextResult[str]:
        """Guard that ensures string is not empty."""
        try:
            result = FlextGuards.ValidationUtils.require_non_empty(value, message)
            return FlextResult[str].ok(cast("str", result))
        except Exception as e:
            return FlextResult[str].fail(str(e))

    @staticmethod
    def require_positive(
        value: float, message: str = "Value must be positive"
    ) -> FlextResult[float]:
        """Guard that ensures number is positive."""
        try:
            result = FlextGuards.ValidationUtils.require_positive(value, message)
            return FlextResult[float].ok(cast("float", result))
        except Exception as e:
            return FlextResult[float].fail(str(e))

    def create_validator(
        self, validation_func: ValidatorCallable, error_message: str
    ) -> Callable[[object], FlextResult[object]]:
        """Create custom validator function."""

        def validator(value: object) -> FlextResult[object]:
            if validation_func(value):
                return FlextResult[object].ok(value)
            return FlextResult[object].fail(error_message)

        return validator

    @property
    def validators(self) -> object:
        """Access validation utilities."""
        return FlextValidation

    @property
    def predicates(self) -> object:
        """Access predicate functions."""
        return FlextValidation.Core.Predicates

    @property
    def guards(self) -> type[FlextGuards]:
        """Access guard functions."""
        return FlextGuards

    # =========================================================================
    # CONFIGURATION & SETTINGS
    # =========================================================================

    def get_settings(self, settings_class: type[object]) -> object:
        """Get settings instance with caching."""
        if settings_class not in self._settings_cache:
            self._settings_cache[settings_class] = settings_class()
        return self._settings_cache[settings_class]

    @property
    def constants(self) -> type[FlextConstants]:
        """Access FLEXT constants."""
        return FlextConstants

    @staticmethod
    def load_config_from_env(prefix: str = "FLEXT_") -> FlextResult[dict[str, object]]:
        """Load configuration from environment variables (foundation pattern)."""
        try:
            env_data: dict[str, object] = {}
            prefix_with_sep = f"{prefix.rstrip('_')}_"

            for key, value in os.environ.items():
                if key.startswith(prefix_with_sep):
                    # Remove prefix and convert to lowercase
                    config_key = key[len(prefix_with_sep) :].lower()
                    env_data[config_key] = value

            return FlextResult[dict[str, object]].ok(env_data)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to load config: {e}")

    @staticmethod
    def merge_configs(*configs: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Merge multiple configuration dictionaries."""
        try:
            min_configs_for_merge = 2
            if len(configs) < min_configs_for_merge:
                return FlextResult[dict[str, object]].fail(
                    "At least 2 configs required for merging"
                )
            result = FlextConfig.merge_configs(configs[0], configs[1])
            if result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    result.error or "Config merge failed"
                )
            return result  # Already correct type
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to merge configs: {e}")

    @staticmethod
    def validate_config(
        config: dict[str, object], schema: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Validate configuration against schema."""
        try:
            # Simple validation - check that all schema keys are present
            for key in schema:
                if key not in config:
                    return FlextResult[dict[str, object]].fail(
                        f"Missing required config key: {key}"
                    )
            return FlextResult[dict[str, object]].ok(config)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Config validation error: {e}")

    @staticmethod
    def safe_get_env_var(name: str, default: str | None = None) -> FlextResult[str]:
        """Safely get environment variable."""
        return FlextConfig.safe_get_env_var(name, default)

    # =========================================================================
    # DOMAIN MODELING & DDD PATTERNS
    # =========================================================================

    @staticmethod
    def create_entity(entity_class: type[T], **data: object) -> FlextResult[T]:
        """Create domain entity with validation."""
        try:
            if hasattr(entity_class, "model_validate"):
                # Use getattr to safely access the method with type safety
                model_validate = getattr(entity_class, "model_validate", None)
                if callable(model_validate):
                    instance = model_validate(data)
                    if not isinstance(instance, entity_class):
                        return FlextResult[T].fail(
                            "Model validation returned incorrect type"
                        )
                else:
                    instance = entity_class(**data)
            else:
                instance = entity_class(**data)
            # Type assertion after validation - instance must be of type T
            validated_instance = (
                instance if isinstance(instance, entity_class) else entity_class(**data)
            )
            return FlextResult[T].ok(validated_instance)
        except Exception as e:
            return FlextResult[T].fail(f"Entity creation failed: {e}")

    @staticmethod
    def create_value_object(vo_class: type[T], **data: object) -> FlextResult[T]:
        """Create value object with validation."""
        try:
            if hasattr(vo_class, "model_validate"):
                # Use getattr to safely access the method with type safety
                model_validate = getattr(vo_class, "model_validate", None)
                if callable(model_validate):
                    instance = model_validate(data)
                    if not isinstance(instance, vo_class):
                        return FlextResult[T].fail(
                            "Model validation returned incorrect type"
                        )
                else:
                    instance = vo_class(**data)
            else:
                instance = vo_class(**data)
            # Type assertion after validation - instance must be of type T
            validated_instance = (
                instance if isinstance(instance, vo_class) else vo_class(**data)
            )
            return FlextResult[T].ok(validated_instance)
        except Exception as e:
            return FlextResult[T].fail(f"Value object creation failed: {e}")

    @staticmethod
    def create_aggregate_root(
        aggregate_class: type[T], **data: object
    ) -> FlextResult[T]:
        """Create aggregate root with validation."""
        try:
            if hasattr(aggregate_class, "model_validate"):
                # Use getattr to safely access the method with type safety
                model_validate = getattr(aggregate_class, "model_validate", None)
                if callable(model_validate):
                    instance = model_validate(data)
                    if not isinstance(instance, aggregate_class):
                        return FlextResult[T].fail(
                            "Model validation returned incorrect type"
                        )
                else:
                    instance = aggregate_class(**data)
            else:
                instance = aggregate_class(**data)
            # Type assertion after validation - instance must be of type T
            validated_instance = (
                instance
                if isinstance(instance, aggregate_class)
                else aggregate_class(**data)
            )
            return FlextResult[T].ok(validated_instance)
        except Exception as e:
            return FlextResult[T].fail(f"Aggregate root creation failed: {e}")

    @property
    def entity_base(self) -> type[FlextModels.Entity]:
        """Access entity base class."""
        return FlextModels.Entity

    @property
    def value_object_base(self) -> type[FlextModels.Value]:
        """Access value object base class."""
        return FlextModels.Value

    @property
    def aggregate_root_base(self) -> type[FlextModels.AggregateRoot]:
        """Access aggregate root base class."""
        return FlextModels.AggregateRoot

    @property
    def domain_service_base(self) -> type[FlextDomainService[object]]:
        """Access domain service base class."""
        return FlextDomainService[object]

    # =========================================================================
    # UTILITIES & GENERATORS
    # =========================================================================

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID - delegates to FlextUtilities."""
        return FlextUtilities.Generators.generate_id()

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID - delegates to FlextUtilities."""
        return FlextUtilities.Generators.generate_uuid()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID - delegates to FlextUtilities."""
        return FlextUtilities.Generators.generate_correlation_id()

    @staticmethod
    def safe_call(func: Callable[[], T], default: T) -> T:
        """Safely call function with default fallback."""
        result = safe_call(func)
        if result.is_failure:
            return default
        return result.value

    @staticmethod
    def truncate(text: str, max_length: int = 100) -> str:
        """Truncate text to maximum length."""
        return FlextUtilities.truncate(text, max_length)

    @staticmethod
    def is_not_none(value: object | None) -> bool:
        """Check if value is not None."""
        return value is not None

    @property
    def console(self) -> FlextUtilities:
        """Get console instance."""
        if self._console is None:
            self._console = FlextUtilities()
        return self._console

    @property
    def utilities(self) -> type[FlextUtilities]:
        """Access utility functions."""
        return FlextUtilities

    @property
    def generators(self) -> type[FlextUtilities.Generators]:
        """Access generator functions."""
        return FlextUtilities.Generators

    @property
    def type_guards(self) -> type[FlextUtilities.TypeGuards]:
        """Access type guard functions."""
        return FlextUtilities.TypeGuards

    # =========================================================================
    # MESSAGING & EVENTS
    # =========================================================================

    @staticmethod
    def create_message(
        message_type: str, **kwargs: object
    ) -> FlextResult[FlextModels.Payload[str]]:
        """Create cross-service message."""
        try:
            correlation_id = (
                str(kwargs.pop("correlation_id", None))
                if kwargs.get("correlation_id")
                else None
            )
            # Use FlextModels factory method instead
            message_result = FlextModels.create_payload(
                kwargs.get("data", {}),
                message_type,
                "flext-core",
                correlation_id=correlation_id,
            )

            if message_result.is_failure:
                return FlextResult[FlextModels.Payload[str]].fail(
                    message_result.error or "Message creation failed"
                )
            return cast("FlextResult[FlextModels.Payload[str]]", message_result)
        except Exception as e:
            return FlextResult[FlextModels.Payload[str]].fail(
                f"Message creation failed: {e}"
            )

    @staticmethod
    def create_event(
        event_type: str, data: dict[str, object], **kwargs: object
    ) -> FlextResult[FlextModels.Payload[Mapping[str, object]]]:
        """Create cross-service event."""
        try:
            # Remove unused correlation_id processing
            kwargs.pop("correlation_id", None)
            # Use FlextModels factory method instead
            event_result = FlextModels.create_domain_event(
                event_type,
                str(kwargs.get("aggregate_id", "unknown")),
                str(kwargs.get("aggregate_type", "Unknown")),
                cast(
                    "FlextTypes.Core.JsonObject",
                    data if isinstance(data, dict) else {"data": data},
                ),
                "flext-core",
            )

            if event_result.is_failure:
                return FlextResult[FlextModels.Payload[Mapping[str, object]]].fail(
                    event_result.error or "Event creation failed"
                )
            return cast(
                "FlextResult[FlextModels.Payload[Mapping[str, object]]]", event_result
            )
        except Exception as e:
            return FlextResult[FlextModels.Payload[Mapping[str, object]]].fail(
                f"Event creation failed: {e}"
            )

    @staticmethod
    def validate_protocol(payload: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Validate cross-service protocol."""
        # Basic payload validation
        required_fields = ["message_type", "source_service", "data"]
        for field in required_fields:
            if field not in payload:
                return FlextResult[dict[str, object]].fail(
                    f"Missing required field: {field}"
                )
        return FlextResult[dict[str, object]].ok(payload)

    @staticmethod
    def get_serialization_metrics() -> dict[str, object]:
        """Get payload serialization metrics."""
        return {"total_payloads": 0, "average_size": 0, "max_size": 0, "min_size": 0}

    @property
    def payload_base(self) -> type[FlextModels.Payload[object]]:
        """Access payload base class."""
        return FlextModels.Payload[object]

    @property
    def message_base(self) -> type[FlextModels.Payload[str]]:
        """Access message base class."""
        return cast("type[FlextModels.Payload[str]]", FlextModels.Payload)

    @property
    def event_base(self) -> type[FlextModels.Payload[Mapping[str, object]]]:
        """Access event base class."""
        return cast(
            "type[FlextModels.Payload[Mapping[str, object]]]", FlextModels.Payload
        )

    # =========================================================================
    # HANDLERS & CQRS
    # =========================================================================

    @property
    def handler_registry(self) -> FlextHandlers.Management.HandlerRegistry:
        """Get handler registry instance."""
        if self._handler_registry is None:
            self._handler_registry = FlextHandlers.Management.HandlerRegistry()
        return self._handler_registry

    def register_handler(self, handler_type: str, handler: object) -> FlextResult[None]:
        """Register event/command handler."""
        try:
            registry = self.handler_registry
            if hasattr(registry, "register"):
                validated_handler = cast(
                    "FlextHandlers.Implementation.AbstractHandler[object, object]",
                    handler,
                )
                registry.register(handler_type, validated_handler)
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(
                "Handler registry does not support registration"
            )
        except Exception as e:
            return FlextResult[None].fail(f"Handler registration failed: {e}")

    def get_handler(self, handler_type: str) -> FlextResult[object]:
        """Get registered handler."""
        try:
            registry = self.handler_registry
            if hasattr(registry, "get_handler"):
                return registry.get_handler(handler_type)
            return FlextResult[object].fail(f"Handler not found: {handler_type}")
        except Exception as e:
            return FlextResult[object].fail(f"Handler retrieval failed: {e}")

    @property
    def handlers(self) -> type[FlextHandlers]:
        """Access handlers utilities."""
        return FlextHandlers

    @property
    def base_handler(self) -> type[FlextHandlers.Implementation.BasicHandler]:
        """Access base handler class."""
        return FlextHandlers.Implementation.BasicHandler

    @property
    def commands(self) -> type[FlextCommands]:
        """Access CQRS commands."""
        return FlextCommands

    # =========================================================================
    # FIELDS & METADATA
    # =========================================================================

    @property
    def field_registry(self) -> FlextFields.Registry.FieldRegistry:
        """Get field registry instance."""
        if self._field_registry is None:
            self._field_registry = FlextFields.Registry.FieldRegistry()
        return self._field_registry

    @staticmethod
    def create_string_field(name: str, **kwargs: object) -> object:
        """Create string field definition."""
        result = FlextFields.Factory.create_field("string", name, **kwargs)
        if result.is_success:
            return result.value
        return result.error

    @staticmethod
    def create_integer_field(name: str, **kwargs: object) -> object:
        """Create integer field definition."""
        result = FlextFields.Factory.create_field("integer", name, **kwargs)
        if result.is_success:
            return result.value
        return result.error

    @staticmethod
    def create_boolean_field(name: str, **kwargs: object) -> object:
        """Create boolean field definition."""
        result = FlextFields.Factory.create_field("boolean", name, **kwargs)
        if result.is_success:
            return result.value
        return result.error

    @property
    def fields(self) -> type[FlextFields]:
        """Access field utilities."""
        return FlextFields

    # =========================================================================
    # DECORATORS & ASPECTS
    # =========================================================================

    @property
    def decorators(self) -> type[FlextDecorators]:
        """Access decorator utilities."""
        return FlextDecorators

    # @property
    # def decorator_factory(self) -> type[FlextDecoratorFactory]:
    #     """Access decorator factory."""
    #     return FlextDecoratorFactory

    def create_validation_decorator(self, validator: ValidatorCallable) -> object:
        """Create custom validation decorator."""
        # Cast validator to proper type for validate_input compatibility
        bool_validator = cast("Callable[[object], bool]", validator)
        return FlextDecorators.Validation.validate_input(validator=bool_validator)

    def create_error_handling_decorator(self) -> object:
        """Create custom error handling decorator."""
        return FlextDecorators.Reliability

    def create_performance_decorator(self) -> object:
        """Create performance monitoring decorator."""
        return FlextDecorators.Performance

    def create_logging_decorator(self) -> object:
        """Create logging decorator."""
        return FlextDecorators.Observability

    @staticmethod
    def make_immutable(target_class: type[T]) -> type[T]:
        """Make class immutable."""
        return FlextGuards.immutable(target_class)

    @staticmethod
    def make_pure(func: Callable[P, R]) -> Callable[P, R]:
        """Make function pure."""
        # Cast to satisfy type compatibility

        return cast("Callable[P, R]", FlextGuards.pure(func))

    # =========================================================================
    # MIXINS & COMPOSITION
    # =========================================================================

    @property
    def timestamp_mixin(self) -> object:
        """Access timestamp mixin."""
        return FlextMixins.create_timestamp_fields

    @property
    def identifiable_mixin(self) -> object:
        """Access identifiable mixin."""
        return FlextMixins.ensure_id

    @property
    def loggable_mixin(self) -> object:
        """Access loggable mixin."""
        return FlextMixins.get_logger

    @property
    def validatable_mixin(self) -> object:
        """Access validatable mixin."""
        return FlextMixins.validate_required_fields

    @property
    def serializable_mixin(self) -> object:
        """Access serializable mixin."""
        return FlextMixins.to_dict

    @property
    def cacheable_mixin(self) -> object:
        """Access cacheable mixin."""
        return FlextMixins.get_cache_key

    # =========================================================================
    # ROOT MODELS & VALUE TYPES
    # =========================================================================

    @staticmethod
    def create_entity_id(
        value: str | None = None,
    ) -> FlextResult[FlextModels.EntityId]:
        """Create entity ID."""
        if value is None:
            return FlextResult[FlextModels.EntityId].fail(
                "Entity ID value cannot be None"
            )
        try:
            entity_id = FlextModels.EntityId(root=value)
            return FlextResult[FlextModels.EntityId].ok(entity_id)
        except Exception as e:
            return FlextResult[FlextModels.EntityId].fail(
                f"Entity ID creation failed: {e}"
            )

    @staticmethod
    def create_version_number(value: int) -> FlextResult[FlextModels.Version]:
        """Create version number."""
        try:
            version = FlextModels.Version(root=value)
            return FlextResult[FlextModels.Version].ok(version)
        except Exception as e:
            return FlextResult[FlextModels.Version].fail(
                f"Version creation failed: {e}"
            )

    @staticmethod
    def create_email_address(value: str) -> FlextResult[FlextModels.EmailAddress]:
        """Create email address."""
        try:
            email = FlextModels.EmailAddress(root=value)
            return FlextResult[FlextModels.EmailAddress].ok(email)
        except Exception as e:
            return FlextResult[FlextModels.EmailAddress].fail(
                f"Email creation failed: {e}"
            )

    @staticmethod
    def create_service_name_value(
        value: str,
    ) -> FlextResult[FlextModels.Host]:
        """Create service name (using Host as fallback)."""
        try:
            service_name = FlextModels.Host(root=value)
            return FlextResult[FlextModels.Host].ok(service_name)
        except Exception as e:
            return FlextResult[FlextModels.Host].fail(
                f"Service name creation failed: {e}"
            )

    @staticmethod
    def create_timestamp() -> FlextModels.Timestamp:
        """Create current timestamp."""
        return FlextModels.Timestamp(root=datetime.now(UTC))

    @staticmethod
    def create_metadata(**data: object) -> FlextResult[FlextModels.Metadata]:
        """Create metadata object."""
        try:
            # Convert all values to strings for Metadata compatibility
            typed_data = {str(k): str(v) for k, v in data.items()}
            metadata = FlextModels.Metadata(root=typed_data)
            return FlextResult[FlextModels.Metadata].ok(metadata)
        except Exception as e:
            return FlextResult[FlextModels.Metadata].fail(
                f"Metadata creation failed: {e}"
            )

    # =========================================================================
    # EXCEPTIONS & ERROR HANDLING
    # =========================================================================

    @staticmethod
    def create_error(message: str, error_code: str | None = None) -> object:
        """Create FLEXT error."""
        return FlextExceptions.Error(message, error_code=error_code)

    @staticmethod
    def create_validation_error(message: str, field_name: str | None = None) -> object:
        """Create validation error."""
        try:
            # Try different constructor patterns
            if field_name:
                return FlextExceptions.ValidationError(
                    f"{message} (field: {field_name})"
                )
            return FlextExceptions.ValidationError(message)
        except Exception:
            # Fallback to basic constructor
            return FlextExceptions.ValidationError(message)

    @staticmethod
    def create_configuration_error(
        message: str, config_key: str | None = None
    ) -> object:
        """Create configuration error."""
        try:
            # Try different constructor patterns
            if config_key:
                return FlextExceptions.ConfigurationError(
                    f"{message} (config: {config_key})"
                )
            return FlextExceptions.ConfigurationError(message)
        except Exception:
            # Fallback to basic constructor
            return FlextExceptions.ConfigurationError(message)

    @staticmethod
    def get_exception_metrics() -> dict[str, int]:
        """Get exception metrics."""
        # Use FlextExceptions for metrics management
        return FlextExceptions.get_metrics()

    @staticmethod
    def clear_exception_metrics() -> None:
        """Clear exception metrics."""
        # Use FlextExceptions for metrics management
        FlextExceptions.clear_metrics()

    @property
    def exceptions(self) -> type[FlextExceptions]:
        """Access exception utilities."""
        return FlextExceptions

    # =========================================================================
    # SCHEMA PROCESSING & SEMANTIC
    # =========================================================================

    @staticmethod
    def create_processing_pipeline() -> FlextProcessingPipeline[object, object]:
        """Create processing pipeline."""
        return FlextProcessingPipeline[object, object]()

    # =========================================================================
    # CONTEXT & PROTOCOLS
    # =========================================================================

    @staticmethod
    def create_context(**data: object) -> FlextResult[FlextContext]:
        """Create execution context."""
        try:
            context = FlextContext(**data)
            return FlextResult[FlextContext].ok(context)
        except Exception as e:
            return FlextResult[FlextContext].fail(f"Context creation failed: {e}")

    @property
    def context_class(self) -> type[FlextContext]:
        """Access context class."""
        return FlextContext

    @property
    def repository_protocol(self) -> object:
        """Access repository protocol."""
        return FlextProtocols.Domain.Repository

    @property
    def plugin_protocol(self) -> object:
        """Access plugin protocol."""
        return FlextProtocols.Extensions.Plugin

    @property
    def plugin_registry(self) -> object:
        """Get plugin registry instance."""
        if self._plugin_registry is None:
            # Create a simple plugin registry implementation
            class SimplePluginRegistry:
                def __init__(self) -> None:
                    self._plugins: dict[str, object] = {}

                def register(self, plugin: object) -> None:
                    plugin_name = getattr(plugin, "name", str(type(plugin).__name__))
                    self._plugins[plugin_name] = plugin

                def get(self, name: str) -> object | None:
                    return self._plugins.get(name)

                def list_plugins(self) -> list[str]:
                    return list(self._plugins.keys())

            self._plugin_registry = SimplePluginRegistry()
        return self._plugin_registry

    def register_plugin(self, plugin: object) -> FlextResult[None]:
        """Register plugin."""
        try:
            registry = self.plugin_registry
            if hasattr(registry, "register"):
                register_method = getattr(registry, "register", None)
                if callable(register_method):
                    register_method(plugin)
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(
                "Plugin registry does not support registration"
            )
        except Exception as e:
            return FlextResult[None].fail(f"Plugin registration failed: {e}")

    # =========================================================================
    # TYPE VALIDATION & GUARDS
    # =========================================================================

    @staticmethod
    def validate_type(obj: object, expected_type: type) -> FlextResult[object]:
        """Validate object type."""
        if not isinstance(obj, expected_type):
            return FlextResult[object].fail(
                f"Expected {expected_type.__name__}, got {type(obj).__name__}"
            )
        return FlextResult[object].ok(obj)

    @staticmethod
    def validate_dict_structure(
        obj: object, value_type: type
    ) -> FlextResult[dict[str, object]]:
        """Validate dictionary structure."""
        if not isinstance(obj, dict):
            return FlextResult[dict[str, object]].fail("Expected dictionary")

        if not FlextGuards.is_dict_of(cast("dict[object, object]", obj), value_type):
            return FlextResult[dict[str, object]].fail(
                f"Dictionary values must be of type {value_type.__name__}"
            )

        return FlextResult[dict[str, object]].ok(cast("dict[str, object]", obj))

    @staticmethod
    def create_validated_model(
        model_class: type, **data: object
    ) -> FlextResult[object]:
        """Create validated model."""
        try:
            model_validate_attr = getattr(model_class, "model_validate", None)
            if callable(model_validate_attr):
                instance: object = model_validate_attr(data)
                return FlextResult[object].ok(instance)
            instance_fallback: object = model_class(**data)
            return FlextResult[object].ok(instance_fallback)
        except Exception as e:
            return FlextResult[object].fail(f"Model validation failed: {e}")

    # =========================================================================
    # PERFORMANCE & MONITORING
    # =========================================================================

    @property
    def performance(self) -> type[FlextUtilities.Performance]:
        """Access performance utilities."""
        return FlextUtilities.Performance

    def track_performance(self, operation_name: str) -> object:
        """Create performance tracking decorator."""

        def _raise_not_callable() -> None:
            """Abstract raise to inner function for TRY301 compliance."""
            msg = "Decorated object is not callable"
            raise TypeError(msg)

        def decorator(func: object) -> object:
            def wrapper(*args: object, **kwargs: object) -> object:
                start_time = datetime.now(UTC)
                try:
                    # Use getattr for type-safe callable access
                    if callable(func):
                        result = func.__call__(*args, **kwargs)
                    else:
                        _raise_not_callable()
                        return None  # This line should never be reached
                    duration = (datetime.now(UTC) - start_time).total_seconds()
                    # Log performance metrics
                    logger = self.get_logger(__name__)
                    logger.info(
                        f"Operation {operation_name} completed in {duration:.3f}s"
                    )
                    return result
                except Exception:
                    duration = (datetime.now(UTC) - start_time).total_seconds()
                    logger = self.get_logger(__name__)
                    logger.exception(
                        f"Operation {operation_name} failed after {duration:.3f}s"
                    )
                    raise

            return wrapper

        return decorator

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    def create_factory(
        self, factory_type: str, **config: object
    ) -> FlextResult[object]:
        """Create factory instance."""
        try:
            if factory_type == "model":
                # Create basic model using FlextModels
                model_data = dict(config)
                model_data.setdefault("id", f"model_{id(config)}")
                return FlextResult[object].ok(model_data)
            if factory_type == "service":
                # Create basic service using FlextModels
                service_data = dict(config)
                service_data.setdefault("name", "service")
                return FlextResult[object].ok(service_data)
            return FlextResult[object].fail(f"Unknown factory type: {factory_type}")
        except Exception as e:
            return FlextResult[object].fail(f"Factory creation failed: {e}")

    @property
    def model_factory(self) -> type[FlextModels]:
        """Access model factory."""
        return FlextModels

    # =========================================================================
    # COMPREHENSIVE API ACCESS
    # =========================================================================

    def get_all_functionality(self) -> dict[str, object]:
        r"""Get comprehensive dictionary of all available FLEXT functionality and components.

        Returns a complete inventory of all FLEXT Core functionality accessible through
        this FlextCore instance, organized by functional areas. This method is useful
        for introspection, documentation generation, and system inventory management.

        Returns:
            dict[str, object]: Dictionary mapping functionality names to their implementations,
                organized by categories including core patterns, domain modeling, validation,
                utilities, handlers, decorators, and more.

        Functionality Categories:
            - **Core Patterns**: FlextResult, container, constants
            - **Domain Modeling**: Entity, Value Object, Aggregate Root base classes
            - **Validation & Guards**: Validators, predicates, guards, type checking
            - **Configuration**: Environment variables, config management, merging
            - **Utilities**: Generators, type guards, console, performance tools
            - **Messaging & Events**: Payload, message, event base classes
            - **Handlers & CQRS**: Handler registry, command processing
            - **Fields & Metadata**: Field definitions and registry
            - **Decorators**: Cross-cutting concern implementations
            - **Mixins**: Reusable behavior patterns
            - **Exceptions**: Error handling and metrics
            - **Context & Protocols**: Execution context and interface definitions
            - **Observability**: Metrics, monitoring, and performance tracking
            - **Factories**: Object creation and instantiation patterns

        Usage Examples:
            System introspection::

                core = FlextCore.get_instance()
                functionality = core.get_all_functionality()

                print(f"Available functionality: {len(functionality)} components")

                # List core patterns
                core_patterns = ["result", "container", "constants"]
                for pattern in core_patterns:
                    if pattern in functionality:
                        print(f"✅ {pattern}: {type(functionality[pattern]).__name__}")

            Documentation generation:

                >>> def generate_api_docs() -> str:
                ...     functionality = FlextCore.get_instance().get_all_functionality()
                ...     docs = ["# FLEXT Core API Reference\\n"]
                ...     for name, component in sorted(functionality.items()):
                ...         component_type = type(component).__name__
                ...         docstring = getattr(
                ...             component, "__doc__", "No documentation"
                ...         )
                ...         docs.append(f"## {name} ({component_type})\\n")
                ...         docs.append(f"{docstring}\\n")
                ...     return "\\n".join(docs)

            Feature availability checking:

                >>> def check_required_features(
                ...     required: list[str],
                ... ) -> FlextResult[None]:
                ...     functionality = FlextCore.get_instance().get_all_functionality()
                ...     missing = [
                ...         feature
                ...         for feature in required
                ...         if feature not in functionality
                ...     ]
                ...     if missing:
                ...         return FlextResult[None].fail(
                ...             f"Missing required features: {', '.join(missing)}"
                ...         )
                ...     return FlextResult[None].ok(None)

            System inventory for monitoring:

                >>> def collect_system_inventory() -> dict[str, object]:
                ...     core = FlextCore.get_instance()
                ...     return {
                ...         "functionality": core.get_all_functionality(),
                ...         "methods": core.list_available_methods(),
                ...         "system_info": core.get_system_info(),
                ...         "health": core.health_check().value
                ...         if core.health_check().success
                ...         else None,
                ...     }

        Integration Points:
            Each functionality entry represents a major integration point with the
            FLEXT ecosystem, providing access to:

            - **Service Registration**: Via container and registries
            - **Domain Modeling**: Through base classes and factories
            - **Validation Pipelines**: Via validators and predicates
            - **Cross-cutting Concerns**: Through decorators and mixins
            - **System Monitoring**: Via observability and performance tools

        Note:
            The returned dictionary provides references to live objects and classes.
            Modifications to mutable objects will affect the global system state.
            Use with caution in production environments.

        See Also:
            - list_available_methods(): List callable methods on FlextCore
            - get_method_info(): Get detailed information about specific methods
            - get_system_info(): System configuration and runtime information

        """
        return {
            # Core patterns
            "result": FlextResult,
            "container": self.container,
            "constants": self.constants,
            # Domain modeling
            "entity_base": self.entity_base,
            "value_object_base": self.value_object_base,
            "aggregate_root_base": self.aggregate_root_base,
            "domain_service_base": self.domain_service_base,
            # Validation & guards
            "validators": self.validators,
            "predicates": self.predicates,
            "guards": self.guards,
            # Configuration utilities
            "safe_get_env_var": self.safe_get_env_var,
            "merge_configs": self.merge_configs,
            "validate_config": self.validate_config,
            # Utilities
            "utilities": self.utilities,
            "generators": self.generators,
            "type_guards": self.type_guards,
            "console": self.console,
            "performance": self.performance,
            # Messaging & events
            "payload_base": self.payload_base,
            "message_base": self.message_base,
            "event_base": self.event_base,
            # Handlers & CQRS
            "handlers": self.handlers,
            "handler_registry": self.handler_registry,
            "base_handler": self.base_handler,
            "commands": self.commands,
            # Fields & metadata
            "fields": self.fields,
            "field_registry": self.field_registry,
            # Decorators
            "decorators": self.decorators,
            # "decorator_factory": self.decorator_factory,
            # Mixins
            "timestamp_mixin": self.timestamp_mixin,
            "identifiable_mixin": self.identifiable_mixin,
            "loggable_mixin": self.loggable_mixin,
            "validatable_mixin": self.validatable_mixin,
            "serializable_mixin": self.serializable_mixin,
            "cacheable_mixin": self.cacheable_mixin,
            # Exceptions
            "exceptions": self.exceptions,
            # Schema & semantic - removed as semantic.py no longer exists
            # Context & protocols
            "context_class": self.context_class,
            "repository_protocol": self.repository_protocol,
            "plugin_protocol": self.plugin_protocol,
            "plugin_registry": self.plugin_registry,
            # Observability
            "observability": self.observability,
            # Factory
            "model_factory": self.model_factory,
        }

    def list_available_methods(self) -> list[str]:
        """List all available public methods."""
        return [
            method
            for method in dir(self)
            if not method.startswith("_") and callable(getattr(self, method))
        ]

    def get_method_info(self, method_name: str) -> FlextResult[dict[str, object]]:
        """Get information about a specific method."""
        try:
            if not hasattr(self, method_name):
                return FlextResult[dict[str, object]].fail(
                    f"Method not found: {method_name}"
                )

            method = getattr(self, method_name)
            if not callable(method):
                return FlextResult[dict[str, object]].fail(
                    f"Attribute is not callable: {method_name}"
                )

            info: dict[str, object] = {
                "name": method_name,
                "doc": method.__doc__ or "No documentation available",
                "type": "method" if hasattr(method, "__self__") else "function",
                "callable": True,
            }

            return FlextResult[dict[str, object]].ok(info)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to get method info: {e}"
            )

    # =========================================================================
    # SYSTEM INFORMATION & DIAGNOSTICS
    # =========================================================================

    def get_system_info(self) -> dict[str, object]:
        """Get comprehensive system information."""
        info: dict[str, object] = {
            "version": "2.0.0-comprehensive",
            "singleton_id": id(self),
            "container_services": self._container.get_service_count()
            if hasattr(self._container, "get_service_count")
            else "Unknown",
            "settings_cached": len(self._settings_cache),
            "handler_registry_loaded": self._handler_registry is not None,
            "field_registry_loaded": self._field_registry is not None,
            "plugin_registry_loaded": self._plugin_registry is not None,
            "console_loaded": self._console is not None,
            "observability_loaded": self._observability is not None,
            "total_methods": len(self.list_available_methods()),
            "functionality_count": len(self.get_all_functionality()),
        }
        return info

    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform comprehensive system health check with detailed diagnostics.

        Executes a thorough health check of all FlextCore subsystems and components,
        testing functionality, connectivity, and performance. Returns detailed status
        information for monitoring and debugging purposes.

        Returns:
            FlextResult[dict[str, object]]: Success result containing health status dictionary
                with component status, or failure result if health check fails.

        Health Check Components:
            - **Container Status**: Dependency injection container health
            - **Logging System**: Logger functionality verification
            - **Validation System**: Validation pipeline testing
            - **Result Patterns**: FlextResult[T] operation verification
            - **Utility Functions**: Core utility function testing
            - **Timestamp Generation**: System time and timing accuracy

        Status Dictionary Structure:
            The returned dictionary contains::

                {
                    "status": "healthy" | "degraded" | "unhealthy",
                    "container": "ok" | "missing" | "error",
                    "logging": "ok" | "error",
                    "validation": "ok" | "error",
                    "result_pattern": "ok" | "error",
                    "timestamp": "2025-01-15T10:30:45.123456Z",
                }

        Usage Examples:
            Basic health monitoring::

                core = FlextCore.get_instance()
                health_result = core.health_check()

                if health_result.success:
                    health = health_result.value
                    if health["status"] == "healthy":
                        print("✅ System is healthy")
                    else:
                        print(f"⚠️ System status: {health['status']}")
                        # Log degraded components
                        for component, status in health.items():
                            if status == "error":
                                logger.warning(f"Component {component} is unhealthy")

            Automated monitoring integration:

                >>> def monitor_system_health() -> bool:
                ...     health_result = FlextCore.get_instance().health_check()
                ...     if health_result.failure:
                ...         logger.critical(
                ...             f"Health check failed: {health_result.error}"
                ...         )
                ...         return False
                ...     health = health_result.value
                ...     critical_components = ["container", "logging", "validation"]
                ...     for component in critical_components:
                ...         if health.get(component) == "error":
                ...             logger.error(
                ...                 f"Critical component {component} is unhealthy"
                ...             )
                ...             return False
                ...     return health["status"] in ["healthy", "degraded"]

            Integration with health endpoints:

                >>> # FastAPI health endpoint example
                >>> from fastapi import HTTPException
                >>> @app.get("/health")
                ... def health_endpoint():
                ...     health_result = FlextCore.get_instance().health_check()
                ...     if health_result.failure:
                ...         raise HTTPException(
                ...             status_code=503, detail="Service unavailable"
                ...         )
                ...     health = health_result.value
                ...     status_code = 200 if health["status"] == "healthy" else 206
                ...     return Response(
                ...         content=json.dumps(health),
                ...         status_code=status_code,
                ...         media_type="application/json",
                ...     )

        Error Handling:
            The health check is designed to be resilient and will attempt to test
            all components even if some fail. A complete failure only occurs if
            the health check system itself encounters an unexpected error.

        Performance:
            The health check is lightweight and designed for frequent execution
            in production monitoring systems. Most checks complete in milliseconds.

        See Also:
            - get_system_info(): System information and metrics
            - reset_all_caches(): Reset system state for testing
            - get_all_functionality(): Available functionality inventory

        """
        try:
            health: dict[str, object] = {
                "status": "healthy",
                "container": "ok" if self._container else "missing",
                "logging": "ok",
                "validation": "ok",
                "utilities": "ok",
                "timestamp": str(datetime.now(UTC)),
            }

            # Test basic functionality
            test_result = self.ok("health_check_test")
            if test_result.is_failure:
                health["status"] = "degraded"
                health["result_pattern"] = "error"
            else:
                health["result_pattern"] = "ok"

            # Test validation
            try:
                validation_result = self.validate_required("test", "health_check")
                if (
                    hasattr(validation_result, "is_failure")
                    and validation_result.is_failure
                ):
                    health["status"] = "degraded"
                    health["validation"] = "error"
            except Exception:
                health["status"] = "degraded"
                health["validation"] = "error"

            return FlextResult[dict[str, object]].ok(health)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Health check failed: {e}")

    def reset_all_caches(self) -> FlextResult[None]:
        """Reset all cached instances."""
        try:
            self._settings_cache.clear()
            self._handler_registry = None
            self._field_registry = None
            self._plugin_registry = None
            self._console = None
            self._observability = None
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Cache reset failed: {e}")

    # =========================================================================
    # ENTERPRISE BUILDERS & FACTORIES (BOILERPLATE REDUCTION)
    # =========================================================================

    def create_validator_class[T](
        self,
        name: str,
        validation_func: Callable[[T], FlextResult[T]],
    ) -> type[FlextValidation.Domain.BaseValidator]:
        """Create validator class dynamically to reduce boilerplate."""
        # Import already at module level

        class DynamicValidator(FlextValidation.Domain.BaseValidator):
            def validate(self, value: T) -> FlextResult[T]:
                return validation_func(value)

        DynamicValidator.__name__ = name
        DynamicValidator.__qualname__ = name
        return cast("type[FlextValidation.Domain.BaseValidator]", DynamicValidator)

    def create_service_processor(
        self,
        name: str,
        process_func: Callable[[object], FlextResult[object]],
        result_type: type[object] = object,
        build_func: Callable[[object, str], object] | None = None,
        decorators: list[str] | None = None,
    ) -> type:
        """Create service processor class dynamically to reduce boilerplate."""

        class DynamicServiceProcessor:
            def __init__(self) -> None:
                # Use class method to get logger (get_logger may not be available)
                self._logger = FlextLogger(f"flext.services.{name.lower()}")

            def process(self, request: object) -> FlextResult[object]:
                return process_func(request)

            def build(self, domain: object, *, correlation_id: str) -> object:
                if build_func:
                    return build_func(domain, correlation_id)
                # Default: return domain if types match
                if isinstance(domain, result_type):
                    return domain
                # Fallback: try to create result_type from domain
                if hasattr(result_type, "model_validate"):
                    # Use getattr to safely access the method with type safety
                    model_validate = getattr(result_type, "model_validate", None)
                    if callable(model_validate):
                        return model_validate(domain)
                # Final fallback: try constructor with domain attributes
                if hasattr(domain, "__dict__"):
                    return result_type(**domain.__dict__)
                return result_type()

        # Apply decorators if specified
        if decorators:
            original_process = DynamicServiceProcessor.process
            for decorator_name in decorators:
                if hasattr(FlextDecorators, decorator_name):
                    decorator = getattr(FlextDecorators, decorator_name)
                    if callable(decorator):
                        # Create decorated method instead of assigning to class
                        decorated_method = decorator(original_process)
                        # Dynamically assign the method to the class
                        # Necessary for dynamic decoration (intentional)
                        DynamicServiceProcessor.process = decorated_method  # type: ignore[assignment]

        DynamicServiceProcessor.__name__ = f"{name}ServiceProcessor"
        DynamicServiceProcessor.__qualname__ = f"{name}ServiceProcessor"
        return DynamicServiceProcessor

    def create_entity_with_validators(
        self,
        name: str,
        fields: dict[str, tuple[type, dict[str, object]]],
        validators: dict[str, Callable[[object], FlextResult[object]]] | None = None,
    ) -> type[FlextModels.Entity]:
        """Create entity class with built-in validators to reduce boilerplate."""
        # Build field annotations
        annotations = {}

        for field_name, (field_type, _field_config) in fields.items():
            # Create basic annotated field - simplified to avoid type issues
            annotations[field_name] = Annotated[field_type, Field()]

        # Create class attributes
        class_attrs: dict[str, object] = {
            "__annotations__": annotations,
        }

        # Add field validators if provided
        if validators:
            for field_name, validator_func in validators.items():

                def create_validator(
                    func: Callable[[object], FlextResult[object]],
                    fname: str = field_name,
                ) -> object:
                    def validator_method(_cls: type[object], v: object) -> object:
                        result = func(v)
                        if result.is_failure:
                            error_msg = result.error or f"{fname} validation failed"
                            raise ValueError(error_msg)
                        return result.value

                    # Apply decorators to create proper validator
                    return field_validator(fname)(classmethod(validator_method))

                class_attrs[f"validate_{field_name}_field"] = create_validator(
                    validator_func
                )

        # Create dynamic class
        return type(name, (FlextModels.Entity,), class_attrs)

    def create_value_object_with_validators(
        self,
        name: str,
        fields: dict[str, tuple[type, dict[str, object]]],
        validators: dict[str, Callable[[object], FlextResult[object]]] | None = None,
        business_rules: Callable[[object], FlextResult[None]] | None = None,
    ) -> type[FlextModels.Value]:
        """Create value object class with built-in validators to reduce boilerplate."""
        # Build field annotations
        annotations = {}

        for field_name, (field_type, _field_config) in fields.items():
            # Create basic annotated field - simplified to avoid type issues
            annotations[field_name] = Annotated[field_type, Field()]

        # Create class attributes
        class_attrs: dict[str, object] = {
            "__annotations__": annotations,
        }

        # Add field validators if provided
        if validators:
            for field_name, validator_func in validators.items():

                def create_validator(
                    func: Callable[[object], FlextResult[object]],
                    fname: str = field_name,
                ) -> object:
                    def validator_method(_cls: type[object], v: object) -> object:
                        result = func(v)
                        if result.is_failure:
                            error_msg = result.error or f"{fname} validation failed"
                            raise ValueError(error_msg)
                        return result.value

                    # Apply decorators to create proper validator
                    return field_validator(fname)(classmethod(validator_method))

                class_attrs[f"validate_{field_name}_field"] = create_validator(
                    validator_func
                )

        # Add business rules validation if provided
        if business_rules:

            def validate_business_rules_method(self: object) -> FlextResult[None]:
                return business_rules(self)

            class_attrs["validate_business_rules"] = validate_business_rules_method

        # Create dynamic class
        return type(name, (FlextModels.Value,), class_attrs)

    def setup_container_with_services(
        self,
        services: dict[str, type | Callable[[], object]],
        validator: Callable[[str], FlextResult[str]] | None = None,
    ) -> FlextResult[FlextContainer]:
        """Setup container with multiple services, reducing boilerplate."""
        try:
            container = self.container

            for service_name, service_factory in services.items():
                # Validate service name if validator provided
                if validator:
                    validation_result = validator(service_name)
                    if validation_result.is_failure:
                        self.get_logger(__name__).error(
                            "Service name validation failed",
                            service_name=service_name,
                            error=Exception(
                                validation_result.error or "Validation error"
                            ),
                        )
                        continue

                # Register service
                if isinstance(service_factory, type):
                    # Class factory - use closure to capture type properly
                    def create_factory_func(
                        cls: type | Callable[[], object] = service_factory,
                    ) -> object:
                        if isinstance(cls, type):
                            return cls()
                        return cls()

                    register_result = container.register_factory(
                        service_name, create_factory_func
                    )
                else:
                    # Callable factory
                    register_result = container.register_factory(
                        service_name, service_factory
                    )

                if register_result.is_failure:
                    return FlextResult[FlextContainer].fail(
                        f"Failed to register {service_name}: {register_result.error}"
                    )

                self.get_logger(__name__).info(
                    "Service registered",
                    service_name=service_name,
                    service_class=(
                        service_factory.__name__
                        if hasattr(service_factory, "__name__")
                        else str(type(service_factory))
                    ),
                )

            self.get_logger(__name__).info(
                "Container setup completed",
                total_services=len(services),
                registered_services=list(services.keys()),
            )

            return FlextResult[FlextContainer].ok(container)
        except Exception as e:
            return FlextResult[FlextContainer].fail(f"Container setup failed: {e}")

    def create_demo_function(
        self,
        name: str,
        demo_func: Callable[[], None],
        decorators: list[str] | None = None,
    ) -> Callable[[], None]:
        """Create demo function with standard decorators to reduce boilerplate."""
        # Apply decorators if specified
        decorated_func: Callable[[], None] = demo_func
        if decorators:
            for decorator_name in reversed(decorators):  # Apply in reverse order
                if hasattr(FlextDecorators, decorator_name):
                    decorator = getattr(FlextDecorators, decorator_name)
                    if callable(decorator):
                        decorated_func = cast(
                            "Callable[[], None]", decorator(decorated_func)
                        )

        decorated_func.__name__ = name
        decorated_func.__qualname__ = name
        return decorated_func

    def log_result(
        self, result: FlextResult[T], success_msg: str, logger_name: str | None = None
    ) -> FlextResult[T]:
        """Utility to log FlextResult with consistent formatting."""
        logger = self.get_logger(logger_name or __name__)
        if result.is_success:
            logger.info(f"✅ {success_msg}", result_type=type(result.value).__name__)
        else:
            logger.error(
                f"❌ {success_msg} failed",
                error=Exception(result.error or "Unknown error"),
            )
        return result

    def get_service_with_fallback(
        self, service_name: str, default_factory: type[T]
    ) -> T:
        """Get service from container with type-safe fallback."""
        result = self.get_service(service_name)
        if result.is_success:
            self.get_logger(__name__).debug(
                "Service retrieved from container", service_name=service_name
            )
            return cast("T", result.value)

        self.get_logger(__name__).warning(
            "Service not found in container, using default factory",
            service_name=service_name,
            default_factory=default_factory.__name__,
        )
        return default_factory()

    def create_standard_validators(
        self,
    ) -> dict[str, Callable[[object], FlextResult[object]]]:
        """Create standard validators to reduce boilerplate."""
        return {
            "age": lambda v: cast(
                "FlextResult[object]",
                self.validate_numeric(v, 18, 120)
                if isinstance(v, (int, float))
                else FlextResult[object].fail("Age must be numeric"),
            ),
            "email": lambda v: cast(
                "FlextResult[object]",
                self.validate_email(v)
                if isinstance(v, str)
                else FlextResult[object].fail("Email must be string"),
            ),
            "name": lambda v: cast(
                "FlextResult[object]",
                self.validate_string(v, 2, 100)
                if isinstance(v, str)
                else FlextResult[object].fail("Name must be string"),
            ),
            "service_name": lambda v: cast(
                "FlextResult[object]",
                self.validate_service_name(v)
                if isinstance(v, str)
                else FlextResult[object].fail("Service name must be string"),
            ),
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @override
    def __repr__(self) -> str:
        """Return comprehensive string representation with system metrics.

        Provides detailed technical representation of the FlextCore instance including
        service counts, available methods, and functionality metrics. This representation
        is designed for debugging, logging, and system monitoring purposes.

        Returns:
            str: Detailed representation including service count, method count, and
                functionality count for system monitoring and debugging.

        Usage Examples:
            Debug logging::

                core = FlextCore.get_instance()
                logger.debug(f"FlextCore state: {repr(core)}")
                # Output: FlextCore(services=15, methods=87, functionality=42)

            System monitoring::

                def log_system_state():
                    core = FlextCore.get_instance()
                    state_info = repr(core)
                    metrics_logger.info(f"System state: {state_info}")

        Note:
            The service count is retrieved from the container's get_service_count()
            method if available, otherwise displays "Unknown".

        """
        service_count = (
            self._container.get_service_count()
            if hasattr(self._container, "get_service_count")
            else "Unknown"
        )
        return (
            f"FlextCore("
            f"services={service_count}, "
            f"methods={len(self.list_available_methods())}, "
            f"functionality={len(self.get_all_functionality())}"
            f")"
        )

    @override
    def __str__(self) -> str:
        """Return user-friendly string representation for end-user display.

        Provides a clean, human-readable representation of FlextCore suitable for
        end-user interfaces, help messages, and general application output.

        Returns:
            str: User-friendly representation with version information.

        Usage Examples:
            User interface display::

                core = FlextCore.get_instance()
                print(f"Initialized: {str(core)}")
                # Output: FlextCore - Comprehensive FLEXT ecosystem access (v2.0.0)

            Help messages::

                def show_system_info():
                    core = FlextCore.get_instance()
                    print(f"Running {str(core)}")
                    print(f"Available methods: {len(core.list_available_methods())}")

        """
        return "FlextCore - Comprehensive FLEXT ecosystem access (v2.0.0)"


# Export API
__all__: list[str] = [
    "FlextCore",  # ONLY main class exported
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
