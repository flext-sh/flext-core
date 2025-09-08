"""FLEXT Configuration Management - Type-safe configuration with environment integration.

This module provides enterprise-grade configuration management using Pydantic
settings with proper validation, environment variable integration, and
FlextResult error handling patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self, cast

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Version validation constants
_SEMANTIC_VERSION_MIN_PARTS = 3

# Business rule validation constants
_MIN_PRODUCTION_WORKERS = 2
_HIGH_TIMEOUT_THRESHOLD = 120
_MIN_WORKERS_FOR_HIGH_TIMEOUT = 4
_MAX_WORKERS_THRESHOLD = 50

# Configuration profile constants
_PROFILE_WEB_SERVICE = "web_service"
_PROFILE_DATA_PROCESSOR = "data_processor"
_PROFILE_API_CLIENT = "api_client"
_PROFILE_BATCH_JOB = "batch_job"
_PROFILE_MICROSERVICE = "microservice"

# Type aliases removed - using direct typing


class FlextConfig(BaseSettings):
    """Type-safe configuration management with environment integration.

    Provides declarative configuration fields with automatic environment
    variable binding, validation, and FlextResult-based error handling.
    Follows Single Responsibility Principle - only handles configuration
    loading, validation, and access.

    Features:
    - Automatic environment variable mapping with FLEXT_ prefix
    - Type-safe field validation using Pydantic v2
    - FlextResult integration for error handling
    - Configuration sealing to prevent runtime modifications
    - Structured configuration metadata tracking
    """

    # Core application identity fields
    app_name: str = Field(
        default="flext-app",
        description="Application identifier name",
        min_length=1,
        max_length=100,
    )

    name: str = Field(
        default=FlextConstants.Core.NAME.lower(),
        description="Configuration instance name",
        min_length=1,
        max_length=50,
    )

    version: str = Field(
        default=FlextConstants.Core.VERSION,
        description="Configuration version string",
        pattern=r"^\d+\.\d+\.\d+.*$",
    )

    description: str = Field(
        default="FLEXT application configuration",
        description="Configuration description",
        max_length=500,
    )

    # Environment and runtime settings
    environment: FlextTypes.Config.Environment = Field(
        default="development",
        description="Deployment environment identifier",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode and verbose logging",
    )

    # Observability configuration
    log_level: str = Field(
        default=FlextConstants.Observability.DEFAULT_LOG_LEVEL,
        description="Global logging level",
    )

    # Configuration management fields
    config_source: str = Field(
        default="default",
        description="Source of configuration values",
    )

    config_priority: int = Field(
        default=5,
        description="Configuration priority level",
        ge=1,
        le=10,
    )

    # Performance configuration
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker threads",
        ge=1,
    )

    timeout_seconds: int = Field(
        default=30,
        description="Default timeout in seconds",
        ge=1,
    )

    # Feature flags
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection",
    )

    enable_caching: bool = Field(
        default=True,
        description="Enable caching functionality",
    )

    enable_auth: bool = Field(
        default=False,
        description="Enable authentication and authorization",
    )

    enable_rate_limiting: bool = Field(
        default=False,
        description="Enable API rate limiting",
    )

    enable_circuit_breaker: bool = Field(
        default=False,
        description="Enable circuit breaker pattern for resilience",
    )

    # Web/API server configuration (common across many FLEXT projects)
    host: str = Field(
        default="127.0.0.1",
        description="Server host address",
        min_length=1,
    )

    port: int = Field(
        default=8000,
        description="Server port number",
        ge=1,
        le=65535,
    )

    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for service endpoints",
    )

    # Database configuration (common across data-centric FLEXT projects)
    database_url: str = Field(
        default="",
        description="Database connection URL",
    )

    database_pool_size: int = Field(
        default=5,
        description="Database connection pool size",
        ge=1,
        le=50,
    )

    database_timeout: int = Field(
        default=30,
        description="Database query timeout in seconds",
        ge=1,
    )

    # Message queue configuration (for async processing)
    message_queue_url: str = Field(
        default="",
        description="Message queue connection URL (Redis, RabbitMQ, etc.)",
    )

    message_queue_max_retries: int = Field(
        default=3,
        description="Maximum number of message processing retries",
        ge=0,
    )

    # Monitoring and health check configuration
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds",
        ge=1,
    )

    metrics_port: int = Field(
        default=9090,
        description="Port for metrics endpoint",
        ge=1,
        le=65535,
    )

    # Security configuration
    api_key: str = Field(
        default="",
        description="API key for external service authentication",
    )

    cors_origins: FlextTypes.Core.StringList = Field(
        default_factory=list,
        description="Allowed CORS origins for web APIs",
    )

    # Internal state management
    _metadata: FlextTypes.Core.Headers = PrivateAttr(default_factory=dict)
    _sealed: bool = PrivateAttr(default=False)

    # Pydantic configuration for environment integration
    model_config = SettingsConfigDict(
        # Environment integration
        env_prefix="FLEXT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Validation behavior
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        extra="ignore",
        # Schema generation
        title="FLEXT Configuration",
        json_schema_extra={
            "description": "FLEXT application configuration with type safety",
            "examples": [
                {
                    "app_name": "flext-data-processor",
                    "environment": "production",
                    "debug": False,
                    "log_level": "INFO",
                }
            ],
        },
    )

    def __init__(self, /, *, _factory_mode: bool = False, **data: object) -> None:
        """Override BaseSettings init to preserve pure default semantics.

        Args:
            _factory_mode: Internal flag to indicate factory creation (allows env vars)
            **data: Additional configuration data

        """
        environment_explicit = "environment" in data
        debug_explicit = "debug" in data
        log_level_explicit = "log_level" in data

        # Call parent constructor without any arguments to avoid type issues
        # Let Pydantic handle the data validation through its normal mechanisms
        super().__init__()

        # Then update fields individually from data if provided
        if data:
            for field_name, value in data.items():
                if isinstance(field_name, str) and hasattr(self, field_name):
                    # Set attribute directly - validation will happen via Pydantic
                    # Don't catch validation errors to allow proper test failures
                    setattr(self, field_name, value)
        # Only force defaults when NOT in factory mode
        if not _factory_mode:
            # Helper for field defaults avoiding direct attribute coupling for type checker
            def _get_field_default(field_name: str) -> object:
                fields_obj = getattr(
                    type(self), "model_fields", {}
                )  # pydantic v2 attribute
                if isinstance(fields_obj, dict):
                    field_info = fields_obj.get(field_name)
                    if field_info is not None and hasattr(field_info, "default"):
                        return getattr(field_info, "default")
                return getattr(self, field_name)

            if not environment_explicit:
                object.__setattr__(
                    self, "environment", _get_field_default("environment")
                )
            if not debug_explicit:
                object.__setattr__(self, "debug", _get_field_default("debug"))
            if not log_level_explicit:
                object.__setattr__(self, "log_level", _get_field_default("log_level"))

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Validate environment is in allowed set."""
        allowed = set(FlextConstants.Config.ENVIRONMENTS)
        if value not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            # Padronizar mensagem para testes que procuram prefixo 'Invalid environment'
            # mantendo substring anterior para compatibilidade reversa.
            msg = f"Invalid environment. Environment must be one of: {allowed_str}"
            raise ValueError(msg)
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate log level is recognized by logging system."""
        allowed = {level.value for level in FlextConstants.Config.LogLevel}
        normalized = value.upper()
        if normalized not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            msg = f"Invalid log_level. Log level must be one of: {allowed_str}"
            raise ValueError(msg)
        return normalized

    @field_validator("config_source")
    @classmethod
    def validate_config_source(cls, value: str) -> str:
        """Validate config source is in allowed set."""
        allowed = {source.value for source in FlextConstants.Config.ConfigSource}
        if value not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            msg = f"Config source must be one of: {allowed_str}"
            raise ValueError(msg)
        return value

    @field_validator(
        "max_workers",
        "timeout_seconds",
        "config_priority",
        "port",
        "database_pool_size",
        "database_timeout",
        "health_check_interval",
        "metrics_port",
    )
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        """Validate that integer fields are positive."""
        if value <= 0:
            msg = f"Value must be positive, got {value}"
            raise ValueError(msg)
        return value

    @field_validator("message_queue_max_retries")
    @classmethod
    def validate_non_negative_integers(cls, value: int) -> int:
        """Validate that integer fields are non-negative."""
        if value < 0:
            msg = f"Value must be non-negative, got {value}"
            raise ValueError(msg)
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        """Validate host is not empty."""
        if not value.strip():
            msg = "Host cannot be empty"
            raise ValueError(msg)
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Validate base URL has proper protocol."""
        if not value.strip():
            msg = "Base URL cannot be empty"
            raise ValueError(msg)
        if not value.startswith(("http://", "https://")):
            msg = "Base URL must start with http:// or https://"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_configuration_consistency(self) -> Self:
        """Validate cross-field configuration consistency."""
        # Development should use appropriate log levels
        if self.environment == "development" and self.log_level in {
            "CRITICAL",
            "ERROR",
        }:
            msg = f"Log level {self.log_level} too restrictive for development"
            raise ValueError(msg)

        return self

    @classmethod
    def get_env_var(cls, var_name: str) -> FlextResult[str]:
        """Get environment variable with FlextResult error handling.

        Args:
            var_name: Name of environment variable to retrieve

        Returns:
            FlextResult containing variable value or error details

        """
        try:
            value = os.environ.get(var_name)
            if value is None:
                return FlextResult[str].fail(
                    f"Environment variable {var_name} not set",
                    error_code="ENV_VAR_NOT_FOUND",
                )
            return FlextResult[str].ok(value)
        except Exception as error:
            return FlextResult[str].fail(
                f"Failed to get environment variable '{var_name}': {error}",
                error_code="ENV_VAR_ERROR",
            )

    @classmethod
    def validate_config_value(
        cls, value: object, expected_type: type
    ) -> FlextResult[bool]:
        """Validate that a configuration value matches expected type.

        Args:
            value: Value to validate
            expected_type: Expected type for the value

        Returns:
            FlextResult containing True if valid, False if invalid

        """
        try:
            is_valid = isinstance(value, expected_type)
            return FlextResult[bool].ok(is_valid)
        except Exception as error:
            return FlextResult[bool].fail(
                f"Type validation failed: {error}",
                error_code="TYPE_VALIDATION_ERROR",
            )

    @classmethod
    def merge_configs(
        cls, config1: FlextTypes.Core.Dict, config2: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Merge two configuration dictionaries with conflict resolution.

        Args:
            config1: First configuration dictionary
            config2: Second configuration dictionary (takes precedence)

        Returns:
            FlextResult containing merged configuration dictionary

        """
        try:
            # Create a copy of config1 to avoid modifying original
            merged = config1.copy()
            # Update with config2 values (config2 takes precedence)
            merged.update(config2)
            return FlextResult[FlextTypes.Core.Dict].ok(merged)
        except Exception as error:
            return FlextResult[FlextTypes.Core.Dict].fail(
                f"Config merge failed: {error}",
                error_code="CONFIG_MERGE_ERROR",
            )

    @classmethod
    def create(
        cls,
        *,
        constants: FlextTypes.Core.Dict | None = None,
        cli_overrides: FlextTypes.Core.Dict | None = None,
        env_file: str | Path | None = None,
    ) -> FlextResult[Self]:
        """Create configuration instance with constants and environment integration.

        Args:
            constants: Dictionary of configuration values to set
            cli_overrides: CLI command-line override values (highest priority)
            env_file: Optional path to environment file

        Returns:
            FlextResult containing configured instance or error details

        """
        settings: FlextTypes.Core.Dict = {}
        try:
            # Start with constants if provided
            settings = constants.copy() if constants else {}

            # Apply CLI overrides (highest priority)
            if cli_overrides:
                settings.update(cli_overrides)

            # Configure environment file if provided
            if env_file:
                env_path = Path(env_file)
                if not env_path.exists():
                    return FlextResult[Self].fail(
                        f"Environment file not found: {env_file}",
                        error_code="ENV_FILE_NOT_FOUND",
                    )

            # Create instance with validation
            if env_file:
                # Create a temporary settings class with the specific env_file
                config_dict = cls.model_config.copy()  # type: ignore[attr-defined]
                config_dict["env_file"] = str(env_file)

                # Create a temporary settings class with the env_file config
                class TempConfig(cls):  # type: ignore[misc,valid-type]
                    model_config = config_dict

                instance = cast("Self", TempConfig(_factory_mode=True, **settings))  # type: ignore[call-arg]
            else:
                instance = cls(_factory_mode=True, **settings)  # type: ignore[misc]

            # Track creation metadata
            instance._metadata["created_from"] = "factory"
            if constants:
                instance._metadata["constants_provided"] = "true"
            if cli_overrides:
                instance._metadata["cli_overrides_provided"] = "true"
            if env_file:
                instance._metadata["env_file"] = str(env_file)

            return FlextResult[Self].ok(instance)

        except ValidationError as exc:
            # Reformat environment validation errors para satisfazer testes
            invalid_env = (
                settings.get("environment") if isinstance(settings, dict) else None
            )
            for err in exc.errors():
                loc = err.get("loc")
                if loc and loc[0] == "environment":
                    allowed = ", ".join(sorted(FlextConstants.Config.ENVIRONMENTS))
                    msg = f"Configuration creation failed: Invalid environment '{invalid_env}'. Environment must be one of: {allowed}"
                    return FlextResult[Self].fail(
                        msg, error_code="CONFIG_CREATION_ERROR"
                    )
            return FlextResult[Self].fail(
                f"Configuration creation failed: {exc}",
                error_code="CONFIG_CREATION_ERROR",
            )
        except Exception as error:
            return FlextResult[Self].fail(
                f"Configuration creation failed: {error}",
                error_code="CONFIG_CREATION_ERROR",
            )

    @classmethod
    def create_from_environment(
        cls,
        *,
        env_file: str | Path | None = None,
        extra_settings: FlextTypes.Core.Dict | None = None,
    ) -> FlextResult[Self]:
        """Create configuration instance from environment with validation.

        Args:
            env_file: Optional path to environment file
            extra_settings: Additional settings to override defaults

        Returns:
            FlextResult containing configured instance or error details

        """
        settings: FlextTypes.Core.Dict = {}
        try:
            # Prepare settings for Pydantic
            settings = {}
            if extra_settings:
                settings.update(extra_settings)

            # Configure environment file if provided
            if env_file:
                env_path = Path(env_file)
                if not env_path.exists():
                    return FlextResult[Self].fail(
                        f"Environment file not found: {env_file}",
                        error_code="ENV_FILE_NOT_FOUND",
                    )
                settings["_env_file"] = str(env_file)

            # Create instance with validation
            instance = cast(
                "Self", cast("type[BaseModel]", cls).model_validate(settings)
            )

            # Track creation metadata
            instance._metadata["created_from"] = "environment"
            if env_file:
                instance._metadata["env_file"] = str(env_file)

            return FlextResult[Self].ok(instance)

        except ValidationError as exc:
            invalid_env = (
                settings.get("environment") if isinstance(settings, dict) else None
            )
            for err in exc.errors():
                loc = err.get("loc")
                if loc and loc[0] == "environment":
                    allowed = ", ".join(sorted(FlextConstants.Config.ENVIRONMENTS))
                    msg = f"Invalid environment '{invalid_env}'. Environment must be one of: {allowed}"
                    return FlextResult[Self].fail(
                        msg, error_code="CONFIG_CREATION_ERROR"
                    )
            return FlextResult[Self].fail(
                f"Configuration creation failed: {exc}",
                error_code="CONFIG_CREATION_ERROR",
            )
        except Exception as error:
            return FlextResult[Self].fail(
                f"Configuration creation failed: {error}",
                error_code="CONFIG_CREATION_ERROR",
            )

    def seal(self) -> FlextResult[None]:
        """Seal configuration to prevent further modifications.

        Returns:
            FlextResult indicating success or failure of sealing operation

        """
        if self._sealed:
            return FlextResult[None].fail(
                "Configuration is already sealed", error_code="CONFIG_ALREADY_SEALED"
            )

        try:
            self._sealed = True
            return FlextResult[None].ok(None)
        except Exception as error:
            return FlextResult[None].fail(
                f"Failed to seal configuration: {error}", error_code="CONFIG_SEAL_ERROR"
            )

    def is_sealed(self) -> bool:
        """Check if configuration is sealed against modifications."""
        return self._sealed

    def get_metadata(self) -> FlextTypes.Core.Headers:
        """Get configuration creation and modification metadata."""
        return dict(self._metadata)

    def to_api_payload(self) -> FlextResult[FlextTypes.Core.Dict]:
        """Export configuration as API-safe payload.

        Returns:
            FlextResult containing serialized configuration data

        """
        try:
            payload = {
                "app_name": self.app_name,
                "environment": self.environment,
                "debug": self.debug,
            }
            return FlextResult[FlextTypes.Core.Dict].ok(payload)
        except Exception as error:
            return FlextResult[FlextTypes.Core.Dict].fail(
                f"Failed to create API payload: {error}",
                error_code="CONFIG_SERIALIZATION_ERROR",
            )

    def as_api_payload(self) -> FlextResult[FlextTypes.Core.Dict]:
        """Alias for to_api_payload for backward compatibility.

        Returns:
            FlextResult containing serialized configuration data

        """
        return self.to_api_payload()

    def validate_runtime_requirements(self) -> FlextResult[None]:
        """Validate configuration meets runtime requirements.

        Returns:
            FlextResult indicating validation success or specific failures

        """
        errors = []

        # Check required fields are not empty
        if not self.app_name.strip():
            errors.append("app_name cannot be empty")

        if not self.name.strip():
            errors.append("name cannot be empty")

        # Validate version format
        if (
            not self.version
            or len(self.version.split(".")) < _SEMANTIC_VERSION_MIN_PARTS
        ):
            errors.append("version must follow semantic versioning (x.y.z)")

        if errors:
            error_msg = "; ".join(errors)
            return FlextResult[None].fail(
                f"Runtime validation failed: {error_msg}",
                error_code="CONFIG_RUNTIME_VALIDATION_ERROR",
            )

        return FlextResult[None].ok(None)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for configuration consistency.

        Returns:
            FlextResult indicating validation success or specific failures

        """
        errors = []

        # Production environment validation
        if self.environment == "production":
            if self.debug and self.config_source != "default":
                errors.append(
                    "Debug mode in production requires explicit configuration"
                )

            if self.max_workers < _MIN_PRODUCTION_WORKERS:
                errors.append("Production environment should have at least 2 workers")

        # Performance consistency checks
        if (
            self.timeout_seconds > _HIGH_TIMEOUT_THRESHOLD
            and self.max_workers < _MIN_WORKERS_FOR_HIGH_TIMEOUT
        ):
            errors.append(
                "High timeout with low worker count may cause performance issues"
            )

        # Resource validation
        if self.max_workers > _MAX_WORKERS_THRESHOLD:
            errors.append("Worker count above 50 may cause resource exhaustion")

        if errors:
            error_msg = "; ".join(errors)
            return FlextResult[None].fail(
                f"Business rule validation failed: {error_msg}",
                error_code="CONFIG_BUSINESS_RULE_ERROR",
            )

        return FlextResult[None].ok(None)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent modification of sealed configuration fields."""
        if (
            getattr(self, "_sealed", False)
            and name in cast("BaseModel", self).model_fields
            and not name.startswith("_")
        ):
            msg = f"Cannot modify field '{name}' - configuration is sealed"
            raise AttributeError(msg)

        super().__setattr__(name, value)

    def to_dict(self) -> FlextTypes.Core.Dict:
        """Export configuration as dictionary.

        Returns:
            Dictionary representation of configuration

        """
        return cast("BaseModel", self).model_dump(
            exclude={"_metadata", "_sealed"},
        )

    def to_json(
        self,
        *,
        indent: int | None = None,
        by_alias: bool = True,
    ) -> str:
        """Export configuration as JSON string with consistent formatting."""
        return cast("BaseModel", self).model_dump_json(
            exclude={"_metadata", "_sealed"},
            by_alias=by_alias,
            indent=indent,
        )

    # =============================================================================
    # FACTORY METHODS FOR SPECIALIZED CONFIGURATIONS
    # =============================================================================

    @classmethod
    def create_web_service_config(
        cls,
        *,
        app_name: str = "flext-web-service",
        environment: str = "development",
        port: int = 8000,
        enable_cors: bool = True,
        **overrides: object,
    ) -> FlextResult[Self]:
        """Create optimized configuration for web service applications.

        Args:
            app_name: Name of the web service
            environment: Deployment environment
            port: Service port
            enable_cors: Whether to enable CORS
            **overrides: Additional configuration overrides

        Returns:
            FlextResult containing configured web service instance

        """
        config_data = {
            "app_name": app_name,
            "environment": environment,
            "port": port,
            "enable_auth": True,
            "enable_rate_limiting": True,
            "enable_metrics": True,
            "enable_caching": True,
            "max_workers": 4,
            "timeout_seconds": 30,
            "health_check_interval": 30,
            "cors_origins": ["*"] if enable_cors else [],
        }
        config_data.update(overrides)

        result = cls.create(constants=config_data)
        if result.is_success:
            instance = result.value
            instance._metadata["profile"] = _PROFILE_WEB_SERVICE
            instance._metadata["created_with"] = "web_service_factory"

        return result

    @classmethod
    def create_data_processor_config(
        cls,
        *,
        app_name: str = "flext-data-processor",
        environment: str = "development",
        batch_size: int = 1000,
        **overrides: object,
    ) -> FlextResult[Self]:
        """Create optimized configuration for data processing applications.

        Args:
            app_name: Name of the data processor
            environment: Deployment environment
            batch_size: Processing batch size
            **overrides: Additional configuration overrides

        Returns:
            FlextResult containing configured data processor instance

        """
        config_data = {
            "app_name": app_name,
            "environment": environment,
            "enable_metrics": True,
            "enable_caching": True,
            "enable_auth": False,
            "max_workers": 8,  # Higher for data processing
            "timeout_seconds": 300,  # Longer for data operations
            "database_pool_size": 10,
            "database_timeout": 60,
            "message_queue_max_retries": 5,
            "health_check_interval": 60,
        }
        config_data.update(overrides)

        result = cls.create(constants=config_data)
        if result.is_success:
            instance = result.value
            instance._metadata["profile"] = _PROFILE_DATA_PROCESSOR
            instance._metadata["created_with"] = "data_processor_factory"
            instance._metadata["batch_size"] = str(batch_size)

        return result

    @classmethod
    def create_microservice_config(
        cls,
        *,
        app_name: str = "flext-microservice",
        environment: str = "development",
        port: int = 8080,
        **overrides: object,
    ) -> FlextResult[Self]:
        """Create optimized configuration for microservice applications.

        Args:
            app_name: Name of the microservice
            environment: Deployment environment
            port: Service port
            **overrides: Additional configuration overrides

        Returns:
            FlextResult containing configured microservice instance

        """
        config_data = {
            "app_name": app_name,
            "environment": environment,
            "port": port,
            "enable_auth": True,
            "enable_rate_limiting": True,
            "enable_circuit_breaker": True,
            "enable_metrics": True,
            "enable_caching": True,
            "max_workers": 4,
            "timeout_seconds": 15,  # Faster for microservices
            "health_check_interval": 15,
            "metrics_port": 9090,
        }
        config_data.update(overrides)

        result = cls.create(constants=config_data)
        if result.is_success:
            instance = result.value
            instance._metadata["profile"] = _PROFILE_MICROSERVICE
            instance._metadata["created_with"] = "microservice_factory"

        return result

    @classmethod
    def create_api_client_config(
        cls,
        *,
        app_name: str = "flext-api-client",
        base_url: str = "https://api.example.com",
        api_key: str = "",
        **overrides: object,
    ) -> FlextResult[Self]:
        """Create optimized configuration for API client applications.

        Args:
            app_name: Name of the API client
            base_url: Target API base URL
            api_key: API authentication key
            **overrides: Additional configuration overrides

        Returns:
            FlextResult containing configured API client instance

        """
        config_data = {
            "app_name": app_name,
            "base_url": base_url,
            "api_key": api_key,
            "enable_metrics": True,
            "enable_caching": True,
            "enable_circuit_breaker": True,
            "max_workers": 2,  # Lower for client apps
            "timeout_seconds": 60,  # Higher for external API calls
            "message_queue_max_retries": 3,
        }
        config_data.update(overrides)

        result = cls.create(constants=config_data)
        if result.is_success:
            instance = result.value
            instance._metadata["profile"] = _PROFILE_API_CLIENT
            instance._metadata["created_with"] = "api_client_factory"

        return result

    @classmethod
    def create_batch_job_config(
        cls,
        *,
        app_name: str = "flext-batch-job",
        environment: str = "development",
        **overrides: object,
    ) -> FlextResult[Self]:
        """Create optimized configuration for batch job applications.

        Args:
            app_name: Name of the batch job
            environment: Deployment environment
            **overrides: Additional configuration overrides

        Returns:
            FlextResult containing configured batch job instance

        """
        config_data = {
            "app_name": app_name,
            "environment": environment,
            "enable_auth": False,
            "enable_metrics": True,
            "enable_caching": False,  # Usually not needed for batch jobs
            "max_workers": 16,  # High for batch processing
            "timeout_seconds": 3600,  # Very high for long-running jobs
            "database_pool_size": 20,
            "database_timeout": 120,
            "health_check_interval": 300,  # Less frequent for batch jobs
        }
        config_data.update(overrides)

        result = cls.create(constants=config_data)
        if result.is_success:
            instance = result.value
            instance._metadata["profile"] = _PROFILE_BATCH_JOB
            instance._metadata["created_with"] = "batch_job_factory"

        return result

    # =============================================================================
    # ADVANCED UTILITY METHODS
    # =============================================================================

    def get_profile(self) -> str:
        """Get the configuration profile if set.

        Returns:
            Configuration profile name or 'custom' if none set

        """
        return self._metadata.get("profile", "custom")

    def is_production_ready(self) -> FlextResult[bool]:
        """Check if configuration is ready for production deployment.

        Returns:
            FlextResult containing boolean indicating production readiness

        """
        try:
            errors = []

            # Check critical production settings
            if self.environment != "production":
                errors.append("Environment must be 'production'")

            if self.debug:
                errors.append("Debug mode must be disabled in production")

            if self.log_level not in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
                errors.append("Log level too verbose for production")

            # Check security requirements
            if self.enable_auth and not self.api_key.strip():
                errors.append("API key required when authentication is enabled")

            # Check performance settings
            if self.max_workers < _MIN_PRODUCTION_WORKERS:
                errors.append(
                    f"Production requires at least {_MIN_PRODUCTION_WORKERS} workers"
                )

            if errors:
                return FlextResult[bool].fail(
                    f"Production readiness check failed: {'; '.join(errors)}",
                    error_code="PRODUCTION_READINESS_ERROR",
                )

            return FlextResult[bool].ok(data=True)

        except Exception as error:
            return FlextResult[bool].fail(
                f"Production readiness check failed: {error}",
                error_code="PRODUCTION_READINESS_ERROR",
            )

    def get_connection_string(self, service_type: str) -> FlextResult[str]:
        """Generate connection string for various service types.

        Args:
            service_type: Type of service ('database', 'message_queue', 'api')

        Returns:
            FlextResult containing connection string

        """
        try:
            if service_type == "database" and self.database_url:
                return FlextResult[str].ok(self.database_url)
            if service_type == "message_queue" and self.message_queue_url:
                return FlextResult[str].ok(self.message_queue_url)
            if service_type == "api":
                api_url = f"{self.base_url.rstrip('/')}"
                if self.api_key:
                    return FlextResult[str].ok(f"{api_url}?api_key={self.api_key}")
                return FlextResult[str].ok(api_url)
            return FlextResult[str].fail(
                f"No connection configuration found for service type: {service_type}",
                error_code="CONNECTION_STRING_ERROR",
            )
        except Exception as error:
            return FlextResult[str].fail(
                f"Failed to generate connection string: {error}",
                error_code="CONNECTION_STRING_ERROR",
            )

    def get_feature_flags(self) -> dict[str, bool]:
        """Get all feature flags as a dictionary.

        Returns:
            Dictionary of feature flag names and their values

        """
        return {
            "metrics": self.enable_metrics,
            "caching": self.enable_caching,
            "auth": self.enable_auth,
            "rate_limiting": self.enable_rate_limiting,
            "circuit_breaker": self.enable_circuit_breaker,
        }

    def apply_environment_overrides(
        self, env_overrides: FlextTypes.Core.Dict
    ) -> FlextResult[None]:
        """Apply environment-specific configuration overrides.

        Args:
            env_overrides: Dictionary of configuration overrides

        Returns:
            FlextResult indicating success or failure

        """
        if self.is_sealed():
            return FlextResult[None].fail(
                "Cannot apply overrides to sealed configuration",
                error_code="CONFIG_SEALED_ERROR",
            )

        try:
            for key, value in env_overrides.items():
                if hasattr(self, key) and key in cast("BaseModel", self).model_fields:
                    setattr(self, key, value)

            # Track the override in metadata
            self._metadata["overrides_applied"] = "true"
            self._metadata["override_count"] = str(len(env_overrides))

            return FlextResult[None].ok(None)

        except Exception as error:
            return FlextResult[None].fail(
                f"Failed to apply environment overrides: {error}",
                error_code="OVERRIDE_APPLICATION_ERROR",
            )

    def validate_service_dependencies(
        self, required_services: FlextTypes.Core.StringList
    ) -> FlextResult[dict[str, bool]]:
        """Validate that required service dependencies are properly configured.

        Args:
            required_services: List of required service names

        Returns:
            FlextResult containing validation results for each service

        """
        try:
            results = {}

            for service in required_services:
                if service == "database":
                    results[service] = bool(self.database_url.strip())
                elif service == "message_queue":
                    results[service] = bool(self.message_queue_url.strip())
                elif service == "api":
                    results[service] = bool(self.base_url.strip())
                elif service == "metrics":
                    results[service] = self.enable_metrics
                elif service == "auth":
                    results[service] = self.enable_auth and bool(self.api_key.strip())
                else:
                    results[service] = False

            return FlextResult[dict[str, bool]].ok(results)

        except Exception as error:
            return FlextResult[dict[str, bool]].fail(
                f"Service dependency validation failed: {error}",
                error_code="SERVICE_DEPENDENCY_ERROR",
            )

    def create_child_config(self, **overrides: object) -> FlextResult[Self]:
        """Create a child configuration with specific overrides.

        Args:
            **overrides: Configuration overrides for the child config

        Returns:
            FlextResult containing new configuration instance

        """
        try:
            # Get current config as dict
            current_config = self.to_dict()

            # Apply overrides
            current_config.update(overrides)

            # Create new instance
            result = self.__class__.create(constants=current_config)

            if result.is_success:
                child_instance = result.value
                child_instance._metadata["parent_profile"] = self.get_profile()
                child_instance._metadata["created_with"] = "child_config_factory"
                child_instance._metadata["override_keys"] = ",".join(overrides.keys())

            return result

        except Exception as error:
            return FlextResult[Self].fail(
                f"Failed to create child configuration: {error}",
                error_code="CHILD_CONFIG_ERROR",
            )

    @classmethod
    def create_from_template(
        cls,
        template_name: str,
    ) -> FlextResult[Self]:
        """Create configuration from a predefined template.

        Args:
            template_name: Name of the template to use

        Returns:
            FlextResult containing configured instance

        """
        try:
            if template_name == "web_service":
                return cls.create_web_service_config()
            if template_name == "data_processor":
                return cls.create_data_processor_config()
            if template_name == "microservice":
                return cls.create_microservice_config()
            if template_name == "api_client":
                return cls.create_api_client_config()
            if template_name == "batch_job":
                return cls.create_batch_job_config()
            available = (
                "web_service, data_processor, microservice, api_client, batch_job"
            )
            return FlextResult[Self].fail(
                f"Unknown template '{template_name}'. Available templates: {available}",
                error_code="TEMPLATE_NOT_FOUND",
            )
        except Exception as error:
            return FlextResult[Self].fail(
                f"Failed to create configuration from template '{template_name}': {error}",
                error_code="TEMPLATE_CREATION_ERROR",
            )


__all__ = ["FlextConfig"]
