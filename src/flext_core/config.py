"""Enterprise configuration management with type-safe validation and environment integration.

This module provides comprehensive configuration management for the FLEXT ecosystem using
Pydantic v2 BaseModel patterns with FlextResult error handling, environment variable
integration, JSON serialization/deserialization, and business rule validation for
enterprise-grade configuration reliability.

Module Organization:
    Core Configuration: FlextConfig with nested SystemDefaults and Settings
    Environment Integration: BaseSettings with automatic environment variable loading
    Validation System: Multi-layer validation with business rules and type checking
    Utility Functions: Safe loading, serialization, and configuration operations
    Type Safety: FlextTypes.Core integration for consistent typing

Classes:
    FlextConfig: Main configuration management class with consolidated functionality
        └── SystemDefaults: Centralized system defaults organized by domain
            ├── Security: Password length, secret key requirements, authentication
            ├── Network: Timeouts, retries, connection limits, service discovery
            ├── Pagination: Page sizes, limits, sorting defaults
            ├── Logging: Log levels, formats, rotation settings
            └── Environment: Development, staging, production configurations
        └── Settings(BaseSettings): Environment-aware configuration loading
            • model_config: SettingsConfigDict with environment variable mapping
            • field validation: Automatic type coercion and constraint checking
            • nested configuration: Support for complex hierarchical configurations
        └── BaseConfigModel: Backward-compatibility alias for Settings

Functions:
    validate_business_rules(config_data) -> FlextResult[dict]
        Validate business-specific configuration rules and constraints

    create_with_validation(config_data) -> FlextResult[FlextConfig]
        Create validated configuration instance with comprehensive error handling

    load_from_file(file_path, validate_business=True) -> FlextResult[FlextConfig]
        Load configuration from JSON file with optional business validation

    load_and_validate_from_file(file_path, required_keys) -> FlextResult[FlextConfig]
        Load configuration with required key validation and business rules

    safe_load_json_file(file_path) -> FlextResult[dict]
        Safely load JSON file with comprehensive error handling

    safe_get_env_var(var_name, default_value) -> FlextResult[str]
        Safely retrieve environment variable with default fallback

Type Integration:
    dict[str, object]: Configuration dictionary type
    FlextTypes.Core.Value: Generic configuration value type
    FlextTypes.Core.String: String configuration type
    FlextTypes.Core.Serializer: JSON serialization interface

Integration with FlextCore:
    >>> from flext_core import FlextConfig, FlextResult
    >>> from flext_core.core import FlextCore
    >>> # Environment-aware configuration loading
    >>> core = FlextCore.get_instance()
    >>> config_result = FlextConfig.load_from_file("config/production.json")
    >>> if config_result.success:
    ...     config = config_result.value
    ...     core.logger.info(f"Configuration loaded: {config.model_dump()}")
    >>> # Business rule validation
    >>> validation_result = FlextConfig.validate_business_rules(
    ...     {
    ...         "database_url": "postgresql://localhost/prod",
    ...         "secret_key": "secure-key-with-sufficient-length",
    ...         "log_level": "INFO",
    ...     }
    ... )

Environment Configuration Examples:
    >>> # Development configuration
    >>> dev_config = FlextConfig.Settings(
    ...     debug=True,
    ...     log_level="DEBUG",
    ...     database_url="sqlite:///dev.db",
    ...     enable_profiling=True,
    ... )
    >>> # Production configuration with environment variables
    >>> # Set via: export FLEXT_DATABASE_URL="postgresql://prod-server/db"
    >>> prod_config = FlextConfig.Settings()  # Automatically loads from env
    >>> # Validate and apply configuration
    >>> validation = FlextConfig.validate_business_rules(prod_config.model_dump())
    >>> if validation.success:
    ...     core.configure_from_settings(prod_config)

Security Configuration Examples:
    >>> # Access security defaults
    >>> max_password = FlextConfig.SystemDefaults.Security.max_password_length
    >>> min_secret_key = (
    ...     FlextConfig.SystemDefaults.Security.min_secret_key_length_strong
    ... )
    >>> # Validate security settings
    >>> security_config = {
    ...     "password_policy": {"min_length": 12, "require_special": True},
    ...     "secret_key": "very-long-secret-key-for-production-security",
    ...     "token_expiry_seconds": 3600,
    ... }
    >>> security_result = FlextConfig.validate_business_rules(security_config)

File-Based Configuration:
    >>> # Load from JSON with validation
    >>> config_file = "configs/api-service.json"
    >>> required_keys = ["database_url", "secret_key", "service_name"]
    >>> config_result = FlextConfig.load_and_validate_from_file(
    ...     config_file, required_keys
    ... )
    >>> # Handle configuration errors gracefully
    >>> if config_result.failure:
    ...     core.logger.error(f"Configuration error: {config_result.error}")
    ...     fallback_config = FlextConfig.SystemDefaults()

Nested Configuration Structure:
    >>> # Complex hierarchical configuration
    >>> complex_config = FlextConfig(
    ...     database={
    ...         "primary": {"url": "postgresql://primary/db", "pool_size": 20},
    ...         "readonly": {"url": "postgresql://readonly/db", "pool_size": 10},
    ...     },
    ...     services={
    ...         "auth": {"endpoint": "https://auth.example.com", "timeout": 30},
    ...         "metrics": {"endpoint": "https://metrics.example.com", "enabled": True},
    ...     },
    ...     feature_flags={"new_ui": True, "beta_features": False},
    ... )

Notes:
    - All configuration operations return FlextResult for type-safe error handling
    - Environment variable integration follows FLEXT_* naming convention
    - Business rule validation enforces enterprise security and operational requirements
    - SystemDefaults provide consistent baseline values across FLEXT ecosystem
    - Configuration serialization supports both JSON and environment variable formats
    - Backward compatibility maintained through BaseConfigModel alias
    - Integration with FlextCore enables centralized configuration management

"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextConfig(BaseModel):
    """Main FLEXT configuration class using pure Pydantic BaseModel patterns.

    Core configuration model for the FLEXT ecosystem providing type-safe
    configuration with automatic validation, serialization, and environment
    variable integration.

    Attributes:
        SystemDefaults: Nested class containing centralized system defaults.
        Settings: BaseSettings subclass for environment-aware configuration.

    Methods:
        validate_business_rules: Validate business-specific configuration rules.
        create_with_validation: Create validated configuration instance.
        load_from_file: Load configuration from JSON file.

    """

    # =========================================================================
    # NESTED CLASSES - Core configuration components consolidated
    # =========================================================================

    class SystemDefaults:
        """Centralized system defaults for the FLEXT ecosystem.

        Provides nested classes containing default values for security,
        network, pagination, logging, and environment configuration.
        Values are sourced from FlextConstants for consistency.
        """

        class Security:
            """Security-related configuration defaults.

            Attributes:
                max_password_length: Maximum allowed password length.
                max_username_length: Maximum allowed username length.
                min_secret_key_length_strong: Minimum length for strong secret keys.
                min_secret_key_length_adequate: Minimum length for adequate secret keys.

            """

            # Use FlextConstants.Validation directly for all validation constants
            max_password_length = FlextConstants.Validation.MAX_PASSWORD_LENGTH
            max_username_length = FlextConstants.Validation.MAX_SERVICE_NAME_LENGTH // 2
            min_secret_key_length_strong = (
                FlextConstants.Validation.MIN_SECRET_KEY_LENGTH * 2
            )
            min_secret_key_length_adequate = (
                FlextConstants.Validation.MIN_SECRET_KEY_LENGTH
            )

        class Network:
            """Network and service defaults.

            Attributes:
                TIMEOUT: Default network timeout in seconds.
                RETRIES: Maximum number of retry attempts.
                CONNECTION_TIMEOUT: Connection establishment timeout.

            """

            TIMEOUT = FlextConstants.Network.DEFAULT_TIMEOUT
            RETRIES = FlextConstants.Defaults.MAX_RETRIES
            CONNECTION_TIMEOUT = FlextConstants.Network.CONNECTION_TIMEOUT

        class Pagination:
            """Pagination defaults.

            Attributes:
                PAGE_SIZE: Default number of items per page.
                MAX_PAGE_SIZE: Maximum allowed items per page.

            """

            PAGE_SIZE = FlextConstants.Defaults.PAGE_SIZE
            MAX_PAGE_SIZE = FlextConstants.Defaults.MAX_PAGE_SIZE

        class Logging:
            """Logging configuration defaults.

            Attributes:
                LOG_LEVEL: Default logging level from observability constants.

            """

            LOG_LEVEL = FlextConstants.Observability.DEFAULT_LOG_LEVEL

        class Environment:
            """Environment configuration defaults.

            Attributes:
                DEFAULT_ENV: Default environment name from configuration constants.

            """

            DEFAULT_ENV = FlextConstants.Config.DEFAULT_ENVIRONMENT

    class Settings(BaseSettings):
        """Environment-aware configuration using Pydantic BaseSettings.

        Foundation class for configuration that automatically loads from environment
        variables with FLEXT_ prefix. Provides validation and business rule checking.

        Configuration:
            env_prefix: "FLEXT_" - All environment variables must start with this.
            env_file: ".env" - Loads from .env file if present.
            validate_assignment: True - Validates on field assignment.
            extra: "ignore" - Ignores unknown fields.

        Methods:
            validate_business_rules: Override for custom validation logic.
            serialize_settings_for_api: JSON serializer with metadata.
            create_with_validation: Class method to create validated instances.

        """

        model_config = SettingsConfigDict(
            # Environment integration
            env_prefix="FLEXT_",
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            # Validation and safety
            validate_assignment=True,
            extra="ignore",
            str_strip_whitespace=True,
            # JSON schema generation
            json_schema_extra={
                "examples": [],
                "description": "FLEXT settings with environment variable support",
            },
        )

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate business-specific configuration rules.

            Returns:
                FlextResult[None] indicating validation success or failure.

            Note:
                Default implementation always succeeds. Override in subclasses
                to implement specific business validation logic.

            """
            # Default implementation: no business rules => success
            return FlextResult[None].ok(None)

        # Note: Do not use field_serializer for model_config; it's not a model field.

        @model_serializer(mode="wrap", when_used="json")
        def serialize_settings_for_api(
            self,
            serializer: FlextTypes.Config.ConfigSerializer,
            info: SerializationInfo,
        ) -> FlextTypes.Config.ConfigDict:
            """Serialize settings for API output with metadata.

            Args:
                serializer: Pydantic serializer function.
                info: Serialization context information.

            Returns:
                Dictionary with settings data plus _settings metadata including
                type, environment loading status, and API version information.

            """
            _ = info  # Acknowledge parameter for future use
            _ = serializer  # Acknowledge serializer parameter for Pydantic compatibility
            # Get the base dict representation first
            base_data = self.model_dump()
            # Ensure all values are of the correct types for ConfigDict
            data: FlextTypes.Config.ConfigDict = {}
            for key, value in base_data.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    # Cast to ConfigValue after type check
                    data[key] = cast("FlextTypes.Config.ConfigValue", value)
                else:
                    data[key] = str(value)

            # Add settings-specific API metadata
            data["_settings"] = {
                "type": "FlextConfig",
                "env_loaded": True,
                "validation_enabled": True,
                "api_version": "v2",
                "serialization_format": FlextConstants.InfrastructureMessages.CONFIG_FORMAT_JSON,
            }
            return data

        @classmethod
        def create_with_validation(
            cls,
            overrides: Mapping[str, FlextTypes.Core.Value] | None = None,
            **kwargs: FlextTypes.Core.Value,
        ) -> FlextResult[FlextConfig.Settings]:
            """Create settings instance with validation and proper override handling."""
            try:
                # Start with default instance
                instance = cls()

                # Prepare overrides dict - support both overrides parameter and kwargs
                all_overrides: dict[str, object] = {}
                if overrides:
                    # Mapping -> dict
                    all_overrides.update(dict(overrides))
                all_overrides.update(kwargs)

                # Apply overrides if any provided
                if all_overrides:
                    # Get current values as dict
                    current_data = instance.model_dump()
                    # Update with overrides
                    current_data.update(all_overrides)
                    # Create new instance with merged data
                    instance = cls.model_validate(current_data)

                validation_result = instance.validate_business_rules()
                if validation_result.is_failure:
                    return FlextResult[FlextConfig.Settings].fail(
                        validation_result.error
                        or FlextConstants.Messages.VALIDATION_FAILED
                    )
                return FlextResult[FlextConfig.Settings].ok(instance)
            except Exception as e:
                return FlextResult[FlextConfig.Settings].fail(
                    f"Settings creation failed: {e}"
                )

    class BaseConfigModel(Settings):
        """Backward-compatible base for configuration models.

        Subclassing this class is equivalent to subclassing ``FlextConfig.Settings``.
        """

    # Core identification
    name: str = Field(
        default=FlextConstants.Core.NAME.lower(), description="Configuration name"
    )
    version: str = Field(
        default=FlextConstants.Core.VERSION, description="Configuration version"
    )
    description: str = Field(
        default="FLEXT configuration",
        description="Configuration description",
    )

    # Environment settings
    environment: FlextTypes.Config.Environment = Field(
        default="development",
        description="Environment name (development, staging, production)",
    )
    debug: bool = Field(default=False, description="Debug mode enabled")

    # Configuration source tracking
    config_source: str = Field(
        default="default",
        description="Source of configuration (file, env, cli, default)",
    )
    config_priority: int = Field(
        default=FlextConstants.Config.CONSTANTS_PRIORITY,
        description="Configuration provider priority",
    )
    config_namespace: str = Field(
        default="flext",
        description="Configuration namespace for isolation",
    )

    # Core operational settings
    log_level: str = Field(
        default=FlextConstants.Observability.DEFAULT_LOG_LEVEL,
        description="Logging level",
    )
    timeout: int = Field(
        default=FlextConstants.Network.DEFAULT_TIMEOUT,
        description="Default timeout in seconds",
    )
    retries: int = Field(
        default=FlextConstants.Defaults.MAX_RETRIES, description="Default retry count"
    )
    page_size: int = Field(
        default=FlextConstants.Defaults.PAGE_SIZE, description="Default page size"
    )

    # Feature flags
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value with shorthand mapping.

        Args:
            v: Environment value to validate.

        Returns:
            Normalized environment name.

        Raises:
            ValueError: If environment is not in allowed values from FlextConstants.Config.ENVIRONMENTS.

        """
        mapping = {
            "dev": FlextConstants.Config.ENVIRONMENTS[0],  # development
            "prod": FlextConstants.Config.ENVIRONMENTS[2],  # production
            "stage": FlextConstants.Config.ENVIRONMENTS[1],  # staging
            "stg": FlextConstants.Config.ENVIRONMENTS[1],  # staging
        }
        normalized = mapping.get(v.lower(), v)
        allowed_set = set(FlextConstants.Config.ENVIRONMENTS)
        if normalized not in allowed_set:
            msg = f"Environment must be one of: {FlextConstants.Config.ENVIRONMENTS}"
            raise ValueError(msg)
        return normalized

    @field_validator("config_source")
    @classmethod
    def validate_config_source(cls, v: str) -> str:
        """Validate configuration source using FlextConstants.Config.ConfigSource values.

        Args:
            v: Configuration source to validate.

        Returns:
            Valid configuration source string.

        Raises:
            ValueError: If source is not in FlextConstants.Config.ConfigSource values.

        """
        allowed_sources = {
            source.value for source in FlextConstants.Config.ConfigSource
        }
        if v not in allowed_sources:
            msg = f"Config source must be one of: {list(allowed_sources)}"
            raise ValueError(msg)
        return v

    @field_validator("config_priority")
    @classmethod
    def validate_config_priority(cls, v: int) -> int:
        """Validate configuration priority within allowed range.

        Args:
            v: Configuration priority value to validate.

        Returns:
            Valid configuration priority integer.

        Raises:
            ValueError: If priority is not within CLI_PRIORITY to CONSTANTS_PRIORITY range.

        """
        min_priority = FlextConstants.Config.CLI_PRIORITY
        max_priority = FlextConstants.Config.CONSTANTS_PRIORITY
        if not (min_priority <= v <= max_priority):
            msg = f"Config priority must be between {min_priority} and {max_priority}"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level against allowed values.

        Args:
            v: Log level string to validate.

        Returns:
            Uppercase log level string.

        Raises:
            ValueError: If log level is not in FlextConstants.Config.LogLevel values.

        """
        allowed = {level.value for level in FlextConstants.Config.LogLevel}
        if v.upper() not in allowed:
            msg = f"Log level must be one of: {list(allowed)}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("timeout", "retries", "page_size")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        """Validate positive integer values for timeout, retries, and page_size fields.

        Args:
            v: Integer value to validate.

        Returns:
            Validated positive integer value.

        Raises:
            ValueError: If value is less than or equal to 0.

        """
        if v <= 0:
            msg = FlextConstants.Messages.INVALID_INPUT
            raise ValueError(msg)
        return v

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules specific to FLEXT configuration.

        Validates critical business constraints including:
        - Debug mode not allowed in production environment
        - Critical fields (database_url, key) cannot be None when present
        - Extra field validation for configuration integrity

        Returns:
            FlextResult[None]: Success if all business rules pass, failure with error message otherwise.

        """
        if self.debug and self.environment == FlextConstants.Config.ENVIRONMENTS[2]:
            return FlextResult[None].fail(
                FlextConstants.ValidationSystem.BUSINESS_RULE_VIOLATED,
            )

        # Validate critical fields are not None when they exist as extra fields
        extra_data = (
            dict(self.__pydantic_extra__.items()) if self.__pydantic_extra__ else {}
        )

        # Check for None values in critical fields
        critical_fields = ["database_url", "key"]
        critical_none_fields = {
            field
            for field in critical_fields
            if field in extra_data and extra_data[field] is None
        }
        if critical_none_fields:
            fields_str = ", ".join(sorted(critical_none_fields))
            return FlextResult[None].fail(
                f"Config validation failed for {fields_str}",
            )

        return FlextResult[None].ok(None)

    @field_serializer("environment", when_used="json")
    def serialize_environment(self, value: str) -> dict[str, object]:
        """Serialize environment field with metadata for JSON serialization.

        Args:
            value: Environment string value.

        Returns:
            Dictionary containing environment name and computed metadata including
            production status, debug allowance, and configuration profile.

        """
        return {
            "name": value,
            "is_production": value == FlextConstants.Config.ENVIRONMENTS[2],
            "debug_allowed": value != FlextConstants.Config.ENVIRONMENTS[2],
            "config_profile": f"flext-{value}",
        }

    @field_serializer("log_level", when_used="json")
    def serialize_log_level(self, value: str) -> dict[str, object]:
        """Serialize log level field with metadata for JSON serialization.

        Args:
            value: Log level string value.

        Returns:
            Dictionary containing log level, numeric level, verbosity flag,
            and production safety indicator.

        """
        level_hierarchy = FlextConstants.Config.LogLevel.get_numeric_levels()
        return {
            "level": value,
            "numeric_level": level_hierarchy.get(
                value, FlextConstants.Config.LogLevel.get_numeric_levels()["INFO"]
            ),
            "verbose": value == FlextConstants.Observability.LOG_LEVELS[1],
            "production_safe": value
            in {
                FlextConstants.Observability.LOG_LEVELS[2],
                FlextConstants.Observability.LOG_LEVELS[3],
                FlextConstants.Observability.LOG_LEVELS[4],
                FlextConstants.Observability.LOG_LEVELS[5],
            },
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_config_for_api(
        self,
        serializer: FlextTypes.Core.Serializer,
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Serialize complete configuration model for API output with metadata.

        Args:
            serializer: Pydantic serializer function for base model serialization.
            info: Serialization context information.

        Returns:
            Dictionary containing serialized configuration data with additional
            API-specific metadata including version, timestamps, and environment flags.

        """
        _ = info  # Acknowledge parameter for future use
        data: dict[str, object] = serializer(self)
        # Add config-specific API metadata
        if data and hasattr(data, "get"):
            data["_config"] = {
                "type": "FlextConfig",
                "version": data.get("version", "1.0.0"),
                "environment": data.get("environment", "development"),
                "features_enabled": {
                    "caching": data.get("enable_caching", True),
                    "metrics": data.get("enable_metrics", True),
                    "tracing": data.get("enable_tracing", False),
                },
                "api_version": "v2",
                "cross_service_ready": True,
            }
        return data

    @classmethod
    def create_complete_config(
        cls,
        config_data: Mapping[str, object],
        defaults: dict[str, object] | None = None,
        *,
        apply_defaults: bool = True,
        validate_all: bool = True,
    ) -> FlextResult[dict[str, object]]:
        """Create complete configuration with defaults and validation.

        Args:
            config_data: Base configuration data mapping.
            defaults: Optional default values to apply before user configuration.
            apply_defaults: Whether to apply default values from model or provided defaults.
            validate_all: Whether to run business rule validation after creation.

        Returns:
            FlextResult containing validated configuration dictionary on success,
            or error message on validation failure.

        """
        try:
            # Convert config_data to dict for manipulation
            working_config: dict[str, object] = dict(config_data)

            # Apply defaults if requested
            if apply_defaults:
                if defaults:
                    # Use provided defaults
                    working_config = {**defaults, **working_config}
                else:
                    # Use model defaults
                    default_config = cls().model_dump()
                    working_config = {**default_config, **working_config}

            # Create and validate instance
            instance = cls.model_validate(working_config)

            if validate_all:
                validation_result = instance.validate_business_rules()
                if validation_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        validation_result.error
                        or FlextConstants.Messages.VALIDATION_FAILED,
                    )

            return FlextResult[dict[str, object]].ok(instance.model_dump())
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to create complete config: {e}"
            )

    @classmethod
    def load_and_validate_from_file(
        cls,
        file_path: str,
        required_keys: list[str] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Load and validate configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file.
            required_keys: Optional list of keys that must be present in configuration.

        Returns:
            FlextResult containing validated configuration dictionary on success,
            or error message if file loading or validation fails.

        """
        try:
            file_result = cls.safe_load_json_file(file_path)
            if file_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    file_result.error or "Failed to load file"
                )

            # file_result.value is typed as dict[str, object] on success
            data = file_result.value
            # safe_load_json_file already ensures data is dict[str, object] on success

            # Check for required keys if specified
            if required_keys is not None:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    missing_str = ", ".join(missing_keys)
                    return FlextResult[dict[str, object]].fail(
                        f"Missing required configuration keys: {missing_str}",
                    )

            instance = cls.model_validate(data)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or FlextConstants.Messages.VALIDATION_FAILED
                )

            return FlextResult[dict[str, object]].ok(instance.model_dump())
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to load and validate from file: {e}"
            )

    @classmethod
    def safe_load_from_dict(
        cls,
        config_data: Mapping[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Safely load and validate configuration from dictionary mapping.

        Args:
            config_data: Configuration data as key-value mapping.

        Returns:
            FlextResult containing validated configuration dictionary on success,
            or error message if validation fails.

        """
        try:
            instance = cls.model_validate(dict(config_data))
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or FlextConstants.Messages.VALIDATION_FAILED
                )

            return FlextResult[dict[str, object]].ok(instance.model_dump())
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to load from dict: {e}")

    @classmethod
    def merge_and_validate_configs(
        cls,
        base_config: Mapping[str, object],
        override_config: Mapping[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge base and override configurations with validation.

        Args:
            base_config: Base configuration mapping (lower priority).
            override_config: Override configuration mapping (higher priority).

        Returns:
            FlextResult containing merged and validated configuration dictionary,
            or error message if merge contains None values or validation fails.

        """
        try:
            merged = {**dict(base_config), **dict(override_config)}

            # Check for None values which are not allowed in config
            none_keys = [k for k, v in merged.items() if v is None]
            if none_keys:
                keys_str = ", ".join(none_keys)
                return FlextResult[dict[str, object]].fail(
                    f"Configuration cannot contain None values for keys: {keys_str}",
                )

            instance = cls.model_validate(merged)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or FlextConstants.Messages.VALIDATION_FAILED
                )

            return FlextResult[dict[str, object]].ok(instance.model_dump())
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to merge and validate configs: {e}"
            )

    @classmethod
    def get_env_with_validation(
        cls,
        env_var: str,
        *,
        validate_type: type = str,
        default: object = None,
        required: bool = False,
    ) -> FlextResult[str]:
        """Get environment variable with type validation and defaults.

        Args:
            env_var: Environment variable name to retrieve.
            validate_type: Expected type for validation (str, int, bool, etc.).
            default: Default value if environment variable is not set.
            required: Whether environment variable is required (fails if missing).

        Returns:
            FlextResult containing validated environment variable value as string,
            or error message if required variable is missing or validation fails.

        """
        try:
            env_result = cls.safe_get_env_var(
                env_var,
                str(default) if default is not None else None,
            )
            if env_result.is_failure and required:
                return env_result

            # Get value from result or use default
            if env_result.is_success:
                value: str = env_result.value
            else:
                value = str(default) if default is not None else ""
            if value == "" and default is not None:
                return FlextResult[str].ok(str(default))

            # Type validation
            if validate_type is str:
                return FlextResult[str].ok(value)

            # Type-specific validation using helper method
            result = cls._validate_type_value(value=value, validate_type=validate_type)
            if result.success:
                # result.value may be of various types; coerce to str for env API
                return FlextResult[str].ok(str(result.value))
            return FlextResult[str].fail(result.error or "Type validation failed")
        except Exception as e:
            return FlextResult[str].fail(f"Failed to get env with validation: {e}")

    @classmethod
    def _validate_type_value(
        cls,
        *,
        value: object,
        validate_type: type,
    ) -> FlextResult[object]:
        """Helper method to validate and convert value to specific type."""
        if validate_type is int:
            try:
                return FlextResult[object].ok(int(str(value)))
            except ValueError:
                return FlextResult[object].fail(f"Cannot convert '{value}' to int")
        elif validate_type is bool:
            if isinstance(value, str):
                return FlextResult[object].ok(
                    value.lower() in {"true", "1", "yes", "on"}
                )
            return FlextResult[object].ok(bool(value))
        else:
            # Default case for any other type
            return FlextResult[object].ok(value)

    @classmethod
    def validate_config_value(
        cls,
        value: object,
        validator: object,
        error_message: str = FlextConstants.Messages.VALIDATION_FAILED,
    ) -> FlextResult[bool]:
        """Validate a configuration value using a validator function."""
        try:
            if not callable(validator):
                return FlextResult[bool].fail("Validator must be callable")

            try:
                result = validator(value)
                if not result:
                    return FlextResult[bool].fail(error_message)
                return FlextResult[bool].ok(data=True)
            except Exception as e:
                return FlextResult[bool].fail(f"Validation error: {e}")
        except Exception as e:
            return FlextResult[bool].fail(f"Validation failed: {e}")

    @staticmethod
    def get_model_config(
        description: str = "Base configuration model",
        *,
        frozen: bool = True,
        extra: str = "forbid",
        validate_assignment: bool = True,
        use_enum_values: bool = True,
        str_strip_whitespace: bool = True,
        validate_all: bool = True,
        allow_reuse: bool = True,
    ) -> dict[str, object]:
        """Get model configuration parameters as a dictionary.

        Args:
            description: Model description for documentation.
            frozen: Whether model instances should be immutable after creation.
            extra: How to handle extra fields ("forbid", "ignore", "allow").
            validate_assignment: Whether to validate field assignments after creation.
            use_enum_values: Whether to use enum values instead of names in serialization.
            str_strip_whitespace: Whether to strip whitespace from string fields.
            validate_all: Whether to validate all fields on model creation.
            allow_reuse: Whether to allow model class reuse in recursive definitions.

        Returns:
            Dictionary containing Pydantic model configuration parameters.

        """
        return {
            "description": description,
            "frozen": frozen,
            "extra": extra,
            "validate_assignment": validate_assignment,
            "use_enum_values": use_enum_values,
            "str_strip_whitespace": str_strip_whitespace,
            "validate_all": validate_all,
            "allow_reuse": allow_reuse,
        }

    # =========================================================================
    # TIER 1 MODULE PATTERN - Consolidated static interface
    # =========================================================================

    @staticmethod
    def get_system_defaults() -> type[FlextConfig.SystemDefaults]:
        """Get reference to SystemDefaults class with centralized configuration values.

        Returns:
            FlextConfig.SystemDefaults class containing nested classes for Security,
            Network, Pagination, Logging, and Environment defaults sourced from FlextConstants.

        """
        return FlextConfig.SystemDefaults

    @staticmethod
    def get_env_var(
        var_name: str,
        default: str | None = None,
    ) -> FlextResult[str]:
        """Get environment variable safely with error handling.

        Args:
            var_name: Environment variable name to retrieve.
            default: Optional default value if variable is not set.

        Returns:
            FlextResult containing environment variable value or error message.

        """
        return FlextConfig.safe_get_env_var(var_name, default)

    @staticmethod
    def load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Load JSON file safely with validation and error handling.

        Args:
            file_path: Path to JSON file as string or Path object.

        Returns:
            FlextResult containing parsed JSON dictionary or error message
            if file not found, invalid JSON, or not a dictionary.

        """
        return FlextConfig.safe_load_json_file(file_path)

    @staticmethod
    def merge_config_dicts(
        base_config: dict[str, object],
        override_config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge base and override configuration dictionaries safely.

        Args:
            base_config: Base configuration dictionary (lower priority).
            override_config: Override configuration dictionary (higher priority).

        Returns:
            FlextResult containing merged configuration dictionary or error message
            if merge fails or contains None values.

        """
        return FlextConfig.merge_configs(base_config, override_config)

    @classmethod
    def create_settings(
        cls,
        overrides: Mapping[str, FlextTypes.Core.Value] | None = None,
        **kwargs: FlextTypes.Core.Value,
    ) -> FlextResult[FlextConfig.Settings]:
        """Create validated Settings instance with environment variable loading.

        Args:
            overrides: Optional mapping of override values for settings.
            **kwargs: Additional keyword arguments for setting overrides.

        Returns:
            FlextResult containing validated FlextConfig.Settings instance
            or error message if validation fails.

        """
        return cls.Settings.create_with_validation(overrides, **kwargs)

    @classmethod
    def create_validated_settings(
        cls,
        overrides: Mapping[str, FlextTypes.Core.Value] | None = None,
        **kwargs: FlextTypes.Core.Value,
    ) -> FlextResult[FlextConfig.Settings]:
        """Alias for create_settings for backward compatibility."""
        return cls.create_settings(overrides, **kwargs)

    # =========================================================================
    # COMPATIBILITY FACADES - Access to all config classes
    # =========================================================================

    # Class-level access to all configuration components (updated for nested classes)
    Defaults: ClassVar[type[SystemDefaults]] = SystemDefaults

    # =============================================================================
    # UTILITY FUNCTIONS (Foundation patterns - these should remain)
    # =============================================================================

    @staticmethod
    def safe_get_env_var(
        var_name: FlextTypes.Core.String,
        default: str | None = None,
    ) -> FlextResult[str]:
        """Safely get environment variable with optional default value.

        Foundation utility function providing safe environment variable access
        with proper error handling and FlextResult integration.

        Args:
            var_name: Environment variable name to retrieve.
            default: Optional default value if variable is not set.

        Returns:
            FlextResult containing environment variable value or error message
            if variable is not set and no default provided.

        """
        try:
            value = os.getenv(var_name, default)
            if value is None:
                return FlextResult[str].fail(f"Environment variable {var_name} not set")
            return FlextResult[str].ok(value)
        except Exception as e:
            return FlextResult[str].fail(f"{FlextConstants.Errors.CONFIG_ERROR}: {e}")

    @staticmethod
    def safe_load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Safely load JSON configuration file with validation.

        Foundation utility function providing safe JSON file loading with
        comprehensive error handling for file access, parsing, and type validation.

        Args:
            file_path: Path to JSON file as string or Path object.

        Returns:
            FlextResult containing parsed JSON dictionary or error message
            for file not found, invalid JSON format, or non-dictionary content.

        """
        try:
            with Path(file_path).open(encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return FlextResult[dict[str, object]].fail(
                    FlextConstants.Messages.TYPE_MISMATCH
                )

            return FlextResult[dict[str, object]].ok(cast("dict[str, object]", data))
        except FileNotFoundError:
            return FlextResult[dict[str, object]].fail(
                f"{FlextConstants.Errors.NOT_FOUND}: {file_path}"
            )
        except json.JSONDecodeError as e:
            return FlextResult[dict[str, object]].fail(
                f"{FlextConstants.Errors.SERIALIZATION_ERROR}: {e}"
            )
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"{FlextConstants.Errors.CONFIG_ERROR}: {e}"
            )

    @staticmethod
    def merge_configs(
        base_config: dict[str, object],
        override_config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge two configuration dictionaries with validation.

        Foundation utility function providing safe configuration dictionary merging
        with None value validation and proper error handling.

        Args:
            base_config: Base configuration dictionary (lower priority).
            override_config: Override configuration dictionary (higher priority).

        Returns:
            FlextResult containing merged configuration dictionary or error message
            if merge contains None values or operation fails.

        """
        try:
            merged = {**base_config, **override_config}

            # Validate for None values which are invalid
            for key, value in merged.items():
                if value is None:
                    return FlextResult[dict[str, object]].fail(
                        f"{FlextConstants.Messages.VALIDATION_FAILED} for {key}: {FlextConstants.Messages.NULL_DATA}",
                    )

            return FlextResult[dict[str, object]].ok(merged)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Config merge failed: {e}")


# Export only the classes and functions defined in this module
__all__ = [
    "FlextConfig",  # Main class
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
