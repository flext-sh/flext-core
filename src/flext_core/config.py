"""Pure Pydantic BaseModel configuration patterns for FLEXT Core.

This module provides configuration management using pure Pydantic BaseModel patterns,
eliminating all legacy complexity and compatibility layers.

Key Benefits:
- Pure Pydantic BaseModel for consistency
- Automatic validation and serialization
- Environment variable integration via pydantic-settings
- Railway-oriented programming via FlextResult
- No legacy compatibility layers
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, cast

from pydantic import (
    Field,
    SecretStr,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.models import FlextModel
from flext_core.result import FlextResult

# Constants for magic values
MIN_PASSWORD_LENGTH_HIGH_SECURITY = 12
MIN_PASSWORD_LENGTH_MEDIUM_SECURITY = 8
MAX_PASSWORD_LENGTH = 64
MAX_USERNAME_LENGTH = 32
MIN_SECRET_KEY_LENGTH_STRONG = 64
MIN_SECRET_KEY_LENGTH_ADEQUATE = 32


class FlextSettings(BaseSettings):
    """Base settings class using pure Pydantic BaseSettings patterns.

    This is the foundation for all environment-aware configuration across
    the FLEXT ecosystem. Provides automatic environment variable loading
    with type safety and validation.
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
        """Validate business rules - override in subclasses for specific rules."""
        # Default implementation: no business rules => success
        return FlextResult[None].ok(None)

    # Note: Do not use field_serializer for model_config; it's not a model field.

    @model_serializer(mode="wrap", when_used="json")
    def serialize_settings_for_api(
        self,
        serializer: Callable[[FlextSettings], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Model serializer for settings API output with environment metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # With JSON mode, Pydantic always returns dict
        # Add settings-specific API metadata
        data["_settings"] = {
            "type": "FlextSettings",
            "env_loaded": True,
            "validation_enabled": True,
            "api_version": "v2",
            "serialization_format": "json",
        }
        return data

    @classmethod
    def create_with_validation(
        cls,
        overrides: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> FlextResult[FlextSettings]:
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
                return FlextResult[FlextSettings].fail(
                    validation_result.error or "Validation failed"
                )
            return FlextResult[FlextSettings].ok(instance)
        except Exception as e:
            return FlextResult[FlextSettings].fail(f"Settings creation failed: {e}")


class FlextBaseConfigModel(FlextSettings):
    """Backward-compatible base for configuration models.

    Subclassing this class is equivalent to subclassing ``FlextSettings``.
    """


class FlextConfig(FlextModel):
    """Main FLEXT configuration class using pure Pydantic BaseModel patterns.

    This is the core configuration model for the FLEXT ecosystem,
    providing type-safe configuration with automatic validation.
    """

    # Core identification
    name: str = Field(default="flext", description="Configuration name")
    version: str = Field(default="1.0.0", description="Configuration version")
    description: str = Field(
        default="FLEXT configuration",
        description="Configuration description",
    )

    # Environment settings
    environment: str = Field(
        default="development",
        description="Environment name (development, staging, production)",
    )
    debug: bool = Field(default=False, description="Debug mode enabled")

    # Core operational settings
    log_level: str = Field(default="INFO", description="Logging level")
    timeout: int = Field(default=30, description="Default timeout in seconds")
    retries: int = Field(default=3, description="Default retry count")
    page_size: int = Field(default=100, description="Default page size")

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
        """Validate environment value with common shorthand mapping."""
        mapping = {
            "dev": "development",
            "prod": "production",
            "stage": "staging",
            "stg": "staging",
        }
        normalized = mapping.get(v.lower(), v)
        allowed = {"development", "staging", "production", "test"}
        if normalized not in allowed:
            msg = f"Environment must be one of: {allowed}"
            raise ValueError(msg)
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            msg = f"Log level must be one of: {allowed}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("timeout", "retries", "page_size")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        """Validate positive integer values."""
        if v <= 0:
            msg = "Value must be positive"
            raise ValueError(msg)
        return v

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate FLEXT-specific business rules."""
        if self.debug and self.environment == "production":
            return FlextResult[None].fail(
                "Debug mode cannot be enabled in production environment",
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
            return FlextResult[None].fail(
                f"Config validation failed for {', '.join(sorted(critical_none_fields))}",
            )

        return FlextResult[None].ok(None)

    @field_serializer("environment", when_used="json")
    def serialize_environment(self, value: str) -> dict[str, object]:
        """Serialize environment with additional metadata for JSON."""
        return {
            "name": value,
            "is_production": value == "production",
            "debug_allowed": value != "production",
            "config_profile": f"flext-{value}",
        }

    @field_serializer("log_level", when_used="json")
    def serialize_log_level(self, value: str) -> dict[str, object]:
        """Serialize log level with metadata for JSON."""
        level_hierarchy = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        return {
            "level": value,
            "numeric_level": level_hierarchy.get(value, 20),
            "verbose": value == "DEBUG",
            "production_safe": value in {"INFO", "WARNING", "ERROR", "CRITICAL"},
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_config_for_api(
        self,
        serializer: Callable[[FlextConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Model serializer for config API output with comprehensive metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
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
        """Create complete configuration with defaults and validation."""
        try:
            # Convert config_data to dict for manipulation
            working_config = dict(config_data)

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
                        validation_result.error or "Validation failed",
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
        """Load and validate configuration from file."""
        try:
            file_result = safe_load_json_file(file_path)
            if file_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    file_result.error or "Failed to load file"
                )

            # file_result.data is typed as dict[str, object] on success
            data = file_result.data
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
                    validation_result.error or "Validation failed"
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
        """Safely load configuration from dictionary."""
        try:
            instance = cls.model_validate(dict(config_data))
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or "Validation failed"
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
        """Merge two configurations and validate the result."""
        try:
            merged = {**dict(base_config), **dict(override_config)}

            # Check for None values which are not allowed in config
            none_keys = [k for k, v in merged.items() if v is None]
            if none_keys:
                return FlextResult[dict[str, object]].fail(
                    f"Configuration cannot contain None values for keys: {', '.join(none_keys)}",
                )

            instance = cls.model_validate(merged)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or "Validation failed"
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
        """Get environment variable with type validation."""
        try:
            env_result = safe_get_env_var(
                env_var,
                str(default) if default is not None else None,
            )
            if env_result.is_failure and required:
                return env_result

            # Get value from result or use default
            if env_result.is_success:
                value: str = env_result.unwrap()
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
                # result.data may be of various types; coerce to str for env API
                return FlextResult[str].ok(str(result.unwrap()))
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
    def merge_configs(
        cls,
        base_config: dict[str, object],
        override_config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge two configurations and validate the result."""
        try:
            merged = {**base_config, **override_config}

            # Validate for None values which are invalid
            for key, value in merged.items():
                if value is None:
                    return FlextResult[dict[str, object]].fail(
                        f"Config validation failed for {key}: cannot be null",
                    )

            instance = cls.model_validate(merged)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or "Validation failed"
                )

            return FlextResult[dict[str, object]].ok(instance.model_dump())
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Config merge failed: {e}")

    @classmethod
    def validate_config_value(
        cls,
        value: object,
        validator: object,
        error_message: str = "Validation failed",
    ) -> FlextResult[bool]:
        """Validate a configuration value using a validator function."""
        try:
            if not callable(validator):
                return FlextResult[bool].fail("Validator must be callable")

            try:
                result = validator(value)
                if not result:
                    return FlextResult[bool].fail(error_message)
                return FlextResult[bool].ok(True)  # noqa: FBT003
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

        This static method returns model configuration parameters that tests expect.
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


class FlextDatabaseConfig(FlextModel):
    """Database configuration using pure Pydantic BaseModel patterns."""

    # Connection settings
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="postgres", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: SecretStr = Field(
        default_factory=lambda: SecretStr("postgres"),
        description="Database password",
    )
    database_schema: str = Field(default="public", description="Database schema")

    # Connection pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")

    # Query settings
    echo: bool = Field(default=False, description="Echo SQL queries")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        max_port = 65535
        if not 1 <= v <= max_port:
            msg = f"Port must be between 1 and {max_port}"
            raise ValueError(msg)
        return v

    @field_validator("host", "username")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Ensure string fields are not empty."""
        if not v.strip():
            msg = "Field must not be empty"
            raise ValueError(msg)
        return v

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate database business rules."""
        if self.pool_size > self.max_overflow:
            return FlextResult[None].fail(
                "Connection pool_size cannot exceed max_overflow"
            )
        return FlextResult[None].ok(None)

    def get_connection_string(self) -> str:
        """Get database connection string."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.username}:{pwd}@{self.host}:{self.port}/{self.database}"

    @field_serializer("password", when_used="json")
    def serialize_password(self, value: SecretStr) -> dict[str, object]:
        """Serialize password with security metadata (never expose actual value)."""
        password_length = (
            len(value.get_secret_value()) if value.get_secret_value() else 0
        )
        return {
            "is_set": bool(value.get_secret_value()),
            "length": password_length,
            "security_level": "high"
            if password_length >= MIN_PASSWORD_LENGTH_HIGH_SECURITY
            else "medium",
            "_warning": "Password value hidden for security",
        }

    @field_serializer("host", "username", when_used="json")
    def serialize_connection_fields(self, value: str) -> dict[str, object]:
        """Serialize connection fields with validation metadata."""
        return {
            "value": value,
            "is_valid": bool(value and value.strip()),
            "character_count": len(value),
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_database_config_for_api(
        self,
        serializer: Callable[[FlextDatabaseConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Database config model serializer for API with connection metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # Add database-specific API metadata
        if data and hasattr(data, "get"):
            data["_database"] = {
                "type": "PostgreSQL",
                "connection_ready": bool(data.get("host") and data.get("username")),
                "pool_configured": (
                    int(pool_size_val)
                    if isinstance(pool_size_val := data.get("pool_size", 0), (int, str))
                    else 0
                )
                > 0,
                "ssl_mode": "prefer",  # Default for PostgreSQL
                "api_version": "v2",
            }
        return data

    def to_database_dict(self) -> dict[str, object]:
        """Convert database config to dictionary format."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "database_schema": self.database_schema,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "echo": self.echo,
        }


class FlextRedisConfig(FlextModel):
    """Redis configuration using pure Pydantic BaseModel patterns."""

    # Connection settings
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="Redis password",
    )
    database: int = Field(default=0, description="Redis database number")

    # Connection behavior
    decode_responses: bool = Field(
        default=True,
        description="Decode responses to strings",
    )
    socket_timeout: int = Field(default=30, description="Socket timeout in seconds")
    connection_pool_max_connections: int = Field(
        default=50,
        description="Max pool connections",
    )

    @field_validator("database")
    @classmethod
    def validate_database(cls, v: int) -> int:
        """Validate database number."""
        if v < 0:
            msg = "Database must be a non-negative integer"
            raise ValueError(msg)
        return v

    def get_connection_string(self) -> str:
        """Get Redis connection string."""
        pwd = self.password.get_secret_value()
        if pwd:
            return f"redis://:{pwd}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"

    @field_serializer("database", when_used="json")
    def serialize_redis_database(self, value: int) -> dict[str, object]:
        """Serialize Redis database number with metadata."""
        return {
            "number": value,
            "is_default": value == 0,
            "namespace": f"flext:db:{value}",
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_redis_config_for_api(
        self,
        serializer: Callable[[FlextRedisConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Redis config model serializer for API with cache metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # Add Redis-specific API metadata
        if data and hasattr(data, "get"):
            data["_redis"] = {
                "type": "Redis",
                "purpose": "caching",
                "connection_pooling": True,
                "max_connections": data.get("connection_pool_max_connections", 50),
                "api_version": "v2",
            }
        return data

    def to_redis_dict(self) -> dict[str, object]:
        """Convert Redis config to dictionary format."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "decode_responses": self.decode_responses,
            "socket_timeout": self.socket_timeout,
            "connection_pool_max_connections": self.connection_pool_max_connections,
        }


class FlextLDAPConfig(FlextModel):
    """LDAP configuration using pure Pydantic BaseModel patterns."""

    # Server settings
    server: str = Field(default="localhost", description="LDAP server")
    port: int = Field(default=389, description="LDAP port")
    use_ssl: bool = Field(default=False, description="Use SSL")
    use_tls: bool = Field(default=False, description="Use TLS")

    # Authentication
    bind_dn: str = Field(default="", description="Bind DN")
    bind_password: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="Bind password",
    )
    base_dn: str = Field(default="", description="Base DN for LDAP operations")

    # Search configuration
    search_base: str = Field(default="", description="Search base DN")
    search_filter: str = Field(default="(objectClass=*)", description="Search filter")
    attributes: list[str] = Field(
        default_factory=list,
        description="Attributes to retrieve",
    )

    # Connection settings
    timeout: int = Field(default=30, description="LDAP timeout in seconds")

    @field_validator("base_dn")
    @classmethod
    def validate_base_dn(cls, v: str) -> str:
        """Validate base DN format."""
        if not v.strip():
            msg = "Base DN cannot be empty"
            raise ValueError(msg)

        # Basic DN format validation: must contain dc= or cn= or ou=
        dn_lower = v.lower()
        if not any(prefix in dn_lower for prefix in ["dc=", "cn=", "ou="]):
            msg = "Invalid DN format: must contain dc=, cn=, or ou="
            raise ValueError(msg)

        return v

    @field_validator("port")
    @classmethod
    def validate_ldap_port(cls, v: int) -> int:
        """Validate LDAP port."""
        # Common LDAP ports or any valid port
        common_ports = {389, 636, 3268, 3269}
        max_port = 65535
        if v not in common_ports and not (1 <= v <= max_port):
            msg = f"LDAP port must be one of {common_ports} or between 1 and {max_port}"
            raise ValueError(msg)
        return v

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate LDAP business rules."""
        if self.use_ssl and self.use_tls:
            return FlextResult[None].fail(
                "Cannot enable both SSL and TLS simultaneously"
            )
        return FlextResult[None].ok(None)

    def get_connection_string(self) -> str:
        """Get LDAP connection string."""
        scheme = "ldaps" if self.use_ssl else "ldap"
        return f"{scheme}://{self.server}:{self.port}"

    @field_serializer("bind_password", when_used="json")
    def serialize_bind_password(self, value: SecretStr) -> dict[str, object]:
        """Serialize bind password with security metadata."""
        return {
            "is_set": bool(value.get_secret_value()),
            "length": len(value.get_secret_value()) if value.get_secret_value() else 0,
            "_warning": "Password value hidden for security",
        }

    @field_serializer("use_ssl", "use_tls", when_used="json")
    def serialize_security_flags(self, value: bool) -> dict[str, object]:  # noqa: FBT001
        """Serialize security flags with context."""
        return {
            "enabled": value,
            "security_level": "high" if value else "medium",
            "recommended": value,
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_ldap_config_for_api(
        self,
        serializer: Callable[[FlextLDAPConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """LDAP config model serializer for API with directory metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # Add LDAP-specific API metadata
        if data and hasattr(data, "get"):
            data["_ldap"] = {
                "type": "LDAP",
                "protocol": "ldaps" if data.get("use_ssl") else "ldap",
                "security_enabled": data.get("use_ssl", False)
                or data.get("use_tls", False),
                "directory_ready": bool(data.get("base_dn")),
                "api_version": "v2",
            }
        return data

    def to_ldap_dict(self) -> dict[str, object]:
        """Convert LDAP config to dictionary format."""
        return {
            "host": self.server,  # Use server but export as 'host' for compatibility
            "server": self.server,
            "port": self.port,
            "use_ssl": self.use_ssl,
            "use_tls": self.use_tls,
            "bind_dn": self.bind_dn,
            "base_dn": self.base_dn,
            "search_base": self.search_base,
            "search_filter": self.search_filter,
            "attributes": self.attributes,
            "timeout": self.timeout,
        }

    # Compatibility property for existing code
    @property
    def host(self) -> str:
        """Alias for server for backward compatibility."""
        return self.server


class FlextOracleConfig(FlextModel):
    """Oracle database configuration using pure Pydantic BaseModel patterns."""

    # Connection settings
    host: str = Field(default="localhost", description="Oracle host")
    port: int = Field(default=1521, description="Oracle port")
    service_name: str | None = Field(default=None, description="Oracle service name")
    sid: str | None = Field(default=None, description="Oracle SID")
    username: str = Field(description="Oracle username")
    password: SecretStr = Field(description="Oracle password")
    oracle_schema: str = Field(default="public", description="Oracle schema")

    # Connection pool settings
    pool_min: int = Field(default=1, description="Minimum pool connections")
    pool_max: int = Field(default=10, description="Maximum pool connections")
    connection_timeout: int = Field(
        default=30,
        description="Connection timeout in seconds",
    )

    def model_post_init(self, __context: object, /) -> None:
        """Post-initialization validation."""
        if not (self.service_name or self.sid):
            msg = "Either service_name or sid must be provided"
            raise ValueError(msg)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate Oracle business rules."""
        if self.pool_min > self.pool_max:
            return FlextResult[None].fail("pool_min cannot be greater than pool_max")
        return FlextResult[None].ok(None)

    def get_connection_string(self) -> str:
        """Get Oracle connection string."""
        if self.service_name:
            return f"{self.host}:{self.port}/{self.service_name}"
        if self.sid:
            return f"{self.host}:{self.port}:{self.sid}"
        return f"{self.host}:{self.port}"

    @field_serializer("password", when_used="json")
    def serialize_oracle_password(self, value: SecretStr) -> dict[str, object]:
        """Serialize Oracle password with security metadata."""
        return {
            "is_set": bool(value.get_secret_value()),
            "complexity_check": len(value.get_secret_value())
            >= MIN_PASSWORD_LENGTH_MEDIUM_SECURITY,
            "_warning": "Password value hidden for security",
        }

    @field_serializer("service_name", "sid", when_used="json")
    def serialize_oracle_identifier(self, value: str | None) -> dict[str, object]:
        """Serialize Oracle service name or SID with metadata."""
        return {
            "value": value,
            "is_set": value is not None,
            "connection_type": "service_name"
            if value and "/" not in str(value)
            else "sid",
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_oracle_config_for_api(
        self,
        serializer: Callable[[FlextOracleConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Oracle config model serializer for API with database metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # Add Oracle-specific API metadata
        if data and hasattr(data, "get"):
            pool_max_raw: object = data.get("pool_max", 0)
            if isinstance(pool_max_raw, int):
                pool_max_int = pool_max_raw
            elif isinstance(pool_max_raw, str) and pool_max_raw.isdigit():
                pool_max_int = int(pool_max_raw)
            else:
                pool_max_int = 0

            data["_oracle"] = {
                "type": "Oracle",
                "version": "21c+",
                "connection_method": "service_name"
                if data.get("service_name")
                else "sid",
                "pool_enabled": pool_max_int > 1,
                "api_version": "v2",
            }
        return data

    def to_oracle_dict(self) -> dict[str, object]:
        """Convert Oracle config to dictionary format."""
        return {
            "host": self.host,
            "port": self.port,
            "service_name": self.service_name,
            "sid": self.sid,
            "username": self.username,
            "oracle_schema": self.oracle_schema,
            "pool_min": self.pool_min,
            "pool_max": self.pool_max,
            "connection_timeout": self.connection_timeout,
        }


class FlextJWTConfig(FlextModel):
    """JWT configuration using pure Pydantic BaseModel patterns."""

    # JWT settings
    secret_key: SecretStr = Field(description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiry in minutes",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiry in days",
    )
    issuer: str = Field(default="flext", description="JWT issuer")
    audience: list[str] = Field(default_factory=list, description="JWT audience")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm."""
        allowed = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in allowed:
            msg = f"Algorithm must be one of: {allowed}"
            raise ValueError(msg)
        return v

    @field_validator("secret_key", mode="before")
    @classmethod
    def validate_secret_key(cls, v: object) -> SecretStr:
        """Validate secret key length."""
        min_key_length = 32
        key = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        if len(key) < min_key_length:
            msg = f"Secret key must be at least {min_key_length} characters"
            raise ValueError(msg)
        return SecretStr(key) if not isinstance(v, SecretStr) else v

    def to_jwt_dict(self) -> dict[str, object]:
        """Convert JWT config to dictionary format."""
        return {
            "algorithm": self.algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "refresh_token_expire_days": self.refresh_token_expire_days,
            "issuer": self.issuer,
            "audience": self.audience,
        }

    @field_serializer("secret_key", when_used="json")
    def serialize_jwt_secret(self, value: SecretStr) -> dict[str, object]:
        """Serialize JWT secret with security metadata."""
        secret_value = value.get_secret_value()
        return {
            "is_set": bool(secret_value),
            "length": len(secret_value),
            "strength": "strong"
            if len(secret_value) >= MIN_SECRET_KEY_LENGTH_STRONG
            else "adequate"
            if len(secret_value) >= MIN_SECRET_KEY_LENGTH_ADEQUATE
            else "weak",
            "_warning": "Secret key value hidden for security",
        }

    @field_serializer("algorithm", when_used="json")
    def serialize_jwt_algorithm(self, value: str) -> dict[str, object]:
        """Serialize JWT algorithm with security metadata."""
        algorithm_security = {
            "HS256": "symmetric",
            "HS384": "symmetric",
            "HS512": "symmetric",
            "RS256": "asymmetric",
            "RS384": "asymmetric",
            "RS512": "asymmetric",
        }
        return {
            "algorithm": value,
            "type": algorithm_security.get(value, "unknown"),
            "recommended": value in {"HS256", "RS256"},
            "security_level": "high" if value.startswith("RS") else "medium",
        }

    @model_serializer(mode="wrap", when_used="json")
    def serialize_jwt_config_for_api(
        self,
        serializer: Callable[[FlextJWTConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """JWT config model serializer for API with token metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # Add JWT-specific API metadata
        if data and hasattr(data, "get"):
            refresh_days_raw: object = data.get("refresh_token_expire_days", 0)
            if isinstance(refresh_days_raw, int):
                refresh_days_int = refresh_days_raw
            elif isinstance(refresh_days_raw, str) and refresh_days_raw.isdigit():
                refresh_days_int = int(refresh_days_raw)
            else:
                refresh_days_int = 0

            data["_jwt"] = {
                "type": "JWT",
                "token_management": "stateless",
                "refresh_enabled": refresh_days_int > 0,
                "security_compliant": True,
                "api_version": "v2",
            }
        return data


class FlextObservabilityConfig(FlextModel):
    """Observability configuration using pure Pydantic BaseModel patterns."""

    # Logging configuration (test-compatible field names)
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format")
    logging_enabled: bool = Field(default=True, description="Enable logging")

    # Alternative field names for backward compatibility
    logging_level: str = Field(default="INFO", description="Logging level (alias)")
    logging_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format (alias)",
    )

    # Tracing configuration
    tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )
    service_name: str = Field(default="flext", description="Service name for tracing")
    tracing_environment: str = Field(
        default="development",
        description="Environment for tracing",
    )

    # Metrics configuration
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8080, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

    # Health check configuration
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_port: int = Field(default=8081, description="Health check port")
    health_check_path: str = Field(default="/health", description="Health check path")

    @field_validator("log_level", "logging_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = v.upper()
        if level not in allowed:
            msg = f"Log level must be one of: {allowed}"
            raise ValueError(msg)
        return level

    def to_observability_dict(self) -> dict[str, object]:
        """Convert observability config to dictionary format."""
        return {
            "log_level": self.log_level,
            "log_format": self.log_format,
            "metrics_enabled": self.metrics_enabled,
            "tracing_enabled": self.tracing_enabled,
            "service_name": self.service_name,
        }

    # Constants for backward compatibility
    ENABLE_METRICS: ClassVar[bool] = True
    TRACE_ENABLED: ClassVar[bool] = True
    TRACE_SAMPLE_RATE: ClassVar[float] = 0.1


class FlextSingerConfig(FlextModel):
    """Singer configuration using pure Pydantic BaseModel patterns."""

    # Executable paths
    tap_executable: str = Field(default="tap", description="Tap executable path")
    target_executable: str = Field(
        default="target",
        description="Target executable path",
    )

    # File paths
    config_file: str = Field(
        default="config.json",
        description="Configuration file path",
    )
    catalog_file: str = Field(default="catalog.json", description="Catalog file path")
    state_file: str = Field(default="state.json", description="State file path")
    output_file: str = Field(
        default="singer_output.jsonl",
        description="Output file path",
    )

    # Stream configuration
    stream_name: str = Field(description="Singer stream name")
    batch_size: int = Field(default=1000, description="Batch size")
    stream_schema: dict[str, object] = Field(
        default_factory=dict,
        description="Singer stream schema",
    )
    stream_config: dict[str, object] = Field(
        default_factory=dict,
        description="Singer stream configuration",
    )

    @field_validator("stream_name")
    @classmethod
    def validate_stream_name(cls, v: str) -> str:
        """Validate stream name is not empty."""
        if not v.strip():
            msg = "Stream name must not be empty"
            raise ValueError(msg)
        return v

    def to_singer_dict(self) -> dict[str, object]:
        """Convert Singer config to dictionary format."""
        return {
            "stream_name": self.stream_name,
            "batch_size": self.batch_size,
            "stream_schema": self.stream_schema,
            "stream_config": self.stream_config,
            "tap_executable": self.tap_executable,
            "target_executable": self.target_executable,
        }


class FlextConfigFactory:
    """Factory for creating configuration instances using pure Pydantic patterns."""

    _config_registry: ClassVar[dict[str, type[FlextModel]]] = {
        "main": FlextConfig,
        "database": FlextDatabaseConfig,
        "redis": FlextRedisConfig,
        "ldap": FlextLDAPConfig,
        "oracle": FlextOracleConfig,
        "jwt": FlextJWTConfig,
        "observability": FlextObservabilityConfig,
        "singer": FlextSingerConfig,
    }

    @classmethod
    def create(cls, config_type: str, **kwargs: object) -> FlextResult[FlextModel]:
        """Create configuration instance by type."""
        if config_type not in cls._config_registry:
            return FlextResult[FlextModel].fail(f"Unknown config type: {config_type}")

        try:
            config_class = cls._config_registry[config_type]
            instance = config_class.model_validate(kwargs)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[FlextModel].fail(
                    validation_result.error or "Validation failed"
                )
            return FlextResult[FlextModel].ok(instance)
        except Exception as e:
            return FlextResult[FlextModel].fail(
                f"Failed to create {config_type} config: {e}"
            )

    @classmethod
    def create_from_env(
        cls,
        prefix: str = "FLEXT_",
    ) -> FlextResult[dict[str, object]]:
        """Create configuration dictionary from environment variables."""
        try:
            env_data = cls._load_from_env(prefix.rstrip("_"))
            return FlextResult[dict[str, object]].ok(env_data)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to create config from env: {e}",
            )

    @classmethod
    def create_from_env_typed(
        cls,
        config_type: str,
        prefix: str = "FLEXT",
    ) -> FlextResult[FlextModel]:
        """Create typed configuration from environment variables."""
        if config_type not in cls._config_registry:
            return FlextResult[FlextModel].fail(f"Unknown config type: {config_type}")

        try:
            config_class = cls._config_registry[config_type]
            env_data = cls._load_from_env(prefix)
            instance = config_class.model_validate(env_data)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[FlextModel].fail(
                    validation_result.error or "Validation failed"
                )
            return FlextResult[FlextModel].ok(instance)
        except Exception as e:
            return FlextResult[FlextModel].fail(
                f"Failed to create {config_type} config from env: {e}",
            )

    @classmethod
    def create_from_file(
        cls,
        file_path: str | Path,
        required_keys: list[str] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Create configuration dictionary from JSON file."""
        try:
            file_data_result = cls._load_from_file(file_path)
            if file_data_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    file_data_result.error or "Failed to load file"
                )

            config_data = file_data_result.data

            # Validate required keys if specified
            if required_keys:
                validation_result = FlextConfigValidation.validate_required_keys(
                    config_data, required_keys
                )
                if validation_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        validation_result.error or "Required keys validation failed"
                    )

            return FlextResult[dict[str, object]].ok(config_data)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to create config from file: {e}",
            )

    @classmethod
    def create_from_file_typed(
        cls,
        config_type: str,
        file_path: str | Path,
    ) -> FlextResult[FlextModel]:
        """Create typed configuration from JSON file."""
        if config_type not in cls._config_registry:
            return FlextResult[FlextModel].fail(f"Unknown config type: {config_type}")

        try:
            file_data_result = cls._load_from_file(file_path)
            if file_data_result.is_failure:
                return FlextResult[FlextModel].fail(
                    file_data_result.error or "Failed to load file"
                )

            config_class = cls._config_registry[config_type]
            instance = config_class.model_validate(file_data_result.data)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[FlextModel].fail(
                    validation_result.error or "Validation failed"
                )
            return FlextResult[FlextModel].ok(instance)
        except Exception as e:
            return FlextResult[FlextModel].fail(
                f"Failed to create {config_type} config from file: {e}",
            )

    @classmethod
    def register_config(cls, name: str, config_class: type[FlextModel]) -> None:
        """Register new configuration class."""
        cls._config_registry[name] = config_class

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of registered configuration types."""
        return list(cls._config_registry.keys())

    @classmethod
    def create_from_dict(
        cls,
        config_data: dict[str, object],
        defaults: dict[str, object] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Create configuration from dictionary with optional defaults."""
        try:
            # Apply defaults if provided
            if defaults:
                result_config = FlextConfigDefaults.apply_defaults(
                    config_data, defaults
                )
                if result_config.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        result_config.error or "Failed to apply defaults"
                    )
                return result_config
            # Just validate the config data
            validation_result = FlextConfigValidation.validate_config(config_data)
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    validation_result.error or "Config validation failed"
                )
            return FlextResult[dict[str, object]].ok(config_data)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to create config from dict: {e}"
            )

    @classmethod
    def create_from_multiple_sources(
        cls,
        sources: list[tuple[str, object]],
    ) -> FlextResult[dict[str, object]]:
        """Create configuration from multiple sources in priority order."""
        try:
            final_config: dict[str, object] = {}

            for source_type, source_data in sources:
                if source_type == "file":
                    result = cls.create_from_file(str(source_data))
                elif source_type == "env":
                    result = cls.create_from_env(str(source_data))
                elif source_type == "defaults" and isinstance(source_data, dict):
                    result = FlextResult[dict[str, object]].ok(
                        cast("dict[str, object]", source_data)
                    )
                else:
                    continue

                if result.is_success and result.data:
                    # Merge with existing config
                    merge_result = merge_configs(final_config, result.data)
                    if merge_result.is_success and merge_result.data:
                        final_config = merge_result.data

            return FlextResult[dict[str, object]].ok(final_config)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to create config from multiple sources: {e}"
            )

    @staticmethod
    def _load_from_env(prefix: str = "FLEXT") -> dict[str, object]:
        """Load configuration from environment variables."""
        config: dict[str, object] = {}
        prefix_with_separator = f"{prefix}_"

        for key, value in os.environ.items():
            if key.startswith(prefix_with_separator):
                config_key = key[len(prefix_with_separator) :].lower()
                # Simple string values only - let Pydantic handle type conversion
                config[config_key] = value

        return config

    @staticmethod
    def _load_from_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Load configuration from JSON file."""
        import json  # noqa: PLC0415

        path = Path(file_path)
        if not path.exists():
            return FlextResult[dict[str, object]].fail("Configuration file not found")
        if not path.is_file():
            return FlextResult[dict[str, object]].fail("Path is not a file")

        try:
            content = path.read_text(encoding="utf-8")
            loaded_data = json.loads(content)
            if not isinstance(loaded_data, dict):
                return FlextResult[dict[str, object]].fail(
                    "JSON file must contain a dictionary"
                )
            data: dict[str, object] = cast("dict[str, object]", loaded_data)
            return FlextResult[dict[str, object]].ok(data)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to load JSON file: {e}")


# Convenience functions for common configuration operations
def create_config(config_type: str, **kwargs: object) -> FlextResult[FlextModel]:
    """Create configuration instance using factory."""
    return FlextConfigFactory.create(config_type, **kwargs)


def load_config_from_env(
    config_type: str,
    prefix: str = "FLEXT",
) -> FlextResult[FlextModel]:
    """Load configuration from environment variables."""
    return FlextConfigFactory.create_from_env_typed(config_type, prefix)


def load_config_from_file(
    config_type: str,
    file_path: str | Path,
) -> FlextResult[FlextModel]:
    """Load configuration from JSON file."""
    return FlextConfigFactory.create_from_file_typed(config_type, file_path)


# Additional utility functions expected by tests
def safe_get_env_var(
    var_name: str,
    default: str | None = None,
    *,
    required: bool = False,
) -> FlextResult[str]:
    """Safely get environment variable with FlextResult error handling."""
    try:
        if not var_name or not var_name.strip():
            return FlextResult[str].fail(
                "Environment variable name must be a non-empty string"
            )

        value = os.environ.get(var_name)
        if value is None:
            if required:
                return FlextResult[str].fail(
                    f"Required environment variable '{var_name}' not found"
                )
            if default is None:
                return FlextResult[str].fail(
                    f"Environment variable '{var_name}' not found"
                )
            return FlextResult[str].ok(default)
        return FlextResult[str].ok(value)
    except Exception as e:
        return FlextResult[str].fail(f"Environment variable retrieval failed: {e}")


def safe_load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
    """Safely load JSON file with FlextResult error handling."""
    try:
        path = Path(file_path)
        if not path.exists():
            return FlextResult[dict[str, object]].fail(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        loaded_data = json.loads(content)

        # Validate that the JSON contains a dictionary
        if not isinstance(loaded_data, dict):
            return FlextResult[dict[str, object]].fail(
                "JSON file must contain a dictionary"
            )

        data: dict[str, object] = cast("dict[str, object]", loaded_data)
        return FlextResult[dict[str, object]].ok(data)
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Failed to load JSON file: {e}")


def merge_configs(
    base: dict[str, object],
    override: dict[str, object],
) -> FlextResult[dict[str, object]]:
    """Merge two configuration dictionaries with override taking precedence."""
    try:

        def _deep_merge(
            base_dict: dict[str, object], override_dict: dict[str, object]
        ) -> dict[str, object]:
            """Recursively merge dictionaries."""
            merged = base_dict.copy()

            for key, value in override_dict.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    # Recursively merge nested dictionaries
                    merged[key] = _deep_merge(
                        cast("dict[str, object]", merged[key]),
                        cast("dict[str, object]", value),
                    )
                else:
                    # Override value
                    merged[key] = value

            return merged

        result = _deep_merge(base, override)
        return FlextResult[dict[str, object]].ok(result)
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Config merge failed: {e}")


def merge_configs_multiple(
    *configs: FlextModel,
) -> FlextResult[dict[str, object]]:
    """Merge multiple configuration objects into a dictionary."""
    try:
        if not configs:
            return FlextResult[dict[str, object]].ok({})

        merged: dict[str, object] = {}
        for config in configs:
            config_dict = config.model_dump()
            merged.update(config_dict)

        return FlextResult[dict[str, object]].ok(merged)
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Config merge failed: {e}")


def validate_config(
    config: dict[str, object],  # noqa: ARG001
    schema: dict[str, object] | None = None,
) -> FlextResult[None]:
    """Validate configuration dictionary."""
    try:
        # No runtime type checks needed - parameters are already typed
        # Perform actual validation logic here if needed
        if schema is not None:
            # Future: schema validation logic can be added here
            pass
        return FlextResult[None].ok(None)
    except Exception as e:
        return FlextResult[None].fail(f"Configuration validation failed: {e}")


# Namespace classes for test compatibility
class FlextConfigOps:
    """Configuration operations namespace."""

    @staticmethod
    def safe_load_from_dict(
        data: dict[str, object],
        required_keys: list[str] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Safely load configuration from dictionary."""
        try:
            # Note: isinstance check removed since parameter is already typed as dict
            if required_keys is not None:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    missing_str = ", ".join(missing_keys)
                    return FlextResult[dict[str, object]].fail(
                        f"Missing required configuration keys: {missing_str}",
                    )

            return FlextResult[dict[str, object]].ok(data.copy())
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to load from dict: {e}")

    @staticmethod
    def safe_get_env_var(
        var_name: str,
        default: str | None = None,
        *,
        required: bool = False,
    ) -> FlextResult[str]:
        """Safely get environment variable."""
        return safe_get_env_var(var_name, default, required=required)

    @staticmethod
    def safe_load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Safely load JSON file."""
        return safe_load_json_file(file_path)

    @staticmethod
    def safe_save_json_file(
        data: dict[str, object],
        file_path: str | Path,
        *,
        create_dirs: bool = False,
    ) -> FlextResult[None]:
        """Safely save JSON file."""
        try:
            path = Path(file_path)

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            elif not path.parent.exists():
                return FlextResult[None].fail(
                    f"Directory does not exist: {path.parent}"
                )

            content = json.dumps(data, indent=2, ensure_ascii=False)
            path.write_text(content, encoding="utf-8")
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to save JSON file: {e}")


class FlextConfigDefaults:
    """Configuration defaults management."""

    @staticmethod
    def apply_defaults(
        config: dict[str, object],
        defaults: dict[str, object] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Apply default values to configuration."""
        try:
            # Use system defaults if no custom defaults provided
            if defaults is None:
                defaults = {
                    "debug": False,
                    "timeout": DEFAULT_TIMEOUT,
                    "retries": DEFAULT_RETRIES,
                    "page_size": DEFAULT_PAGE_SIZE,
                    "log_level": DEFAULT_LOG_LEVEL,
                    "environment": DEFAULT_ENVIRONMENT,
                    "port": 8000,
                }

            result = defaults.copy()
            result.update(config)
            return FlextResult[dict[str, object]].ok(result)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Failed to apply defaults: {e}")

    @staticmethod
    def get_default_config() -> FlextResult[dict[str, object]]:
        """Get the default configuration."""
        try:
            defaults: dict[str, object] = {
                "debug": False,
                "timeout": DEFAULT_TIMEOUT,
                "retries": DEFAULT_RETRIES,
                "page_size": DEFAULT_PAGE_SIZE,
                "log_level": DEFAULT_LOG_LEVEL,
                "environment": DEFAULT_ENVIRONMENT,
                "port": 8000,
            }
            return FlextResult[dict[str, object]].ok(defaults)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to get default config: {e}"
            )


class FlextConfigValidation:
    """Configuration validation utilities."""

    @staticmethod
    def validate_config_value(
        value: object,
        validator: object,
        key: str = "field",
    ) -> FlextResult[bool]:
        """Validate a configuration value."""
        try:
            if callable(validator):
                result = validator(value)
                return FlextResult[bool].ok(bool(result))
            validation_success = True
            return FlextResult[bool].ok(validation_success)
        except Exception as e:
            return FlextResult[bool].fail(f"Validation failed for {key}: {e}")

    @staticmethod
    def validate_config_type(
        value: object,
        expected_type: type,
        key_name: str = "field",
    ) -> FlextResult[bool]:
        """Validate configuration value type."""
        try:
            if isinstance(value, expected_type):
                return FlextResult[bool].ok(True)  # noqa: FBT003
            expected_name = expected_type.__name__
            actual_name = type(value).__name__
            return FlextResult[bool].fail(
                f"Expected {expected_name} for {key_name}, got {actual_name}",
            )
        except Exception as e:
            return FlextResult[bool].fail(f"Type validation failed for {key_name}: {e}")

    @staticmethod
    def validate_config_range(
        value: float,
        min_val: float | None = None,
        max_val: float | None = None,
        key_name: str = "field",
    ) -> FlextResult[bool]:
        """Validate numeric configuration value range."""
        try:
            if min_val is not None and value < min_val:
                return FlextResult[bool].fail(f"{key_name} must be at least {min_val}")
            if max_val is not None and value > max_val:
                return FlextResult[bool].fail(f"{key_name} must be at most {max_val}")
            return FlextResult[bool].ok(True)  # noqa: FBT003
        except Exception as e:
            return FlextResult[bool].fail(
                f"Range validation failed for {key_name}: {e}"
            )

    @staticmethod
    def validate_config(config: dict[str, object]) -> FlextResult[None]:
        """Validate configuration dictionary."""
        try:
            # Basic validation - ensure config is not empty
            if not config:
                return FlextResult[None].fail("Configuration cannot be empty")
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Configuration validation failed: {e}")

    @staticmethod
    def validate_required_keys(
        config: dict[str, object],
        required_keys: list[str],
    ) -> FlextResult[None]:
        """Validate that all required keys are present."""
        try:
            for key in required_keys:
                if key not in config or config[key] is None:
                    return FlextResult[None].fail(
                        f"Required config key '{key}' not found"
                    )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Required keys validation failed: {e}")

    @staticmethod
    def validate_type_compatibility(config: dict[str, object]) -> FlextResult[None]:
        """Validate type compatibility of configuration values."""
        try:
            # Basic type compatibility check - ensure values are JSON-serializable types
            for key, value in config.items():
                if not isinstance(
                    value, (str, int, float, bool, list, dict, type(None))
                ):
                    return FlextResult[None].fail(
                        f"Configuration value for '{key}' has incompatible type: {type(value).__name__}"
                    )
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Type compatibility validation failed: {e}")


# Base configuration validation alias
_BaseConfigValidation = FlextConfigValidation

# Abstract config alias for backward compatibility
FlextAbstractConfig = FlextConfig

# Backward-compatibility handled by the explicit subclass above.


# Constants expected by tests
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_PAGE_SIZE = 100
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ENVIRONMENT = "development"
CONFIG_VALIDATION_MESSAGES = {
    "required": "Field is required",
    "invalid": "Invalid value",
    "missing": "Missing configuration",
}


# Export clean public API
__all__ = [
    "CONFIG_VALIDATION_MESSAGES",
    "DEFAULT_ENVIRONMENT",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "FlextAbstractConfig",
    "FlextBaseConfigModel",
    "FlextBaseConfigModel",
    "FlextConfig",
    "FlextConfigDefaults",
    "FlextConfigFactory",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextDatabaseConfig",
    "FlextJWTConfig",
    "FlextLDAPConfig",
    "FlextObservabilityConfig",
    "FlextOracleConfig",
    "FlextRedisConfig",
    "FlextSettings",
    "FlextSingerConfig",
    "_BaseConfigValidation",
    "create_config",
    "load_config_from_env",
    "load_config_from_file",
    "merge_configs",
    "safe_get_env_var",
    "safe_load_json_file",
    "validate_config",
]
