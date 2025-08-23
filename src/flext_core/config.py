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
from typing import ClassVar, cast, override

from pydantic import (
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.models import FlextModel
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# CONFIGURATION DEFAULTS AND CONSTANTS
# =============================================================================


# Direct constants (use FlextConfig.SystemDefaults in new code)
MIN_PASSWORD_LENGTH_HIGH_SECURITY = 12
MIN_PASSWORD_LENGTH_MEDIUM_SECURITY = 8
MAX_PASSWORD_LENGTH = 64
MAX_USERNAME_LENGTH = 32
MIN_SECRET_KEY_LENGTH_STRONG = 64
MIN_SECRET_KEY_LENGTH_ADEQUATE = 32


# FlextSettings and FlextBaseConfigModel facades will be defined after FlextConfig class
# to avoid circular reference issues


class FlextConfig(FlextModel):
    """Main FLEXT configuration class using pure Pydantic BaseModel patterns.

    This is the core configuration model for the FLEXT ecosystem,
    providing type-safe configuration with automatic validation.

    Following the [module].py + Flext[Module] pattern, this class consolidates
    all configuration functionality while maintaining backward compatibility
    through nested classes and facades.
    """

    # =========================================================================
    # NESTED CLASSES - Core configuration components consolidated
    # =========================================================================

    class SystemDefaults:
        """Centralized system defaults for the FLEXT ecosystem."""

        class Security:
            """Security-related configuration defaults."""

            MIN_PASSWORD_LENGTH_HIGH_SECURITY = 12
            MIN_PASSWORD_LENGTH_MEDIUM_SECURITY = 8
            MAX_PASSWORD_LENGTH = 64
            MAX_USERNAME_LENGTH = 32
            MIN_SECRET_KEY_LENGTH_STRONG = 64
            MIN_SECRET_KEY_LENGTH_ADEQUATE = 32

        class Network:
            """Network and service defaults."""

            TIMEOUT = 30
            RETRIES = 3
            CONNECTION_TIMEOUT = 10

        class Pagination:
            """Pagination defaults."""

            PAGE_SIZE = 100
            MAX_PAGE_SIZE = 1000

        class Logging:
            """Logging configuration defaults."""

            LOG_LEVEL = "INFO"

        class Environment:
            """Environment defaults."""

            DEFAULT_ENV = "development"

    class Settings(BaseSettings):
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
            serializer: Callable[[FlextConfig.Settings], dict[str, object]],
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
            overrides: Mapping[str, FlextTypes.Core.Value] | None = None,
            **kwargs: FlextTypes.Core.Value,
        ) -> FlextResult[FlextConfig.Settings]:
            """Create settings instance with validation and proper override handling."""
            try:
                # Start with default instance
                instance = cls()

                # Prepare overrides dict - support both overrides parameter and kwargs
                all_overrides: FlextTypes.Core.Dict = {}
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
                        validation_result.error or "Validation failed"
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

    @override
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
            fields_str = ", ".join(sorted(critical_none_fields))
            return FlextResult[None].fail(
                f"Config validation failed for {fields_str}",
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

        This static method returns model configuration parameters that .
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
    def get_system_defaults() -> type[FlextSystemDefaults]:
        """Access to system defaults - Tier 1 consolidated interface."""
        return FlextSystemDefaults

    @staticmethod
    def get_env_var(
        var_name: str,
        default: str | None = None,
    ) -> FlextResult[str]:
        """Get environment variable safely - Tier 1 consolidated interface."""
        return safe_get_env_var(var_name, default)

    @staticmethod
    def load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Load JSON file safely - Tier 1 consolidated interface."""
        return safe_load_json_file(file_path)

    @staticmethod
    def merge_config_dicts(
        base_config: dict[str, object],
        override_config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge configuration dictionaries - Tier 1 consolidated interface."""
        return merge_configs(base_config, override_config)

    @classmethod
    def create_settings(
        cls,
        overrides: Mapping[str, FlextTypes.Core.Value] | None = None,
        **kwargs: FlextTypes.Core.Value,
    ) -> FlextResult[FlextConfig.Settings]:
        """Create Settings instance - Tier 1 consolidated interface."""
        return cls.Settings.create_with_validation(overrides, **kwargs)

    # =========================================================================
    # COMPATIBILITY FACADES - Access to all config classes
    # =========================================================================

    # Class-level access to all configuration components (updated for nested classes)
    Defaults: ClassVar[type[SystemDefaults]] = SystemDefaults

    # Backward compatibility constants
    MIN_PASSWORD_LENGTH_HIGH_SECURITY: ClassVar[int] = (
        SystemDefaults.Security.MIN_PASSWORD_LENGTH_HIGH_SECURITY
    )
    MIN_PASSWORD_LENGTH_MEDIUM_SECURITY: ClassVar[int] = (
        SystemDefaults.Security.MIN_PASSWORD_LENGTH_MEDIUM_SECURITY
    )
    MAX_PASSWORD_LENGTH: ClassVar[int] = SystemDefaults.Security.MAX_PASSWORD_LENGTH
    MAX_USERNAME_LENGTH: ClassVar[int] = SystemDefaults.Security.MAX_USERNAME_LENGTH
    MIN_SECRET_KEY_LENGTH_STRONG: ClassVar[int] = (
        SystemDefaults.Security.MIN_SECRET_KEY_LENGTH_STRONG
    )
    MIN_SECRET_KEY_LENGTH_ADEQUATE: ClassVar[int] = (
        SystemDefaults.Security.MIN_SECRET_KEY_LENGTH_ADEQUATE
    )


# =============================================================================
# COMPATIBILITY FACADES - Defined after FlextConfig to avoid circular references
# =============================================================================


# Compatibility facades for backward compatibility
FlextSettings = FlextConfig.Settings
FlextBaseConfigModel = FlextConfig.BaseConfigModel
FlextSystemDefaults = FlextConfig.SystemDefaults


# =============================================================================
# UTILITY FUNCTIONS (Foundation patterns - these should remain)
# =============================================================================


def safe_get_env_var(
    var_name: str,
    default: str | None = None,
) -> FlextResult[str]:
    """Safely get environment variable with optional default.

    This is a foundation utility function that all FLEXT libraries can use
    for environment variable handling with proper error handling.
    """
    try:
        value = os.getenv(var_name, default)
        if value is None:
            return FlextResult[str].fail(f"Environment variable {var_name} not set")
        return FlextResult[str].ok(value)
    except Exception as e:
        return FlextResult[str].fail(f"Failed to get environment variable: {e}")


def safe_load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
    """Safely load JSON configuration file.

    This is a foundation utility function that all FLEXT libraries can use
    for safe JSON file loading with proper error handling.
    """
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return FlextResult[dict[str, object]].fail(
                "JSON file must contain an object"
            )

        return FlextResult[dict[str, object]].ok(cast("dict[str, object]", data))
    except FileNotFoundError:
        return FlextResult[dict[str, object]].fail(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        return FlextResult[dict[str, object]].fail(f"Invalid JSON: {e}")
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Failed to load JSON file: {e}")


def merge_configs(
    base_config: dict[str, object],
    override_config: dict[str, object],
) -> FlextResult[dict[str, object]]:
    """Merge two configuration dictionaries.

    This is a foundation utility function that all FLEXT libraries can use
    for safe configuration merging with proper validation.
    """
    try:
        merged = {**base_config, **override_config}

        # Validate for None values which are invalid
        for key, value in merged.items():
            if value is None:
                return FlextResult[dict[str, object]].fail(
                    f"Config validation failed for {key}: cannot be null",
                )

        return FlextResult[dict[str, object]].ok(merged)
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Config merge failed: {e}")


# Export only the classes and functions defined in this module
__all__ = [
    "MAX_PASSWORD_LENGTH",
    "MAX_USERNAME_LENGTH",
    "MIN_PASSWORD_LENGTH_HIGH_SECURITY",
    "MIN_PASSWORD_LENGTH_MEDIUM_SECURITY",
    "MIN_SECRET_KEY_LENGTH_ADEQUATE",
    "MIN_SECRET_KEY_LENGTH_STRONG",
    "FlextBaseConfigModel",
    "FlextConfig",
    "FlextSettings",
    "FlextSystemDefaults",
    "merge_configs",
    "safe_get_env_var",
    "safe_load_json_file",
]
