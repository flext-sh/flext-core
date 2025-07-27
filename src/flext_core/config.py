"""FLEXT Core Configuration Module.

Comprehensive configuration management system for the FLEXT Core library providing
consolidated functionality through multiple inheritance patterns and Pydantic
integration.

Architecture:
    - Multiple inheritance from specialized configuration base classes
    - Pydantic BaseSettings integration for automatic environment loading
    - Configuration validation with comprehensive error handling
    - File-based configuration loading with JSON support
    - Environment variable management with type validation
    - Configuration merging and override capabilities

Configuration Categories:
    - Base configuration: Core configuration loading and management
    - Configuration defaults: Default value application and management
    - Configuration operations: Merging, loading, and transformation
    - Configuration validation: Type checking and constraint validation
    - Environment integration: Environment variable loading and validation
    - Pydantic integration: Automatic settings with model validation

Maintenance Guidelines:
    - Add new configuration types to appropriate specialized base classes first
    - Use multiple inheritance for configuration capability combination
    - Integrate FlextResult pattern for all operations that can fail
    - Maintain backward compatibility through function aliases
    - Keep configuration operations stateless when possible

Design Decisions:
    - Multiple inheritance pattern for maximum configuration capability reuse
    - Pydantic BaseSettings integration for automatic environment handling
    - FlextResult integration for consistent error handling patterns
    - File-based configuration with JSON format support
    - Type validation for environment variables and configuration values

Configuration Features:
    - JSON file loading with automatic error handling
    - Environment variable access with default value support
    - Configuration merging with deep dictionary combination
    - Type validation for configuration values and environment variables
    - Complete configuration creation with validation and defaults
    - Pydantic model integration for settings management

Dependencies:
    - _config_base: Foundation configuration implementations
    - pydantic_settings: BaseSettings for environment integration
    - result: FlextResult pattern for consistent error handling
    - types: Type definitions for configuration data structures

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict

from flext_core._config_base import (
    _BaseConfig,
    _BaseConfigDefaults,
    _BaseConfigOps,
    _BaseConfigValidation,
)
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from flext_core.types import TAnyDict

# =============================================================================
# FLEXT CONFIG - Consolidado com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextConfig(
    _BaseConfig,
    _BaseConfigDefaults,
    _BaseConfigOps,
    _BaseConfigValidation,
):
    """Consolidated configuration management with multiple inheritance.

    Ultimate configuration orchestration class combining four specialized
    configuration bases through multiple inheritance, adding complex functionality
    impossible to achieve
    with single inheritance patterns. Provides comprehensive configuration operations
    with FlextResult integration for enterprise error handling.

    Architecture:
        - Multiple inheritance from four specialized configuration base classes
        - Complex orchestration methods combining multiple configuration types
        - FlextResult integration for all operations that can fail
        - Enterprise-grade configuration validation and error handling
        - File-based configuration loading with comprehensive validation

    Inherited Configuration Categories:
        - Base Configuration: Core loading and management (_BaseConfig)
        - Configuration Defaults: Default value management (_BaseConfigDefaults)
        - Configuration Operations: Merging and transformation (_BaseConfigOps)
        - Configuration Validation: Type and constraint checking (_BaseConfigValidation)

    Enterprise Features:
        - Complete configuration creation with validation and defaults
        - File-based configuration loading with comprehensive error handling
        - Configuration merging with validation of merged results
        - Environment variable access with type validation
        - Multi-step validation with detailed error reporting

    Orchestration Methods:
        - create_complete_config: Full configuration creation workflow
        - load_and_validate_from_file: File loading with validation
        - merge_and_validate_configs: Configuration merging with validation
        - get_env_with_validation: Environment variable access with validation

    Usage Patterns:
        # Complete configuration creation
        config_result = FlextConfig.create_complete_config(
            {"database_url": "sqlite:///app.db", "debug": True},
            apply_defaults=True,
            validate_all=True
        )

        # File-based configuration loading
        file_result = FlextConfig.load_and_validate_from_file(
            "config.json",
            required_keys=["database_url", "secret_key"]
        )

        # Configuration merging
        merge_result = FlextConfig.merge_and_validate_configs(
            base_config,
            override_config
        )

        # Environment variable access
        env_result = FlextConfig.get_env_with_validation(
            "DATABASE_URL",
            required=True,
            validate_type=str
        )
    """

    # =========================================================================
    # FUNCIONALIDADES ESPECÍFICAS (combinam múltiplas bases)
    # =========================================================================

    @classmethod
    def create_complete_config(
        cls,
        config_data: TAnyDict,
        *,
        apply_defaults: bool = True,
        validate_all: bool = True,
    ) -> FlextResult[TAnyDict]:
        """Create complete configuration orchestrating multiple inherited bases."""
        try:
            # Use inherited validation methods directly
            if validate_all:
                for key, value in config_data.items():
                    # Use basic validation - non-None check
                    validation_result = cls.validate_config_value(
                        value,
                        lambda x: x is not None,
                        f"Config value for '{key}' cannot be None",
                    )
                    if validation_result.is_failure:
                        error_msg = (
                            f"Config validation failed for {key}: "
                            f"{validation_result.error}"
                        )
                        return FlextResult.fail(error_msg)

            # Use inherited operations
            load_result = cls.safe_load_from_dict(config_data)
            if load_result.is_failure:
                return FlextResult.fail(f"Config load failed: {load_result.error}")

            final_config = load_result.unwrap()

            # Apply defaults using inherited method
            if apply_defaults:
                # Use basic defaults for common configuration keys
                default_values = {
                    "debug": False,
                    "timeout": 30,
                    "port": 8000,
                }
                defaults_result = cls.apply_defaults(final_config, default_values)
                if defaults_result.is_failure:
                    return FlextResult.fail(
                        f"Applying defaults failed: {defaults_result.error}",
                    )
                final_config = defaults_result.unwrap()

            return FlextResult.ok(final_config)

        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Complete config creation failed: {e}")

    @classmethod
    def load_and_validate_from_file(
        cls,
        file_path: str,
        *,
        required_keys: list[str] | None = None,
    ) -> FlextResult[TAnyDict]:
        """Load and validate config from file using inherited methods."""
        # Use inherited file loading
        load_result = cls.safe_load_json_file(file_path)
        if load_result.is_failure:
            return load_result

        config_data = load_result.unwrap()

        # Validate required keys using inherited validation
        if required_keys:
            for key in required_keys:
                if key not in config_data:
                    return FlextResult.fail(f"Required config key '{key}' not found")

                # Use inherited validation
                validation_result = cls.validate_config_value(
                    config_data[key],
                    lambda x: x is not None,
                    f"Config value for '{key}' cannot be None",
                )
                if validation_result.is_failure:
                    return FlextResult.fail(
                        f"Invalid config value for '{key}': {validation_result.error}",
                    )

        return FlextResult.ok(config_data)

    @classmethod
    def merge_and_validate_configs(
        cls,
        base_config: TAnyDict,
        override_config: TAnyDict,
    ) -> FlextResult[TAnyDict]:
        """Merge configs and validate using inherited methods."""
        try:
            # Use inherited merge
            merge_result = cls.merge_configs(base_config, override_config)
            if merge_result.is_failure:
                return FlextResult.fail(f"Config merge failed: {merge_result.error}")

            merged = merge_result.unwrap()

            # Validate merged result using inherited validation
            for key, value in merged.items():
                validation_result = cls.validate_config_value(
                    value,
                    lambda x: x is not None,
                    f"Merged config value for '{key}' cannot be None",
                )
                if validation_result.is_failure:
                    error_msg = (
                        f"Merged config validation failed for {key}: "
                        f"{validation_result.error}"
                    )
                    return FlextResult.fail(error_msg)

            return FlextResult.ok(merged)

        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Config merge failed: {e}")

    @classmethod
    def get_env_with_validation(
        cls,
        var_name: str,
        *,
        required: bool = False,
        default: str | None = None,
        validate_type: type | None = None,
    ) -> FlextResult[str]:
        """Get environment variable with validation using inherited methods."""
        # Use inherited env access
        env_result = cls.safe_get_env_var(var_name, default, required=required)
        if env_result.is_failure:
            return FlextResult.fail(env_result.error)

        value = env_result.unwrap()

        # Validate type using inherited validation if specified
        if validate_type:
            type_validation = cls.validate_config_type(value, validate_type, var_name)
            if type_validation.is_failure:
                error_msg = (
                    f"Environment variable '{var_name}' type validation failed: "
                    f"{type_validation.error}"
                )
                return FlextResult.fail(error_msg)

        return FlextResult.ok(value)


# =============================================================================
# FLEXT BASE SETTINGS - Simplified Pydantic integration
# =============================================================================


class FlextBaseSettings(PydanticBaseSettings):
    """Enterprise-grade settings class with Pydantic and FlextConfig integration.

    Advanced settings management combining Pydantic BaseSettings automatic
    environment loading with FlextConfig validation capabilities for enterprise
    configuration management with comprehensive error handling.

    Architecture:
        - Pydantic BaseSettings inheritance for automatic environment loading
        - FlextConfig integration for validation and error handling
        - Model configuration with strict validation and encoding settings
        - FlextResult integration for consistent error reporting

    Configuration Features:
        - Automatic .env file loading with UTF-8 encoding support
        - Case-insensitive environment variable matching
        - Strict validation with assignment-time checking
        - Extra field prohibition for configuration security
        - Integration with FlextConfig validation pipeline

    Enterprise Benefits:
        - Environment-based configuration with automatic type conversion
        - Comprehensive validation with detailed error reporting
        - FlextResult integration for consistent error handling patterns
        - Security-focused configuration with extra field prohibition
        - Development and production environment support

    Usage Patterns:
        class AppSettings(FlextBaseSettings):
            database_url: str = "sqlite:///app.db"
            secret_key: str
            debug: bool = False

            class Config:
                env_prefix = "APP_"

        # Create with validation
        settings_result = AppSettings.create_with_validation(
            debug=True,
            database_url="postgresql://localhost/myapp"
        )

        if settings_result.is_success:
            settings = settings_result.data
            print(settings.database_url)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        validate_assignment=True,
    )

    @classmethod
    def create_with_validation(
        cls,
        **overrides: object,
    ) -> FlextResult[FlextBaseSettings]:
        """Create settings with Pydantic validation."""
        try:
            # Let Pydantic handle validation directly
            instance = cls(**overrides)  # type: ignore[arg-type]
            return FlextResult.ok(instance)

        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Failed to create settings: {e}")


# =============================================================================
# EXPOSIÇÃO DIRETA DAS BASES ÚTEIS (aliases limpos sem herança vazia)
# =============================================================================

# Expose useful base classes directly with clean names
FlextConfigOps = _BaseConfigOps
FlextConfigDefaults = _BaseConfigDefaults
FlextConfigValidation = _BaseConfigValidation

# =============================================================================
# ESSENTIAL COMPATIBILITY FUNCTIONS (mantém apenas interface crítica)
# =============================================================================


# Mantém apenas merge_configs como função essencial para uso direto
def merge_configs(base: TAnyDict, override: TAnyDict) -> TAnyDict:
    """Merge two configuration dictionaries with deep combination.

    Essential function providing direct access to configuration merging.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary

    """
    merge_result = FlextConfig.merge_configs(base, override)
    return merge_result.unwrap() if merge_result.is_success else {}


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    # Simplified settings class
    "FlextBaseSettings",
    # Main consolidated class with multiple inheritance
    "FlextConfig",
    # Direct base exports (no inheritance overhead)
    "FlextConfigDefaults",
    "FlextConfigOps",
    "FlextConfigValidation",
    # Essential compatibility function
    "merge_configs",
]
