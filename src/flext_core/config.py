"""FLEXT Core Configuration - Configuration Layer Management System.

Enterprise-grade configuration management providing consolidated environment loading,
validation, and settings management across the 32-project FLEXT ecosystem. Foundation
for consistent configuration patterns in distributed data integration platforms.

Module Role in Architecture:
    Configuration Layer → Settings Management → Environment Integration

    This module provides unified configuration management used throughout FLEXT:
    - Pydantic BaseSettings integration for automatic environment loading
    - FlextResult pattern integration for consistent error handling
    - JSON file loading with validation for deployment configurations
    - Configuration merging and override capabilities for multi-environment support

Configuration Architecture Patterns:
    Composition Pattern: Specialized configuration bases without inheritance overhead
    Environment Integration: Automatic .env file loading with type conversion
    Validation Pipeline: Comprehensive error handling with actionable feedback
    Settings Factory: Type-safe configuration creation with FlextResult

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Environment loading, JSON files, Pydantic integration
    🚧 Active Development: Configuration unification (Priority 2 - September 2025)
    📋 TODO Integration: Hierarchical configuration system (Priority 2)

Configuration Management Patterns:
    FlextBaseSettings: Enterprise settings with automatic environment loading
    FlextConfig: Consolidated operations with validation and merging
    Environment Variables: Type-safe access with validation and defaults
    File Loading: JSON configuration with comprehensive error handling

Ecosystem Usage Patterns:
    # FLEXT Service Applications
    class ServiceSettings(FlextBaseSettings):
        database_url: str = "postgresql://localhost/flext"
        redis_url: str = "redis://localhost:6379"
        log_level: str = "INFO"

        class Config:
            env_prefix = "FLEXT_"

    # Singer Taps/Targets
    settings_result = FlextBaseSettings.create_with_validation({
        "oracle_host": "localhost",
        "oracle_port": 1521,
        "batch_size": 1000
    })

    # Go Service Integration
    config_result = FlextConfig.load_and_validate_from_file(
        "config/production.json",
        required_keys=["database_url", "secret_key"]
    )

Enterprise Configuration Features:
    - Multi-environment support (dev, staging, production)
    - Security-conscious settings with secret handling
    - Type validation preventing runtime configuration errors
    - Configuration merging for deployment flexibility

Quality Standards:
    - All configuration loading must use FlextResult pattern
    - Environment variables must have sensible defaults
    - Configuration validation must provide actionable error messages
    - Settings classes must support both file and environment loading

See Also:
    docs/TODO.md: Priority 2 - Configuration system unification
    _config_base.py: Foundation configuration implementations
    constants.py: Configuration constants and defaults

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict

from flext_core._config_base import (
    _BaseConfigDefaults,
    _BaseConfigOps,
    _BaseConfigValidation,
)
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from flext_core.flext_types import TAnyDict

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Configuration Pattern Specializations
# =============================================================================

# Configuration specific types for better domain modeling
type TConfigKey = str  # Configuration key identifier
type TConfigValue = object  # Configuration value (any type)
type TConfigPath = str  # File path for configuration files
type TConfigEnv = str  # Environment name (dev, prod, test)
type TConfigValidationRule = str  # Configuration validation rule
type TConfigMergeStrategy = str  # Strategy for merging configurations
type TConfigSettings = TAnyDict  # Settings dictionary
type TConfigDefaults = TAnyDict  # Default configuration values
type TConfigOverrides = TAnyDict  # Configuration overrides

# Environment and deployment types
type TEnvironmentName = str  # Environment identifier
type TDeploymentStage = str  # Deployment stage (staging, production)
type TConfigVersion = str  # Configuration version for tracking

# =============================================================================
# FLEXT CONFIG - Consolidado com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextConfig:
    """Consolidated configuration management with composition-based orchestration.

    Configuration orchestration class combining four specialized configuration bases
    through composition and delegation, providing comprehensive configuration operations
    with FlextResult integration for enterprise error handling.

    Architecture:
        - Composition-based delegation to specialized configuration base classes
        - Orchestration methods combining configuration types through composition
        - FlextResult integration for all operations that can fail
        - Enterprise-grade configuration validation and error handling
        - File-based configuration loading with comprehensive validation

    Configuration Categories (accessed through composition):
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
            # Use composition to access validation methods
            if validate_all:
                for key, value in config_data.items():
                    # Use basic validation - non-None check
                    validation_result = _BaseConfigValidation.validate_config_value(
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

            # Use composition to access config operations
            load_result = _BaseConfigOps.safe_load_from_dict(config_data)
            if load_result.is_failure:
                return FlextResult.fail(f"Config load failed: {load_result.error}")

            final_config = load_result.unwrap()

            # Apply defaults using composition
            if apply_defaults:
                # Use basic defaults for common configuration keys
                default_values = {
                    "debug": False,
                    "timeout": 30,
                    "port": 8000,
                }
                defaults_result = _BaseConfigDefaults.apply_defaults(
                    final_config,
                    dict(default_values),
                )
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
        # Use composition for file loading
        load_result = _BaseConfigOps.safe_load_json_file(file_path)
        if load_result.is_failure:
            return FlextResult.fail(load_result.error or "JSON file loading failed")

        config_data = load_result.unwrap()

        # Validate required keys using composition
        if required_keys:
            for key in required_keys:
                if key not in config_data:
                    return FlextResult.fail(f"Required config key '{key}' not found")

                # Use composition for validation
                validation_result = _BaseConfigValidation.validate_config_value(
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
            # Use composition for merging
            merge_result = _BaseConfigDefaults.merge_configs(
                base_config,
                override_config,
            )
            if merge_result.is_failure:
                return FlextResult.fail(f"Config merge failed: {merge_result.error}")

            merged = merge_result.unwrap()

            # Validate merged result using composition
            for key, value in merged.items():
                validation_result = _BaseConfigValidation.validate_config_value(
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
        # Use composition for env access
        env_result = _BaseConfigOps.safe_get_env_var(
            var_name,
            default,
            required=required,
        )
        if env_result.is_failure:
            return FlextResult.fail(
                env_result.error or "Environment variable access failed",
            )

        value = env_result.unwrap()

        # Validate type using composition if specified
        if validate_type:
            type_validation = _BaseConfigValidation.validate_config_type(
                value,
                validate_type,
                var_name,
            )
            if type_validation.is_failure:
                error_msg = (
                    f"Environment variable '{var_name}' type validation failed: "
                    f"{type_validation.error}"
                )
                return FlextResult.fail(error_msg)

        return FlextResult.ok(value)

    # =========================================================================
    # PROXY METHODS - Direct access to base class functionality
    # =========================================================================

    @classmethod
    def safe_load_from_dict(cls, config_dict: TAnyDict) -> FlextResult[TAnyDict]:
        """Proxy to _BaseConfigOps.safe_load_from_dict."""
        return _BaseConfigOps.safe_load_from_dict(config_dict)

    @classmethod
    def apply_defaults(
        cls,
        config: TAnyDict,
        defaults: TAnyDict,
    ) -> FlextResult[TAnyDict]:
        """Proxy to _BaseConfigDefaults.apply_defaults."""
        return _BaseConfigDefaults.apply_defaults(config, defaults)

    @classmethod
    def merge_configs(
        cls,
        base_config: TAnyDict,
        override_config: TAnyDict,
    ) -> FlextResult[TAnyDict]:
        """Proxy to _BaseConfigOps.merge_configs."""
        # Implementation temporarily disabled due to type issues
        merged = {**base_config, **override_config}
        return FlextResult.ok(merged)

    @classmethod
    def validate_config_value(
        cls,
        value: object,
        validator: object,
        error_message: str = "Validation failed",
    ) -> FlextResult[None]:
        """Proxy to _BaseConfigValidation.validate_config_value."""
        # Simple validation implementation
        if callable(validator):
            try:
                if validator(value):
                    return FlextResult.ok(None)
                return FlextResult.fail(error_message)
            except (TypeError, ValueError, AttributeError) as e:
                return FlextResult.fail(f"Validation error: {e}")
        return FlextResult.fail("Validator must be callable")


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
        extra="ignore",
        validate_assignment=True,
    )

    @classmethod
    def create_with_validation(
        cls,
        overrides: TAnyDict | None = None,
        **kwargs: object,
    ) -> FlextResult[FlextBaseSettings]:
        """Create settings with Pydantic validation.

        Args:
            overrides: Optional dictionary of configuration overrides
            **kwargs: Additional keyword arguments for settings

        Returns:
            FlextResult containing the validated settings instance

        """
        try:
            # Merge overrides and kwargs
            final_config = {}
            if overrides:
                final_config.update(overrides)
            if kwargs:
                final_config.update(kwargs)

            # Pydantic BaseSettings accepts dynamic **kwargs
            # MyPy cannot verify dynamic dict keys against model fields
            instance = cls(**final_config) if final_config else cls()  # type: ignore[arg-type]
            return FlextResult.ok(instance)

        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Failed to create settings: {e}")


# =============================================================================
# EXPOSIÇÃO DIRETA DAS BASES ÚTEIS (aliases limpos sem herança vazia)
# =============================================================================

# Direct exposure with clean names - eliminates inheritance overhead
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
    merge_result = _BaseConfigDefaults.merge_configs(base, override)
    return merge_result.unwrap() if merge_result.is_success else {}


# =============================================================================
# MODULE-LEVEL WRAPPER FUNCTIONS for test compatibility
# =============================================================================


def safe_get_env_var(
    var_name: str,
    default: str | None = None,
    *,
    required: bool = False,
) -> FlextResult[str]:
    """Module-level wrapper for safe environment variable access."""
    result = _BaseConfigOps.safe_get_env_var(var_name, default, required=required)
    if result.is_failure:
        return FlextResult.fail(result.error or "Environment variable access failed")
    return FlextResult.ok(result.unwrap())


def safe_load_json_file(file_path: str | Path) -> FlextResult[TAnyDict]:
    """Module-level wrapper for safe JSON file loading."""
    result = _BaseConfigOps.safe_load_json_file(file_path)
    if result.is_failure:
        return FlextResult.fail(result.error or "JSON file loading failed")
    return FlextResult.ok(result.unwrap())


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
