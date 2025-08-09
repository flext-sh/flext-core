"""Concrete configuration implementations using config_base abstractions.

Provides domain-specific configuration classes and Pydantic-based settings
with environment variable loading using SOLID principles.

Classes:
    FlextConfig: Main configuration class with business rule validation.
    FlextSettings: Environment-aware settings using base abstractions.
    FlextConfigManager: High-level configuration orchestration.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict

from flext_core.config_base import (
    FlextAbstractConfig,
    FlextAbstractSettings,
    FlextConfigOperations,
)
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.config_models import (
        FlextObservabilityConfig,
        FlextPerformanceConfig,
    )
else:
    # Import at runtime to avoid circular imports
    try:
        from flext_core.config_models import (
            FlextObservabilityConfig,
            FlextPerformanceConfig,
        )
    except ImportError:
        # Create stub classes for backward compatibility
        class FlextObservabilityConfig:  # type: ignore[no-redef]
            """Stub class for observability configuration."""

        class FlextPerformanceConfig:  # type: ignore[no-redef]
            """Stub class for performance configuration."""


# =============================================================================
# CONFIGURATION MODELS - Imported from models or created here
# =============================================================================


class FlextConfig:
    """Main configuration class for FlexT system."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize configuration with provided values."""
        # Set values from kwargs or defaults
        self.debug = kwargs.get("debug", False)
        self.environment = kwargs.get("environment", "development")
        self.log_level = kwargs.get("log_level", "INFO")

        # Store all kwargs for dynamic access
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate configuration business rules."""
        return FlextResult.ok(None)

    # -----------------------------------------------------------------
    # Backward-compat: expose high-level manager APIs on FlextConfig
    # -----------------------------------------------------------------
    @classmethod
    def create_complete_config(
        cls,
        config_data: TAnyDict,
        *,
        apply_defaults: bool = True,
        validate_all: bool = True,
    ) -> FlextResult[TAnyDict]:
        """Create complete configuration from data."""
        # Delegate to FlextConfigManager defined later in this module
        # Use getattr to avoid forward reference issues
        mgr_cls = __import__(
            "flext_core.config", fromlist=["FlextConfigManager"],
        ).FlextConfigManager
        return mgr_cls.create_complete_config(  # type: ignore[no-any-return]
            config_data,
            apply_defaults=apply_defaults,
            validate_all=validate_all,
        )

    @classmethod
    def merge_and_validate_configs(
        cls,
        base_config: TAnyDict,
        override_config: TAnyDict,
    ) -> FlextResult[TAnyDict]:
        """Merge and validate configurations."""
        # Delegate to FlextConfigManager defined later in this module
        mgr_cls = __import__(
            "flext_core.config", fromlist=["FlextConfigManager"],
        ).FlextConfigManager
        return mgr_cls.merge_and_validate_configs(base_config, override_config)  # type: ignore[no-any-return]

    @classmethod
    def get_env_with_validation(
        cls,
        var_name: str,
        *,
        required: bool = False,
        default: str | None = None,
        validate_type: type | None = None,
    ) -> FlextResult[str]:
        """Get environment variable with validation."""
        # Delegate to FlextConfigManager defined later in this module
        mgr_cls = __import__(
            "flext_core.config", fromlist=["FlextConfigManager"],
        ).FlextConfigManager
        return mgr_cls.get_env_with_validation(  # type: ignore[no-any-return]
            var_name,
            required=required,
            default=default,
            validate_type=validate_type,
        )

    @classmethod
    def load_and_validate_from_file(
        cls, file_path: str, *, required_keys: list[str] | None = None,
    ) -> FlextResult[FlextConfig]:
        """Load configuration from file and validate."""
        try:
            path = Path(file_path)
            if not path.exists():
                return FlextResult.fail(f"Configuration file not found: {file_path}")

            with path.open() as f:
                data = json.load(f)

            if required_keys:
                missing = [k for k in required_keys if k not in data]
                if missing:
                    return FlextResult.fail(
                        f"Missing required keys: {', '.join(missing)}",
                    )

            config = cls(**data)
            validation_result = config.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")

            return FlextResult.ok(config)

        except Exception as e:
            return FlextResult.fail(f"Failed to load configuration: {e}")


# =============================================================================
# DOMAIN-SPECIFIC CONFIGURATION DEFAULTS
# =============================================================================


class FlextConfigDefaults:
    """Domain-specific configuration defaults extending base abstractions."""

    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_DEBUG = False
    DEFAULT_ENVIRONMENT = "development"
    DEFAULT_PORT = 8000
    DEFAULT_TIMEOUT = 30

    @classmethod
    def apply_defaults(
        cls,
        config: dict[str, object],
        defaults: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Apply domain defaults using base operations.

        Args:
            config: Configuration dictionary.
            defaults: Default values dictionary.

        Returns:
            FlextResult containing merged configuration.

        """
        result = FlextConfigOperations.merge_configs(config, defaults)
        return FlextResult.ok(result)

    @classmethod
    def merge_configs(
        cls,
        base_config: dict[str, object],
        override_config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge configuration dictionaries using base operations.

        Args:
            base_config: Base configuration dictionary.
            override_config: Override configuration dictionary.

        Returns:
            FlextResult containing merged configuration.

        """
        result = FlextConfigOperations.merge_configs(base_config, override_config)
        return FlextResult.ok(result)

    @classmethod
    def get_domain_defaults(cls) -> dict[str, object]:
        """Get domain-specific default values."""
        return {
            "debug": cls.DEFAULT_DEBUG,
            "log_level": cls.DEFAULT_LOG_LEVEL,
            "environment": cls.DEFAULT_ENVIRONMENT,
            "port": cls.DEFAULT_PORT,
            "timeout": cls.DEFAULT_TIMEOUT,
        }


# Use FlextConfigOperations from config_base.py - eliminates duplication
# Domain-specific operations are composition over base operations


class FlextConfigOps:
    """Domain configuration operations using base abstractions.

    Provides domain-specific configuration loading operations by composing
    base configuration operations from config_base.py. Eliminates code
    duplication while maintaining clean interface.
    """

    @classmethod
    def safe_load_from_dict(
        cls,
        config_dict: dict[str, object],
        required_keys: list[str] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Safely load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary to load.
            required_keys: List of required keys to validate (optional).

        Returns:
            FlextResult containing loaded configuration.

        """
        if required_keys:
            missing = [k for k in required_keys if k not in config_dict]
            if missing:
                return FlextResult.fail(
                    f"Missing required keys: {', '.join(missing)}",
                )
        return FlextResult.ok(dict(config_dict))

    @classmethod
    def safe_get_env_var(
        cls,
        var_name: str,
        *,
        required: bool = False,
        default: str | None = None,
    ) -> FlextResult[str | None]:
        """Safely get environment variable using base operations.

        Args:
            var_name: Environment variable name.
            default: Default value if not found.
            required: Whether variable is required.

        Returns:
            FlextResult containing variable value.

        """
        env_result = FlextConfigOperations.load_from_env(
            f"{var_name}_",
            [var_name] if required else None,
        )
        if env_result.is_failure:
            return FlextResult.fail(env_result.error or "Env load failed")
        value = env_result.unwrap().get(var_name)
        if value is None:
            return FlextResult.ok(default)
        return FlextResult.ok(value)

    @classmethod
    def safe_load_json_file(cls, file_path: str) -> FlextResult[dict[str, object]]:
        """Safely load JSON file using base operations.

        Args:
            file_path: Path to JSON file.

        Returns:
            FlextResult containing loaded JSON data.

        """
        return FlextConfigOperations.load_from_json(file_path)


# Use validation from validation_base.py - eliminates duplication
# Domain-specific validation is composition over base validation


class FlextConfigValidation:
    """Domain configuration validation using base abstractions.

    Provides configuration validation capabilities by composing base
    validation patterns from validation_base.py. Eliminates code
    duplication while providing domain-specific validation logic.
    """

    @classmethod
    def validate_config_value(
        cls,
        value: object,
        validator: object,
        error_message: str = "Validation failed",
    ) -> FlextResult[object]:
        """Validate configuration value with custom validator.

        Args:
            value: Value to validate.
            validator: Validation function.
            error_message: Error message if validation fails.

        Returns:
            FlextResult containing validated value.

        """
        try:
            if callable(validator) and validator(value):
                return FlextResult.ok(value)
            return FlextResult.fail(error_message)
        except Exception as e:
            return FlextResult.fail(f"Validation error: {e}")

    @classmethod
    def validate_config_type(
        cls,
        value: object,
        expected_type: type,
        var_name: str,
    ) -> FlextResult[object]:
        """Validate configuration value type.

        Args:
            value: Value to validate.
            expected_type: Expected type.
            var_name: Variable name for error messages.

        Returns:
            FlextResult containing validated value.

        """
        try:
            if isinstance(value, expected_type):
                return FlextResult.ok(value)
            return FlextResult.fail(
                f"Expected {expected_type.__name__} for '{var_name}', "
                f"got {type(value).__name__}",
            )
        except Exception as e:
            return FlextResult.fail(f"Type validation error for '{var_name}': {e}")


if TYPE_CHECKING:
    from flext_core.typings import TAnyDict

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Configuration Pattern Specializations
# =============================================================================

# Configuration specific types for better domain modeling

# Environment and deployment types

# =============================================================================
# FLEXT CONFIG - Consolidado com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextConfigManager:
    """Configuration management with validation and file loading.

    Provides comprehensive configuration operations including file loading,
    environment variable access, validation, and configuration merging.
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
        """Create complete configuration with validation and defaults.

        Args:
            config_data: Configuration data dictionary.
            apply_defaults: Whether to apply default values.
            validate_all: Whether to validate all values.

        Returns:
            FlextResult containing complete configuration.

        """
        try:
            # Use composition to access validation methods
            if validate_all:
                for key, value in config_data.items():
                    # Use basic validation - non-None check
                    validation_result = FlextConfigValidation.validate_config_value(
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
            load_result = FlextConfigOps.safe_load_from_dict(config_data)
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
                defaults_result = FlextConfigDefaults.apply_defaults(
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
        """Load and validate configuration from JSON file.

        Args:
            file_path: Path to configuration file.
            required_keys: List of required configuration keys.

        Returns:
            FlextResult containing loaded configuration.

        """
        # Use composition for file loading
        load_result = FlextConfigOps.safe_load_json_file(file_path)
        if load_result.is_failure:
            return FlextResult.fail(load_result.error or "JSON file loading failed")

        config_data = load_result.unwrap()

        # Validate required keys using composition
        if required_keys:
            for key in required_keys:
                if key not in config_data:
                    return FlextResult.fail(f"Required config key '{key}' not found")

                # Use composition for validation
                validation_result = FlextConfigValidation.validate_config_value(
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
        """Merge and validate two configurations.

        Args:
            base_config: Base configuration dictionary.
            override_config: Override configuration dictionary.

        Returns:
            FlextResult containing merged configuration.

        """
        try:
            # Use composition for merging
            merge_result = FlextConfigDefaults.merge_configs(
                base_config,
                override_config,
            )
            if merge_result.is_failure:
                return FlextResult.fail(f"Config merge failed: {merge_result.error}")

            merged = merge_result.unwrap()

            # Validate merged result using composition
            for key, value in merged.items():
                validation_result = FlextConfigValidation.validate_config_value(
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
        """Get environment variable with validation.

        Args:
            var_name: Environment variable name.
            required: Whether variable is required.
            default: Default value if not found.
            validate_type: Type to validate against.

        Returns:
            FlextResult containing environment variable value.

        """
        # Use composition for env access
        env_result = FlextConfigOps.safe_get_env_var(
            var_name,
            required=required,
        )
        if env_result.is_failure:
            return FlextResult.fail(
                env_result.error or "Environment variable access failed",
            )

        value = env_result.unwrap()

        # Handle default value if variable is None/not found
        if value is None:
            if default is not None:
                value = default
            elif required:
                return FlextResult.fail(
                    f"Required environment variable '{var_name}' is None",
                )
            else:
                return FlextResult.fail(f"Environment variable '{var_name}' is None")

        # Validate type using composition if specified
        if validate_type:
            type_validation = FlextConfigValidation.validate_config_type(
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

        # Ensure value is string for type safety
        return FlextResult.ok(str(value))

    # =========================================================================
    # PROXY METHODS - Direct access to base class functionality
    # =========================================================================

    @classmethod
    def safe_load_from_dict(cls, config_dict: TAnyDict) -> FlextResult[TAnyDict]:
        """Load configuration from dictionary."""
        return FlextConfigOps.safe_load_from_dict(config_dict)

    @classmethod
    def apply_defaults(
        cls,
        config: TAnyDict,
        defaults: TAnyDict,
    ) -> FlextResult[TAnyDict]:
        """Apply default values to configuration."""
        return FlextConfigDefaults.apply_defaults(config, defaults)

    @classmethod
    def merge_configs(
        cls,
        base_config: TAnyDict,
        override_config: TAnyDict,
    ) -> FlextResult[TAnyDict]:
        """Merge two configuration dictionaries."""
        # Use centralized implementation
        return FlextConfigDefaults.merge_configs(base_config, override_config)

    @classmethod
    def validate_config_value(
        cls,
        value: object,
        validator: Callable[[object], bool],
        error_message: str = "Validation failed",
    ) -> FlextResult[None]:
        """Validate configuration value."""
        # Use centralized implementation
        validation_result = FlextConfigValidation.validate_config_value(
            value,
            validator,
            error_message,
        )
        # Convert to None result for compatibility
        if validation_result.is_success:
            return FlextResult.ok(None)
        return FlextResult.fail(validation_result.error or error_message)


# =============================================================================
# FLEXT BASE SETTINGS - Simplified Pydantic integration
# =============================================================================


class FlextSettings(FlextAbstractSettings, PydanticBaseSettings):
    """Base settings class with automatic environment loading.

    Extends base abstractions and Pydantic BaseSettings with FlextResult
    integration for type-safe configuration management with .env file support.
    Uses config_base.py abstractions following SOLID principles.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    def validate_settings(self) -> FlextResult[None]:
        """Validate settings - implements abstract method from base.

        Returns:
            FlextResult indicating validation success or failure.

        """
        return FlextResult.ok(None)

    @classmethod
    def create_with_validation(
        cls,
        overrides: TAnyDict | None = None,
        **kwargs: object,
    ) -> FlextResult[FlextSettings]:
        """Create settings with Pydantic validation.

        Args:
            overrides: Optional dictionary of configuration overrides.
            **kwargs: Additional keyword arguments for settings.

        Returns:
            FlextResult containing the validated settings instance.

        """
        try:
            # Merge overrides and kwargs with type compatibility
            final_config: dict[str, object] = {}
            if overrides:
                # Overrides are TAnyDict compatible, convert to object
                final_config.update(overrides)
            if kwargs:
                final_config.update(kwargs)

            # Pydantic BaseSettings accepts dynamic **kwargs for model construction
            # We need to filter only the model fields, not BaseSettings init params
            # BaseSettings __init__ has specific params that shouldn't be
            # passed as kwargs

            # Create instance by directly passing values to the model initializer
            # BaseSettings validation will handle the fields appropriately
            if final_config:
                # Filter out any BaseSettings-specific init params if present
                model_fields = {
                    k: v
                    for k, v in final_config.items()
                    if not k.startswith("_")
                    and k
                    not in {
                        "_case_sensitive",
                        "_env_prefix",
                        "_env_file",
                        "_env_file_encoding",
                        "_env_nested_delimiter",
                        "_secrets_dir",
                        "_cli_settings_source",
                    }
                }
                instance = cls.model_validate(model_fields)
            else:
                instance = cls()
            return FlextResult.ok(instance)

        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Failed to create settings: {e}")


# =============================================================================
# FLEXT CONFIG - Main Configuration Class
# =============================================================================


class FlextMainConfig(FlextAbstractConfig, FlextSettings):  # type: ignore[misc]
    """Main configuration class with environment integration.

    Provides immutable configuration with environment variable loading
    and business rule validation support. Uses both abstract config
    and settings patterns from base abstractions.
    """

    def validate_config(self) -> FlextResult[None]:
        """Validate configuration specifics - implements abstract method.

        Returns:
            FlextResult indicating validation success or failure.

        """
        return self.validate_business_rules()

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules - override in subclasses.

        Returns:
            FlextResult indicating validation success or failure.

        """
        return FlextResult.ok(None)

    # -----------------------------------------------------------------
    # Test compatibility: keep old API available on FlextConfig
    # -----------------------------------------------------------------
    @classmethod
    def create_complete_config(
        cls,
        config_data: TAnyDict,
        *,
        apply_defaults: bool = True,
        validate_all: bool = True,
    ) -> FlextResult[TAnyDict]:
        """Compatibility alias to manager's create_complete_config."""
        return FlextConfigManager.create_complete_config(
            config_data,
            apply_defaults=apply_defaults,
            validate_all=validate_all,
        )

    @classmethod
    def load_and_validate_from_file(
        cls, file_path: str, required_keys: list[str] | None = None,
    ) -> FlextResult[FlextConfig]:
        """Load configuration from file and validate."""
        try:
            path = Path(file_path)
            if not path.exists():
                return FlextResult.fail(f"Configuration file not found: {file_path}")

            with path.open() as f:
                data = json.load(f)

            # Check required keys if provided
            if required_keys:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    return FlextResult.fail(f"Missing required keys: {missing_keys}")

            config = cls.model_validate(data)
            validation_result = config.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")

            return FlextResult.ok(config)  # type: ignore[arg-type]

        except Exception as e:
            return FlextResult.fail(f"Failed to load configuration: {e}")


# =============================================================================
# DIRECT EXPOSURE OF USEFUL BASES (clean aliases without empty inheritance)
# =============================================================================

# ARCHITECTURAL DECISION: Direct exposure with clean names - completely
# eliminates empty inheritance and code duplication. Each assignment provides
# full functionality from centralized base implementation.

# All implementations come from base_config.py - NO duplication

# Configuration classes already imported - provides full functionality
# FlextConfigOps already imported - provides file loading and environment management
# FlextConfigDefaults already imported - provides default value management
# FlextConfigValidation already imported - provides type and constraint validation

# =============================================================================
# ESSENTIAL COMPATIBILITY FUNCTIONS (mantém apenas interface crítica)
# =============================================================================


# Mantém apenas merge_configs como função essencial para uso direto
def merge_configs(base: TAnyDict, override: TAnyDict) -> TAnyDict:
    """Merge two configuration dictionaries.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.

    """
    merge_result = FlextConfigDefaults.merge_configs(base, override)
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
    """Safe environment variable access.

    Args:
        var_name: Environment variable name.
        default: Default value if not found.
        required: Whether variable is required.

    Returns:
        FlextResult containing variable value.

    """
    result = FlextConfigOps.safe_get_env_var(var_name, required=required)
    if result.is_failure:
        return FlextResult.fail(result.error or "Environment variable access failed")
    value = result.unwrap()
    if value is None:
        if default is not None:
            return FlextResult.ok(default)
        return FlextResult.fail(f"Environment variable '{var_name}' is None")
    return FlextResult.ok(value)


def safe_load_json_file(file_path: str | Path) -> FlextResult[TAnyDict]:
    """Safe JSON file loading.

    Args:
        file_path: Path to JSON file.

    Returns:
        FlextResult containing loaded JSON data.

    """
    # Convert Path to str if needed
    file_path_str = str(file_path) if isinstance(file_path, Path) else file_path
    result = FlextConfigOps.safe_load_json_file(file_path_str)
    if result.is_failure:
        return FlextResult.fail(result.error or "JSON file loading failed")
    return FlextResult.ok(result.unwrap())


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    "FlextConfig",
    "FlextConfigDefaults",
    "FlextConfigManager",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextObservabilityConfig",  # Re-exported from config_models
    "FlextPerformanceConfig",  # Re-exported from config_models
    "FlextSettings",
    "merge_configs",
]
