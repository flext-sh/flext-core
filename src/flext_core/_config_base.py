"""FLEXT Core Configuration Base Module.

Comprehensive configuration management system providing foundational patterns for
configuration loading, validation, and management across the FLEXT Core library.
Implements consolidated architecture with file operations and environment integration.

Architecture:
    - Single source of truth pattern for configuration operations
    - File-based configuration loading with JSON support and validation
    - Environment variable management with type checking and defaults
    - Configuration validation with predicate-based checking patterns
    - Default value application with merging and filtering capabilities
    - Exception-safe operations with comprehensive error handling

Configuration System Components:
    - _BaseConfigOps: Core configuration operations for file and environment handling
    - _BaseConfigValidation: Validation patterns for configuration values and types
    - _BaseConfigDefaults: Default value management and configuration merging
    - _BaseConfig: Utility functions for model configuration and standardization
    - Performance and observability configuration constants for system tuning

Maintenance Guidelines:
    - Maintain exception-safe operations with comprehensive error context
    - Use FlextResult pattern for all operations that can fail
    - Validate inputs using FlextValidators for consistency
    - Preserve configuration immutability through defensive copying
    - Keep file operations atomic with proper resource management
    - Follow path validation patterns for security and reliability

Design Decisions:
    - FlextResult integration for consistent error handling patterns
    - Path-based file operations with pathlib for cross-platform compatibility
    - Environment variable access with optional defaults and requirement checking
    - JSON-based configuration files with UTF-8 encoding and validation
    - Dictionary-based configuration with type checking and key validation
    - Defensive copying preventing configuration mutation

Configuration Management Features:
    - JSON file loading and saving with comprehensive error handling
    - Environment variable access with defaults and requirement validation
    - Configuration merging with precedence control and conflict resolution
    - Default value application preserving existing configuration values
    - Configuration filtering for security and namespace isolation
    - Type validation with clear error messages and context

File Operation Patterns:
    - Atomic file operations with proper encoding and error handling
    - Directory creation with parent directory support for file saving
    - Path validation ensuring file existence and type checking
    - JSON parsing with type validation and comprehensive error reporting
    - Resource management with proper file handle cleanup

Environment Integration:
    - Environment variable retrieval with optional default values
    - Required variable validation with clear error messages
    - Type-safe environment variable access with validation
    - Cross-platform environment variable handling

Dependencies:
    - json: JSON parsing and serialization for configuration files
    - os: Environment variable access and system integration
    - pathlib: Path operations and file system interaction
    - _result_base: FlextResult pattern for error handling
    - validation: FlextValidators for input validation

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core._result_base import _BaseResult
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# BASE CONFIGURATION OPERATIONS - Foundation for configuration management
# =============================================================================


class _BaseConfigOps:
    """Base configuration operations without external dependencies."""

    @staticmethod
    def safe_load_from_dict(
        config_dict: dict[str, object],
        required_keys: list[str] | None = None,
    ) -> _BaseResult[dict[str, object]]:
        """Safely load configuration from dictionary.

        Args:
            config_dict: Dictionary to load from
            required_keys: Optional list of required keys

        Returns:
            Result with validated configuration or error

        """
        if not FlextValidators.is_dict(config_dict):
            return _BaseResult.fail("Configuration must be a dictionary")

        # Validate required keys if provided
        if required_keys:
            if not FlextValidators.is_list(required_keys):
                return _BaseResult.fail("Required keys must be a list")

            missing_keys = [key for key in required_keys if key not in config_dict]
            if missing_keys:
                return _BaseResult.fail(
                    f"Missing required configuration keys: {', '.join(missing_keys)}",
                )

        # Create clean copy
        try:
            clean_config = dict(config_dict)
            return _BaseResult.ok(clean_config)
        except (TypeError, ValueError) as e:
            return _BaseResult.fail(f"Configuration loading failed: {e}")

    @staticmethod
    def safe_get_env_var(
        var_name: str,
        default: str | None = None,
        *,
        required: bool = False,
    ) -> _BaseResult[str]:
        """Safely get environment variable.

        Args:
            var_name: Environment variable name
            default: Default value if not found
            required: Whether variable is required

        Returns:
            Result with environment variable value or error

        """
        if not FlextValidators.is_non_empty_string(var_name):
            return _BaseResult.fail("Variable name must be non-empty string")

        try:
            value = os.environ.get(var_name)

            if value is None:
                if required:
                    error_msg = f"Required environment variable '{var_name}' not found"
                    return _BaseResult.fail(error_msg)
                if default is not None:
                    return _BaseResult.ok(default)
                error_msg = f"Environment variable '{var_name}' not found"
                return _BaseResult.fail(error_msg)

            return _BaseResult.ok(value)
        except (TypeError, OSError) as e:
            return _BaseResult.fail(f"Environment variable access failed: {e}")

    @staticmethod
    def safe_load_json_file(file_path: str | Path) -> _BaseResult[dict[str, object]]:
        """Safely load JSON configuration file.

        Args:
            file_path: Path to JSON file

        Returns:
            Result with parsed JSON data or error

        """
        try:
            path = Path(file_path)

            if not path.exists():
                return _BaseResult.fail(f"Configuration file not found: {path}")

            if not path.is_file():
                return _BaseResult.fail(f"Path is not a file: {path}")

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return _BaseResult.fail("JSON file must contain a dictionary")

            return _BaseResult.ok(data)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            return _BaseResult.fail(f"JSON file loading failed: {e}")

    @staticmethod
    def safe_save_json_file(
        data: dict[str, object],
        file_path: str | Path,
        *,
        create_dirs: bool = True,
    ) -> _BaseResult[None]:
        """Safely save configuration to JSON file.

        Args:
            data: Data to save
            file_path: Path to save to
            create_dirs: Whether to create parent directories

        Returns:
            Result indicating success or error

        """
        if not FlextValidators.is_dict(data):
            return _BaseResult.fail("Data must be a dictionary")

        try:
            path = Path(file_path)

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return _BaseResult.ok(None)
        except (OSError, TypeError, ValueError) as e:
            return _BaseResult.fail(f"JSON file saving failed: {e}")


# =============================================================================
# BASE CONFIGURATION VALIDATION - Validation patterns
# =============================================================================


class _BaseConfigValidation:
    """Base configuration validation operations without external dependencies."""

    @staticmethod
    def validate_config_value(
        value: object,
        validator: Callable[[object], bool],
        error_message: str = "Configuration value validation failed",
    ) -> _BaseResult[object]:
        """Validate configuration value with custom validator.

        Args:
            value: Value to validate
            validator: Validation function
            error_message: Error message if validation fails

        Returns:
            Result with value if valid or error

        """
        if not FlextValidators.is_callable(validator):
            return _BaseResult.fail("Validator must be callable")

        try:
            if callable(validator) and validator(value):
                return _BaseResult.ok(value)
            return _BaseResult.fail(error_message)
        except (TypeError, ValueError, AttributeError) as e:
            return _BaseResult.fail(f"Validation failed: {e}")

    @staticmethod
    def validate_config_type(
        value: object,
        expected_type: type,
        key_name: str = "value",
    ) -> _BaseResult[object]:
        """Validate configuration value type.

        Args:
            value: Value to validate
            expected_type: Expected type
            key_name: Name of configuration key

        Returns:
            Result with value if correct type or error

        """
        try:
            if isinstance(value, expected_type):
                return _BaseResult.ok(value)
            return _BaseResult.fail(
                f"Configuration '{key_name}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}",
            )
        except (TypeError, AttributeError) as e:
            return _BaseResult.fail(f"Type validation failed: {e}")

    @staticmethod
    def validate_config_range(
        value: float,
        min_value: float | None = None,
        max_value: float | None = None,
        key_name: str = "value",
    ) -> _BaseResult[int | float]:
        """Validate numeric configuration value range.

        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            key_name: Name of configuration key

        Returns:
            Result with value if in range or error

        """
        try:
            if min_value is not None and value < min_value:
                return _BaseResult.fail(
                    f"Configuration '{key_name}' must be >= {min_value}, got {value}",
                )

            if max_value is not None and value > max_value:
                return _BaseResult.fail(
                    f"Configuration '{key_name}' must be <= {max_value}, got {value}",
                )

            return _BaseResult.ok(value)
        except (TypeError, ValueError) as e:
            return _BaseResult.fail(f"Range validation failed: {e}")


# =============================================================================
# BASE CONFIGURATION DEFAULTS - Default value management
# =============================================================================


class _BaseConfigDefaults:
    """Base configuration defaults management without external dependencies."""

    @staticmethod
    def apply_defaults(
        config: dict[str, object],
        defaults: dict[str, object],
    ) -> _BaseResult[dict[str, object]]:
        """Apply default values to configuration.

        Args:
            config: Configuration dictionary
            defaults: Default values dictionary

        Returns:
            Result with configuration including defaults or error

        """
        if not FlextValidators.is_dict(config):
            return _BaseResult.fail("Configuration must be a dictionary")

        if not FlextValidators.is_dict(defaults):
            return _BaseResult.fail("Defaults must be a dictionary")

        try:
            # Create copy to avoid mutation
            result_config = dict(config)

            # Apply defaults for missing keys
            for key, default_value in defaults.items():
                if key not in result_config:
                    result_config[key] = default_value

            return _BaseResult.ok(result_config)
        except (TypeError, ValueError) as e:
            return _BaseResult.fail(f"Applying defaults failed: {e}")

    @staticmethod
    def merge_configs(
        *configs: dict[str, object],
    ) -> _BaseResult[dict[str, object]]:
        """Merge multiple configuration dictionaries.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Result with merged configuration or error

        """
        if not configs:
            return _BaseResult.ok({})

        try:
            merged: dict[str, object] = {}

            for i, config in enumerate(configs):
                if not FlextValidators.is_dict(config):
                    return _BaseResult.fail(f"Configuration {i} must be a dictionary")

                # Update with each config (later configs override earlier ones)
                merged.update(config)

            return _BaseResult.ok(merged)
        except (TypeError, ValueError, AttributeError) as e:
            return _BaseResult.fail(f"Configuration merging failed: {e}")

    @staticmethod
    def filter_config_keys(
        config: dict[str, object],
        allowed_keys: list[str],
    ) -> _BaseResult[dict[str, object]]:
        """Filter configuration to only include allowed keys.

        Args:
            config: Configuration dictionary
            allowed_keys: List of allowed keys

        Returns:
            Result with filtered configuration or error

        """
        if not FlextValidators.is_dict(config):
            return _BaseResult.fail("Configuration must be a dictionary")

        if not FlextValidators.is_list(allowed_keys):
            return _BaseResult.fail("Allowed keys must be a list")

        try:
            filtered = {
                key: value for key, value in config.items() if key in allowed_keys
            }

            return _BaseResult.ok(filtered)
        except (TypeError, ValueError, AttributeError) as e:
            return _BaseResult.fail(f"Configuration filtering failed: {e}")


# =============================================================================
# BASE CONFIGURATION CONSTANTS - Performance and observability
# =============================================================================


class _PerformanceConfig:
    """Performance configuration constants."""

    DEFAULT_CACHE_SIZE = 1000
    DEFAULT_TIMEOUT = 30
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_POOL_SIZE = 10
    DEFAULT_MAX_RETRIES = 3


class _ObservabilityConfig:
    """Observability configuration constants."""

    ENABLE_METRICS = True
    TRACE_ENABLED = True
    TRACE_SAMPLE_RATE = 0.1
    SLOW_OPERATION_THRESHOLD = 1000  # milliseconds
    CRITICAL_OPERATION_THRESHOLD = 5000  # milliseconds


class _BaseConfig:
    """Base configuration utilities."""

    @staticmethod
    def get_model_config(
        description: str = "Base configuration model",
        *,
        frozen: bool = True,
        extra: str = "forbid",
        validate_assignment: bool = True,
        use_enum_values: bool = True,
    ) -> dict[str, object]:
        """Get standardized model configuration."""
        return {
            "description": description,
            "frozen": frozen,
            "extra": extra,
            "validate_assignment": validate_assignment,
            "use_enum_values": use_enum_values,
            "str_strip_whitespace": True,
            "validate_all": True,
            "allow_reuse": True,
        }


# =============================================================================
# EXPORTS - Base configuration functionality only
# =============================================================================

__all__ = [
    # Base configuration classes
    "_BaseConfig",
    "_BaseConfigDefaults",
    "_BaseConfigOps",
    "_BaseConfigValidation",
    "_ObservabilityConfig",
    "_PerformanceConfig",
]
