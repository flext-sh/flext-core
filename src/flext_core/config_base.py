"""Base configuration abstractions following SOLID principles.

This module provides abstract configuration patterns and interfaces
used across the FLEXT ecosystem. Concrete implementations should be
in their respective domain modules.

Architecture:
    - Protocol-based abstractions (Interface Segregation)
    - Single responsibility classes (SRP)
    - Composition over inheritance (OCP)
    - Type-safe operations (LSP)
    - Dependency inversion through protocols (DIP)

Classes:
    FlextAbstractConfig: Base class for domain configurations.
    FlextAbstractSettings: Base class with environment support.
    FlextConfigOperations: Generic configuration operations.
    FlextConfigBuilder: Builder pattern for configurations.
    FlextConfigValidator: Validation operations.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict as PydanticConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


# =============================================================================
# PROTOCOLS - Interface Segregation Principle (ISP)
# =============================================================================


@runtime_checkable
class IConfigValidator(Protocol):
    """Protocol for configuration validation."""

    def validate(self) -> FlextResult[None]:
        """Validate configuration."""
        ...


@runtime_checkable
class IConfigLoader(Protocol):
    """Protocol for configuration loading."""

    def load(
        self, source: str | Path | dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Load configuration from source."""
        ...


@runtime_checkable
class IConfigMerger(Protocol):
    """Protocol for configuration merging."""

    def merge(
        self, base: dict[str, object], override: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Merge configurations."""
        ...


@runtime_checkable
class IConfigSerializer(Protocol):
    """Protocol for configuration serialization."""

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        ...

    def to_json(self) -> str:
        """Convert to JSON string."""
        ...


# =============================================================================
# BASE ABSTRACTIONS - Single Responsibility Principle (SRP)
# =============================================================================


class FlextAbstractConfig(ABC, BaseModel):
    """Abstract base class for configurations.

    Follows SRP by focusing only on configuration validation and serialization.
    """

    model_config = PydanticConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        frozen=True,
    )

    @abstractmethod
    def validate_config(self) -> FlextResult[None]:
        """Validate configuration specifics."""
        ...

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return self.model_dump(exclude_unset=True)

    def to_json(self) -> str:
        """Convert to JSON."""
        return self.model_dump_json(exclude_unset=True)

    def validate_instance(self) -> FlextResult[None]:
        """Validate configuration instance."""
        try:
            self.model_validate(self.model_dump())
        except Exception as e:
            return FlextResult.fail(f"Configuration validation failed: {e}")
        return self.validate_config()


class FlextAbstractSettings(ABC, BaseSettings):
    """Abstract base class for settings with environment support.

    Follows SRP by focusing on environment-aware configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid",
    )

    @abstractmethod
    def validate_settings(self) -> FlextResult[None]:
        """Validate settings specifics."""
        ...

    def validate_config(self) -> FlextResult[None]:
        """Validate settings."""
        return self.validate_settings()


# =============================================================================
# CONFIGURATION OPERATIONS - Open/Closed Principle (OCP)
# =============================================================================


class FlextConfigOperations:
    """Configuration operations following OCP.

    Open for extension through composition, closed for modification.
    """

    @staticmethod
    def merge_configs(
        base: dict[str, object],
        override: dict[str, object],
    ) -> dict[str, object]:
        """Deep merge two configuration dictionaries.

        Args:
            base: Base configuration dictionary.
            override: Override configuration dictionary.

        Returns:
            Merged configuration dictionary.

        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursive merge for nested dicts
                result[key] = FlextConfigOperations.merge_configs(
                    result[key],  # type: ignore[arg-type]
                    value,
                )
            else:
                result[key] = value

        return result

    @staticmethod
    def load_from_env(
        prefix: str = "",
        required: list[str] | None = None,
    ) -> FlextResult[dict[str, str]]:
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix to filter.
            required: List of required environment variables.

        Returns:
            FlextResult containing environment configuration.

        """
        config: dict[str, str] = {}
        required = required or []

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :] if prefix else key
                config[config_key] = value

        missing = [
            f"{prefix}{req}" if prefix else req for req in required if req not in config
        ]

        if missing:
            return FlextResult.fail(
                f"Missing required environment variables: {', '.join(missing)}",
            )

        return FlextResult.ok(config)

    @staticmethod
    def load_from_json(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Load configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file.

        Returns:
            FlextResult containing loaded configuration.

        """
        try:
            path = Path(file_path) if isinstance(file_path, str) else file_path

            if not path.exists():
                return FlextResult.fail(f"Configuration file not found: {path}")

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return FlextResult.fail(
                    "JSON file must contain an object at root level",
                )

            return FlextResult.ok(data)

        except json.JSONDecodeError as e:
            return FlextResult.fail(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            return FlextResult.fail(f"Failed to load configuration: {e}")

    # -----------------------------------------------------------------
    # Backward-compat helper names used by tests via patching
    # -----------------------------------------------------------------
    @staticmethod
    def safe_load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
        """Load JSON file safely with error handling."""
        return FlextConfigOperations.load_from_json(file_path)

    @staticmethod
    def safe_get_env_var(
        var_name: str, *, required: bool = False,
    ) -> FlextResult[str | None]:
        """Get environment variable safely with error handling."""
        result = FlextConfigOperations.load_from_env(
            "", [var_name] if required else None,
        )
        if result.is_failure:
            return FlextResult.fail(
                result.error or "Environment variable access failed",
            )
        data = result.unwrap()
        return FlextResult.ok(data.get(var_name))

    @staticmethod
    def safe_load_from_dict(
        config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Load configuration from dictionary safely."""
        try:
            return FlextResult.ok(dict(config))
        except Exception as e:
            return FlextResult.fail(f"Failed to load dict: {e}")

    @staticmethod
    def apply_defaults(
        config: dict[str, object], defaults: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Apply default values to configuration dictionary."""
        try:
            merged = FlextConfigOperations.merge_configs(defaults, config)
            return FlextResult.ok(merged)
        except Exception as e:
            return FlextResult.fail(f"Failed to apply defaults: {e}")

    @staticmethod
    def validate_required_fields(
        config: dict[str, object],
        required: list[str],
    ) -> FlextResult[None]:
        """Validate that required fields are present.

        Args:
            config: Configuration dictionary.
            required: List of required field names.

        Returns:
            FlextResult indicating validation result.

        """
        missing = [field for field in required if field not in config]

        if missing:
            return FlextResult.fail(f"Missing required fields: {', '.join(missing)}")

        return FlextResult.ok(None)


# =============================================================================
# VALIDATION OPERATIONS - Liskov Substitution Principle (LSP)
# =============================================================================


class FlextConfigValidator:
    """Configuration validation operations.

    All validators follow LSP - they can be substituted without affecting behavior.
    """

    @staticmethod
    def validate_required(
        config: dict[str, object], required_keys: list[str],
    ) -> FlextResult[None]:
        """Validate required keys exist.

        Args:
            config: Configuration dictionary.
            required_keys: List of required keys.

        Returns:
            FlextResult indicating validation result.

        """
        missing = [k for k in required_keys if k not in config]
        if missing:
            return FlextResult.fail(f"Missing required keys: {missing}")
        return FlextResult.ok(None)

    @staticmethod
    def validate_types(
        config: dict[str, object], type_map: dict[str, type],
    ) -> FlextResult[None]:
        """Validate value types.

        Args:
            config: Configuration dictionary.
            type_map: Mapping of keys to expected types.

        Returns:
            FlextResult indicating validation result.

        """
        for key, expected_type in type_map.items():
            if key in config and not isinstance(config[key], expected_type):
                actual = type(config[key]).__name__
                expected = expected_type.__name__
                return FlextResult.fail(
                    f"Type mismatch for '{key}': expected {expected}, got {actual}",
                )
        return FlextResult.ok(None)

    @staticmethod
    def validate_range(
        config: dict[str, object],
        key: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> FlextResult[None]:
        """Validate numeric value is within range.

        Args:
            config: Configuration dictionary.
            key: Key to validate.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            FlextResult indicating validation result.

        """
        if key not in config:
            return FlextResult.ok(None)

        value = config[key]
        if not isinstance(value, (int, float)):
            return FlextResult.fail(f"'{key}' must be numeric")

        if min_val is not None and value < min_val:
            return FlextResult.fail(f"'{key}' must be >= {min_val}")

        if max_val is not None and value > max_val:
            return FlextResult.fail(f"'{key}' must be <= {max_val}")

        return FlextResult.ok(None)

    @staticmethod
    def validate_pattern(
        config: dict[str, object], key: str, pattern: str,
    ) -> FlextResult[None]:
        """Validate string value matches pattern.

        Args:
            config: Configuration dictionary.
            key: Key to validate.
            pattern: Regular expression pattern.

        Returns:
            FlextResult indicating validation result.

        """
        if key not in config:
            return FlextResult.ok(None)

        value = config[key]
        if not isinstance(value, str):
            return FlextResult.fail(f"'{key}' must be a string")

        if not re.match(pattern, value):
            return FlextResult.fail(f"'{key}' does not match pattern '{pattern}'")

        return FlextResult.ok(None)


# =============================================================================
# BUILDER PATTERN - Dependency Inversion Principle (DIP)
# =============================================================================


class FlextConfigBuilder[T: FlextAbstractConfig]:
    """Builder pattern for configuration construction.

    Follows DIP by depending on abstractions, not concrete implementations.
    """

    def __init__(self, config_class: type[T]) -> None:
        """Initialize configuration builder.

        Args:
            config_class: Configuration class to build.

        """
        self._config_class = config_class
        self._data: dict[str, object] = {}
        self._validators: list[Callable[[dict[str, object]], FlextResult[None]]] = []

    def set(self, key: str, value: object) -> FlextConfigBuilder[T]:
        """Set configuration value.

        Args:
            key: Configuration key.
            value: Configuration value.

        Returns:
            Self for method chaining.

        """
        self._data[key] = value
        return self

    def from_dict(self, data: dict[str, object]) -> FlextConfigBuilder[T]:
        """Load from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            Self for method chaining.

        """
        self._data.update(data)
        return self

    def from_env(self, prefix: str = "") -> FlextConfigBuilder[T]:
        """Load from environment variables.

        Args:
            prefix: Environment variable prefix.

        Returns:
            Self for method chaining.

        """
        result = FlextConfigOperations.load_from_env(prefix)
        if result.is_success:
            self._data.update(result.unwrap())
        return self

    def from_json(self, file_path: str | Path) -> FlextConfigBuilder[T]:
        """Load from JSON file.

        Args:
            file_path: Path to JSON file.

        Returns:
            Self for method chaining.

        """
        result = FlextConfigOperations.load_from_json(file_path)
        if result.is_success:
            self._data.update(result.unwrap())
        return self

    def with_validator(
        self, validator: Callable[[dict[str, object]], FlextResult[None]],
    ) -> FlextConfigBuilder[T]:
        """Add a validator to the builder.

        Args:
            validator: Validation function.

        Returns:
            Self for method chaining.

        """
        self._validators.append(validator)
        return self

    def build(self) -> FlextResult[T]:
        """Build final configuration with validation.

        Returns:
            FlextResult containing built configuration.

        """
        # Run all validators
        for validator in self._validators:
            result = validator(self._data)
            if result.is_failure:
                return FlextResult.fail(f"Validation failed: {result.error}")

        try:
            config = self._config_class(**self._data)
            validation_result = config.validate_instance()

            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")

            return FlextResult.ok(config)

        except Exception as e:
            return FlextResult.fail(f"Failed to build configuration: {e}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FlextAbstractConfig",
    "FlextAbstractSettings",
    "FlextConfigBuilder",
    "FlextConfigOperations",
    "FlextConfigValidator",
    "IConfigLoader",
    "IConfigMerger",
    "IConfigSerializer",
    "IConfigValidator",
]
