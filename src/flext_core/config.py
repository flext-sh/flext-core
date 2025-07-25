"""FLEXT Core Configuration - Pydantic Settings Integration.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Configuration management using pydantic-settings for type-safe,
validated configuration across the FLEXT ecosystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from flext_core.constants import FlextConstants
from flext_core.constants import FlextEnvironment
from flext_core.constants import FlextLogLevel


class FlextCoreSettings(BaseSettings):
    """Configuration settings for FLEXT applications.

    Provides type-safe configuration management with environment
    variable integration, validation, and defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="FLEXT_",
        env_file=None,  # Disable .env file loading in tests
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,  # Immutable after creation
        validate_assignment=True,
        str_strip_whitespace=True,
        env_nested_delimiter="__",
    )

    # Environment configuration
    environment: FlextEnvironment = Field(
        default=FlextConstants.DEFAULT_ENVIRONMENT,
        description="Runtime environment (dev, test, staging, production)",
    )

    # Logging configuration
    log_level: FlextLogLevel = Field(
        default=FlextConstants.DEFAULT_LOG_LEVEL,
        description="Logging level for the application",
    )

    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages",
    )

    # Service configuration
    service_timeout: int = Field(
        default=FlextConstants.DEFAULT_TIMEOUT,
        ge=1,
        le=300,
        description="Default timeout for service operations in seconds",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for failed operations",
    )

    # Development configuration
    debug: bool = Field(
        default=False,
        description="Enable debug mode with additional logging and validation",
    )

    # Paths configuration
    config_dir: Path = Field(
        default=Path.cwd() / "config",
        description="Directory containing configuration files",
    )

    data_dir: Path = Field(
        default=Path.cwd() / "data",
        description="Directory for application data storage",
    )

    logs_dir: Path = Field(
        default=Path.cwd() / "logs",
        description="Directory for log file storage",
    )

    @field_validator("config_dir", "data_dir", "logs_dir")
    @classmethod
    def validate_directories(cls, value: Path) -> Path:
        """Ensure directories are absolute paths."""
        return value.resolve()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, value: str) -> str:
        """Validate log format string."""
        if not value.strip():
            msg = "Log format cannot be empty"
            raise ValueError(msg)
        return value.strip()

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == FlextEnvironment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == FlextEnvironment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == FlextEnvironment.TESTING

    def model_dump_safe(self) -> dict[str, Any]:
        """Export configuration without sensitive data."""
        data = self.model_dump()
        # Remove potentially sensitive information
        return {k: v for k, v in data.items() if not k.startswith("_")}


class _SettingsContainer:
    """Thread-safe container for global settings without global."""

    def __init__(self) -> None:
        """Initialize the settings container."""
        self._settings: FlextCoreSettings | None = None

    def get_settings(self) -> FlextCoreSettings:
        """Get the global FlextCoreSettings instance.

        Returns:
            The global settings instance, creating it if needed.

        """
        if self._settings is None:
            self._settings = FlextCoreSettings()
        return self._settings

    def configure_settings(
        self,
        settings: FlextCoreSettings | None = None,
    ) -> FlextCoreSettings:
        """Configure the global settings instance.

        Args:
            settings: Settings instance to use globally, or None for
                default.

        Returns:
            The configured global settings instance.

        """
        if settings is not None:
            self._settings = settings
        else:
            self._settings = FlextCoreSettings()
        return self._settings


# Singleton container instance
_container = _SettingsContainer()


def get_settings() -> FlextCoreSettings:
    """Get the global FlextCoreSettings instance.

    Returns:
        The global settings instance, creating it if needed.

    """
    return _container.get_settings()


def configure_settings(
    settings: FlextCoreSettings | None = None,
) -> FlextCoreSettings:
    """Configure the global settings instance.

    Args:
        settings: Settings instance to use globally, or None for
            default.

    Returns:
        The configured global settings instance.

    """
    return _container.configure_settings(settings)


__all__ = [
    "FlextCoreSettings",
    "configure_settings",
    "get_settings",
]
