"""Enhanced base configuration for FLEXT framework.

This module provides enhanced configuration patterns for enterprise applications.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Environment(StrEnum):
    """Environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnhancedBaseConfig(BaseModel):
    """Enhanced base configuration with advanced features."""

    project_name: str = Field(description="Project name")
    project_version: str = Field(default="0.1.0", description="Project version")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Environment type",
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.environment == Environment.STAGING

    def is_test(self) -> bool:
        """Check if running in test."""
        return self.environment == Environment.TEST


class APIConfig(BaseModel):
    """API configuration model."""

    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(default=8000, description="API port")
    prefix: str = Field(default="/api/v1", description="API prefix")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    rate_limit: int = Field(default=100, description="Rate limit per minute")
    timeout: int = Field(default=30, description="Request timeout in seconds")
