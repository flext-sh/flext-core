"""CLI configuration adapter - minimal adapter for flext-cli integration.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Base configuration model for CLI applications.

This is a minimal adapter - actual CLI functionality should be in flext-cli.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from flext_core.config.base import BaseConfig
from flext_core.config.base import BaseSettings


class CLIConfig(BaseConfig):
    """Base configuration model for CLI applications."""

    # Output settings
    output_format: str = Field(
        default="text",
        description="Output format (text, json, yaml, table)",
        pattern="^(text|json|yaml|table)$",
    )
    verbose: bool = Field(default=False, description="Enable verbose output")
    quiet: bool = Field(default=False, description="Suppress non-error output")


class CLISettings(BaseSettings):
    """Base settings for CLI applications with environment variable support.

    This is a minimal adapter - actual CLI functionality should be in flext-cli.
    """

    model_config = SettingsConfigDict(env_prefix="FLEXT_")

    # CLI-specific environment settings
    default_profile: str = Field(
        default="default",
        description="Default profile name",
    )


def cli_config_to_dict(config: CLIConfig) -> dict[str, Any]:
    """Convert CLI config to dictionary for use by flext-cli.

    Args:
        config: CLI configuration instance

    Returns:
        Dictionary representation of CLI config

    """
    return config.model_dump(exclude_unset=True)


__all__ = [
    "CLIConfig",
    "CLISettings",
    "cli_config_to_dict",
]
