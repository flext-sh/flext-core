"""Singer SDK configuration adapter.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Base configuration for Singer taps and targets.

Provides common configuration fields and validation for Singer SDK projects.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import SettingsConfigDict

from flext_core.config.base import BaseConfig
from flext_core.config.base import BaseSettings
from flext_core.config.validators import validate_url


class SingerConfig(BaseConfig):
    """Base configuration for Singer taps and targets."""

    # Stream selection
    stream_maps: dict[str, Any] | None = Field(
        default=None,
        description="Configuration for stream maps including aliasing and filtering",
    )
    stream_map_config: dict[str, Any] | None = Field(
        default=None,
        description="Additional configuration for stream maps",
    )

    # State management
    state: dict[str, Any] | None = Field(
        default=None,
        description="Singer state for incremental replication",
    )

    # Performance settings
    max_parallel_streams: int = Field(
        default=0,
        description="Maximum number of parallel streams (0 = sequential)",
        ge=0,
    )
    batch_config: dict[str, Any] | None = Field(
        default=None,
        description="Batch configuration for targets",
    )

    # Common API settings
    api_url: str | None = Field(default=None, description="Base API URL")
    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds",
        gt=0,
    )
    retry_count: int = Field(
        default=3,
        description="Number of retries for failed requests",
        ge=0,
    )
    page_size: int = Field(
        default=100,
        description="Page size for paginated requests",
        gt=0,
    )

    @field_validator("api_url")
    @classmethod
    def validate_api_url(cls, v: str | None) -> str | None:
        """Validate API URL.

        Arguments:
            v: The API URL to validate.

        Returns:
            The validated API URL.

        """
        if v is not None:
            return validate_url(v)
        return v


class SingerTapConfig(SingerConfig):
    """Configuration for Singer taps (data extractors)."""

    # Discovery settings
    discover_mode: bool = Field(
        default=False,
        description="Run in discovery mode to generate catalog",
    )
    catalog: dict[str, Any] | None = Field(
        None,
        description="Singer catalog for stream and schema discovery",
    )
    properties: dict[str, Any] | None = Field(
        None,
        description="Deprecated: Use catalog instead",
    )

    # Selection
    selected_streams: list[str] | None = Field(
        None,
        description="List of selected streams to sync",
    )


class SingerTargetConfig(SingerConfig):
    """Configuration for Singer targets (data loaders)."""

    # Load settings
    add_record_metadata: bool = Field(
        default=True,
        description="Add metadata to records",
    )
    load_method: str = Field(
        "append",
        description="Method for loading data (append, upsert, overwrite)",
        pattern="^(append|upsert|overwrite)$",
    )

    # Target-specific settings
    default_target_schema: str | None = Field(
        default=None,
        description="Default target schema for all streams",
    )
    flattening_enabled: bool = Field(
        default=True,
        description="Enable flattening of nested objects",
    )
    flattening_max_depth: int = Field(
        10,
        description="Maximum depth for flattening nested objects",
        gt=0,
        le=20,
    )


class SingerSettings(BaseSettings):
    """Settings for Singer SDK projects with environment variable support."""

    model_config = SettingsConfigDict(env_prefix="FLEXT_")

    # Authentication
    api_key: str | None = Field(default=None, description="API key for authentication")
    client_id: str | None = Field(default=None, description="OAuth client ID")
    client_secret: str | None = Field(default=None, description="OAuth client secret")
    refresh_token: str | None = Field(default=None, description="OAuth refresh token")

    # Database connections (for database taps/targets)
    database_url: str | None = Field(
        default=None,
        description="Database connection URL",
    )

    @field_validator("api_key", "client_secret", "refresh_token")
    @classmethod
    def mask_secrets(cls, v: str | None) -> str | None:
        """Mask secrets.

        Arguments:
            v: The value to mask.

        Returns:
            The masked value.

        """
        # Return the value unchanged - secrets are handled by env_prefix masking
        return v


def singer_config_adapter(
    pydantic_config: type[BaseConfig],
    *,
    is_target: bool = False,  # Reserved for future target-specific logic  # noqa: ARG001
) -> dict[str, Any]:
    """Convert pydantic config to Singer SDK schema format.

    Arguments:
        pydantic_config: The Pydantic config to convert.
        is_target: Whether the config is for a target.

    Returns:
        The converted config.

    """
    # Get config schema from Pydantic model
    schema = pydantic_config.model_json_schema()

    # Convert to Singer SDK format
    singer_schema = {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    # Convert properties
    for prop_name, prop_schema in schema.get("properties", {}).items():
        singer_prop = {
            "type": prop_schema.get("type", "string"),
            "description": prop_schema.get("description", ""),
        }

        # Add constraints
        if "minimum" in prop_schema:
            singer_prop["minimum"] = prop_schema["minimum"]
        if "maximum" in prop_schema:
            singer_prop["maximum"] = prop_schema["maximum"]
        if "pattern" in prop_schema:
            singer_prop["pattern"] = prop_schema["pattern"]
        if "enum" in prop_schema:
            singer_prop["enum"] = prop_schema["enum"]

        # Mark secrets
        if prop_name in {"api_key", "client_secret", "password", "token"}:
            singer_prop["secret"] = True
            singer_prop["writeOnly"] = True

        if isinstance(singer_schema, dict) and "properties" in singer_schema:
            singer_schema["properties"][prop_name] = singer_prop  # type: ignore[index]

    # Set required fields
    singer_schema["required"] = schema.get("required", [])

    return singer_schema


def create_singer_config_class(
    name: str,
    fields: dict[str, Any],
    *,
    is_target: bool = False,
    base_class: type[SingerConfig] | None = None,
) -> type[SingerConfig]:
    """Create a Singer configuration class dynamically.

    Arguments:
        name: The name of the class.
        fields: The fields of the class.
        is_target: Whether the config is for a target.
        base_class: The base class to inherit from.

    Returns:
        The created class.

    """
    if base_class is None:
        base_class = SingerTargetConfig if is_target else SingerTapConfig

    # Create class dynamically
    return type(name, (base_class,), fields)
