"""App configuration schema (new) using Pydantic v2 and centralized Flext* modules.

Single-class module: defines `FlextConfigSchemaAppConfig` only.

Design:
- Base class is `FlextModels` to inherit unified model_config.
- Uses enums from `FlextConstants.Config` directly for strict values.
- Uses `FlextTypes` for common aliases and computed helpers.
- Business rules via Pydantic v2 `@field_validator` and `@model_validator`.

No compatibility helpers are provided here. This module is standalone and
intended to be referenced by the facade in `config_new/core.py`.
"""

from __future__ import annotations

from pydantic import Field, computed_field, field_validator, model_validator

from flext_core.constants import FlextConstants
from flext_core.models import FlextModels


class FlextConfigSchemaAppConfig(FlextModels.BaseConfig):
    """Application configuration schema with strict types and rules.

    Fields are intentionally minimal for initial migration and can be extended
    safely. Environment and log level use centralized enums from FlextConstants.
    """

    # Core
    app_name: str = Field(default="flext-app", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default=FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
        description="Runtime environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Logging
    log_level: str = Field(
        default=FlextConstants.Config.LogLevel.INFO.value,
        description="Logging severity level",
    )

    # Optional integration endpoints and customizations
    database_url: str | None = Field(default=None, description="Database URL")
    redis_url: str | None = Field(default=None, description="Redis URL")
    api_keys: dict[str, str] = Field(default_factory=dict, description="API keys")
    feature_flags: dict[str, bool] = Field(
        default_factory=dict, description="Feature flags"
    )
    custom_settings: dict[str, object] = Field(
        default_factory=dict, description="Custom settings"
    )

    # Network/defaults
    timeout_seconds: int = Field(
        default=FlextConstants.Network.DEFAULT_TIMEOUT,
        ge=1,
        description="Default request timeout in seconds",
    )
    max_retries: int = Field(
        default=FlextConstants.Defaults.MAX_RETRIES,
        ge=0,
        description="Default retry attempts",
    )

    # Feature flags (kept minimal)
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    enable_tracing: bool = Field(default=False, description="Enable tracing")

    # Source metadata
    config_source: str = Field(
        default=FlextConstants.Config.ConfigSource.DEFAULT.value,
        description="Source of the configuration",
    )

    @field_validator("config_source")
    @classmethod
    def _validate_config_source(cls, v: str) -> str:
        # Be permissive for source labeling to ease testing and integrations
        return v

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        allowed = {e.value for e in FlextConstants.Config.ConfigEnvironment}
        if v not in allowed:
            msg = f"Environment must be one of: {sorted(allowed)}"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {log.value for log in FlextConstants.Config.LogLevel}
        if v not in allowed:
            msg = f"Log level must be one of: {sorted(allowed)}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _business_rules(self) -> FlextConfigSchemaAppConfig:
        # Rule: debug is not allowed in production
        if (
            self.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
            and self.debug
        ):
            msg = "Debug must be disabled in production environment"
            raise ValueError(msg)
        return self

    @property
    @computed_field
    def is_production(self) -> bool:
        """Return True when environment is production."""
        return (
            self.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
        )

    @property
    @computed_field
    def numeric_log_level(self) -> int:
        """Return numeric level mapped from `log_level`."""
        mapping = FlextConstants.Config.LogLevel.get_numeric_levels()
        return mapping.get(self.log_level, mapping["INFO"])

    # No legacy alias fields; keep schema strictly defined
